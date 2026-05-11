from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import torch
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm
from ultralytics import YOLO


# ---
# Config
# ---

@dataclass(frozen=True)
class PipelineConfig:
    # required
    parent_folder:    str
    output_folder:    str

    # paths
    las_folder:       str   = ""
    model_path:       str   = "models/manhole_detector_modelv4.pt"

    # detection thresholds
    confidence:       float = 0.75
    iou_threshold:    float = 0.45
    cluster_radius_m:       float = 2.00   # tight same-camera dedup radius (metres)
    cross_camera_radius_m:  float = 8.00   # wider cross-camera merge radius (metres).
                                            # rays from different camera angles land a few metres
                                            # apart even for the same firida, so we need a looser
                                            # threshold here than the same-camera pass.

    # camera / image settings
    image_width:      int   = 1280
    image_height:     int   = 1632
    use_tta:          bool  = True   # test-time augmentation, helps catch edge cases
    batch_size:       int   = 24
    camera_height:    float = 2.45   # ladybug sits about 2.45m off the ground
    h_fov:            float = 60.0   # horizontal field of view per camera in degrees

    # vertical datum correction — if None we calculate it automatically from the data.
    # for this recording area in Romania it ends up being ~39.1m.
    # you can hardcode it here if you already know it for a given recording area.
    geoid_undulation: Optional[float] = None

    # fisheye lens calibration coefficients (Scaramuzza model).
    # leave these as None to use the simpler equirectangular approximation instead.
    # if you have the OCamCalib output for the ladybug, plug the values in here.
    fisheye_a0: Optional[float] = None
    fisheye_a2: Optional[float] = None
    fisheye_a4: Optional[float] = None

    camera_angles: dict = field(default_factory=lambda: {
        "Camera3":  60.0,   # front-right
        "Camera2": 300.0,   # front-left
    })


# bounding box colors in BGR because OpenCV is BGR for some reason
COLOR_GREEN  = (0, 255,   0)   # high confidence
COLOR_ORANGE = (0, 165, 255)   # medium confidence
COLOR_RED    = (0,   0, 255)   # low confidence, worth reviewing
COLOR_YELLOW = (0, 255, 255)   # clustered — seen from multiple angles

# single shared transformer, no need to recreate it every call


# ---
# LiDAR loading
# ---

def load_lidar_kdtree(las_folder: str, side: str) -> Tuple[Optional[KDTree], Optional[np.ndarray]]:
    """
    Loads a left or right LAS file and builds a KDTree from it.
    Returns (tree, points) or (None, None) if something goes wrong.

    One thing to watch out for: laspy 1.x returns raw integers from las.x/las.y/las.z,
    not actual coordinates. Those integers are things like -2,193,189 instead of 417,248.
    We handle both versions — laspy 2.x has las.xyz which just works, and for 1.x we
    manually apply the scale and offset from the file header.
    """
    if not os.path.isdir(las_folder):
        print(f"  [!] LAS folder not found: {las_folder}")
        return None, None

    las_files = [f for f in os.listdir(las_folder) if f.lower().endswith(".las")]
    target    = next((os.path.join(las_folder, f) for f in las_files
                      if side in f.lower()), None)
    if not target:
        print(f"  [!] No '{side}' .las file found in {las_folder}")
        return None, None

    print(f"  [*] Loading {side.upper()}: {os.path.basename(target)}")
    try:
        las = laspy.read(target)

        try:
            # laspy 2.x — this just gives us the real coordinates directly
            points = np.asarray(las.xyz, dtype=np.float64)
        except AttributeError:
            # laspy 1.x — have to do it manually
            h = las.header
            points = np.column_stack([
                np.array(las.X, dtype=np.float64) * h.scale[0] + h.offset[0],
                np.array(las.Y, dtype=np.float64) * h.scale[1] + h.offset[1],
                np.array(las.Z, dtype=np.float64) * h.scale[2] + h.offset[2],
            ])

        # quick sanity check — Stereo70 easting should be somewhere between 100k and 900k.
        # if we're getting something wildly outside that range we probably got raw integers
        # instead of real coordinates and the KDTree will be useless.
        x_med = float(np.median(points[:, 0]))
        if not (100_000 < x_med < 900_000):
            raise ValueError(
                f"X median is {x_med:.0f}, which is outside the expected Stereo70 range. "
                "Looks like we got raw unscaled integers from laspy. "
                "Try upgrading: pip install 'laspy[lazrs]'"
            )

        tree = KDTree(points)
        print(f"  [*] {side.upper()} KDTree ready — {len(points):,} points  "
              f"X=[{points[:,0].min():.1f}..{points[:,0].max():.1f}]  "
              f"Z=[{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        return tree, points

    except Exception as exc:
        print(f"  [!] Failed to load {side} LAS: {exc}")
        return None, None


# ---
# Z datum calibration
# ---

def estimate_geoid_undulation(
    telemetry_df:  pd.DataFrame,
    points:        np.ndarray,
    camera_height: float = 2.45,
    n_samples:     int   = 40,
    xy_radius_m:   float = 5.0,
) -> float:
    """
    Figures out the vertical offset between the GPS altitude in the CSV
    and the Z values in the LAS file.

    The CSV gives us WGS84 ellipsoidal height. The LAS gives us orthometric
    height above the Black Sea. Those are different things and the gap between
    them (the geoid undulation) needs to be subtracted before we can raycast.

    We measure it empirically: for each vehicle position we find the lowest
    LAS point within a few metres, subtract the camera height, and compare
    to the GPS altitude. The median across ~40 samples is our correction value.

    We use a 2D KDTree (XY only) for the search here because if we searched
    in 3D the ~39m Z gap would mean we'd never find any nearby points.
    """
    tree2d = KDTree(points[:, :2])
    step   = max(1, len(telemetry_df) // n_samples)
    undulations: list[float] = []

    for _, row in telemetry_df.iloc[::step].head(n_samples).iterrows():
        vx = float(row["X_Stereo70"])
        vy = float(row["Y_Stereo70"])
        vz = float(row["Z"])   # this is the ellipsoidal GPS altitude

        idxs = tree2d.query_ball_point([vx, vy], r=xy_radius_m)
        if not idxs:
            continue

        # lowest Z in the neighbourhood = road surface
        z_ground = float(points[idxs, 2].min())
        u = vz - z_ground - camera_height

        # 15–65m is the sane range for Romania, anything outside is probably noise
        if 15.0 < u < 65.0:
            undulations.append(u)

    if not undulations:
        # shouldn't happen with a valid LAS file, but just in case
        default = 39.1
        print(f"  [!] Couldn't calibrate Z offset automatically, falling back to {default}m")
        return default

    med = float(np.median(undulations))
    std = float(np.std(undulations))
    print(f"  [*] Z offset: {med:.3f}m  (std={std:.3f}m across {len(undulations)} samples)")
    return med


# ---
# Ray math
# ---

def _unproject_pixel(u: float, v: float, cfg: PipelineConfig) -> Tuple[float, float, float]:
    """
    Converts a pixel coordinate to a unit direction vector in camera space.

    Returns (rx, ry, rz) where:
      rx > 0 means pointing right
      ry > 0 means pointing downward
      rz > 0 means pointing forward

    If we have Scaramuzza fisheye coefficients we use those, otherwise we fall
    back to a simple equirectangular approximation which is good enough for the
    centre of the frame but gets a bit off near the edges.
    """
    cx    = cfg.image_width  / 2.0
    cy    = cfg.image_height / 2.0
    dx    = u - cx
    dy    = v - cy
    r_pix = math.sqrt(dx**2 + dy**2)

    if r_pix < 1e-6:
        # pixel is exactly at the optical centre, just return straight forward
        return 0.0, 0.0, 1.0

    if cfg.fisheye_a0 is not None and cfg.fisheye_a2 is not None and cfg.fisheye_a4 is not None:
        # Scaramuzza polynomial model — more accurate, needs calibration data
        rz   = cfg.fisheye_a0 + cfg.fisheye_a2 * r_pix**2 + cfg.fisheye_a4 * r_pix**4
        norm = math.sqrt(dx**2 + dy**2 + rz**2)
        return dx/norm, dy/norm, rz/norm

    # equirectangular approximation — works fine for detections near image center
    f     = cx / math.radians(cfg.h_fov / 2.0)
    theta = r_pix / f
    sin_t = math.sin(theta)
    return sin_t*(dx/r_pix), sin_t*(dy/r_pix), math.cos(theta)


def _raycast_cylinder(
    origin:     np.ndarray,
    direction:  np.ndarray,
    kdtree:     KDTree,
    points:     np.ndarray,
    min_dist:   float = 2.0,
    max_dist:   float = 30.0,
    cyl_radius: float = 1.50,   # 1.5m works well for MX2 point density (~0.8m wall spacing)
    min_strike: int   = 2,      # need at least 2 points to call it a real hit
    step_m:     float = 0.40,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Casts a ray and checks if it hits anything in the point cloud.
    Returns (centroid_xyz, distance_m) of the first surface it hits, or (None, None).

    The approach: sample points along the ray at regular intervals, collect all
    cloud points that fall within cyl_radius of any sample, then filter those
    down to ones that are actually inside the cylinder (not just near a sample point).
    Take the closest cluster that has at least min_strike points in it.

    On the radius choice: we measured the MX2 LAS files for this project and the
    average nearest-neighbour spacing on walls is about 0.8m. A 0.4m radius cylinder
    would miss most of the time. 1.5m catches things reliably while still being
    narrow enough to give a decent centroid position.
    """
    steps   = np.arange(min_dist, max_dist, step_m)
    ray_pts = origin + np.outer(steps, direction)

    # collect candidate point indices from all sample positions along the ray
    cand_set: set = set()
    for pt in ray_pts:
        cand_set.update(kdtree.query_ball_point(pt, r=cyl_radius))

    if not cand_set:
        return None, None

    # now filter to points actually inside the cylinder, not just near a sample
    cands  = points[list(cand_set)]
    vecs   = cands - origin
    t_vals = vecs @ direction                          # distance along ray axis
    proj   = origin + np.outer(t_vals, direction)     # closest point on ray to each candidate
    perp   = np.linalg.norm(cands - proj, axis=1)     # perpendicular distance to ray axis

    mask   = (perp < cyl_radius) & (t_vals >= min_dist) & (t_vals <= max_dist)
    if not mask.any():
        return None, None

    valid   = cands[mask]
    valid_t = t_vals[mask]
    t_min   = valid_t.min()

    # grab everything within 0.5m of the first hit — that's our surface cluster
    near = np.abs(valid_t - t_min) < 0.5
    if near.sum() < min_strike:
        return None, None

    return valid[near].mean(axis=0), float(t_min)


def _raycast_ground_plane(
    origin:            np.ndarray,
    direction:         np.ndarray,
    kdtree:            KDTree,
    points:            np.ndarray,
    expected_ground_z: float,
    min_dist:          float = 2.0,
    max_dist:          float = 30.0,
    xy_radius_m:       float = 1.0,
    z_tolerance_m:     float = 0.5,
    min_strike:        int   = 3,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Ray-into-LiDAR specifically for FLAT targets like manholes.

    The cylinder version doesn't work here because the ray hits the road at a
    really shallow angle (~14° at 10m range). A cylinder along that ray catches
    a long smear of asphalt starting around 4m out, and the "first strike" logic
    locks onto that smear instead of the actual manhole. So we do this instead:

      1. Analytically solve for where the ray crosses the expected ground plane
      2. Look up LAS points within a small XY neighborhood of that crossing
      3. Filter to points that are actually at ground level (rejects nearby
         cars, curbs, anything taller than the road surface)
      4. Return the centroid of those ground points

    This pinpoints flat targets the same way the cylinder pinpoints walls.
    """
    # ray needs to be tilted downward to ever hit the ground
    if direction[2] >= -1e-6:
        return None, None

    # parametric line/plane intersection:
    #   origin.z + t * direction.z = expected_ground_z
    t = (expected_ground_z - origin[2]) / direction[2]
    if not (min_dist <= t <= max_dist):
        return None, None

    # where the ray crosses the ground plane in world coords
    target = origin + t * direction

    # gather LAS points near the predicted intersection.
    # we use a slightly bigger radius here so we have something to filter on.
    nearby_idxs = kdtree.query_ball_point(target, r=xy_radius_m * 1.5)
    if len(nearby_idxs) < min_strike:
        return None, None

    nearby = points[nearby_idxs]

    # keep only points that are actually at ground level.
    # this filter is what rejects curbs (~15cm above road), parked cars, etc.
    z_mask = np.abs(nearby[:, 2] - expected_ground_z) < z_tolerance_m
    ground_pts = nearby[z_mask]
    if len(ground_pts) < min_strike:
        return None, None

    # final XY filter — keep only points inside the actual disc around the target
    xy_dist = np.linalg.norm(ground_pts[:, :2] - target[:2], axis=1)
    inside  = ground_pts[xy_dist < xy_radius_m]
    if len(inside) < min_strike:
        return None, None

    return inside.mean(axis=0), float(t)


# ---
# Geolocation
# ---

def calculate_gps_offset_3d(
    car_x:              float,
    car_y:              float,
    car_z:              float,   # GPS altitude, WGS84 ellipsoidal
    car_heading:        float,   # degrees, compass (0=north, 90=east)
    bbox_center_x:      float,
    bbox_ref_y:         float,   # for manholes this is the box CENTER, not the bottom
    camera_mount_angle: float,
    kdtree:             Optional[KDTree],
    points:             Optional[np.ndarray],
    cfg:                PipelineConfig,
    geoid_undulation:   float,
) -> dict:
    """
    Given a YOLO detection on a flat ground target (manhole), figures out
    where that manhole actually sits in world coordinates.

    Steps:
      1. Convert the pixel position to a direction vector in camera space
      2. Rotate that into world space using the car heading + mount angle
      3. Correct the Z so it's in the same datum as the LAS file
      4. Cast the ray into the point cloud using the GROUND-PLANE method
         (the cylinder method doesn't work for flat targets, see comments
         in _raycast_ground_plane for why)
      5. If the cloud misses, fall back to flat-ground trig
      6. Return Stereo70 X, Y, Z plus quality flags

    Why bbox_ref_y is the box CENTER, not the bottom:
      For a flat manhole on the road, the bbox CENTER pixel corresponds to
      the physical centre of the manhole. The bbox bottom corresponds to
      the NEAR edge (closest to camera), which would bias every manhole
      position about half a metre toward the vehicle.
    """
    rx, ry, rz = _unproject_pixel(bbox_center_x, bbox_ref_y, cfg)

    # flag detections in the outer 35% of the frame — the equirectangular model
    # gets noticeably less accurate out there
    px_edge_flag = abs(bbox_center_x - cfg.image_width / 2.0) > cfg.image_width * 0.35

    # if ry is zero or negative the ray is pointing up or sideways, which means
    # the flat-ground fallback would give a nonsensical result (negative time).
    # just skip it.
    if ry <= 0:
        return {"x": None, "y": None, "z": None, "lidar_hit": False,
                "px_edge_flag": px_edge_flag, "range_m": None,
                "true_heading_deg": None}

    # figure out which direction in world space this pixel is pointing
    alpha_deg        = math.degrees(math.atan2(rx, rz))
    true_heading_deg = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng             = math.radians(true_heading_deg)

    vert_angle = math.atan2(-ry, math.sqrt(rx**2 + rz**2))
    cos_v      = math.cos(vert_angle)
    direction  = np.array([math.sin(brng)*cos_v,
                            math.cos(brng)*cos_v,
                            math.sin(vert_angle)], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    # the key correction: GPS gives ellipsoidal altitude, LAS uses orthometric.
    # subtract the geoid undulation to get them into the same reference system.
    origin_z = car_z - geoid_undulation
    origin   = np.array([car_x, car_y, origin_z], dtype=np.float64)

    # the road sits camera_height below the camera by definition.
    # this is the Z value we expect the manhole to be at in the LAS file.
    expected_ground_z = origin_z - cfg.camera_height

    # debug block — flip this to True if you want to see exactly what the ray
    # is doing and whether it's landing in the right Z range
    DEBUG_RAYCAST = False
    if DEBUG_RAYCAST and kdtree is not None and points is not None:
        print("\n[DEBUG] -------------------------------------------------")
        print(f"  GPS Z (ellipsoidal):  {car_z:.3f}m")
        print(f"  Geoid undulation:     {geoid_undulation:.3f}m")
        print(f"  Ray origin Z (LAS):   {origin_z:.3f}m")
        print(f"  Expected ground Z:    {expected_ground_z:.3f}m")
        print(f"  LAS Z range:          [{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        print(f"  Bearing:              {true_heading_deg:.1f}°")
        for d in [5, 10, 15, 20]:
            pt  = origin + direction * d
            n   = len(kdtree.query_ball_point(pt, r=1.5))
            print(f"  @{d:2d}m  Z={pt[2]:.2f}  nearby_pts={n}")
        x_ok = 100_000 < float(np.median(points[:,0])) < 900_000
        print(f"  X coordinates look valid: {x_ok}")
        print("[DEBUG] -------------------------------------------------\n")

    # try to hit the point cloud using the ground-plane method.
    # NOT the cylinder method — that one doesn't work for flat targets.
    centroid_xyz: Optional[np.ndarray] = None
    range_m:      Optional[float]      = None
    lidar_hit    = False

    if kdtree is not None and points is not None:
        centroid_xyz, range_m = _raycast_ground_plane(
            origin, direction, kdtree, points, expected_ground_z
        )
        if centroid_xyz is not None:
            lidar_hit = True

    # if the cloud didn't give us anything, estimate using flat-ground geometry.
    # for manholes this is actually pretty accurate because the manhole IS on
    # the road at exactly camera_height below the camera — the only assumption
    # we're making is that the road is locally flat, which is almost always true.
    if centroid_xyz is None:
        t        = cfg.camera_height / ry
        dist_gnd = min(math.sqrt((t*rx)**2 + (t*rz)**2), 100.0)
        centroid_xyz = np.array([
            car_x + math.sin(brng) * dist_gnd,
            car_y + math.cos(brng) * dist_gnd,
            expected_ground_z,
        ], dtype=np.float64)

    return {"x": float(centroid_xyz[0]), "y": float(centroid_xyz[1]), "z": float(centroid_xyz[2]), "lidar_hit": lidar_hit,
            "px_edge_flag": px_edge_flag, "range_m": range_m,
            "true_heading_deg": true_heading_deg}


# ---
# Helpers
# ---

def euclidean_distance(x1, y1, x2, y2) -> float:
    """Straight-line 2D distance in metres between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def _check_heading_convention(samples: list, cfg: PipelineConfig) -> None:
    """
    Sanity check on the heading values. If the mean offset between the resolved
    ray bearing and the car heading is around 90°, that usually means the CSV
    is using math angles (east=0) instead of compass angles (north=0).
    Logs a warning rather than crashing — easy to miss otherwise.
    """
    deltas = []
    for d in samples:
        th = d.get("true_heading_deg")
        ch = d.get("_car_heading_deg")
        if th is not None and ch is not None:
            deltas.append((th - ch + 180) % 360 - 180)
    if not deltas:
        return
    mean_abs = abs(sum(deltas) / len(deltas))
    if 60 < mean_abs < 120:
        warnings.warn(
            f"[HEADING] Average ray-vs-GPS bearing difference is {mean_abs:.1f}°. "
            "If positions look rotated 90°, check that Heading_deg in the CSV "
            "uses compass convention (north=0°, east=90°).",
            stacklevel=2,
        )


# ---
# Main pipeline
# ---

def run_enterprise_pipeline(cfg: PipelineConfig) -> None:
    print("[PHASE 1] Starting up...")
    print(f"          Source : {cfg.parent_folder}")
    print(f"          Output : {cfg.output_folder}")
    print(f"          LiDAR  : {cfg.las_folder or '(none — will use planar fallback)'}")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        print("          GPU    : CUDA available, using half precision")

    model = YOLO(cfg.model_path)
    os.makedirs(cfg.output_folder, exist_ok=True)
    start_time = time.time()

    # load both LAS files up front so we don't have to reload them per camera.
    # Camera1/2 are left-facing, Camera3/4 are right-facing.
    left_tree  = left_pts  = None
    right_tree = right_pts = None
    if cfg.las_folder:
        left_tree,  left_pts  = load_lidar_kdtree(cfg.las_folder, "left")
        right_tree, right_pts = load_lidar_kdtree(cfg.las_folder, "right")

    # figure out the Z correction needed for this recording.
    # if the user passed --geoid_undulation we use that directly,
    # otherwise we calculate it from the data.
    geoid_undulation = cfg.geoid_undulation
    if geoid_undulation is None:
        calib_pts = left_pts if left_pts is not None else right_pts
        if calib_pts is not None:
            print("[PHASE 1] Calculating Z datum offset from LAS vs GPS...")
            calib_df: Optional[pd.DataFrame] = None

            # just grab the first coordonate file we can find for calibration
            for fd in sorted(os.listdir(cfg.parent_folder)):
                fp = os.path.join(cfg.parent_folder, fd)
                if not os.path.isdir(fp):
                    continue
                for fn in os.listdir(fp):
                    fl = fn.lower()
                    if "coordonate" in fl and not fl.startswith("~$"):
                        try:
                            p = os.path.join(fp, fn)
                            calib_df = (pd.read_csv(p) if fn.endswith(".csv")
                                        else pd.read_excel(p))
                            break
                        except Exception:
                            pass
                if calib_df is not None:
                    break

            if calib_df is not None and "X_Stereo70" in calib_df.columns:
                geoid_undulation = estimate_geoid_undulation(
                    calib_df, calib_pts, cfg.camera_height
                )
            else:
                geoid_undulation = 39.1
                print(f"  [!] Couldn't find telemetry for calibration, using {geoid_undulation}m default")
        else:
            geoid_undulation = 0.0   # no LAS loaded, correction doesn't matter

    print(f"          Z correction: {geoid_undulation:.3f}m")

    # process left cameras first so the sort is deterministic and matches our preloaded trees
    def _cam_sort(name: str) -> int:
        for k, a in cfg.camera_angles.items():
            if k in name:
                return 0 if a >= 180 else 1
        return 2

    all_folders = sorted(
        [f for f in os.listdir(cfg.parent_folder)
         if os.path.isdir(os.path.join(cfg.parent_folder, f))],
        key=_cam_sort,
    )

    all_detections:       list[dict] = []
    heading_check_sample: list[dict] = []

    for folder_name in all_folders:
        folder_path = os.path.join(cfg.parent_folder, folder_name)

        cam_key   = None
        mount_ang = None
        for k, a in cfg.camera_angles.items():
            if k in folder_name:
                cam_key   = k
                mount_ang = a
                break
        if cam_key is None or mount_ang is None:
            continue

        print(f"\n---> {cam_key} (mounted at {mount_ang}° from nose)")

        # cameras with mount angle >= 180° face left, the rest face right
        is_left = mount_ang >= 180.0
        kdtree  = left_tree  if is_left else right_tree
        points  = left_pts   if is_left else right_pts

        # find the coordonate file for this camera folder
        coord_file = None
        for fn in os.listdir(folder_path):
            fl = fn.lower()
            if "coordonate" in fl and not fl.startswith("~$"):
                coord_file = os.path.join(folder_path, fn)
                break
        if not coord_file:
            print(f"  [!] No coordonate file found, skipping {cam_key}.")
            continue

        df = (pd.read_csv(coord_file) if coord_file.endswith(".csv")
              else pd.read_excel(coord_file))

        required = ["X_Stereo70", "Y_Stereo70", "Z", "Heading_deg", "Imagine"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [!] Missing columns {missing}, skipping {cam_key}.")
            continue

        # lowercase everything once here so we don't have to worry about
        # case mismatches when looking up image names later
        df["Imagine"] = df["Imagine"].astype(str).str.strip().str.lower()
        lookup = {row["Imagine"]: row for _, row in df.iterrows()}

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"  [*] {len(images)} images to process")

        results = model.predict(
            source=folder_path, conf=cfg.confidence, iou=cfg.iou_threshold,
            imgsz=cfg.image_width, augment=cfg.use_tta, half=use_half,
            device=device, stream=True, verbose=False, batch=cfg.batch_size,
        )

        n_det = 0
        for r in tqdm(results, total=len(images), desc=f"Scanning {cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue

            img_name  = os.path.basename(r.path).strip().lower()
            telemetry = lookup.get(img_name)
            if telemetry is None:
                continue   # image not in the coordonate file, skip it

            car_x = float(telemetry["X_Stereo70"])
            car_y = float(telemetry["Y_Stereo70"])
            car_z = float(telemetry["Z"])
            car_h = float(telemetry["Heading_deg"])

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                bbox_cx = (x1 + x2) / 2.0
                bbox_cy = (y1 + y2) / 2.0   # centre, not bottom — see notes in calculate_gps_offset_3d

                geo = calculate_gps_offset_3d(
                    car_x, car_y, car_z, car_h,
                    bbox_cx, bbox_cy, mount_ang,
                    kdtree, points, cfg, geoid_undulation,
                )

                if geo["x"] is None:
                    # ray was pointing upward, nothing we can do with this one
                    continue

                det = {
                    "image":        img_name,
                    "cam_key":      cam_key,
                    "folder_path":  folder_path,
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf":         conf,
                    "x":            geo["x"],
                    "y":            geo["y"],
                    "z":            geo["z"],
                    "lidar_hit":    geo["lidar_hit"],     # True = sub-metre accuracy
                    "px_edge_flag": geo["px_edge_flag"],  # True = near frame edge, less accurate
                    "range_m":      geo["range_m"],       # how far the ray travelled to hit something
                    "_car_heading_deg":  car_h,           # kept for the heading sanity check, not exported
                    "true_heading_deg":  geo["true_heading_deg"],
                }
                all_detections.append(det)
                n_det += 1

                if len(heading_check_sample) < 5:
                    heading_check_sample.append(det)

        print(f"  [✓] {cam_key} done — {n_det} detections")

    _check_heading_convention(heading_check_sample, cfg)
    print(f"\nAll cameras done — {len(all_detections)} total raw detections.")

    # ---
    # Phase 3: merge detections that are pointing at the same firida
    # ---
    print("[PHASE 3] Deduplicating and clustering nearby detections...")
    unique_firidas: list[dict] = []

    for det in all_detections:
        matched = False
        for uf in unique_firidas:
            dist = euclidean_distance(det["x"], det["y"], uf["x"], uf["y"])
            if dist <= cfg.cluster_radius_m:
                # same image, same camera — definitely a duplicate, just skip it
                if det["image"] == uf["image"] and det["cam_key"] == uf["cam_key"]:
                    matched = True
                    break
                # close enough to count as the same firida
                matched = True
                uf["seen_count"] = uf.get("seen_count", 1) + 1
                uf["clustered"]  = True
                if "cluster_members" not in uf:
                    uf["cluster_members"] = [dict(uf)]
                uf["cluster_members"].append(dict(det))
                # keep whichever detection has the higher confidence score
                if det["conf"] > uf["conf"]:
                    det["clustered"]       = True
                    det["seen_count"]      = uf["seen_count"]
                    det["cluster_members"] = uf["cluster_members"]
                    uf.update(det)
                break
        if not matched:
            det["clustered"]       = False
            det["seen_count"]      = 1
            det["cluster_members"] = [dict(det)]
            unique_firidas.append(det)

    # Cross-camera merge pass.
    #
    # The first pass above uses a tight 2m radius which works great for the same
    # camera seeing the same firida multiple times as the car drives past. But when
    # two different cameras both spot the same firida, their ray angles are different
    # and the calculated coordinates can land 3-8m apart even for the same object.
    # The 2m threshold misses those matches.
    #
    # This pass uses a wider radius and, crucially, actually merges the cluster_members
    # lists so the gallery ends up with all the photos together in one group. The old
    # version only set clustered=True without combining anything, which is why you'd
    # see two separate yellow entries for the same firida.
    #
    # We loop until nothing changes because merging two groups can create new pairs.
    did_merge = True
    while did_merge:
        did_merge = False
        i = 0
        while i < len(unique_firidas):
            j = i + 1
            while j < len(unique_firidas):
                fi = unique_firidas[i]
                fj = unique_firidas[j]
                dist = euclidean_distance(fi["x"], fi["y"], fj["x"], fj["y"])
                if dist <= cfg.cross_camera_radius_m:
                    # merge fj into fi — combine their photo lists
                    if "cluster_members" not in fi:
                        fi["cluster_members"] = [dict(fi)]
                    fi["cluster_members"].extend(fj.get("cluster_members", [dict(fj)]))
                    fi["seen_count"] = fi.get("seen_count", 1) + fj.get("seen_count", 1)
                    fi["clustered"]  = True

                    # if fj had the better confidence, promote it as the representative
                    # but keep the merged member list we just built
                    if fj["conf"] > fi["conf"]:
                        best = dict(fj)
                        best["cluster_members"] = fi["cluster_members"]
                        best["seen_count"]      = fi["seen_count"]
                        best["clustered"]       = True
                        unique_firidas[i]       = best
                        fi = unique_firidas[i]

                    unique_firidas.pop(j)
                    did_merge = True
                    # don't increment j — the list just got shorter
                else:
                    j += 1
            i += 1

    hit_count = sum(1 for d in unique_firidas if d.get("lidar_hit"))
    total     = len(unique_firidas)
    print(f"Done — {total} unique firidas found.")
    print(f"LiDAR positioned: {hit_count}/{total} ({100*hit_count//max(total,1)}%)")

    # save annotated images so you can review what got detected
    for f in unique_firidas:
        if f["clustered"]:
            color = COLOR_YELLOW
            label = f"CLUSTERED: {f['conf']:.2f}"
        elif f["conf"] >= 0.85:
            color = COLOR_GREEN
            label = f"Firida: {f['conf']:.2f}"
        elif f["conf"] >= 0.80:
            color = COLOR_ORANGE
            label = f"Firida: {f['conf']:.2f}"
        else:
            color = COLOR_RED
            label = f"WARNING: {f['conf']:.2f}"
        if not f.get("lidar_hit"):
            label += " [PLANAR]"
        if f.get("px_edge_flag"):
            label += " [EDGE]"

        for m in f.get("cluster_members", [f]):
            img = cv2.imread(os.path.join(m["folder_path"], m["image"]))
            if img is not None:
                cv2.rectangle(img, (m["x1"], m["y1"]), (m["x2"], m["y2"]), color, 4)
                cv2.putText(img, f"{m['cam_key']} - {label}",
                            (m["x1"], m["y1"]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite(
                    os.path.join(cfg.output_folder,
                                 f"{m['cam_key']}_{m['image']}"), img)

    # ---
    # Phase 4: export shapefile and JSON for QGIS
    # ---
    print("[PHASE 4] Exporting results...")
    flat: list[dict] = []
    for f in unique_firidas:
        for m in f.get("cluster_members", [f]):
            mc = {k: v for k, v in m.items() if k != "cluster_members"}
            mc["clustered"] = f.get("clustered", False)
            flat.append(mc)

    if flat:
        df_exp = pd.DataFrame(flat)

        # drop columns we don't need in the final output
        drop = [c for c in ["x1","y1","x2","y2","folder_path","_car_heading_deg"]
                if c in df_exp.columns]
        df_exp = df_exp.drop(columns=drop)

        gdf = gpd.GeoDataFrame(
            df_exp,
            geometry=gpd.points_from_xy(df_exp["x"], df_exp["y"], z=df_exp["z"]),
            crs="EPSG:3844",
        )


        shp  = os.path.join(cfg.output_folder, "Export_Camine.shp")
        jsn  = os.path.join(cfg.output_folder, "Export_Camine.json")
        gdf.to_file(pathlib.Path(shp))
        df_exp.to_json(jsn, orient="records", indent=2)
        print(f"  Shapefile : {shp}")
        print(f"  JSON      : {jsn}")
    else:
        print("  No firidas to export.")

    print(f"\nFinished in {round(time.time()-start_time, 2)}s.")


# ---
# Entry point
# ---

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gauss firida detector")
    p.add_argument("--folder",           required=True,  help="Recording directory (contains Camera1-4 subfolders)")
    p.add_argument("--output",           required=True,  help="Where to save results")
    p.add_argument("--las_folder",       default="",     help="Folder with the left/right .las files")
    p.add_argument("--conf",             type=float, default=0.75, help="YOLO confidence threshold")
    p.add_argument("--cluster",              type=float, default=2.00, help="Same-camera dedup radius in metres")
    p.add_argument("--cross_camera_radius",  type=float, default=8.00,
                   help="Cross-camera merge radius in metres. Wider than --cluster because "
                        "rays from different angles land a few metres apart for the same firida.")
    p.add_argument("--batch",            type=int,   default=24,   help="YOLO batch size")
    p.add_argument("--geoid_undulation", type=float, default=None,
                   help="Z offset in metres between GPS altitude and LAS orthometric height. "
                        "Leave blank to calculate automatically. For this area of Romania it's ~39m.")
    p.add_argument("--intrinsics",       type=str,   default=None,
                   help='Scaramuzza fisheye coefficients as JSON, e.g. \'{"a0":-616.2,"a2":0.0,"a4":-4.1e-7}\'')
    args = p.parse_args()

    if not os.path.exists(args.folder):
        print(f"Error: folder not found: {args.folder}")
    else:
        intr = json.loads(args.intrinsics) if args.intrinsics else {}
        cfg  = PipelineConfig(
            parent_folder         = args.folder,
            output_folder         = args.output,
            las_folder            = args.las_folder,
            confidence            = args.conf,
            cluster_radius_m      = args.cluster,
            cross_camera_radius_m = args.cross_camera_radius,
            batch_size            = args.batch,
            geoid_undulation      = args.geoid_undulation,
            fisheye_a0            = intr.get("a0"),
            fisheye_a2            = intr.get("a2"),
            fisheye_a4            = intr.get("a4"),
        )
        run_enterprise_pipeline(cfg)