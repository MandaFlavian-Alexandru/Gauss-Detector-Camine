from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Restrict threading at the C/C++ level BEFORE importing heavy numerical libraries
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"


import cv2
import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import torch
import torchvision
import scipy.interpolate
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm
from ultralytics import YOLO

# Limit PyTorch and OpenCV threads to reduce excessive CPU usage on high-core-count processors (like i9 13900K)
torch.set_num_threads(2)
cv2.setNumThreads(2)
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
    batch_size:       int   = 96
    camera_height:    float = 2.45   # ladybug sits about 2.45m off the ground
    h_fov:            float = 60.0   # horizontal field of view per camera in degrees

    # ---
    # tiled inference settings (for cross-domain model deployment)
    # ---
    # we're running a model trained on dashcam images (wide landscape, rectilinear)
    # against ladybug images (tall portrait, fisheye). that's a big visual gap and
    # straight model.predict() at the native resolution doesn't catch anything.
    #
    # the workaround is:
    #   1. crop out the sky/upper region — training had a "blind zone" mask there
    #   2. slice the result into overlapping square tiles
    #   3. run YOLO on each tile at the tile's native size (no letterboxing squash)
    #   4. merge detections back to original image coords with NMS
    #
    # tweak the values below if you want to tune speed vs recall.
    use_tiled_inference: bool   = True   # turn off to fall back to plain model.predict
    roi_crop_top:        float  = 0.40   # fraction of image height to crop off the top
    tile_size:           int    = 640    # YOLOv8's native training size — best feature match
    tile_overlap:        float  = 0.30   # overlap between adjacent tiles (0–1)
    base_conf:           float  = 0.10   # very low threshold during the scan; we filter
                                          # by `confidence` at the end. this catches manholes
                                          # that the model is uncertain about (which is most of
                                          # them given the domain shift).

    # vertical datum correction — if None we calculate it automatically from the data.
    # for this recording area in Romania it ends up being ~39.1m.
    # you can hardcode it here if you already know it for a given recording area.
    geoid_undulation: Optional[float] = None

    # Ladybug hardware calibration file (.cal).  When set, the precise per-lens
    # pinhole intrinsics + extrinsic rotation matrices from the file are used for
    # unprojection using the .cal B-Spline mesh for all active lenses.
    ladybug_cal_path: Optional[str] = None

    # cam_key → Ladybug camera ID mapping derived from angular geometry of the rig.
    camera_lb_ids: dict = field(default_factory=lambda: {
        "Camera1": 3,   # rear-left   (~216° CW)
        "Camera2": 4,   # front-left  (~288° CW)
        "Camera3": 1,   # front-right (~72°  CW)
        "Camera4": 2,   # rear-right  (~144° CW)
    })


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
_stereo70_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3844", "EPSG:4326", always_xy=True)


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
#
# Ladybug 5+ calibration constants — properties of the camera hardware,
# fixed for a given serial number. The .cal file encodes intrinsics and
# distortion for a 2448x2048 sensor frame per camera.
LADYBUG_SENSOR_W      = 2448
LADYBUG_SENSOR_H      = 2048
LADYBUG_SPLINE_DEGREE = 3      # cubic, indicated by 4-fold repeated end knots

_LB_CAL_CACHE: dict = {}  # path -> {'cameras': {...}, 'warps': {...}}
_LB_LUT_CACHE: dict = {}  # (path, lb_id) -> ndarray of shape (W, H, 3)


def _parse_ladybug_cal(path: str) -> dict:
    """
    Parse a Ladybug .cal file fully — cameras AND warp blocks — and validate
    everything aggressively. Cached so this only happens once per run.

    Parses:
      - Per-camera intrinsics:  focalLength, Center (principal point)
      - Per-camera extrinsics:  CamToLadybugEulerZYX
      - Per-camera warp IDs:    RectifiedSpline, DistortedSpline
      - Tensor-product B-spline warp blocks: knots, coefficient grids

    Validates (raises ValueError if anything is off):
      - File exists and is readable
      - Every camera has all required fields
      - Focal lengths are non-zero (catches corrupt cal files)
      - Rotation matrices are orthonormal (det == 1, R @ R.T == I)
      - Warp ID references resolve to defined warp blocks
      - Knot count matches the cubic spline relation N_knots == N_coefs + degree + 1

    Returns:
        {
          'cameras': { lb_id (int) -> {
              'fl_x', 'fl_y', 'cx_n', 'cy_n', 'R',
              'rect_warp_u_id', 'rect_warp_v_id',
              'dist_warp_u_id', 'dist_warp_v_id',
          }, ... },
          'warps': { warp_id (int) -> {
              'knots_x', 'knots_y', 'coefs_2d', 'kx', 'ky',
          }, ... },
        }

    Cam frame convention:    X=right, Y=down, Z=forward (standard pinhole)
    Ladybug body convention: X=forward, Y=left, Z=up
    Rotation R takes camera frame → Ladybug body frame.
    """
    if path in _LB_CAL_CACHE:
        return _LB_CAL_CACHE[path]

    if not path or not os.path.isfile(path):
        raise ValueError(f"Ladybug calibration file not found: {path!r}")

    cameras: dict  = {}
    warps:   dict  = {}
    cur_cam: dict  = {}
    cur_warp: dict = {}
    cur_block      = None
    reading_coefs  = False

    with open(path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            key = tok[0]

            if key == 'BeginCamera':
                cur_cam = {}
                cur_block = 'camera'
                continue
            if key == 'EndCamera':
                if 'id' in cur_cam:
                    cameras[cur_cam['id']] = cur_cam
                cur_cam   = {}
                cur_block = None
                continue
            if key == 'BeginWarp':
                cur_warp      = {'_coef_buffer': []}
                cur_block     = 'warp'
                reading_coefs = False
                continue
            if key == 'EndWarp':
                if 'id' in cur_warp:
                    nx, ny = cur_warp.get('num_coefs', (0, 0))
                    buf    = cur_warp.pop('_coef_buffer')
                    if nx == 0 or ny == 0:
                        raise ValueError(f"Warp {cur_warp['id']}: NumberCoefs missing")
                    if len(buf) != nx * ny:
                        raise ValueError(
                            f"Warp {cur_warp['id']}: expected {nx*ny} coefs "
                            f"({nx} x {ny}), got {len(buf)}"
                        )
                    cur_warp['coefs_2d'] = np.array(buf, dtype=np.float64).reshape(nx, ny)
                    cur_warp['kx']       = LADYBUG_SPLINE_DEGREE
                    cur_warp['ky']       = LADYBUG_SPLINE_DEGREE
                    warps[cur_warp['id']] = cur_warp
                cur_warp      = {}
                cur_block     = None
                reading_coefs = False
                continue

            if cur_block == 'camera':
                if   key == 'Id':               cur_cam['id']   = int(tok[1])
                elif key == 'focalLength':      cur_cam['fl_x'] = float(tok[1]); cur_cam['fl_y'] = float(tok[2])
                elif key == 'Center':           cur_cam['cx_n'] = float(tok[4]); cur_cam['cy_n'] = float(tok[5])
                elif key == 'RectifiedSpline':  cur_cam['rect_warp_u_id'] = int(tok[1]); cur_cam['rect_warp_v_id'] = int(tok[2])
                elif key == 'DistortedSpline':  cur_cam['dist_warp_u_id'] = int(tok[1]); cur_cam['dist_warp_v_id'] = int(tok[2])
                elif key == 'CamToLadybugEulerZYX':
                    cur_cam['euler'] = (float(tok[1]), float(tok[2]), float(tok[3]))
                continue

            if cur_block == 'warp':
                if key == 'Id':
                    cur_warp['id'] = int(tok[1])
                    reading_coefs = False
                elif key == 'NumberKnots':
                    cur_warp['num_knots'] = int(tok[1])
                    reading_coefs = False
                elif key == 'KnotsX':
                    cur_warp['knots_x'] = np.array([float(x) for x in tok[1:]], dtype=np.float64)
                    reading_coefs = False
                elif key == 'KnotsY':
                    cur_warp['knots_y'] = np.array([float(x) for x in tok[1:]], dtype=np.float64)
                    reading_coefs = False
                elif key == 'NumberCoefs':
                    cur_warp['num_coefs'] = (int(tok[1]), int(tok[2]))
                    reading_coefs = False
                elif key == 'Coefs':
                    reading_coefs = True
                    cur_warp['_coef_buffer'].extend(float(x) for x in tok[1:])
                elif reading_coefs:
                    cur_warp['_coef_buffer'].extend(float(x) for x in tok)
                continue

    if not cameras:
        raise ValueError(f"No camera blocks parsed from {path}")

    for cam_id, cam in cameras.items():
        for req in ('fl_x', 'fl_y', 'cx_n', 'cy_n', 'euler',
                    'rect_warp_u_id', 'rect_warp_v_id'):
            if req not in cam:
                raise ValueError(f"Camera {cam_id}: missing required field '{req}'")

        if cam['fl_x'] == 0.0 or cam['fl_y'] == 0.0:
            raise ValueError(f"Camera {cam_id}: focal length is zero — corrupt cal file")

        rx_e, ry_e, rz_e = cam['euler']
        cX, sX = math.cos(rx_e), math.sin(rx_e)
        cY, sY = math.cos(ry_e), math.sin(ry_e)
        cZ, sZ = math.cos(rz_e), math.sin(rz_e)
        mat_Rx = np.array([[1, 0, 0], [0, cX, -sX], [0, sX, cX]], dtype=np.float64)
        mat_Ry = np.array([[cY, 0, sY], [0, 1, 0], [-sY, 0, cY]], dtype=np.float64)
        mat_Rz = np.array([[cZ, -sZ, 0], [sZ, cZ, 0], [0, 0, 1]], dtype=np.float64)
        R = mat_Rz @ mat_Ry @ mat_Rx

        det = float(np.linalg.det(R))
        if abs(det - 1.0) > 1e-6:
            raise ValueError(
                f"Camera {cam_id}: rotation matrix not orthonormal "
                f"(det={det:.6f}, expected 1.0)"
            )
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError(
                f"Camera {cam_id}: rotation matrix is not orthogonal (R @ R.T != I)"
            )
        cam['R'] = R

    for cam_id, cam in cameras.items():
        for ref_key in ('rect_warp_u_id', 'rect_warp_v_id'):
            wid = cam[ref_key]
            if wid not in warps:
                raise ValueError(
                    f"Camera {cam_id}: references undefined warp ID {wid} ({ref_key})"
                )

    for warp_id, warp in warps.items():
        for req in ('knots_x', 'knots_y', 'num_coefs', 'coefs_2d'):
            if req not in warp:
                raise ValueError(f"Warp {warp_id}: missing '{req}'")
        kx, ky = warp['kx'], warp['ky']
        nx, ny = warp['num_coefs']
        if len(warp['knots_x']) != nx + kx + 1:
            raise ValueError(
                f"Warp {warp_id}: knots_x has {len(warp['knots_x'])} entries, "
                f"expected {nx + kx + 1} (num_coefs_x + kx + 1)"
            )
        if len(warp['knots_y']) != ny + ky + 1:
            raise ValueError(
                f"Warp {warp_id}: knots_y has {len(warp['knots_y'])} entries, "
                f"expected {ny + ky + 1} (num_coefs_y + ky + 1)"
            )

    result = {'cameras': cameras, 'warps': warps}
    _LB_CAL_CACHE[path] = result
    return result


def _lut_disk_path(cal_path: str, lb_id: int) -> str:
    """
    Path on disk where a per-camera LUT is cached. The filename includes a hash
    of the .cal file's contents so that if the cal file ever changes, the cache
    is automatically invalidated.
    """
    with open(cal_path, 'rb') as fh:
        cal_hash = hashlib.md5(fh.read()).hexdigest()[:12]
    cache_dir = os.path.join(tempfile.gettempdir(), 'gauss_ladybug_lut_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"lut_{cal_hash}_cam{lb_id}.npy")


def _build_ladybug_lut(cal_path: str, lb_id: int) -> np.ndarray:
    """
    Build (or load from disk cache) a (sensor_w, sensor_h, 3) lookup table
    that maps every DISTORTED sensor pixel to its corresponding unit ray
    vector in our working frame: (rx=right, ry=down, rz=forward).

    Steps for each sensor pixel (u, v):
      1. Normalise (u, v) to (0..1) over the sensor dimensions
      2. Evaluate the RectifiedSpline B-spline at that normalised location to
         get the corresponding rectified normalised position (u_r, v_r).
      3. Apply linear pinhole math on the rectified coords using the cal file's
         focal length and principal point to get a ray in camera frame.
      4. Rotate camera frame → Ladybug body frame using R from CamToLadybugEulerZYX.
      5. Convert Ladybug body frame (X=forward, Y=left, Z=up) to working frame
         (rx=right, ry=down, rz=forward): rx=-Y_lb, ry=-Z_lb, rz=+X_lb.
      6. Normalise.
    """
    cache_key = (cal_path, lb_id)
    if cache_key in _LB_LUT_CACHE:
        return _LB_LUT_CACHE[cache_key]

    disk_path = _lut_disk_path(cal_path, lb_id)
    if os.path.isfile(disk_path):
        try:
            lut = np.load(disk_path)
            if lut.shape == (LADYBUG_SENSOR_W, LADYBUG_SENSOR_H, 3):
                _LB_LUT_CACHE[cache_key] = lut
                return lut
        except Exception:
            pass

    data = _parse_ladybug_cal(cal_path)
    cam  = data['cameras'].get(lb_id)
    if cam is None:
        raise ValueError(
            f"Camera id {lb_id} not present in calibration file {cal_path}"
        )

    warp_u = data['warps'][cam['rect_warp_u_id']]
    warp_v = data['warps'][cam['rect_warp_v_id']]

    tck_u = (
        warp_u['knots_x'], warp_u['knots_y'],
        warp_u['coefs_2d'].flatten(),
        warp_u['kx'], warp_u['ky'],
    )
    tck_v = (
        warp_v['knots_x'], warp_v['knots_y'],
        warp_v['coefs_2d'].flatten(),
        warp_v['kx'], warp_v['ky'],
    )

    sensor_w, sensor_h = LADYBUG_SENSOR_W, LADYBUG_SENSOR_H
    u_grid_n = np.arange(sensor_w, dtype=np.float64) / (sensor_w - 1)
    v_grid_n = np.arange(sensor_h, dtype=np.float64) / (sensor_h - 1)

    print(f"        evaluating rectification spline for camera {lb_id}...")
    u_rect_grid = scipy.interpolate.bisplev(u_grid_n, v_grid_n, tck_u)
    v_rect_grid = scipy.interpolate.bisplev(u_grid_n, v_grid_n, tck_v)

    if (np.abs(u_rect_grid) > 5).any() or (np.abs(v_rect_grid) > 5).any():
        raise ValueError(
            f"Camera {lb_id}: rectification spline output out of plausible range "
            f"(u in [{u_rect_grid.min():.2f}, {u_rect_grid.max():.2f}], "
            f"v in [{v_rect_grid.min():.2f}, {v_rect_grid.max():.2f}]). "
            "FITPACK coefficient ordering may be wrong."
        )

    fl_x, fl_y = cam['fl_x'], cam['fl_y']
    cx_n, cy_n = cam['cx_n'], cam['cy_n']
    dx_cam = (u_rect_grid - cx_n) / fl_x
    dy_cam = (v_rect_grid - cy_n) / fl_y

    rays_cam = np.stack(
        [dx_cam, dy_cam, np.ones_like(dx_cam)],
        axis=-1,
    )
    norms    = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_cam = rays_cam / np.maximum(norms, 1e-12)

    R       = cam['R']
    rays_lb = rays_cam @ R.T

    rays_work = np.stack([
        -rays_lb[..., 1],   # rx = -Y_lb
        -rays_lb[..., 2],   # ry = -Z_lb
         rays_lb[..., 0],   # rz = +X_lb
    ], axis=-1)
    norms     = np.linalg.norm(rays_work, axis=-1, keepdims=True)
    rays_work = rays_work / np.maximum(norms, 1e-12)

    cu, cv     = sensor_w // 2, sensor_h // 2
    center_ray = rays_work[cu, cv]
    print(f"        camera {lb_id} centre-pixel ray: "
          f"({center_ray[0]:+.3f}, {center_ray[1]:+.3f}, {center_ray[2]:+.3f})")

    rays_work = rays_work.astype(np.float32)

    try:
        np.save(disk_path, rays_work)
    except Exception as e:
        print(f"        [warn] couldn't write LUT cache to {disk_path}: {e}")

    _LB_LUT_CACHE[cache_key] = rays_work
    return rays_work


def get_ray_for_pixel(u: float, v: float, cfg: PipelineConfig, cam_key: str) -> Tuple[float, float, float]:
    """
    Returns a unit ray vector (rx, ry, rz) in the working frame for a pixel
    in the current processing resolution.

    Steps:
      1. Scale (u, v) from (image_width, image_height) → (sensor_w, sensor_h).
      2. Bilinear-interpolate the camera's pre-built LUT at that location.
      3. Re-normalise (averaging unit vectors mildly shortens them).

    No silent fallbacks — if the cal isn't usable, this raises ValueError.
    """
    if not cfg.ladybug_cal_path:
        raise ValueError(
            "get_ray_for_pixel called but no ladybug_cal_path configured in cfg"
        )

    lb_id = cfg.camera_lb_ids.get(cam_key)
    if lb_id is None:
        raise ValueError(
            f"No Ladybug camera ID mapped for camera key '{cam_key}'. "
            f"Configured keys: {list(cfg.camera_lb_ids.keys())}"
        )

    lut      = _build_ladybug_lut(cfg.ladybug_cal_path, lb_id)
    sensor_w = LADYBUG_SENSOR_W
    sensor_h = LADYBUG_SENSOR_H
    img_w    = float(cfg.image_width)
    img_h    = float(cfg.image_height)

    u_sensor = u * (sensor_w - 1) / max(img_w - 1, 1.0)
    v_sensor = v * (sensor_h - 1) / max(img_h - 1, 1.0)
    u_sensor = max(0.0, min(float(sensor_w - 1), u_sensor))
    v_sensor = max(0.0, min(float(sensor_h - 1), v_sensor))

    u0 = int(np.floor(u_sensor))
    v0 = int(np.floor(v_sensor))
    u1 = min(u0 + 1, sensor_w - 1)
    v1 = min(v0 + 1, sensor_h - 1)
    fu = u_sensor - u0
    fv = v_sensor - v0

    r00 = lut[u0, v0]
    r01 = lut[u0, v1]
    r10 = lut[u1, v0]
    r11 = lut[u1, v1]

    r_v0 = r00 * (1.0 - fu) + r10 * fu
    r_v1 = r01 * (1.0 - fu) + r11 * fu
    ray  = r_v0 * (1.0 - fv) + r_v1 * fv

    n = float(np.linalg.norm(ray))
    if n < 1e-9:
        raise ValueError(
            f"Interpolated ray vanished at sensor pixel ({u_sensor:.1f}, {v_sensor:.1f}). "
            "LUT may be corrupted."
        )
    return float(ray[0] / n), float(ray[1] / n), float(ray[2] / n)


def precompute_ladybug_luts(cfg: PipelineConfig) -> None:
    """
    Force-build the LUT for every camera configured in cfg.camera_lb_ids before
    the main loop starts. This pays the spline-evaluation cost once up-front
    instead of as a stall on the first detection per camera.
    """
    if not cfg.ladybug_cal_path:
        return
    t0 = time.time()
    print("[PHASE 1] Loading Ladybug calibration...")
    data = _parse_ladybug_cal(cfg.ladybug_cal_path)
    print(f"        parsed in {time.time()-t0:.2f}s — "
          f"{len(data['cameras'])} cameras, {len(data['warps'])} warp blocks")

    seen: set = set()
    for cam_key, lb_id in cfg.camera_lb_ids.items():
        if lb_id in seen or lb_id not in data['cameras']:
            continue
        seen.add(lb_id)
        t1 = time.time()
        _build_ladybug_lut(cfg.ladybug_cal_path, lb_id)
        print(f"        LUT ready for {cam_key} (lb_id {lb_id}) "
              f"in {time.time()-t1:.2f}s")


def _unproject_pixel(u: float, v: float, cfg: PipelineConfig, cam_key: str = "") -> Tuple[float, float, float]:
    """
    Converts a pixel coordinate to a unit direction vector in the working frame.
    Returns (rx, ry, rz) where rx>0=right, ry>0=down, rz>0=forward.

    If the Ladybug .cal calibration is configured, uses the LUT-backed
    get_ray_for_pixel for full B-spline-accurate unprojection. Failures here
    raise ValueError — we deliberately do NOT silently fall back to equirectangular,
    because that produced "looks plausible but is wrong" results when the cal
    was misconfigured.

    The equirectangular branch is only entered when no .cal is configured at all.
    """
    if cfg.ladybug_cal_path:
        if not cam_key:
            raise ValueError(
                "_unproject_pixel: ladybug_cal_path is set but no cam_key supplied. "
                "Every call site must pass cam_key when calibration is active."
            )
        return get_ray_for_pixel(u, v, cfg, cam_key)

    # No calibration configured → equirectangular approximation (legacy behaviour)
    W, H  = float(cfg.image_width), float(cfg.image_height)
    cx    = W / 2.0
    cy    = H / 2.0
    dx    = u - cx
    dy    = v - cy
    r_pix = math.sqrt(dx**2 + dy**2)
    if r_pix < 1e-6:
        return 0.0, 0.0, 1.0
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
    cam_key:            str = "",
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
    rx, ry, rz = _unproject_pixel(bbox_center_x, bbox_ref_y, cfg, cam_key)

    px_edge_flag = abs(bbox_center_x - cfg.image_width / 2.0) > cfg.image_width * 0.35

    # if ry is zero or negative the ray is pointing up or sideways, which means
    # the flat-ground fallback would give a nonsensical result (negative time).
    # just skip it.
    if ry <= 0:
        return {"x": None, "y": None, "z": None, "lat": None, "lon": None, "lidar_hit": False,
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
        # Same planar method as GaussDetectorFiride — identical bearing math,
        # capped at 100 m. Manholes are on the ground so z = expected_ground_z
        # rather than origin_z (which is camera height, 2.45 m too high).
        t        = cfg.camera_height / ry
        dist_gnd = min(math.sqrt((t*rx)**2 + (t*rz)**2), 100.0)
        centroid_xyz = np.array([
            car_x + math.sin(brng) * dist_gnd,
            car_y + math.cos(brng) * dist_gnd,
            expected_ground_z,
        ], dtype=np.float64)

    lon, lat = _stereo70_to_wgs84.transform(float(centroid_xyz[0]), float(centroid_xyz[1]))
    return {"x": float(centroid_xyz[0]), "y": float(centroid_xyz[1]), "z": float(centroid_xyz[2]),
            "lat": lat, "lon": lon,
            "lidar_hit": lidar_hit, "px_edge_flag": px_edge_flag, "range_m": range_m,
            "true_heading_deg": true_heading_deg}


# ---
# Helpers
# ---

# ---
# Tiled inference helpers (for cross-domain model deployment)
# ---

def _preprocess_for_inference(img: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, int]:
    """
    Crops the top portion of the image to roughly mimic the "AI BLIND ZONE"
    that was masked out during the model's training. The training dashcam
    images had the upper ~30% ignored — sky, distant buildings, anything
    above the road. For our taller ladybug images, cropping the top 40%
    leaves us with the road surface, which is where manholes actually live.

    Returns (cropped_image, y_offset). The y_offset is how many pixels were
    sliced off the top — we need it to map detection coordinates back to
    the original image after inference.
    """
    h = img.shape[0]
    y_offset = int(h * cfg.roi_crop_top)
    return img[y_offset:, :], y_offset


def _generate_tile_origins(image_w: int, image_h: int, tile: int, overlap: float) -> list:
    """
    Lays out a grid of tile origin positions that fully covers the image,
    with the requested overlap between adjacent tiles. The last column/row
    is always shifted back so it touches the image edge — no gaps at the
    right or bottom.
    """
    step = max(1, int(tile * (1 - overlap)))

    xs = list(range(0, max(1, image_w - tile + 1), step))
    if not xs:
        xs = [0]
    if xs[-1] + tile < image_w:
        xs.append(max(0, image_w - tile))

    ys = list(range(0, max(1, image_h - tile + 1), step))
    if not ys:
        ys = [0]
    if ys[-1] + tile < image_h:
        ys.append(max(0, image_h - tile))

    return [(x, y) for y in ys for x in xs]


def _predict_with_tiles_batch(
    model:     YOLO,
    imgs:      list[np.ndarray],
    cfg:       PipelineConfig,
    device:    str,
    use_half:  bool,
) -> list[list[dict]]:
    """
    GPU-native batched tile inference.

    Images are uploaded one at a time (~3-4 MB each) via torch.from_numpy zero-copy
    views, avoiding a single massive np.stack that would block the main thread with
    hundreds of MB of CPU memcpy. Tiles are sliced as GPU tensor views + F.pad.
    Box coordinates stay on GPU until the very end: r.boxes.xyxy returns the raw
    GPU tensor so we accumulate, offset, and NMS entirely on-device, then do one
    bulk .cpu() pull per image instead of one CUDA sync per box.
    """
    if not imgs:
        return []

    h, w = imgs[0].shape[:2]
    tile    = cfg.tile_size
    origins = _generate_tile_origins(w, h, tile, cfg.tile_overlap)
    dtype   = torch.float16 if use_half else torch.float32

    tile_list:       list[torch.Tensor] = []
    tile_to_img_idx: list[tuple]        = []

    for img_idx, img in enumerate(imgs):
        # torch.from_numpy is a zero-copy view of the numpy buffer.
        # Uploading one image at a time (~3-4 MB) avoids the giant
        # np.stack() call that would pin hundreds of MB in the main thread.
        img_t = (
            torch.from_numpy(img)
            .to(device=device, dtype=dtype)   # small upload, no intermediate CPU copy
            .permute(2, 0, 1)                 # HWC → CHW on GPU
            .contiguous()
            .div_(255.0)
        )
        for x0, y0 in origins:
            x1   = min(x0 + tile, w)
            y1   = min(y0 + tile, h)
            crop = img_t[:, y0:y1, x0:x1]    # GPU view — no copy
            ph   = tile - crop.shape[1]
            pw   = tile - crop.shape[2]
            if ph > 0 or pw > 0:
                crop = torch.nn.functional.pad(crop, (0, pw, 0, ph), mode='reflect')
            tile_list.append(crop)
            tile_to_img_idx.append((img_idx, x0, y0))

    if not tile_list:
        return [[] for _ in imgs]

    batch_tensor = torch.stack(tile_list, dim=0)   # (N_tiles, C, H, W) on GPU

    with torch.inference_mode():
        results = model.predict(
            source=batch_tensor,
            batch=cfg.batch_size,
            conf=cfg.base_conf,
            iou=cfg.iou_threshold,
            imgsz=tile,
            augment=cfg.use_tta,
            half=use_half,
            device=device,
            workers=0,
            verbose=False,
        )

    # Accumulate box tensors entirely on GPU.
    # r.boxes.xyxy is already a GPU tensor — applying the tile offset here on GPU
    # means we never call .tolist() per box, eliminating one CUDA sync per detection.
    all_boxes_gpu: list[list[torch.Tensor]] = [[] for _ in imgs]
    for r, (img_idx, x0, y0) in zip(results, tile_to_img_idx):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy   = r.boxes.xyxy                                          # GPU (N, 4)
        conf   = r.boxes.conf.unsqueeze(1)                             # GPU (N, 1)
        offset = torch.tensor([x0, y0, x0, y0], dtype=xyxy.dtype, device=device)
        all_boxes_gpu[img_idx].append(
            torch.cat([xyxy + offset, conf], dim=1)                    # GPU (N, 5)
        )

    # One GPU→CPU transfer per image after NMS — not one per box
    final_detections_per_img = []
    for box_list in all_boxes_gpu:
        if not box_list:
            final_detections_per_img.append([])
            continue
        arr  = torch.cat(box_list, dim=0)                              # (N_total, 5)
        keep = torchvision.ops.nms(arr[:, :4], arr[:, 4], cfg.iou_threshold)
        kept = arr[keep]
        kept = kept[kept[:, 4] >= cfg.base_conf].cpu().numpy()         # single sync
        final_detections_per_img.append([
            {"xyxy": [float(v) for v in row[:4]], "conf": float(row[4])}
            for row in kept
        ])

    return final_detections_per_img


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

    # build the Ladybug LUTs up-front so the first detection per camera isn't
    # blocked by a multi-second spline-evaluation stall.
    precompute_ladybug_luts(cfg)

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

        # plain model.predict() on the full ladybug image produces zero detections
        # for the manhole model — see comments on _predict_with_tiles for why.
        # we now go image-by-image, crop out the sky region, tile the rest, and
        # run YOLO on each tile separately.
        n_det          = 0
        n_raw_total    = 0   # detections found at base_conf, before final threshold
        n_after_filter = 0   # detections that survived the final cfg.confidence filter

        # chunk_size = images fed to _predict_with_tiles_batch per call.
        # Each image produces ~6 tiles, so this gives batch_size tiles per YOLO call
        # (e.g. 96 batch_size / 6 = 16 images → 96 tiles). Larger than the old 4-image
        # chunks so the GPU stays busier, but small enough that the per-image upload
        # loop and torch.stack don't pin the main thread for hundreds of ms.
        chunk_size = max(1, cfg.batch_size // 6)

        def process_chunk(files, img_data_list, offsets):
            nonlocal n_det, n_raw_total, n_after_filter, all_detections, heading_check_sample

            if cfg.use_tiled_inference:
                batch_detections = _predict_with_tiles_batch(
                    model, img_data_list, cfg, device, use_half
                )
            else:
                # legacy single-shot inference on the full images
                results = model.predict(
                    source=img_data_list, batch=cfg.batch_size, conf=cfg.base_conf, iou=cfg.iou_threshold,
                    imgsz=cfg.image_width, augment=cfg.use_tta, half=use_half,
                    device=device, workers=0, verbose=False,
                )
                batch_detections = []
                for r in results:
                    dets = []
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        dets.append({
                            "xyxy": [bx1, by1, bx2, by2],
                            "conf": float(box.conf[0]),
                        })
                    batch_detections.append(dets)

            for idx, file_name in enumerate(files):
                detections = batch_detections[idx]
                y_offset = offsets[idx]
                
                n_raw_total += len(detections)
                detections = [d for d in detections if d["conf"] >= cfg.confidence]
                n_after_filter += len(detections)

                if not detections:
                    continue

                telemetry = lookup.get(file_name.strip().lower())
                car_x = float(telemetry["X_Stereo70"])
                car_y = float(telemetry["Y_Stereo70"])
                car_z = float(telemetry["Z"])
                car_h = float(telemetry["Heading_deg"])

                for det in detections:
                    x1, y1, x2, y2 = det["xyxy"]
                    y1 += y_offset
                    y2 += y_offset
                    conf = det["conf"]
                    bbox_cx = (x1 + x2) / 2.0
                    bbox_cy = (y1 + y2) / 2.0

                    effective_mount = 0.0 if cfg.ladybug_cal_path else mount_ang
                    geo = calculate_gps_offset_3d(
                        car_x, car_y, car_z, car_h,
                        bbox_cx, bbox_cy, effective_mount,
                        kdtree, points, cfg, geoid_undulation,
                        cam_key=cam_key,
                    )

                    if geo["x"] is None:
                        continue

                    det_record = {
                        "image":        file_name,
                        "cam_key":      cam_key,
                        "folder_path":  folder_path,
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "conf":         conf,
                        "x":            geo["x"],
                        "y":            geo["y"],
                        "z":            geo["z"],
                        "lat":          geo["lat"],
                        "lon":          geo["lon"],
                        "lidar_hit":    geo["lidar_hit"],
                        "px_edge_flag": geo["px_edge_flag"],
                        "range_m":      geo["range_m"],
                        "_car_heading_deg":  car_h,
                        "true_heading_deg":  geo["true_heading_deg"],
                    }
                    all_detections.append(det_record)
                    n_det += 1

                    if len(heading_check_sample) < 5:
                        heading_check_sample.append(det_record)

        def _load_and_prep(img_file):
            img_name = img_file.strip().lower()
            telemetry = lookup.get(img_name)
            if telemetry is None:
                return None
            
            img_path = os.path.join(folder_path, img_file)
            img_full = cv2.imread(img_path)
            if img_full is None:
                return None
                
            if cfg.use_tiled_inference:
                img_cropped, y_offset = _preprocess_for_inference(img_full, cfg)
                return (img_name, img_cropped, y_offset)
            else:
                return (img_name, img_full, 0)

        pbar = tqdm(total=len(images), desc=f"Scanning {cam_key}", unit="img")
        
        from concurrent.futures import ThreadPoolExecutor

        # Break images into explicit chunks so we only load a small bounded amount at a time
        image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
        
        # Use ThreadPoolExecutor to overlap I/O and Compute using a Double-Buffer pattern
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit the very first chunk to the threads
            if image_chunks:
                next_futures = [executor.submit(_load_and_prep, img) for img in image_chunks[0]]
            else:
                next_futures = []
                
            for i in range(len(image_chunks)):
                current_futures = next_futures
                
                # Submit the NEXT chunk so it downloads while the GPU processes the current one
                if i + 1 < len(image_chunks):
                    next_futures = [executor.submit(_load_and_prep, img) for img in image_chunks[i+1]]
                    
                batch_files = []
                batch_data = []
                batch_offsets = []
                
                # Gather the results of the CURRENT chunk
                for future in current_futures:
                    result = future.result()
                    if result is not None:
                        img_name, img_data, offset = result
                        batch_files.append(img_name)
                        batch_data.append(img_data)
                        batch_offsets.append(offset)
                        
                # Process the CURRENT chunk on the GPU
                if batch_files:
                    process_chunk(batch_files, batch_data, batch_offsets)
                    pbar.update(len(batch_files))
            
        pbar.close()

        # diagnostic summary — if n_raw_total is 0, the model isn't finding
        # ANYTHING even at the very low base_conf threshold. that means the
        # visual gap between training and inference is too wide and we need
        # to try other things (lower base_conf even further, undistort the
        # fisheye, etc).
        print(f"  [✓] {cam_key} done")
        print(f"        Raw candidates at conf >= {cfg.base_conf}:  {n_raw_total}")
        print(f"        After conf >= {cfg.confidence} filter:     {n_after_filter}")
        print(f"        Geolocated detections kept:                 {n_det}")

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

    # Cross-camera merge — KDTree + Union-Find, O(N log N) vs the old O(N^3) triple loop.
    # query_pairs finds every pair within the radius in one pass; union-find groups
    # connected components in near-linear time; a single loop then merges each group.
    if len(unique_firidas) > 1:
        coords  = np.array([[f["x"], f["y"]] for f in unique_firidas])
        cc_tree = KDTree(coords)
        pairs   = cc_tree.query_pairs(r=cfg.cross_camera_radius_m)

        parent = list(range(len(unique_firidas)))

        def _uf_find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = _uf_find(a), _uf_find(b)
            if ra != rb:
                parent[rb] = ra   # union by first root

        groups: dict[int, list[int]] = {}
        for idx in range(len(unique_firidas)):
            groups.setdefault(_uf_find(idx), []).append(idx)

        merged_firidas: list[dict] = []
        for members in groups.values():
            if len(members) == 1:
                merged_firidas.append(unique_firidas[members[0]])
                continue
            best_idx = max(members, key=lambda i: unique_firidas[i]["conf"])
            rep      = dict(unique_firidas[best_idx])
            all_photos: list[dict] = []
            total_seen = 0
            for idx in members:
                f = unique_firidas[idx]
                all_photos.extend(f.get("cluster_members", [dict(f)]))
                total_seen += f.get("seen_count", 1)
            rep["cluster_members"] = all_photos
            rep["seen_count"]      = total_seen
            rep["clustered"]       = True
            merged_firidas.append(rep)

        unique_firidas = merged_firidas

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
    # Phase 4: export GeoJSON (EPSG:4326, WGS84)
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
        drop = [c for c in ["folder_path","_car_heading_deg"]
                if c in df_exp.columns]
        df_exp = df_exp.drop(columns=drop)

        # Round coordinates to 4 decimal places
        for col in ('x', 'y', 'z', 'lat', 'lon'):
            if col in df_exp.columns:
                df_exp[col] = df_exp[col].round(4)

        # Export in WGS84 (EPSG:4326) to avoid the EPSG:3844 axis-order ambiguity
        # that causes QGIS 3.x to swap Northing/Easting, placing Romanian points in Bulgaria.
        gdf = gpd.GeoDataFrame(
            df_exp,
            geometry=gpd.points_from_xy(df_exp["lon"], df_exp["lat"]),
            crs="EPSG:4326",
        )

        geo = os.path.join(cfg.output_folder, "Export_Camine.geojson")
        jsn = os.path.join(cfg.output_folder, "Export_Camine.json")
        gdf.to_file(pathlib.Path(geo), driver="GeoJSON")
        df_exp.to_json(jsn, orient="records", indent=2)
        print(f"  GeoJSON   : {geo}")
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
    p.add_argument("--cal",              type=str,   default=None,
                   help='Path to the Ladybug .cal hardware calibration file (e.g. ladybug15295546.cal)')
    args = p.parse_args()

    if not os.path.exists(args.folder):
        print(f"Error: folder not found: {args.folder}")
    else:
        cfg  = PipelineConfig(
            parent_folder         = args.folder,
            output_folder         = args.output,
            las_folder            = args.las_folder,
            confidence            = args.conf,
            cluster_radius_m      = args.cluster,
            cross_camera_radius_m = args.cross_camera_radius,
            batch_size            = args.batch,
            geoid_undulation      = args.geoid_undulation,
            ladybug_cal_path      = args.cal,
        )
        run_enterprise_pipeline(cfg)