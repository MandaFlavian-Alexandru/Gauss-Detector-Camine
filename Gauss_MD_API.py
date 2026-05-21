import os
import sys
import json
import contextlib
import logging
import subprocess
import threading
import re
import zipfile
import glob
import tempfile
import signal
import time
import uuid
import pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Gauss_FastAPI")

# only these two values are accepted for execution_mode
ALLOWED_EXECUTION_MODES = {"parallel", "sequential"}

# Path to the Ladybug factory calibration file. The backend uses this to build
# the B-spline LUT that maps each pixel to a unit ray vector — without it, the
# backend silently falls back to a pinhole approximation that drifts at the
# image edges. The path is resolved at import time so the API fails fast with
# a clear error if the file is missing, rather than every detection run quietly
# running uncalibrated.
LADYBUG_CAL_PATH = os.path.abspath("ladybug15295546.cal")


def _require_ladybug_cal() -> str:
    """
    Returns the absolute path to the Ladybug .cal file. Raises HTTPException(500)
    with an actionable message if the file is missing — better to fail loudly at
    the start of a detection run than to silently produce drifted coordinates.
    """
    if not os.path.isfile(LADYBUG_CAL_PATH):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ladybug calibration file not found at {LADYBUG_CAL_PATH}. "
                "The detector requires this file for accurate ray geometry. "
                "Place ladybug15295546.cal in the working directory and retry."
            ),
        )
    return LADYBUG_CAL_PATH


def _validate_session_id(session_id: str) -> str:
    """
    Checks that a session_id is a valid UUID before we use it to build file paths.
    A crafted value like '../../etc/passwd' would be a path traversal attack otherwise.
    """
    try:
        return str(uuid.UUID(session_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid session ID format.") from e

def _validate_directory_path(path: str) -> str:
    """
    Resolves a user-supplied directory path to its real absolute form and
    checks it actually exists as a directory.

    os.path.realpath resolves symlinks and any ../ traversal, so we get the
    true path the OS would use. We also confirm it's a directory, not a file
    or device node or anything else someone might try to slip in.
    """
    if not path or not isinstance(path, str):
        raise HTTPException(status_code=400, detail="Path must be a non-empty string.")
    resolved = os.path.realpath(os.path.abspath(path))
    if not os.path.exists(resolved):
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    if not os.path.isdir(resolved):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
    return resolved

def _validate_numeric(value: float, name: str, lo: float, hi: float) -> float:
    """Range check for numeric params that end up in the subprocess command."""
    if not (lo <= value <= hi):
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {lo} and {hi}, got {value}."
        )
    return value

app = FastAPI(title="Gauss Detector Camine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Centralized State Management for Concurrent Sessions ---
def get_state_file(session_id: str):
    return os.path.join(tempfile.gettempdir(), f"gauss_state_{session_id}.json")

def load_state(session_id: str):
    try:
        with open(get_state_file(session_id), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "session_id": session_id,
            "is_running": False,
            "progress": 0,
            "camera_progress": {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0},
            "logs": [],
            "results_ready": False,
            "current_output_dir": "", 
            "source_folder": "",
            "las_folder": "",
            "process_id": None,
            "is_pending": False,
            "is_cancelled": False
        }

def save_state(session_id: str, state):
    try:
        with open(get_state_file(session_id), 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

class AnalysisRequest(BaseModel):
    folder_path: str
    las_folder_path: str
    min_confidence: float
    cluster_radius: float
    batch_size: int
    execution_mode: str = "sequential"

class DeleteRequest(BaseModel):
    session_id: str
    image_name: str
    cam_key: str

class FinalExportRequest(BaseModel):
    session_id: str
    results: list

def run_subprocess(session_id: str, folder_path: str, las_folder_path: str, conf: float, cluster: float, batch_size: int, output_dir: str):
    """Executes the Gauss backend pipeline as a subprocess and monitors its stdout."""

    # all three paths were resolved and validated before this function was called,
    # but we resolve them again here as a defence-in-depth measure in case this
    # function is ever called from somewhere else in the future.
    folder_path    = os.path.realpath(folder_path)
    las_folder_path = os.path.realpath(las_folder_path) if las_folder_path else ""
    output_dir     = os.path.realpath(output_dir)

    for f in glob.glob(os.path.join(output_dir, "Export_Camine.*")): 
        try: 
            os.remove(f)
        except OSError as e:
            logger.warning(f"Could not remove old file {f}: {e}")

    st = load_state(session_id)
    if st.get("is_cancelled"): 
        return
        
    st["is_running"] = True
    st["is_pending"] = False
    st["logs"] = [f"Initializing AI Analysis in: {folder_path}"]
    st["progress"] = 0
    st["camera_progress"] = {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0}
    st["results_ready"] = False

    # build the command as a list (never a string) so the OS never passes it
    # through a shell. each argument is a separate list element, which means
    # special characters like semicolons, pipes, or backticks in a path are
    # treated as literal characters and cannot be used for injection.
    cmd = [
        sys.executable, os.path.abspath("Gauss_MD_Backend.py"),
        "--folder",     folder_path,
        "--las_folder", las_folder_path,
        "--cal",        _require_ladybug_cal(),
        "--conf",       str(round(conf / 100.0, 4)),
        "--cluster",    str(round(cluster, 4)),
        "--batch",      str(int(batch_size)),
        "--output",     output_dir,
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,   # never True — shell=True is what enables injection
    )
    st["process_id"] = process.pid
    save_state(session_id, st)
    
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            if line := line.strip():
                st = load_state(session_id) # Refresh state before updating
                
                # Clean ANSI characters that tqdm outputs for a cleaner UI log
                clean_log = re.sub(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]', '', line)
                
                # Tqdm naturally emits percentage blocks, so we skip adding pure percentage strings to logs
                if '%' not in clean_log or "[PHASE" in clean_log:
                     st["logs"].append(clean_log)
                     if len(st["logs"]) > 50: 
                         st["logs"] = st["logs"][-50:]
                    
                # Regex for literal `Scanning Camera1:  63%|` outputs
                if cam_m := re.search(r'Scanning.*?(Camera[1-4]).*?(\d+)\s*%', clean_log):
                    st["camera_progress"][cam_m[1]] = int(cam_m[2])
                    
                if "[PHASE 1]" in clean_log:
                    st["progress"] = 5
                elif "[PHASE 3]" in clean_log:
                    st["progress"] = 75
                elif "[PHASE 4]" in clean_log:
                    st["progress"] = 90
                
                # Dynamically calc phase 2 progress using tqdm stats
                if st["progress"] < 75:
                    # Up to 200 cumulative points total across 2 cameras
                    cam_total = sum(st["camera_progress"].values())
                    cam_overall = cam_total / 200.0
                    st["progress"] = 5 + int(cam_overall * 70)

                save_state(session_id, st)
            
    if process.stdout:
        process.stdout.close()
    process.wait()
    
    st = load_state(session_id)
    st["progress"] = 100
    st["camera_progress"] = {"Camera1": 100, "Camera2": 100, "Camera3": 100, "Camera4": 100}
    st["is_running"] = False
    st["results_ready"] = True
    st["process_id"] = None
    save_state(session_id, st)

def run_sequential_batch(sessions_to_run):
    """Executes a list of sessions one by one. If a user cancels one, it skips to the next."""
    for sess in sessions_to_run:
        # Verify it wasn't cancelled while waiting in queue
        st = load_state(sess["session_id"])
        if st.get("is_cancelled", False):
            continue
            
        run_subprocess(
            sess["session_id"], sess["folder_path"], sess["las_folder_path"], 
            sess["conf"], sess["cluster"], sess["batch"], sess["output"]
        )

@app.post("/api/analyze")
def start_analysis(req: AnalysisRequest):
    print(f"*** Batch Request Received. Target Execution Mode: {req.execution_mode.upper()} ***")

    # validate execution_mode against an explicit allowlist.
    # without this, any string would silently fall through to the else branch
    # and be treated as "parallel" — not dangerous on its own, but still wrong.
    if req.execution_mode not in ALLOWED_EXECUTION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid execution_mode '{req.execution_mode}'. Must be one of: {ALLOWED_EXECUTION_MODES}"
        )

    # validate and resolve paths — realpath strips any ../ traversal attempts
    try:
        folder_path     = _validate_directory_path(req.folder_path)
        las_folder_path = _validate_directory_path(req.las_folder_path) if req.las_folder_path else ""
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": f"Invalid path: {e}"}

    # validate numeric parameters before they go anywhere near the subprocess
    _validate_numeric(req.min_confidence, "min_confidence", 0.0, 100.0)
    _validate_numeric(req.cluster_radius, "cluster_radius",  0.1,  50.0)
    _validate_numeric(req.batch_size,     "batch_size",      1.0, 128.0)

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return {"status": "error", "message": "Directory path is invalid or cannot be reached."}

    recording_dirs = []
    try:
        # 1. Check if the directory itself is a Recording folder
        is_direct_recording = any("Camera" in item and os.path.isdir(os.path.join(folder_path, item)) for item in os.listdir(folder_path))
                
        if is_direct_recording:
            recording_dirs.append({
                "name": os.path.basename(folder_path),
                "path": folder_path
            })
        else:
            # 2. Iterate as a Master Directory
            for subd in os.listdir(folder_path):
                full_path = os.path.join(folder_path, subd)
                if os.path.isdir(full_path):
                    has_cameras = any("Camera" in inner_d and os.path.isdir(os.path.join(full_path, inner_d)) for inner_d in os.listdir(full_path))
                    
                    if has_cameras:
                        recording_dirs.append({
                            "name": subd,
                            "path": full_path
                        })
    except Exception as e:
        return {"status": "error", "message": f"Error scanning directory: {str(e)}"}

    if not recording_dirs:
         return {"status": "error", "message": "No valid recording directories containing 'CameraX' folders were found."}

    sessions_started = []
    
    if req.execution_mode == "sequential":
        sessions_to_run = []
        for rec in recording_dirs:
            session_id = str(uuid.uuid4())
            st = load_state(session_id)
            st["source_folder"] = rec["path"]
            st["las_folder"] = las_folder_path
            st["is_pending"] = True
            
            dynamic_output_dir = os.path.join(tempfile.gettempdir(), f"Gauss_App_Staging_{session_id}")
            os.makedirs(dynamic_output_dir, exist_ok=True)
            st["current_output_dir"] = dynamic_output_dir
            save_state(session_id, st)
            
            sessions_started.append({"session_id": session_id, "folder_name": rec["name"], "path": rec["path"]})
            sessions_to_run.append({
                "session_id": session_id, "folder_path": rec["path"], "las_folder_path": las_folder_path, 
                "conf": req.min_confidence, "cluster": req.cluster_radius, "batch": req.batch_size, "output": dynamic_output_dir
            })
            
        thread = threading.Thread(target=run_sequential_batch, args=(sessions_to_run,))
        thread.start()
        
    else:
        for rec in recording_dirs:
            session_id = str(uuid.uuid4())
            st = load_state(session_id)
            
            st["source_folder"] = rec["path"]
            st["las_folder"] = las_folder_path
            
            dynamic_output_dir = os.path.join(tempfile.gettempdir(), f"Gauss_App_Staging_{session_id}")
            os.makedirs(dynamic_output_dir, exist_ok=True)
            
            st["current_output_dir"] = dynamic_output_dir
            save_state(session_id, st)
            
            thread = threading.Thread(
                target=run_subprocess, 
                args=(session_id, rec["path"], las_folder_path, req.min_confidence, req.cluster_radius, req.batch_size, dynamic_output_dir)
            )
            thread.start()
            
            sessions_started.append({
                "session_id": session_id,
                "folder_name": rec["name"],
                "path": rec["path"]
            })
        
    return {"status": "started", "sessions": sessions_started}

@app.get("/api/status")
def get_status(session_id: str):
    st = load_state(session_id)
    return {
        "session_id": session_id,
        "is_running": st["is_running"],
        "is_pending": st.get("is_pending", False),
        "is_cancelled": st.get("is_cancelled", False),
        "progress": min(st["progress"], 100),
        "camera_progress": st["camera_progress"],
        "logs": st["logs"][-20:],
        "results_ready": st["results_ready"]
    }

@app.get("/images/{session_id}/{image_name}")
def get_image(session_id: str, image_name: str):
    # validate session_id is a real UUID before using it in file operations
    session_id = _validate_session_id(session_id)

    st = load_state(session_id)
    if not st.get("current_output_dir"):
        return Response(status_code=404)

    output_dir = os.path.realpath(st["current_output_dir"])

    # resolve the full path and then confirm it's still inside the output directory.
    # without this check, a crafted image_name like '../../etc/passwd' would let
    # anyone read arbitrary files from the server.
    image_path = os.path.realpath(os.path.join(output_dir, image_name))
    if not image_path.startswith(output_dir + os.sep):
        logger.warning(f"Path traversal attempt blocked: {image_name}")
        return Response(status_code=400)

    if os.path.exists(image_path) and os.path.isfile(image_path):
        return FileResponse(image_path)
    return Response(status_code=404)

@app.get("/api/results")
def get_results(session_id: str):
    st = load_state(session_id)
    if not st["results_ready"] or not st["current_output_dir"]:
        return {"status": "not_ready"}
    try:
        json_path = os.path.join(st["current_output_dir"], "Export_Camine.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        return {"error": str(e)}
    return []

# Removed deprecated /api/delete_result - Frontend now handles logical Soft Delete in app state!

@app.post("/api/generate_final_export")
def generate_final_export(req: FinalExportRequest):
    st = load_state(req.session_id)
    if not st["current_output_dir"]:
        return {"status": "error", "message": "No active directory"}
    
    # We explicitly accept only verifications sent by the payload, bypassing the trash bin entirely!
    final_data = [d for d in req.results if d.get('verified')]
    
    if not final_data:
        return {"status": "error", "message": "No verified detections to export."}
        
    df = pd.DataFrame(final_data)

    # Round coordinates to 4 decimal places
    for col in ("x", "y", "z", "lat", "lon"):
        if col in df.columns:
            df[col] = df[col].round(4)

    # Inject required empty columns for Camine schema
    null_cols = ["Id_Camin", "CT", "CR1", "CR2", "CR3", "CR4"]
    for col in null_cols:
        df[col] = ""

    retain_cols = [
        "image", "x", "y", "z", "lat", "lon", "lidar_hit", "px_edge_flag",
        "range_m", "conf", "cam_key",
    ] + null_cols

    drop_cols = [c for c in df.columns if c not in retain_cols]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Prefer WGS84 (lon/lat) — avoids the EPSG:3844 axis-order swap in QGIS 3.x.
    if 'lon' in df.columns and 'lat' in df.columns:
        geometry = gpd.points_from_xy(df['lon'], df['lat'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        geometry = gpd.points_from_xy(df['x'], df['y'], z=df['z'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.set_crs(epsg=3844, inplace=True)

    base_name = "Export_Camine"

    json_path = os.path.join(st["current_output_dir"], f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f)

    geo_path = os.path.join(st["current_output_dir"], f"{base_name}.geojson")
    gdf.to_file(pathlib.Path(geo_path), driver="GeoJSON")

    return {"status": "success"}

@app.get("/api/download_geojson")
def download_geojson(session_id: str):
    st = load_state(session_id)
    target_dir = st["current_output_dir"]
    base_name = "Export_Camine"
    
    if not target_dir or not os.path.exists(os.path.join(target_dir, f"{base_name}.geojson")):
        return Response(content="GeoJSON not generated yet", status_code=404)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    final_output_dir = os.path.join(desktop_path, f"Gauss_Results_{session_id}")
    os.makedirs(final_output_dir, exist_ok=True)

    zip_path = os.path.join(final_output_dir, f"{base_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for ext in ['.geojson', '.json']:
            file_path = os.path.join(target_dir, f"{base_name}{ext}")
            if os.path.exists(file_path):
                import shutil
                shutil.copy2(file_path, os.path.join(final_output_dir, f"{base_name}{ext}"))
                zip_file.write(file_path, arcname=f"{base_name}{ext}")
                
    return FileResponse(
        path=zip_path, 
        filename=f"{base_name}.zip", 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={base_name}.zip"}
    )

@app.get("/api/download_briefcase")
def download_briefcase(session_id: str):
    st = load_state(session_id)
    target_dir = st.get("current_output_dir")
    source_dir = st.get("source_folder")
    
    if not target_dir or not os.path.exists(os.path.join(target_dir, "Export_Camine.json")):
        return Response(content="Results not generated yet", status_code=404)
        
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    final_output_dir = os.path.join(desktop_path, f"Gauss_Briefcase_{session_id}")
    os.makedirs(final_output_dir, exist_ok=True)
    
    zip_path = os.path.join(final_output_dir, "Gauss_Briefcase.zip")
    
    # ZIP_STORED: JPEGs are already compressed — DEFLATE gains nothing but wastes
    # 60-90s of CPU time, which can cause the process to be killed mid-write.
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zip_file:
        # Add the JSON (small — compress it)
        json_path = os.path.join(target_dir, "Export_Camine.json")
        zip_file.write(json_path, arcname="Export_Camine.json", compress_type=zipfile.ZIP_DEFLATED)
        
        # Read the JSON to find which images are needed
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            # Keep track of unique (cam_key, image) pairs
            unique_photos = set()
            for r in results:
                if r.get("image") and r.get("cam_key"):
                    unique_photos.add((r["cam_key"], r["image"]))
                # Include clustered images as well
                for member in r.get("cluster_members", []):
                    if member.get("image") and member.get("cam_key"):
                        unique_photos.add((member["cam_key"], member["image"]))
            
            # Add all required images.
            # Primary source: annotated copies in target_dir (guaranteed to exist —
            # they were just written by the pipeline).  These already have the bbox
            # rectangle drawn, which is ideal for the validator review UI.
            # Fallback: original file from source_dir (useful if target_dir is cleaned).
            for cam_key, img_name in unique_photos:
                annotated = os.path.join(target_dir, f"{cam_key}_{img_name}")
                original  = (os.path.join(source_dir, cam_key, img_name)
                             if source_dir and os.path.exists(source_dir) else "")
                if os.path.exists(annotated):
                    zip_file.write(annotated, arcname=f"{cam_key}/{img_name}")
                elif original and os.path.exists(original):
                    zip_file.write(original, arcname=f"{cam_key}/{img_name}")
        except Exception as e:
            logger.error(f"Failed to package briefcase images: {e}")
            
    return FileResponse(
        path=zip_path, 
        filename="Gauss_Briefcase.zip", 
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=Gauss_Briefcase.zip"}
    )

@app.post("/api/cancel")
def cancel_analysis(session_id: str = Query(...)):
    st = load_state(session_id)
    
    if pid := st.get("process_id"):
        with contextlib.suppress(Exception):
            os.kill(pid, signal.SIGTERM)
            
    st["is_running"] = False
    st["is_pending"] = False
    st["is_cancelled"] = True
    st["camera_progress"] = {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0}
    st["logs"].append("ERROR: Analysis cancelled by user.")
    st["process_id"] = None
    save_state(session_id, st)
    return {"status": "cancelled"}