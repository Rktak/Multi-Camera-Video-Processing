"""
main.py - FastAPI Backend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

How to run:
    uvicorn main:app --reload --port 8000

Interactive API docs (Swagger UI):
    http://localhost:8000/docs

Endpoints:
    POST  /upload-videos/        Upload videos and start processing job
    GET   /status/{job_id}       Poll job progress
    GET   /results/              List all processed output videos
    GET   /video/{job_id}/{file} Stream a processed output video
    GET   /tracks/               Fetch latest DB records (JSON)
    GET   /stats/                Aggregate DB statistics
    GET   /diagnostics/          Per-video track-level diagnostics  [NEW v2]
    GET   /                      Health check

CHANGED (v2 — track-level Re-ID upgrade):
    • POST /upload-videos/ accepts new optional parameters:
        - gallery_size         (int,   default 8)
        - weight_appearance    (float, default 0.70)
        - weight_temporal      (float, default 0.20)
        - weight_transition    (float, default 0.10)
        - camera_topology      (str,   JSON string, optional)
        - min_crop_quality     (float, default 0.40)
    • GET /diagnostics/ returns per-video diagnostics from the engine.
    • _sync_track_video passes TrackingConfig to the engine.
    • tracker.track_video() now returns richer summary with local track count.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

import aiofiles
from dotenv import load_dotenv
load_dotenv()
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import PersonTrack, SessionLocal, create_tables, get_db
from tracker import MultiVideoTracker, TrackingConfig

# ─────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────

# Job registry: tracks status of each processing job in memory.
JOB_REGISTRY: dict = {}

# Thread pool for CPU-bound video processing tasks.
THREAD_POOL = ThreadPoolExecutor(max_workers=2)

# Single shared tracker instance (models loaded once at startup)
tracker: Optional[MultiVideoTracker] = None


# ─────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_instance):
    """Startup: load models + DB.  Shutdown: drain thread pool."""
    global tracker

    print("[Server] Starting MCMOT API server (v2 — track-level Re-ID)...")
    create_tables()

    # Build default config from environment (overridden per-job via API params)
    yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt")
    reid_model = os.getenv("REID_MODEL", "osnet_ain")
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "480"))
    reid_half = os.getenv("REID_HALF", "true").lower() in ("true", "1", "yes")
    default_config = TrackingConfig(
        similarity_threshold=0.70,
        device="auto",
        yolo_model=yolo_model,
        yolo_imgsz=yolo_imgsz,
        reid_model=reid_model,
        reid_half=reid_half,
    )
    tracker = MultiVideoTracker(config=default_config)

    # Restore identity gallery from DB
    db = SessionLocal()
    try:
        tracker.load_cache_from_db(db)
    finally:
        db.close()

    print("[Server] ✓ Ready — http://localhost:8000/docs")
    yield

    THREAD_POOL.shutdown(wait=True)
    print("[Server] Thread pool shut down cleanly.")


# ─────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MCMOT API",
    description=(
        "Multi-Camera Multi-Object Tracking with Re-Identification (v2).\n\n"
        "Upload multiple videos, process them concurrently with YOLOv8 + "
        "ByteTrack, build per-track appearance galleries, and assign global "
        "person IDs across cameras using appearance + temporal + topology scoring."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Health check")
def root() -> dict:
    """Returns server status. Use this to verify the backend is running."""
    return {
        "status": "ok",
        "message": "MCMOT Server v2 is running (track-level Re-ID)",
        "docs": "http://localhost:8000/docs",
    }

from fastapi import Form, BackgroundTasks, Depends
from database import get_db, PersonTrack, PersonIdentity
from sqlalchemy import func

@app.post("/teach-person/", tags=["Processing"], summary="Teach a new person identity")
async def teach_person(
    person_name: str = Form(..., description="Name of the person"),
    file: UploadFile = File(..., description="Video of the person"),
    db = Depends(get_db)
) -> dict:
    """
    Teach the system a specific person. The person who appears the longest
    in the video will be registered with the given name.
    """
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # Save file
    teach_dir = os.path.join(UPLOAD_DIR, "teaching")
    os.makedirs(teach_dir, exist_ok=True)
    safe_name = f"teach_{int(datetime.utcnow().timestamp())}_{file.filename.replace(' ', '_')}"
    dest_path = os.path.join(teach_dir, safe_name)
    
    content = await file.read()
    with open(dest_path, "wb") as f_out:
        f_out.write(content)

    print(f"[Teach] Processing {dest_path} for person: {person_name}")

    # Build a tracker that will NOT merge (threshold=1.0)
    yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt")
    reid_model = os.getenv("REID_MODEL", "osnet_ain")
    teach_config = TrackingConfig(
        similarity_threshold=1.0,  # Never merge
        supervised_mode=False,     # Allow creating new IDs
        device="auto",
        yolo_model=yolo_model,
        reid_model=reid_model,
    )
    
    from tracker import MultiVideoTrackingEngine
    engine = MultiVideoTrackingEngine(teach_config)
    engine.load_cache_from_db(db)
    
    def dummy_callback(frames, total): pass
    
    # Process the video
    dummy_output = os.path.join(teach_dir, f"out_{safe_name}")
    engine.track_video(dest_path, dummy_output, db, 50, 1, dummy_callback)

    # Find the most prominent person in this video
    top_person = (
        db.query(PersonTrack.global_person_id, func.count(PersonTrack.id).label('ct'))
        .filter(PersonTrack.video_source == safe_name)
        .group_by(PersonTrack.global_person_id)
        .order_by(func.count(PersonTrack.id).desc())
        .first()
    )

    if not top_person:
        return {"status": "error", "message": "No person detected in the video."}

    gid = top_person.global_person_id
    
    # Assign the name
    identity = db.query(PersonIdentity).filter(PersonIdentity.global_person_id == gid).first()
    if identity:
        identity.name = person_name
        db.commit()
        
    # Reload global tracker cache so it knows about the new name
    global tracker
    with tracker.identity_mgr._lock:
        tracker.identity_mgr._names[gid] = person_name

    return {
        "status": "success", 
        "message": f"Successfully registered '{person_name}' (Global ID: {gid})",
        "global_id": gid
    }


@app.post("/upload-videos/", tags=["Processing"], summary="Upload and process videos")
async def upload_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more video files (.mp4, .avi, .mov)"),
    # ── Existing parameters ────────────────────────────────────────────
    similarity_threshold: float = 0.70,
    frame_skip: int = 1,
    # ── New v2 parameters ──────────────────────────────────────────────
    gallery_size: int = 8,
    weight_appearance: float = 0.70,
    weight_temporal: float = 0.20,
    weight_transition: float = 0.10,
    camera_topology: Optional[str] = None,
    min_crop_quality: float = 0.40,
    reid_model: str = "osnet_ain",
    supervised_mode: bool = False,
    use_onnx: bool = False,
) -> dict:
    """
    Accept multiple video uploads and start asynchronous processing.

    Processing runs in the background — this endpoint returns immediately
    with a `job_id`. Poll `/status/{job_id}` to track progress.

    Args:
        files:                List of video files to process.
        similarity_threshold: Re-ID cosine similarity threshold (0.5–0.99).
        frame_skip:           Process every Nth frame (1=all, 2=every other).
        gallery_size:         [NEW v2] Max crops per local track gallery (4–16).
        weight_appearance:    [NEW v2] Weight for appearance score (default 0.70).
        weight_temporal:      [NEW v2] Weight for temporal consistency score (default 0.20).
        weight_transition:    [NEW v2] Weight for camera transition prior (default 0.10).
        camera_topology:      [NEW v2] JSON string describing camera travel times.
                              Example: '{"cam1": {"cam2": [2, 10]}}'
        min_crop_quality:     [NEW v2] Minimum YOLO confidence for gallery crops (default 0.40).
        supervised_mode:      If True, only taught identities are tracked. Others are Unknown.

    Returns:
        job_id and polling URL.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for f in files:
        ext = os.path.splitext(f.filename)[-1].lower()
        if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {f.filename}. Use .mp4, .avi, .mov"
            )

    # Parse camera topology JSON if provided
    topology_dict: Optional[dict] = None
    if camera_topology:
        try:
            topology_dict = json.loads(camera_topology)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid camera_topology JSON: {exc}"
            )

    # Create unique job ID and directories
    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)

    # Save uploaded files asynchronously
    saved_paths: List[str] = []
    for file in files:
        safe_name = file.filename.replace(" ", "_")
        dest_path = os.path.join(job_upload_dir, safe_name)
        content = await file.read()
        async with aiofiles.open(dest_path, "wb") as f_out:
            await f_out.write(content)
        saved_paths.append(dest_path)
        print(f"[Upload] Saved: {dest_path} ({len(content)/1024:.1f} KB)")

    # Build TrackingConfig for this job
    job_config = TrackingConfig(
        similarity_threshold=similarity_threshold,
        gallery_size=max(2, min(gallery_size, 20)),
        weight_appearance=weight_appearance,
        weight_temporal=weight_temporal,
        weight_transition=weight_transition,
        camera_topology=topology_dict,
        min_crop_confidence=min_crop_quality,
        frame_skip=frame_skip,
        device="auto",
        yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        yolo_imgsz=int(os.getenv("YOLO_IMGSZ", "480")),
        reid_model=reid_model,
        reid_half=os.getenv("REID_HALF", "true").lower() in ("true", "1", "yes"),
        supervised_mode=supervised_mode,
        use_onnx=use_onnx,
    )

    # Register job state
    JOB_REGISTRY[job_id] = {
        "status": "pending",
        "videos": [os.path.basename(p) for p in saved_paths],
        "video_progress": {
            os.path.basename(p): {
                "frames_done":  0,
                "total_frames": 0,
                "status":       "pending",
            }
            for p in saved_paths
        },
        "results": [],
        "error":   None,
        "started_at": datetime.utcnow().isoformat(),
        "done_at":    None,
        "settings": {
            "similarity_threshold": similarity_threshold,
            "frame_skip":           frame_skip,
            "gallery_size":         gallery_size,
            "weight_appearance":    round(job_config.weight_appearance, 3),
            "weight_temporal":      round(job_config.weight_temporal, 3),
            "weight_transition":    round(job_config.weight_transition, 3),
            "min_crop_quality":     min_crop_quality,
            "camera_topology":      topology_dict,
            "reid_model":           reid_model,
            "use_onnx":             use_onnx,
        },
    }

    # Update global tracker threshold (backward compat)
    tracker.update_threshold(similarity_threshold)

    background_tasks.add_task(
        _process_videos_job,
        job_id,
        saved_paths,
        job_output_dir,
        job_config,
    )

    return {
        "job_id": job_id,
        "message": f"Processing {len(files)} video(s) in the background.",
        "poll_url": f"/status/{job_id}",
        "settings": JOB_REGISTRY[job_id]["settings"],
    }


async def _process_videos_job(
    job_id: str,
    video_paths: List[str],
    output_dir: str,
    job_config: TrackingConfig,
) -> None:
    """
    Background coroutine: process all videos concurrently.

    Uses asyncio.gather() + ThreadPoolExecutor so YOLO/OpenCV
    (which release the GIL) run in true parallel threads.
    """
    JOB_REGISTRY[job_id]["status"] = "processing"

    # Load Re-ID cache from DB before processing
    db = SessionLocal()
    try:
        tracker.load_cache_from_db(db)
    except Exception as exc:
        print(f"[Job {job_id}] Warning loading cache: {exc}")
    finally:
        db.close()

    # Update tracker config for this job
    tracker.config = job_config
    tracker.identity_mgr.config = job_config

    try:
        loop = asyncio.get_event_loop()
        tasks = []

        for video_path in video_paths:
            output_filename = f"processed_{os.path.basename(video_path)}"
            output_filename = os.path.splitext(output_filename)[0] + ".mp4"
            output_path = os.path.join(output_dir, output_filename)
            video_key   = os.path.basename(video_path)

            JOB_REGISTRY[job_id]["video_progress"][video_key]["status"] = "processing"

            task = loop.run_in_executor(
                THREAD_POOL,
                _sync_track_video,
                video_path,
                output_path,
                job_config,
                job_id,
                video_key,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries, errors = [], []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Video {idx + 1}: {str(result)}")
                video_keys = list(JOB_REGISTRY[job_id]["video_progress"].keys())
                if idx < len(video_keys):
                    JOB_REGISTRY[job_id]["video_progress"][video_keys[idx]]["status"] = "error"
            else:
                summaries.append(result)
                v_source = result.get("video_source", "")
                if v_source in JOB_REGISTRY[job_id]["video_progress"]:
                    JOB_REGISTRY[job_id]["video_progress"][v_source]["status"] = "done"

        if errors:
            JOB_REGISTRY[job_id]["status"] = "error"
            JOB_REGISTRY[job_id]["error"]  = " | ".join(errors)
        else:
            JOB_REGISTRY[job_id]["status"]  = "done"
            JOB_REGISTRY[job_id]["results"] = summaries
            JOB_REGISTRY[job_id]["done_at"] = datetime.utcnow().isoformat()

        print(f"[Job {job_id}] ✓ Completed. {len(summaries)} video(s) processed.")

    except Exception as exc:
        JOB_REGISTRY[job_id]["status"] = "error"
        JOB_REGISTRY[job_id]["error"]  = str(exc)
        print(f"[Job {job_id}] ✗ Fatal error: {exc}")


def _sync_track_video(
    video_path: str,
    output_path: str,
    job_config: TrackingConfig,
    job_id: str,
    video_key: str,
) -> dict:
    """
    Synchronous (blocking) wrapper called from the thread pool.
    Each thread gets its own DB session. Progress is reported every 10 frames.
    """
    def _progress_cb(frames_done: int, total_frames: int) -> None:
        try:
            vp = JOB_REGISTRY[job_id]["video_progress"][video_key]
            vp["frames_done"]  = frames_done
            vp["total_frames"] = total_frames
        except KeyError:
            pass

    db = SessionLocal()
    try:
        # Pre-seed total_frames so the UI has a denominator immediately
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(video_path)
        if _cap.isOpened():
            _total = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()
            JOB_REGISTRY[job_id]["video_progress"][video_key]["total_frames"] = _total

        return tracker.track_video(
            video_path=video_path,
            output_path=output_path,
            db_session=db,
            db_batch_size=50,
            frame_skip=job_config.frame_skip,
            progress_callback=_progress_cb,
        )
    finally:
        db.close()


@app.get("/status/{job_id}", tags=["Processing"], summary="Poll job status")
def get_job_status(job_id: str) -> dict:
    """Check the current status of a processing job."""
    if job_id not in JOB_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOB_REGISTRY[job_id]


@app.get("/results/", tags=["Results"], summary="List all output videos")
def list_results() -> dict:
    """Return a list of all processed output video files across all jobs."""
    all_outputs = []
    if os.path.exists(OUTPUT_DIR):
        for job_folder in sorted(os.listdir(OUTPUT_DIR)):
            folder_path = os.path.join(OUTPUT_DIR, job_folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".mp4", ".avi")):
                    file_path = os.path.join(folder_path, fname)
                    all_outputs.append({
                        "job_id":   job_folder,
                        "filename": fname,
                        "size_mb":  round(os.path.getsize(file_path) / 1024 / 1024, 2),
                        "url":      f"/video/{job_folder}/{fname}",
                    })
    return {"count": len(all_outputs), "outputs": all_outputs}


@app.get("/video/{job_id}/{filename}", tags=["Results"], summary="Stream a processed video")
def serve_video(job_id: str, filename: str) -> FileResponse:
    """Stream a processed output video file (supports HTTP range requests for seeking)."""
    safe_filename = os.path.basename(filename)
    video_path    = os.path.join(OUTPUT_DIR, job_id, safe_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
    return FileResponse(video_path, media_type="video/mp4", filename=safe_filename)


@app.get("/tracks/", tags=["Database"], summary="Fetch latest tracking records")
def get_tracks(
    limit: int = 200,
    video_source: Optional[str] = None,
    global_person_id: Optional[int] = None,
    db: Session = Depends(get_db),
) -> list:
    """Fetch the most recent tracking records from PostgreSQL."""
    limit = min(limit, 1000)
    query = db.query(PersonTrack)
    if video_source:
        query = query.filter(PersonTrack.video_source == video_source)
    if global_person_id is not None:
        query = query.filter(PersonTrack.global_person_id == global_person_id)

    records = query.order_by(PersonTrack.detected_at.desc()).limit(limit).all()
    return [
        {
            "id":               r.id,
            "global_person_id": r.global_person_id,
            "local_track_id":   r.local_track_id,
            "video_source":     r.video_source,
            "frame_number":     r.frame_number,
            "bbox_coordinates": r.bbox_coordinates,
            "confidence":       r.confidence,
            "detected_at":      r.detected_at.isoformat() if r.detected_at else None,
            # v2 track-level fields (may be None for old rows)
            "track_start_frame": r.track_start_frame,
            "track_end_frame":   r.track_end_frame,
            "quality_score":     r.quality_score,
            "match_score":       r.match_score,
        }
        for r in records
    ]


@app.get("/stats/", tags=["Database"], summary="Aggregate tracking statistics")
def get_stats(db: Session = Depends(get_db)) -> dict:
    """Return high-level statistics from the person_tracks table."""
    total_detections = db.query(func.count(PersonTrack.id)).scalar() or 0
    unique_persons   = db.query(func.count(PersonTrack.global_person_id.distinct())).scalar() or 0
    unique_sources   = db.query(func.count(PersonTrack.video_source.distinct())).scalar() or 0
    avg_confidence   = db.query(func.avg(PersonTrack.confidence)).scalar() or 0.0
    avg_quality      = db.query(func.avg(PersonTrack.quality_score)).scalar() or 0.0
    avg_match        = db.query(func.avg(PersonTrack.match_score)).scalar() or 0.0
    return {
        "total_detections":      total_detections,
        "unique_global_persons": unique_persons,
        "unique_video_sources":  unique_sources,
        "average_confidence":    round(float(avg_confidence), 4),
        # v2 additions
        "average_quality_score": round(float(avg_quality), 4),
        "average_match_score":   round(float(avg_match), 4),
    }


@app.get("/diagnostics/", tags=["Database"], summary="Per-video track-level diagnostics")
def get_diagnostics() -> dict:
    """
    Return per-video diagnostics from the last processing run.

    Includes:
        - local_track_count  : number of ByteTrack local tracks found
        - global_id_count    : number of unique global IDs assigned
        - avg_match_score    : average combined matching score
        - unmatched_tracks   : tracks that got new IDs (no match found)

    Also returns the current state of the GlobalIdentityManager.
    """
    if tracker is None:
        return {"error": "Tracker not initialised."}

    per_video = tracker.get_diagnostics()
    global_stats = tracker.get_global_stats()

    return {
        "per_video":    per_video,
        "global_state": global_stats,
    }


@app.delete("/tracks/", tags=["Database"], summary="Clear all tracking records")
def clear_tracks(db: Session = Depends(get_db)) -> dict:
    """Delete ALL rows from person_tracks. Resets the tracker cache."""
    count = db.query(PersonTrack).count()
    db.query(PersonTrack).delete()
    from database import PersonIdentity
    db.query(PersonIdentity).delete()
    db.commit()

    if tracker:
        tracker.reset_reid_cache()

    return {"deleted_rows": count, "message": "All tracking records and identities deleted."}


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
