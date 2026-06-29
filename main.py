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
    GET   /                      Health check

Design Notes:
    - Video processing (YOLO + OpenCV) is CPU/GPU-bound — it BLOCKS
      the thread. We run it in a ThreadPoolExecutor so the FastAPI
      async event loop stays free to handle other HTTP requests.
    - asyncio.gather() processes multiple videos CONCURRENTLY (true
      parallelism via threads, not just interleaving).
    - One global `MultiVideoTracker` instance is reused across all
      jobs to avoid reloading heavy models on every request.
    - Job state is stored in a plain Python dict (JOB_REGISTRY).
      For a production system you'd use Redis or a task queue (Celery).
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import PersonTrack, SessionLocal, create_tables, get_db
from tracker import MultiVideoTracker

# ─────────────────────────────────────────────────────────────────
# DIRECTORY SETUP  (created before app so lifespan can reference them)
# ─────────────────────────────────────────────────────────────────

UPLOAD_DIR = "uploads"    # Raw uploaded videos (one subfolder per job)
OUTPUT_DIR = "outputs"    # Processed annotated videos (one subfolder per job)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────

# Job registry: tracks status of each processing job in memory.
# Schema per entry:
#   {
#     "status":     "pending" | "processing" | "done" | "error"
#     "videos":     [<filename>, ...]          # Input filenames
#     "results":    [{summary_dict}, ...]      # Output summaries
#     "error":      <str> | None
#     "started_at": <ISO timestamp>
#     "done_at":    <ISO timestamp> | None
#   }
JOB_REGISTRY: dict = {}

# Thread pool for CPU-bound video processing tasks.
# max_workers=2 means 2 videos can process in parallel at once.
# Increase if you have more CPU cores / GPU memory.
THREAD_POOL = ThreadPoolExecutor(max_workers=2)

# Single shared tracker instance (models loaded once at startup)
tracker: Optional[MultiVideoTracker] = None


# ─────────────────────────────────────────────────────────────────
# LIFESPAN — Modern FastAPI startup / shutdown (replaces @app.on_event)
# ─────────────────────────────────────────────────────────────────
# The @asynccontextmanager pattern is the recommended approach in
# FastAPI 0.93+ and avoids DeprecationWarning from on_event.
# Everything BEFORE `yield` runs at startup; AFTER yield runs at shutdown.

@asynccontextmanager
async def lifespan(app_instance):
    """Startup: load models + DB.  Shutdown: drain thread pool."""
    global tracker

    # ── STARTUP ──────────────────────────────────────────────────
    print("[Server] Starting MCMOT API server...")
    # Create the person_tracks table if it doesn't exist yet
    create_tables()
    # Load YOLOv8n + MobileNetV3 onto GPU (or CPU as fallback)
    # Done here so the first request is fast (no cold-start delay)
    tracker = MultiVideoTracker(similarity_threshold=0.70, device="cuda")
    # Load Re-ID cache from database on startup
    db = SessionLocal()
    try:
        tracker.load_cache_from_db(db)
    finally:
        db.close()
    print("[Server] ✓ Ready — http://localhost:8000/docs")

    yield  # Server is now running and handling requests

    # ── SHUTDOWN ─────────────────────────────────────────────────
    # Wait for any in-progress video processing threads to finish
    THREAD_POOL.shutdown(wait=True)
    print("[Server] Thread pool shut down cleanly.")


# ─────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MCMOT API",
    description=(
        "Multi-Camera Multi-Object Tracking with Re-Identification.\n\n"
        "Upload multiple videos, process them concurrently with YOLOv8 + "
        "ByteTrack, and assign global person IDs across cameras."
    ),
    version="1.0.0",
    docs_url="/docs",   # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc", # ReDoc UI    → http://localhost:8000/redoc
    lifespan=lifespan,  # Attach the lifespan context manager
)

# Allow all origins so the Streamlit frontend (different port) can call this API.
# In production, restrict this to your specific domain.
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
        "message": "MCMOT Server is running",
        "docs": "http://localhost:8000/docs",
    }


@app.post("/upload-videos/", tags=["Processing"], summary="Upload and process videos")
async def upload_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more video files (.mp4, .avi)"),
    similarity_threshold: float = 0.70,
    frame_skip: int = 1,
) -> dict:
    """
    Accept multiple video uploads and start asynchronous processing.

    Processing runs in the background — this endpoint returns immediately
    with a `job_id`. Poll `/status/{job_id}` to track progress.

    Args:
        files:               List of video files to process.
        similarity_threshold: Re-ID cosine similarity threshold (0.5–0.99).
        frame_skip:          Process every Nth frame (1=all, 2=every other).
                             Use higher values for faster (less accurate) results.

    Returns:
        job_id and polling URL.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate file types
    for f in files:
        ext = os.path.splitext(f.filename)[-1].lower()
        if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {f.filename}. Use .mp4, .avi, .mov"
            )

    # Create unique job ID and directories
    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)

    # Save uploaded files to disk asynchronously
    # aiofiles.open() doesn't block the event loop while writing
    saved_paths: List[str] = []
    for file in files:
        # Sanitize filename (replace spaces with underscores)
        safe_name = file.filename.replace(" ", "_")
        dest_path = os.path.join(job_upload_dir, safe_name)
        content = await file.read()  # Read into memory first
        async with aiofiles.open(dest_path, "wb") as f_out:
            await f_out.write(content)
        saved_paths.append(dest_path)
        print(f"[Upload] Saved: {dest_path} ({len(content)/1024:.1f} KB)")

    # Register job state
    JOB_REGISTRY[job_id] = {
        "status": "pending",
        "videos": [os.path.basename(p) for p in saved_paths],
        "video_progress": {
            # Per-video progress — updated in real-time by progress_callback
            os.path.basename(p): {
                "frames_done": 0,
                "total_frames": 0,
                "status": "pending",   # pending | processing | done | error
            }
            for p in saved_paths
        },
        "results": [],
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "done_at": None,
        "similarity_threshold": similarity_threshold,
        "frame_skip": frame_skip,
    }

    # Update tracker settings for this job
    tracker.update_threshold(similarity_threshold)
    # tracker.reset_reid_cache()  # Commented out to enable persistent cross-job Re-ID

    # Schedule background processing
    background_tasks.add_task(
        _process_videos_job,
        job_id,
        saved_paths,
        job_output_dir,
        frame_skip,
    )

    return {
        "job_id": job_id,
        "message": f"Processing {len(files)} video(s) in the background.",
        "poll_url": f"/status/{job_id}",
        "settings": {
            "similarity_threshold": similarity_threshold,
            "frame_skip": frame_skip,
        },
    }


async def _process_videos_job(
    job_id: str,
    video_paths: List[str],
    output_dir: str,
    frame_skip: int,
) -> None:
    """
    Background coroutine: process all videos concurrently.

    Why asyncio + ThreadPoolExecutor?
        - YOLO/OpenCV are synchronous (blocking) C++ extensions.
        - Running them directly in `async` code would block the event loop.
        - run_in_executor() offloads each video to a thread, allowing
          true concurrent processing (limited by Python GIL for pure Python,
          but YOLO/OpenCV release the GIL during heavy computation).
        - asyncio.gather() waits for ALL threads to finish before marking done.
    """
    JOB_REGISTRY[job_id]["status"] = "processing"
    
    # Load Re-ID cache from the database before starting video threads
    db = SessionLocal()
    try:
        tracker.load_cache_from_db(db)
    except Exception as exc:
        print(f"[Job {job_id}] Warning loading cache: {exc}")
    finally:
        db.close()

    try:
        loop = asyncio.get_event_loop()
        tasks = []

        for video_path in video_paths:
            output_filename = f"processed_{os.path.basename(video_path)}"
            # Ensure output is always .mp4 (even if input was .avi)
            output_filename = os.path.splitext(output_filename)[0] + ".mp4"
            output_path = os.path.join(output_dir, output_filename)
            video_key = os.path.basename(video_path)

            # Mark this video as "processing" in the registry
            JOB_REGISTRY[job_id]["video_progress"][video_key]["status"] = "processing"

            # run_in_executor submits the blocking function to the thread pool
            task = loop.run_in_executor(
                THREAD_POOL,
                _sync_track_video,      # The blocking function
                video_path,             # Arguments to pass to it
                output_path,
                frame_skip,
                job_id,                 # For progress callback
                video_key,              # For progress callback
            )
            tasks.append(task)

        # Await all tasks concurrently — returns when ALL are done
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any errors in the results
        summaries = []
        errors = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Video {idx + 1}: {str(result)}")
                # Mark the specific video as errored if we can identify it
                video_keys = list(JOB_REGISTRY[job_id]["video_progress"].keys())
                if idx < len(video_keys):
                    JOB_REGISTRY[job_id]["video_progress"][video_keys[idx]]["status"] = "error"
            else:
                summaries.append(result)
                # Mark the specific video as done
                v_source = result.get("video_source", "")
                if v_source in JOB_REGISTRY[job_id]["video_progress"]:
                    JOB_REGISTRY[job_id]["video_progress"][v_source]["status"] = "done"

        if errors:
            JOB_REGISTRY[job_id]["status"] = "error"
            JOB_REGISTRY[job_id]["error"] = " | ".join(errors)
        else:
            JOB_REGISTRY[job_id]["status"] = "done"
            JOB_REGISTRY[job_id]["results"] = summaries
            JOB_REGISTRY[job_id]["done_at"] = datetime.utcnow().isoformat()

        print(f"[Job {job_id}] ✓ Completed. {len(summaries)} video(s) processed.")

    except Exception as exc:
        JOB_REGISTRY[job_id]["status"] = "error"
        JOB_REGISTRY[job_id]["error"] = str(exc)
        print(f"[Job {job_id}] ✗ Fatal error: {exc}")


def _sync_track_video(
    video_path: str,
    output_path: str,
    frame_skip: int,
    job_id: str,
    video_key: str,
) -> dict:
    """
    Synchronous (blocking) wrapper called from the thread pool.
    This is the actual CPU-heavy work that runs in a separate thread.
    Each thread creates and manages its own independent database session.

    A progress_callback is wired up here so that track_video() can
    report (frames_done, total_frames) every 10 frames, allowing the
    frontend to show per-video progress bars in real-time.
    """
    def _progress_cb(frames_done: int, total_frames: int) -> None:
        """Write progress into the shared JOB_REGISTRY (thread-safe for dict key writes)."""
        try:
            vp = JOB_REGISTRY[job_id]["video_progress"][video_key]
            vp["frames_done"]  = frames_done
            vp["total_frames"] = total_frames
        except KeyError:
            pass  # Job was cleared — silently ignore

    db = SessionLocal()
    try:
        # Seed total_frames immediately from the video header so the UI
        # can show a denominator before the first 10-frame callback fires.
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
            frame_skip=frame_skip,
            progress_callback=_progress_cb,
        )
    finally:
        db.close()


@app.get("/status/{job_id}", tags=["Processing"], summary="Poll job status")
def get_job_status(job_id: str) -> dict:
    """
    Check the current status of a processing job.

    Status values:
        pending    → Job queued, not yet started
        processing → Videos are currently being processed
        done       → All videos processed successfully
        error      → Processing failed (see 'error' field for details)
    """
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
                        "job_id": job_folder,
                        "filename": fname,
                        "size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
                        "url": f"/video/{job_folder}/{fname}",
                    })
    return {"count": len(all_outputs), "outputs": all_outputs}


@app.get("/video/{job_id}/{filename}", tags=["Results"], summary="Stream a processed video")
def serve_video(job_id: str, filename: str) -> FileResponse:
    """
    Stream a processed output video file.
    FastAPI's FileResponse handles HTTP range requests automatically,
    which enables HTML5 video seeking.
    """
    # Sanitize path to prevent directory traversal attacks
    safe_filename = os.path.basename(filename)
    video_path = os.path.join(OUTPUT_DIR, job_id, safe_filename)

    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {filename}"
        )
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=safe_filename,
    )


@app.get("/tracks/", tags=["Database"], summary="Fetch latest tracking records")
def get_tracks(
    limit: int = 200,
    video_source: Optional[str] = None,
    global_person_id: Optional[int] = None,
    db: Session = Depends(get_db),
) -> list:
    """
    Fetch the most recent tracking records from PostgreSQL.

    Args:
        limit:            Max rows to return (default 200, max 1000).
        video_source:     Filter by video filename (optional).
        global_person_id: Filter by specific global person ID (optional).
    """
    limit = min(limit, 1000)  # Hard cap to prevent huge responses
    query = db.query(PersonTrack)

    if video_source:
        query = query.filter(PersonTrack.video_source == video_source)
    if global_person_id is not None:
        query = query.filter(PersonTrack.global_person_id == global_person_id)

    records = query.order_by(PersonTrack.detected_at.desc()).limit(limit).all()

    return [
        {
            "id": r.id,
            "global_person_id": r.global_person_id,
            "local_track_id": r.local_track_id,
            "video_source": r.video_source,
            "frame_number": r.frame_number,
            "bbox_coordinates": r.bbox_coordinates,
            "confidence": r.confidence,
            "detected_at": r.detected_at.isoformat() if r.detected_at else None,
        }
        for r in records
    ]


@app.get("/stats/", tags=["Database"], summary="Aggregate tracking statistics")
def get_stats(db: Session = Depends(get_db)) -> dict:
    """Return high-level statistics from the person_tracks table."""
    total_detections = db.query(func.count(PersonTrack.id)).scalar() or 0
    unique_persons = (
        db.query(func.count(PersonTrack.global_person_id.distinct())).scalar() or 0
    )
    unique_sources = (
        db.query(func.count(PersonTrack.video_source.distinct())).scalar() or 0
    )
    avg_confidence = (
        db.query(func.avg(PersonTrack.confidence)).scalar() or 0.0
    )
    return {
        "total_detections": total_detections,
        "unique_global_persons": unique_persons,
        "unique_video_sources": unique_sources,
        "average_confidence": round(float(avg_confidence), 4),
    }


@app.delete("/tracks/", tags=["Database"], summary="Clear all tracking records")
def clear_tracks(db: Session = Depends(get_db)) -> dict:
    """
    Delete ALL rows from person_tracks.
    Useful for resetting the database between experiments.
    """
    count = db.query(PersonTrack).count()
    db.query(PersonTrack).delete()
    from database import PersonIdentity
    db.query(PersonIdentity).delete()
    db.commit()
    
    # Reset the tracker's in-memory cache
    if tracker:
        tracker.reset_reid_cache()
        
    return {"deleted_rows": count, "message": "All tracking records and identities deleted."}


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT — allows running directly with `python main.py`
# ─────────────────────────────────────────────────────────────────
# When you run `python main.py`, Python sets __name__ to "__main__".
# This block starts a uvicorn ASGI server programmatically.
# Equivalent to:  uvicorn main:app --reload --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",       # Module:variable pointing to the FastAPI instance
        host="0.0.0.0",   # Listen on all interfaces (use 127.0.0.1 for local-only)
        port=8000,
        reload=True,      # Auto-restart when code changes (great for development)
        log_level="info",
    )
