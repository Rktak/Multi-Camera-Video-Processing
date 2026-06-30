# Project Architecture for AI Assistants

This document is intended for LLMs and AI coding assistants to quickly understand the architecture, models, and flow of the Multi-Camera Multi-Object Tracking (MCMOT) System v2.

## Tech Stack
- **Backend:** FastAPI (async REST API), Uvicorn, Python 3
- **Frontend:** Streamlit (`app.py`)
- **Database:** PostgreSQL via SQLAlchemy (sync engine)
- **Computer Vision:** OpenCV, Ultralytics YOLOv8, Torchreid (OSNet), ByteTrack

## Core Components (`tracker.py`)
The tracking logic lives entirely in `tracker.py` and is heavily object-oriented.

1. **`PersonDetectorTracker`**: Wraps Ultralytics `YOLOv8n`. Uses its built-in ByteTrack (`model.track()`) to assign local `track_id`s within a single video. It borrows/returns YOLO instances from a thread-safe Queue pool to avoid reloading weights.
2. **`LocalTrackProfile`**: As a person is tracked across frames by ByteTrack, this class accumulates cropped images of them. It maintains an "appearance gallery" (default size 8). It filters bad crops (low confidence, too small).
3. **`ReIDFeatureExtractor`**: Takes the crops from `LocalTrackProfile` and runs them through a Re-Identification CNN (default `osnet_ain`, fallback `mobilenetv3`). Returns a 512-dim (or 576-dim) L2-normalized embedding tensor.
4. **`GlobalIdentityManager`**: The core Re-ID logic. Matches the `LocalTrackProfile` gallery against known global galleries stored in memory (and DB). Scoring uses three factors:
   - Appearance (Cosine Similarity)
   - Temporal consistency (Are they in two places at once?)
   - Camera transition (Did they travel fast enough between cameras based on topology?)
5. **`MultiVideoTrackingEngine`**: The orchestrator. Creates the YOLO pool, handles video reading/writing (with `cv2.VideoCapture`), frame downscaling, and DB flushing.

## Database Schema (`database.py`)
- **`PersonTrack` (`person_tracks`)**: A row for every single bounding box detection in every frame. Very high volume.
- **`PersonIdentity` (`person_identities`)**: A single row for each unique global person. Stores their `name` (if taught) and their `embedding` gallery as a JSON array of floats.

## Operating Modes

### Unsupervised Mode (Clustering)
If `supervised_mode = False`, the `GlobalIdentityManager` will dynamically cluster people. If a person's embedding doesn't match existing global IDs above the `similarity_threshold`, it spawns a brand new Global ID and saves their gallery to the DB.

### Supervised Mode (Teaching)
If `supervised_mode = True`, the `GlobalIdentityManager` **disables** clustering and dynamic gallery updates. It will only match against identities that have an explicit `name` in the DB. If a person falls below the threshold, they are assigned `global_id = -1` (labeled "Unknown" in red).
- **Teach Endpoint**: `/teach-person/` in `main.py`. This endpoint processes a single video of a specific person with `supervised_mode=False` and `similarity_threshold=1.0` to force the creation of a new ID. It then updates that ID in the database with the provided `name`.

## Performance Optimizations Applied
- **Frame Downscaling**: Videos larger than 1280px in width are automatically resized using `cv2.resize` in `track_video()` before being passed to YOLO.
- **FP16 Inference**: YOLO is instantiated with `half=True` if `device="cuda"`, halving VRAM usage and doubling inference speed on modern GPUs.
- **Batched Re-ID**: `ReIDFeatureExtractor.extract_batch()` processes multiple bounding box crops in a single forward pass tensor.
- **Asynchronous Execution**: In `main.py`, multiple videos are processed concurrently using a `ThreadPoolExecutor`, limited by `MAX_WORKERS`.
