# 🎯 Multi-Camera Multi-Object Tracking (MCMOT) System v2

A complete end-to-end computer vision pipeline for tracking people across multiple video streams and assigning globally consistent identities using **track-level Re-Identification (Re-ID)**.

> **v2 upgrade**: Moved from frame-level cosine matching to a full **track-level gallery Re-ID** system with appearance + temporal + camera-transition scoring.

---

## 📁 Project Structure

```
VideoProcessing/
├── database.py       # SQLAlchemy ORM model + PostgreSQL connection
├── tracker.py        # Modular tracking engine — 6 classes (see Architecture)
├── main.py           # FastAPI backend (REST API + background processing)
├── app.py            # Streamlit frontend (UI dashboard)
├── requirements.txt  # All Python dependencies
├── .env              # PostgreSQL credentials (do not commit this!)
├── uploads/          # Raw uploaded videos (auto-created on first run)
└── outputs/          # Processed annotated videos (auto-created on first run)
```

---

## ⚙️ Setup Instructions

### 1. Create the PostgreSQL Database

Connect to your local PostgreSQL instance and create the database:

```sql
CREATE DATABASE mcmot_db;
```

The `.env` file is already configured for your setup:
```
DATABASE_URL=postgresql://postgres:admin@localhost:5432/mcmot_db
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Note:** If you have CUDA, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt` for the rest.

### 3. Start the FastAPI Backend

Open **Terminal 1**:
```bash
uvicorn main:app --reload --port 8000
```

On first startup the server will:
- Create / migrate the `person_tracks` table in PostgreSQL automatically
- Add new v2 columns (`track_start_frame`, `track_end_frame`, `quality_score`, `match_score`) safely to existing tables
- Download `yolov8n.pt` weights (~6 MB) if not present
- Download MobileNetV3-Small weights (~10 MB) if not present
- Restore identity galleries from DB if any exist

API Explorer (Swagger UI): http://localhost:8000/docs

### 4. Start the Streamlit Frontend

Open **Terminal 2**:
```bash
streamlit run app.py
```

Browser will open automatically at: http://localhost:8501

---

## 🚀 Usage

1. Upload **2 or more** `.mp4` / `.avi` / `.mov` video files in the **📤 Upload & Process** tab
2. Adjust settings in the sidebar (see [Tuning Parameters](#-key-tuning-parameters) below)
3. Click **🚀 Process Videos** and watch per-video progress bars
4. Switch to the **🎬 Results** tab to see annotated videos side-by-side
   - Overlay format: **`L{local} / G{global}`** — local ByteTrack ID + global Re-ID person ID
5. Check the **📊 Tracking Dashboard** tab to browse the PostgreSQL tracking log and live diagnostics

---

## 🧠 Architecture

The pipeline is split into six modular classes inside `tracker.py`:

```
Video Frame Loop
    │
    ▼
PersonDetectorTracker        YOLOv8 detection + ByteTrack → local_track_id
    │
    ▼
LocalTrackProfile            Accumulates high-quality crops into a gallery
                             (quality gating: min size, min confidence, near-duplicate guard)
    │
    ▼
ReIDFeatureExtractor         MobileNetV3-Small → 576-dim L2-normalised embedding
    │
    ▼  (deferred until track ends)
GlobalIdentityManager        Gallery-to-gallery matching with combined score:
                             score = w_app·appearance + w_temp·temporal + w_trans·transition
    │
    ▼
Annotated Video + PostgreSQL DB (with track metadata)
```

### `TrackingConfig` — Central Parameter Dataclass
All tunable parameters in one place. Weights are auto-normalised to sum to 1.0.

### `ReIDFeatureExtractor` — Pluggable Model
MobileNetV3-Small (576-dim) is the default. To swap in a stronger model (OSNet, ResNet-IBN, CLIP): subclass `ReIDFeatureExtractor`, override `_forward_batch()`, pass to `MultiVideoTrackingEngine`. No other changes needed.

### `LocalTrackProfile` — Per-Track Gallery
- Stores up to `gallery_size` (default 8) high-quality crops per ByteTrack local ID
- **Crop quality gate**: minimum size 32×64 px, minimum confidence 0.40, sampled every 5 frames
- **Near-duplicate guard**: skips crops with cosine similarity ≥ 0.97 to last gallery entry
- **Representative embedding**: L2-normalised mean of gallery
- **Quality score**: weighted combination of gallery fill ratio, mean confidence, mean crop area

### `GlobalIdentityManager` — Multi-Factor Scoring
```
final_score = w_appearance  × gallery-to-gallery cosine sim
            + w_temporal    × temporal compatibility score
            + w_transition  × camera topology prior
```

- **Temporal score**: 0.0 if person appears in two cameras simultaneously (impossible match); 0.9 for same-camera continuation; 1.0 for no overlap
- **Transition score**: uses optional camera topology config (`{"cam1.mp4": {"cam2.mp4": [2, 10]}}`) to check expected travel-time windows
- Thread-safe; persists gallery to `PersonIdentity` table across server restarts

### Detection: YOLOv8n
- Detects only `person` class (COCO class 0), conf ≥ 0.35, NMS IoU 0.45
- YOLO pool: 2 instances pre-loaded at startup to avoid weight reload overhead

### Database: PostgreSQL via SQLAlchemy
- Two tables: `person_tracks` (one row per detection) + `person_identities` (gallery persistence)
- Batch inserts every 50 records; track-level metadata updated after each video finalizes

---

## 📊 Database Schema

### `person_tracks`

```sql
CREATE TABLE person_tracks (
    id                SERIAL PRIMARY KEY,
    global_person_id  INTEGER NOT NULL,     -- cross-camera Re-ID identity
    local_track_id    INTEGER,              -- ByteTrack ID within this video
    video_source      VARCHAR(255) NOT NULL,
    frame_number      INTEGER NOT NULL,
    bbox_coordinates  JSON NOT NULL,        -- [x1, y1, x2, y2]
    confidence        FLOAT NOT NULL,
    detected_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    -- v2 track-level metadata (NULL for legacy rows)
    track_start_frame INTEGER,             -- first frame of local track
    track_end_frame   INTEGER,             -- last frame of local track
    quality_score     FLOAT,              -- track gallery quality 0–1
    match_score       FLOAT               -- combined Re-ID score at assignment
);
```

### `person_identities`

```sql
CREATE TABLE person_identities (
    global_person_id  INTEGER PRIMARY KEY,
    embedding         JSON NOT NULL,        -- gallery: [[576 floats], ...]
    last_updated      TIMESTAMP
);
```

---

## 🔧 Key Tuning Parameters

All parameters can be set per-job via the Streamlit sidebar or the FastAPI `/upload-videos/` endpoint.

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | 0.70 | Combined score cutoff for merging a track into an existing global ID. |
| `frame_skip` | 1 | Process every Nth frame. Higher = faster but less smooth tracking. |
| `gallery_size` | 8 | Max crops per track gallery. More = more robust Re-ID, more RAM. |
| `min_crop_quality` | 0.40 | Minimum YOLO confidence to admit a crop into the gallery. |
| `weight_appearance` | 0.70 | Weight for gallery cosine similarity in combined score. |
| `weight_temporal` | 0.20 | Weight for temporal compatibility score. |
| `weight_transition` | 0.10 | Weight for camera topology transition prior. |
| `camera_topology` | None | JSON string mapping cameras to expected travel-time ranges. |
| `yolo_model` | `yolov8n.pt` | Replace with `yolov8s.pt` / `yolov8m.pt` for better accuracy. |
| `conf=0.35` | 0.35 | YOLO per-detection confidence threshold (inside `tracker.py`). |

### Camera Topology Example

```json
{
  "cam1.mp4": {"cam2.mp4": [2, 10], "cam3.mp4": [8, 20]},
  "cam2.mp4": {"cam1.mp4": [2, 10]}
}
```
Values are expected travel times **in seconds**. Matches outside this window are penalised.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload-videos/` | Upload videos + start processing job |
| `GET` | `/status/{job_id}` | Poll job progress |
| `GET` | `/results/` | List all output videos |
| `GET` | `/video/{job_id}/{file}` | Stream an annotated output video |
| `GET` | `/tracks/` | Fetch latest DB records |
| `GET` | `/stats/` | Aggregate statistics |
| `GET` | `/diagnostics/` | Per-video track-level diagnostics *(v2 new)* |
| `DELETE` | `/tracks/` | Clear all records and reset Re-ID cache |

Interactive docs: http://localhost:8000/docs

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `psycopg2.OperationalError` | Check PostgreSQL is running and `.env` credentials are correct. |
| `CUDA out of memory` | Reduce video resolution or set `device="cpu"` in `.env` / `TrackingConfig`. |
| All detections get unique IDs | Lower `similarity_threshold` (try 0.55). |
| Same ID assigned to different people | Raise `similarity_threshold` (try 0.85). |
| Short tracks never match across cameras | Lower `min_gallery_for_match` in `TrackingConfig` (default 2). |
| `Cannot open video` error | Ensure video codec is H.264; re-encode with HandBrake if needed. |
| Streamlit can't connect to backend | Make sure uvicorn is running on port 8000. |
| Gallery always empty | Lower `min_crop_quality` or `gallery_sample_every` in `TrackingConfig`. |

---

## 🔌 Swapping the Re-ID Model

The `ReIDFeatureExtractor` is fully pluggable. Example:

```python
from tracker import ReIDFeatureExtractor, MultiVideoTrackingEngine, TrackingConfig

class MyStrongerExtractor(ReIDFeatureExtractor):
    EMB_DIM = 512

    def __init__(self, device):
        self.device = device
        self.model = load_my_model(...)   # OSNet, ResNet-IBN, CLIP, etc.

    def _forward_batch(self, chw_arrays):
        # return numpy [N, 512] array
        ...

engine = MultiVideoTrackingEngine(
    config=TrackingConfig(...),
    extractor=MyStrongerExtractor(device="cuda"),
)
```

No changes to `main.py`, `app.py`, or `database.py` required.

---

## 📚 Learning Resources

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Re-ID Survey](https://arxiv.org/abs/1610.02984)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [Person Re-ID Awesome List](https://github.com/bismex/Awesome-person-re-identification)
