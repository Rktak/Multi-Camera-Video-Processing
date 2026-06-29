# 🎯 Multi-Camera Multi-Object Tracking (MCMOT) System

A complete end-to-end computer vision pipeline for tracking people across multiple video streams and assigning globally consistent identities using Re-Identification (Re-ID).

---

## 📁 Project Structure

```
VideoProcessing/
├── database.py       # SQLAlchemy ORM model + PostgreSQL connection
├── tracker.py        # Core tracking engine (YOLO + ByteTrack + Re-ID)
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
cd "d:\Python Project\VideoProcessing"
pip install -r requirements.txt
```

> **GPU Note:** If you have CUDA, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run pip install for the rest.

### 3. Start the FastAPI Backend

Open **Terminal 1**:
```bash
cd "d:\Python Project\VideoProcessing"
uvicorn main:app --reload --port 8000
```

The server will:
- Create the `person_tracks` table in PostgreSQL automatically
- Download `yolov8n.pt` weights (~6MB) on first run
- Download MobileNetV3-Small weights (~10MB) on first run

API Explorer (Swagger UI): http://localhost:8000/docs

### 4. Start the Streamlit Frontend

Open **Terminal 2**:
```bash
cd "d:\Python Project\VideoProcessing"
streamlit run app.py
```

Browser will open automatically at: http://localhost:8501

---

## 🚀 Usage

1. Upload 2 or more `.mp4` / `.avi` video files in the **Upload & Process** tab
2. Adjust the **Re-ID Threshold** slider in the sidebar if needed (default: 0.70)
3. Click **🚀 Process Videos** and wait for completion
4. Switch to the **🎬 Results** tab to see annotated videos side-by-side
5. Check the **📊 Tracking Dashboard** tab to browse the PostgreSQL tracking log

---

## 🧠 Architecture

### Detection: YOLOv8n
- Ultralytics YOLOv8 Nano — fastest model, ~6MB weights
- Detects only `person` class (COCO class 0)
- Min confidence: 0.35, NMS IoU: 0.45

### Tracking: ByteTrack (per-video local IDs)
- Built into Ultralytics — no separate install
- Maintains tracking state across frames (`persist=True`)
- Assigns **local** track IDs within one video

### Re-Identification: MobileNetV3-Small + Cosine Similarity
- Extracts 576-dim appearance embedding from each person crop
- L2-normalized embeddings stored in a memory cache
- Cosine similarity comparison → assigns **global** person IDs across all cameras
- Cache uses EMA (α=0.9) to adapt to slow appearance changes

### Database: PostgreSQL via SQLAlchemy
- Table: `person_tracks`
- One row per detected person per frame
- Batch inserts every 50 records for performance

### Concurrency: asyncio + ThreadPoolExecutor
- Each video processed in a separate thread
- `asyncio.gather()` waits for all threads concurrently
- FastAPI event loop stays unblocked

---

## 📊 Database Schema

```sql
CREATE TABLE person_tracks (
    id               SERIAL PRIMARY KEY,
    global_person_id INTEGER NOT NULL,
    video_source     VARCHAR(255) NOT NULL,
    frame_number     INTEGER NOT NULL,
    bbox_coordinates JSON NOT NULL,      -- [x1, y1, x2, y2]
    confidence       FLOAT NOT NULL,
    detected_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
```

---

## 🔧 Key Tuning Parameters

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | 0.70 | Re-ID cosine similarity cutoff. Lower = more merges. |
| `frame_skip` | 1 | Process every Nth frame. Higher = faster but less smooth. |
| `yolov8n.pt` | nano | Replace with `yolov8s.pt` for better accuracy. |
| `conf=0.35` | 0.35 | YOLO confidence threshold in `tracker.py`. |
| `db_batch_size` | 50 | DB insert batch size in `track_video()`. |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `psycopg2.OperationalError` | Check PostgreSQL is running and `.env` credentials are correct. |
| `CUDA out of memory` | Reduce video resolution or set `device="cpu"` in `tracker.py`. |
| All detections get unique IDs | Lower `similarity_threshold` (try 0.55). |
| Same ID assigned to different people | Raise `similarity_threshold` (try 0.85). |
| `Cannot open video` error | Ensure video codec is H.264; re-encode with HandBrake if needed. |
| Streamlit can't connect to backend | Make sure uvicorn is running on port 8000. |

---

## 📚 Learning Resources

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Re-ID Survey](https://arxiv.org/abs/1610.02984)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
