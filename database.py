"""
database.py - SQLAlchemy Database Models & Connection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

What this file does:
    1. Reads the DATABASE_URL from .env
    2. Creates a SQLAlchemy engine (connection pool) to PostgreSQL
    3. Defines the `person_tracks` table as a Python ORM class
    4. Provides helper functions used by FastAPI (create_tables, get_db)

CHANGED (v2 — track-level Re-ID upgrade):
    • PersonTrack gains 4 new nullable columns:
        - track_start_frame  : first frame of the local ByteTrack track
        - track_end_frame    : last frame of the local ByteTrack track
        - quality_score      : 0–1 track quality (gallery size, crop quality, confidence)
        - match_score        : combined appearance+temporal+transition score used to
                               assign the global_person_id
    • create_tables() runs ALTER TABLE IF NOT EXISTS for each new column so
      existing deployments are upgraded safely without data loss.
"""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────
# DATABASE CONNECTION SETUP
# ─────────────────────────────────────────────────────────────────

# Read connection string; fall back to constructing it from individual env variables or a sane default
DATABASE_URL: str = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASSWORD", "admin")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "mcmot_db")

    # URL-encode password in case it contains special characters like '@'
    from urllib.parse import quote_plus
    encoded_pass = quote_plus(db_pass)
    DATABASE_URL = f"postgresql://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"


# `create_engine` sets up the connection pool.
# echo=False suppresses SQL query logging (set True to see raw SQL for debugging)
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# `SessionLocal` is a factory: calling it creates a new database session.
# autocommit=False means you must explicitly call session.commit()
# autoflush=False prevents automatic flushing before every query
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# `Base` is the parent class for all ORM models
Base = declarative_base()


# ─────────────────────────────────────────────────────────────────
# ORM MODEL — person_tracks TABLE
# ─────────────────────────────────────────────────────────────────

class PersonTrack(Base):
    """
    Represents one detection event: a single person detected in a
    single frame of a single video.

    Table: person_tracks
    ┌──────────────────────┬──────────────┬──────────────────────────────────────────────────────┐
    │ Column               │ Type         │ Description                                          │
    ├──────────────────────┼──────────────┼──────────────────────────────────────────────────────┤
    │ id                   │ SERIAL PK    │ Auto-incrementing row identifier                     │
    │ global_person_id     │ INTEGER      │ Cross-video Re-ID — same person = same ID            │
    │ local_track_id       │ INTEGER      │ ByteTrack ID within the video (nullable)             │
    │ video_source         │ VARCHAR      │ Filename of the source video                         │
    │ frame_number         │ INTEGER      │ Frame index within the video (0-based)               │
    │ bbox_coordinates     │ JSON         │ Bounding box as [x1, y1, x2, y2] pixels             │
    │ confidence           │ FLOAT        │ YOLO detection confidence score (0–1)                │
    │ detected_at          │ TIMESTAMP    │ UTC wall-clock time when row was created             │
    │ track_start_frame    │ INTEGER      │ [NEW v2] First frame where this local track appeared │
    │ track_end_frame      │ INTEGER      │ [NEW v2] Last frame where this local track appeared  │
    │ quality_score        │ FLOAT        │ [NEW v2] Track quality score 0–1                     │
    │ match_score          │ FLOAT        │ [NEW v2] Re-ID combined score used for assignment    │
    └──────────────────────┴──────────────┴──────────────────────────────────────────────────────┘
    """
    __tablename__ = "person_tracks"

    # Primary key — auto-increments for every new row
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # The GLOBAL person identity (assigned by Re-ID engine in tracker.py)
    # Two rows with the same global_person_id represent the SAME person,
    # potentially across different cameras.
    global_person_id = Column(Integer, index=True, nullable=False)

    # The LOCAL person identity (assigned by ByteTrack within this video stream)
    local_track_id = Column(Integer, nullable=True)

    # Which video file this detection came from (e.g., "camera1.mp4")
    video_source = Column(String(255), nullable=False)

    # Frame number within the video (helps reconstruct timelines)
    frame_number = Column(Integer, nullable=False)

    # Bounding box stored as a JSON array: [x1, y1, x2, y2]
    # x1,y1 = top-left corner; x2,y2 = bottom-right corner (in pixels)
    bbox_coordinates = Column(JSON, nullable=False)

    # YOLOv8 confidence score for this detection (0.0 to 1.0)
    confidence = Column(Float, nullable=False)

    # When this record was written to the DB (UTC)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # ── [NEW v2] Track-level metadata ─────────────────────────────
    # The frame range gives temporal context for the local track,
    # enabling temporal consistency checks in GlobalIdentityManager.

    # First frame index where this local ByteTrack track was seen
    track_start_frame = Column(Integer, nullable=True)

    # Last frame index where this local ByteTrack track was seen
    track_end_frame = Column(Integer, nullable=True)

    # Composite quality score for this track's gallery:
    # Higher = more reliable Re-ID (bigger gallery, higher confidence, sharper crops)
    quality_score = Column(Float, nullable=True)

    # Combined Re-ID score (appearance + temporal + transition weights) used
    # when this track was matched to (or created as) a global person ID.
    # NULL if this was the first observation of this global ID (no match needed).
    match_score = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<PersonTrack id={self.id} "
            f"global_id={self.global_person_id} "
            f"local_id={self.local_track_id} "
            f"source={self.video_source!r} "
            f"frame={self.frame_number}>"
        )


class PersonIdentity(Base):
    """
    Stores the persistent appearance embedding gallery for each global person ID.
    This enables Re-ID matching to persist across server runs/restarts.

    CHANGED (v2): The embedding field now stores a list of embeddings (gallery)
    rather than a single EMA embedding. Each embedding is a 576-dim float list.
    Backward-compatible: load_cache_from_db() handles both formats.
    """
    __tablename__ = "person_identities"

    global_person_id = Column(Integer, primary_key=True, index=True)
    # The name of the taught person. Null if unsupervised.
    name = Column(String(100), nullable=True)
    # v2: stores List[List[float]] — a gallery of up to gallery_size embeddings
    embedding = Column(JSON, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PersonIdentity global_id={self.global_person_id} name={self.name}>"


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

# Columns added in v2 — each is added via ALTER TABLE IF NOT EXISTS for safety
_V2_MIGRATIONS = [
    "ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS local_track_id INTEGER;",
    "ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS track_start_frame INTEGER;",
    "ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS track_end_frame INTEGER;",
    "ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS quality_score FLOAT;",
    "ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS match_score FLOAT;",
    "ALTER TABLE person_identities ADD COLUMN IF NOT EXISTS name VARCHAR(100);",
]


def create_tables() -> None:
    """
    Create all tables defined via ORM models in PostgreSQL.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS
    semantics so existing tables are never overwritten.

    Also runs v2 schema migrations to add new nullable columns to
    person_tracks without touching existing data.
    """
    Base.metadata.create_all(bind=engine)

    # Run v2 migrations — ADD COLUMN IF NOT EXISTS is idempotent
    with engine.begin() as conn:
        for stmt in _V2_MIGRATIONS:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                # Log but don't crash — column may already exist in older PG versions
                # that don't support IF NOT EXISTS
                print(f"[DB] Migration note ({stmt[:50]}...): {exc}")

    print("[DB] Tables created/verified successfully (v2 schema).")
    print(f"[DB] Connected to: {DATABASE_URL.split('@')[-1]}")  # Hide credentials


def get_db():
    """
    FastAPI dependency injector that yields a database session.

    Usage in a FastAPI endpoint:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    The `finally` block guarantees the session is always closed,
    even if an exception is raised inside the endpoint.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
