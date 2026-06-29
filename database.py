"""
database.py - SQLAlchemy Database Models & Connection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

What this file does:
    1. Reads the DATABASE_URL from .env
    2. Creates a SQLAlchemy engine (connection pool) to PostgreSQL
    3. Defines the `person_tracks` table as a Python ORM class
    4. Provides helper functions used by FastAPI (create_tables, get_db)

Learning note:
    SQLAlchemy ORM lets you work with database rows as Python objects
    instead of writing raw SQL. The engine handles the connection pool
    so multiple concurrent requests share connections efficiently.
"""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────
# DATABASE CONNECTION SETUP
# ─────────────────────────────────────────────────────────────────

# Read connection string; fall back to a sane default for local dev
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:admin@localhost:5432/mcmot_db"
)

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
    ┌─────────────────────┬──────────────┬────────────────────────────────────────────┐
    │ Column              │ Type         │ Description                                │
    ├─────────────────────┼──────────────┼────────────────────────────────────────────┤
    │ id                  │ SERIAL PK    │ Auto-incrementing row identifier           │
    │ global_person_id    │ INTEGER      │ Cross-video Re-ID — same person = same ID  │
    │ video_source        │ VARCHAR      │ Filename of the source video               │
    │ frame_number        │ INTEGER      │ Frame index within the video (0-based)     │
    │ bbox_coordinates    │ JSON         │ Bounding box as [x1, y1, x2, y2] pixels   │
    │ confidence          │ FLOAT        │ YOLO detection confidence score (0–1)      │
    │ detected_at         │ TIMESTAMP    │ UTC wall-clock time when row was created   │
    └─────────────────────┴──────────────┴────────────────────────────────────────────┘
    """
    __tablename__ = "person_tracks"

    # Primary key — auto-increments for every new row
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # The GLOBAL person identity (assigned by Re-ID engine in tracker.py)
    # Two rows with the same global_person_id represent the SAME person,
    # potentially across different cameras.
    global_person_id = Column(Integer, index=True, nullable=False)

    # The LOCAL person identity (assigned by tracker in tracker.py) within this stream
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

    def __repr__(self) -> str:
        return (
            f"<PersonTrack id={self.id} "
            f"global_id={self.global_person_id} "
            f"source={self.video_source!r} "
            f"frame={self.frame_number}>"
        )


class PersonIdentity(Base):
    """
    Stores the persistent appearance embedding for each global person ID.
    This enables Re-ID matching to persist across server runs/restarts.
    """
    __tablename__ = "person_identities"

    global_person_id = Column(Integer, primary_key=True, index=True)
    embedding = Column(JSON, nullable=False)  # 576-dim float list
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PersonIdentity global_id={self.global_person_id}>"


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def create_tables() -> None:
    """
    Create all tables defined via ORM models in PostgreSQL.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS
    semantics so existing tables are never overwritten.
    """
    Base.metadata.create_all(bind=engine)
    
    # Run manual migration to add local_track_id column if it doesn't exist
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE person_tracks ADD COLUMN IF NOT EXISTS local_track_id INTEGER;"))
        print("[DB] Schema updated successfully (local_track_id verified).")
    except Exception as exc:
        print(f"[DB] Migration warning: {exc}")

    print("[DB] Tables created/verified successfully.")
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
