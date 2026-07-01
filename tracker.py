"""
tracker.py - Multi-Video Tracking & Re-Identification Engine (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

Architecture Overview (v2 — track-level Re-ID):
    ┌──────────────────────────────────────────────────────────┐
    │                    Video Frame Loop                      │
    └───────────────────────┬──────────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  PersonDetectorTracker                                   │
    │  • YOLOv8n Detection (person class, conf≥0.35)          │
    │  • ByteTrack (persist=True) → local_track_id            │
    └───────────────────────┬──────────────────────────────────┘
                            │ local_track_id + bbox + conf + crop
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  LocalTrackProfile (one per ByteTrack ID)               │
    │  • Accumulates high-quality crops into a gallery        │
    │  • Crop quality filter: min size, min conf, blur check  │
    │  • Near-duplicate guard: skip if too similar to last    │
    │  • ReIDFeatureExtractor → L2-normalized embeddings      │
    │  • Computes representative embedding (gallery mean)     │
    │  • Computes quality_score from gallery richness         │
    └───────────────────────┬──────────────────────────────────┘
                            │  (deferred — after video ends)
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  GlobalIdentityManager                                   │
    │  • Matches LocalTrackProfile → global_person_id         │
    │  • Score = w_app*appearance + w_temp*temporal +         │
    │            w_trans*transition                           │
    │  • Temporal filter: reject same-time same-camera clash  │
    │  • Camera topology prior (optional)                     │
    │  • Thread-safe; persists gallery to PersonIdentity DB   │
    └───────────────────────┬──────────────────────────────────┘
                            │  global_person_id + match_score
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Annotated Video (L{local} / G{global} overlay)         │
    │  + PostgreSQL DB Log (person_tracks with track metadata) │
    └──────────────────────────────────────────────────────────┘

CHANGED vs v1:
    • Frame-level Re-ID replaced with track-level gallery matching.
    • Each ByteTrack ID now has a LocalTrackProfile with up to
      gallery_size high-quality crops and embeddings.
    • GlobalIdentityManager uses gallery-to-gallery cosine similarity
      combined with temporal + camera-transition scoring.
    • Output video overlay shows both local and global ID: "L3 / G7"
    • TrackingConfig dataclass centralises all tunable parameters.
    • ReIDFeatureExtractor is pluggable — subclass to swap models.
    • MultiVideoTracker kept as a backward-compatible alias for main.py.
"""

import json
import logging
import os
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

from database import PersonTrack, SessionLocal

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("mcmot.tracker")
if not logger.handlers:
    # Console Handler
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)
    
    # File Handler
    try:
        _fh = logging.FileHandler("mcmot_tracker.log", mode="a", encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
        logger.addHandler(_fh)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")
        
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────
# TRACKING CONFIG — All tunable parameters in one dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrackingConfig:
    """
    Central configuration object for the MCMOT pipeline.

    All parameters have sensible defaults and can be overridden
    per-job via the FastAPI upload endpoint.

    Args:
        similarity_threshold:
            Cosine similarity threshold to merge a track into an
            existing global ID. Range 0.0–1.0.
        gallery_size:
            Maximum number of embedding crops stored per local track.
            More crops → more robust representative embedding, but
            uses more RAM. Recommended: 6–12.
        gallery_sample_every:
            Minimum frame gap between successive gallery crops.
            Prevents near-duplicate frames from filling the gallery.
        min_crop_confidence:
            YOLO detection confidence required for a crop to be added
            to the gallery. Lower-confidence detections are still
            tracked and displayed, but not used for Re-ID.
        min_crop_width:
            Minimum pixel width for a gallery crop.
        min_crop_height:
            Minimum pixel height for a gallery crop.
        near_duplicate_threshold:
            Cosine similarity threshold above which a new crop is
            considered a near-duplicate of the last gallery embedding
            and is discarded. 0.97 = very similar (fine-tune if needed).
        weight_appearance:
            Weight for appearance (gallery cosine similarity) in the
            combined matching score.
        weight_temporal:
            Weight for temporal consistency score.
        weight_transition:
            Weight for camera-transition prior score.
        camera_topology:
            Optional dict mapping camera names to expected travel-time
            ranges in seconds: {"cam1": {"cam2": [2, 10]}, ...}
            If None, camera transition score defaults to 1.0 (neutral).
        frame_skip:
            Process every Nth frame (1 = all frames).
        device:
            "auto" | "cuda" | "cpu"
        yolo_model:
            Path or name of YOLO weights file.
        yolo_pool_size:
            Number of pre-loaded YOLO instances (should match
            ThreadPoolExecutor max_workers in main.py).
        db_batch_size:
            Flush DB records every N detections.
        min_gallery_for_match:
            Minimum number of gallery embeddings a track must have
            before it is eligible for global ID matching (vs always
            creating a new ID). Prevents single-frame ghost tracks
            from fragmenting global IDs.
    """
    similarity_threshold: float = 0.70
    gallery_size: int = 8
    gallery_sample_every: int = 5
    min_crop_confidence: float = 0.40
    min_crop_width: int = 32
    min_crop_height: int = 64
    near_duplicate_threshold: float = 0.97
    weight_appearance: float = 0.70
    weight_temporal: float = 0.20
    weight_transition: float = 0.10
    camera_topology: Optional[Dict] = None
    frame_skip: int = 1
    device: str = "auto"
    yolo_model: str = "yolov8n.pt"
    yolo_pool_size: int = 2
    db_batch_size: int = 50
    min_gallery_for_match: int = 2
    reid_model: str = "osnet_ain"          # "osnet_ain" | "osnet" | "mobilenetv3"
    reid_weights_path: Optional[str] = None  # Custom .pth path (auto-download if None)
    supervised_mode: bool = False            # If True, only match against taught named identities

    def __post_init__(self):
        # Normalise weights so they always sum to 1.0
        total = self.weight_appearance + self.weight_temporal + self.weight_transition
        if total > 0:
            self.weight_appearance /= total
            self.weight_temporal   /= total
            self.weight_transition /= total


# ─────────────────────────────────────────────────────────────────
# RE-ID FEATURE EXTRACTOR  (pluggable)
# ─────────────────────────────────────────────────────────────────

class ReIDFeatureExtractor:
    """
    Legacy appearance feature extractor for Re-ID (MobileNetV3-Small).

    Produces a 576-dimensional L2-normalised embedding per crop using
    MobileNetV3-Small with the classifier head removed.

    For better Re-ID accuracy, use OSNetReIDExtractor instead (the new
    default). This class is kept for backward compatibility and as a
    lightweight fallback when torchreid is not installed.

    The rest of the pipeline (LocalTrackProfile, GlobalIdentityManager)
    is model-agnostic — they just work with L2-normalised numpy arrays.
    """

    EMB_DIM = 576  # MobileNetV3-Small avgpool output dimension

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

        # Load MobileNetV3-Small (backbone only — no classifier)
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.model = torch.nn.Sequential(
            mobilenet.features,
            mobilenet.avgpool,
            torch.nn.Flatten(),  # → [batch, 576]
        ).to(self.device)
        self.model.eval()
        logger.info("MobileNetV3-Small Re-ID extractor loaded (%d-dim) on %s.",
                    self.EMB_DIM, self.device.upper())

        # ImageNet normalisation constants
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert a BGR crop → normalised CHW float32 numpy array.
        Returns None if the crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        resized = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return rgb.transpose(2, 0, 1)  # HWC → CHW

    def _forward_batch(self, chw_arrays: List[np.ndarray]) -> np.ndarray:
        """Run a list of CHW float32 arrays through the model. Returns [N, EMB_DIM]."""
        batch = torch.from_numpy(np.stack(chw_arrays)).to(self.device)
        with torch.no_grad():
            return self.model(batch).cpu().numpy()

    def extract_batch(
        self, crops: List[Optional[np.ndarray]]
    ) -> List[Optional[np.ndarray]]:
        """
        Batch feature extraction: one GPU forward pass for all crops.

        Args:
            crops: List of BGR numpy arrays (or None for invalid slots).

        Returns:
            List of L2-normalised float32 embeddings (same length as crops).
            Entries are None for invalid/empty crops.
        """
        results: List[Optional[np.ndarray]] = [None] * len(crops)
        if not crops:
            return results

        tensors: List[np.ndarray] = []
        valid_idx: List[int] = []

        for i, crop in enumerate(crops):
            chw = self._preprocess(crop)
            if chw is not None:
                tensors.append(chw)
                valid_idx.append(i)

        if not tensors:
            return results

        embeddings = self._forward_batch(tensors)  # [N, EMB_DIM]
        for out_i, orig_i in enumerate(valid_idx):
            emb = embeddings[out_i].astype(np.float32)
            norm = np.linalg.norm(emb)
            results[orig_i] = emb / norm if norm > 0 else None

        return results

    def extract_single(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Convenience wrapper for a single crop (uses batch internally)."""
        result = self.extract_batch([crop_bgr])
        return result[0]


class OSNetReIDExtractor:
    """
    Person Re-ID feature extractor using OSNet / OSNet-AIN from torchreid.

    OSNet (Omni-Scale Network) is a lightweight CNN specifically designed
    for person re-identification. OSNet-AIN adds Instance Normalization
    for robustness to cross-camera style/lighting variations.

    Produces a 512-dimensional L2-normalised embedding per crop.
    Pretrained on Market1501 for person Re-ID (not ImageNet).

    Input crop size: 256×128 (standard ReID resolution).

    Supported model_name values:
        - 'osnet_ain_x1_0'  (recommended — best cross-camera accuracy)
        - 'osnet_x1_0'      (slightly faster, no instance norm)
        - 'osnet_x0_75'     (smaller variant)

    The public API (extract_batch, extract_single) is identical to
    ReIDFeatureExtractor so the rest of the pipeline works unchanged.
    """

    EMB_DIM = 512  # OSNet output dimension

    # Model name mapping for user-friendly config values
    _MODEL_MAP = {
        "osnet_ain": "osnet_ain_x1_0",
        "osnet":     "osnet_x1_0",
    }

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "osnet_ain",
        weights_path: Optional[str] = None,
    ) -> None:
        self.device = device
        self._torchreid_name = self._MODEL_MAP.get(model_name, model_name)

        try:
            from torchreid.reid.utils import FeatureExtractor as TorchreidFE
        except ImportError:
            try:
                from torchreid.utils import FeatureExtractor as TorchreidFE
            except ImportError:
                raise ImportError(
                    "torchreid is required for OSNet Re-ID. "
                    "Install with: pip install torchreid"
                )

        # torchreid.FeatureExtractor handles weight download, model build,
        # and eval mode internally. It accepts image paths OR PIL Images
        # OR numpy arrays (HWC uint8 BGR/RGB).
        fe_kwargs = {
            "model_name": self._torchreid_name,
            "device": device,
        }
        if weights_path:
            fe_kwargs["model_path"] = weights_path

        self._fe = TorchreidFE(**fe_kwargs)
        logger.info(
            "OSNet Re-ID extractor loaded: %s (%d-dim) on %s.",
            self._torchreid_name, self.EMB_DIM, device.upper(),
        )

        # ImageNet normalisation constants (same as torchreid uses internally)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert a BGR crop → normalised CHW float32 numpy array.
        Uses 256×128 input resolution (standard for person ReID).
        Returns None if the crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        resized = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return rgb.transpose(2, 0, 1)  # HWC → CHW

    def _forward_batch(self, chw_arrays: List[np.ndarray]) -> np.ndarray:
        """Run a list of CHW float32 arrays through OSNet. Returns [N, 512]."""
        batch = torch.from_numpy(np.stack(chw_arrays)).to(self.device)
        with torch.no_grad():
            features = self._fe.model(batch)
        return features.cpu().numpy()

    def extract_batch(
        self, crops: List[Optional[np.ndarray]]
    ) -> List[Optional[np.ndarray]]:
        """
        Batch feature extraction: one GPU forward pass for all crops.

        Args:
            crops: List of BGR numpy arrays (or None for invalid slots).

        Returns:
            List of L2-normalised float32 embeddings (same length as crops).
            Entries are None for invalid/empty crops.
        """
        results: List[Optional[np.ndarray]] = [None] * len(crops)
        if not crops:
            return results

        tensors: List[np.ndarray] = []
        valid_idx: List[int] = []

        for i, crop in enumerate(crops):
            chw = self._preprocess(crop)
            if chw is not None:
                tensors.append(chw)
                valid_idx.append(i)

        if not tensors:
            return results

        embeddings = self._forward_batch(tensors)  # [N, 512]
        for out_i, orig_i in enumerate(valid_idx):
            emb = embeddings[out_i].astype(np.float32)
            norm = np.linalg.norm(emb)
            results[orig_i] = emb / norm if norm > 0 else None

        return results

    def extract_single(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Convenience wrapper for a single crop (uses batch internally)."""
        result = self.extract_batch([crop_bgr])
        return result[0]


def _build_reid_extractor(
    config: TrackingConfig,
    device: str,
) -> "ReIDFeatureExtractor | OSNetReIDExtractor":
    """
    Factory function: build the appropriate Re-ID feature extractor
    based on config.reid_model.

    Falls back to MobileNetV3 if torchreid is not installed.
    """
    model_key = config.reid_model.lower().strip()

    if model_key in ("osnet_ain", "osnet", "osnet_ain_x1_0", "osnet_x1_0"):
        try:
            return OSNetReIDExtractor(
                device=device,
                model_name=model_key,
                weights_path=config.reid_weights_path,
            )
        except ImportError:
            logger.warning(
                "torchreid not installed — falling back to MobileNetV3. "
                "Install with: pip install torchreid"
            )
            return ReIDFeatureExtractor(device=device)
    elif model_key == "mobilenetv3":
        return ReIDFeatureExtractor(device=device)
    else:
        logger.warning(
            "Unknown reid_model '%s' — falling back to OSNet-AIN.", model_key
        )
        try:
            return OSNetReIDExtractor(device=device, model_name="osnet_ain")
        except ImportError:
            return ReIDFeatureExtractor(device=device)


# ─────────────────────────────────────────────────────────────────
# LOCAL TRACK PROFILE — per-ByteTrack-ID state
# ─────────────────────────────────────────────────────────────────

class LocalTrackProfile:
    """
    Accumulates detection evidence for one ByteTrack local track ID
    across all frames of a single video.

    Key responsibilities:
        • Collect high-quality crops for the embedding gallery
        • Maintain frame range (start/end) for temporal reasoning
        • Compute a representative embedding (gallery mean)
        • Compute a quality score based on gallery richness
        • Expose these to GlobalIdentityManager for matching

    Near-duplicate guard:
        If a new embedding is cosine-similar (>= near_duplicate_threshold)
        to the most recently accepted gallery embedding, it is skipped.
        This prevents a static/barely-moving person from having a gallery
        of 8 nearly identical embeddings.
    """

    def __init__(
        self,
        local_track_id: int,
        video_source: str,
        config: TrackingConfig,
    ) -> None:
        self.local_track_id = local_track_id
        self.video_source = video_source
        self.config = config

        # Temporal range
        self.start_frame: int = -1
        self.end_frame: int = -1

        # Raw detection history (for DB logging)
        self.frame_records: List[dict] = []  # one dict per frame detection

        # Gallery: list of L2-normalised embeddings (up to gallery_size)
        self.gallery: List[np.ndarray] = []

        # Quality metadata
        self.confidences: List[float] = []
        self.crop_sizes: List[Tuple[int, int]] = []  # (w, h) per accepted gallery crop

        # Internal: frame counter since last gallery sample
        self._frames_since_sample: int = 0

        # Assigned global ID (set by GlobalIdentityManager)
        self.global_person_id: Optional[int] = None
        self.match_score: Optional[float] = None

        # Cache: invalidated when gallery changes
        self._rep_emb: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        crop_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray],
    ) -> None:
        """
        Record a new frame detection for this track.

        Args:
            frame_number: 0-based frame index.
            bbox:         (x1, y1, x2, y2) in pixels.
            confidence:   YOLO detection confidence.
            crop_bgr:     Person crop (may be None if out of bounds).
            embedding:    L2-normalised embedding (may be None).
        """
        if self.start_frame == -1:
            self.start_frame = frame_number
        self.end_frame = frame_number

        # Always record for DB logging
        self.frame_records.append({
            "frame_number": frame_number,
            "bbox_coordinates": list(bbox),
            "confidence": round(confidence, 4),
        })
        self.confidences.append(confidence)

        # Gallery gating — only add if the crop passes quality filters
        self._frames_since_sample += 1
        if self._frames_since_sample < self.config.gallery_sample_every:
            return  # Too soon since last sample

        if embedding is None or crop_bgr is None:
            return

        # Minimum crop size
        x1, y1, x2, y2 = bbox
        cw, ch = x2 - x1, y2 - y1
        if cw < self.config.min_crop_width or ch < self.config.min_crop_height:
            return

        # Minimum confidence
        if confidence < self.config.min_crop_confidence:
            return

        # Near-duplicate guard: skip if very similar to last gallery entry
        if self.gallery:
            sim = float(np.dot(embedding, self.gallery[-1]))
            if sim >= self.config.near_duplicate_threshold:
                return

        # Accept this crop into the gallery
        if len(self.gallery) < self.config.gallery_size:
            self.gallery.append(embedding.astype(np.float32))
            self.crop_sizes.append((cw, ch))
            self._rep_emb = None  # Invalidate cache
            self._frames_since_sample = 0

    @property
    def representative_embedding(self) -> Optional[np.ndarray]:
        """
        L2-normalised mean of all gallery embeddings.
        Cached — recomputed only when gallery changes.
        Returns None if gallery is empty.
        """
        if not self.gallery:
            return None
        if self._rep_emb is None:
            mean_emb = np.mean(self.gallery, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            self._rep_emb = mean_emb / norm if norm > 0 else mean_emb
        return self._rep_emb

    @property
    def quality_score(self) -> float:
        """
        Composite track quality score in [0, 1].

        Combines:
            • Gallery fill ratio (gallery_size / max_gallery_size)
            • Mean detection confidence
            • Mean crop size relative to a reference 64×128 crop

        Higher score → more reliable Re-ID embedding.
        """
        if not self.gallery:
            return 0.0

        gallery_ratio = len(self.gallery) / max(self.config.gallery_size, 1)
        mean_conf = float(np.mean(self.confidences)) if self.confidences else 0.0

        # Crop area ratio: compare mean crop area to reference 64×128=8192 px
        if self.crop_sizes:
            mean_area = np.mean([w * h for w, h in self.crop_sizes])
            area_ratio = min(float(mean_area) / 8192.0, 1.0)
        else:
            area_ratio = 0.0

        # Weighted combination
        score = 0.5 * gallery_ratio + 0.3 * mean_conf + 0.2 * area_ratio
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    @property
    def frame_count(self) -> int:
        """Total number of frames this track was detected in."""
        return len(self.frame_records)

    def gallery_to_gallery_similarity(self, other: "LocalTrackProfile") -> float:
        """
        Compute the maximum cosine similarity between the galleries
        of this track and another track.

        Strategy: compute all pairwise dot products (L2-normalised →
        dot == cosine sim) and return the maximum.
        This is more robust than rep-to-rep for short or varied galleries.

        Falls back to representative-to-representative if either gallery
        is small or empty.
        """
        emb_a = self.representative_embedding
        emb_b = other.representative_embedding
        if emb_a is None or emb_b is None:
            return 0.0

        # Use representative embeddings for speed when galleries are large
        if len(self.gallery) <= 2 or len(other.gallery) <= 2:
            return float(np.dot(emb_a, emb_b))

        # Gallery-to-gallery: [|A|, dim] @ [dim, |B|] → [|A|, |B|]
        A = np.array(self.gallery, dtype=np.float32)  # [n_a, dim]
        B = np.array(other.gallery, dtype=np.float32)  # [n_b, dim]
        sim_matrix = A @ B.T  # [n_a, n_b]
        return float(np.max(sim_matrix))

    def __repr__(self) -> str:
        return (
            f"<LocalTrackProfile lid={self.local_track_id} "
            f"src={self.video_source!r} "
            f"frames={self.start_frame}-{self.end_frame} "
            f"gallery={len(self.gallery)} "
            f"gid={self.global_person_id}>"
        )


# ─────────────────────────────────────────────────────────────────
# GLOBAL IDENTITY MANAGER — cross-video track matching
# ─────────────────────────────────────────────────────────────────

class GlobalIdentityManager:
    """
    Maps LocalTrackProfile objects from one or more videos to a shared
    set of global person IDs using multi-factor scoring.

    Matching score:
        final_score = w_app * appearance
                    + w_temp * temporal
                    + w_trans * transition

    where:
        appearance  = gallery-to-gallery cosine similarity
        temporal    = 1.0 if tracks are temporally compatible, else low score
        transition  = camera-transition prior from topology config

    Thread-safety:
        All public methods acquire self._lock. Video threads can call
        assign_global_id() concurrently; the lock is brief (no heavy
        compute happens inside it after initial setup).

    Persistence:
        The gallery for each global ID is saved to PersonIdentity table.
        On startup, load_from_db() restores the cache.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._lock = threading.Lock()

        # Maps global_person_id → list of L2-normalised embeddings (gallery)
        self._gallery: Dict[int, List[np.ndarray]] = {}

        # Maps global_person_id → temporal range dict:
        # {video_source: (start_frame, end_frame)}
        self._temporal: Dict[int, Dict[str, Tuple[int, int]]] = {}

        # Maps global_person_id → name (for supervised taught identities)
        self._names: Dict[int, Optional[str]] = {}

        # Tracks which global IDs have been modified (need DB flush)
        self._dirty: Dict[int, List[np.ndarray]] = {}

        self._next_id: int = 1

        # Vectorised embedding matrix (rebuilt lazily after mutations)
        self._emb_matrix: Optional[np.ndarray] = None   # [N, dim]
        self._emb_gids: List[int] = []                   # [N] — global ID for each row
        self._matrix_dirty: bool = True

        logger.info("GlobalIdentityManager initialized. Weights: app=%.2f temp=%.2f trans=%.2f",
                    config.weight_appearance, config.weight_temporal, config.weight_transition)

    # ── Core: assign a global ID to a local track ─────────────────

    def assign_global_id(self, profile: LocalTrackProfile) -> Tuple[int, float]:
        """
        Find the best matching global ID for a LocalTrackProfile, or
        create a new global ID if no match is found.

        Returns:
            (global_person_id, match_score)
            match_score is 0.0 for newly created IDs.
        """
        rep = profile.representative_embedding

        # Tracks with insufficient gallery evidence always get a new ID
        # to avoid polluting global IDs with noisy short-track embeddings.
        if rep is None or len(profile.gallery) < self.config.min_gallery_for_match:
            if self.config.supervised_mode:
                return -1, 0.0
            gid = self._create_new_id(profile)
            logger.debug("  Track L%d → NEW global G%d (insufficient gallery: %d/%d)",
                         profile.local_track_id, gid,
                         len(profile.gallery), self.config.min_gallery_for_match)
            return gid, 0.0

        with self._lock:
            if not self._gallery:
                if self.config.supervised_mode:
                    return -1, 0.0
                gid = self._create_new_id_locked(profile)
                return gid, 0.0

            if self._matrix_dirty or (self._emb_matrix is not None and self._emb_matrix.shape[1] != rep.shape[0]):
                self._rebuild_matrix(target_dim=rep.shape[0])

            if self._emb_matrix is None:
                gid = self._create_new_id_locked(profile)
                return gid, 0.0

            # ── Phase 1: fast appearance pre-filter (vectorised) ───
            rep_f32 = rep.astype(np.float32)
            app_scores = self._emb_matrix @ rep_f32   # [N]
            # Deduplicate: per global ID, take max appearance score
            gid_app: Dict[int, float] = {}
            for row_i, gid in enumerate(self._emb_gids):
                s = float(app_scores[row_i])
                if s > gid_app.get(gid, -1.0):
                    gid_app[gid] = s

        # ── Phase 2: full multi-factor scoring (outside lock) ─────
        best_gid:   int   = -1
        best_score: float = -1.0

        for gid, app_sim in gid_app.items():
            # Quick prune: if appearance alone can't reach threshold, skip
            if app_sim * self.config.weight_appearance < (
                self.config.similarity_threshold * self.config.weight_appearance - 0.15
            ):
                continue

            temporal_score   = self._temporal_score(profile, gid)
            transition_score = self._transition_score(profile, gid)

            final_score = (
                self.config.weight_appearance * app_sim
                + self.config.weight_temporal   * temporal_score
                + self.config.weight_transition * transition_score
            )

            if final_score > best_score:
                best_score = final_score
                best_gid   = gid

        with self._lock:
            if best_gid != -1 and best_score >= self.config.similarity_threshold:
                # Merge into existing global ID
                self._update_gallery_locked(best_gid, profile)
                self._update_temporal_locked(best_gid, profile)
                logger.debug(
                    "  Track L%d (%s) → MATCHED G%d "
                    "(score=%.3f app=%.3f)",
                    profile.local_track_id, profile.video_source,
                    best_gid, best_score,
                    gid_app.get(best_gid, 0.0),
                )
                return best_gid, round(best_score, 4)
            else:
                # No match — mint a new global ID
                gid = self._create_new_id_locked(profile)
                logger.debug(
                    "  Track L%d (%s) → NEW G%d "
                    "(best_score=%.3f < threshold=%.2f)",
                    profile.local_track_id, profile.video_source,
                    gid, best_score, self.config.similarity_threshold,
                )
                return gid, 0.0

    # ── Fast inline match (during frame processing) ───────────────

    def quick_match(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Fast single-embedding match against all known global ID galleries.
        Used during frame processing to show a preliminary global ID in the
        output video overlay. This is less reliable than assign_global_id()
        but runs per-frame without waiting for track closure.

        Returns (global_person_id, similarity).
        """
        feat = embedding.astype(np.float32)
        with self._lock:
            if not self._gallery:
                if self.config.supervised_mode:
                    return -1, 0.0
                gid = self._next_id
                self._gallery[gid] = [feat.copy()]
                self._temporal[gid] = {}
                self._dirty[gid] = self._gallery[gid]
                self._next_id += 1
                self._matrix_dirty = True
                return gid, 0.0

            if self._matrix_dirty or (self._emb_matrix is not None and self._emb_matrix.shape[1] != feat.shape[0]):
                self._rebuild_matrix(target_dim=feat.shape[0])

            if self._emb_matrix is None:
                if self.config.supervised_mode:
                    return -1, 0.0
                gid = self._next_id
                self._gallery[gid] = [feat.copy()]
                self._temporal[gid] = {}
                self._dirty[gid] = self._gallery[gid]
                self._next_id += 1
                self._matrix_dirty = True
                return gid, 0.0

            sims = self._emb_matrix @ feat
            best_i   = int(np.argmax(sims))
            best_sim = float(sims[best_i])
            best_gid = self._emb_gids[best_i]

            if best_sim >= self.config.similarity_threshold:
                return best_gid, best_sim
            else:
                if self.config.supervised_mode:
                    return -1, 0.0
                gid = self._next_id
                self._gallery[gid] = [feat.copy()]
                self._temporal[gid] = {}
                self._dirty[gid] = self._gallery[gid]
                self._next_id += 1
                self._matrix_dirty = True
                return gid, 0.0

    # ── Scoring helpers ───────────────────────────────────────────

    def _temporal_score(self, profile: LocalTrackProfile, gid: int) -> float:
        """
        Score temporal compatibility of a track with an existing global ID.

        Rules:
        • If the same video already assigned this gid AND the frame ranges
          overlap significantly, it's likely the same track re-detected —
          score = 0.9 (high, likely continuation).
        • If a DIFFERENT video already assigned this gid AND the frame ranges
          overlap, it means the person was in two places at once → impossible
          → score = 0.0 (hard reject).
        • If no temporal data exists yet → score = 1.0 (neutral/new).
        """
        temporal_map = self._temporal.get(gid, {})
        if not temporal_map:
            return 1.0  # No prior temporal data — neutral

        p_start, p_end = profile.start_frame, profile.end_frame

        for src, (t_start, t_end) in temporal_map.items():
            overlap = min(p_end, t_end) - max(p_start, t_start)
            if overlap > 0:
                if src == profile.video_source:
                    # Same camera, overlapping range → likely continuation
                    return 0.9
                else:
                    # Different camera, overlapping frames → impossible match
                    return 0.0

        return 1.0  # No overlap with any prior track for this gid

    def _transition_score(self, profile: LocalTrackProfile, gid: int) -> float:
        """
        Camera topology prior: how likely is it for a person to move from
        a camera where gid was last seen to this profile's camera?

        If no topology config is provided, returns 1.0 (neutral).
        If topology is provided but no rule matches, returns 0.5 (uncertain).
        If a matching rule exists, checks whether the temporal gap falls in
        the expected travel-time range.
        """
        topology = self.config.camera_topology
        if not topology:
            return 1.0  # No topology — camera-agnostic

        temporal_map = self._temporal.get(gid, {})
        if not temporal_map:
            return 1.0  # No prior observations

        # Find the most recent observation camera and frame
        latest_src, latest_range = max(temporal_map.items(), key=lambda kv: kv[1][1])
        if latest_src == profile.video_source:
            return 1.0  # Same camera — transition not applicable

        # Look up expected travel time from latest_src → profile.video_source
        src_rules = topology.get(latest_src, {})
        rule = src_rules.get(profile.video_source)
        if rule is None:
            return 0.5  # No rule between these cameras

        # rule = [min_seconds, max_seconds] of expected travel time
        try:
            min_s, max_s = float(rule[0]), float(rule[1])
        except (TypeError, IndexError, ValueError):
            return 0.5

        # Convert frame gap to seconds (assume 25 FPS if unknown)
        fps = 25.0
        frame_gap = max(0, profile.start_frame - latest_range[1])
        gap_seconds = frame_gap / fps

        if min_s <= gap_seconds <= max_s:
            return 1.0   # Perfect transition window
        elif gap_seconds < min_s:
            return 0.3   # Too fast — unlikely
        else:
            return 0.6   # Possible but slower than expected

    # ── Gallery management ────────────────────────────────────────

    def _create_new_id(self, profile: LocalTrackProfile) -> int:
        """Create a new global ID without holding the lock."""
        with self._lock:
            return self._create_new_id_locked(profile)

    def _create_new_id_locked(self, profile: LocalTrackProfile) -> int:
        """Create a new global ID. MUST be called while holding self._lock."""
        gid = self._next_id
        gallery_copy = [e.copy() for e in profile.gallery] if profile.gallery else []
        if not gallery_copy and profile.representative_embedding is not None:
            gallery_copy = [profile.representative_embedding.copy()]
        self._gallery[gid] = gallery_copy
        self._temporal[gid] = {
            profile.video_source: (profile.start_frame, profile.end_frame)
        }
        self._dirty[gid] = self._gallery[gid]
        self._next_id += 1
        self._matrix_dirty = True
        return gid

    def _update_gallery_locked(self, gid: int, profile: LocalTrackProfile) -> None:
        """
        Merge profile gallery into the existing global ID gallery.
        Keeps up to gallery_size embeddings, preferring diverse ones.
        """
        existing = self._gallery[gid]
        max_g = self.config.gallery_size

        for emb in profile.gallery:
            if len(existing) >= max_g:
                break
            # Only add if not near-duplicate of any existing embedding
            if existing:
                sims = [float(np.dot(emb, e)) for e in existing]
                if max(sims) >= self.config.near_duplicate_threshold:
                    continue
            existing.append(emb.copy())

        self._dirty[gid] = existing
        self._matrix_dirty = True

    def _update_temporal_locked(self, gid: int, profile: LocalTrackProfile) -> None:
        """Update temporal range for this global ID from the matched profile."""
        src = profile.video_source
        if src in self._temporal[gid]:
            old_start, old_end = self._temporal[gid][src]
            self._temporal[gid][src] = (
                min(old_start, profile.start_frame),
                max(old_end, profile.end_frame),
            )
        else:
            self._temporal[gid][src] = (profile.start_frame, profile.end_frame)

    # ── Matrix helpers ────────────────────────────────────────────

    def _rebuild_matrix(self, target_dim: Optional[int] = None) -> None:
        """
        Flatten _gallery into a [N, dim] matrix and parallel gid list.
        Called lazily when _matrix_dirty is True (inside lock).
        """
        if target_dim is None:
            # Determine the target dimension, prioritizing newly added/dirty embeddings
            for embs in self._dirty.values():
                if embs:
                    target_dim = embs[-1].shape[0]
                    break
                
        if target_dim is None:
            # Fallback to the most frequent dimension
            dim_counts = {}
            for embs in self._gallery.values():
                for emb in embs:
                    dim_counts[emb.shape[0]] = dim_counts.get(emb.shape[0], 0) + 1
            if dim_counts:
                target_dim = max(dim_counts.items(), key=lambda x: x[1])[0]

        rows: List[np.ndarray] = []
        gids: List[int] = []
        
        for gid in list(self._gallery.keys()):
            if self.config.supervised_mode and not self._names.get(gid):
                continue

            embs = self._gallery[gid]
            valid_embs = [e for e in embs if target_dim is None or e.shape[0] == target_dim]
            
            if not valid_embs:
                # Discard global IDs that have no valid embeddings (e.g. from an older model)
                del self._gallery[gid]
                if gid in self._temporal:
                    del self._temporal[gid]
                if gid in self._dirty:
                    del self._dirty[gid]
            else:
                self._gallery[gid] = valid_embs
                for emb in valid_embs:
                    rows.append(emb)
                    gids.append(gid)

        self._emb_gids = gids
        self._emb_matrix = np.array(rows, dtype=np.float32) if rows else None
        self._matrix_dirty = False

    # ── Statistics ────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of current global identity statistics."""
        with self._lock:
            return {
                "total_global_ids": len(self._gallery),
                "next_global_id": self._next_id,
                "gallery_sizes": {gid: len(embs) for gid, embs in self._gallery.items()},
            }

    # ── Persistence ───────────────────────────────────────────────

    def flush_to_db(self, db_session) -> None:
        """
        Write dirty global identity galleries to the PersonIdentity table.
        Safe to call concurrently — snapshots dirty dict under lock.
        """
        if self.config.supervised_mode:
            return

        from database import PersonIdentity

        with self._lock:
            to_flush = self._dirty.copy()
            self._dirty.clear()

        if not to_flush:
            return

        try:
            for gid, embs in to_flush.items():
                emb_list = [e.tolist() for e in embs]
                rec = db_session.query(PersonIdentity).filter(
                    PersonIdentity.global_person_id == gid
                ).first()
                if rec:
                    rec.embedding = emb_list
                    rec.last_updated = datetime.utcnow()
                else:
                    db_session.add(PersonIdentity(
                        global_person_id=gid,
                        embedding=emb_list,
                    ))
            db_session.commit()
            logger.info("Flushed %d global identities to DB.", len(to_flush))
        except Exception as exc:
            db_session.rollback()
            logger.error("Failed to flush identities to DB: %s", exc)

    def load_from_db(self, db_session) -> None:
        """
        Restore global identity galleries from PersonIdentity table.
        Call at startup (or before a new job) to persist identities
        across server restarts.
        """
        from database import PersonIdentity

        with self._lock:
            try:
                records = db_session.query(PersonIdentity).all()
                self._gallery.clear()
                self._temporal.clear()
                self._dirty.clear()
                max_id = 0

                for r in records:
                    gid = r.global_person_id
                    raw = r.embedding
                    # Support both [[float,...]] (multi-view) and [float,...] (v1 compat)
                    if raw and isinstance(raw[0], list):
                        embs = [np.array(e, dtype=np.float32) for e in raw]
                    else:
                        embs = [np.array(raw, dtype=np.float32)]
                    self._gallery[gid] = embs
                    self._temporal[gid] = {}
                    self._names[gid] = r.name
                    if gid > max_id:
                        max_id = gid

                self._next_id = max_id + 1
                self._matrix_dirty = True
                logger.info("Loaded %d global identities from DB. Next GID: %d",
                            len(records), self._next_id)
            except Exception as exc:
                logger.error("Error loading identities from DB: %s", exc)

    def reset(self) -> None:
        """Clear all global state and reset ID counter."""
        with self._lock:
            self._gallery.clear()
            self._temporal.clear()
            self._dirty.clear()
            self._names.clear()
            self._next_id = 1
            self._emb_matrix = None
            self._emb_gids = []
            self._matrix_dirty = True
        logger.info("GlobalIdentityManager reset.")

    def update_threshold(self, new_threshold: float) -> None:
        """Hot-update the similarity threshold."""
        self.config.similarity_threshold = float(new_threshold)
        logger.info("Re-ID threshold updated to %.2f", new_threshold)


# ─────────────────────────────────────────────────────────────────
# PERSON DETECTOR TRACKER — YOLO + ByteTrack wrapper
# ─────────────────────────────────────────────────────────────────

class PersonDetectorTracker:
    """
    Wraps Ultralytics YOLO + ByteTrack for person detection and
    local tracking within a single video stream.

    Manages a pool of YOLO instances so multiple video threads can
    run detection concurrently without reloading weights from disk.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif config.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        logger.info("PersonDetectorTracker: device=%s", self.device.upper())
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
            torch.backends.cudnn.benchmark = True

        # YOLO instance pool
        self._pool: _queue.Queue = _queue.Queue()
        for _ in range(config.yolo_pool_size):
            inst = YOLO(config.yolo_model)
            inst.to(self.device)
            self._pool.put(inst)
        logger.info("YOLO pool: %d × %s loaded.", config.yolo_pool_size, config.yolo_model)

    def borrow_yolo(self) -> YOLO:
        """Borrow a YOLO instance from the pool (blocks until one is free)."""
        return self._pool.get()

    def return_yolo(self, inst: YOLO) -> None:
        """Return a borrowed YOLO instance to the pool."""
        self._pool.put(inst)

    def detect_and_track(
        self,
        frame: np.ndarray,
        yolo_inst: YOLO,
    ) -> Optional[object]:
        """
        Run YOLO + ByteTrack on a single frame.

        Returns the raw ultralytics Results object, or None if no
        detections were found.
        """
        results = yolo_inst.track(
            frame,
            persist=True,   # Carry ByteTrack state across frames
            classes=[0],    # COCO class 0 = "person"
            conf=0.35,
            iou=0.45,
            device=self.device,
            verbose=False,
            half=(self.device == "cuda"),
        )
        if results[0].boxes is None or results[0].boxes.id is None:
            return None
        return results[0]


# ─────────────────────────────────────────────────────────────────
# MULTI-VIDEO TRACKING ENGINE — orchestrator
# ─────────────────────────────────────────────────────────────────

class MultiVideoTrackingEngine:
    """
    Orchestrates the full MCMOT pipeline for one or more video files.

    Per-video processing (track_video):
        1. Frame loop: detect → extract embeddings → update LocalTrackProfiles
        2. Overlay frame with preliminary global IDs (quick_match)
        3. Write annotated video
        4. After the frame loop: finalize all local tracks via
           GlobalIdentityManager.assign_global_id()
        5. Write DB records with track-level metadata

    Thread-safety:
        track_video() is designed to be called from concurrent threads.
        All shared state (GlobalIdentityManager, extractor model) is
        thread-safe or uses Python's GIL appropriately.

    Backward compatibility:
        MultiVideoTracker is defined at the bottom as an alias for this
        class so existing main.py imports continue to work unchanged.
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        extractor: Optional[ReIDFeatureExtractor] = None,
        # Legacy positional params for backward-compat with main.py
        similarity_threshold: float = 0.70,
        device: str = "auto",
        yolo_model: Optional[str] = None,
    ) -> None:
        # Support both new config-object API and legacy keyword-arg API
        if config is None:
            config = TrackingConfig(
                similarity_threshold=similarity_threshold,
                device=device,
                yolo_model=yolo_model or os.getenv("YOLO_MODEL", "yolov8n.pt"),
            )
        self.config = config

        self.detector = PersonDetectorTracker(config)
        self.device = self.detector.device

        if extractor is None:
            extractor = _build_reid_extractor(config, device=self.device)
        self.extractor = extractor

        self.identity_mgr = GlobalIdentityManager(config)

        # Per-video local track profiles — cleared between unrelated jobs
        # Key: video_source → {local_track_id: LocalTrackProfile}
        self._track_profiles: Dict[str, Dict[int, LocalTrackProfile]] = {}
        self._profiles_lock = threading.Lock()

        # Diagnostics counters (populated by track_video)
        self._diagnostics: Dict[str, dict] = {}
        self._diag_lock = threading.Lock()

        # Colour palette for bounding box rendering
        np.random.seed(42)
        self._palette: List[Tuple[int, int, int]] = [
            tuple(int(c) for c in col)
            for col in np.random.randint(80, 230, size=(2000, 3))
        ]

        logger.info("MultiVideoTrackingEngine ready. Threshold=%.2f",
                    config.similarity_threshold)

    # ── Helpers ───────────────────────────────────────────────────

    def _color(self, global_id: int) -> Tuple[int, int, int]:
        return self._palette[global_id % len(self._palette)]

    def _draw_box(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        local_id: int,
        global_id: int,
        confidence: float,
    ) -> None:
        """
        Draw bounding box + dual-label overlay showing both local and global ID.
        Label format: "L{local} / G{global}" or Name if supervised.
        """
        color = self._color(global_id) if global_id != -1 else (0, 0, 255) # Red for unknown
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if global_id == -1:
            label = " Unknown "
        else:
            name = self.identity_mgr._names.get(global_id)
            if name:
                label = f" {name} "
            else:
                label = f" L{local_id} / G{global_id} "

        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly_top = max(y1 - lh - baseline - 6, 0)
        cv2.rectangle(frame, (x1, ly_top), (x1 + lw, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        # Confidence badge at the bottom of the box
        cv2.putText(
            frame, f"{confidence:.0%}",
            (x1 + 4, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

    # ── Frame processing ──────────────────────────────────────────

    def _process_frame(
        self,
        frame: np.ndarray,
        video_source: str,
        frame_number: int,
        yolo_inst: YOLO,
        local_profiles: Dict[int, LocalTrackProfile],
        local_to_global: Dict[int, int],
    ) -> Tuple[np.ndarray, List[dict], float, float]:
        """
        Run detection + embedding extraction + preliminary Re-ID for one frame.

        Returns:
            annotated_frame: Frame with bounding boxes drawn.
            pending_records: List of dicts ready for DB insertion.
            yolo_time: Time taken for YOLO detection and tracking.
            reid_time: Time taken for Re-ID batch embedding extraction.
        """
        t_yolo_start = time.time()
        result = self.detector.detect_and_track(frame, yolo_inst)
        t_yolo = time.time() - t_yolo_start

        if result is None:
            return frame, [], t_yolo, 0.0

        annotated = frame.copy()
        h, w = frame.shape[:2]
        boxes = result.boxes
        n = len(boxes)

        # Pull all coordinates at once (GPU → CPU in one go)
        all_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        all_conf = boxes.conf.cpu().numpy().astype(np.float32)
        all_ids  = boxes.id.cpu().numpy().astype(np.int32)

        # Clamp boxes
        clamped: List[Tuple[int, int, int, int]] = []
        crops:   List[Optional[np.ndarray]] = []
        for i in range(n):
            x1, y1, x2, y2 = all_xyxy[i].astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            clamped.append((x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
            else:
                crops.append(None)

        # One batch GPU inference for all crops
        t_reid_start = time.time()
        embeddings = self.extractor.extract_batch(crops)
        t_reid = time.time() - t_reid_start

        pending_records: List[dict] = []

        for i in range(n):
            local_id = int(all_ids[i])
            conf     = float(all_conf[i])
            x1, y1, x2, y2 = clamped[i]
            emb = embeddings[i]

            # ── Update LocalTrackProfile ───────────────────────────
            if local_id not in local_profiles:
                local_profiles[local_id] = LocalTrackProfile(
                    local_id, video_source, self.config
                )
            local_profiles[local_id].update(
                frame_number, (x1, y1, x2, y2), conf, crops[i], emb
            )

            # ── Preliminary global ID (quick inline match) ─────────
            if local_id in local_to_global:
                global_id = local_to_global[local_id]
            else:
                if emb is not None:
                    global_id, _ = self.identity_mgr.quick_match(emb)
                else:
                    global_id = local_id  # Fallback: use local id as placeholder
                local_to_global[local_id] = global_id

            # ── Draw overlay ───────────────────────────────────────
            self._draw_box(annotated, x1, y1, x2, y2, local_id, global_id, conf)

            # ── Accumulate DB record ───────────────────────────────
            pending_records.append({
                "global_person_id": global_id,
                "local_track_id":   local_id,
                "video_source":     video_source,
                "frame_number":     frame_number,
                "bbox_coordinates": [int(x1), int(y1), int(x2), int(y2)],
                "confidence":       round(conf, 4),
                "detected_at":      datetime.utcnow(),
                # Track-level fields filled after finalization
                "track_start_frame": None,
                "track_end_frame":   None,
                "quality_score":     None,
                "match_score":       None,
            })

        return annotated, pending_records, t_yolo, t_reid

    def _finalize_tracks(
        self,
        video_source: str,
        local_profiles: Dict[int, LocalTrackProfile],
        db_session,
    ) -> Dict[int, int]:
        """
        After the video frame loop completes:
        1. Run GlobalIdentityManager.assign_global_id() for every local track.
        2. Update person_tracks DB rows with final global ID + track metadata.
        3. Flush identity gallery to PersonIdentity table.

        Returns mapping {local_track_id: global_person_id}.
        """
        local_to_global: Dict[int, int] = {}
        track_meta: Dict[int, dict] = {}  # local_id → metadata for DB update

        for local_id, profile in local_profiles.items():
            global_id, match_score = self.identity_mgr.assign_global_id(profile)
            profile.global_person_id = global_id
            profile.match_score = match_score
            local_to_global[local_id] = global_id
            track_meta[local_id] = {
                "global_person_id":  global_id,
                "track_start_frame": profile.start_frame,
                "track_end_frame":   profile.end_frame,
                "quality_score":     profile.quality_score,
                "match_score":       match_score if match_score > 0 else None,
            }
            logger.debug("  Finalized L%d → G%d (quality=%.2f match=%.3f)",
                         local_id, global_id, profile.quality_score, match_score)

        # Bulk-update person_tracks rows for this video with finalized metadata
        if track_meta:
            try:
                for local_id, meta in track_meta.items():
                    db_session.query(PersonTrack).filter(
                        PersonTrack.video_source == video_source,
                        PersonTrack.local_track_id == local_id,
                    ).update(meta, synchronize_session=False)
                db_session.commit()
                logger.info("[%s] Updated %d track records with final global IDs.",
                            video_source, len(track_meta))
            except Exception as exc:
                db_session.rollback()
                logger.error("[%s] Failed to update track metadata: %s", video_source, exc)

        # Flush global identity galleries to DB
        self.identity_mgr.flush_to_db(db_session)

        return local_to_global

    def _flush_pending(self, db_session, pending: List[dict]) -> None:
        """Batch-insert pending DB records. Rolls back on failure."""
        if not pending:
            return
        try:
            db_session.add_all([PersonTrack(**r) for r in pending])
            db_session.commit()
        except Exception as exc:
            db_session.rollback()
            logger.error("Batch insert failed (%d records): %s", len(pending), exc)

    # ── Main video processing method ──────────────────────────────

    def track_video(
        self,
        video_path: str,
        output_path: str,
        db_session,
        db_batch_size: int = 50,
        frame_skip: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Full pipeline for a single video file.

        Phase 1 — Frame loop:
            Read → Detect → Track → Gallery-update → Quick-match → Annotate → Write

        Phase 2 — Track finalization (after frame loop):
            LocalTrackProfile[] → GlobalIdentityManager.assign_global_id()
            → Update DB rows with final global IDs and track metadata

        Args:
            video_path:        Absolute path to the input video.
            output_path:       Absolute path to save the annotated output.
            db_session:        SQLAlchemy session (thread-local).
            db_batch_size:     Flush DB records every N detections.
            frame_skip:        Process every Nth frame (1 = all).
            progress_callback: Optional callable(frames_done, total_frames).

        Returns:
            Summary dict with processing statistics.
        """
        video_source = os.path.basename(video_path)
        logger.info("▶ Starting video tracking job for: %s", video_source)
        logger.info("  Input path: %s", video_path)
        logger.info("  Output path: %s", output_path)
        logger.info("  Config - Supervised Mode: %s, Similarity Threshold: %.2f", self.config.supervised_mode, self.config.similarity_threshold)
        t_start = time.time()

        # ── Open Video ────────────────────────────────────────────
        logger.info("  Opening video file...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("  Failed to open video file: %s", video_path)
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        max_w = 1280
        if orig_width > max_w:
            scale = max_w / orig_width
            width = max_w
            height = int(orig_height * scale)
            logger.info("  Video downscaling enabled: %d×%d -> %d×%d", orig_width, orig_height, width, height)
        else:
            width = orig_width
            height = orig_height

        logger.info("  Video specs: %d frames | %.1f FPS | %d×%d px", total_frames, fps, width, height)

        # ── Output Writer ─────────────────────────────────────────
        logger.info("  Initializing video writer...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning("  VideoWriter could not be opened. Annotated output will not be saved.")

        # Borrow a YOLO instance from the pool for this job
        logger.info("  Borrowing YOLO model instance from the pool...")
        yolo_inst = self.detector.borrow_yolo()
        logger.info("  YOLO model borrowed successfully. Starting processing loop...")

        # Per-video local track profiles (local_id → LocalTrackProfile)
        local_profiles: Dict[int, LocalTrackProfile] = {}

        # Preliminary local→global mapping used during frame processing
        local_to_global: Dict[int, int] = {}

        # ── Phase 1: Frame Loop ───────────────────────────────────
        frame_number    = 0
        pending_records: List[dict] = []
        unique_local_ids: set = set()
        tick_times: List[float] = []
        read_times: List[float] = []
        resize_times: List[float] = []
        proc_times: List[float] = []
        yolo_times: List[float] = []
        reid_times: List[float] = []
        write_times: List[float] = []
        db_times: List[float] = []

        t_phase1_start = time.time()

        try:
            while True:
                t_tick_start = time.time()
                
                t_read_start = time.time()
                ret, frame = cap.read()
                read_times.append(time.time() - t_read_start)
                
                if not ret:
                    break
                    
                t_resize_start = time.time()
                if orig_width > max_w:
                    frame = cv2.resize(frame, (width, height))
                resize_times.append(time.time() - t_resize_start)

                if frame_number % frame_skip != 0:
                    t_write_start = time.time()
                    writer.write(frame)
                    write_times.append(time.time() - t_write_start)
                    
                    frame_number += 1
                    tick_times.append(time.time() - t_tick_start)
                    yolo_times.append(0.0)
                    reid_times.append(0.0)
                    continue

                t_proc_start = time.time()
                annotated, records, t_yolo, t_reid = self._process_frame(
                    frame, video_source, frame_number,
                    yolo_inst, local_profiles, local_to_global,
                )
                proc_times.append(time.time() - t_proc_start)
                yolo_times.append(t_yolo)
                reid_times.append(t_reid)
                
                t_write_start = time.time()
                writer.write(annotated)
                write_times.append(time.time() - t_write_start)

                pending_records.extend(records)
                unique_local_ids.update(r["local_track_id"] for r in records)

                t_db_start = time.time()
                if len(pending_records) >= db_batch_size:
                    self._flush_pending(db_session, pending_records)
                    pending_records.clear()
                db_times.append(time.time() - t_db_start)

                frame_number += 1
                tick_times.append(time.time() - t_tick_start)

                if progress_callback is not None and frame_number % 10 == 0:
                    try:
                        progress_callback(frame_number, total_frames)
                    except Exception:
                        pass

                if frame_number % 100 == 0:
                    pct = (frame_number / max(total_frames, 1)) * 100
                    elapsed = time.time() - t_start
                    avg_tick = np.mean(tick_times[-100:]) if tick_times else 0.0
                    avg_read = np.mean(read_times[-100:]) if read_times else 0.0
                    avg_resize = np.mean(resize_times[-100:]) if resize_times else 0.0
                    avg_proc = np.mean(proc_times[-100:]) if proc_times else 0.0
                    avg_yolo = np.mean(yolo_times[-100:]) if yolo_times else 0.0
                    avg_reid = np.mean(reid_times[-100:]) if reid_times else 0.0
                    avg_write = np.mean(write_times[-100:]) if write_times else 0.0
                    avg_db = np.mean(db_times[-100:]) if db_times else 0.0
                    logger.info(
                        "  [%s] %d/%d (%.1f%%) | elapsed: %.1fs | avg tick: %.4fs "
                        "[read: %.4fs, resize: %.4fs, proc: %.4fs (yolo: %.4fs, reid: %.4fs), write: %.4fs, db: %.4fs]",
                        video_source, frame_number, total_frames, pct, elapsed, avg_tick,
                        avg_read, avg_resize, avg_proc, avg_yolo, avg_reid, avg_write, avg_db
                    )

            # Flush remaining records BEFORE finalization
            if pending_records:
                logger.info("  [%s] Flushing final %d pending frame-level records to DB...", video_source, len(pending_records))
                self._flush_pending(db_session, pending_records)
                pending_records.clear()

        finally:
            logger.info("  [%s] Releasing video capture and writer resources...", video_source)
            cap.release()
            writer.release()
            logger.info("  [%s] Returning YOLO instance to pool...", video_source)
            self.detector.return_yolo(yolo_inst)

        t_phase1_end = time.time()
        phase1_duration = t_phase1_end - t_phase1_start
        logger.info(
            "  [%s] Phase 1 (Frame Loop) completed in %.2fs (avg tick: %.4fs/frame)\n"
            "        Breakdown - avg read: %.4fs | avg resize: %.4fs | avg proc: %.4fs (yolo: %.4fs, reid: %.4fs) | avg write: %.4fs | avg db: %.4fs", 
            video_source, phase1_duration, np.mean(tick_times) if tick_times else 0.0,
            np.mean(read_times) if read_times else 0.0,
            np.mean(resize_times) if resize_times else 0.0,
            np.mean(proc_times) if proc_times else 0.0,
            np.mean(yolo_times) if yolo_times else 0.0,
            np.mean(reid_times) if reid_times else 0.0,
            np.mean(write_times) if write_times else 0.0,
            np.mean(db_times) if db_times else 0.0
        )

        # ── Phase 2: Track Finalization ───────────────────────────
        t_phase2_start = time.time()
        logger.info("  [%s] Finalizing %d local tracks (assigning global IDs)...", video_source, len(local_profiles))
        final_local_to_global = self._finalize_tracks(
            video_source, local_profiles, db_session
        )
        t_phase2_end = time.time()
        phase2_duration = t_phase2_end - t_phase2_start
        logger.info("  [%s] Phase 2 (Track Finalization) completed in %.2fs", 
                    video_source, phase2_duration)

        # Store profiles for diagnostics
        with self._profiles_lock:
            self._track_profiles[video_source] = local_profiles

        # Compute per-video diagnostics
        unique_global_ids = sorted(set(final_local_to_global.values()))
        n_matched = sum(
            1 for p in local_profiles.values()
            if p.match_score is not None and p.match_score > 0
        )
        avg_match = (
            float(np.mean([p.match_score for p in local_profiles.values()
                           if p.match_score and p.match_score > 0]))
            if n_matched > 0 else 0.0
        )
        unmatched = len(local_profiles) - n_matched

        diag = {
            "local_track_count":  len(local_profiles),
            "global_id_count":    len(unique_global_ids),
            "avg_match_score":    round(avg_match, 4),
            "unmatched_tracks":   unmatched,
        }
        with self._diag_lock:
            self._diagnostics[video_source] = diag

        elapsed_total = time.time() - t_start

        # ── Performance Summary ───────────────────────────────────
        total_read   = sum(read_times)
        total_resize = sum(resize_times)
        total_proc   = sum(proc_times)
        total_yolo   = sum(yolo_times)
        total_reid   = sum(reid_times)
        total_write  = sum(write_times)
        total_db     = sum(db_times)
        total_finalize = phase2_duration
        accounted    = total_read + total_resize + total_proc + total_write + total_db + total_finalize
        overhead     = max(0.0, elapsed_total - accounted)

        def _pct(val: float) -> str:
            return f"{(val / elapsed_total * 100):.1f}%" if elapsed_total > 0 else "0.0%"

        logger.info(
            "═══════════════════════════════════════════════════════════\n"
            "  [%s] PERFORMANCE SUMMARY\n"
            "───────────────────────────────────────────────────────────\n"
            "  Total time:         %.2fs\n"
            "  Total frames:       %d  (processed: %d)\n"
            "  Effective FPS:      %.1f frames/sec\n"
            "───────────────────────────────────────────────────────────\n"
            "  Phase 1 (Frame Loop):      %.2fs (%s)\n"
            "    ├─ Frame Read (decode):   %.2fs (%s)\n"
            "    ├─ Frame Resize:          %.2fs (%s)\n"
            "    ├─ Processing (_process): %.2fs (%s)\n"
            "    │   ├─ YOLO Detection:    %.2fs (%s)\n"
            "    │   └─ ReID Extraction:   %.2fs (%s)\n"
            "    ├─ Frame Write (encode):  %.2fs (%s)\n"
            "    └─ DB Batch Flush:        %.2fs (%s)\n"
            "  Phase 2 (Finalization):     %.2fs (%s)\n"
            "  Overhead (misc):           %.2fs (%s)\n"
            "───────────────────────────────────────────────────────────\n"
            "  Local tracks: %d | Global IDs: %s\n"
            "  Matched: %d | Unmatched: %d | Avg match score: %.4f\n"
            "═══════════════════════════════════════════════════════════",
            video_source,
            elapsed_total,
            frame_number, len(proc_times),
            frame_number / elapsed_total if elapsed_total > 0 else 0.0,
            phase1_duration, _pct(phase1_duration),
            total_read, _pct(total_read),
            total_resize, _pct(total_resize),
            total_proc, _pct(total_proc),
            total_yolo, _pct(total_yolo),
            total_reid, _pct(total_reid),
            total_write, _pct(total_write),
            total_db, _pct(total_db),
            total_finalize, _pct(total_finalize),
            overhead, _pct(overhead),
            len(local_profiles), unique_global_ids,
            n_matched, unmatched, avg_match,
        )

        return {
            "video_source":        video_source,
            "total_frames":        frame_number,
            "unique_local_ids":    sorted(unique_local_ids),
            "unique_global_ids":   unique_global_ids,
            "local_track_count":   len(local_profiles),
            "output_path":         output_path,
            "processing_time_s":   round(elapsed_total, 2),
            "diagnostics":         diag,
        }

    # ── Backward-compat helpers (called by main.py) ───────────────

    def update_threshold(self, new_threshold: float) -> None:
        """Hot-update similarity threshold without reloading models."""
        self.identity_mgr.update_threshold(new_threshold)

    def reset_reid_cache(self) -> None:
        """Clear global identity state (called by /tracks/ DELETE endpoint)."""
        self.identity_mgr.reset()
        with self._profiles_lock:
            self._track_profiles.clear()
        logger.info("Re-ID cache and track profiles cleared.")

    def load_cache_from_db(self, db_session) -> None:
        """Load persistent identity gallery from DB (called at startup)."""
        self.identity_mgr.load_from_db(db_session)

    def get_diagnostics(self) -> Dict[str, dict]:
        """Return per-video diagnostics snapshot."""
        with self._diag_lock:
            return dict(self._diagnostics)

    def get_global_stats(self) -> dict:
        """Return current GlobalIdentityManager stats."""
        return self.identity_mgr.stats()


# ─────────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE ALIAS
# ─────────────────────────────────────────────────────────────────
# main.py imports `from tracker import MultiVideoTracker`.
# This alias ensures that import continues to work unchanged.
MultiVideoTracker = MultiVideoTrackingEngine
