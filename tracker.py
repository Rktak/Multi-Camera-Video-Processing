"""
tracker.py - Multi-Video Tracking & Re-Identification Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

Architecture Overview:
    ┌──────────────────────────────────────────────────────────┐
    │                    Video Frame                           │
    └───────────────────────┬──────────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  YOLOv8n Detection (Ultralytics)                        │
    │  • Detects all "person" bounding boxes in the frame     │
    │  • Filters by confidence ≥ 0.35                        │
    └───────────────────────┬──────────────────────────────────┘
                            │  bounding boxes
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  ByteTrack (built into ultralytics)                     │
    │  • Associates detections across consecutive frames      │
    │  • Assigns LOCAL track IDs (unique per-video only)      │
    └───────────────────────┬──────────────────────────────────┘
                            │  local_track_id + bbox
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  MobileNetV3 Feature Extractor                          │
    │  • Crops person bounding box from frame                 │
    │  • Produces 576-dim L2-normalized appearance embedding  │
    └───────────────────────┬──────────────────────────────────┘
                            │  appearance embedding
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Cosine Similarity Re-ID Cache                          │
    │  • Compares embedding against all known global IDs      │
    │  • If similarity > threshold → assign existing ID       │
    │  • Else → create new global ID                         │
    │  • Updates cache with EMA for appearance drift          │
    └───────────────────────┬──────────────────────────────────┘
                            │  global_person_id
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Annotated Frame + PostgreSQL DB Log                    │
    └──────────────────────────────────────────────────────────┘

Learning Notes:
    - LOCAL ID: Track ID assigned by ByteTrack within ONE video stream.
                Resets or duplicates if you restart processing.
    - GLOBAL ID: Our Re-ID assigned ID, consistent ACROSS all cameras.
    - EMA (Exponential Moving Average): Lets the cached embedding slowly
      adapt to the same person appearing under different lighting/angles.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
# scipy removed — cosine similarity replaced with np.dot() (L2-normalized vectors
# make dot product == cosine similarity, and numpy matmul is 5-20x faster)
from ultralytics import YOLO

from database import PersonTrack, SessionLocal


class MultiVideoTracker:
    """
    Unified tracker that handles detection, local tracking, appearance
    feature extraction, cross-video Re-ID, and DB logging for one or
    more video sources.

    Thread-safety note:
        `track_video()` calls are run from a thread pool (in main.py).
        The global_id_cache and next_global_id are SHARED across all
        threads, which is intentional — this is what enables cross-video
        Re-ID. Python's GIL provides basic protection for simple dict ops,
        but for production you'd add a threading.Lock().
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        device: str = "auto",
        yolo_model: str = "yolov8m.pt",
    ) -> None:
        """
        Initialize models and Re-ID state.

        Args:
            similarity_threshold:
                Cosine similarity threshold for Re-ID matching.
                Range 0.0–1.0. Default 0.70 balances precision/recall.
            device:
                "auto" → GPU if CUDA available, else CPU.
                "cuda" → Force GPU (raises error if no GPU).
                "cpu"  → Force CPU.
            yolo_model:
                YOLOv8 model weights file. "yolov8n.pt" is fastest;
                "yolov8s.pt" / "yolov8m.pt" are more accurate but slower.
        """
        # ── Device Configuration ──────────────────────────────────────
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("[Tracker] ⚠  CUDA requested but no NVIDIA GPU detected — falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = device
        print(f"[Tracker] Device: {self.device.upper()}")
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[Tracker] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            # Auto-tune CUDA convolution kernels for the fixed input sizes we use.
            # Adds a small one-time cost on the first batch, pays off every frame after.
            torch.backends.cudnn.benchmark = True

        # ── YOLOv8 Detection + ByteTrack ─────────────────────────────
        # The ultralytics YOLO class bundles detection + tracking together.
        # ByteTrack is activated by calling .track() instead of .predict().
        self.yolo_model = yolo_model
        self.lock = threading.Lock()

        # ── YOLO Instance Pool ────────────────────────────────────────
        # Pre-load _POOL_SIZE YOLO instances at startup.
        # track_video() borrows one from the pool and returns it when done,
        # so we NEVER reload weights from disk mid-run (saves 2–4 s per job).
        # Pool size should match THREAD_POOL max_workers in main.py.
        import queue as _queue
        _POOL_SIZE = 2
        self._yolo_pool: _queue.Queue = _queue.Queue()
        for _i in range(_POOL_SIZE):
            _inst = YOLO(yolo_model)
            _inst.to(self.device)
            self._yolo_pool.put(_inst)
        # Keep a direct reference for callers that still use self.yolo
        self.yolo = self._yolo_pool.queue[0]
        print(f"[Tracker] YOLOv8 ({yolo_model}) — {_POOL_SIZE} instances pre-loaded on {self.device.upper()}.")

        # ── MobileNetV3-Small Feature Extractor ───────────────────────
        # Why MobileNetV3?
        #   • Very small (~2.5MB download)
        #   • Fast inference (especially on GPU)
        #   • Pretrained on ImageNet → captures generic visual textures/colors
        #   • Adequate for appearance Re-ID in a learning project
        #
        # We remove the final classifier layer and use the output of
        # the avgpool layer as our appearance embedding (576 dimensions).
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # Sequential: backbone → global average pool → flatten
        # Result: 576-dim vector per image crop
        self.feature_extractor = torch.nn.Sequential(
            mobilenet.features,   # Convolutional backbone
            mobilenet.avgpool,    # 7x7 → 1x1 spatial pooling
            torch.nn.Flatten(),   # Shape: [batch, 576]
        ).to(self.device)
        self.feature_extractor.eval()  # Disable dropout/batchnorm training mode
        print("[Tracker] MobileNetV3-Small feature extractor loaded (576-dim embeddings).")

        # ── Image Preprocessing Pipeline ─────────────────────────────
        # Standard ImageNet normalization + resize to 128x64 (H x W)
        # 128x64 is the canonical Re-ID input size (tall+narrow for people)
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB mean
                std=[0.229, 0.224, 0.225],   # ImageNet RGB std
            ),
        ])

        # ── Re-ID Global ID Cache ─────────────────────────────────────
        # Maps: global_person_id (int) → mean appearance embedding (ndarray)
        # This cache is the HEART of cross-video Re-ID.
        # When a new detection arrives, its embedding is compared against
        # all entries here. The best match (if above threshold) gets the
        # corresponding global ID.
        self.global_id_cache: Dict[int, np.ndarray] = {}
        self.dirty_identities: Dict[int, np.ndarray] = {}
        self.next_global_id: int = 1
        self.similarity_threshold: float = similarity_threshold

        # ── Vectorized Embedding Matrix Cache ─────────────────────────
        # Flattened view of global_id_cache as a numpy matrix [N, 576].
        # Rebuilt lazily whenever the cache is modified (_cache_dirty=True).
        # Enables one np.matmul() call to score ALL cached identities at once.
        self._emb_matrix: Optional[np.ndarray] = None       # shape [N, 576]
        self._emb_gid_view: List[Tuple[int, int]] = []      # [(gid, view_idx), ...]
        self._cache_dirty: bool = True

        # ── Visualization ─────────────────────────────────────────────
        # Pre-generate 1000 distinct, visually-separated BGR colors
        # so each global ID always gets the same vivid color.
        np.random.seed(42)
        self._color_palette: List[Tuple[int, int, int]] = [
            tuple(int(c) for c in color)
            for color in np.random.randint(80, 255, size=(1000, 3))
        ]
        print(f"[Tracker] Ready. Re-ID threshold: {similarity_threshold:.2f}")

    # ─────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────────

    def extract_features(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert a person bounding box crop into an L2-normalized
        appearance embedding using MobileNetV3.

        Why L2-normalize?
            After normalization, cosine similarity = dot product,
            which is cheaper to compute and numerically stable.

        Args:
            image_crop: BGR numpy array (height x width x 3).

        Returns:
            numpy array of shape [576] (L2-normalized), or None if invalid.
        """
        if image_crop is None or image_crop.size == 0:
            return None

        # OpenCV uses BGR; PyTorch/PIL uses RGB — convert before processing
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply transform and add batch dimension: [1, 3, 128, 64]
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():  # No gradient computation needed (inference only)
            embedding = self.feature_extractor(tensor)

        # Move to CPU, convert to numpy, squeeze batch dim → shape [576]
        embedding_np = embedding.squeeze().cpu().numpy()

        # L2 normalize: divide by the vector's magnitude
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return None  # Degenerate embedding — skip this detection
        return embedding_np / norm

    def _rebuild_emb_matrix(self) -> None:
        """
        Flatten global_id_cache into a numpy matrix and a parallel list of
        (global_id, view_index) tuples. Called lazily when _cache_dirty=True.
        """
        gid_view: List[Tuple[int, int]] = []
        emb_list: List[np.ndarray] = []
        for gid, embeddings in self.global_id_cache.items():
            for v_idx, emb in enumerate(embeddings):
                gid_view.append((gid, v_idx))
                emb_list.append(emb)
        self._emb_gid_view = gid_view
        self._emb_matrix = np.array(emb_list, dtype=np.float32) if emb_list else None
        self._cache_dirty = False

    def extract_features_batch(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Batch version of extract_features() — runs ONE GPU forward pass for ALL
        person crops in a frame instead of N separate passes.

        Why faster?
            GPU throughput is wasted on tiny single-image batches.
            Batching N crops into [N, 3, 128, 64] saturates GPU parallelism.

        Uses cv2 resize (faster than PIL) and inlined numpy normalization
        (avoids torchvision transform overhead per image).

        Args:
            crops: List of BGR numpy arrays, one per detected person.

        Returns:
            List of L2-normalized float32 embeddings (same length as crops).
            Entries are None for invalid/empty crops.
        """
        results: List[Optional[np.ndarray]] = [None] * len(crops)
        if not crops:
            return results

        tensors: List[torch.Tensor] = []
        valid_indices: List[int] = []
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            # cv2.resize is faster than PIL.Image.resize for this size
            resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)  # (W, H)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _mean) / _std   # ImageNet normalization (in-place on float32 copy)
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)))  # HWC → CHW
            valid_indices.append(i)

        if not tensors:
            return results

        # One batched GPU forward pass: [N, 3, 128, 64] → [N, 576]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.feature_extractor(batch)

        embeddings_np = embeddings.cpu().numpy()
        for out_idx, orig_idx in enumerate(valid_indices):
            emb = embeddings_np[out_idx]
            norm = np.linalg.norm(emb)
            results[orig_idx] = (emb / norm) if norm > 0 else None

        return results

    # ─────────────────────────────────────────────────────────────────
    # RE-IDENTIFICATION — GLOBAL ID MATCHING
    # ─────────────────────────────────────────────────────────────────

    def match_global_id(self, features: np.ndarray, is_overlapping: bool = False) -> int:
        """
        Vectorized Re-ID matching — replaces per-embedding Python loop with
        a single numpy matrix multiply.

        Because every embedding is L2-normalized (unit vector):
            cosine_similarity(a, b)  ==  dot(a, b)
        So:  similarities = emb_matrix @ features
        gives cosine similarities to ALL cached identities in one BLAS call.

        Algorithm:
            1. Rebuild the flat [N×576] embedding matrix if the cache changed.
            2. One matmul → [N] similarity scores.
            3. If max score ≥ threshold  → assign that global ID + EMA update.
            4. Else                       → mint a new global ID.

        Args:
            features:      L2-normalized embedding of shape [576].
            is_overlapping: Flag to skip cache update during occlusion.

        Returns:
            Assigned global_person_id (int ≥ 1).
        """
        with self.lock:
            if not self.global_id_cache:
                gid = self.next_global_id
                self.global_id_cache[gid] = [features.copy()]
                self.dirty_identities[gid] = self.global_id_cache[gid]
                self.next_global_id += 1
                self._cache_dirty = True
                return gid

            # Lazy rebuild: only when something was added/updated
            if self._cache_dirty:
                self._rebuild_emb_matrix()

            if self._emb_matrix is None:
                gid = self.next_global_id
                self.global_id_cache[gid] = [features.copy()]
                self.dirty_identities[gid] = self.global_id_cache[gid]
                self.next_global_id += 1
                self._cache_dirty = True
                return gid

            # ── ONE matmul replaces the entire nested for-loop ────────
            # [N, 576] @ [576] → [N]  (dot product == cosine sim for unit vecs)
            feat_f32 = features.astype(np.float32)
            similarities = self._emb_matrix @ feat_f32   # [N]
            best_flat_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_flat_idx])
            best_gid, best_view_idx = self._emb_gid_view[best_flat_idx]

            if best_similarity >= self.similarity_threshold:
                # ── Match found ──────────────────────────────────────────
                if not is_overlapping:
                    alpha = 0.9  # EMA smoothing factor
                    matched_emb = self.global_id_cache[best_gid][best_view_idx]
                    updated = alpha * matched_emb + (1 - alpha) * feat_f32
                    updated_norm = np.linalg.norm(updated)
                    self.global_id_cache[best_gid][best_view_idx] = (
                        updated / updated_norm if updated_norm > 0 else updated
                    )
                    self.dirty_identities[best_gid] = self.global_id_cache[best_gid]
                    self._cache_dirty = True  # Matrix stale after EMA update
                return best_gid
            else:
                # ── New person detected ───────────────────────────────────
                gid = self.next_global_id
                self.global_id_cache[gid] = [features.copy()]
                self.dirty_identities[gid] = self.global_id_cache[gid]
                self.next_global_id += 1
                self._cache_dirty = True
                return gid

    def update_global_cache(self, global_id: int, features: np.ndarray) -> None:
        """Update the appearance cache for an existing global ID, adding or updating views (up to 3).

        Uses np.dot() instead of scipy cosine — equivalent for L2-normalized vectors,
        and avoids a function-call overhead per view.
        """
        with self.lock:
            if global_id not in self.global_id_cache:
                return

            embeddings = self.global_id_cache[global_id]
            feat_f32 = features.astype(np.float32)

            # Vectorized similarity: one dot per view (views are always unit vecs)
            similarities = [float(np.dot(feat_f32, emb)) for emb in embeddings]
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]

            new_view_threshold = 0.20
            max_views = 3

            alpha = 0.9
            if best_sim >= new_view_threshold:
                # Update the closest matching view using EMA
                updated = alpha * embeddings[best_idx] + (1 - alpha) * feat_f32
                updated_norm = np.linalg.norm(updated)
                embeddings[best_idx] = updated / updated_norm if updated_norm > 0 else updated
            elif len(embeddings) < max_views:
                # New view: sufficiently dissimilar and slots available
                embeddings.append(feat_f32.copy())
                print(f"[Tracker] Added new appearance view {len(embeddings)} for Global ID {global_id} (sim: {best_sim:.2f})")
            else:
                # Slots full → update the closest view
                updated = alpha * embeddings[best_idx] + (1 - alpha) * feat_f32
                updated_norm = np.linalg.norm(updated)
                embeddings[best_idx] = updated / updated_norm if updated_norm > 0 else updated

            self.dirty_identities[global_id] = embeddings
            self._cache_dirty = True  # Embedding matrix must be rebuilt next match

    # ─────────────────────────────────────────────────────────────────
    # FRAME-LEVEL PROCESSING
    # ─────────────────────────────────────────────────────────────────

    def _get_color(self, global_id: int) -> Tuple[int, int, int]:
        """Return a consistent BGR color for a given global person ID."""
        return self._color_palette[global_id % len(self._color_palette)]

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = float(area1 + area2 - intersection_area)
        
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def process_frame(
        self,
        frame: np.ndarray,
        video_source: str,
        frame_number: int,
        yolo_instance: YOLO,
        local_to_global: Dict[int, int],
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run the full detection → tracking → Re-ID pipeline on one frame.

        Args:
            frame:        BGR numpy array (H × W × 3) — raw video frame.
            video_source: Filename of the source video (used for DB logging).
            frame_number: 0-based index of this frame within the video.
            yolo_instance: Thread-local/job-local YOLO model instance for safety.
            local_to_global: Mapping of local track IDs to global person IDs for consistency.

        Returns:
            annotated_frame: Frame with bounding boxes and global IDs drawn.
            track_records:   List of dicts, each ready to insert into DB.
        """
        track_records: List[dict] = []

        # ── YOLOv8 + ByteTrack ───────────────────────────────────────
        # .track() instead of .predict() enables ByteTrack state tracking.
        # persist=True: carry ByteTrack state from the PREVIOUS call
        #               (critical for continuous tracking across frames)
        # classes=[0]:  COCO class 0 = "person" — ignore cars, bikes, etc.
        # conf=0.35:    Minimum detection confidence to consider
        # iou=0.45:     NMS overlap threshold (lower = fewer overlapping boxes)
        results = yolo_instance.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.35,
            iou=0.45,
            device=self.device,
            verbose=False,  # Suppress noisy per-frame YOLO output
        )

        # No detections or tracker not yet initialized
        if results[0].boxes is None or results[0].boxes.id is None:
            return frame, track_records   # Return original — no copy needed if nothing to draw

        annotated_frame = frame.copy()   # Defer copy until we know there are detections
        boxes = results[0].boxes
        h, w = frame.shape[:2]
        n_boxes = len(boxes)

        # ── Vectorized IoU Overlap Detection ──────────────────────────
        # Replaces the O(n²) nested Python loop with numpy broadcasting.
        # All box coordinates pulled from GPU to CPU in one go.
        all_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)  # [N, 4]
        overlapping_indices: set = set()
        if n_boxes > 1:
            x1_n = all_xyxy[:, 0][:, None];  y1_n = all_xyxy[:, 1][:, None]
            x2_n = all_xyxy[:, 2][:, None];  y2_n = all_xyxy[:, 3][:, None]
            ix1 = np.maximum(x1_n, all_xyxy[:, 0])
            iy1 = np.maximum(y1_n, all_xyxy[:, 1])
            ix2 = np.minimum(x2_n, all_xyxy[:, 2])
            iy2 = np.minimum(y2_n, all_xyxy[:, 3])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)  # [N, N]
            areas = (all_xyxy[:, 2] - all_xyxy[:, 0]) * (all_xyxy[:, 3] - all_xyxy[:, 1])  # [N]
            union = areas[:, None] + areas[None, :] - inter
            iou_mat = np.where(union > 0, inter / union, 0.0)
            np.fill_diagonal(iou_mat, 0)
            pairs = np.argwhere(iou_mat > 0.02)
            overlapping_indices = {int(idx) for pair in pairs for idx in pair}

        # ── Pass 1: collect all crops for batch GPU inference ─────────
        # We gather every person crop first so MobileNetV3 runs ONE forward
        # pass over all N crops, not N separate single-image passes.
        crops: List[Optional[np.ndarray]] = []
        valid_boxes: List[Tuple[int, int, int, int]] = []   # clamped (x1,y1,x2,y2)
        confs: List[float] = []
        local_ids: List[int] = []

        for i in range(n_boxes):
            x1, y1, x2, y2 = all_xyxy[i].astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                crops.append(None)
            else:
                crops.append(frame[y1:y2, x1:x2])
            valid_boxes.append((x1, y1, x2, y2))
            confs.append(float(boxes.conf[i].item()))
            local_ids.append(int(boxes.id[i].item()))

        # Batched feature extraction — one GPU forward pass for the whole frame
        all_features = self.extract_features_batch(crops)

        # ── Pass 2: Re-ID matching + annotation ───────────────────────
        for i in range(n_boxes):
            if crops[i] is None:
                continue

            x1, y1, x2, y2 = valid_boxes[i]
            confidence = confs[i]
            local_id = local_ids[i]
            features = all_features[i]
            is_overlapping = (i in overlapping_indices)

            if features is None:
                continue

            feat_f32 = features.astype(np.float32)

            # ── Global Re-ID Assignment ───────────────────────────────
            if local_id in local_to_global:
                global_id = local_to_global[local_id]

                if not is_overlapping:
                    # Check for appearance drift (suggests a ByteTrack ID swap after crossing)
                    embeddings = self.global_id_cache.get(global_id, [])
                    current_sim = -1.0
                    if embeddings:
                        # np.dot instead of scipy cosine (L2-normalized → equivalent)
                        current_sim = float(max(np.dot(feat_f32, emb) for emb in embeddings))

                    if current_sim < 0.65 and self._emb_matrix is not None:
                        # Search entire cache for a better match using vectorized matmul
                        with self.lock:
                            if self._cache_dirty:
                                self._rebuild_emb_matrix()
                            if self._emb_matrix is not None:
                                all_sims = self._emb_matrix @ feat_f32
                                best_flat = int(np.argmax(all_sims))
                                best_sim = float(all_sims[best_flat])
                                best_gid, _ = self._emb_gid_view[best_flat]
                                if best_sim >= 0.78 and best_gid != global_id:
                                    print(
                                        f"[Tracker] Corrected ID switch: Local {local_id} "
                                        f"{global_id}→{best_gid} "
                                        f"(sim {current_sim:.2f}→{best_sim:.2f})"
                                    )
                                    global_id = best_gid
                                    local_to_global[local_id] = global_id

                    self.update_global_cache(global_id, feat_f32)
            else:
                # New local track — assign a global Re-ID
                global_id = self.match_global_id(feat_f32, is_overlapping=is_overlapping)
                local_to_global[local_id] = global_id

            color = self._get_color(global_id)  # BGR tuple

            # ── Draw Bounding Box ─────────────────────────────────────
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            label = f" ID: {global_id} "
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y_top = max(y1 - label_h - baseline - 6, 0)
            cv2.rectangle(
                annotated_frame,
                (x1, label_y_top),
                (x1 + label_w, y1),
                color, -1,  # -1 = filled rectangle
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text for readability
                2,
                cv2.LINE_AA,
            )

            # ── Confidence badge (bottom-left of box) ─────────────────
            conf_label = f"{confidence:.0%}"
            cv2.putText(
                annotated_frame,
                conf_label,
                (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            # ── Collect DB Record ─────────────────────────────────────
            track_records.append({
                "global_person_id": global_id,
                "local_track_id": local_id,
                "video_source": video_source,
                "frame_number": frame_number,
                "bbox_coordinates": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(confidence, 4),
                "detected_at": datetime.utcnow(),
            })

        return annotated_frame, track_records

    # ─────────────────────────────────────────────────────────────────
    # VIDEO-LEVEL PROCESSING
    # ─────────────────────────────────────────────────────────────────

    def track_video(
        self,
        video_path: str,
        output_path: str,
        db_session,
        db_batch_size: int = 50,
        frame_skip: int = 1,
    ) -> dict:
        """
        Full pipeline for a single video file:
            Read frames → Detect+Track+Re-ID → Write annotated video → Log to DB

        Args:
            video_path:    Absolute path to the input video file.
            output_path:   Absolute path to save the annotated output video.
            db_session:    SQLAlchemy session (from SessionLocal()).
            db_batch_size: Flush DB records every N detections (improves perf).
            frame_skip:    Process every Nth frame (1=all frames, 2=every other).
                           Use 2 or 3 for faster processing at lower accuracy.

        Returns:
            Summary dict with processing statistics.
        """
        video_source = os.path.basename(video_path)
        print(f"\n[Tracker] ▶ Starting: {video_source}")
        t_start = time.time()

        # ── Open Video ────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[Tracker] Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Tracker]   → {total_frames} frames | {fps:.1f} FPS | {width}×{height}px")

        # ── Setup Output Video Writer ─────────────────────────────────
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # avc1 codec (H.264) → .mp4 container (required for HTML5 browser playback)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Borrow a pre-loaded YOLO instance from the pool.
        # This avoids reloading weights from disk on every job call (saves 2–4 s).
        # The instance is returned to the pool in the finally block.
        yolo_instance = self._yolo_pool.get()

        # Map local track IDs to global person IDs within this video run for consistency
        local_to_global: Dict[int, int] = {}

        # ── Processing Loop ───────────────────────────────────────────
        frame_number = 0
        pending_records: List[dict] = []
        unique_global_ids: set = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Optional frame skipping for performance
            if frame_number % frame_skip != 0:
                writer.write(frame)  # Write unmodified frame to keep audio-sync
                frame_number += 1
                continue

            # Run the detection + Re-ID pipeline on this frame
            annotated_frame, track_records = self.process_frame(
                frame, video_source, frame_number, yolo_instance, local_to_global
            )
            writer.write(annotated_frame)

            # Accumulate DB records
            pending_records.extend(track_records)
            unique_global_ids.update(r["global_person_id"] for r in track_records)

            # Batch insert to reduce DB round-trips
            if len(pending_records) >= db_batch_size:
                self._flush_to_db(db_session, pending_records)
                pending_records.clear()

            frame_number += 1

            # Progress log every 100 frames
            if frame_number % 100 == 0:
                elapsed = time.time() - t_start
                pct = (frame_number / max(total_frames, 1)) * 100
                print(f"[Tracker]   [{video_source}] {frame_number}/{total_frames} frames "
                      f"({pct:.1f}%) | {elapsed:.1f}s elapsed")

        # Flush any remaining DB records
        if pending_records:
            self._flush_to_db(db_session, pending_records)

        cap.release()
        writer.release()

        # Return YOLO instance to the pool so the next job can reuse it
        self._yolo_pool.put(yolo_instance)

        elapsed_total = time.time() - t_start
        print(f"[Tracker] ✓ Done: {video_source} in {elapsed_total:.1f}s")
        print(f"[Tracker]   Unique global IDs found: {sorted(unique_global_ids)}")

        return {
            "video_source": video_source,
            "total_frames": frame_number,
            "unique_global_ids": sorted(unique_global_ids),
            "output_path": output_path,
            "processing_time_s": round(elapsed_total, 2),
        }

    def _flush_to_db(self, db_session, records: List[dict]) -> None:
        """
        Batch-insert a list of tracking records into PostgreSQL.
        Using add_all() is clean and Pythonic for SQLAlchemy 2.0+.
        The try/except ensures a failed batch doesn't kill the whole job.
        """
        if not records:
            return
        try:
            db_session.add_all([PersonTrack(**r) for r in records])
            db_session.commit()
        except Exception as exc:
            db_session.rollback()
            print(f"[DB] Batch insert failed ({len(records)} records): {exc}")

        # Batch-upsert dirty identities to the database
        with self.lock:
            to_flush = self.dirty_identities.copy()
            self.dirty_identities.clear()

        if to_flush:
            from database import PersonIdentity
            try:
                for gid, embs in to_flush.items():
                    emb_list = [emb.tolist() for emb in embs]
                    identity_rec = db_session.query(PersonIdentity).filter(PersonIdentity.global_person_id == gid).first()
                    if identity_rec:
                        identity_rec.embedding = emb_list
                    else:
                        identity_rec = PersonIdentity(global_person_id=gid, embedding=emb_list)
                        db_session.add(identity_rec)
                db_session.commit()
                print(f"[DB] Flushed {len(to_flush)} identities to PostgreSQL.")
            except Exception as exc:
                db_session.rollback()
                print(f"[DB] Failed to flush identities: {exc}")

    def update_threshold(self, new_threshold: float) -> None:
        """Hot-update the Re-ID similarity threshold without reloading models."""
        self.similarity_threshold = float(new_threshold)
        print(f"[Tracker] Re-ID threshold updated to {new_threshold:.2f}")

    def reset_reid_cache(self) -> None:
        """Clear the Re-ID cache and reset the global ID counter.
        Useful between unrelated processing jobs."""
        self.global_id_cache.clear()
        self.dirty_identities.clear()
        self.next_global_id = 1
        print("[Tracker] Re-ID cache cleared. Global IDs reset to 1.")

    def load_cache_from_db(self, db_session) -> None:
        """Load persistent person identities and embeddings from PostgreSQL into memory cache."""
        from database import PersonIdentity
        with self.lock:
            try:
                records = db_session.query(PersonIdentity).all()
                self.global_id_cache.clear()
                self.dirty_identities.clear()
                max_id = 0
                for r in records:
                    # Support both list of lists (multi-view) and single list (backwards compatibility)
                    if isinstance(r.embedding[0], list):
                        self.global_id_cache[r.global_person_id] = [
                            np.array(e, dtype=np.float32) for e in r.embedding
                        ]
                    else:
                        self.global_id_cache[r.global_person_id] = [
                            np.array(r.embedding, dtype=np.float32)
                        ]
                    if r.global_person_id > max_id:
                        max_id = r.global_person_id
                self.next_global_id = max_id + 1
                print(f"[Tracker] Loaded {len(records)} identities from database. Next Global ID: {self.next_global_id}")
            except Exception as exc:
                print(f"[Tracker] Error loading cache from DB: {exc}")
