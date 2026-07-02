"""
app.py - Streamlit Frontend (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

How to run:
    streamlit run app.py

Make sure the FastAPI backend is running first:
    uvicorn main:app --reload --port 8000

CHANGED (v2 — track-level Re-ID upgrade):
    • Sidebar additions:
        - Gallery & Crop Quality section (gallery_size, min_crop_quality)
        - Matching Weights section (appearance, temporal, transition) with
          normalisation preview
        - Camera Topology JSON text input with validation
    • Upload flow passes all new parameters to the backend.
    • Tab 3 (Dashboard) gains a Diagnostics panel showing per-video stats
      fetched from the new /diagnostics/ endpoint.
    • /stats/ response shows avg_quality_score and avg_match_score.
"""

import json
import os
import tempfile
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCMOT Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Multi-Camera Multi-Object Tracking System v2 — Track-Level Re-ID",
    },
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.3rem; line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem;
    }
    .pipeline-step {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; color: #a5b4fc; margin: 3px; font-weight: 500;
    }
    .pipeline-arrow { color: #475569; font-size: 1rem; margin: 0 2px; }

    .badge-done    { background:#064e3b; color:#6ee7b7; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-error   { background:#7f1d1d; color:#fca5a5; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-process { background:#1e3a5f; color:#93c5fd; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-pending { background:#3b2d00; color:#fde68a; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }

    .cam-label {
        font-size: 0.95rem; font-weight: 600; color: #e2e8f0;
        padding: 6px 0 4px 0; border-bottom: 2px solid #667eea; margin-bottom: 8px;
    }
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px; padding: 12px 16px;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(100, 116, 139, 0.2);
    }
    button[data-baseweb="tab"] { font-weight: 600; }
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important; padding: 12px !important;
    }
    .diag-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .weight-bar {
        height: 8px; border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #a78bfa);
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────
DEFAULT_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
MAX_POLL_SECONDS = 900
POLL_INTERVAL_S  = 3


def check_server(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def status_badge(status: str) -> str:
    badges = {
        "done":       '<span class="badge-done">✓ Done</span>',
        "error":      '<span class="badge-error">✗ Error</span>',
        "processing": '<span class="badge-process">⟳ Processing</span>',
        "pending":    '<span class="badge-pending">⏳ Pending</span>',
    }
    return badges.get(status, f"<code>{status}</code>")


def _normalise_weights(w_app: float, w_temp: float, w_trans: float):
    """Return normalised weights that sum to 1.0."""
    total = w_app + w_temp + w_trans
    if total <= 0:
        return 0.7, 0.2, 0.1
    return w_app / total, w_temp / total, w_trans / total


# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
for key, default in {
    "job_id":          None,
    "processing_done": False,
    "job_results":     [],
    "last_error":      None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 MCMOT System v2")
    st.markdown("---")

    api_url = st.text_input(
        "FastAPI Server URL",
        value=DEFAULT_API_URL,
        help="Change this if you're running the backend on a different host/port.",
    )
    server_ok = check_server(api_url)
    if server_ok:
        st.success("🟢 Backend: Online", icon="✅")
    else:
        st.error("🔴 Backend: Offline")
        st.code("uvicorn main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### ⚙️ Re-ID Settings")

    # ReID model selector
    _REID_OPTIONS = {
        "OSNet-AIN (recommended)": "osnet_ain",
        "OSNet":                   "osnet",
        "MobileNetV3 (legacy)":    "mobilenetv3",
    }
    reid_model_label = st.selectbox(
        "Re-ID Backbone Model",
        options=list(_REID_OPTIONS.keys()),
        index=0,
        help=(
            "The neural network used to extract appearance embeddings for Re-ID.\n\n"
            "• **OSNet-AIN** — Purpose-built for person Re-ID with Instance Norm "
            "for cross-camera robustness. 512-dim, ~8 MB. (Recommended)\n"
            "• **OSNet** — Same architecture without Instance Norm, slightly faster.\n"
            "• **MobileNetV3** — General-purpose ImageNet features (576-dim). "
            "Legacy fallback."
        ),
    )
    reid_model_key = _REID_OPTIONS[reid_model_label]

    st.markdown("---")
    st.markdown("### 🔍 Matching & Filtering")

    supervised_mode = st.toggle(
        "Supervised Mode (Taught Persons Only)",
        value=True,
        help="If enabled, only people explicitly taught via the 'Teach Person' tab will be labeled. All others will be labeled as 'Unknown'."
    )

    use_onnx = st.toggle(
        "Enable ONNX Backend",
        value=False,
        help="Use ONNX for YOLO inference. Only beneficial if you have a supported GPU (CUDA). Can be slower on CPU-only machines."
    )

    similarity_threshold = st.slider(
        "Cosine Similarity Threshold",
        min_value=0.50, max_value=0.99, value=0.70, step=0.01,
        help=(
            "Minimum combined score to merge a track into an existing global ID.\n\n"
            "↑ Higher (0.85+): Stricter — fewer false merges, may split one person.\n"
            "↓ Lower (0.55-): More lenient — better cross-camera recall."
        ),
    )

    frame_skip = st.select_slider(
        "Frame Skip",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=1,
        help=(
            "Process every Nth frame.\n"
            "• 1 = all frames (slowest, best)\n"
            "• 2 = every other frame (~2× faster)\n"
            "• 3 = every 3rd frame (~3× faster)"
        ),
    )

    st.markdown("---")
    st.markdown("### 🖼️ Gallery & Crop Quality")

    gallery_size = st.slider(
        "Gallery Size (crops per track)",
        min_value=2, max_value=16, value=8, step=1,
        help=(
            "Maximum number of appearance crops stored per ByteTrack local track.\n"
            "Larger gallery → more robust Re-ID, but uses more memory.\n"
            "Recommended: 6–12."
        ),
    )

    min_crop_quality = st.slider(
        "Min Crop Confidence (gallery gate)",
        min_value=0.30, max_value=0.70, value=0.40, step=0.05,
        help=(
            "Minimum YOLO detection confidence for a crop to be added to the gallery.\n"
            "Lower-confidence detections are still tracked and displayed, but not\n"
            "used to build the Re-ID embedding gallery."
        ),
    )

    st.markdown("---")
    st.markdown("### ⚖️ Matching Weights")
    st.caption("Weights are normalised to sum to 1.0 before use.")

    weight_appearance = st.slider(
        "Appearance weight", 0.0, 1.0, 0.70, 0.05,
        help="Gallery cosine similarity score weight.",
    )
    weight_temporal = st.slider(
        "Temporal weight", 0.0, 0.50, 0.20, 0.05,
        help="Temporal compatibility score weight.",
    )
    weight_transition = st.slider(
        "Transition weight", 0.0, 0.30, 0.10, 0.05,
        help="Camera topology transition prior weight.",
    )

    # Normalised preview
    na, nt, ntr = _normalise_weights(weight_appearance, weight_temporal, weight_transition)
    st.markdown(
        f"**Normalised:** `app={na:.2f}` · `temp={nt:.2f}` · `trans={ntr:.2f}`",
    )

    st.markdown("---")
    st.markdown("### 🗺️ Camera Topology (optional)")
    st.caption(
        "Expected travel times between cameras in seconds.\n"
        'Example: `{"cam1.mp4": {"cam2.mp4": [2, 10]}}`'
    )
    topology_input = st.text_area(
        "Camera Topology JSON",
        value="",
        height=100,
        key="topology_json",
        label_visibility="collapsed",
        placeholder='{\n  "cam1.mp4": {"cam2.mp4": [2, 10]}\n}',
    )
    topology_valid = True
    topology_parsed = None
    if topology_input.strip():
        try:
            topology_parsed = json.loads(topology_input.strip())
            st.success("✅ Valid JSON topology")
        except json.JSONDecodeError as exc:
            st.error(f"❌ Invalid JSON: {exc}")
            topology_valid = False

    st.markdown("---")
    st.markdown("### 📖 Pipeline (v2)")

    # Dynamic model label for the pipeline display
    _emb_label = {
        "osnet_ain":    "OSNet-AIN Embeddings (512-d)",
        "osnet":        "OSNet Embeddings (512-d)",
        "mobilenetv3":  "MobileNetV3 Embeddings (576-d)",
    }.get(reid_model_key, "Re-ID Embeddings")

    st.markdown(f"""
    <div style='line-height:2.2'>
    <span class='pipeline-step'>📹 Video Upload</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🔍 YOLOv8 Detection</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🏃 ByteTrack (Local IDs)</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🖼️ Gallery Accumulation</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🧬 {_emb_label}</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🔗 Track-Level Re-ID</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🗄️ PostgreSQL Logging</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("MCMOT v2 — Track-Level Re-ID")


# ─────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎯 Multi-Camera Multi-Object Tracker</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload multiple video streams · Track people with YOLOv8 + ByteTrack · '
    'Assign global identities across cameras with track-level gallery Re-ID</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab_teach, tab_upload, tab_results, tab_dashboard = st.tabs([
    "🎓 Teach Person",
    "📤 Upload & Process",
    "🎬 Results",
    "📊 Tracking Dashboard",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 0 — TEACH PERSON
# ═══════════════════════════════════════════════════════════════════
with tab_teach:
    st.markdown("### 🎓 Teach the System a New Identity")
    st.markdown(
        "Upload a short video containing **only one person** (or mostly one person) "
        "and provide their name. The system will learn to recognize them."
    )
    
    with st.form("teach_form"):
        teach_name = st.text_input("Person Name", placeholder="e.g., John Doe")
        teach_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="teach_video")
        
        submitted = st.form_submit_button("Teach Identity", type="primary")
        
        if submitted:
            if not teach_name or not teach_video:
                st.error("Please provide both a name and a video.")
            else:
                with st.spinner(f"Teaching system to recognize {teach_name}..."):
                    files = {"file": (teach_video.name, teach_video.getvalue(), teach_video.type)}
                    data = {"person_name": teach_name}
                    try:
                        resp = requests.post(f"{api_url}/teach-person/", data=data, files=files)
                        resp.raise_for_status()
                        result = resp.json()
                        if result.get("status") == "success":
                            st.success(result["message"])
                        else:
                            st.error(result.get("message", "Unknown error"))
                    except Exception as e:
                        st.error(f"Error teaching identity: {e}")

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & PROCESS
# ═══════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📁 Upload Videos")
    st.markdown(
        "Upload **2 or more** video files (`.mp4`, `.avi`, `.mov`). "
        "Each video represents one camera stream."
    )

    uploaded_files = st.file_uploader(
        "Drag and drop video files here, or click to browse",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
        key="video_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        file_info_cols = st.columns([3, 1, 1])
        file_info_cols[0].markdown("**Filename**")
        file_info_cols[1].markdown("**Size**")
        file_info_cols[2].markdown("**Type**")
        for f in uploaded_files:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"📹 `{f.name}`")
            c2.caption(f"{f.size / 1024 / 1024:.2f} MB")
            c3.caption(f.type or "video")

        st.markdown("---")

        # Show a summary of the active settings before processing
        with st.expander("⚙️ Active Settings Summary", expanded=False):
            na, nt, ntr = _normalise_weights(weight_appearance, weight_temporal, weight_transition)
            settings_cols = st.columns(3)
            with settings_cols[0]:
                st.metric("Similarity Threshold", f"{similarity_threshold:.2f}")
                st.metric("Frame Skip", frame_skip)
                st.metric("ONNX Enabled", "Yes" if use_onnx else "No")
            with settings_cols[1]:
                st.metric("Gallery Size", gallery_size)
                st.metric("Min Crop Confidence", f"{min_crop_quality:.2f}")
            with settings_cols[2]:
                st.metric("App / Temp / Trans", f"{na:.2f} / {nt:.2f} / {ntr:.2f}")
                if topology_parsed:
                    st.metric("Camera Topology", f"{len(topology_parsed)} camera(s)")
                else:
                    st.metric("Camera Topology", "None (appearance-only)")

        btn_col1, btn_col2, _ = st.columns([2, 1, 3])
        with btn_col1:
            process_btn = st.button(
                "🚀 Process Videos",
                type="primary",
                use_container_width=True,
                disabled=not server_ok or not topology_valid,
            )
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["job_id"]          = None
                st.session_state["processing_done"] = False
                st.session_state["job_results"]     = []
                st.rerun()

        if not server_ok:
            st.warning("⚠️ FastAPI backend is offline. Start it before processing.")
        if not topology_valid:
            st.warning("⚠️ Fix the camera topology JSON before processing.")

        # ── Processing Logic ──────────────────────────────────────────
        if process_btn and server_ok and topology_valid:
            st.session_state["processing_done"] = False
            st.session_state["job_results"]     = []
            st.session_state["last_error"]      = None

            with st.status("🚀 Submitting to backend...", expanded=True) as status_box:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "video/mp4"))
                    for f in uploaded_files
                ]

                params = {
                    "similarity_threshold": similarity_threshold,
                    "frame_skip":           frame_skip,
                    "gallery_size":         gallery_size,
                    "weight_appearance":    weight_appearance,
                    "weight_temporal":      weight_temporal,
                    "weight_transition":    weight_transition,
                    "min_crop_quality":     min_crop_quality,
                    "reid_model":           reid_model_key,
                    "supervised_mode":      str(supervised_mode).lower(),
                    "use_onnx":             str(use_onnx).lower(),
                }
                if topology_parsed:
                    params["camera_topology"] = json.dumps(topology_parsed)

                st.write(f"⬆️ Uploading {len(files_payload)} file(s)...")
                try:
                    upload_resp = requests.post(
                        f"{api_url}/upload-videos/",
                        files=files_payload,
                        params=params,
                        timeout=120,
                    )
                    upload_resp.raise_for_status()
                    job_data = upload_resp.json()
                    job_id   = job_data["job_id"]
                    st.session_state["job_id"] = job_id
                    st.write(f"✅ Job created: `{job_id}`")

                    # Show active settings
                    applied = job_data.get("settings", {})
                    na2, nt2, ntr2 = _normalise_weights(
                        applied.get("weight_appearance", weight_appearance),
                        applied.get("weight_temporal",   weight_temporal),
                        applied.get("weight_transition",  weight_transition),
                    )
                    st.write(
                        f"   Threshold: `{applied.get('similarity_threshold', similarity_threshold)}` | "
                        f"Gallery: `{applied.get('gallery_size', gallery_size)}` | "
                        f"Weights: `app={na2:.2f} temp={nt2:.2f} trans={ntr2:.2f}`"
                    )

                    # ── Polling Loop ──────────────────────────────────
                    st.write("📡 Processing videos in parallel — polling every 3s...")

                    video_names = [f.name for f in uploaded_files]
                    progress_containers = {}
                    for vname in video_names:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;📹 **{vname}**")
                        progress_containers[vname] = {
                            "bar":   st.progress(0.0),
                            "label": st.empty(),
                        }

                    overall_label = st.empty()
                    elapsed = 0

                    while elapsed < MAX_POLL_SECONDS:
                        time.sleep(POLL_INTERVAL_S)
                        elapsed += POLL_INTERVAL_S

                        poll_resp   = requests.get(f"{api_url}/status/{job_id}", timeout=10)
                        job_status  = poll_resp.json()
                        curr_status = job_status["status"]
                        video_prog  = job_status.get("video_progress", {})

                        for vname, container in progress_containers.items():
                            safe_name = vname.replace(" ", "_")
                            vp = (
                                video_prog.get(safe_name)
                                or video_prog.get(vname)
                                or {}
                            )
                            frames_done  = vp.get("frames_done",  0)
                            total_frames = vp.get("total_frames", 0)
                            vstatus      = vp.get("status", "pending")

                            if vstatus == "done":
                                pct        = 1.0
                                label_text = f"✅ Done — {total_frames:,} frames"
                            elif vstatus == "error":
                                pct        = 1.0
                                label_text = "❌ Error during processing"
                            elif total_frames > 0:
                                pct = min(frames_done / total_frames, 1.0)
                                label_text = f"⚙️ {frames_done:,} / {total_frames:,} frames ({pct:.0%})"
                            elif vstatus == "processing":
                                pct        = 0.02
                                label_text = "⏳ Starting..."
                            else:
                                pct        = 0.0
                                label_text = "⏸ Pending..."

                            container["bar"].progress(pct)
                            container["label"].caption(label_text)

                        overall_label.info(f"🟡 Overall: `{curr_status}` — {elapsed}s elapsed")

                        if curr_status == "done":
                            overall_label.success("✅ All videos complete!")
                            status_box.update(
                                label="✅ All Videos Processed!",
                                state="complete", expanded=False,
                            )
                            st.session_state["processing_done"] = True
                            st.session_state["job_results"]     = job_status.get("results", [])
                            st.success(
                                "🎉 Processing complete! Switch to the **🎬 Results** tab "
                                "to view the annotated videos."
                            )
                            break
                        elif curr_status == "error":
                            overall_label.error("❌ Processing failed")
                            status_box.update(label="❌ Processing Failed", state="error")
                            err_msg = job_status.get("error", "Unknown error")
                            st.error(f"**Error:** {err_msg}")
                            st.session_state["last_error"] = err_msg
                            break
                    else:
                        st.warning("⏱️ Timed out waiting for processing. Check server logs.")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to FastAPI server.\n\n"
                        "Run: `uvicorn main:app --reload --port 8000`"
                    )
                except requests.exceptions.HTTPError as exc:
                    st.error(f"❌ Upload failed (HTTP {exc.response.status_code}): {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    else:
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("📹 **Step 1**\nUpload 2+ video files from different camera angles")
        with col_b:
            st.info("⚙️ **Step 2**\nAdjust gallery size, weights, and topology in the sidebar")
        with col_c:
            st.info("🚀 **Step 3**\nClick Process Videos and wait for completion")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("### 🎬 Side-by-Side Processed Videos")

    all_outputs = []
    try:
        results_resp = requests.get(f"{api_url}/results/", timeout=10)
        results_resp.raise_for_status()
        all_outputs = results_resp.json().get("outputs", [])
    except Exception as exc:
        st.error(f"Failed to fetch results from backend: {exc}")

    if not all_outputs:
        st.info(
            "👈 Upload and process your videos in the **📤 Upload & Process** tab first.\n\n"
            "The annotated videos (with `L{local} / G{global}` overlay) will appear here."
        )
    else:
        unique_job_ids = sorted(list(set(r["job_id"] for r in all_outputs)), reverse=True)

        curr_job_id   = st.session_state.get("job_id")
        default_index = 0
        if curr_job_id in unique_job_ids:
            default_index = unique_job_ids.index(curr_job_id)

        selected_job_id = st.selectbox(
            "📁 Select Processed Job",
            options=unique_job_ids,
            index=default_index,
            format_func=lambda j: (
                f"Job ID: {j} "
                f"({len([r for r in all_outputs if r['job_id'] == j])} video(s))"
            ),
            key="results_job_selector",
        )

        job_outputs = [r for r in all_outputs if r["job_id"] == selected_job_id]

        if not job_outputs:
            st.warning("⚠️ No output videos found for this job.")
            if st.button("🔄 Refresh Results"):
                st.rerun()
        else:
            st.success(f"✅ {len(job_outputs)} processed video(s) ready for review.")

            with st.expander("💡 How to Read the Results (v2)", expanded=True):
                st.markdown("""
                - **Bounding boxes** show every detected person in each frame
                - Label format: **`L{local} / G{global}`**
                    - `L` = local ByteTrack ID (unique within this video only)
                    - `G` = global Re-ID person ID (consistent across ALL cameras)
                - Same **`G` number** = same person across cameras
                - Each ID has a unique color for easy visual tracking
                - **v2 improvement**: Global IDs are assigned using a full per-track
                  gallery of high-quality crops, making cross-camera matching more stable
                """)

            st.markdown("#### 📺 Camera Streams")
            num_videos = len(job_outputs)
            video_cols = st.columns(num_videos)

            for idx, result in enumerate(job_outputs):
                with video_cols[idx]:
                    st.markdown(
                        f'<div class="cam-label">📷 Camera {idx + 1} — '
                        f'<code>{result["filename"]}</code></div>',
                        unsafe_allow_html=True,
                    )
                    video_url = f"{api_url}{result['url']}"
                    try:
                        video_resp = requests.get(video_url, timeout=120, stream=True)
                        video_resp.raise_for_status()
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".mp4", dir=tempfile.gettempdir()
                        ) as tmp_file:
                            tmp_file.write(video_resp.content)
                            tmp_path = tmp_file.name
                        st.video(tmp_path)
                        st.download_button(
                            label="📥 Download Video",
                            data=video_resp.content,
                            file_name=result["filename"],
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"download_{result['job_id']}_{idx}",
                        )
                        st.caption(f"📁 {result['size_mb']} MB")
                    except requests.exceptions.HTTPError as exc:
                        st.error(f"HTTP {exc.response.status_code}: Could not load video.")
                    except Exception as exc:
                        st.error(f"Video load error: {exc}")

            # Per-video summary metrics (from session results)
            if (st.session_state.get("job_results")
                    and selected_job_id == st.session_state.get("job_id")):
                st.markdown("---")
                st.markdown("#### 📈 Processing Summary")
                metric_cols = st.columns(len(st.session_state["job_results"]))
                for idx, summary in enumerate(st.session_state["job_results"]):
                    with metric_cols[idx]:
                        source_name    = summary.get("video_source", f"Video {idx+1}")
                        n_frames       = summary.get("total_frames", 0)
                        unique_globals = summary.get("unique_global_ids", [])
                        n_local        = summary.get("local_track_count", 0)
                        proc_time      = summary.get("processing_time_s", 0)

                        st.metric(
                            label=f"📹 {source_name}",
                            value=f"{n_frames:,} frames",
                            delta=f"{len(unique_globals)} global IDs | {proc_time}s",
                        )
                        if unique_globals:
                            st.caption(f"Global IDs: {unique_globals}")
                        st.caption(f"Local tracks: {n_local}")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TRACKING DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Live Tracking Database")
    st.markdown("Queries the PostgreSQL `person_tracks` table and the live diagnostics endpoint.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True, type="primary")
    with ctrl_col2:
        record_limit = st.selectbox("Max records", [100, 200, 500, 1000], index=1)
    with ctrl_col3:
        st.empty()

    # ── Aggregate Stats ───────────────────────────────────────────
    try:
        stats_resp = requests.get(f"{api_url}/stats/", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🔍 Total Detections",  f"{stats['total_detections']:,}")
            s2.metric("👤 Unique Persons",     stats["unique_global_persons"])
            s3.metric("📹 Video Sources",      stats["unique_video_sources"])
            s4.metric("🎯 Avg Confidence",     f"{stats['average_confidence']:.1%}")

            # v2 additional stats
            if stats.get("average_quality_score") is not None:
                q1, q2, _, _ = st.columns(4)
                q1.metric("⭐ Avg Track Quality", f"{stats['average_quality_score']:.3f}")
                q2.metric("🔗 Avg Match Score",   f"{stats['average_match_score']:.3f}")
        else:
            st.warning("Could not load stats.")
    except Exception:
        st.warning("⚠️ Backend not reachable. Stats unavailable.")

    st.markdown("---")

    # ── [NEW v2] Diagnostics Panel ────────────────────────────────
    st.markdown("#### 🔬 Track-Level Diagnostics")
    st.caption("Live diagnostics from the tracking engine (per-video breakdown).")

    try:
        diag_resp = requests.get(f"{api_url}/diagnostics/", timeout=5)
        if diag_resp.status_code == 200:
            diag_data   = diag_resp.json()
            per_video   = diag_data.get("per_video", {})
            global_info = diag_data.get("global_state", {})

            if not per_video:
                st.info("📭 No diagnostics yet. Process some videos first!")
            else:
                diag_cols = st.columns(len(per_video))
                for col_idx, (vsrc, vdiag) in enumerate(per_video.items()):
                    with diag_cols[col_idx]:
                        st.markdown(
                            f'<div class="diag-card">'
                            f'<strong>📹 {vsrc}</strong><br>'
                            f'Local tracks: <code>{vdiag.get("local_track_count", "—")}</code><br>'
                            f'Global IDs: <code>{vdiag.get("global_id_count", "—")}</code><br>'
                            f'Avg match score: <code>{vdiag.get("avg_match_score", 0):.3f}</code><br>'
                            f'Unmatched tracks: <code>{vdiag.get("unmatched_tracks", "—")}</code>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Global state summary
                if global_info:
                    n_total  = global_info.get("total_global_ids", 0)
                    next_gid = global_info.get("next_global_id", 1)
                    st.info(
                        f"**Global Identity Manager:** {n_total} known persons · "
                        f"Next GID: {next_gid}"
                    )
        else:
            st.warning("Could not load diagnostics.")
    except Exception as exc:
        st.warning(f"⚠️ Diagnostics not available: {exc}")

    st.markdown("---")

    # ── Tracking Records Table ────────────────────────────────────
    try:
        tracks_resp = requests.get(
            f"{api_url}/tracks/",
            params={"limit": record_limit},
            timeout=10,
        )
        tracks_resp.raise_for_status()
        tracks_data = tracks_resp.json()

        if not tracks_data:
            st.info("📭 No tracking data yet. Process some videos first!")
        else:
            df = pd.DataFrame(tracks_data)

            st.markdown("#### 🔍 Filters")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                sources    = ["All"] + sorted(df["video_source"].unique().tolist())
                sel_source = st.selectbox("Video Source", sources, key="filter_source")
            with f_col2:
                person_ids = ["All"] + [str(i) for i in sorted(df["global_person_id"].unique().tolist())]
                sel_pid    = st.selectbox("Global Person ID", person_ids, key="filter_pid")
            with f_col3:
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, key="filter_conf")

            fdf = df.copy()
            if sel_source != "All":
                fdf = fdf[fdf["video_source"] == sel_source]
            if sel_pid != "All":
                fdf = fdf[fdf["global_person_id"] == int(sel_pid)]
            fdf = fdf[fdf["confidence"] >= min_conf]
            fdf = fdf.sort_values(
                by=["global_person_id", "video_source", "frame_number"],
                ascending=[True, True, True],
            )

            st.markdown(f"#### 📋 Track Records — `{len(fdf)}` rows shown")

            # Determine which v2 columns are available
            available_cols = ["global_person_id", "local_track_id", "video_source",
                              "frame_number", "confidence", "detected_at", "bbox_coordinates"]
            rename_map = {
                "global_person_id": "Global ID",
                "local_track_id":   "Local ID",
                "video_source":     "Video Source",
                "frame_number":     "Frame #",
                "confidence":       "Confidence",
                "detected_at":      "Detected At",
                "bbox_coordinates": "BBox [x1,y1,x2,y2]",
            }
            col_config = {
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Global ID": st.column_config.NumberColumn("Global ID", format="%d"),
                "Local ID":  st.column_config.NumberColumn("Local ID",  format="%d"),
                "Frame #":   st.column_config.NumberColumn("Frame #",   format="%d"),
            }

            # Conditionally show v2 columns if they exist and have data
            for v2col, label in [
                ("quality_score", "Quality"),
                ("match_score",   "Match Score"),
                ("track_start_frame", "Start Frame"),
                ("track_end_frame",   "End Frame"),
            ]:
                if v2col in fdf.columns and fdf[v2col].notna().any():
                    available_cols.append(v2col)
                    rename_map[v2col] = label
                    if "score" in v2col:
                        col_config[label] = st.column_config.ProgressColumn(
                            label, min_value=0.0, max_value=1.0, format="%.3f"
                        )

            display_df = fdf[available_cols].rename(columns=rename_map)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=col_config,
            )

            # ── Charts ────────────────────────────────────────────
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### 📊 Detections per Global Person ID")
                id_counts = (
                    fdf.groupby("global_person_id")
                    .size().reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.bar_chart(
                    id_counts.set_index("global_person_id")["count"],
                    x_label="Global Person ID", y_label="Detection Count",
                    color="#667eea",
                )

            with chart_col2:
                st.markdown("##### 📹 Detections per Video Source")
                source_counts = (
                    fdf.groupby("video_source")
                    .size().reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.bar_chart(
                    source_counts.set_index("video_source")["count"],
                    x_label="Video Source", y_label="Detection Count",
                    color="#a78bfa",
                )

            # ── Quality chart (v2) ────────────────────────────────
            if "quality_score" in fdf.columns and fdf["quality_score"].notna().any():
                st.markdown("---")
                st.markdown("##### ⭐ Track Quality Score Distribution")
                quality_by_lid = (
                    fdf.dropna(subset=["quality_score"])
                    .groupby(["local_track_id", "video_source"])["quality_score"]
                    .mean()
                    .reset_index()
                )
                quality_by_lid["label"] = (
                    quality_by_lid["video_source"].str[:8]
                    + " L"
                    + quality_by_lid["local_track_id"].astype(str)
                )
                st.bar_chart(
                    quality_by_lid.set_index("label")["quality_score"],
                    x_label="Track (source + local ID)", y_label="Quality Score",
                    color="#38bdf8",
                )

            # ── Danger Zone ────────────────────────────────────────
            st.markdown("---")
            with st.expander("🗑️ Danger Zone — Clear Database"):
                st.warning(
                    "This will permanently delete ALL tracking records from PostgreSQL "
                    "and reset the Re-ID cache."
                )
                if st.button("🗑️ Clear All Tracking Records", type="secondary"):
                    del_resp = requests.delete(f"{api_url}/tracks/", timeout=10)
                    if del_resp.status_code == 200:
                        result = del_resp.json()
                        st.success(f"✅ Deleted {result['deleted_rows']} records.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed to clear records: HTTP {del_resp.status_code}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to FastAPI backend.\n\n"
            "Start the server: `uvicorn main:app --reload --port 8000`"
        )
    except Exception as exc:
        st.error(f"Unexpected error loading dashboard: {exc}")
