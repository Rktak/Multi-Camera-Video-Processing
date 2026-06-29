"""
app.py - Streamlit Frontend
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Camera Multi-Object Tracking (MCMOT) System

How to run:
    streamlit run app.py

Make sure the FastAPI backend is running first:
    uvicorn main:app --reload --port 8000

What this file does:
    1. Lets you upload multiple video files
    2. Sends them to the FastAPI backend for YOLO + ByteTrack + Re-ID processing
    3. Polls the job status until processing is complete
    4. Shows the processed annotated videos side-by-side
    5. Displays a live tracking database dashboard
"""

import os
import tempfile
import time

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCMOT Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Multi-Camera Multi-Object Tracking System — Learning Project",
    },
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS — Modern dark UI styling
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Pipeline step badges */
    .pipeline-step {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #a5b4fc;
        margin: 3px;
        font-weight: 500;
    }
    .pipeline-arrow {
        color: #475569;
        font-size: 1rem;
        margin: 0 2px;
    }

    /* Status badges */
    .badge-done     { background:#064e3b; color:#6ee7b7; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-error    { background:#7f1d1d; color:#fca5a5; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-process  { background:#1e3a5f; color:#93c5fd; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
    .badge-pending  { background:#3b2d00; color:#fde68a; border-radius:12px; padding:3px 12px; font-weight:600; font-size:0.8rem; }

    /* Camera label above videos */
    .cam-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        padding: 6px 0 4px 0;
        border-bottom: 2px solid #667eea;
        margin-bottom: 8px;
    }

    /* Metric value highlight */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(100, 116, 139, 0.2);
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "http://localhost:8000"
MAX_POLL_SECONDS = 900  # 15 minutes max wait
POLL_INTERVAL_S = 3     # Check status every 3 seconds


def check_server(api_url: str) -> bool:
    """Ping the FastAPI health endpoint. Returns True if online."""
    try:
        r = requests.get(f"{api_url}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def status_badge(status: str) -> str:
    """Return an HTML status badge for a job status string."""
    badges = {
        "done":       '<span class="badge-done">✓ Done</span>',
        "error":      '<span class="badge-error">✗ Error</span>',
        "processing": '<span class="badge-process">⟳ Processing</span>',
        "pending":    '<span class="badge-pending">⏳ Pending</span>',
    }
    return badges.get(status, f"<code>{status}</code>")


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction.
# st.session_state persists values across re-runs.
for key, default in {
    "job_id": None,
    "processing_done": False,
    "job_results": [],
    "last_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 MCMOT System")
    st.markdown("---")

    # API URL configuration
    api_url = st.text_input(
        "FastAPI Server URL",
        value=DEFAULT_API_URL,
        help="Change this if you're running the backend on a different host/port.",
    )

    # Server status indicator
    server_ok = check_server(api_url)
    if server_ok:
        st.success("🟢 Backend: Online", icon="✅")
    else:
        st.error("🔴 Backend: Offline")
        st.code("uvicorn main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### ⚙️ Re-ID Settings")

    similarity_threshold = st.slider(
        "Cosine Similarity Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help=(
            "How similar two person appearances must be to be matched as the same "
            "global identity.\n\n"
            "**↑ Higher** (0.85+): Stricter — safer from false merges, may split one "
            "person into multiple IDs if their appearance changes.\n\n"
            "**↓ Lower** (0.55-): Lenient — better cross-camera recall, risk of "
            "incorrectly merging different people."
        ),
    )

    frame_skip = st.select_slider(
        "Frame Skip",
        options=[1, 2, 3, 5],
        value=1,
        help=(
            "Process every Nth frame.\n"
            "• 1 = every frame (slowest, most accurate)\n"
            "• 2 = every other frame (~2× faster)\n"
            "• 3 = every 3rd frame (~3× faster)"
        ),
    )

    st.markdown("---")
    st.markdown("### 📖 Pipeline")
    st.markdown("""
    <div style='line-height:2.2'>
    <span class='pipeline-step'>📹 Video Upload</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🔍 YOLOv8n Detection</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🏃 ByteTrack (Local IDs)</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🧬 MobileNetV3 Features</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🔗 Cosine Re-ID (Global IDs)</span><br>
    <span class='pipeline-arrow'>↓</span><br>
    <span class='pipeline-step'>🗄️ PostgreSQL Logging</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Learning Project — MCMOT v1.0")


# ─────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎯 Multi-Camera Multi-Object Tracker</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload multiple video streams · Track people with YOLOv8 + ByteTrack · '
    'Assign global identities across cameras with appearance Re-ID</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab_upload, tab_results, tab_dashboard = st.tabs([
    "📤 Upload & Process",
    "🎬 Results",
    "📊 Tracking Dashboard",
])


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
        # Show file preview table
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

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
        with btn_col1:
            process_btn = st.button(
                "🚀 Process Videos",
                type="primary",
                use_container_width=True,
                disabled=not server_ok,
            )
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["job_id"] = None
                st.session_state["processing_done"] = False
                st.session_state["job_results"] = []
                st.rerun()

        if not server_ok:
            st.warning("⚠️ FastAPI backend is offline. Start it before processing.")

        # ── Processing Logic ──────────────────────────────────────────
        if process_btn and server_ok:
            # Reset previous job state
            st.session_state["processing_done"] = False
            st.session_state["job_results"] = []
            st.session_state["last_error"] = None

            with st.status("🚀 Submitting to backend...", expanded=True) as status_box:

                # Build multipart upload payload
                files_payload = [
                    ("files", (f.name, f.getvalue(), "video/mp4"))
                    for f in uploaded_files
                ]

                st.write(f"⬆️ Uploading {len(files_payload)} file(s)...")
                try:
                    upload_resp = requests.post(
                        f"{api_url}/upload-videos/",
                        files=files_payload,
                        params={
                            "similarity_threshold": similarity_threshold,
                            "frame_skip": frame_skip,
                        },
                        timeout=120,
                    )
                    upload_resp.raise_for_status()
                    job_data = upload_resp.json()
                    job_id = job_data["job_id"]
                    st.session_state["job_id"] = job_id
                    st.write(f"✅ Job created: `{job_id}`")
                    st.write(
                        f"   Re-ID threshold: `{similarity_threshold}` | "
                        f"Frame skip: `{frame_skip}`"
                    )

                    # ── Polling Loop ──────────────────────────────────
                    st.write("⏳ Processing (polling every 3s)...")
                    progress_bar = st.progress(0.0, text="Waiting for tracker...")
                    elapsed = 0
                    estimated_pct = 0.0

                    while elapsed < MAX_POLL_SECONDS:
                        time.sleep(POLL_INTERVAL_S)
                        elapsed += POLL_INTERVAL_S

                        poll_resp = requests.get(
                            f"{api_url}/status/{job_id}", timeout=10
                        )
                        job_status = poll_resp.json()
                        current_status = job_status["status"]

                        # Slowly advance progress bar while unknown duration
                        estimated_pct = min(0.92, elapsed / MAX_POLL_SECONDS * 6)
                        progress_bar.progress(
                            estimated_pct,
                            text=f"Status: `{current_status}` — {elapsed}s elapsed"
                        )

                        if current_status == "done":
                            progress_bar.progress(1.0, text="✅ Complete!")
                            status_box.update(
                                label="✅ All Videos Processed!",
                                state="complete",
                                expanded=False,
                            )
                            st.session_state["processing_done"] = True
                            st.session_state["job_results"] = job_status.get("results", [])
                            st.success(
                                "🎉 Processing complete! Switch to the **🎬 Results** tab "
                                "to view the annotated videos."
                            )
                            break

                        elif current_status == "error":
                            progress_bar.progress(estimated_pct, text="❌ Error")
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
        # Empty state — show instructions
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("📹 **Step 1**\nUpload 2+ video files from different camera angles")
        with col_b:
            st.info("⚙️ **Step 2**\nAdjust Re-ID threshold in the sidebar if needed")
        with col_c:
            st.info("🚀 **Step 3**\nClick Process Videos and wait for completion")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("### 🎬 Side-by-Side Processed Videos")

    # Fetch output video list from backend first, regardless of session state
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
            "The annotated videos (with bounding boxes and global person IDs) will appear here."
        )
    else:
        # Group outputs by job_id
        unique_job_ids = sorted(list(set(r["job_id"] for r in all_outputs)), reverse=True)
        
        # Determine default job ID
        curr_job_id = st.session_state.get("job_id")
        default_index = 0
        if curr_job_id in unique_job_ids:
            default_index = unique_job_ids.index(curr_job_id)
            
        selected_job_id = st.selectbox(
            "📁 Select Processed Job",
            options=unique_job_ids,
            index=default_index,
            format_func=lambda j: f"Job ID: {j} (contains {len([r for r in all_outputs if r['job_id'] == j])} video(s))",
            key="results_job_selector"
        )
        
        # Filter outputs to the selected job
        job_outputs = [r for r in all_outputs if r["job_id"] == selected_job_id]

        if not job_outputs:
            st.warning(
                "⚠️ No output videos found for this job. "
                "Try refreshing."
            )
            if st.button("🔄 Refresh Results"):
                st.rerun()
        else:
            st.success(f"✅ {len(job_outputs)} processed video(s) ready for review.")

            # ── Key Explanation ───────────────────────────────────────
            with st.expander("💡 How to Read the Results", expanded=True):
                st.markdown("""
                - **Bounding boxes** show every detected person in each frame
                - The **`ID: N`** label is the **Global Person ID** — consistent across ALL cameras
                - Same number = same person (even if they moved between cameras)
                - Each ID has a unique color for easy visual tracking
                - **Verify Re-ID**: If the same person appears in both videos with the **same ID**, 
                  the cross-camera Re-ID worked correctly!
                """)

            st.markdown("#### 📺 Camera Streams")

            # ── Side-by-Side Video Display ────────────────────────────
            num_videos = len(job_outputs)
            video_cols = st.columns(num_videos)

            for idx, result in enumerate(job_outputs):
                with video_cols[idx]:
                    # Camera label header
                    st.markdown(
                        f'<div class="cam-label">📷 Camera {idx + 1} — '
                        f'<code>{result["filename"]}</code></div>',
                        unsafe_allow_html=True,
                    )

                    # Fetch and render the video
                    video_url = f"{api_url}{result['url']}"
                    try:
                        video_resp = requests.get(video_url, timeout=120, stream=True)
                        video_resp.raise_for_status()

                        # st.video() needs a file path or bytes object.
                        # We write to a temp file so it can be seeked properly.
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
                            key=f"download_{result['job_id']}_{idx}"
                        )
                        st.caption(f"📁 {result['size_mb']} MB")

                    except requests.exceptions.HTTPError as exc:
                        st.error(f"HTTP {exc.response.status_code}: Could not load video.")
                    except Exception as exc:
                        st.error(f"Video load error: {exc}")

            # ── Per-Video Summary Metrics ─────────────────────────────
            if st.session_state.get("job_results") and selected_job_id == st.session_state.get("job_id"):
                st.markdown("---")
                st.markdown("#### 📈 Processing Summary")

                metric_cols = st.columns(len(st.session_state["job_results"]))
                for idx, summary in enumerate(st.session_state["job_results"]):
                    with metric_cols[idx]:
                        source_name = summary.get("video_source", f"Video {idx + 1}")
                        n_frames = summary.get("total_frames", 0)
                        unique_ids = summary.get("unique_global_ids", [])
                        proc_time = summary.get("processing_time_s", 0)

                        st.metric(
                            label=f"📹 {source_name}",
                            value=f"{n_frames:,} frames",
                            delta=f"{len(unique_ids)} person IDs | {proc_time}s",
                        )
                        if unique_ids:
                            st.caption(f"Global IDs seen: {unique_ids}")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TRACKING DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Live Tracking Database")
    st.markdown("Directly queries the PostgreSQL `person_tracks` table via the backend API.")

    # ── Controls ──────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True, type="primary")
    with ctrl_col2:
        record_limit = st.selectbox("Max records", [100, 200, 500, 1000], index=1)
    with ctrl_col3:
        st.empty()  # Spacer

    # ── Aggregate Stats ───────────────────────────────────────────
    try:
        stats_resp = requests.get(f"{api_url}/stats/", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🔍 Total Detections", f"{stats['total_detections']:,}")
            s2.metric("👤 Unique Persons", stats["unique_global_persons"])
            s3.metric("📹 Video Sources", stats["unique_video_sources"])
            s4.metric("🎯 Avg Confidence", f"{stats['average_confidence']:.1%}")
        else:
            st.warning("Could not load stats.")
    except Exception:
        st.warning("⚠️ Backend not reachable. Stats unavailable.")

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

            # ── Filter Controls ───────────────────────────────────────
            st.markdown("#### 🔍 Filters")
            f_col1, f_col2, f_col3 = st.columns(3)

            with f_col1:
                sources = ["All"] + sorted(df["video_source"].unique().tolist())
                sel_source = st.selectbox("Video Source", sources, key="filter_source")

            with f_col2:
                person_ids = ["All"] + [
                    str(i) for i in sorted(df["global_person_id"].unique().tolist())
                ]
                sel_pid = st.selectbox("Global Person ID", person_ids, key="filter_pid")

            with f_col3:
                min_conf = st.slider(
                    "Min Confidence", 0.0, 1.0, 0.0, 0.05, key="filter_conf"
                )

            # Apply filters
            fdf = df.copy()
            if sel_source != "All":
                fdf = fdf[fdf["video_source"] == sel_source]
            if sel_pid != "All":
                fdf = fdf[fdf["global_person_id"] == int(sel_pid)]
            fdf = fdf[fdf["confidence"] >= min_conf]

            # Sort by Global ID, then Video Source, and then Frame # to group same person's track history together
            fdf = fdf.sort_values(by=["global_person_id", "video_source", "frame_number"], ascending=[True, True, True])

            st.markdown(f"#### 📋 Track Records — `{len(fdf)}` rows shown")

            # Rename columns for display
            display_df = fdf[[
                "global_person_id", "local_track_id", "video_source", "frame_number",
                "confidence", "detected_at", "bbox_coordinates"
            ]].rename(columns={
                "global_person_id": "Global ID",
                "local_track_id":  "Local ID",
                "video_source":     "Video Source",
                "frame_number":     "Frame #",
                "confidence":       "Confidence",
                "detected_at":      "Detected At",
                "bbox_coordinates": "BBox [x1,y1,x2,y2]",
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "Global ID": st.column_config.NumberColumn("Global ID", format="%d"),
                    "Local ID": st.column_config.NumberColumn("Local ID", format="%d"),
                    "Frame #": st.column_config.NumberColumn("Frame #", format="%d"),
                },
            )

            # ── Charts ─────────────────────────────────────────────────
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### 📊 Detections per Global Person ID")
                id_counts = (
                    fdf.groupby("global_person_id")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.bar_chart(
                    id_counts.set_index("global_person_id")["count"],
                    x_label="Global Person ID",
                    y_label="Detection Count",
                    color="#667eea",
                )

            with chart_col2:
                st.markdown("##### 📹 Detections per Video Source")
                source_counts = (
                    fdf.groupby("video_source")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.bar_chart(
                    source_counts.set_index("video_source")["count"],
                    x_label="Video Source",
                    y_label="Detection Count",
                    color="#a78bfa",
                )

            # ── Danger Zone ────────────────────────────────────────────
            st.markdown("---")
            with st.expander("🗑️ Danger Zone — Clear Database"):
                st.warning(
                    "This will permanently delete ALL tracking records from PostgreSQL. "
                    "Use this to reset between experiments."
                )
                if st.button("🗑️ Clear All Tracking Records", type="secondary"):
                    del_resp = requests.delete(f"{api_url}/tracks/", timeout=10)
                    if del_resp.status_code == 200:
                        result = del_resp.json()
                        st.success(
                            f"✅ Deleted {result['deleted_rows']} records from person_tracks."
                        )
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
