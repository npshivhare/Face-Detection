"""
Facial Recognition Attendance System — Streamlit GUI  (FIXED)
Requires: attendance_system.py in the same directory
Run     : streamlit run app.py
"""

import os, cv2, pickle, warnings, glob, time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import backend ────────────────────────────────────────────
try:
    from attendance_system import (
        get_embedding, get_haar, detect_faces, draw_box, hud,
        DIRS, STUDENTS_CSV, ENCODINGS_PKL, TMP_FACE,
        CAPTURE_FRAMES, THRESHOLD, CAMERA_INDEX,
    )
    from scipy.spatial.distance import cosine
    BACKEND_OK = True
except ImportError as _e:
    BACKEND_OK = False
    BACKEND_ERR = str(_e)

# ── Ensure dirs ───────────────────────────────────────────────
if BACKEND_OK:
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  CSS STYLING
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.main { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
[data-testid="metric-container"] {
    background: rgba(30, 42, 58, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(79, 195, 247, 0.2);
    border-radius: 16px; padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color: #90a4ae !important; font-size:0.85rem !important; font-weight:600 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#4fc3f7 !important; font-size:2rem !important; font-weight:700 !important; }
.section-title { font-size:2rem; font-weight:700; background:linear-gradient(135deg,#4fc3f7,#81d4fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px; }
.section-sub { font-size:1rem; color:#78909c; margin-bottom:24px; font-weight:500; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1a1f2e,#0f1117); border-right:1px solid rgba(79,195,247,0.1); }
[data-testid="stSidebar"] h2 { color:#4fc3f7; font-weight:700; }
.stButton > button { background:linear-gradient(135deg,#4fc3f7,#2196f3); color:white; border:none; border-radius:12px; padding:12px 24px; font-weight:600; box-shadow:0 4px 16px rgba(79,195,247,0.3); }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(79,195,247,0.4); }
.sidebar-footer { font-size:0.8rem; color:#546e7a; text-align:center; padding:16px 0; margin-top:24px; border-top:1px solid rgba(79,195,247,0.1); }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(79,195,247,0.3),transparent); margin:32px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def load_students():
    if BACKEND_OK and os.path.exists(STUDENTS_CSV):
        return pd.read_csv(STUDENTS_CSV)
    return pd.DataFrame(columns=["ID", "Name", "Registered"])

def load_encodings():
    if BACKEND_OK and os.path.exists(ENCODINGS_PKL):
        with open(ENCODINGS_PKL, "rb") as f:
            return pickle.load(f)
    return {}

def load_attendance_all():
    if not BACKEND_OK:
        return pd.DataFrame(columns=["ID", "Name", "Time", "Date", "Status"])
    files = glob.glob(os.path.join(DIRS["attendance"], "*.xlsx"))
    dfs = []
    for fpath in sorted(files):
        try:
            xl = pd.ExcelFile(fpath)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                if not df.empty:
                    dfs.append(df)
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["ID", "Name", "Time", "Date", "Status"])

def img_count(name, sid):
    p = os.path.join(DIRS["images"], f"{name}_{sid}")
    return len(glob.glob(os.path.join(p, "*.jpg"))) if os.path.exists(p) else 0

def attendance_files():
    return sorted(glob.glob(os.path.join(DIRS["attendance"], "*.xlsx")))

def open_camera_st():
    """Open camera and configure it."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        # Try index 1 if 0 fails
        cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Smart Attendance")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "👤 Register", "🧠 Train Model", "📋 Attendance", "📈 Reports"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    df_stu = load_students()
    enc_db = load_encodings()
    df_att = load_attendance_all()

    col1, col2 = st.columns(2)
    col1.metric("Students", len(df_stu))
    col2.metric("Records", len(df_att))
    model_status = f"✅ {len(enc_db)} enrolled" if enc_db else "❌ Not trained"
    st.metric("Model", model_status)
    st.markdown('<p class="sidebar-footer">Powered by DeepFace • Facenet</p>', unsafe_allow_html=True)

# ── Backend guard ─────────────────────────────────────────────
if not BACKEND_OK:
    st.error(f"❌ Could not import `attendance_system.py`. Make sure it is in the same folder.\n\nError: `{BACKEND_ERR}`")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time system overview and analytics</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Enrolled Students", len(df_stu))
    c2.metric("🧠 Model Status", "Trained" if enc_db else "Untrained")
    c3.metric("📋 Total Records", len(df_att))
    c4.metric("📁 Attendance Files", len(attendance_files()))

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("👥 Registered Students")
        if df_stu.empty:
            st.info("💡 No students registered yet.")
        else:
            display = df_stu.copy()
            display["Images"] = display.apply(lambda r: img_count(r["Name"], r["ID"]), axis=1)
            st.dataframe(display, use_container_width=True, hide_index=True, height=400)

    with col_right:
        st.subheader("📋 Recent Attendance")
        if df_att.empty:
            st.info("💡 No attendance records yet.")
        else:
            recent = df_att.sort_values("Date", ascending=False).head(15)
            st.dataframe(recent, use_container_width=True, hide_index=True, height=400)

# ══════════════════════════════════════════════════════════════
#  PAGE: REGISTER  ← MAIN FIX IS HERE
# ══════════════════════════════════════════════════════════════
elif page == "👤 Register":
    st.markdown('<p class="section-title">Register Student</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Capture facial data and enroll new students</p>', unsafe_allow_html=True)

    # ── Session state keys for registration ──────────────────
    for key, default in [
        ("reg_phase", "idle"),       # idle | preview | capturing | done
        ("reg_saved", 0),
        ("reg_cap", None),
        ("reg_haar", None),
        ("reg_student_dir", ""),
        ("reg_sid", ""),
        ("reg_name", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_form, col_info = st.columns([1, 1])

    with col_form:
        # Show form only when not actively capturing
        if st.session_state.reg_phase == "idle":
            sid  = st.text_input("🆔 Student ID (numeric)", placeholder="e.g. 1001")
            name = st.text_input("📝 Student Name", placeholder="e.g. John Doe")

            if st.button("📸 Open Camera & Preview", type="primary", use_container_width=True):
                if not sid.strip().isdigit():
                    st.error("❌ Student ID must be numeric.")
                elif not name.strip():
                    st.error("❌ Name cannot be empty.")
                else:
                    student_dir = os.path.join(DIRS["images"], f"{name.strip()}_{sid.strip()}")
                    os.makedirs(student_dir, exist_ok=True)

                    cap = open_camera_st()
                    if not cap.isOpened():
                        st.error("❌ Cannot open camera. Check your camera connection and try again.")
                    else:
                        # Warmup
                        for _ in range(30):
                            cap.read()
                        st.session_state.reg_cap         = cap
                        st.session_state.reg_haar        = get_haar()
                        st.session_state.reg_phase       = "preview"
                        st.session_state.reg_saved       = 0
                        st.session_state.reg_student_dir = student_dir
                        st.session_state.reg_sid         = sid.strip()
                        st.session_state.reg_name        = name.strip()
                        st.rerun()

        # ── Preview phase: show live feed, user clicks Start ──
        if st.session_state.reg_phase in ("preview", "capturing", "done"):
            cap  = st.session_state.reg_cap
            haar = st.session_state.reg_haar

            cam_placeholder = st.empty()
            status_text     = st.empty()
            progress_bar    = st.empty()

            btn_col1, btn_col2 = st.columns(2)

            if st.session_state.reg_phase == "preview":
                start_clicked = btn_col1.button("🟢 Start Capturing", type="primary", use_container_width=True)
                cancel_clicked = btn_col2.button("❌ Cancel", use_container_width=True)

                if cancel_clicked:
                    if cap:
                        cap.release()
                    st.session_state.reg_phase = "idle"
                    st.session_state.reg_cap   = None
                    st.rerun()

                if start_clicked:
                    st.session_state.reg_phase = "capturing"
                    st.session_state.reg_saved = 0
                    st.rerun()

                # Show live preview frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.equalizeHist(gray, gray)
                    faces = detect_faces(gray, haar)
                    for (x, y, w, h) in faces:
                        draw_box(frame, x, y, w, h, "Face Detected ✓", (0, 220, 220))
                    if not faces:
                        hud(frame, "No face detected — adjust position", color=(0, 100, 220))
                    else:
                        hud(frame, f"{len(faces)} face(s) detected — click Start Capturing")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_placeholder.image(rgb, channels="RGB", use_container_width=True)
                    status_text.info(f"👁️ Preview active — **{len(faces)} face(s)** detected. Click **Start Capturing** when ready.")

                # Auto-refresh preview
                time.sleep(0.05)
                st.rerun()

            elif st.session_state.reg_phase == "capturing":
                stop_clicked = btn_col1.button("⏹ Stop Early", use_container_width=True)
                saved = st.session_state.reg_saved
                student_dir = st.session_state.reg_student_dir

                # Capture one frame per Streamlit rerun
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.equalizeHist(gray, gray)
                    faces = detect_faces(gray, haar)

                    for (x, y, w, h) in faces:
                        draw_box(frame, x, y, w, h,
                                 f"Capturing {saved+1}/{CAPTURE_FRAMES}", (0, 255, 0))

                    # Save frame to disk
                    save_path = os.path.join(student_dir, f"{saved}.jpg")
                    cv2.imwrite(save_path, frame)
                    st.session_state.reg_saved += 1
                    saved = st.session_state.reg_saved

                    hud(frame, f"Capturing {saved}/{CAPTURE_FRAMES} — hold still!", color=(0, 200, 80))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_placeholder.image(rgb, channels="RGB", use_container_width=True)
                    progress_bar.progress(min(saved / CAPTURE_FRAMES, 1.0))
                    status_text.success(f"🟢 Capturing... **{saved}/{CAPTURE_FRAMES}** frames saved")

                if stop_clicked or st.session_state.reg_saved >= CAPTURE_FRAMES:
                    st.session_state.reg_phase = "done"
                    st.rerun()
                else:
                    time.sleep(0.05)
                    st.rerun()

            elif st.session_state.reg_phase == "done":
                # Release camera
                if cap:
                    cap.release()
                    st.session_state.reg_cap = None

                saved       = st.session_state.reg_saved
                sid         = st.session_state.reg_sid
                name        = st.session_state.reg_name
                student_dir = st.session_state.reg_student_dir

                cam_placeholder.empty()
                progress_bar.empty()

                if saved < 5:
                    st.error(f"⚠️ Only {saved} frames captured. Please try again.")
                else:
                    # Save to CSV
                    new_row = pd.DataFrame(
                        [[int(sid), name, datetime.now().strftime("%Y-%m-%d %H:%M")]],
                        columns=["ID", "Name", "Registered"],
                    )
                    if os.path.exists(STUDENTS_CSV):
                        df = pd.read_csv(STUDENTS_CSV)
                        df = df[df["ID"] != int(sid)]
                        df = pd.concat([df, new_row], ignore_index=True)
                    else:
                        df = new_row
                    df.to_csv(STUDENTS_CSV, index=False)

                    st.success(f"✅ **{name}** registered successfully! ({saved} frames saved)")
                    st.info(f"📁 Images saved to: `{student_dir}`")
                    st.info("➡️ Go to **🧠 Train Model** next to generate embeddings.")

                if st.button("🔄 Register Another Student", use_container_width=True):
                    st.session_state.reg_phase = "idle"
                    st.rerun()

    with col_info:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Registration Steps:**
        1. Enter student ID (must be numeric)
        2. Enter student name
        3. Click **Open Camera & Preview**
        4. Position face in the camera frame
        5. Click **Start Capturing** (no keyboard needed!)
        6. Hold still while frames are captured automatically
        7. Registration completes automatically at 30 frames

        **Tips for Best Results:**
        - Ensure good lighting
        - Face the camera directly
        - Stay still during capture
        - Remove glasses if possible
        """)

        if not df_stu.empty:
            st.markdown("### 👥 Recently Registered")
            recent_students = df_stu.sort_values("Registered", ascending=False).head(5)
            for _, row in recent_students.iterrows():
                st.markdown(f"**{row['Name']}** (ID: {row['ID']})")
                st.caption(f"Registered: {row['Registered']}")

# ══════════════════════════════════════════════════════════════
#  PAGE: TRAIN
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":
    st.markdown('<p class="section-title">Train Recognition Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Generate facial embeddings for enrolled students</p>', unsafe_allow_html=True)

    folders = []
    if os.path.exists(DIRS["images"]):
        folders = [
            d for d in os.listdir(DIRS["images"])
            if os.path.isdir(os.path.join(DIRS["images"], d))
        ]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Student Folders", len(folders))
    with col2:
        st.metric("🧠 Current Model", f"{len(enc_db)} students" if enc_db else "Not trained")

    if not folders:
        st.warning("⚠️ No student data found. Please register students first.")
        st.stop()

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("🔄 Training model... This may take a few minutes on first run (downloads ~90MB model)..."):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            log_area     = st.empty()
            logs = []

            encodings_db   = {}
            total_folders  = len(folders)

            for idx, folder in enumerate(folders):
                parts = folder.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    logs.append(f"⚠️ Skipping: {folder}")
                    continue

                name        = parts[0].replace("_", " ")
                sid         = int(parts[1])
                folder_path = os.path.join(DIRS["images"], folder)
                img_files   = sorted([
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith(".jpg")
                ])

                status_text.info(f"Processing: **{name}** (ID: {sid})")
                logs.append(f"👤 {name} (ID: {sid}) — {len(img_files)} images")

                enc_list = []
                for img_file in img_files:
                    img_path = os.path.join(folder_path, img_file)
                    bgr = cv2.imread(img_path)
                    if bgr is None:
                        logs.append(f"  ⚠️ Cannot read {img_file}")
                        continue
                    try:
                        enc = get_embedding(bgr)
                        if enc is not None:
                            enc_list.append(enc)
                            logs.append(f"  ✓ {img_file}")
                        else:
                            logs.append(f"  – No embedding: {img_file}")
                    except Exception as e:
                        logs.append(f"  ⚠️ {img_file}: {e}")

                if enc_list:
                    encodings_db[sid] = {"name": name, "encodings": enc_list}
                    logs.append(f"  ✅ {len(enc_list)} embeddings saved\n")
                else:
                    logs.append(f"  ❌ No embeddings for {name}\n")

                progress_bar.progress((idx + 1) / total_folders)
                log_area.text_area("Training Log", "\n".join(logs[-20:]), height=300)

            if not encodings_db:
                st.error("❌ No encodings created. Training failed.")
            else:
                with open(ENCODINGS_PKL, "wb") as f:
                    pickle.dump(encodings_db, f)

                total_embeddings = sum(len(v["encodings"]) for v in encodings_db.values())
                progress_bar.empty()
                status_text.empty()

                st.success("✅ **Training Complete!**")
                col_a, col_b = st.columns(2)
                col_a.metric("📊 Students Trained", len(encodings_db))
                col_b.metric("🎯 Total Embeddings", total_embeddings)
                st.balloons()

    st.markdown("---")
    st.markdown("""
    **What happens during training?**
    Face detection on registered images → Facenet embeddings (128-D) → saved for live recognition.
    Retrain after registering new students. First run downloads Facenet model (~90 MB).
    """)

# ══════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE  ← ALSO FIXED (no cv2.imshow/waitKey)
# ══════════════════════════════════════════════════════════════
elif page == "📋 Attendance":
    st.markdown('<p class="section-title">Mark Attendance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time facial recognition for attendance tracking</p>', unsafe_allow_html=True)

    if not os.path.exists(ENCODINGS_PKL):
        st.error("❌ No trained model found. Please train the model first.")
        st.stop()

    # Session state
    for key, default in [
        ("att_running", False),
        ("att_log", []),
        ("att_marked", set()),
        ("att_cap", None),
        ("att_frame_count", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_cam, col_log = st.columns([3, 2])

    with col_cam:
        cam_placeholder  = st.empty()
        info_placeholder = st.empty()

    with col_log:
        st.subheader("✅ Attendance Log")
        log_placeholder   = st.empty()
        count_placeholder = st.empty()

    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    start_btn = btn_col1.button("▶ Start Recognition", type="primary", use_container_width=True)
    stop_btn  = btn_col2.button("⏹ Stop & Save",       use_container_width=True)

    if start_btn and not st.session_state.att_running:
        cap = open_camera_st()
        if not cap.isOpened():
            st.error("❌ Cannot open camera.")
        else:
            for _ in range(30): cap.read()   # warmup
            st.session_state.att_cap         = cap
            st.session_state.att_running     = True
            st.session_state.att_log         = []
            st.session_state.att_marked      = set()
            st.session_state.att_frame_count = 0
            st.rerun()

    if stop_btn and st.session_state.att_running:
        st.session_state.att_running = False
        if st.session_state.att_cap:
            st.session_state.att_cap.release()
            st.session_state.att_cap = None
        st.rerun()

    if st.session_state.att_running:
        cap  = st.session_state.att_cap
        haar = get_haar()
        enc_db_att   = load_encodings()
        session_date = datetime.now().strftime("%Y-%m-%d")
        EVERY        = 5  # run recognition every N frames

        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠️ Camera read failed. Try stopping and restarting.")
        else:
            frame = cv2.flip(frame, 1)
            st.session_state.att_frame_count += 1
            frame_count = st.session_state.att_frame_count

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            faces = detect_faces(gray, haar)

            for (x, y, w, h) in faces:
                if frame_count % EVERY != 0:
                    draw_box(frame, x, y, w, h, "scanning…", (120, 120, 120))
                    continue

                fc = frame[y:y+h, x:x+w]
                if fc.size == 0:
                    continue

                try:
                    enc = get_embedding(fc)
                    if enc is None:
                        draw_box(frame, x, y, w, h, "?", (80, 80, 200))
                        continue

                    best_id, best_dist, best_name = None, 1.0, "Unknown"
                    for s_id, data in enc_db_att.items():
                        dists = [cosine(enc, e) for e in data["encodings"]]
                        avg   = float(np.mean(sorted(dists)[:5]))
                        if avg < best_dist:
                            best_dist = avg
                            best_id   = s_id
                            best_name = data["name"]

                    if best_dist < THRESHOLD:
                        already = best_id in st.session_state.att_marked
                        color   = (0, 180, 60) if not already else (200, 140, 0)
                        draw_box(frame, x, y, w, h,
                                 f"{best_name} {'(marked)' if already else 'PRESENT'}", color)
                        if not already:
                            now = datetime.now().strftime("%H:%M:%S")
                            st.session_state.att_marked.add(best_id)
                            st.session_state.att_log.append(
                                [best_id, best_name, now, session_date, "Present"]
                            )
                    else:
                        draw_box(frame, x, y, w, h,
                                 f"Unknown ({best_dist:.2f})", (60, 60, 200))

                except Exception:
                    draw_box(frame, x, y, w, h, "error", (0, 0, 180))

            hud(frame, f"Date:{session_date}  Marked:{len(st.session_state.att_marked)}  Click Stop to save")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, channels="RGB", use_container_width=True)
            info_placeholder.caption(f"🗓 {session_date}  |  ✅ {len(st.session_state.att_marked)} marked")

            if st.session_state.att_log:
                df_live = pd.DataFrame(
                    st.session_state.att_log,
                    columns=["ID", "Name", "Time", "Date", "Status"]
                )
                log_placeholder.dataframe(df_live[["Name", "Time"]], use_container_width=True, hide_index=True, height=400)
                count_placeholder.success(f"**{len(st.session_state.att_marked)} student(s) marked present**")

        time.sleep(0.05)
        st.rerun()

    elif not st.session_state.att_running and st.session_state.att_log:
        # Save on stop
        session_date = datetime.now().strftime("%Y-%m-%d")
        excel_path   = os.path.join(DIRS["attendance"], f"Attendance_{session_date}.xlsx")
        df_save = pd.DataFrame(
            st.session_state.att_log,
            columns=["ID", "Name", "Time", "Date", "Status"]
        )
        mode = "a" if os.path.exists(excel_path) else "w"
        kw   = {"if_sheet_exists": "replace"} if mode == "a" else {}
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode, **kw) as writer:
            df_save.to_excel(writer, sheet_name=session_date, index=False)

        st.success(f"✅ Attendance saved → `{excel_path}`")
        st.dataframe(df_save, use_container_width=True, hide_index=True)

        if os.path.exists(TMP_FACE):
            try:
                os.remove(TMP_FACE)
            except Exception:
                pass

        # Clear log so it doesn't re-save on next rerun
        st.session_state.att_log    = []
        st.session_state.att_marked = set()

# ══════════════════════════════════════════════════════════════
#  PAGE: REPORTS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Reports":
    st.markdown('<p class="section-title">Attendance Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Analytics, insights, and downloadable records</p>', unsafe_allow_html=True)

    att_files = attendance_files()

    if not att_files:
        st.info("📂 No attendance files found yet. Mark attendance first.")
    else:
        file_names    = [os.path.basename(f) for f in att_files]
        selected_file = st.selectbox("📁 Select attendance file", file_names, index=len(file_names)-1)
        sel_path      = os.path.join(DIRS["attendance"], selected_file)

        try:
            xl     = pd.ExcelFile(sel_path)
            sheets = xl.sheet_names

            if len(sheets) > 1:
                sel_sheet = st.selectbox("📄 Select date", sheets, index=len(sheets)-1)
            else:
                sel_sheet = sheets[0]

            df_report = xl.parse(sel_sheet)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("📋 Present Today", len(df_report))
            if os.path.exists(STUDENTS_CSV):
                total_reg      = len(pd.read_csv(STUDENTS_CSV))
                attendance_pct = len(df_report) / max(total_reg, 1) * 100
                mc2.metric("👥 Total Enrolled", total_reg)
                mc3.metric("📊 Attendance Rate", f"{attendance_pct:.1f}%")

            st.markdown("---")

            search = st.text_input("🔍 Search by name or ID", placeholder="Type to filter…")
            if search:
                mask = (
                    df_report["Name"].astype(str).str.contains(search, case=False, na=False) |
                    df_report["ID"].astype(str).str.contains(search, case=False, na=False)
                )
                df_report = df_report[mask]

            st.dataframe(df_report, use_container_width=True, hide_index=True, height=400)

            csv_data = df_report.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_data,
                file_name=f"{selected_file.replace('.xlsx','')}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    st.markdown("---")
    st.subheader("📊 Historical Analytics")
    df_all = load_attendance_all()
    if df_all.empty:
        st.info("No historical records available.")
    else:
        by_date = df_all.groupby("Date").size().reset_index(name="Count")
        st.bar_chart(by_date.set_index("Date")["Count"])

        st.subheader("🏆 Top Attendees")
        top_students = (
            df_all.groupby("Name").size()
            .reset_index(name="Days Present")
            .sort_values("Days Present", ascending=False)
            .head(10)
        )
        st.dataframe(top_students, use_container_width=True, hide_index=True)
