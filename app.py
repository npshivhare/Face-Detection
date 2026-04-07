"""
Facial Recognition Attendance System — Streamlit GUI (Streamlit Cloud / webrtc version)
Requires: attendance_system.py in the same directory
Run     : streamlit run app.py
"""

import os, cv2, pickle, warnings, glob, time, threading
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Page config  (MUST be first st call) ─────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import webrtc ─────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False


from streamlit_webrtc import webrtc_streamer, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="camera",
    rtc_configuration=RTC_CONFIGURATION
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

# ── RTC config (STUN server — required on Streamlit Cloud) ────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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

# ══════════════════════════════════════════════════════════════
#  VIDEO PROCESSORS (streamlit-webrtc)
# ══════════════════════════════════════════════════════════════

class RegisterProcessor(VideoProcessorBase):
    """Captures face frames for student registration."""
    def __init__(self):
        self.lock         = threading.Lock()
        self.capturing    = False
        self.saved        = 0
        self.student_dir  = ""
        self.target       = CAPTURE_FRAMES if BACKEND_OK else 30
        self._haar        = get_haar() if BACKEND_OK else None

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = detect_faces(gray, self._haar) if BACKEND_OK else []

        with self.lock:
            capturing   = self.capturing
            saved       = self.saved
            student_dir = self.student_dir
            target      = self.target

        if capturing and student_dir and faces is not None and len(faces) > 0 and saved < target:
            x, y, w, h = faces[0]
            face_crop   = img[y:y+h, x:x+w]
            path        = os.path.join(student_dir, f"{saved}.jpg")
            cv2.imwrite(path, face_crop)
            with self.lock:
                self.saved += 1
                saved = self.saved

        # Draw overlay
        for (x, y, w, h) in (faces if faces is not None else []):
            if capturing:
                label = f"Saving {saved}/{target}"
                color = (0, 255, 0)
            else:
                label = "Face Detected ✓"
                color = (0, 220, 220)
            if BACKEND_OK:
                draw_box(img, x, y, w, h, label, color)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = f"Captured {saved}/{target}" if capturing else "Preview — position your face"
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class AttendanceProcessor(VideoProcessorBase):
    """Runs live face recognition for attendance marking."""
    def __init__(self):
        self.lock        = threading.Lock()
        self.enc_db      = load_encodings()
        self.marked      = set()   # set of student IDs already marked
        self.log         = []      # list of [id, name, time, date, status]
        self._haar       = get_haar() if BACKEND_OK else None
        self._frame_no   = 0
        self._threshold  = THRESHOLD if BACKEND_OK else 0.4

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self._frame_no += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = detect_faces(gray, self._haar) if BACKEND_OK else []

        for (x, y, w, h) in (faces if faces is not None else []):
            # Only run heavy recognition every 5 frames
            if self._frame_no % 5 != 0:
                if BACKEND_OK:
                    draw_box(img, x, y, w, h, "scanning…", (120, 120, 120))
                continue

            fc = img[y:y+h, x:x+w]
            if fc.size == 0:
                continue

            try:
                enc = get_embedding(fc) if BACKEND_OK else None
                if enc is None:
                    if BACKEND_OK:
                        draw_box(img, x, y, w, h, "?", (80, 80, 200))
                    continue

                best_id, best_dist, best_name = None, 1.0, "Unknown"
                for s_id, data in self.enc_db.items():
                    dists = [cosine(enc, e) for e in data["encodings"]]
                    avg   = float(np.mean(sorted(dists)[:5]))
                    if avg < best_dist:
                        best_dist = avg
                        best_id   = s_id
                        best_name = data["name"]

                with self.lock:
                    already = best_id in self.marked

                if best_dist < self._threshold:
                    color = (0, 180, 60) if not already else (200, 140, 0)
                    label = f"{best_name} {'(marked)' if already else 'PRESENT'}"
                    if BACKEND_OK:
                        draw_box(img, x, y, w, h, label, color)
                    if not already:
                        now = datetime.now().strftime("%H:%M:%S")
                        date = datetime.now().strftime("%Y-%m-%d")
                        with self.lock:
                            self.marked.add(best_id)
                            self.log.append([best_id, best_name, now, date, "Present"])
                else:
                    if BACKEND_OK:
                        draw_box(img, x, y, w, h, f"Unknown ({best_dist:.2f})", (60, 60, 200))

            except Exception:
                if BACKEND_OK:
                    draw_box(img, x, y, w, h, "error", (0, 0, 180))

        with self.lock:
            count = len(self.marked)
        date_str = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(img, f"{date_str}  Marked: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
    df_stu  = load_students()
    enc_db  = load_encodings()
    df_att  = load_attendance_all()

    col1, col2 = st.columns(2)
    col1.metric("Students", len(df_stu))
    col2.metric("Records",  len(df_att))
    model_status = f"✅ {len(enc_db)} enrolled" if enc_db else "❌ Not trained"
    st.metric("Model", model_status)
    st.markdown('<p class="sidebar-footer">Powered by DeepFace • Facenet</p>', unsafe_allow_html=True)

# ── Guards ────────────────────────────────────────────────────
if not WEBRTC_OK:
    st.error("❌ `streamlit-webrtc` not installed. Add it to requirements.txt")
    st.stop()

if not BACKEND_OK:
    st.error(f"❌ Could not import `attendance_system.py`.\n\nError: `{BACKEND_ERR}`")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time system overview and analytics</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Enrolled Students", len(df_stu))
    c2.metric("🧠 Model Status",      "Trained" if enc_db else "Untrained")
    c3.metric("📋 Total Records",     len(df_att))
    c4.metric("📁 Attendance Files",  len(attendance_files()))

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
#  PAGE: REGISTER
# ══════════════════════════════════════════════════════════════
elif page == "👤 Register":
    st.markdown('<p class="section-title">Register Student</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Capture facial data and enroll new students</p>', unsafe_allow_html=True)

    col_form, col_info = st.columns([1, 1])

    with col_form:
        sid  = st.text_input("🆔 Student ID (numeric)", placeholder="e.g. 1001")
        name = st.text_input("📝 Student Name",         placeholder="e.g. John Doe")

        if sid and name:
            student_dir = os.path.join(DIRS["images"], f"{name.strip()}_{sid.strip()}")
            os.makedirs(student_dir, exist_ok=True)
        else:
            student_dir = ""

        st.markdown("---")
        st.markdown("**Step 1 — Allow camera access, then position your face**")
        st.markdown("**Step 2 — Click *Start Capturing* below**")

        ctx = webrtc_streamer(
            key="register",
            video_processor_factory=RegisterProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Wire student_dir into the processor as soon as it exists
        if ctx.video_processor and student_dir:
            ctx.video_processor.student_dir = student_dir

        # Controls
        col_a, col_b = st.columns(2)
        start_btn  = col_a.button("🟢 Start Capturing", type="primary", use_container_width=True)
        stop_btn   = col_b.button("⏹ Stop",              use_container_width=True)

        if start_btn:
            if not sid.strip().isdigit():
                st.error("❌ Student ID must be numeric.")
            elif not name.strip():
                st.error("❌ Name cannot be empty.")
            elif ctx.video_processor:
                ctx.video_processor.capturing = True
                ctx.video_processor.saved     = 0
                st.info("🟢 Capturing started — hold still!")
            else:
                st.warning("⚠️ Camera not ready yet. Wait for the stream to start.")

        if stop_btn and ctx.video_processor:
            ctx.video_processor.capturing = False

        # Live progress
        if ctx.video_processor:
            saved  = ctx.video_processor.saved
            target = ctx.video_processor.target
            st.progress(min(saved / target, 1.0))
            st.caption(f"Frames captured: {saved} / {target}")

            if saved >= target and student_dir:
                # Auto-save student to CSV
                new_row = pd.DataFrame(
                    [[int(sid.strip()), name.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")]],
                    columns=["ID", "Name", "Registered"],
                )
                if os.path.exists(STUDENTS_CSV):
                    df_existing = pd.read_csv(STUDENTS_CSV)
                    df_existing = df_existing[df_existing["ID"] != int(sid.strip())]
                    df_existing = pd.concat([df_existing, new_row], ignore_index=True)
                else:
                    df_existing = new_row
                df_existing.to_csv(STUDENTS_CSV, index=False)

                st.success(f"✅ **{name.strip()}** registered! ({saved} frames saved)")
                st.info("➡️ Go to **🧠 Train Model** to generate embeddings.")
                ctx.video_processor.capturing = False

    with col_info:
        st.markdown("### 📋 Instructions")
        st.markdown("""
**Registration Steps:**
1. Enter Student ID (numeric) and Name
2. Click **Start** on the camera widget (allow browser camera access)
3. Click **Start Capturing** once your face is visible
4. Hold still — frames are captured automatically
5. Registration saves when target frames are reached

**Tips for Best Results:**
- Ensure good lighting
- Face the camera directly
- Stay still during capture
- Remove glasses if possible
        """)
        if not df_stu.empty:
            st.markdown("### 👥 Recently Registered")
            for _, row in df_stu.sort_values("Registered", ascending=False).head(5).iterrows():
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
        with st.spinner("🔄 Training model... First run downloads ~90 MB Facenet model..."):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            log_area     = st.empty()
            logs = []

            encodings_db  = {}
            total_folders = len(folders)

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
                col_a.metric("📊 Students Trained",  len(encodings_db))
                col_b.metric("🎯 Total Embeddings",  total_embeddings)
                st.balloons()

    st.markdown("---")
    st.markdown("""
**What happens during training?**
Face detection on registered images → Facenet embeddings (128-D) → saved for live recognition.
Retrain after registering new students. First run downloads Facenet model (~90 MB).
    """)


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE
# ══════════════════════════════════════════════════════════════
elif page == "📋 Attendance":
    st.markdown('<p class="section-title">Mark Attendance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time facial recognition for attendance tracking</p>', unsafe_allow_html=True)

    if not os.path.exists(ENCODINGS_PKL):
        st.error("❌ No trained model found. Please train the model first.")
        st.stop()

    col_cam, col_log = st.columns([3, 2])

    with col_cam:
        st.markdown("**Allow camera access and click Start ▶**")
        ctx = webrtc_streamer(
            key="attendance",
            video_processor_factory=AttendanceProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_log:
        st.subheader("✅ Attendance Log")
        log_placeholder   = st.empty()
        count_placeholder = st.empty()
        save_placeholder  = st.empty()

    # Poll the processor state every refresh
    if ctx.video_processor:
        with ctx.video_processor.lock:
            log_snapshot    = list(ctx.video_processor.log)
            marked_count    = len(ctx.video_processor.marked)

        if log_snapshot:
            df_live = pd.DataFrame(log_snapshot, columns=["ID", "Name", "Time", "Date", "Status"])
            log_placeholder.dataframe(
                df_live[["Name", "Time"]], use_container_width=True, hide_index=True, height=400
            )
            count_placeholder.success(f"**{marked_count} student(s) marked present**")

            # Save button
            if save_placeholder.button("💾 Save Attendance to Excel", type="primary"):
                session_date = datetime.now().strftime("%Y-%m-%d")
                excel_path   = os.path.join(DIRS["attendance"], f"Attendance_{session_date}.xlsx")
                df_save = pd.DataFrame(log_snapshot, columns=["ID", "Name", "Time", "Date", "Status"])
                mode = "a" if os.path.exists(excel_path) else "w"
                kw   = {"if_sheet_exists": "replace"} if mode == "a" else {}
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode, **kw) as writer:
                    df_save.to_excel(writer, sheet_name=session_date, index=False)
                st.success(f"✅ Saved → `{excel_path}`")

                # Also offer CSV download (works even on ephemeral cloud storage)
                csv_bytes = df_save.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"Attendance_{session_date}.csv",
                    mime="text/csv",
                )
        else:
            log_placeholder.info("No attendance recorded yet. Start the camera and face the lens.")
    else:
        st.info("👆 Click **START** on the camera widget above to begin recognition.")


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

            sel_sheet = st.selectbox("📄 Select date", sheets, index=len(sheets)-1) if len(sheets) > 1 else sheets[0]
            df_report = xl.parse(sel_sheet)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("📋 Present Today", len(df_report))
            if os.path.exists(STUDENTS_CSV):
                total_reg      = len(pd.read_csv(STUDENTS_CSV))
                attendance_pct = len(df_report) / max(total_reg, 1) * 100
                mc2.metric("👥 Total Enrolled",   total_reg)
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