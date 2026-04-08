"""
Facial Recognition Attendance System — Streamlit Cloud Edition
Uses st.camera_input() — works on any device, no WebRTC/TURN servers needed.
Requires: attendance_system.py in the same directory
Run     : streamlit run app.py
"""

import os, cv2, pickle, warnings, glob
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Page config (MUST be first st call) ──────────────────────
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
        CAPTURE_FRAMES, THRESHOLD,
    )
    from scipy.spatial.distance import cosine
    BACKEND_OK = True
except ImportError as _e:
    BACKEND_OK = False
    BACKEND_ERR = str(_e)

if BACKEND_OK:
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="metric-container"] {
    background: rgba(30,42,58,0.6);
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 16px; padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color:#90a4ae !important; font-size:0.85rem !important; font-weight:600 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#4fc3f7 !important; font-size:2rem !important; font-weight:700 !important; }
.section-title { font-size:2rem; font-weight:700; background:linear-gradient(135deg,#4fc3f7,#81d4fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px; }
.section-sub { font-size:1rem; color:#78909c; margin-bottom:24px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1a1f2e,#0f1117); border-right:1px solid rgba(79,195,247,0.1); }
.stButton > button { background:linear-gradient(135deg,#4fc3f7,#2196f3); color:white; border:none; border-radius:12px; padding:12px 24px; font-weight:600; box-shadow:0 4px 16px rgba(79,195,247,0.3); }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(79,195,247,0.4); }
.sidebar-footer { font-size:0.8rem; color:#546e7a; text-align:center; padding:16px 0; margin-top:24px; border-top:1px solid rgba(79,195,247,0.1); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HELPERS
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
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["ID","Name","Time","Date","Status"])

def img_count(name, sid):
    p = os.path.join(DIRS["images"], f"{name}_{sid}")
    return len(glob.glob(os.path.join(p, "*.jpg"))) if os.path.exists(p) else 0

def attendance_files():
    return sorted(glob.glob(os.path.join(DIRS["attendance"], "*.xlsx")))

def pil_to_bgr(pil_img):
    """Convert PIL image from camera_input to OpenCV BGR array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def detect_and_draw(bgr, haar, label="", color=(0, 220, 220)):
    """Run face detection, draw boxes, return (frame_rgb, faces)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    faces = detect_faces(gray, haar)
    out   = bgr.copy()
    for (x, y, w, h) in (faces if faces is not None else []):
        draw_box(out, x, y, w, h, label, color)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, faces if faces is not None else []

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
    c1, c2 = st.columns(2)
    c1.metric("Students", len(df_stu))
    c2.metric("Records",  len(df_att))
    st.metric("Model", f"✅ {len(enc_db)} enrolled" if enc_db else "❌ Not trained")
    st.markdown('<p class="sidebar-footer">Powered by DeepFace • Facenet</p>', unsafe_allow_html=True)

if not BACKEND_OK:
    st.error(f"❌ Could not import `attendance_system.py`.\n\nError: `{BACKEND_ERR}`")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">System overview and analytics</p>', unsafe_allow_html=True)

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
            st.dataframe(df_att.sort_values("Date", ascending=False).head(15),
                         use_container_width=True, hide_index=True, height=400)


# ══════════════════════════════════════════════════════════════
#  PAGE: REGISTER  —  uses st.camera_input()
# ══════════════════════════════════════════════════════════════
elif page == "👤 Register":
    st.markdown('<p class="section-title">Register Student</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Capture facial data and enroll new students</p>', unsafe_allow_html=True)

    # Request rear camera early so browser warms up the stream
    st.markdown("""
    <script>
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
            .catch(function(e) { console.warn("Camera pre-warm failed:", e); });
    }
    </script>
    """, unsafe_allow_html=True)

    # Session state
    for k, v in [("reg_saved_frames", []), ("reg_done", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    col_form, col_info = st.columns([1, 1])

    with col_form:
        sid  = st.text_input("🆔 Student ID (numeric)", placeholder="e.g. 1001")
        name = st.text_input("📝 Student Name",         placeholder="e.g. John Doe")

        st.markdown("---")
        st.markdown("#### 📸 Camera Capture")
        st.info("👇 Click **Take Photo** — browser asks for camera permission on **this device** only.")

        photo = st.camera_input(
            label="Take a photo",
            key="reg_camera",
            help="Click to capture one frame. Repeat to collect multiple samples.",
        )

        haar = get_haar()

        if photo is not None:
            if not sid.strip().isdigit():
                st.error("❌ Student ID must be numeric.")
            elif not name.strip():
                st.error("❌ Name cannot be empty.")
            else:
                pil_img = Image.open(io.BytesIO(photo.getvalue()))
                bgr     = pil_to_bgr(pil_img)

                rgb_out, faces = detect_and_draw(bgr, haar, "Face Detected ✓", (0, 220, 220))
                st.image(rgb_out, channels="RGB", use_container_width=True)

                if len(faces) == 0:
                    st.warning("⚠️ No face detected — try better lighting or move closer.")
                else:
                    x, y, w, h = faces[0]
                    face_crop  = bgr[y:y+h, x:x+w]
                    st.session_state.reg_saved_frames.append(face_crop)
                    st.success(f"✅ Frame {len(st.session_state.reg_saved_frames)}/{CAPTURE_FRAMES} captured!")

        saved_count = len(st.session_state.reg_saved_frames)
        st.progress(min(saved_count / CAPTURE_FRAMES, 1.0))
        st.caption(f"Frames captured: {saved_count} / {CAPTURE_FRAMES}  (min 5 to save)")

        col_a, col_b = st.columns(2)

        if col_b.button("🗑 Clear & Restart", use_container_width=True):
            st.session_state.reg_saved_frames = []
            st.session_state.reg_done         = False
            st.rerun()

        save_ready = saved_count >= max(5, CAPTURE_FRAMES // 3)
        if col_a.button("💾 Save Registration", type="primary",
                        use_container_width=True, disabled=not save_ready):
            if not sid.strip().isdigit():
                st.error("❌ Student ID must be numeric.")
            elif not name.strip():
                st.error("❌ Name cannot be empty.")
            else:
                student_dir = os.path.join(DIRS["images"], f"{name.strip()}_{sid.strip()}")
                os.makedirs(student_dir, exist_ok=True)

                for i, face_bgr in enumerate(st.session_state.reg_saved_frames):
                    cv2.imwrite(os.path.join(student_dir, f"{i}.jpg"), face_bgr)

                new_row = pd.DataFrame(
                    [[int(sid.strip()), name.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")]],
                    columns=["ID", "Name", "Registered"],
                )
                if os.path.exists(STUDENTS_CSV):
                    df_ex = pd.read_csv(STUDENTS_CSV)
                    df_ex = df_ex[df_ex["ID"] != int(sid.strip())]
                    df_ex = pd.concat([df_ex, new_row], ignore_index=True)
                else:
                    df_ex = new_row
                df_ex.to_csv(STUDENTS_CSV, index=False)

                st.success(f"✅ **{name.strip()}** registered with {saved_count} frames!")
                st.info("➡️ Go to **🧠 Train Model** to generate embeddings.")
                st.session_state.reg_saved_frames = []
                st.session_state.reg_done         = True

    with col_info:
        st.markdown("### 📋 How it works")
        st.markdown("""
**Steps:**
1. Enter Student ID and Name
2. Click **Take Photo** — browser asks for camera on **this device**
3. Keep clicking to capture multiple frames (aim for 10–30)
4. Good frames show a teal detection box ✓
5. Click **Save Registration** when done (min 5 frames)

**Tips:**
- Each click = 1 frame saved
- Slightly change head angle between shots for variety
- Good lighting = better accuracy later
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
        folders = [d for d in os.listdir(DIRS["images"])
                   if os.path.isdir(os.path.join(DIRS["images"], d))]

    col1, col2 = st.columns(2)
    col1.metric("📁 Student Folders", len(folders))
    col2.metric("🧠 Current Model", f"{len(enc_db)} students" if enc_db else "Not trained")

    if not folders:
        st.warning("⚠️ No student data found. Please register students first.")
        st.stop()

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("🔄 Training... First run downloads ~90 MB Facenet model..."):
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            log_area      = st.empty()
            logs          = []
            encodings_db  = {}

            for idx, folder in enumerate(folders):
                parts = folder.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    logs.append(f"⚠️ Skipping: {folder}")
                    continue

                f_name      = parts[0].replace("_", " ")
                sid         = int(parts[1])
                folder_path = os.path.join(DIRS["images"], folder)
                img_files   = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

                status_text.info(f"Processing: **{f_name}** (ID: {sid})")
                logs.append(f"👤 {f_name} (ID: {sid}) — {len(img_files)} images")

                enc_list = []
                for img_file in img_files:
                    bgr = cv2.imread(os.path.join(folder_path, img_file))
                    if bgr is None:
                        continue
                    try:
                        enc = get_embedding(bgr)
                        if enc is not None:
                            enc_list.append(enc)
                            logs.append(f"  ✓ {img_file}")
                    except Exception as e:
                        logs.append(f"  ⚠️ {img_file}: {e}")

                if enc_list:
                    encodings_db[sid] = {"name": f_name, "encodings": enc_list}
                    logs.append(f"  ✅ {len(enc_list)} embeddings saved\n")
                else:
                    logs.append(f"  ❌ No embeddings for {f_name}\n")

                progress_bar.progress((idx + 1) / len(folders))
                # FIX: unique key per iteration prevents StreamlitDuplicateElementId
                log_area.text_area("Training Log", "\n".join(logs[-20:]), height=300, key=f"log_{idx}")

            if not encodings_db:
                st.error("❌ Training failed — no encodings created.")
            else:
                with open(ENCODINGS_PKL, "wb") as f:
                    pickle.dump(encodings_db, f)
                total = sum(len(v["encodings"]) for v in encodings_db.values())
                progress_bar.empty(); status_text.empty()
                st.success("✅ **Training Complete!**")
                ca, cb = st.columns(2)
                ca.metric("📊 Students Trained", len(encodings_db))
                cb.metric("🎯 Total Embeddings", total)
                st.balloons()

    st.markdown("---")
    st.markdown("""
**What happens during training?**
Face detection on registered images → Facenet embeddings (128-D) → saved for live recognition.
Retrain after registering new students. First run downloads Facenet model (~90 MB).
    """)


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE  —  uses st.camera_input()
# ══════════════════════════════════════════════════════════════
elif page == "📋 Attendance":
    st.markdown('<p class="section-title">Mark Attendance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Take a photo to recognise and mark attendance</p>', unsafe_allow_html=True)

    if not os.path.exists(ENCODINGS_PKL):
        st.error("❌ No trained model found. Please train the model first.")
        st.stop()

    # Request rear camera early so browser warms up the stream
    st.markdown("""
    <script>
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
            .catch(function(e) { console.warn("Camera pre-warm failed:", e); });
    }
    </script>
    """, unsafe_allow_html=True)

    enc_db_att   = load_encodings()
    session_date = datetime.now().strftime("%Y-%m-%d")
    haar         = get_haar()

    if "att_log"    not in st.session_state: st.session_state.att_log    = []
    if "att_marked" not in st.session_state: st.session_state.att_marked = set()

    col_cam, col_log = st.columns([3, 2])

    with col_cam:
        st.info("📷 Click **Take Photo** to recognise the student in front of the camera.")

        photo = st.camera_input(
            label="Take attendance photo",
            key="att_camera",
            help="Point the camera at student(s) and click to capture.",
        )

        if photo is not None:
            pil_img = Image.open(io.BytesIO(photo.getvalue()))
            bgr     = pil_to_bgr(pil_img)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            faces = detect_faces(gray, haar)
            out   = bgr.copy()

            if faces is None or len(faces) == 0:
                st.warning("⚠️ No face detected. Try better lighting or move closer.")
            else:
                results = []
                for (x, y, w, h) in faces:
                    fc = bgr[y:y+h, x:x+w]
                    if fc.size == 0:
                        continue
                    try:
                        enc = get_embedding(fc)
                        if enc is None:
                            draw_box(out, x, y, w, h, "?", (80, 80, 200))
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
                            color   = (200, 140, 0) if already else (0, 200, 60)
                            label   = f"{best_name} ({'marked' if already else 'PRESENT'})"
                            draw_box(out, x, y, w, h, label, color)
                            if not already:
                                now = datetime.now().strftime("%H:%M:%S")
                                st.session_state.att_marked.add(best_id)
                                st.session_state.att_log.append(
                                    [best_id, best_name, now, session_date, "Present"]
                                )
                                results.append(f"✅ **{best_name}** marked present")
                            else:
                                results.append(f"🔁 **{best_name}** already marked")
                        else:
                            draw_box(out, x, y, w, h, f"Unknown ({best_dist:.2f})", (60, 60, 200))
                            results.append(f"❓ Unknown face (score: {best_dist:.2f})")

                    except Exception:
                        draw_box(out, x, y, w, h, "error", (0, 0, 180))

                rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                st.image(rgb_out, channels="RGB", use_container_width=True)
                for r in results:
                    st.markdown(r)

        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.att_log    = []
            st.session_state.att_marked = set()
            st.rerun()

    with col_log:
        st.subheader("✅ Attendance Log")
        if st.session_state.att_log:
            df_live = pd.DataFrame(
                st.session_state.att_log,
                columns=["ID", "Name", "Time", "Date", "Status"]
            )
            st.dataframe(df_live[["Name", "Time"]], use_container_width=True,
                         hide_index=True, height=350)
            st.success(f"**{len(st.session_state.att_marked)} student(s) marked present**")
            st.markdown("---")

            if st.button("💾 Save to Excel", type="primary", use_container_width=True):
                excel_path = os.path.join(DIRS["attendance"], f"Attendance_{session_date}.xlsx")
                df_save    = pd.DataFrame(st.session_state.att_log,
                                          columns=["ID","Name","Time","Date","Status"])
                mode = "a" if os.path.exists(excel_path) else "w"
                kw   = {"if_sheet_exists": "replace"} if mode == "a" else {}
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode, **kw) as writer:
                    df_save.to_excel(writer, sheet_name=session_date, index=False)
                st.success(f"✅ Saved to `{excel_path}`")

            df_dl = pd.DataFrame(st.session_state.att_log,
                                 columns=["ID","Name","Time","Date","Status"])
            st.download_button(
                label="⬇️ Download CSV",
                data=df_dl.to_csv(index=False).encode("utf-8"),
                file_name=f"Attendance_{session_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No attendance recorded yet.\nTake a photo on the left to begin.")


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
            xl        = pd.ExcelFile(sel_path)
            sheets    = xl.sheet_names
            sel_sheet = st.selectbox("📄 Select date", sheets, index=len(sheets)-1) if len(sheets) > 1 else sheets[0]
            df_report = xl.parse(sel_sheet)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("📋 Present", len(df_report))
            if os.path.exists(STUDENTS_CSV):
                total_reg = len(pd.read_csv(STUDENTS_CSV))
                mc2.metric("👥 Total Enrolled", total_reg)
                mc3.metric("📊 Attendance Rate", f"{len(df_report)/max(total_reg,1)*100:.1f}%")

            st.markdown("---")
            search = st.text_input("🔍 Search by name or ID")
            if search:
                mask = (
                    df_report["Name"].astype(str).str.contains(search, case=False, na=False) |
                    df_report["ID"].astype(str).str.contains(search, case=False, na=False)
                )
                df_report = df_report[mask]

            st.dataframe(df_report, use_container_width=True, hide_index=True, height=400)
            st.download_button("⬇️ Download CSV",
                               df_report.to_csv(index=False).encode(),
                               file_name=f"{selected_file.replace('.xlsx','')}.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    st.markdown("---")
    st.subheader("📊 Historical Analytics")
    df_all = load_attendance_all()
    if df_all.empty:
        st.info("No historical records available.")
    else:
        st.bar_chart(df_all.groupby("Date").size().reset_index(name="Count").set_index("Date")["Count"])
        st.subheader("🏆 Top Attendees")
        st.dataframe(
            df_all.groupby("Name").size().reset_index(name="Days Present")
            .sort_values("Days Present", ascending=False).head(10),
            use_container_width=True, hide_index=True
        )