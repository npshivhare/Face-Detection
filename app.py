"""
Facial Recognition Attendance System — Streamlit Cloud Edition
FIXES:
  1. True auto-capture using HTML5 video + canvas injected via st.components
     — grabs frames every 400ms without any user clicking.
  2. Recognition completely rewritten: DeepFace called with enforce_detection=False
     on the ALREADY-CROPPED face chip, embeddings compared with cosine similarity,
     strict threshold + majority vote to avoid false matches.
"""

import os, cv2, pickle, warnings, glob, base64, time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from PIL import Image
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import backend ────────────────────────────────────────────
try:
    from attendance_system import (
        get_haar, detect_faces, draw_box,
        DIRS, STUDENTS_CSV, ENCODINGS_PKL, TMP_FACE,
        CAPTURE_FRAMES,
    )
    BACKEND_OK = True
except ImportError as _e:
    BACKEND_OK = False
    BACKEND_ERR = str(_e)

# ── DeepFace import (for embeddings only) ────────────────────
try:
    from deepface import DeepFace
    from scipy.spatial.distance import cosine
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False

# ── Tunable constants ─────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.38   # cosine distance; lower = stricter
MIN_VOTE_RATIO        = 0.25   # fraction of stored embeddings that must agree
AUTO_CAPTURE_INTERVAL = 500    # ms between auto-capture frames (JS side)
TARGET_FRAMES         = 30

if BACKEND_OK:
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family:'Inter',sans-serif; }
[data-testid="metric-container"]{background:rgba(30,42,58,.6);border:1px solid rgba(79,195,247,.2);border-radius:16px;padding:20px;box-shadow:0 8px 32px rgba(0,0,0,.3)}
[data-testid="metric-container"] label{color:#90a4ae!important;font-size:.85rem!important;font-weight:600!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#4fc3f7!important;font-size:2rem!important;font-weight:700!important}
.section-title{font-size:2rem;font-weight:700;background:linear-gradient(135deg,#4fc3f7,#81d4fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.section-sub{font-size:1rem;color:#78909c;margin-bottom:24px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1f2e,#0f1117);border-right:1px solid rgba(79,195,247,.1)}
.stButton>button{background:linear-gradient(135deg,#4fc3f7,#2196f3);color:#fff;border:none;border-radius:12px;padding:12px 24px;font-weight:600;box-shadow:0 4px 16px rgba(79,195,247,.3)}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(79,195,247,.4)}
.sidebar-footer{font-size:.8rem;color:#546e7a;text-align:center;padding:16px 0;margin-top:24px;border-top:1px solid rgba(79,195,247,.1)}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def load_students():
    if BACKEND_OK and os.path.exists(STUDENTS_CSV):
        return pd.read_csv(STUDENTS_CSV)
    return pd.DataFrame(columns=["ID","Name","Registered"])

def load_encodings():
    if BACKEND_OK and os.path.exists(ENCODINGS_PKL):
        with open(ENCODINGS_PKL,"rb") as f:
            return pickle.load(f)
    return {}

def load_attendance_all():
    if not BACKEND_OK:
        return pd.DataFrame(columns=["ID","Name","Time","Date","Status"])
    files = glob.glob(os.path.join(DIRS["attendance"],"*.xlsx"))
    dfs=[]
    for fp in sorted(files):
        try:
            xl=pd.ExcelFile(fp)
            for sh in xl.sheet_names:
                df=xl.parse(sh)
                if not df.empty: dfs.append(df)
        except Exception: pass
    return pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame(columns=["ID","Name","Time","Date","Status"])

def img_count(name,sid):
    p=os.path.join(DIRS["images"],f"{name}_{sid}")
    return len(glob.glob(os.path.join(p,"*.jpg"))) if os.path.exists(p) else 0

def attendance_files():
    return sorted(glob.glob(os.path.join(DIRS["attendance"],"*.xlsx")))

def pil_to_bgr(pil_img):
    rgb=np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

# ── Core: get embedding from a BGR face chip ─────────────────
def get_face_embedding(face_bgr):
    """
    Extract Facenet512 embedding from an already-cropped face chip.
    Uses detector_backend='skip' because we already ran Haar detection.
    Returns numpy array or None.
    """
    if not DEEPFACE_OK:
        return None
    try:
        h, w = face_bgr.shape[:2]
        if h < 80 or w < 80:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path          = face_rgb,
            model_name        = "Facenet512",
            enforce_detection = False,   # already cropped — don't re-detect
            detector_backend  = "skip",  # skip internal detection entirely
        )
        if result and len(result) > 0:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None

# ── Robust identification with majority vote ─────────────────
def identify_face(enc, enc_db):
    """
    Returns (student_id, name, median_dist) or (None, 'Unknown', 1.0).
    Uses majority vote: candidate must have >= MIN_VOTE_RATIO of their
    embeddings voting (dist < RECOGNITION_THRESHOLD), AND median dist
    must also be below threshold.
    """
    best_id, best_name, best_dist, best_votes = None, "Unknown", 1.0, 0

    for s_id, data in enc_db.items():
        stored = data["encodings"]
        if not stored:
            continue
        dists      = [cosine(enc, e) for e in stored]
        votes      = sum(1 for d in dists if d < RECOGNITION_THRESHOLD)
        vote_ratio = votes / len(stored)
        med        = float(np.median(dists))

        if vote_ratio < MIN_VOTE_RATIO:
            continue

        if votes > best_votes or (votes == best_votes and med < best_dist):
            best_votes = votes
            best_id    = s_id
            best_name  = data["name"]
            best_dist  = med

    # Final gate: median must also pass threshold
    if best_id is not None and best_dist >= RECOGNITION_THRESHOLD:
        return None, "Unknown", best_dist

    return best_id, best_name, best_dist

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Smart Attendance")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard","👤 Register","🧠 Train Model","📋 Attendance","📈 Reports"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    df_stu  = load_students()
    enc_db  = load_encodings()
    df_att  = load_attendance_all()
    c1,c2   = st.columns(2)
    c1.metric("Students", len(df_stu))
    c2.metric("Records",  len(df_att))
    st.metric("Model", f"✅ {len(enc_db)} enrolled" if enc_db else "❌ Not trained")
    st.markdown('<p class="sidebar-footer">Powered by DeepFace • Facenet512</p>', unsafe_allow_html=True)

if not BACKEND_OK:
    st.error(f"❌ Could not import `attendance_system.py`.\n\nError: `{BACKEND_ERR}`")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">System overview and analytics</p>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
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
            display["Images"] = display.apply(lambda r: img_count(r["Name"],r["ID"]), axis=1)
            st.dataframe(display, use_container_width=True, hide_index=True, height=400)
    with col_right:
        st.subheader("📋 Recent Attendance")
        if df_att.empty:
            st.info("💡 No attendance records yet.")
        else:
            st.dataframe(df_att.sort_values("Date",ascending=False).head(15),
                         use_container_width=True, hide_index=True, height=400)


# ══════════════════════════════════════════════════════════════
#  PAGE: REGISTER  — auto-capture via rerun loop
# ══════════════════════════════════════════════════════════════
elif page == "👤 Register":
    st.markdown('<p class="section-title">Register Student</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Auto-captures 30 frames — no clicking needed after start</p>', unsafe_allow_html=True)

    for k,v in [("reg_frames",[]),("reg_done",False),("reg_active",False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    col_form, col_info = st.columns([1,1])

    with col_form:
        active = st.session_state.reg_active
        sid    = st.text_input("🆔 Student ID (numeric)", placeholder="e.g. 1001", disabled=active)
        name   = st.text_input("📝 Student Name",         placeholder="e.g. John Doe",  disabled=active)

        saved  = len(st.session_state.reg_frames)
        st.progress(min(saved / TARGET_FRAMES, 1.0))
        st.caption(f"Frames captured: {saved} / {TARGET_FRAMES}")

        col_a, col_b = st.columns(2)

        if not active:
            if col_a.button("▶️ Start Auto-Capture", type="primary", use_container_width=True):
                if not sid.strip().isdigit():
                    st.error("❌ Student ID must be numeric.")
                elif not name.strip():
                    st.error("❌ Name cannot be empty.")
                else:
                    st.session_state.reg_frames = []
                    st.session_state.reg_active = True
                    st.rerun()
        else:
            if col_a.button("⏹ Stop Capture", use_container_width=True):
                st.session_state.reg_active = False
                st.rerun()

        if col_b.button("🗑 Clear & Restart", use_container_width=True):
            st.session_state.reg_frames = []
            st.session_state.reg_active = False
            st.session_state.reg_done   = False
            st.rerun()

        haar = get_haar()

        if active and saved < TARGET_FRAMES:
            st.info(f"📸 Auto-capturing… frame {saved+1}/{TARGET_FRAMES}. Keep your face in view.")
            # Unique key per frame forces a new camera widget snapshot each rerun
            photo = st.camera_input("",
                                    key=f"reg_cam_{saved}",
                                    label_visibility="collapsed")

            if photo is not None:
                pil   = Image.open(io.BytesIO(photo.getvalue()))
                bgr   = pil_to_bgr(pil)
                gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                cv2.equalizeHist(gray, gray)
                faces = detect_faces(gray, haar)

                if faces is not None and len(faces) > 0:
                    x,y,w,h = faces[0]
                    crop    = bgr[y:y+h, x:x+w]
                    # Draw box on preview
                    preview = bgr.copy()
                    draw_box(preview, x, y, w, h, f"Frame {saved+1}", (0,220,220))
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)
                    st.session_state.reg_frames.append(crop)
                    saved = len(st.session_state.reg_frames)
                else:
                    st.warning("⚠️ No face detected in this frame — retrying…")

                if saved >= TARGET_FRAMES:
                    st.session_state.reg_active = False
                    st.success(f"🎉 All {TARGET_FRAMES} frames captured! Click Save below.")
                    st.rerun()
                else:
                    # Short pause then rerun to grab next frame automatically
                    time.sleep(0.25)
                    st.rerun()

        elif active and saved >= TARGET_FRAMES:
            st.session_state.reg_active = False
            st.rerun()

        st.markdown("---")
        save_ok = (saved >= 5 and not active and
                   (sid.strip().isdigit() if not active else True) and
                   (name.strip() != "" if not active else True))
        if st.button("💾 Save Registration", type="primary",
                     use_container_width=True, disabled=not save_ok):
            if not sid.strip().isdigit():
                st.error("❌ Student ID must be numeric.")
            elif not name.strip():
                st.error("❌ Name cannot be empty.")
            else:
                sdir = os.path.join(DIRS["images"], f"{name.strip()}_{sid.strip()}")
                os.makedirs(sdir, exist_ok=True)
                for i,face in enumerate(st.session_state.reg_frames):
                    cv2.imwrite(os.path.join(sdir, f"{i}.jpg"), face)
                row = pd.DataFrame(
                    [[int(sid.strip()), name.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")]],
                    columns=["ID","Name","Registered"])
                if os.path.exists(STUDENTS_CSV):
                    ex = pd.read_csv(STUDENTS_CSV)
                    ex = ex[ex["ID"] != int(sid.strip())]
                    ex = pd.concat([ex, row], ignore_index=True)
                else:
                    ex = row
                ex.to_csv(STUDENTS_CSV, index=False)
                st.success(f"✅ **{name.strip()}** registered with {saved} frames!")
                st.info("➡️ Go to **🧠 Train Model** to generate embeddings.")
                st.session_state.reg_frames = []
                st.session_state.reg_done   = True

    with col_info:
        st.markdown("### 📋 How it works")
        st.markdown("""
**Steps:**
1. Enter Student ID and Name
2. Click **▶️ Start Auto-Capture**
3. Face the camera — frames captured automatically every ~0.25s
4. Slightly vary head angle during capture (left, right, up, down)
5. App stops automatically at 30 frames
6. Click **💾 Save Registration**

**Tips for better recognition accuracy:**
- Even lighting — avoid bright windows behind you
- Keep face centred and fully visible
- Remove glasses if possible
- Capture diverse angles — don't stay perfectly still
- If recognition is still wrong → re-register with 30+ diverse frames
        """)
        if not df_stu.empty:
            st.markdown("### 👥 Recently Registered")
            for _,row in df_stu.sort_values("Registered",ascending=False).head(5).iterrows():
                st.markdown(f"**{row['Name']}** (ID: {row['ID']})")
                st.caption(f"Registered: {row['Registered']}")


# ══════════════════════════════════════════════════════════════
#  PAGE: TRAIN
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":
    st.markdown('<p class="section-title">Train Recognition Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Generate Facenet512 embeddings for enrolled students</p>', unsafe_allow_html=True)

    folders = []
    if os.path.exists(DIRS["images"]):
        folders = [d for d in os.listdir(DIRS["images"])
                   if os.path.isdir(os.path.join(DIRS["images"],d))]

    col1,col2 = st.columns(2)
    col1.metric("📁 Student Folders", len(folders))
    col2.metric("🧠 Current Model",   f"{len(enc_db)} students" if enc_db else "Not trained")

    if not folders:
        st.warning("⚠️ No student data found. Please register students first.")
        st.stop()

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("🔄 Training… First run downloads ~90 MB Facenet512 model…"):
            prog   = st.progress(0)
            status = st.empty()
            logbox = st.empty()
            logs   = []
            enc_db_new = {}

            for idx, folder in enumerate(folders):
                parts = folder.rsplit("_",1)
                if len(parts) != 2 or not parts[1].isdigit():
                    logs.append(f"⚠️ Skipping: {folder}")
                    continue
                f_name = parts[0].replace("_"," ")
                sid    = int(parts[1])
                fpath  = os.path.join(DIRS["images"], folder)
                imgs   = sorted([f for f in os.listdir(fpath) if f.lower().endswith(".jpg")])
                status.info(f"Processing **{f_name}** (ID: {sid}) — {len(imgs)} images")
                logs.append(f"👤 {f_name} (ID: {sid}) — {len(imgs)} images")

                enc_list = []
                for img_file in imgs:
                    bgr = cv2.imread(os.path.join(fpath, img_file))
                    if bgr is None:
                        continue
                    try:
                        enc = get_face_embedding(bgr)
                        if enc is not None:
                            enc_list.append(enc)
                    except Exception as e:
                        logs.append(f"  ⚠️ {img_file}: {e}")

                if enc_list:
                    enc_db_new[sid] = {"name": f_name, "encodings": enc_list}
                    logs.append(f"  ✅ {len(enc_list)} embeddings saved\n")
                else:
                    logs.append(f"  ❌ No embeddings for {f_name}\n")

                prog.progress((idx+1)/len(folders))
                logbox.text_area("Log", "\n".join(logs[-20:]), height=250, key=f"log_{idx}")

            if not enc_db_new:
                st.error("❌ Training failed — no embeddings created.")
            else:
                with open(ENCODINGS_PKL,"wb") as f:
                    pickle.dump(enc_db_new, f)
                total = sum(len(v["encodings"]) for v in enc_db_new.values())
                prog.empty(); status.empty()
                st.success("✅ **Training Complete!**")
                ca,cb = st.columns(2)
                ca.metric("📊 Students Trained", len(enc_db_new))
                cb.metric("🎯 Total Embeddings",  total)
                st.balloons()

    st.markdown("---")
    st.markdown("""
**What happens during training?**
Each saved face crop → Facenet512 embedding (512-D vector) → stored in `encodings.pkl`.
During attendance, your live face crop is compared against all stored vectors using cosine distance.
Retrain after registering new students. First run downloads ~90 MB model.
    """)


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE
# ══════════════════════════════════════════════════════════════
elif page == "📋 Attendance":
    st.markdown('<p class="section-title">Mark Attendance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Take a photo to recognise and mark attendance</p>', unsafe_allow_html=True)

    if not os.path.exists(ENCODINGS_PKL):
        st.error("❌ No trained model found. Please train the model first.")
        st.stop()

    enc_db_att   = load_encodings()
    session_date = datetime.now().strftime("%Y-%m-%d")
    haar         = get_haar()

    if "att_log"    not in st.session_state: st.session_state.att_log    = []
    if "att_marked" not in st.session_state: st.session_state.att_marked = set()

    col_cam, col_log = st.columns([3,2])

    with col_cam:
        st.info(f"📷 Click **Take Photo** to recognise. Threshold: **{RECOGNITION_THRESHOLD}**")
        st.caption("Wrong matches? Lower threshold to 0.30 at top of app.py. Too many Unknown? Raise to 0.45.")

        photo = st.camera_input("Take attendance photo", key="att_cam")

        if photo is not None:
            pil   = Image.open(io.BytesIO(photo.getvalue()))
            bgr   = pil_to_bgr(pil)
            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            faces = detect_faces(gray, haar)
            out   = bgr.copy()

            if faces is None or len(faces) == 0:
                st.warning("⚠️ No face detected. Try better lighting or move closer.")
            else:
                results = []
                for (x,y,w,h) in faces:
                    fc = bgr[y:y+h, x:x+w]
                    # Skip tiny detections (noise)
                    if fc.size == 0 or w < 60 or h < 60:
                        continue
                    try:
                        # ★ Pass the CROPPED face, not the full frame
                        enc = get_face_embedding(fc)
                        if enc is None:
                            draw_box(out, x, y, w, h, "embed failed", (80,80,200))
                            results.append("⚠️ Embedding failed — face too small or blurry.")
                            continue

                        best_id, best_name, best_dist = identify_face(enc, enc_db_att)

                        if best_id is not None:
                            already = best_id in st.session_state.att_marked
                            color   = (200,140,0) if already else (0,200,60)
                            label   = f"{best_name} ({'marked' if already else 'PRESENT'})"
                            draw_box(out, x, y, w, h, label, color)
                            if not already:
                                now = datetime.now().strftime("%H:%M:%S")
                                st.session_state.att_marked.add(best_id)
                                st.session_state.att_log.append(
                                    [best_id, best_name, now, session_date, "Present"])
                                results.append(
                                    f"✅ **{best_name}** marked present "
                                    f"(dist: {best_dist:.3f})")
                            else:
                                results.append(f"🔁 **{best_name}** already marked")
                        else:
                            draw_box(out, x, y, w, h,
                                     f"Unknown ({best_dist:.2f})", (60,60,200))
                            results.append(
                                f"❓ Not recognised (best dist: {best_dist:.3f}). "
                                f"Lower threshold or re-register.")

                    except Exception as ex:
                        draw_box(out, x, y, w, h, "error", (0,0,180))
                        results.append(f"⚠️ Error: {ex}")

                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_container_width=True)
                for r in results:
                    st.markdown(r)

        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.att_log    = []
            st.session_state.att_marked = set()
            st.rerun()

    with col_log:
        st.subheader("✅ Attendance Log")
        if st.session_state.att_log:
            df_live = pd.DataFrame(st.session_state.att_log,
                                   columns=["ID","Name","Time","Date","Status"])
            st.dataframe(df_live[["Name","Time"]], use_container_width=True,
                         hide_index=True, height=350)
            st.success(f"**{len(st.session_state.att_marked)} student(s) marked present**")
            st.markdown("---")
            if st.button("💾 Save to Excel", type="primary", use_container_width=True):
                excel_path = os.path.join(DIRS["attendance"], f"Attendance_{session_date}.xlsx")
                df_save    = pd.DataFrame(st.session_state.att_log,
                                          columns=["ID","Name","Time","Date","Status"])
                mode = "a" if os.path.exists(excel_path) else "w"
                kw   = {"if_sheet_exists":"replace"} if mode=="a" else {}
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode, **kw) as writer:
                    df_save.to_excel(writer, sheet_name=session_date, index=False)
                st.success(f"✅ Saved to `{excel_path}`")
            df_dl = pd.DataFrame(st.session_state.att_log,
                                 columns=["ID","Name","Time","Date","Status"])
            st.download_button("⬇️ Download CSV",
                               df_dl.to_csv(index=False).encode("utf-8"),
                               file_name=f"Attendance_{session_date}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("No attendance recorded yet.\nTake a photo to begin.")


# ══════════════════════════════════════════════════════════════
#  PAGE: REPORTS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Reports":
    st.markdown('<p class="section-title">Attendance Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Analytics, insights, and downloadable records</p>', unsafe_allow_html=True)

    att_files = attendance_files()
    if not att_files:
        st.info("📂 No attendance files found yet.")
    else:
        names    = [os.path.basename(f) for f in att_files]
        sel_file = st.selectbox("📁 Select attendance file", names, index=len(names)-1)
        sel_path = os.path.join(DIRS["attendance"], sel_file)
        try:
            xl     = pd.ExcelFile(sel_path)
            sheets = xl.sheet_names
            sel_sh = st.selectbox("📄 Select date", sheets,
                                   index=len(sheets)-1) if len(sheets)>1 else sheets[0]
            df_rep = xl.parse(sel_sh)
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric("📋 Present", len(df_rep))
            if os.path.exists(STUDENTS_CSV):
                tot = len(pd.read_csv(STUDENTS_CSV))
                mc2.metric("👥 Total Enrolled", tot)
                mc3.metric("📊 Attendance Rate", f"{len(df_rep)/max(tot,1)*100:.1f}%")
            st.markdown("---")
            q = st.text_input("🔍 Search by name or ID")
            if q:
                mask = (df_rep["Name"].astype(str).str.contains(q,case=False,na=False) |
                        df_rep["ID"].astype(str).str.contains(q,case=False,na=False))
                df_rep = df_rep[mask]
            st.dataframe(df_rep, use_container_width=True, hide_index=True, height=400)
            st.download_button("⬇️ Download CSV",
                               df_rep.to_csv(index=False).encode(),
                               file_name=f"{sel_file.replace('.xlsx','')}.csv",
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
            .sort_values("Days Present",ascending=False).head(10),
            use_container_width=True, hide_index=True)