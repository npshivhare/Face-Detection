"""
Facial Recognition Attendance System
Engine  : OpenCV (detection) + DeepFace/Facenet (recognition)
Platform: Windows / VS Code
Run     : py -3.11 attendance_system.py
"""

import os, cv2, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cosine

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── lazy-import DeepFace so menu appears instantly ────────────
_deepface = None
def get_deepface():
    global _deepface
    if _deepface is None:
        print("   Loading DeepFace model (first time only)…", flush=True)
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface

# ── CONFIG ────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CAPTURE_FRAMES = 30
THRESHOLD      = 0.55   # cosine distance — raise if too many unknowns

DIRS = {
    "images"    : "StudentData/images",
    "encodings" : "StudentData/encodings",
    "attendance": "Attendance",
}
for p in DIRS.values():
    os.makedirs(p, exist_ok=True)

STUDENTS_CSV  = "StudentData/students.csv"
ENCODINGS_PKL = "StudentData/encodings/encodings.pkl"
TMP_FACE      = "StudentData/tmp_face.jpg"

# ── HELPERS ───────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_INDEX}. "
            "Try changing CAMERA_INDEX at the top of this file."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    return cap

def warmup(cap, n=60):
    print("   Warming up camera…", end="", flush=True)
    for _ in range(n):
        cap.read()
    print(" ready!")

def get_haar():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(gray, haar):
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(50, 50)
    )
    return faces if len(faces) > 0 else []

def draw_box(frame, x, y, w, h, label, color=(0, 220, 80)):
    """Draw rectangle and label with FIXED coordinate types for OpenCV"""
    # Convert all coordinates to Python int (not numpy.int64 or float)
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
    
    # Ensure org is a tuple of Python ints
    cv2.putText(frame, label, (x+4, y-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def hud(frame, text, y=None, color=(0, 220, 80)):
    """Draw HUD text with FIXED coordinate types"""
    if y is None:
        y = frame.shape[0] - 12
    
    # Convert y to Python int
    y = int(y)
    
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

def get_embedding(img_bgr):
    """Return 128-d Facenet embedding for a BGR image crop."""
    DeepFace = get_deepface()
    cv2.imwrite(TMP_FACE, img_bgr)
    result = DeepFace.represent(
        img_path          = TMP_FACE,
        model_name        = "Facenet",
        enforce_detection = False,
        detector_backend  = "skip",
    )
    if result:
        return np.array(result[0]["embedding"], dtype=np.float32)
    return None

# ── 1. REGISTER ───────────────────────────────────────────────
def register_student():
    print("\n" + "─"*52)
    print("  STUDENT REGISTRATION")
    print("─"*52)

    sid = input("Enter Student ID (numeric): ").strip()
    if not sid.isdigit():
        print("❌  ID must be numeric."); return

    name = input("Enter Student Name   : ").strip()
    if not name:
        print("❌  Name cannot be empty."); return

    # Duplicate check
    if os.path.exists(STUDENTS_CSV):
        df_ex = pd.read_csv(STUDENTS_CSV)
        if int(sid) in df_ex["ID"].values:
            ans = input(f"⚠️  ID {sid} already exists. Re-register? (y/n): ")
            if ans.lower() != "y":
                print("Cancelled."); return

    student_dir = os.path.join(DIRS["images"], f"{name}_{sid}")
    os.makedirs(student_dir, exist_ok=True)

    print(f"\n📸  Camera will open.")
    print("    ① Click the camera window")
    print("    ② Press SPACE to start capturing")
    print("    ③ Stay still — 30 frames save automatically")
    print("    ④ Press Q when done\n")

    cap  = open_camera()
    haar = get_haar()
    warmup(cap)

    saved = 0
    phase = "waiting"  # waiting → capturing → done

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = detect_faces(gray, haar)

        # Draw face boxes
        for (x, y, w, h) in faces:
            col = (0, 255, 0) if phase == "capturing" else (0, 220, 220)
            draw_box(frame, x, y, w, h,
                     f"{saved}/{CAPTURE_FRAMES}" if phase == "capturing" else "Face OK",
                     col)

        if phase == "waiting":
            hud(frame, f"Faces:{len(faces)}  SPACE=start  Q=quit")

        elif phase == "capturing":
            cv2.imwrite(os.path.join(student_dir, f"{saved}.jpg"), frame)
            saved += 1
            hud(frame, f"Capturing {saved}/{CAPTURE_FRAMES} — keep still!", color=(0,200,80))
            if saved >= CAPTURE_FRAMES:
                phase = "done"

        elif phase == "done":
            hud(frame, "Capture complete!  Press Q", color=(0,200,80))

        cv2.imshow("Registration — press SPACE then Q", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and phase == "waiting":
            phase = "capturing"
            print("▶  Capturing frames…")

        if key == ord("q") or phase == "done":
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved < 5:
        print(f"⚠️  Only {saved} frames captured. Try again."); return

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

    print(f"\n✅  Registered {name}  ({saved} frames saved)")
    print(f"    Folder: {student_dir}")

# ── 2. TRAIN ──────────────────────────────────────────────────
def train_model():
    print("\n" + "─"*52)
    print("  TRAINING  (DeepFace / Facenet)")
    print("─"*52)
    print("  First run downloads Facenet model (~90 MB)…\n")

    folders = [
        d for d in os.listdir(DIRS["images"])
        if os.path.isdir(os.path.join(DIRS["images"], d))
    ]
    if not folders:
        print("❌  No student folders found. Register students first.")
        return

    encodings_db = {}

    for folder in folders:
        parts = folder.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            print(f"  Skipping unrecognised folder: {folder}")
            continue

        name        = parts[0].replace("_", " ")
        sid         = int(parts[1])
        folder_path = os.path.join(DIRS["images"], folder)
        img_files   = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg")
        ])

        print(f"  👤  {name}  (ID: {sid})  —  {len(img_files)} images")
        enc_list = []

        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"    ⚠️  Cannot read {img_file}")
                continue
            try:
                enc = get_embedding(bgr)
                if enc is not None:
                    enc_list.append(enc)
                    print(f"    ✓  {img_file}")
                else:
                    print(f"    –  no embedding  {img_file}")
            except Exception as e:
                print(f"    ⚠️  {img_file}: {e}")

        if enc_list:
            encodings_db[sid] = {"name": name, "encodings": enc_list}
            print(f"    ✅  {len(enc_list)} embeddings saved\n")
        else:
            print(f"    ❌  No embeddings for {name}\n")

    if not encodings_db:
        print("❌  No encodings created."); return

    with open(ENCODINGS_PKL, "wb") as f:
        pickle.dump(encodings_db, f)

    total = sum(len(v["encodings"]) for v in encodings_db.values())
    print("─"*52)
    print(f"✅  Model saved!")
    print(f"   Students : {len(encodings_db)}")
    print(f"   Samples  : {total}")

# ── 3. ATTENDANCE ─────────────────────────────────────────────
def mark_attendance():
    print("\n" + "─"*52)
    print("  MARK ATTENDANCE  (press Q to stop & save)")
    print("─"*52)

    if not os.path.exists(ENCODINGS_PKL):
        print("❌  No trained model. Run option 2 first."); return

    with open(ENCODINGS_PKL, "rb") as f:
        encodings_db = pickle.load(f)

    print(f"✅  {len(encodings_db)} student(s) loaded.\n")

    session_date = datetime.now().strftime("%Y-%m-%d")
    marked       = set()
    log          = []

    cap  = open_camera()
    haar = get_haar()
    warmup(cap)

    frame_count  = 0
    PROCESS_EVERY = 5   # run recognition every N frames for speed

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame       = cv2.flip(frame, 1)
        frame_count += 1

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = detect_faces(gray, haar)

        for (x, y, w, h) in faces:
            # Only run deep recognition every PROCESS_EVERY frames
            if frame_count % PROCESS_EVERY != 0:
                draw_box(frame, x, y, w, h, "scanning…", (120, 120, 120))
                continue

            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            try:
                face_enc = get_embedding(face_crop)
                if face_enc is None:
                    draw_box(frame, x, y, w, h, "?", (80, 80, 200))
                    continue

                # Match against all students
                best_id, best_dist, best_name = None, 1.0, "Unknown"
                for s_id, data in encodings_db.items():
                    encs  = data["encodings"]
                    dists = [cosine(face_enc, e) for e in encs]
                    avg   = float(np.mean(sorted(dists)[:5]))
                    if avg < best_dist:
                        best_dist = avg
                        best_id   = s_id
                        best_name = data["name"]

                if best_dist < THRESHOLD:
                    already = best_id in marked
                    color   = (0, 180, 60) if not already else (200, 140, 0)
                    label   = f"{best_name} {'(marked)' if already else 'PRESENT'}"
                    draw_box(frame, x, y, w, h, label, color)

                    if not already:
                        now = datetime.now().strftime("%H:%M:%S")
                        marked.add(best_id)
                        log.append([best_id, best_name, now,
                                    session_date, "Present"])
                        print(f"  ✅  {best_name:<22} ID:{best_id}  {now}")
                else:
                    draw_box(frame, x, y, w, h,
                             f"Unknown ({best_dist:.2f})", (60, 60, 200))

            except Exception:
                draw_box(frame, x, y, w, h, "error", (0, 0, 180))

        hud(frame,
            f"Date:{session_date}  Marked:{len(marked)}  Q=stop & save")
        cv2.imshow("Attendance — press Q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(TMP_FACE):
        os.remove(TMP_FACE)

    # ── Save Excel ────────────────────────────────────────────
    if not log:
        print("ℹ️  No attendance recorded this session."); return

    excel_path = os.path.join(DIRS["attendance"],
                              f"Attendance_{session_date}.xlsx")
    df = pd.DataFrame(log, columns=["ID","Name","Time","Date","Status"])

    mode = "a" if os.path.exists(excel_path) else "w"
    kw   = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(excel_path, engine="openpyxl",
                        mode=mode, **kw) as writer:
        df.to_excel(writer, sheet_name=session_date, index=False)

    print(f"\n📊  Attendance saved  →  {excel_path}")
    print(df.to_string(index=False))

# ── 4. VIEW REPORT ────────────────────────────────────────────
def view_report():
    print("\n" + "─"*52)
    print("  ATTENDANCE REPORT")
    print("─"*52)

    files = sorted([
        f for f in os.listdir(DIRS["attendance"])
        if f.endswith(".xlsx")
    ])
    if not files:
        print("  No attendance files found."); return

    for i, f in enumerate(files, 1):
        print(f"  [{i}]  {f}")

    choice = input("\n  Enter number (Enter = latest): ").strip()
    idx    = int(choice) - 1 if choice.isdigit() else len(files) - 1
    if not (0 <= idx < len(files)):
        print("  Invalid choice."); return

    path = os.path.join(DIRS["attendance"], files[idx])
    xl   = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        print(f"\n── {sheet} " + "─"*30)
        print(xl.parse(sheet).to_string(index=False))

    if os.path.exists(STUDENTS_CSV):
        total = len(pd.read_csv(STUDENTS_CSV))
        print(f"\n  Total registered students: {total}")

# ── MENU ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*52)
    print("  FACIAL RECOGNITION ATTENDANCE SYSTEM")
    print("  Engine: DeepFace + Facenet  |  No dlib")
    print("="*52)

    menu = {
        "1": ("Register student",            register_student),
        "2": ("Train model",                 train_model),
        "3": ("Mark attendance (live cam)",  mark_attendance),
        "4": ("View attendance report",      view_report),
        "0": ("Exit",                        None),
    }

    while True:
        print("\n  Menu:")
        for k, (label, _) in menu.items():
            print(f"    [{k}]  {label}")
        choice = input("\n  Select option: ").strip()

        if choice == "0":
            print("Goodbye!"); break
        elif choice in menu:
            try:
                menu[choice][1]()
            except KeyboardInterrupt:
                print("\n  Stopped.")
            except Exception as e:
                print(f"\n❌  Error: {e}")
        else:
            print("  Invalid — try again.")

if __name__ == "__main__":
    main()
