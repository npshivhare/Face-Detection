# 👤 Face Recognition Attendance System using FaceNet

A real-time face recognition-based attendance system leveraging a **pre-trained FaceNet model**, OpenCV for detection, and cosine similarity for accurate identity matching.

🚀 **Live App:** https://face-detection-84kwe6szpukadkp4o2tjyt.streamlit.app

📂 **GitHub Repo:** https://github.com/npshivhare/Face-Detection  

🎥 **Demo Video:** [Add YouTube Link]

---

## 📌 Overview

This project presents a **complete end-to-end face recognition pipeline** for automated attendance management.

The system:
- Detects faces using OpenCV Haar Cascade  
- Extracts **128-dimensional embeddings** using FaceNet (via DeepFace)  
- Matches identities using **cosine similarity**  
- Logs attendance automatically into Excel files  

It provides a **scalable, real-time, and contactless solution** for attendance tracking.

---

## 🎯 Key Features

- 🎥 Real-time face detection via webcam  
- 🧠 Face recognition using pre-trained FaceNet  
- 📊 Cosine similarity-based identity matching  
- 🧾 Automated Excel attendance logging  
- 🌐 Interactive Streamlit web interface  
- 👥 Multi-student registration system  
- 🚫 Unknown face detection (threshold-based)  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - OpenCV (Face Detection)  
  - DeepFace (FaceNet embeddings)  
  - NumPy, Pandas  
  - SciPy (Cosine Similarity)  
  - Streamlit (Frontend UI)  
  - OpenPyXL (Excel Handling)  
- **Model:** Pre-trained FaceNet (128-D embeddings)

---

## ⚙️ System Architecture

1. Capture image/video from webcam  
2. Detect face using Haar Cascade  
3. Crop & preprocess face  
4. Generate embedding using FaceNet  
5. Compare with stored embeddings  
6. Apply cosine similarity threshold  
7. Mark attendance or label as "Unknown"  

---

## 🧠 Model Selection

| Model     | Accuracy | Scalability | Robustness | Final Choice |
|----------|---------|------------|------------|-------------|
| FaceNet  | ~99.6% | High       | High       | ⭐ Selected  |
| OpenFace | ~92.9% | High       | Moderate   | Not Used    |
| DeepID   | ~97%   | Low        | Low        | Not Used    |

✅ **FaceNet chosen due to high accuracy, scalability, and real-world robustness**

---

## 📊 Methodology

### 🔹 Face Detection
- OpenCV Haar Cascade  
- Fast and efficient for real-time CPU usage  

### 🔹 Embedding Extraction
- DeepFace (FaceNet model)  
- Generates **128-D feature vectors**  

### 🔹 Identity Matching
- Cosine similarity metric  
- Threshold: **0.55**  
- Average of nearest embeddings used  

### 🔹 Attendance Logging
- Stored in **date-stamped Excel files**  
- Duplicate prevention within session  

---

## 🖥️ Application Features

### 📌 Modules

- Dashboard → System overview  
- Register → Capture student images  
- Train Model → Generate embeddings  
- Attendance → Real-time recognition  
- Reports → Analytics + CSV export  

---


## 📊 Results & Performance

- ✅ Reliable face detection under normal lighting  
- ⚡ Embedding extraction: ~0.5–1 sec per face (CPU)  
- 🎯 Accurate real-time recognition  
- 🚫 Unknown faces correctly rejected using threshold  
- 📉 Slight performance drop under extreme lighting  

---

## 🏆 Key Achievements

- Built **complete face recognition pipeline from scratch**  
- Implemented **real-time biometric attendance system**  
- Used **FaceNet embeddings (128-D)** for high accuracy  
- Designed **dual interface (CLI + Streamlit GUI)**  
- Automated **Excel-based attendance tracking system**  

---

## 👨‍💻 Team Members

- Krish Naik  
- Nrependre Shivhare  
- Prasham Godha  

---

## 🙏 Mentors

- Dr. K. K. Sharma  
- Dr. Lalit Purohit  
- Dr. Upendra Singh  
- Mr. Akshay Gupta  

---

## 🔮 Future Work

- Multi-face recognition in single frame  
- Liveness detection (anti-spoofing)  
- Mobile/web deployment  
- Integration with college ERP systems  
- Replace Haar Cascade with MTCNN/RetinaFace  

---

## 📚 References

- FaceNet (Google Research, CVPR 2015)  
- DeepFace Framework  
- OpenCV Documentation  
- Research papers on Face Recognition  

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
