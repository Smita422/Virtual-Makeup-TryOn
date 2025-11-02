# 💄 Virtual Makeup Try-On App

An AI-powered **Virtual Lipstick Try-On** system built entirely using **Python, OpenCV, and Mediapipe**.  
This project allows users to **try different dark lipstick shades** on their own face in real time or by uploading an image — all without any external ML models or heavy frameworks!

---

## 🚀 Features

- 🎥 **Real-Time Lipstick Application** using your webcam  
- 🖼️ **Upload Mode** – apply lipstick to uploaded photos instantly  
- 🎨 Multiple **dark and bold lipstick shades** including Dark Red, Maroon, Plum, Wine, and more  
- 🌿 “**None / Natural Look**” option for a realistic before-after effect  
- 💻 Built with **Streamlit** for a beautiful web interface  
- ⚡ Lightweight, fast, and easy to use  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 🐍 |
| Computer Vision | OpenCV |
| Face Landmark Detection | Mediapipe FaceMesh |
| Web Framework | Streamlit |
| Libraries Used | `opencv-python`, `mediapipe`, `numpy`, `streamlit` |

---

## 🧩 How It Works

1. The app detects **468 facial landmarks** using Mediapipe’s FaceMesh.
2. Specific landmark indices for the **outer and inner lips** are extracted.
3. A colored mask is generated on top of the lips region.
4. The lipstick color is blended smoothly with the original image using OpenCV’s `addWeighted()`.

---

## 🧰 Installation & Run

### 🔹 Clone the repository:
```bash
git clone https://github.com/Smita422/Virtual-Makeup-TryOn.git
cd Virtual-Makeup-TryOn
