import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="💄 Virtual Lipstick Try-On", layout="centered")

st.title("💋 Virtual Lipstick Try-On (Dark & Natural Shades)")
st.write("Try realistic dark lipstick shades or go **natural** using your camera or uploaded image 💄")

# Mode selection
mode = st.radio("Choose Mode:", ["📷 Camera Mode", "🖼️ Upload Image Mode"])

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Lip landmarks
outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270,
              409, 291, 375, 321, 405, 314, 17, 84,
              181, 91, 146]
inner_lips = [78, 95, 88, 178, 87, 14, 317, 402,
              318, 324, 308, 415, 310, 311, 312,
              13, 82, 81, 80, 191]

# Lipstick shades
color_option = st.selectbox(
    "Select Lipstick Shade 💄",
    [
        "None (Natural Lips)",
        "Dark Red",
        "Dark Maroon",
        "Dark Pink",
        "Wine",
        "Deep Plum",
        "Burgundy",
        "Brick Red",
        "Black Cherry",
        "Chocolate Brown"
    ]
)

color_map = {
    "Dark Red": (0, 0, 139),
    "Dark Maroon": (0, 0, 90),
    "Dark Pink": (147, 20, 255),
    "Wine": (35, 0, 75),
    "Deep Plum": (90, 0, 120),
    "Burgundy": (50, 0, 70),
    "Brick Red": (10, 10, 160),
    "Black Cherry": (10, 0, 40),
    "Chocolate Brown": (30, 30, 100)
}

# Function to apply lipstick
def apply_lipstick(image, color, apply=True):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    face_mesh.close()

    if not results.multi_face_landmarks or not apply:
        return image

    h, w, _ = image.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

    mask = np.zeros_like(image)
    outer_pts = np.array([landmarks[i] for i in outer_lips], np.int32)
    inner_pts = np.array([landmarks[i] for i in inner_lips], np.int32)

    cv2.fillPoly(mask, [outer_pts], color)
    cv2.fillPoly(mask, [inner_pts], (0, 0, 0))

    return cv2.addWeighted(image, 1, mask, 0.45, 0)

# ---------------- Upload Image Mode ----------------
if mode == "🖼️ Upload Image Mode":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if color_option == "None (Natural Lips)":
            output = image
        else:
            color = color_map[color_option]
            output = apply_lipstick(image, color)

        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

# ---------------- Real-Time Camera Mode (WebRTC) ----------------
elif mode == "📷 Camera Mode":
    st.info("Allow camera access when asked. Works on Streamlit Cloud & mobile browsers 📱")

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class LipstickProcessor(VideoProcessorBase):
        def __init__(self):
            self.color = color_map.get(color_option, (0, 0, 0))
            self.apply = color_option != "None (Natural Lips)"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            if self.apply:
                img = apply_lipstick(img, self.color)
            return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

    webrtc_streamer(
        key="lipstick",
        video_processor_factory=LipstickProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
