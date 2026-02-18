import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="💄 Virtual Lipstick Try-On", layout="centered")

st.title("💋 Virtual Lipstick Try-On (Dark Shades Fixed Version)")
st.write("Real-time lipstick try-on using your camera 💄")

mode = st.radio("Choose Mode:", ["📷 Camera Mode", "🖼️ Upload Image Mode"])

mp_face_mesh = mp.solutions.face_mesh

outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270,
              409, 291, 375, 321, 405, 314, 17, 84,
              181, 91, 146]
inner_lips = [78, 95, 88, 178, 87, 14, 317, 402,
              318, 324, 308, 415, 310, 311, 312,
              13, 82, 81, 80, 191]

color_option = st.selectbox(
    "Select Lipstick Shade 💄",
    [
        "None (Natural Lips)",
        "Dark Red",
        "Deep Plum",
        "Wine",
        "Burgundy",
        "Brick Red",
        "Black Cherry",
        "Vamp Purple",
        "Chocolate Brown",
        "Midnight Black",
        "Cranberry Crush",
        "Berry Wine"
    ]
)

color_map = {
    "Dark Red": (0, 0, 200),
    "Deep Plum": (90, 0, 150),
    "Wine": (40, 0, 80),
    "Burgundy": (60, 0, 90),
    "Brick Red": (0, 0, 160),
    "Black Cherry": (10, 0, 40),
    "Vamp Purple": (70, 0, 130),
    "Chocolate Brown": (40, 20, 90),
    "Midnight Black": (5, 5, 5),
    "Cranberry Crush": (80, 0, 80),
    "Berry Wine": (120, 0, 140)
}

opacity = st.slider("💧 Lipstick Intensity", 0.2, 1.0, 0.55, 0.05)

def apply_lipstick(image, results, color, opacity, apply=True, debug=False):
    if not results.multi_face_landmarks:
        return image

    h, w, _ = image.shape
    landmarks = [(int(lm.x * w), int(lm.y * h))
                 for lm in results.multi_face_landmarks[0].landmark]

    outer_pts = np.array([landmarks[i] for i in outer_lips], np.int32)
    inner_pts = np.array([landmarks[i] for i in inner_lips], np.int32)

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [outer_pts], color)
    cv2.fillPoly(mask, [inner_pts], (0, 0, 0))

    # Stronger smoothing for natural blend
    mask = cv2.GaussianBlur(mask, (25, 25), 15)

    # Overlay lipstick with better visibility
    output = cv2.addWeighted(image, 1.0, mask, opacity, 0)

    if debug:
        for (x, y) in outer_pts:
            cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in inner_pts:
            cv2.circle(output, (x, y), 2, (0, 255, 255), -1)

    return output


# ---------------- Upload Image Mode ----------------
if mode == "🖼️ Upload Image Mode":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6
        ) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if color_option == "None (Natural Lips)":
                output = image
            else:
                color = color_map[color_option]
                output = apply_lipstick(image, results, color, opacity)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

# ---------------- Camera Mode ----------------
elif mode == "📷 Camera Mode":
    st.info("Allow camera access when asked. Use good front light for best results 💡")

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class LipstickProcessor(VideoProcessorBase):
        def __init__(self):
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.color = color_map.get(color_option, (0, 0, 0))
            self.apply = color_option != "None (Natural Lips)"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if self.apply and results.multi_face_landmarks:
                img = apply_lipstick(img, results, self.color, opacity)

            return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

    webrtc_streamer(
        key="lipstick",
        video_processor_factory=LipstickProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
