import cv2
import numpy as np
import face_recognition
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="💄 Virtual Lipstick Try-On", layout="centered")
st.title("💄 Virtual Lipstick Try-On (Python 3.13 Compatible)")

lipstick_color = st.color_picker("🎨 Choose Lipstick Color", "#FF007F")
opacity = st.slider("💧 Lipstick Opacity", 0.1, 1.0, 0.6)

# Convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

rgb_color = hex_to_rgb(lipstick_color)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class LipstickProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_landmarks_list = face_recognition.face_landmarks(rgb_img)
        overlay = img.copy()

        for face_landmarks in face_landmarks_list:
            lips = face_landmarks["top_lip"] + face_landmarks["bottom_lip"]
            pts = np.array(lips, np.int32)
            cv2.fillPoly(overlay, [pts], rgb_color)

        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="lipstick",
    mode="video",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=LipstickProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

