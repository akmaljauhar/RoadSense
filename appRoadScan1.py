import streamlit as st
import os
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer
import pandas as pd
import pydeck as pdk
from ultralytics import YOLO
import cv2

# Replace the relative path to your weight file
model_path = "weights/jepang.pt"

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

uploaded_video = st.file_uploader("Pilih video", type=["mp4", "avi", "mkv"])

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# st.write("Model loaded successfully!")

if uploaded_video is not None:
        video_dir = "./video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.write("Video telah diunggah.")

        with open(str(video_path), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button('Detect Objects'):
            vid_cap = cv2.VideoCapture(
                video_path)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model.predict(image, conf=confidence)
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                                )
                else:
                    vid_cap.release()
                    break