import streamlit as st
import os
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer
import pandas as pd
import pydeck as pdk
from ultralytics import YOLO
import cv2

model_path = "weights/jepang.pt"

# Setting page layout
st.set_page_config(
    page_title="RoadSense",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Real Time", "Upload Video"],
    icons=['house-door', 'camera', 'upload'],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Home":
    st.markdown("<h1 style='text-align: center;'>Selamat Datang di Aplikasi RoadSense</h1>", unsafe_allow_html=True)

if selected == "Real Time":
    webrtc_streamer(key="key")

if selected == "Upload Video":

    uploaded_video = st.file_uploader("Pilih video", type=["mp4", "avi", "mkv"])
    uploaded_csv = st.file_uploader("Pilih csv", type=["csv"])

    if uploaded_video is not None:
        video_dir = "./video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.write("Video telah diunggah.")

        model_path = "weights/jepang.pt"
        try:
            model = YOLO(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

        if st.button('Detect Objects'):
            vid_cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model.predict(image, conf=0.4)
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                                )
                else:
                    vid_cap.release()
                    break

            if uploaded_csv is not None:
                data = pd.read_csv(uploaded_csv)
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    mid_latitude = data['latitude'].mean()
                    mid_longitude = data['longitude'].mean()
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=data,
                        get_position=["longitude", "latitude"],
                        get_radius=10,
                        get_fill_color=[255, 0, 0, 255],
                    )

                    view_state = pdk.ViewState(
                        latitude=mid_latitude,
                        longitude=mid_longitude,
                        zoom=14,
                        bearing=0,
                        pitch=0
                    )

                    map = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state
                    )

                    st.pydeck_chart(map)
                else:
                    st.error("Data CSV harus memiliki kolom 'latitude' dan 'longitude'.")
