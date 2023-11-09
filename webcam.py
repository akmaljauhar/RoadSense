from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import Detection
import cv2

model = YOLO("weights/jepang.pt")
model.predict(source="0", show=True, conf=0.4)