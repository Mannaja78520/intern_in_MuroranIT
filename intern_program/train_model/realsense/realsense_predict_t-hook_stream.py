import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ---------- CONFIG ----------
model_path = "realsense_rgb.pt"   # หรือ model ที่คุณ train เอง
conf_thres = 0.8

# ---------- LOAD MODEL ----------
model = YOLO(model_path)

# ---------- REALSENSE SETUP ----------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("🚀 Start streaming... press 'q' to quit")

try:
    while True:
        # รอ frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # convert to numpy
        frame = np.asanyarray(color_frame.get_data())

        # ---------- YOLO INFERENCE ----------
        results = model(frame, conf=conf_thres)

        # ---------- DRAW RESULT ----------
        annotated_frame = results[0].plot()

        # ---------- SHOW ----------
        cv2.imshow("YOLO RealSense RGB", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()