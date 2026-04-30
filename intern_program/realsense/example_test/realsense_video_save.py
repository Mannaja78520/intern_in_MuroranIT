import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = rs.pipeline()
config = rs.config()

# เปิด 3 stream
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(timestamp, exist_ok=True)

rgb_out = cv2.VideoWriter(f'{timestamp}/rgb.avi', fourcc, FPS, (WIDTH, HEIGHT))
ir1_out = cv2.VideoWriter(f'{timestamp}/ir1.avi', fourcc, FPS, (WIDTH, HEIGHT), False)
ir2_out = cv2.VideoWriter(f'{timestamp}/ir2.avi', fourcc, FPS, (WIDTH, HEIGHT), False)

print("🎬 Recording RGB + IR1 + IR2... press q to stop")

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)

        if not color_frame or not ir1_frame or not ir2_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        ir1 = np.asanyarray(ir1_frame.get_data())
        ir2 = np.asanyarray(ir2_frame.get_data())

        # save
        rgb_out.write(color)
        ir1_out.write(ir1)
        ir2_out.write(ir2)

        # preview
        ir1_vis = cv2.cvtColor(ir1, cv2.COLOR_GRAY2BGR)
        ir2_vis = cv2.cvtColor(ir2, cv2.COLOR_GRAY2BGR)

        color = cv2.resize(color, (320, 240))
        ir1_vis = cv2.resize(ir1_vis, (320, 240))
        ir2_vis = cv2.resize(ir2_vis, (320, 240))
        combined = np.hstack((color, ir1_vis, ir2_vis))
        cv2.imshow("RGB | IR1 | IR2", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    rgb_out.release()
    ir1_out.release()
    ir2_out.release()
    cv2.destroyAllWindows()