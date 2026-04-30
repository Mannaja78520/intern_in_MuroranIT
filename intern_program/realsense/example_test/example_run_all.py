import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# Use supported stable modes from rs-enumerate-devices
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

profile = pipeline.start(config)

print("Streaming... press q to quit")

align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)

        aligned = align.process(frames)

        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()

        if not depth or not color:
            continue

        depth_img = np.asanyarray(depth.get_data())
        color_img = np.asanyarray(color.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        combined = np.hstack((color_img, depth_colormap))
        cv2.imshow("RealSense", combined)

        # IMU
        for f in frames:
            if f.is_motion_frame():
                data = f.as_motion_frame().get_motion_data()
                if f.get_profile().stream_type() == rs.stream.accel:
                    print(f"Accel: {data.x:.2f}, {data.y:.2f}, {data.z:.2f}")
                elif f.get_profile().stream_type() == rs.stream.gyro:
                    print(f"Gyro : {data.x:.2f}, {data.y:.2f}, {data.z:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()