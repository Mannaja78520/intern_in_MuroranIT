import cv2
import os

os.makedirs("rgb_frames", exist_ok=True)
os.makedirs("ir1_frames", exist_ok=True)
os.makedirs("ir2_frames", exist_ok=True)

rgb_cap = cv2.VideoCapture("rgb.avi")
ir1_cap = cv2.VideoCapture("ir1.avi")
ir2_cap = cv2.VideoCapture("ir2.avi")

# 👉 save every 5 frame
skip = 5
i = 0
while True:
    ret1, rgb = rgb_cap.read()
    ret2, ir1 = ir1_cap.read()
    ret3, ir2 = ir2_cap.read()

    if not ret1 or not ret2 or not ret3:
        break

    if i % skip == 0:
        cv2.imwrite(f"rgb_frames/frame_{i}.png", rgb)
        cv2.imwrite(f"ir1_frames/frame_{i}.png", ir1)
        cv2.imwrite(f"ir2_frames/frame_{i}.png", ir2)

    i += 1

print("✅ Done extracting frames")
