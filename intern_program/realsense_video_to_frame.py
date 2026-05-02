import cv2
import os

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "videos", "realsense", "recordings")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pictures", "realsense")

# Save every N-th frame
SKIP = 5


def extract_frames(video_path, out_dir, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] cannot open {label}")
        return

    os.makedirs(out_dir, exist_ok=True)
    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % SKIP == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.png"), frame)
            saved += 1
        i += 1

    cap.release()
    print(f"  {label}: {saved} frames saved  ({i} total)")


for timestamp in sorted(os.listdir(RECORDINGS_DIR)):
    rec_path = os.path.join(RECORDINGS_DIR, timestamp)
    if not os.path.isdir(rec_path):
        continue

    rgb_video = os.path.join(rec_path, "rgb.mp4")
    ir_video = os.path.join(rec_path, "ir.mp4")

    has_rgb = os.path.exists(rgb_video)
    has_ir = os.path.exists(ir_video)

    if not has_rgb and not has_ir:
        print(f"[skip] {timestamp} — no rgb.mp4 or ir.mp4")
        continue

    print(f"Processing {timestamp} ...")

    if has_rgb:
        extract_frames(rgb_video, os.path.join(OUTPUT_DIR, timestamp, "rgb"), "rgb")

    if has_ir:
        extract_frames(ir_video, os.path.join(OUTPUT_DIR, timestamp, "ir"), "ir")

print("Done.")
