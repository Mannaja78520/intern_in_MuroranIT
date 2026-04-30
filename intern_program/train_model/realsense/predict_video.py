import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "/home/mannaja/intern_in_MuroranIT/intern_program/models/realsense_merged.pt"
CONF_THRES = 0.8


def predict_video(video_path: str, conf: float = CONF_THRES, show: bool = True):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(video_path).stem + "_predicted.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Model  : {MODEL_PATH}")
    print(f"Input  : {video_path}  ({width}x{height} @ {fps:.1f} fps, {total} frames)")
    print(f"Output : {out_path}")
    print("Processing... press 'q' to stop early.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        writer.write(annotated)
        frame_idx += 1

        if total > 0:
            pct = frame_idx / total * 100
            print(f"\r  {frame_idx}/{total} ({pct:.1f}%)", end="", flush=True)

        if show:
            cv2.imshow("Prediction", annotated)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone. Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on a video using realsense_merged.pt")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable live preview window")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    predict_video(args.video, conf=args.conf, show=args.show)
