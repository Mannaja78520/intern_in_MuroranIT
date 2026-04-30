"""
Review a recorded session: plays rgb, rgb_mask, ir, ir_mask in a single 2x2 window.

Usage:
    python review_recording.py                        # pick latest session
    python review_recording.py recordings/20260427_143022
    python review_recording.py recordings/20260427_143022/combined.mp4  # play combined only

Controls:
    Space       pause / resume
    q / Esc     quit
    d           step forward 1 frame (while paused)
    r           restart from beginning
"""

import cv2
import numpy as np
import os
import sys


RECORDINGS_DIR = "recordings"
VIDEO_NAMES = ["rgb.mp4", "rgb_mask.mp4", "ir.mp4", "ir_mask.mp4"]
LABELS      = ["RGB",     "RGB Mask",     "IR",     "IR Mask"]


def find_latest_session(base_dir):
    sessions = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    if not sessions:
        raise FileNotFoundError(f"No session folders found in '{base_dir}'")
    return os.path.join(base_dir, sessions[-1])


def labeled(frame, title):
    cv2.putText(frame, title, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def play_combined(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return

    paused = False
    print(f"Playing combined: {path}")
    print("Space=pause  d=step  r=restart  q/Esc=quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        cv2.imshow("Review - Combined", frame)
        key = cv2.waitKey(33) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') and paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False

    cap.release()
    cv2.destroyAllWindows()


def play_session(session_dir):
    caps = []
    for name in VIDEO_NAMES:
        path = os.path.join(session_dir, name)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Cannot open: {path}")
            for c in caps:
                c.release()
            return
        caps.append(cap)

    paused = False
    frame  = None
    print(f"Playing session: {session_dir}")
    print("Space=pause  d=step  r=restart  q/Esc=quit")

    while True:
        if not paused:
            frames = []
            end_of_any = False
            for cap in caps:
                ret, f = cap.read()
                if not ret:
                    end_of_any = True
                    break
                frames.append(f)

            if end_of_any:
                for cap in caps:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Ensure all frames are 640x480 BGR
            resized = []
            for f in frames:
                if f.shape[:2] != (480, 640):
                    f = cv2.resize(f, (640, 480))
                if len(f.shape) == 2:
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                resized.append(f)

            panels = [labeled(resized[i].copy(), LABELS[i]) for i in range(4)]
            frame = np.vstack([
                np.hstack([panels[0], panels[1]]),
                np.hstack([panels[2], panels[3]]),
            ])

        if frame is not None:
            cv2.imshow("Review - 4 Streams", frame)

        key = cv2.waitKey(33) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') and paused:
            frames = []
            for cap in caps:
                ret, f = cap.read()
                if ret:
                    frames.append(f)
            if len(frames) == 4:
                resized = []
                for f in frames:
                    if f.shape[:2] != (480, 640):
                        f = cv2.resize(f, (640, 480))
                    if len(f.shape) == 2:
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    resized.append(f)
                panels = [labeled(resized[i].copy(), LABELS[i]) for i in range(4)]
                frame = np.vstack([
                    np.hstack([panels[0], panels[1]]),
                    np.hstack([panels[2], panels[3]]),
                ])
        elif key == ord('r'):
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None:
        session = find_latest_session(RECORDINGS_DIR)
        print(f"No path given — using latest session: {session}")
        play_session(session)
    elif arg.endswith(".mp4"):
        play_combined(arg)
    elif os.path.isdir(arg):
        combined_path = os.path.join(arg, "combined.mp4")
        if all(os.path.exists(os.path.join(arg, n)) for n in VIDEO_NAMES):
            play_session(arg)
        elif os.path.exists(combined_path):
            play_combined(combined_path)
        else:
            print(f"No recognisable videos found in: {arg}")
    else:
        print(f"Path not found: {arg}")
