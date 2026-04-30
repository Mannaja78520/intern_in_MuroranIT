import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs
import os
from datetime import datetime


class RealSenseDashboardRecorder:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.recording = False
        self.writers = {}
        self.output_dir = "recordings"

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error initializing streams: {e}")
            self.window.destroy()
            return

        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.ir_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # --- UI Layout ---
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window)
        self.frame_bottom.pack()
        self.frame_controls = tk.Frame(self.window)
        self.frame_controls.pack(pady=5)

        self.lbl_color = tk.Label(self.frame_top, text="RGB Waiting...")
        self.lbl_color.pack(side="left", padx=5, pady=5)

        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Mask Waiting...")
        self.lbl_color_mask.pack(side="left", padx=5, pady=5)

        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Waiting...")
        self.lbl_ir.pack(side="left", padx=5, pady=5)

        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Mask Waiting...")
        self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        # --- Combined 2x2 view ---
        self.frame_combined = tk.Frame(self.window)
        self.frame_combined.pack()
        self.lbl_combined = tk.Label(self.frame_combined, text="Combined Waiting...")
        self.lbl_combined.pack(padx=5, pady=5)

        # --- Record Button ---
        self.btn_record = tk.Button(
            self.frame_controls, text="Start Recording",
            bg="green", fg="white", font=("Arial", 12, "bold"),
            command=self.toggle_recording, width=20
        )
        self.btn_record.pack(side="left", padx=10)

        self.lbl_status = tk.Label(self.frame_controls, text="Not Recording", fg="gray", font=("Arial", 10))
        self.lbl_status.pack(side="left", padx=10)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0
        size     = (640, 480)
        size_2x2 = (1280, 960)

        self.writers = {
            "rgb":      cv2.VideoWriter(os.path.join(session_dir, "rgb.mp4"),      fourcc, fps, size),
            "rgb_mask": cv2.VideoWriter(os.path.join(session_dir, "rgb_mask.mp4"), fourcc, fps, size),
            "ir":       cv2.VideoWriter(os.path.join(session_dir, "ir.mp4"),       fourcc, fps, size),
            "ir_mask":  cv2.VideoWriter(os.path.join(session_dir, "ir_mask.mp4"),  fourcc, fps, size),
            "combined": cv2.VideoWriter(os.path.join(session_dir, "combined.mp4"), fourcc, fps, size_2x2),
        }

        self.recording = True
        self.btn_record.config(text="Stop Recording", bg="red")
        self.lbl_status.config(text=f"Recording → {session_dir}/", fg="red")
        print(f"Recording started: {session_dir}/")

    def stop_recording(self):
        self.recording = False
        for writer in self.writers.values():
            writer.release()
        self.writers = {}
        self.btn_record.config(text="Start Recording", bg="green")
        self.lbl_status.config(text="Not Recording", fg="gray")
        print("Recording stopped. Videos saved.")

    def update_feed(self):
        try:
            frames = self.pipeline.wait_for_frames(10)
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)

            if color_frame and ir_frame:
                self.process_frames(color_frame, ir_frame)

        except RuntimeError:
            pass

        self.window.after(15, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        imageFrame = np.asanyarray(color_frame.get_data())
        ir_8bit = np.asanyarray(ir_frame.get_data())

        # --- Color processing ---
        fg_mask = self.backSub.apply(imageFrame)
        hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0, 80, 40]),   np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255]))
        color_mask = cv2.bitwise_or(mask1, mask2)

        combined_color_mask = cv2.bitwise_and(color_mask, fg_mask)
        kernel = np.ones((10, 3), np.uint8)
        final_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 20:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         if h > w * 1.5 or w > h * 1.5:
        #             cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #             cv2.putText(imageFrame, "RGB", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- IR processing ---
        ir_display = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)
        ir_fg_mask = self.ir_backSub.apply(ir_8bit)
        _, ir_thresh = cv2.threshold(ir_fg_mask, 100, 255, cv2.THRESH_BINARY)
        ir_kernel = np.ones((10, 3), np.uint8)
        ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, ir_kernel)

        ir_contours, _ = cv2.findContours(ir_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in ir_contours:
        #     if cv2.contourArea(cnt) > 20:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         if h > w * 1.5 or w > h * 1.5:
        #             cv2.rectangle(ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #             cv2.putText(ir_display, "IR", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --- Build combined 2x2 frame (each quadrant full 640x480 → total 1280x960) ---
        rgb_mask_bgr = cv2.cvtColor(final_color_mask, cv2.COLOR_GRAY2BGR)
        ir_mask_bgr  = cv2.cvtColor(ir_cleaned,       cv2.COLOR_GRAY2BGR)

        def labeled(img, title):
            out = img.copy()
            cv2.putText(out, title, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            return out

        combined = np.vstack([
            np.hstack([labeled(imageFrame,   "RGB"),      labeled(rgb_mask_bgr, "RGB Mask")]),
            np.hstack([labeled(ir_display,   "IR"),       labeled(ir_mask_bgr,  "IR Mask")]),
        ])

        # Resize combined down for tkinter display (1280x960 → 640x480)
        combined_display = cv2.resize(combined, (640, 480))

        # --- Write frames to video ---
        if self.recording:
            self.writers["rgb"].write(imageFrame)
            self.writers["rgb_mask"].write(rgb_mask_bgr)
            self.writers["ir"].write(ir_display)
            self.writers["ir_mask"].write(ir_mask_bgr)
            self.writers["combined"].write(combined)

        # --- Update Tkinter labels ---
        self.set_image(self.lbl_color,      imageFrame,       is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)
        self.set_image(self.lbl_ir,         ir_display,       is_bgr=True)
        self.set_image(self.lbl_ir_mask,    ir_cleaned,       is_gray=True)
        self.set_image(self.lbl_combined,   combined_display, is_bgr=True)

    def set_image(self, label, cv_img, is_bgr=False, is_gray=False):
        if is_bgr:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif is_gray:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def on_closing(self):
        if self.recording:
            self.stop_recording()
        print("Shutting down RealSense Pipeline...")
        self.pipeline.stop()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseDashboardRecorder(root, "RealSense - Cable Tracking (Recorder)")
    root.mainloop()
