import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs
import datetime
import os

class RealSenseDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # --- Recording Variables ---
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "./recordings" 
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 1. Initialize RealSense Camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: {e}")
            self.window.destroy()
            return

        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.ir_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # 3. UI Layout
        self.frame_controls = tk.Frame(self.window)
        self.frame_controls.pack(pady=5)
        self.btn_record = tk.Button(self.frame_controls, text="Start Recording", command=self.toggle_recording, bg="green", fg="white")
        self.btn_record.pack()

        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window)
        self.frame_bottom.pack()

        self.lbl_color = tk.Label(self.frame_top, text="RGB Waiting...")
        self.lbl_color.pack(side="left", padx=5, pady=5)
        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Mask Waiting...")
        self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Waiting...")
        self.lbl_ir.pack(side="left", padx=5, pady=5)
        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Mask Waiting...")
        self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def toggle_recording(self):
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"Realsense_Record_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 20 FPS สำหรับ RealSense
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 960))
            self.is_recording = True
            self.btn_record.config(text="Stop Recording", bg="red")
            print(f"RealSense Recording started: {filename}")
        else:
            self.stop_recording_logic()

    def stop_recording_logic(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.btn_record.config(text="Start Recording", bg="green")
        print("RealSense Recording stopped.")

    def update_feed(self):
        try:
            frames = self.pipeline.wait_for_frames(100) # เพิ่ม timeout เป็น 100ms
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            if color_frame and ir_frame:
                self.process_frames(color_frame, ir_frame)
        except RuntimeError: pass
        self.window.after(20, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        img_color = np.asanyarray(color_frame.get_data()).copy()
        ir_raw = np.asanyarray(ir_frame.get_data())

        fg_mask = self.backSub.apply(img_color)
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255])),
                                  cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255])))
        final_color_mask = cv2.morphologyEx(cv2.bitwise_and(mask_red, fg_mask), cv2.MORPH_CLOSE, np.ones((10, 3), np.uint8))
        
        img_ir = cv2.cvtColor(ir_raw, cv2.COLOR_GRAY2BGR)
        ir_fg_mask = self.ir_backSub.apply(ir_raw)
        _, ir_thresh = cv2.threshold(ir_fg_mask, 100, 255, cv2.THRESH_BINARY)
        ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, np.ones((10, 3), np.uint8))

        if self.is_recording and self.video_writer:
            top = np.hstack((img_color, cv2.cvtColor(final_color_mask, cv2.COLOR_GRAY2BGR)))
            bottom = np.hstack((img_ir, cv2.cvtColor(ir_cleaned, cv2.COLOR_GRAY2BGR)))
            self.video_writer.write(np.vstack((top, bottom)))

        self.set_image(self.lbl_color, img_color, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)
        self.set_image(self.lbl_ir, img_ir, is_bgr=True)
        self.set_image(self.lbl_ir_mask, ir_cleaned, is_gray=True)

    def set_image(self, label, cv_img, is_bgr=False, is_gray=False):
        if is_bgr: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif is_gray: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def on_closing(self):
        self.stop_recording_logic()
        self.pipeline.stop()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseDashboard(root, "RealSense - Dashboard")
    root.mainloop()