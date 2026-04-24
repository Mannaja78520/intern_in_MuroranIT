import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import os

from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet
from utils import frame_to_bgr_image

class AstraDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # --- Recording Variables ---
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "./recordings" 
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 1. Initialize Camera
        self.pipeline = Pipeline()
        self.config = Config()
        
        try:
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            ir_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            ir_profile = ir_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(ir_profile)
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
            filename = os.path.join(self.output_dir, f"Astra_Record_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 10 FPS ตาม Log ของ Astra
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (1280, 960))
            self.is_recording = True
            self.btn_record.config(text="Stop Recording", bg="red")
            print(f"Astra Recording started: {filename}")
        else:
            self.stop_recording_logic()

    def stop_recording_logic(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.btn_record.config(text="Start Recording", bg="green")
        print("Astra Recording stopped.")

    def update_feed(self):
        try:
            frames = self.pipeline.wait_for_frames(100) # เพิ่ม timeout เป็น 100ms
            if frames:
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_ir_frame()
                if color_frame and ir_frame:
                    self.process_frames(color_frame, ir_frame)
        except: pass
        self.window.after(20, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        img_color = frame_to_bgr_image(color_frame)
        if img_color is None: return
        img_color = cv2.resize(img_color, (640, 480))
        
        fg_mask = self.backSub.apply(img_color)
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255])),
                                  cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255])))
        final_color_mask = cv2.morphologyEx(cv2.bitwise_and(mask_red, fg_mask), cv2.MORPH_CLOSE, np.ones((10,3), np.uint8))

        ir_data = ir_frame.get_data()
        ir_height, ir_width = ir_frame.get_height(), ir_frame.get_width()
        ir_np = np.frombuffer(ir_data, dtype=np.uint16).reshape((ir_height, ir_width))
        ir_8bit = cv2.resize(np.clip(ir_np / 16, 0, 255).astype(np.uint8), (640, 480))
        
        ir_fg_mask = self.ir_backSub.apply(ir_8bit)
        _, ir_thresh = cv2.threshold(ir_fg_mask, 100, 255, cv2.THRESH_BINARY)
        ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, np.ones((10,3), np.uint8))

        if self.is_recording and self.video_writer:
            top = np.hstack((img_color, cv2.cvtColor(final_color_mask, cv2.COLOR_GRAY2BGR)))
            bottom = np.hstack((cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ir_cleaned, cv2.COLOR_GRAY2BGR)))
            self.video_writer.write(np.vstack((top, bottom)))

        self.set_image(self.lbl_color, img_color, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)
        self.set_image(self.lbl_ir, ir_8bit, is_gray=True)
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
    app = AstraDashboard(root, "Astra Pro - Dashboard")
    root.mainloop()