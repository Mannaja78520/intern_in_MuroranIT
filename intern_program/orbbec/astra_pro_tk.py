import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

class AstraDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # ---------------------------
        # 1. Initialize Camera
        # ---------------------------
        self.config = Config()
        self.pipeline = Pipeline()
        
        try:
            # Color Profile
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = color_profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
            except OBError:
                color_profile = color_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            # IR Profile
            ir_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            try:
                ir_profile = ir_profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
            except OBError:
                ir_profile = ir_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(ir_profile)

        except Exception as e:
            print(f"Error initializing streams: {e}")
            self.window.destroy()
            return
            
        self.pipeline.start(self.config)

        # ---------------------------
        # 2. Setup Vision Parameters
        # ---------------------------
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.ir_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # ---------------------------
        # 3. Setup Tkinter UI Layout (2x2 Grid)
        # ---------------------------
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window)
        self.frame_bottom.pack()

        # Labels to hold the images
        self.lbl_color = tk.Label(self.frame_top, text="RGB Waiting...")
        self.lbl_color.pack(side="left", padx=5, pady=5)
        
        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Mask Waiting...")
        self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Waiting...")
        self.lbl_ir.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Mask Waiting...")
        self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        # Safe shutdown binding
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start loop
        self.update_feed()

    def update_feed(self):
        try:
            frames: FrameSet = self.pipeline.wait_for_frames(10) # Short timeout so UI doesn't freeze
            if frames is not None:
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_ir_frame()
                
                if color_frame and ir_frame:
                    self.process_frames(color_frame, ir_frame)

        except Exception as e:
            pass # Ignore read timeouts

        # Re-call this function every 15 milliseconds
        self.window.after(15, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        # ==========================================
        # --- PROCESS COLOR FRAME ---
        # ==========================================
        imageFrame = frame_to_bgr_image(color_frame)
        if imageFrame is not None:
            imageFrame = cv2.resize(imageFrame, (640, 480)) # Resize for UI
            fg_mask = self.backSub.apply(imageFrame)

            hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            # WIDENED range for distant, darker red
            lower_red = np.array([0, 80, 40]) 
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red2 = np.array([160, 80, 40])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            color_mask = cv2.bitwise_or(mask1, mask2)

            combined_color_mask = cv2.bitwise_and(color_mask, fg_mask)

            # SMALLER kernel to preserve thin distant cables
            kernel = np.ones((10, 3), np.uint8)
            final_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 20: # LOWER area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > w * 1.5 or w > h * 1.5:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(imageFrame, "RGB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert for Tkinter
            self.set_image(self.lbl_color, imageFrame, is_bgr=True)
            self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)

        # ==========================================
        # --- PROCESS IR FRAME ---
        # ==========================================
        ir_format = ir_frame.get_format()
        ir_height = ir_frame.get_height()
        ir_width = ir_frame.get_width()
        ir_data = ir_frame.get_data()

        if ir_format == OBFormat.Y16:
            ir_np = np.frombuffer(ir_data, dtype=np.uint16).reshape((ir_height, ir_width))
            ir_8bit = np.clip(ir_np / 16, 0, 255).astype(np.uint8) 
        elif ir_format == OBFormat.Y8:
            ir_8bit = np.frombuffer(ir_data, dtype=np.uint8).reshape((ir_height, ir_width))

        ir_display = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)
        ir_display = cv2.resize(ir_display, (640, 480))
        ir_8bit_resized = cv2.resize(ir_8bit, (640, 480))

        ir_fg_mask = self.ir_backSub.apply(ir_8bit_resized)

        # LOWER threshold to catch weak reflections >1m away
        _, ir_thresh = cv2.threshold(ir_fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        ir_kernel = np.ones((10, 3), np.uint8) # SMALLER kernel
        ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, ir_kernel)

        ir_contours, _ = cv2.findContours(ir_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in ir_contours:
            if cv2.contourArea(cnt) > 20: # LOWER area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(ir_display, "IR", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Convert for Tkinter
        self.set_image(self.lbl_ir, ir_display, is_bgr=True)
        self.set_image(self.lbl_ir_mask, ir_cleaned, is_gray=True)

    def set_image(self, label, cv_img, is_bgr=False, is_gray=False):
        """Helper to convert OpenCV image to Tkinter Image"""
        if is_bgr:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif is_gray:
            # Tkinter needs 3 channels or specific handling, simplest is to convert gray to RGB
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk # Keep reference to prevent garbage collection
        label.configure(image=imgtk)

    def on_closing(self):
        print("Shutting down Pipeline...")
        self.pipeline.stop()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AstraDashboard(root, "Astra Pro - Cable Tracking")
    root.mainloop()