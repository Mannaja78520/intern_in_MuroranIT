import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs

class RealSenseDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # ---------------------------
        # 1. Initialize RealSense Camera
        # ---------------------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        try:
            # Request Color and IR streams
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error initializing streams: {e}")
            self.window.destroy()
            return

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
            frames = self.pipeline.wait_for_frames(10) # 10ms timeout
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            
            if color_frame and ir_frame:
                self.process_frames(color_frame, ir_frame)

        except RuntimeError:
            pass # Ignore read timeouts from RealSense

        # Re-call this function every 15 milliseconds
        self.window.after(15, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        # Convert directly to numpy arrays
        imageFrame = np.asanyarray(color_frame.get_data())
        ir_8bit = np.asanyarray(ir_frame.get_data())

        # ==========================================
        # --- PROCESS COLOR FRAME ---
        # ==========================================
        fg_mask = self.backSub.apply(imageFrame)

        hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 80, 40]) 
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red2 = np.array([160, 80, 40])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        combined_color_mask = cv2.bitwise_and(color_mask, fg_mask)

        kernel = np.ones((10, 3), np.uint8)
        final_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(imageFrame, "RGB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.set_image(self.lbl_color, imageFrame, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)

        # ==========================================
        # --- PROCESS IR FRAME ---
        # ==========================================
        ir_display = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)
        ir_fg_mask = self.ir_backSub.apply(ir_8bit)

        _, ir_thresh = cv2.threshold(ir_fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        ir_kernel = np.ones((10, 3), np.uint8)
        ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, ir_kernel)

        ir_contours, _ = cv2.findContours(ir_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in ir_contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(ir_display, "IR", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        self.set_image(self.lbl_ir, ir_display, is_bgr=True)
        self.set_image(self.lbl_ir_mask, ir_cleaned, is_gray=True)

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
        print("Shutting down RealSense Pipeline...")
        self.pipeline.stop()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseDashboard(root, "RealSense - Cable Tracking")
    root.mainloop()