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
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
        self.depth_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # ---------------------------
        # 3. Setup Tkinter UI Layout (3 Columns x 2 Rows)
        # ---------------------------
        self.frame_row1 = tk.Frame(self.window)
        self.frame_row1.pack()
        self.frame_row2 = tk.Frame(self.window)
        self.frame_row2.pack()

        # แถวที่ 1: ภาพดิบ/ภาพที่ตีกรอบแล้ว (RGB, IR, Depth)
        self.lbl_color = tk.Label(self.frame_row1, text="RGB Waiting...")
        self.lbl_color.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir = tk.Label(self.frame_row1, text="IR Waiting...")
        self.lbl_ir.pack(side="left", padx=5, pady=5)
        
        self.lbl_depth = tk.Label(self.frame_row1, text="Depth Waiting...")
        self.lbl_depth.pack(side="left", padx=5, pady=5)

        # แถวที่ 2: ภาพ Mask สำหรับเช็คการทำงาน (RGB Mask, IR Mask, Depth Mask)
        self.lbl_color_mask = tk.Label(self.frame_row2, text="RGB Mask Waiting...")
        self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir_mask = tk.Label(self.frame_row2, text="IR Mask Waiting...")
        self.lbl_ir_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_depth_mask = tk.Label(self.frame_row2, text="Depth Mask Waiting...")
        self.lbl_depth_mask.pack(side="left", padx=5, pady=5)

        # Safe shutdown binding
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start loop
        self.update_feed()

    def update_feed(self):
        try:
            frames = self.pipeline.wait_for_frames(10)
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = frames.get_depth_frame()
            
            if color_frame and ir_frame and depth_frame:
                self.process_frames(color_frame, ir_frame, depth_frame)

        except RuntimeError:
            pass

        self.window.after(15, self.update_feed)

    def process_frames(self, color_frame, ir_frame, depth_frame):
        imageFrame = np.asanyarray(color_frame.get_data())
        ir_8bit = np.asanyarray(ir_frame.get_data())
        depth_16bit = np.asanyarray(depth_frame.get_data())

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

        # ==========================================
        # --- PROCESS DEPTH FRAME ---
        # ==========================================
        depth_8bit = cv2.convertScaleAbs(depth_16bit, alpha=0.03) 
        depth_display = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        depth_fg_mask = self.depth_backSub.apply(depth_8bit)
        _, depth_thresh = cv2.threshold(depth_fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        depth_kernel = np.ones((10, 3), np.uint8)
        depth_cleaned = cv2.morphologyEx(depth_thresh, cv2.MORPH_CLOSE, depth_kernel)

        depth_contours, _ = cv2.findContours(depth_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in depth_contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(depth_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(depth_display, "Depth", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.set_image(self.lbl_depth, depth_display, is_bgr=True)
        self.set_image(self.lbl_depth_mask, depth_cleaned, is_gray=True)

    def set_image(self, label, cv_img, is_bgr=False, is_gray=False):
        # ย่อขนาดภาพลง 50% (เหลือ 320x240) เพื่อให้เรียง 3 คอลัมน์ได้พอดีจอ
        cv_img = cv2.resize(cv_img, (320, 240))
        
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