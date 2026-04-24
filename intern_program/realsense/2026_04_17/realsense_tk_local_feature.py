import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs
import time

class RealSenseHueSubtraction:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.last_color = None
        self.last_color_mask = None
        self.last_ir = None
        self.last_ir_mask = None
        
        self.bg_hsv = None
        self.bg_ir = None
        
        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: {e}"); self.window.destroy(); return

        self.frame_btn = tk.Frame(self.window, pady=10)
        self.frame_btn.pack()
        
        self.btn_capture = tk.Button(self.frame_btn, text="Capture Static Background", 
                                     font=("Arial", 12, "bold"), bg="yellow", command=self.set_background)
        self.btn_capture.pack(side="left", padx=5)

        self.btn_save = tk.Button(self.frame_btn, text="Save Current Masks", 
                                  font=("Arial", 12, "bold"), bg="cyan", command=self.save_dataset)
        self.btn_save.pack(side="left", padx=5)

        self.frame_top = tk.Frame(self.window); self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window); self.frame_bottom.pack()

        self.lbl_color = tk.Label(self.frame_top, text="RGB Output"); self.lbl_color.pack(side="left", padx=5, pady=5)
        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Hue Diff Mask"); self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Output"); self.lbl_ir.pack(side="left", padx=5, pady=5)
        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Diff Mask"); self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def save_dataset(self):
        if self.last_color_mask is not None:
            ts = int(time.time())
            cv2.imwrite(f"color_raw_{ts}.png", self.last_color)
            cv2.imwrite(f"color_mask_{ts}.png", self.last_color_mask)
            cv2.imwrite(f"ir_raw_{ts}.png", self.last_ir)
            cv2.imwrite(f"ir_mask_{ts}.png", self.last_ir_mask)
            print(f"Saved dataset with timestamp: {ts}")
        else:
            print("No data to save!")

    def set_background(self):
        self.bg_hsv = None
        self.bg_ir = None
        print("Background Reset! Waiting to capture new background...")

    def get_red_mask(self, hsv_image):
        mask1 = cv2.inRange(hsv_image, np.array([0, 80, 40]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_image, np.array([160, 80, 40]), np.array([180, 255, 255]))
        return cv2.bitwise_or(mask1, mask2)

    def update_feed(self):
        try:
            frames = self.pipeline.wait_for_frames(10)
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            if color_frame and ir_frame:
                self.process_frames(color_frame, ir_frame)
        except RuntimeError: pass 
        self.window.after(15, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        curr_color = np.asanyarray(color_frame.get_data())
        curr_ir = np.asanyarray(ir_frame.get_data())
        
        curr_color_blur = cv2.GaussianBlur(curr_color, (5, 5), 0)
        curr_ir_blur = cv2.GaussianBlur(curr_ir, (5, 5), 0)
        curr_hsv = cv2.cvtColor(curr_color_blur, cv2.COLOR_BGR2HSV)

        if self.bg_hsv is None or self.bg_ir is None:
            self.bg_hsv = curr_hsv.copy()
            self.bg_ir = curr_ir_blur.copy()
            print("Background Captured Successfully!")
            return 

        # ==========================================
        # 🔴 COLOR: Denoise & Masking
        # ==========================================
        bg_h, _, _ = cv2.split(self.bg_hsv)
        curr_h, _, _ = cv2.split(curr_hsv)

        diff_h = np.abs(bg_h.astype(np.int16) - curr_h.astype(np.int16))
        diff_h_circular = np.minimum(diff_h, 180 - diff_h).astype(np.uint8)

        _, hue_changed_mask = cv2.threshold(diff_h_circular, 15, 255, cv2.THRESH_BINARY)
        curr_red_mask = self.get_red_mask(curr_hsv)
        
        # 1. รวม Mask เริ่มต้น
        raw_color_mask = cv2.bitwise_and(curr_red_mask, hue_changed_mask)

        # 2. 🔥 Denoising: ลบ Noise จุดเล็กๆ (Salt-and-pepper)
        denoised_color = cv2.medianBlur(raw_color_mask, 5) 
        
        # 3. 🔥 Erode (ลบติ่ง) ก่อนแล้วค่อย Close (เชื่อมรู)
        kernel = np.ones((5, 3), np.uint8)
        final_color_mask = cv2.erode(denoised_color, kernel, iterations=1)
        final_color_mask = cv2.morphologyEx(final_color_mask, cv2.MORPH_CLOSE, kernel)

        self.last_color = curr_color.copy()
        self.last_color_mask = final_color_mask.copy()

        contours_color, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_color:
            if cv2.contourArea(cnt) > 50: # ปรับพื้นที่ขั้นต่ำเพิ่มเล็กน้อยกันสั่น
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.2 or w > h * 1.2:
                    cv2.rectangle(curr_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ==========================================
        # 🌑 IR: Denoise & Masking
        # ==========================================
        curr_ir_display = cv2.cvtColor(curr_ir, cv2.COLOR_GRAY2BGR)
        diff_ir = cv2.absdiff(self.bg_ir, curr_ir_blur)
        _, diff_ir_mask = cv2.threshold(diff_ir, 20, 255, cv2.THRESH_BINARY) 

        # 🔥 Denoising IR
        denoised_ir = cv2.medianBlur(diff_ir_mask, 5)
        final_ir_mask = cv2.erode(denoised_ir, kernel, iterations=1)
        final_ir_mask = cv2.morphologyEx(final_ir_mask, cv2.MORPH_CLOSE, kernel)

        self.last_ir = curr_ir.copy()
        self.last_ir_mask = final_ir_mask.copy()

        contours_ir, _ = cv2.findContours(final_ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_ir:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.2 or w > h * 1.2:
                    cv2.rectangle(curr_ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)

        self.set_image(self.lbl_color, curr_color, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)
        self.set_image(self.lbl_ir, curr_ir_display, is_bgr=True)
        self.set_image(self.lbl_ir_mask, final_ir_mask, is_gray=True)

    def set_image(self, label, cv_img, is_bgr=False, is_gray=False):
        cv_img = cv2.resize(cv_img, (320, 240))
        if is_bgr: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif is_gray: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk; label.configure(image=imgtk)

    def on_closing(self):
        self.pipeline.stop(); self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk(); app = RealSenseHueSubtraction(root, "RealSense - Denoised Masking"); root.mainloop()