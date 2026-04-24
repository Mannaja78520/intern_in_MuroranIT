import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs

class RealSenseHueSubtraction:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # 1. กล้องและตัวแปรเก็บฉากหลัง
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # เก็บภาพฉากหลังในรูปแบบ HSV
        self.bg_hsv = None
        self.bg_ir = None
        
        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: {e}"); self.window.destroy(); return

        # 2. UI Layout
        self.frame_btn = tk.Frame(self.window, pady=10)
        self.frame_btn.pack()
        
        self.btn_capture = tk.Button(self.frame_btn, text="📸 Capture Static Background", 
                                     font=("Arial", 14, "bold"), bg="yellow", command=self.set_background)
        self.btn_capture.pack()

        self.frame_top = tk.Frame(self.window); self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window); self.frame_bottom.pack()

        self.lbl_color = tk.Label(self.frame_top, text="RGB Output"); self.lbl_color.pack(side="left", padx=5, pady=5)
        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Hue Diff Mask"); self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Output"); self.lbl_ir.pack(side="left", padx=5, pady=5)
        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Diff Mask"); self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def set_background(self):
        self.bg_hsv = None
        self.bg_ir = None
        print("Background Reset! Waiting to capture new background...")

    def get_red_mask(self, hsv_image):
        """ดึงเฉพาะสีแดงจากภาพ HSV"""
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

        # 2. จำฉากหลัง
        if self.bg_hsv is None or self.bg_ir is None:
            self.bg_hsv = curr_hsv.copy() # เก็บฉากหลังเป็น HSV เลย
            self.bg_ir = curr_ir_blur.copy()
            print("Background Captured Successfully!")
            return 

        # ==========================================
        # 🔴 PROCESS COLOR: HUE CIRCULAR SUBTRACTION
        # ==========================================
        # 1. แยกช่องสี H, S, V ของทั้งฉากหลังและภาพปัจจุบัน
        bg_h, bg_s, bg_v = cv2.split(self.bg_hsv)
        curr_h, curr_s, curr_v = cv2.split(curr_hsv)

        # 2. คำนวณความต่างของ Hue (Circular Distance)
        # แปลงเป็น int16 ก่อนลบเพื่อป้องกันค่าติดลบแล้วเพี้ยน (Underflow)
        diff_h = np.abs(bg_h.astype(np.int16) - curr_h.astype(np.int16))
        
        # หาวิยะทางที่สั้นที่สุดในวงกลม 180 องศา (เช่น 179 กับ 1 ห่างกันแค่ 2)
        diff_h_circular = np.minimum(diff_h, 180 - diff_h).astype(np.uint8)

        # 3. สร้าง Mask ว่า "สีเปลี่ยนไป" อย่างเห็นได้ชัด (เช่น Hue เปลี่ยนไปเกิน 15 หน่วย)
        _, hue_changed_mask = cv2.threshold(diff_h_circular, 15, 255, cv2.THRESH_BINARY)

        # 4. สร้าง Mask "สีแดง" ในภาพปัจจุบัน
        curr_red_mask = self.get_red_mask(curr_hsv)

        # 5. รวมร่าง: ต้องเป็น "สีแดง" AND "เฉดสีเปลี่ยนไปจากฉากหลัง"
        final_color_mask = cv2.bitwise_and(curr_red_mask, hue_changed_mask)
        
        kernel = np.ones((5, 3), np.uint8)
        final_color_mask = cv2.morphologyEx(final_color_mask, cv2.MORPH_CLOSE, kernel)

        # ตีกรอบ RGB
        contours_color, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_color:
            if cv2.contourArea(cnt) > 20: 
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(curr_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(curr_color, "Red Cable", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==========================================
        # 🌑 PROCESS IR
        # ==========================================
        curr_ir_display = cv2.cvtColor(curr_ir, cv2.COLOR_GRAY2BGR)
        diff_ir = cv2.absdiff(self.bg_ir, curr_ir_blur)
        _, diff_ir_mask = cv2.threshold(diff_ir, 20, 255, cv2.THRESH_BINARY) 
        final_ir_mask = cv2.morphologyEx(diff_ir_mask, cv2.MORPH_CLOSE, kernel)

        contours_ir, _ = cv2.findContours(final_ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_ir:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(curr_ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(curr_ir_display, "IR Cable", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ==========================================
        # 🖥️ นำส่งเข้า UI
        # ==========================================
        self.set_image(self.lbl_color, curr_color, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True) # โชว์ผลลัพธ์จาก Hue Subtraction
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
    root = tk.Tk(); app = RealSenseHueSubtraction(root, "RealSense - Advanced Hue Subtraction"); root.mainloop()