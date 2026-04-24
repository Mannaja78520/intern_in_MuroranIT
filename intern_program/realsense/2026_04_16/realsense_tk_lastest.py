import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs

class RealSenseStaticBackground:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # ตัวแปรเก็บภาพพื้นหลังตอนแรก
        self.bg_color = None
        self.bg_ir = None
        
        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: {e}")
            self.window.destroy()
            return

        # UI Layout
        self.frame_btn = tk.Frame(self.window, pady=10)
        self.frame_btn.pack()
        
        self.btn_capture = tk.Button(self.frame_btn, text="Capture Static Background", 
                                     font=("Arial", 14, "bold"), bg="yellow", command=self.set_background)
        self.btn_capture.pack()

        self.frame_top = tk.Frame(self.window); self.frame_top.pack()
        self.frame_bottom = tk.Frame(self.window); self.frame_bottom.pack()

        self.lbl_color = tk.Label(self.frame_top, text="RGB Output"); self.lbl_color.pack(side="left", padx=5, pady=5)
        self.lbl_color_mask = tk.Label(self.frame_top, text="RGB Mask"); self.lbl_color_mask.pack(side="left", padx=5, pady=5)
        
        self.lbl_ir = tk.Label(self.frame_bottom, text="IR Output"); self.lbl_ir.pack(side="left", padx=5, pady=5)
        self.lbl_ir_mask = tk.Label(self.frame_bottom, text="IR Mask"); self.lbl_ir_mask.pack(side="left", padx=5, pady=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def set_background(self):
        self.bg_color = None
        self.bg_ir = None
        print("Background Reset! Waiting to capture new background...")

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
        
        # เบลอเพื่อลด Noise เม็ดเล็กๆ
        curr_color_blur = cv2.GaussianBlur(curr_color, (5, 5), 0)
        curr_ir_blur = cv2.GaussianBlur(curr_ir, (5, 5), 0)

        # จำฉากหลัง
        if self.bg_color is None or self.bg_ir is None:
            self.bg_color = curr_color_blur.copy()
            self.bg_ir = curr_ir_blur.copy()
            print("Background Captured Successfully!")
            return 

        # ==========================================
        # 🔥 PROCESS COLOR: โฟกัสจับเชือกย้อนแสง
        # ==========================================
        # 1. หาว่ามีอะไรเปลี่ยนไปบ้างจากฉากหลัง
        diff_color = cv2.absdiff(self.bg_color, curr_color_blur)
        diff_gray = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        
        # 2. ดึงค่าสีแดงแบบ "LOOSE" (ยอมรับสีแดงที่มืดและหม่นลงจากการย้อนแสง)
        # ลด Saturation จาก 80 เหลือ 40, ลด Value จาก 40 เหลือ 20
        hsv = cv2.cvtColor(curr_color_blur, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 40, 20]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 40, 20]), np.array([180, 255, 255]))
        loose_red_mask = cv2.bitwise_or(mask1, mask2)

        # 3. รวมร่าง: ต้องเป็นของใหม่ AND มีความอมแดง (ถึงจะมืดก็จับ)
        final_color_mask = cv2.bitwise_and(diff_mask, loose_red_mask)
        
        # ปิดรอยแหว่งด้วย Kernel 
        kernel = np.ones((10, 3), np.uint8)
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
        _, diff_ir_mask = cv2.threshold(diff_ir, 25, 255, cv2.THRESH_BINARY) 
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
    root = tk.Tk(); app = RealSenseStaticBackground(root, "RealSense - Perfect Static RGB"); root.mainloop()