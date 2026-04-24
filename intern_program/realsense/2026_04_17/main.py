import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyrealsense2 as rs
from feature_detector import CableFeatureDetector

class RealSenseApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.pipeline = rs.pipeline(); self.config = rs.config()
        self.bg_hsv = None; self.bg_ir = None; self.f_count = 0
        
        # ส่ง Path รูป Template ทั้ง 2 แบบ (สี และ IR)
        self.t_detector = CableFeatureDetector("color_raw_1776416226.png", "ir_raw_1776416226.png")
        
        try:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.pipeline.start(self.config)
        except Exception as e: print(f"Error: {e}"); self.window.destroy(); return

        self.frame_btn = tk.Frame(self.window, pady=5); self.frame_btn.pack()
        tk.Button(self.frame_btn, text="📸 Capture Background", bg="yellow", command=self.set_background).pack()
        
        self.f_top = tk.Frame(self.window); self.f_top.pack()
        self.f_bot = tk.Frame(self.window); self.f_bot.pack()

        self.l_c_res = tk.Label(self.f_top, text="Color: Cable+Hook"); self.l_c_res.pack(side="left", padx=5)
        self.l_c_mask = tk.Label(self.f_top, text="Color Mask"); self.l_c_mask.pack(side="left", padx=5)
        self.l_i_res = tk.Label(self.f_bot, text="IR: Cable+Hook"); self.l_i_res.pack(side="left", padx=5)
        self.l_i_mask = tk.Label(self.f_bot, text="IR Mask"); self.l_i_mask.pack(side="left", padx=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_feed()

    def set_background(self): self.bg_hsv = None; self.bg_ir = None

    def get_red_mask(self, hsv):
        m1 = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255]))
        return cv2.bitwise_or(m1, m2)

    def process_frames(self, color_f, ir_f):
        c_raw = np.asanyarray(color_f.get_data())
        i_raw = np.asanyarray(ir_f.get_data())
        c_blur = cv2.GaussianBlur(c_raw, (5, 5), 0)
        i_blur = cv2.GaussianBlur(i_raw, (5, 5), 0)
        c_hsv = cv2.cvtColor(c_blur, cv2.COLOR_BGR2HSV)

        if self.bg_hsv is None: 
            self.bg_hsv = c_hsv.copy(); self.bg_ir = i_blur.copy(); return

        self.f_count += 1
        kernel = np.ones((5,3), np.uint8)

        # --- [1] COLOR WORLD ---
        bg_h, _, _ = cv2.split(self.bg_hsv); c_h, _, _ = cv2.split(c_hsv)
        diff_h = np.minimum(np.abs(bg_h.astype(np.int16) - c_h.astype(np.int16)), 180 - np.abs(bg_h.astype(np.int16) - c_h.astype(np.int16))).astype(np.uint8)
        _, h_mask = cv2.threshold(diff_h, 15, 255, cv2.THRESH_BINARY)
        final_c_mask = cv2.morphologyEx(cv2.bitwise_and(self.get_red_mask(c_hsv), h_mask), cv2.MORPH_CLOSE, kernel)
        
        c_disp = c_raw.copy()
        conts_c, _ = cv2.findContours(final_c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts_c:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(c_disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # --- [2] IR WORLD ---
        diff_ir = cv2.absdiff(self.bg_ir, i_blur)
        _, i_mask_thr = cv2.threshold(diff_ir, 25, 255, cv2.THRESH_BINARY)
        final_i_mask = cv2.morphologyEx(cv2.medianBlur(i_mask_thr, 5), cv2.MORPH_CLOSE, kernel)
        i_disp = cv2.cvtColor(i_raw, cv2.COLOR_GRAY2BGR)
        
        conts_i, _ = cv2.findContours(final_i_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts_i:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(i_disp, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # --- [3] SIFT DETECTION (Every 2nd frame) ---
        if self.f_count % 2 == 0:
            # ไม่ต้องส่งมาส์กเก่า (final_c_mask) เข้าไปแล้ว ให้มันหาเองข้างใน
            box_c = self.t_detector.detect_t_hook(c_disp, mode='color')
            if box_c: 
                cv2.rectangle(c_disp, (box_c[0], box_c[1]), (box_c[2], box_c[3]), (255, 0, 255), 3)
            
            box_i = self.t_detector.detect_t_hook(i_disp, mode='ir')
            if box_i: 
                cv2.rectangle(i_disp, (box_i[0], box_i[1]), (box_i[2], box_i[3]), (255, 0, 255), 3)

        self.set_img(self.l_c_res, c_disp, True); self.set_img(self.l_c_mask, final_c_mask, False, True)
        self.set_img(self.l_i_res, i_disp, True); self.set_img(self.l_i_mask, final_i_mask, False, True)

    def set_img(self, lbl, img, is_bgr=False, is_gray=False):
        img = cv2.resize(img, (320, 240))
        if is_bgr: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif is_gray: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
        lbl.imgtk = tk_img; lbl.configure(image=tk_img)

    def update_feed(self):
        try:
            fs = self.pipeline.wait_for_frames(100)
            if fs: self.process_frames(fs.get_color_frame(), fs.get_infrared_frame(1))
        except: pass
        self.window.after(15, self.update_feed)

    def on_closing(self): self.pipeline.stop(); self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk(); app = RealSenseApp(root, "Dual World SIFT System"); root.mainloop()