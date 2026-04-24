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

        # Background Subtractors
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.ir_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

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
            frames = self.pipeline.wait_for_frames(100)
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            if color_frame and ir_frame:
                self.process_frames(color_frame, ir_frame)
        except RuntimeError: pass
        self.window.after(20, self.update_feed)

    def process_frames(self, color_frame, ir_frame):
        img_color = np.asanyarray(color_frame.get_data()).copy()
        ir_raw = np.asanyarray(ir_frame.get_data())
        H, W = img_color.shape[:2]

        # =========================================================================
        # --- การจัดการภาพ RGB (คงเดิม - จับสายไฟโค้งงอได้ดีมากจากรูปที่ส่งมา) ---
        # =========================================================================
        fg_mask_color = self.backSub.apply(img_color)
        win_h, win_w = 240, 320
        combined_cable_mask = np.zeros((H, W), dtype=np.uint8)

        for y in range(0, H, win_h):
            for x in range(0, W, win_w):
                roi_color = img_color[y:y+win_h, x:x+win_w]
                roi_fg = fg_mask_color[y:y+win_h, x:x+win_w]

                roi_color_up = cv2.resize(roi_color, (win_w * 2, win_h * 2), interpolation=cv2.INTER_LINEAR)
                roi_fg_up = cv2.resize(roi_fg, (win_w * 2, win_h * 2), interpolation=cv2.INTER_NEAREST)

                gray_up = cv2.cvtColor(roi_color_up, cv2.COLOR_BGR2GRAY)
                blur_up = cv2.GaussianBlur(gray_up, (5, 5), 0)
                edges_up = cv2.Canny(blur_up, 40, 120)

                edges_fg_up = cv2.bitwise_and(edges_up, roi_fg_up)
                dilated_edges = cv2.dilate(edges_fg_up, np.ones((5, 5), np.uint8), iterations=2)
                
                contours_up, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cable_mask_up = np.zeros_like(edges_fg_up)
                
                for cnt in contours_up:
                    length = cv2.arcLength(cnt, closed=False)
                    if length > 80: 
                        cv2.drawContours(cable_mask_up, [cnt], -1, 255, 10)

                cable_mask_down = cv2.resize(cable_mask_up, (win_w, win_h), interpolation=cv2.INTER_NEAREST)
                combined_cable_mask[y:y+win_h, x:x+win_w] = cable_mask_down

        kernel_global = np.ones((35, 35), np.uint8)
        combined_cable_mask = cv2.morphologyEx(combined_cable_mask, cv2.MORPH_CLOSE, kernel_global)

        contours, _ = cv2.findContours(combined_cable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            cx, cy, cw, ch = cv2.boundingRect(all_points)
            if cw > 20 or ch > 20: 
                cv2.rectangle(img_color, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 3) 
                cv2.putText(img_color, "CABLE DETECTED", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # =========================================================================
        # --- การจัดการภาพ IR (อัปเดตใหม่ แก้อาการเส้นหาย) ---
        # =========================================================================
        img_ir = cv2.cvtColor(ir_raw, cv2.COLOR_GRAY2BGR)
        ir_fg_mask = self.ir_backSub.apply(ir_raw)
        
        # ตัดเงาออก เอาเฉพาะส่วนที่ขยับจริงๆ (MOG2 จะให้ค่า 255 สำหรับวัตถุแท้)
        _, ir_thresh = cv2.threshold(ir_fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # 1. ขยายจุดไข่ปลาให้ใหญ่และอ้วนขึ้นก่อน (Dilation)
        kernel_ir_dilate = np.ones((5, 5), np.uint8)
        ir_dilated = cv2.dilate(ir_thresh, kernel_ir_dilate, iterations=2)

        # 2. ถมช่องว่างระหว่างจุดให้กลายเป็นเส้นเดียวกัน (Closing)
        kernel_ir_close = np.ones((35, 35), np.uint8)
        ir_closed = cv2.morphologyEx(ir_dilated, cv2.MORPH_CLOSE, kernel_ir_close)

        # 3. ลบจุดรบกวนเล็กๆ ทิ้ง (Opening) - ทำหลังสุดสายไฟจะได้ไม่หาย
        kernel_ir_open = np.ones((9, 9), np.uint8)
        ir_cleaned = cv2.morphologyEx(ir_closed, cv2.MORPH_OPEN, kernel_ir_open)

        # 4. หา Bounding Box ของฝั่ง IR
        contours_ir, _ = cv2.findContours(ir_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_ir:
            all_points_ir = np.vstack(contours_ir)
            cx_ir, cy_ir, cw_ir, ch_ir = cv2.boundingRect(all_points_ir)
            
            # กรองขนาดเหมือนฝั่ง RGB เพื่อไม่ให้ตีกรอบอะไรที่เล็กเกินไป
            if cw_ir > 20 or ch_ir > 20: 
                cv2.rectangle(img_ir, (cx_ir, cy_ir), (cx_ir+cw_ir, cy_ir+ch_ir), (0, 0, 255), 3) 
                cv2.putText(img_ir, "CABLE DETECTED", (cx_ir, cy_ir-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # การบันทึกวิดีโอ
        if self.is_recording and self.video_writer:
            top = np.hstack((img_color, cv2.cvtColor(combined_cable_mask, cv2.COLOR_GRAY2BGR)))
            bottom = np.hstack((img_ir, cv2.cvtColor(ir_cleaned, cv2.COLOR_GRAY2BGR)))
            self.video_writer.write(np.vstack((top, bottom)))

        # อัปเดตขึ้นจอ UI
        self.set_image(self.lbl_color, img_color, is_bgr=True)
        self.set_image(self.lbl_color_mask, combined_cable_mask, is_gray=True)
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