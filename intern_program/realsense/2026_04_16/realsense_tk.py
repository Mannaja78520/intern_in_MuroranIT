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

    # ==========================================
    # --- FUSE + CLEAN PIPELINE ---
    # ==========================================

    def fuse_and_clean(self, color_mask, ir_mask, ir_shape, color_shape):
        """
        รวม mask → เชื่อมเส้น → ตัด noise ด้วย top-bottom connectivity
        """
        # 1. Resize IR mask ให้ตรงกับ color
        ir_resized = cv2.resize(ir_mask, (color_shape[1], color_shape[0]))
        
        # 2. รวม mask
        fused = cv2.bitwise_or(color_mask, ir_resized)

        # ------------------------------------------
        # STEP A: Opening ตัด noise จุดเล็กๆก่อน
        # (erode กิน noise → dilate คืนเส้นจริง)
        # ------------------------------------------
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # ------------------------------------------
        # STEP B: Dilate แนวตั้งเชื่อมจุดที่ขาด
        # ใช้แค่แนวตั้งก่อน ยังไม่แนวนอน
        # ------------------------------------------
        v_kernel = np.ones((20, 1), np.uint8)
        fused_v = cv2.dilate(fused, v_kernel, iterations=2)

        # ------------------------------------------
        # STEP C: Closing แนวนอน เชื่อม 2 ขอบสาย
        # ------------------------------------------
        h_kernel = np.ones((1, 20), np.uint8)
        fused_h = cv2.morphologyEx(fused_v, cv2.MORPH_CLOSE, h_kernel)

        # ------------------------------------------
        # STEP D: Connected Component — ตัด component
        # ที่ไม่แตะขอบบนและขอบล่างพร้อมกัน
        # ------------------------------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fused_h, connectivity=8)
        
        h, w = fused_h.shape
        valid_mask = np.zeros_like(fused_h)
        
        for lbl in range(1, num_labels):  # 0 = background
            component = (labels == lbl).astype(np.uint8)
            
            # ตรวจว่า component นี้แตะขอบบน (top 20%) หรือขอบล่าง (bottom 20%) ไหม
            top_zone    = component[:h//5, :]       # 20% บน
            bottom_zone = component[h*4//5:, :]     # 20% ล่าง
            
            touches_top    = np.any(top_zone > 0)
            touches_bottom = np.any(bottom_zone > 0)
            
            area = stats[lbl, cv2.CC_STAT_AREA]
            
            # เก็บถ้า: แตะทั้งบนและล่าง OR ใหญ่มากพอ (สายยาวที่เริ่มกลางภาพ)
            if (touches_top and touches_bottom) or area > 500:
                valid_mask = cv2.bitwise_or(valid_mask, component * 255)

        # ------------------------------------------
        # STEP E: Closing ขั้นสุดท้าย อุดช่องว่างเล็กน้อย
        # ------------------------------------------
        final_kernel = np.ones((10, 10), np.uint8)
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, final_kernel)

        # ------------------------------------------
        # STEP F: Skeleton
        # ------------------------------------------
        skeleton = cv2.ximgproc.thinning(valid_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        vis = cv2.dilate(skeleton, np.ones((3,3), np.uint8), iterations=1)
        
        return valid_mask, vis
    

    def process_frames(self, color_frame, ir_frame):
        imageFrame = np.asanyarray(color_frame.get_data())
        ir_8bit = np.asanyarray(ir_frame.get_data())

        # ==========================================
        # --- PROCESS COLOR FRAME → color_mask ---
        # ==========================================
        fg_mask = self.backSub.apply(imageFrame)
        hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 80, 40]),   np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255]))
        color_mask = cv2.bitwise_and(cv2.bitwise_or(mask1, mask2), fg_mask)
        kernel = np.ones((10, 3), np.uint8)
        final_color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # วาด bounding box บน RGB
        contours, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 1.5 or w > h * 1.5:
                    cv2.rectangle(imageFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(imageFrame, "RGB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.set_image(self.lbl_color, imageFrame, is_bgr=True)
        self.set_image(self.lbl_color_mask, final_color_mask, is_gray=True)

        # ==========================================
        # --- PROCESS IR FRAME ---
        # ==========================================
        ir_display = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)
        ir_fg_mask = self.ir_backSub.apply(ir_8bit)
        _, ir_thresh = cv2.threshold(ir_fg_mask, 80, 255, cv2.THRESH_BINARY)
        merge_kernel = np.ones((1, 20), np.uint8)
        ir_merged = cv2.dilate(ir_thresh, merge_kernel, iterations=1)
        close_kernel = np.ones((15, 1), np.uint8)
        ir_mask = cv2.morphologyEx(ir_merged, cv2.MORPH_CLOSE, close_kernel)

        # --- FUSE + CLEAN ---
        valid_mask, ir_vis = self.fuse_and_clean(
            final_color_mask, ir_mask,
            ir_shape=ir_8bit.shape,
            color_shape=imageFrame.shape
        )

        # วาด bounding box เฉพาะ component ที่ผ่าน filter
        contours_fused, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fused:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            asp = max(h, w) / (min(h, w) + 1e-5)
            if area > 20 and asp > 1.5:
                sx = ir_8bit.shape[1] / imageFrame.shape[1]
                sy = ir_8bit.shape[0] / imageFrame.shape[0]
                xi, yi, wi, hi = int(x*sx), int(y*sy), int(w*sx), int(h*sy)
                cv2.rectangle(ir_display, (xi, yi), (xi+wi, yi+hi), (255, 255, 0), 2)
                cv2.putText(ir_display, "FUSED", (xi, yi-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        self.set_image(self.lbl_ir, ir_display, is_bgr=True)
        self.set_image(self.lbl_ir_mask, ir_vis, is_gray=True)

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