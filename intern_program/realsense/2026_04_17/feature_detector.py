import cv2
import numpy as np

class CableFeatureDetector:
    def __init__(self, color_temp_path, ir_temp_path):
        self.detector = cv2.SIFT_create(nfeatures=1000)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # โหลด Template และตัดพื้นหลัง Template ออก (ถ้าทำได้)
        self.img_c = cv2.imread(color_temp_path, 0)
        self.img_i = cv2.imread(ir_temp_path, 0)
        
        # สกัด Feature จาก Template
        self.kp_c, self.des_c = self.detector.detectAndCompute(self.img_c, None)
        self.kp_i, self.des_i = self.detector.detectAndCompute(self.img_i, None)
        print("SIFT Optimized Engine Ready.")

    def _create_live_mask(self, gray_frame):
        """ สกัดเฉพาะส่วนที่เด่นออกมาจากพื้นหลังรกๆ สดๆ """
        # ใช้ GaussianBlur เพื่อลด Noise พื้นหลัง
        blur = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        
        # ใช้ Adaptive Threshold เพื่อแยกวัตถุออกจากพื้นหลังที่ไม่สม่ำเสมอ
        live_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        
        # ขยาย Mask เพื่อให้คลุมตัววัตถุทั้งหมด
        kernel = np.ones((5, 5), np.uint8)
        live_mask = cv2.dilate(live_mask, kernel, iterations=2)
        return live_mask

    def detect_t_hook(self, frame, mode='color'):
        target_des = self.des_c if mode == 'color' else self.des_i
        if target_des is None: return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 🔥 สร้าง Mask สดๆ เพื่อตัดพื้นหลังรกๆ ออกก่อนสแกน Feature
        auto_mask = self._create_live_mask(gray)
        
        # สแกนเฉพาะจุดที่ auto_mask บอก (ตัดชั้นวางของ/พื้นหลังทิ้ง)
        kp_f, des_f = self.detector.detectAndCompute(gray, mask=auto_mask)
        
        if des_f is None or len(des_f) < 5: return None

        matches = self.matcher.knnMatch(target_des, des_f, k=2)
        good = [m for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.7 * m_n[1].distance for m in [m_n[0]]]

        if len(good) >= 8: # ลดเหลือ 8 จุดเพื่อให้เจอง่ายขึ้น
            pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            return (np.int32(pts.min(axis=0).ravel()), np.int32(pts.max(axis=0).ravel()))
        return None