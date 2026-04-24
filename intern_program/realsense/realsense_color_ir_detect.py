import cv2
import numpy as np
import pyrealsense2 as rs

ESC_KEY = 27

def main():
    # ---------------------------
    # 1. Initialize RealSense Camera
    # ---------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Enable Color stream (Native BGR output!)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Enable Infrared stream (Stream 1 is usually the left IR camera)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        
        # Start the pipeline
        pipeline.start(config)
        print("RealSense Pipeline started with Color and IR. Press 'q' or ESC to quit.")
    except Exception as e:
        print(f"Error initializing RealSense streams: {e}")
        return

    # Initialize Background Subtractors
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    ir_backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # ---------------------------
    # 2. Main Processing Loop
    # ---------------------------
    try:
        while True:
            # Grab frame set from RealSense
            frames = pipeline.wait_for_frames()
            
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            
            if not color_frame or not ir_frame:
                continue
                
            # Convert directly to numpy arrays
            imageFrame = np.asanyarray(color_frame.get_data())
            ir_8bit = np.asanyarray(ir_frame.get_data())

            # ==========================================
            # --- PROCESS COLOR FRAME ---
            # ==========================================
            fg_mask = backSub.apply(imageFrame)

            hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 100, 50]) 
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red2 = np.array([160, 100, 50])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            color_mask = cv2.bitwise_or(mask1, mask2)

            combined_color_mask = cv2.bitwise_and(color_mask, fg_mask)

            kernel = np.ones((30, 5), np.uint8)
            final_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(final_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > w * 1.5 or w > h * 1.5:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(imageFrame, "RGB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Color Detection", imageFrame)
            cv2.imshow("Color Threshold Mask", final_color_mask)

            # ==========================================
            # --- PROCESS IR FRAME ---
            # ==========================================
            ir_display = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)
            ir_fg_mask = ir_backSub.apply(ir_8bit)

            _, ir_thresh = cv2.threshold(ir_fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            ir_kernel = np.ones((20, 5), np.uint8)
            ir_cleaned = cv2.morphologyEx(ir_thresh, cv2.MORPH_CLOSE, ir_kernel)

            ir_contours, _ = cv2.findContours(ir_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in ir_contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > w * 1.5 or w > h * 1.5:
                        cv2.rectangle(ir_display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        cv2.putText(ir_display, "IR", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("IR Detection", ir_display)
            cv2.imshow("IR Threshold Mask", ir_cleaned)

            # Handle keystrokes
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break
                
    finally:
        # ---------------------------
        # 3. Cleanup
        # ---------------------------
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()