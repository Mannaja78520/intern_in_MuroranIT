import cv2
import numpy as np
import pyrealsense2 as rs

ESC_KEY = 27

def main():
    # ---------------------------
    # 1. Initialize RealSense Camera (Color Only)
    # ---------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Request a 1280x720 RGB stream at 30fps. 
        # RealSense outputs native BGR, so no manual color conversion is needed!
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        print("RealSense Pipeline started. Press 'q' or ESC to quit.")
    except Exception as e:
        print(f"Error initializing RealSense stream: {e}")
        # Fallback to 640x480 if 1280x720 is not supported by your specific RealSense model
        print("Falling back to 640x480 resolution...")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

    # Initialize Background Subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # ---------------------------
    # 2. Main Processing Loop
    # ---------------------------
    try:
        while True:
            # Grab frame set from RealSense pipeline
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            # Convert RealSense frame directly to numpy array for OpenCV
            imageFrame = np.asanyarray(color_frame.get_data())

            # --- START OF CABLE DETECTION LOGIC ---
            
            # 1. Get Movement Mask
            fg_mask = backSub.apply(imageFrame)

            # 2. Get Color Mask (Lowered thresholds to see the wire better)
            hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 100, 50]) # Lower saturation to catch "still" red
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red2 = np.array([160, 100, 50])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            color_mask = cv2.bitwise_or(mask1, mask2)

            # 3. Combine them smartly (Movement AND Color)
            combined_mask = cv2.bitwise_and(color_mask, fg_mask) 
            
            # Bridge the gaps with a TALL kernel
            kernel = np.ones((30, 5), np.uint8)
            final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # 4. Filter by Area to remove noise
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Only draw if it's "Line-shaped" (Tall or Wide)
                    if h > w * 1.5 or w > h * 1.5:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(imageFrame, "RGB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- END OF CABLE DETECTION LOGIC ---

            # Display the results
            cv2.imshow("Detection + Segmentation", cv2.resize(imageFrame, (640, 480)))
            cv2.imshow("Final Segmented Red Cable", cv2.resize(final_mask, (640, 480)))

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