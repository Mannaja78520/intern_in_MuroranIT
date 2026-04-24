import cv2
import numpy as np

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27

def main():
    # ---------------------------
    # 1. Initialize Orbbec Camera
    # ---------------------------
    config = Config()
    pipeline = Pipeline()
    
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # Try to request 1280x720 to match your original OpenCV settings
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(f"Requested resolution not supported, falling back to default: {e}")
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
            
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
        
    pipeline.start(config)

    # Initialize Background Subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    print("Pipeline started. Press 'q' or ESC to quit.")

    # ---------------------------
    # 2. Main Processing Loop
    # ---------------------------
    while True:
        try:
            # Grab frame set from Orbbec pipeline
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
                
            # Convert Orbbec frame to BGR numpy array for OpenCV
            imageFrame = frame_to_bgr_image(color_frame)
            if imageFrame is None:
                print("failed to convert frame to image")
                continue

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

            # 3. Combine them smartly
            combined_mask = cv2.bitwise_and(color_mask, color_mask, mask=None) 
            
            # Bridge the gaps with a TALL kernel
            kernel = np.ones((30, 5), np.uint8)
            final_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # 4. Filter by Area to remove face/lips noise
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Only draw if it's "Line-shaped" (Tall or Wide)
                    if h > w * 1.5 or w > h * 1.5:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- END OF CABLE DETECTION LOGIC ---

            # Display the results
            cv2.imshow("Detection + Segmentation", cv2.resize(imageFrame, (640, 480)))
            cv2.imshow("Final Segmented Red Cable", cv2.resize(final_mask, (640, 480)))

            # Handle keystrokes
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break
                
        except KeyboardInterrupt:
            break

    # ---------------------------
    # 3. Cleanup
    # ---------------------------
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()