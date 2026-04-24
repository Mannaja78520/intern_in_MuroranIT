import cv2
import numpy as np
import pyrealsense2 as rs

TARGET_SIZE = (640, 360)
ESC_KEY = 27

def process_and_stack(im, target_size):
    """Applies resizing, HSV splitting, colormapping, and stitches into a 2x2 grid."""
    im_resized = cv2.resize(im, target_size)
    im_flipped = cv2.flip(im_resized, 1)

    im_gray = cv2.cvtColor(im_flipped, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(im_flipped, cv2.COLOR_BGR2HSV)
    
    hue = im_hsv[:, :, 0] # uint8 0-179
    sat = im_hsv[:, :, 1] # uint8 0-255
    val = im_hsv[:, :, 2] # uint8 0-255

    # 1. Apply the ColorMap to the Hue channel
    hue_mapped = cv2.applyColorMap((hue * (255 / 179)).astype('uint8'), cv2.COLORMAP_HSV)

    # 2. Convert the 1-channel images to 3-channel so they can be stacked with hue_mapped
    im_gray_3c = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
    sat_3c = cv2.cvtColor(sat, cv2.COLOR_GRAY2BGR)
    val_3c = cv2.cvtColor(val, cv2.COLOR_GRAY2BGR)

    # 3. Stitch into a 2x2 grid
    top_row = np.hstack((im_gray_3c, hue_mapped))
    bottom_row = np.hstack((sat_3c, val_3c))
    return np.vstack((top_row, bottom_row))

def main():
    # Initialize Intel RealSense D435i
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        print("Intel RealSense D435i started. Press 'q' or ESC to quit.")
    except Exception as e:
        print(f"Failed to start RealSense camera: {e}")
        return

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            img = np.asanyarray(color_frame.get_data())
            
            view = process_and_stack(img, TARGET_SIZE)
            cv2.imshow('RealSense D435i: [Gray | Hue] over [Sat | Val]', view)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()