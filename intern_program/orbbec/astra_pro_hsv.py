import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBFormat, FrameSet
from utils import frame_to_bgr_image

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
    # Initialize Orbbec Astra Pro
    config = Config()
    pipeline = Pipeline()
    
    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
        pipeline.start(config)
        print("Orbbec Astra Pro started. Press 'q' or ESC to quit.")
    except Exception as e:
        print(f"Failed to start Orbbec camera: {e}")
        return

    try:
        while True:
            frames: FrameSet = pipeline.wait_for_frames(100) 
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
                
            img = frame_to_bgr_image(color_frame)
            if img is not None:
                view = process_and_stack(img, TARGET_SIZE)
                cv2.imshow('Orbbec Astra: [Gray | Hue] over [Sat | Val]', view)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()