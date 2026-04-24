import numpy as np
import cv2

webcam = cv2.VideoCapture(4)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while True:
    success, imageFrame = webcam.read()
    if not success: break

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

    # 3. THE FIX: Combine them smartly
    # Use movement to clean noise, but prioritize color for the line
    # We use a smaller weight for fg_mask or skip bitwise_and if we want it always visible
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

    cv2.imshow("Detection + Segmentation", cv2.resize(imageFrame, (640, 480)))
    cv2.imshow("Final Segmented Red Cable", cv2.resize(final_mask, (640, 480)))

    if cv2.waitKey(10) & 0xFF == ord('q'): break

webcam.release()
cv2.destroyAllWindows()