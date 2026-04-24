import numpy as np
import cv2

# Initialize webcam
webcam = cv2.VideoCapture(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 1. Initialize Background Subtractor
# history: how many previous frames it remembers
# varThreshold: higher means less sensitive to noise/light changes
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 20),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for tracking
ret, old_frame = webcam.read()
if not ret:
    print("Cannot read from webcam")
    exit()
    
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = None  

while True:
    success, imageFrame = webcam.read()
    if not success:
        break

    # --- 2. BACKGROUND SEGMENTATION ---
    # Generates a mask where moving objects are white/gray and background is black
    fg_mask = backSub.apply(imageFrame)
    
    # --- 3. COLOR DETECTION LOGIC ---
    blurred = cv2.GaussianBlur(imageFrame, (5, 5), 0)
    hsvFrame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red Range 1 and 2
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsvFrame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsvFrame, lower_red2, upper_red2)
    color_mask = cv2.bitwise_or(mask1, mask2)

    # --- 4. COMBINE COLOR + MOTION ---
    # This is the "Image Segmentation" part - keeping only moving red parts
    combined_mask = cv2.bitwise_and(color_mask, fg_mask)

    # Morphological Operations: Connect the cable into one line
    # Using a slightly taller kernel to help bridge vertical wire gaps
    kernel = np.ones((25, 5), np.uint8)
    final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- 5. OPTICAL FLOW & CONTOURS ---
    frame_gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centers = []
    
    for cnt in contours:
        # Ignore small noise, focus on the moving cable
        if cv2.contourArea(cnt) > 150:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centers.append([[np.float32(cx), np.float32(cy)]])
                
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    p_new = np.array(current_centers, dtype=np.float32)

    if p0 is not None and len(p0) > 0 and len(p_new) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(imageFrame, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 3)
                cv2.circle(imageFrame, (int(a), int(b)), 5, (255, 255, 0), -1)

    # --- 6. UPDATING STATE ---
    old_gray = frame_gray.copy()
    p0 = p_new

    # Show results
    cv2.imshow("Detection + Segmentation", imageFrame)
    cv2.imshow("Movement Mask (FG)", fg_mask)
    cv2.imshow("Final Segmented Red Cable", final_mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()