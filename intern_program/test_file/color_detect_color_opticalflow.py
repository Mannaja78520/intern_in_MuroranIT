import numpy as np
import cv2

# Initialize webcam
webcam = cv2.VideoCapture(1)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 20),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for tracking
ret, old_frame = webcam.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = None  # Points to track

while True:
    success, imageFrame = webcam.read()
    if not success:
        break

    frame_gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. YOUR COLOR DETECTION LOGIC ---
    blurred = cv2.GaussianBlur(imageFrame, (5, 5), 0)
    hsvFrame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red Range 1 and 2
    lower_red1, upper_red1 = np.array([0, 150, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 150, 70]), np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsvFrame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsvFrame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological Operations to connect line
    kernel = np.ones((15, 15), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- 2. OPTICAL FLOW INTEGRATION ---
    # Find current red centers to feed into Optical Flow
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centers = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centers.append([[np.float32(cx), np.float32(cy)]])
                
                # Draw bounding box (Your original UI)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert centers to numpy array for CV2
    p_new = np.array(current_centers, dtype=np.float32)

    # If we have points from the previous frame, calculate flow
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Draw motion vectors (blue lines) to show tracking
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Line showing where the cable moved from
                cv2.line(imageFrame, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 3)
                cv2.circle(imageFrame, (int(a), int(b)), 5, (255, 255, 0), -1)

    # --- 3. UPDATING STATE ---
    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = p_new

    # Show results
    cv2.imshow("Optical Flow + Color Detection", imageFrame)
    cv2.imshow("Red Mask", red_mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()