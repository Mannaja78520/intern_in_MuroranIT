"""Multi-stage red-cable detector (v2).

This is the "lots of steps" pipeline asked for. It tries to see the
cable EVERYWHERE by stacking several independent cues, and it adds a
T-hook focus view that judges whether the cable is wrapping the hook.

Per frame:
  STEP 1  YOLO            - box the drone + the T-hook (drone_t_hook.pt)
  STEP 2  red mask        - HSV-red UNION LAB-a* (the cable is red)
  STEP 3  motion mask     - MOG2 background subtraction UNION grayscale
                            frame-difference; the cable MOVES, so a
                            moving pixel that the static background
                            never had is cable-candidate. If motion
                            floods the frame (camera was moved) it is
                            dropped for that frame.
  STEP 4  YOLO suppress   - the drone and the T-hook also move, so
                            their YOLO boxes are ERASED from the motion
                            mask; the drone box is erased from the red
                            mask too. What is left moving / red is the
                            cable.
  STEP 5  cable mask      - red UNION (motion kept only where the pixel
                            is at least slightly red) -> Close + Dilate
                            -> tiny blobs removed. This is the
                            "cable everywhere" mask.
  STEP 6  T-hook focus    - crop the T-hook box, detect the cable in it,
                            and estimate a wrap score from how much the
                            cable curves (arc length vs straight chord).

Outputs, per source video:
  rgb_cable_v2_steps.mp4  - a 3x2 panel grid showing every step
  rgb_cable_v2_mask.mp4   - the final binary cable mask

Run:
    conda activate intern_muroranIT_py312
    python cable_detect_v2.py            # all 17 recordings
    python cable_detect_v2.py <rgb.mp4>  # just one video
"""
import os
import sys
import glob
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- paths
_HERE     = os.path.dirname(os.path.abspath(__file__))
_INTERN   = os.path.dirname(os.path.dirname(_HERE))
REC_DIR   = os.path.join(_INTERN, "videos", "realsense", "recordings")
YOLO_PATH = os.path.join(_INTERN, "models", "drone_t_hook.pt")

# ------------------------------------------------------------- tuning
# Red bands (red wraps hue 0/180). Kept strict - the motion cue is
# what gives recall, so the red mask only needs the confident pixels.
HSV_LO_1, HSV_HI_1 = (0,   70, 50), (14,  255, 255)
HSV_LO_2, HSV_HI_2 = (160, 70, 50), (180, 255, 255)
LAB_LO,   LAB_HI   = (0, 150, 115), (255, 255, 180)
A_REDDISH          = 135     # motion pixel kept only if LAB a* >= this

GRAYDIFF_THR = 16            # grayscale frame-difference threshold
MOTION_FLOOD = 0.45          # motion mask above this fraction -> camera
                             # moved, motion cue dropped for the frame
MIN_AREA     = 10            # drop cable specks below this many px

YOLO_CONF  = 0.30
YOLO_EVERY = 2               # run YOLO every Nth frame, reuse between
DRONE = "crazyfile-drone"
THOOK = "muroranIT_t-hook"

# panel / overlay colours (BGR)
CABLE_BGR = (0, 255, 255)                       # yellow
BOX_BGR   = {DRONE: (255, 160, 0), THOOK: (0, 220, 0)}

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# -------------------------------------------------------- small helpers
def red_mask(bgr):
    """STEP 2 - HSV-red UNION LAB-a* red mask (uint8 0/255)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    m = cv2.bitwise_or(cv2.inRange(hsv, HSV_LO_1, HSV_HI_1),
                       cv2.inRange(hsv, HSV_LO_2, HSV_HI_2))
    m = cv2.bitwise_or(m, cv2.inRange(lab, LAB_LO, LAB_HI))
    return m


def zero_boxes(mask, boxes, names, pad=8):
    """Erase (set to 0) the given YOLO box regions from `mask`."""
    out = mask.copy()
    h, w = mask.shape[:2]
    for (x1, y1, x2, y2, name, conf) in boxes:
        if name not in names:
            continue
        out[max(0, y1 - pad):min(h, y2 + pad),
            max(0, x1 - pad):min(w, x2 + pad)] = 0
    return out


def clean(mask):
    """Close gaps, thicken, drop tiny specks."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)
    mask = cv2.dilate(mask, _K3)
    if MIN_AREA > 0:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                keep[lbl == i] = 255
        mask = keep
    return mask


def label(panel, text):
    """Caption a panel in-place (dark strip + text)."""
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    return panel


def gray3(mask):
    """Binary mask -> 3-channel BGR so it can sit in the panel grid."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------- the detector class
class CableDetectorV2:
    """Holds the per-video state (background model, previous frame)
    that the multi-step pipeline needs."""

    def __init__(self, model):
        self.model = model
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self.prev_gray = None
        self.boxes = []

    # -- STEP 1 ---------------------------------------------------------
    def yolo(self, bgr, frame_idx):
        """Box the drone + T-hook. Re-run every YOLO_EVERY-th frame
        and reuse the boxes in between (YOLO is the slow step)."""
        if self.model is None or frame_idx % YOLO_EVERY != 0:
            return self.boxes
        self.boxes = []
        try:
            res = self.model(bgr, verbose=False, conf=YOLO_CONF)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cid = int(b.cls[0])
                self.boxes.append((x1, y1, x2, y2,
                                   res.names.get(cid, str(cid)),
                                   float(b.conf[0])))
        except Exception as e:
            print("  ! yolo error:", e)
        return self.boxes

    # -- STEP 3 ---------------------------------------------------------
    def motion(self, bgr):
        """MOG2 background-subtraction UNION grayscale frame-difference.
        Returns (motion_mask, flooded) - flooded is True when the mask
        covered too much of the frame (camera moved)."""
        fg = self.bg.apply(bgr)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            diff = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray, self.prev_gray)
            diff = cv2.threshold(diff, GRAYDIFF_THR, 255,
                                 cv2.THRESH_BINARY)[1]
        self.prev_gray = gray

        motion = cv2.bitwise_or(fg, diff)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, _K3)
        flooded = motion.mean() > 255 * MOTION_FLOOD
        if flooded:
            motion = np.zeros_like(motion)        # camera moved
        return motion, flooded

    # -- STEP 6 ---------------------------------------------------------
    def thook_focus(self, bgr, boxes):
        """Crop the T-hook box and judge the cable there. Returns a
        640x480 BGR panel and a short status string.

        Wrap score = cable contour arc-length / straight chord. ~1 is
        a straight cable crossing the hook; clearly above 1 means the
        cable bends / loops around it. This is a heuristic, not a
        certainty."""
        th = [b for b in boxes if b[4] == THOOK]
        panel = np.zeros((480, 640, 3), np.uint8)
        if not th:
            cv2.putText(panel, "no T-hook detected", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)
            return panel, "no T-hook"
        x1, y1, x2, y2, _, _ = th[0]
        h, w = bgr.shape[:2]
        pad = 24
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return panel, "T-hook box empty"

        cm = clean(red_mask(crop))
        px = int(np.count_nonzero(cm))
        vis = crop.copy()
        vis[cm > 0] = (0.3 * vis[cm > 0]
                       + 0.7 * np.array(CABLE_BGR)).astype(np.uint8)

        if px < 15:
            status = "cable at hook: NONE"
        else:
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            big = max(cnts, key=cv2.contourArea)
            arc = cv2.arcLength(big, False)
            bx, by, bw, bh = cv2.boundingRect(big)
            chord = max(1.0, (bw ** 2 + bh ** 2) ** 0.5)
            tort = arc / (2.0 * chord)            # ~1 straight, >1 bent
            if tort > 1.7:
                status = "cable WRAPPING hook (curl %.2f)" % tort
            else:
                status = "cable crossing hook (curl %.2f)" % tort

        # fit the crop into the 640x480 panel
        ch, cw = vis.shape[:2]
        s = min(640.0 / cw, 460.0 / ch)
        vis = cv2.resize(vis, (int(cw * s), int(ch * s)))
        oy, ox = (480 - vis.shape[0]) // 2, (640 - vis.shape[1]) // 2
        panel[oy:oy + vis.shape[0], ox:ox + vis.shape[1]] = vis
        col = (0, 165, 255) if "WRAPPING" in status else (0, 220, 0)
        if "NONE" in status:
            col = (90, 90, 90)
        cv2.rectangle(panel, (0, 462), (640, 480), (0, 0, 0), -1)
        cv2.putText(panel, status, (6, 476),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        return panel, status

    # -- the whole pipeline --------------------------------------------
    def process(self, bgr, frame_idx):
        """Run every step. Returns (final_cable_mask, panels list)."""
        boxes = self.yolo(bgr, frame_idx)               # STEP 1
        red   = red_mask(bgr)                           # STEP 2
        motion, flooded = self.motion(bgr)              # STEP 3

        # STEP 4 - erase the drone + T-hook from motion (they move too);
        # erase only the drone from red (the T-hook is metal, not red).
        motion_s = zero_boxes(motion, boxes, {DRONE, THOOK}, pad=10)
        red_s    = zero_boxes(red,    boxes, {DRONE},        pad=6)

        # STEP 5 - cable = red UNION (motion that is at least reddish).
        lab_a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
        motion_red = cv2.bitwise_and(
            motion_s, (lab_a >= A_REDDISH).astype(np.uint8) * 255)
        cable = clean(cv2.bitwise_or(red_s, motion_red))

        # ---- build the 3x2 step panel grid ----
        p1 = bgr.copy()
        for (x1, y1, x2, y2, name, conf) in boxes:
            c = BOX_BGR.get(name, (200, 200, 200))
            cv2.rectangle(p1, (x1, y1), (x2, y2), c, 2)
            cv2.putText(p1, "%s %.2f" % (name, conf),
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        label(p1, "1. RGB + YOLO (drone / T-hook)")

        p2 = label(gray3(red),    "2. red mask (HSV + LAB)")
        m3 = "3. motion (bg-sub + gray-diff)"
        if flooded:
            m3 += "  [camera moved - dropped]"
        p3 = label(gray3(motion), m3)
        p4 = label(gray3(cable),  "4. cable mask (drone/hook removed)")

        p5 = bgr.copy()
        p5[cable > 0] = (0.35 * p5[cable > 0]
                         + 0.65 * np.array(CABLE_BGR)).astype(np.uint8)
        cnts, _ = cv2.findContours(cable, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p5, cnts, -1, CABLE_BGR, 1)
        label(p5, "5. cable highlight  (%d px)"
              % int(np.count_nonzero(cable)))

        p6, _status = self.thook_focus(bgr, boxes)      # STEP 6
        label(p6, "6. T-hook focus / wrap")

        grid = np.vstack([np.hstack([p1, p2, p3]),
                          np.hstack([p4, p5, p6])])
        return cable, grid


# ----------------------------------------------------------- per video
def detect_video(path, model):
    """Process one rgb.mp4 -> writes the step-grid + mask videos."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  ! cannot open", path)
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = os.path.dirname(path)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    grid_w  = cv2.VideoWriter(os.path.join(out_dir, "rgb_cable_v2_steps.mp4"),
                              fourcc, fps, (w * 3, h * 2), isColor=True)
    mask_w  = cv2.VideoWriter(os.path.join(out_dir, "rgb_cable_v2_mask.mp4"),
                              fourcc, fps, (w, h), isColor=False)

    det = CableDetectorV2(model)
    frames, total_px = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cable, grid = det.process(fr, frames)
        total_px += int(np.count_nonzero(cable))
        grid_w.write(grid)
        mask_w.write(cable)
        frames += 1
    cap.release()
    grid_w.release()
    mask_w.release()
    return frames, total_px


def main():
    model = None
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_PATH):
            model = YOLO(YOLO_PATH)
            print("YOLO loaded:", YOLO_PATH, model.names)
        else:
            print("YOLO model missing, motion+colour only:", YOLO_PATH)
    except Exception as e:
        print("ultralytics unavailable, motion+colour only:", e)

    if len(sys.argv) > 1:
        vids = [sys.argv[1]]
    else:
        vids = sorted(glob.glob(os.path.join(REC_DIR, "*", "rgb.mp4")))
    if not vids:
        print("no rgb.mp4 found under", REC_DIR)
        return

    print("processing %d video(s)\n" % len(vids))
    t0 = time.time()
    for i, v in enumerate(vids, 1):
        name = os.path.basename(os.path.dirname(v))
        ts = time.time()
        frames, px = detect_video(v, model)
        print("[%2d/%2d] %s  %d frames  %d cable-px  %.1fs"
              % (i, len(vids), name, frames, px, time.time() - ts))
    print("\ndone: %d video(s) in %.1fs" % (len(vids), time.time() - t0))
    print("outputs: rgb_cable_v2_steps.mp4 + rgb_cable_v2_mask.mp4 "
          "in each recording folder")


if __name__ == "__main__":
    main()
