"""Batch red-cable detector for the RealSense recordings.

Detects the thin RED cable that connects the drone to the T-hook,
across every rgb.mp4 in  videos/realsense/recordings/.

What it does, per source video:
  * red detection      - HSV-red UNION LAB-a* mask (a red pixel is kept
                         if EITHER detector fires, so faint / far parts
                         on a bright background still survive)
  * morphology         - Close then Dilate, to bridge the gaps in the
                         thin cable and thicken it enough to see
  * small-blob removal - drops specks below MIN_AREA
  * YOLO assist        - runs models/drone_t_hook.pt to box the drone
                         and the T-hook; the cable runs between them,
                         so the boxes mark where to expect it
  * writes  rgb_cable_annotated.mp4  (RGB + red overlay + YOLO boxes)
  * writes  rgb_cable_mask.mp4       (binary cable mask)

Tuned from the real footage: the red cable measured roughly
LAB a* 169-188, b* 130-151 and HSV hue near 0/180, S 107-181.
The thresholds below are opened a little wider than that so the
far / motion-blurred parts of the cable are not lost.

Run:
    conda activate intern_muroranIT_py312
    python cable_detect_batch.py            # all 17 recordings
    python cable_detect_batch.py <rgb.mp4>  # just one video
"""
import os
import sys
import glob
import json
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- paths
_HERE     = os.path.dirname(os.path.abspath(__file__))
# .../intern_program/realsense/2026_05_18  ->  .../intern_program
_INTERN   = os.path.dirname(os.path.dirname(_HERE))
REC_DIR   = os.path.join(_INTERN, "videos", "realsense", "recordings")
YOLO_PATH = os.path.join(_INTERN, "models", "drone_t_hook.pt")
CFG_OUT   = os.path.join(_HERE, "cable_detect_config.json")

# ------------------------------------------------------ red-cable tuning
# HSV: red wraps around hue 0 / 180, so two bands are needed.
HSV_LO_1, HSV_HI_1 = (0,   70, 50), (14,  255, 255)   # low-hue red
HSV_LO_2, HSV_HI_2 = (160, 70, 50), (180, 255, 255)   # high-hue red
# LAB: a* is the red-vs-green axis (128 = neutral). The cable measured
# a* >= 169; we open to 155 to keep faint / far pixels. L is left fully
# open (0..255) so a bright white background does not gate the cable
# out, and b* is kept moderately wide.
LAB_LO, LAB_HI = (0, 152, 120), (255, 255, 175)
MIN_AREA   = 12         # drop red specks smaller than this (pixels)
YOLO_CONF  = 0.30       # detection confidence for drone / T-hook
YOLO_EVERY = 2          # run YOLO every Nth frame, reuse boxes between

# --- between-boxes filter -------------------------------------------
# The cable can only run between the drone and the T-hook, so when
# YOLO finds them we restrict red detection to that region. This
# kills the red-tape / background false positives, and lets us use a
# LOOSER red threshold inside the region to recover faint far-away
# cable pixels that the strict threshold would miss.
BETWEEN_BOXES = True
ROI_PAD       = 60      # px padding around the box union (cable sag)
ROI_PAD_ONE   = 160     # bigger pad when only ONE box was found
# Looser red bands, used ONLY inside the box region.
HSV_LO_1L = (0,   45, 40)     # loose low-hue red
HSV_LO_2L = (160, 45, 40)     # loose high-hue red
LAB_LOL   = (0, 138, 112)     # loose LAB lower bound (a* down to 138)

# overlay / box colours (BGR)
CABLE_BGR  = (0, 255, 255)     # yellow highlight for the cable mask
BOX_BGR    = {"crazyfile-drone": (255, 160, 0),     # orange-ish
              "muroranIT_t-hook": (0, 220, 0)}      # green


def roi_from_boxes(boxes, w, h):
    """Region the cable must lie within: the union of the YOLO boxes,
    padded for cable sag. Returns (x0, y0, x1, y1), or None when no
    boxes were found (then no spatial restriction is applied)."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    # One box only -> the far end of the cable is unknown, so pad
    # generously; two+ boxes bracket the cable tightly.
    pad = ROI_PAD if len(boxes) >= 2 else ROI_PAD_ONE
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def red_cable_mask(bgr, roi=None):
    """Return a binary uint8 mask (0/255) of the red cable in `bgr`.

    The mask is the UNION of an HSV-red test and a LAB-a* test, then
    Closed + Dilated to reconnect the thin cable, then cleared of
    blobs below MIN_AREA.

    When `roi` (a padded box-union rect) is given, detection is
    restricted to it and a LOOSER threshold is used inside, so faint
    far-away cable survives without the rest of the frame adding
    noise. `roi` None -> strict threshold over the whole frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    if roi is None:
        hl1, hl2, ll = HSV_LO_1, HSV_LO_2, LAB_LO
    else:
        hl1, hl2, ll = HSV_LO_1L, HSV_LO_2L, LAB_LOL
    hsv_red = cv2.bitwise_or(
        cv2.inRange(hsv, hl1, HSV_HI_1),
        cv2.inRange(hsv, hl2, HSV_HI_2))
    lab_red = cv2.inRange(lab, ll, LAB_HI)
    mask = cv2.bitwise_or(hsv_red, lab_red)

    # Restrict to the drone <-> T-hook corridor: zero everything
    # outside the padded box region.
    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = keep

    # Close bridges gaps along the thin cable; Dilate thickens it.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Remove tiny specks (background red noise).
    if MIN_AREA > 0:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                keep[lbl == i] = 255
        mask = keep
    return mask


def annotate(bgr, mask, boxes, roi=None):
    """Draw the red-cable overlay + YOLO boxes onto a copy of `bgr`."""
    out = bgr.copy()
    # Faint grey rectangle = the box region red was searched within.
    if roi is not None:
        cv2.rectangle(out, (roi[0], roi[1]), (roi[2], roi[3]),
                      (130, 130, 130), 1)
    # Tint the cable pixels yellow and outline the cable contours.
    out[mask > 0] = (0.35 * out[mask > 0]
                     + 0.65 * np.array(CABLE_BGR)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, CABLE_BGR, 1)
    # YOLO boxes for the drone + T-hook the cable links.
    for (x1, y1, x2, y2, name, conf) in boxes:
        col = BOX_BGR.get(name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, "%s %.2f" % (name, conf),
                    (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1)
    px = int(np.count_nonzero(mask))
    cv2.putText(out, "cable px: %d" % px, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CABLE_BGR, 2)
    return out


def detect_video(path, model):
    """Process one rgb.mp4 -> writes *_cable_annotated.mp4 and
    *_cable_mask.mp4 next to it. Returns (frames, total_cable_px)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  ! cannot open", path)
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = os.path.dirname(path)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    ann_w = cv2.VideoWriter(os.path.join(out_dir, "rgb_cable_annotated.mp4"),
                            fourcc, fps, (w, h), isColor=True)
    msk_w = cv2.VideoWriter(os.path.join(out_dir, "rgb_cable_mask.mp4"),
                            fourcc, fps, (w, h), isColor=False)

    frames, total_px, boxes = 0, 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # YOLO first (every YOLO_EVERY-th frame; reuse boxes between)
        # so the box region can scope the red detection.
        if model is not None and frames % YOLO_EVERY == 0:
            boxes = []
            try:
                res = model(fr, verbose=False, conf=YOLO_CONF)[0]
                for b in res.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cid = int(b.cls[0])
                    boxes.append((x1, y1, x2, y2,
                                  res.names.get(cid, str(cid)),
                                  float(b.conf[0])))
            except Exception as e:
                print("  ! yolo error:", e)
        roi = roi_from_boxes(boxes, w, h) if BETWEEN_BOXES else None
        mask = red_cable_mask(fr, roi)
        total_px += int(np.count_nonzero(mask))
        ann_w.write(annotate(fr, mask, boxes, roi))
        msk_w.write(mask)
        frames += 1
    cap.release()
    ann_w.release()
    msk_w.release()
    return frames, total_px


def write_config():
    """Write an analyzer-loadable config JSON holding the tuned
    red-cable settings (HSV hue + LAB red mode + the morphology
    pipeline). Load it from the GUI with Ctrl+O."""
    cfg = {
        "version": 1,
        "params": {
            "H1_low": HSV_LO_1[0], "H1_high": HSV_HI_1[0],
            "H2_low": HSV_LO_2[0], "H2_high": HSV_HI_2[0],
            "S_min": HSV_LO_1[1],  "S_max": 255,
            "V_min": HSV_LO_1[2],  "V_max": 255,
            "L_min": LAB_LO[0], "L_max": LAB_HI[0],
            "A_min": LAB_LO[1], "A_max": LAB_HI[1],
            "B_min": LAB_LO[2], "B_max": LAB_HI[2],
            "YOLO_Conf": int(YOLO_CONF * 100),
            "YOLO_Box_Scale": 100, "YOLO_Box_Pad": 0,
            "Min_area": MIN_AREA,
        },
        "flags": {
            "use_bgsub_rgb": False, "use_bgsub_ir": False,
            "show_boxes": True, "show_overlay": True,
            "use_yolo": True, "loop_var": False, "use_clahe_ir": False,
        },
        "ints": {"clahe_ir_clip": 2, "clahe_ir_tile": 8},
        "strings": {
            "rgb_detect_mode": "HSV hue + LAB (red)",
            "ir_cmap": "Gray", "kernel_shape": "Ellipse",
            "rgb_mask_color": "#00ffff", "ir_mask_color": "#ffff00",
            "yolo_mask_color": "#ff00ff",
            "bgsub_rgb_src": "auto", "bgsub_ir_src": "auto",
            "combine_a": "none", "combine_b": "none",
            "combine_op": "none", "combine_c1": "red",
            "combine_c2": "green",
        },
        # Close (bridge gaps) then Dilate (thicken the thin cable).
        "rgb_pipeline": [
            {"en": True, "op": "Close",  "n": 1, "dir": "Both",
             "kx": 5, "ky": 5, "t": 0},
            {"en": True, "op": "Dilate", "n": 1, "dir": "Both",
             "kx": 3, "ky": 3, "t": 0},
        ],
        "ir_pipeline": [], "rgb_pre_pipeline": [],
        "ir_pre_pipeline": [], "user_pipelines": [],
    }
    with open(CFG_OUT, "w") as f:
        json.dump(cfg, f, indent=2)
    print("config written ->", CFG_OUT)


def main():
    # Load the YOLO model (drone + T-hook). Optional: detection still
    # runs without it, just with no boxes.
    model = None
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_PATH):
            model = YOLO(YOLO_PATH)
            print("YOLO loaded:", YOLO_PATH, model.names)
        else:
            print("YOLO model not found, running colour-only:", YOLO_PATH)
    except Exception as e:
        print("ultralytics unavailable, colour-only:", e)

    if len(sys.argv) > 1:                       # single-video mode
        vids = [sys.argv[1]]
    else:
        vids = sorted(glob.glob(os.path.join(REC_DIR, "*", "rgb.mp4")))
    if not vids:
        print("no rgb.mp4 found under", REC_DIR)
        return

    write_config()
    print("processing %d video(s)\n" % len(vids))
    t0 = time.time()
    for i, v in enumerate(vids, 1):
        name = os.path.basename(os.path.dirname(v))
        ts = time.time()
        frames, px = detect_video(v, model)
        print("[%2d/%2d] %s  %d frames  %d cable-px  %.1fs"
              % (i, len(vids), name, frames, px, time.time() - ts))
    print("\ndone: %d video(s) in %.1fs" % (len(vids), time.time() - t0))
    print("outputs: rgb_cable_annotated.mp4 + rgb_cable_mask.mp4 "
          "in each recording folder")


if __name__ == "__main__":
    main()
