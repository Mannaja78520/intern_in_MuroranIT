"""Configuration constants for the RealSense Cable Video Analyzer.

All tunable constants live here so they can be edited in one place without
touching application logic.
"""
import cv2

# ── File-system paths ─────────────────────────────────────────────────────
RECORDINGS_DIR     = "/home/mannaja/intern_in_MuroranIT/intern_program/videos/realsense/recordings"
YOLO_MODEL_PATH    = "/home/mannaja/intern_in_MuroranIT/intern_program/models/realsense_merged.pt"
SCREENSHOTS_DIR    = "/home/mannaja/intern_in_MuroranIT/intern_program/pictures/screenshots"

# ── Main panel display size ───────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 320, 240

# ── IR colour-map options ─────────────────────────────────────────────────
IR_COLORMAPS = {
    "Gray":    None,
    "Jet":     cv2.COLORMAP_JET,
    "Hot":     cv2.COLORMAP_HOT,
    "Bone":    cv2.COLORMAP_BONE,
    "Rainbow": cv2.COLORMAP_RAINBOW,
    "Ocean":   cv2.COLORMAP_OCEAN,
    "Inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_HOT),
    "Plasma":  getattr(cv2, "COLORMAP_PLASMA",  cv2.COLORMAP_HOT),
    "Magma":   getattr(cv2, "COLORMAP_MAGMA",   cv2.COLORMAP_HOT),
    "Viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
    "Turbo":   getattr(cv2, "COLORMAP_TURBO",   cv2.COLORMAP_JET),
}

# ── Morphological operations ──────────────────────────────────────────────
MORPH_OPS = {
    "Close":    cv2.MORPH_CLOSE,
    "Open":     cv2.MORPH_OPEN,
    "Dilate":   cv2.MORPH_DILATE,
    "Erode":    cv2.MORPH_ERODE,
    "Gradient": cv2.MORPH_GRADIENT,
    "TopHat":   cv2.MORPH_TOPHAT,
    "BlackHat": cv2.MORPH_BLACKHAT,
}

# ── Extra (non-morph) processing operations ───────────────────────────────
EXTRA_OPS = [
    # ── Blur ──────────────────────────────────────────────────────────
    "GaussBlur",          # KX = kernel size
    "MedianBlur",         # KX = kernel size
    "BilateralBlur",      # KX = diameter, KY = sigma
    # ── Threshold ─────────────────────────────────────────────────────
    "Thresh_Binary",      # T = threshold
    "Thresh_Otsu",        # (auto)
    "Thresh_Adaptive",    # KX = block size, KY = C
    # ── Illumination correction ────────────────────────────────────────
    "HistEq",             # global histogram equalisation (no params)
    "CLAHE",              # KX = clip limit, KY = tile grid size
    "Gamma",              # KX = gamma×10  (KX=10 → γ=1.0, KX=5 → γ=0.5)
    "Normalize",          # stretch to 0-255 (no params)
    "Retinex",            # single-scale retinex, KX = sigma
    # ── Edge / gradient ───────────────────────────────────────────────
    "Sharpen",            # unsharp mask, KX = kernel size
    "Laplacian",          # Laplacian edge, KX = kernel size (1/3/5)
    "Sobel",              # gradient magnitude, KX = kernel size
    "Canny",              # KX = low threshold, KY = high threshold
    # ── Combine with other step ────────────────────────────────────────
    "AND_prev",           # AND current mask with previous step snapshot
    "OR_prev",            # OR  current mask with previous step snapshot
    "XOR_prev",           # XOR current mask with previous step snapshot
    "AND_input",          # AND current mask with pipeline input (mask_pre)
    "OR_input",           # OR  current mask with pipeline input
    "XOR_input",          # XOR current mask with pipeline input
    # ── Misc ──────────────────────────────────────────────────────────
    "Invert",
    "FillHoles",
]

ALL_PROC_OPS = list(MORPH_OPS.keys()) + EXTRA_OPS

KERNEL_SHAPES = {
    "Rect":    cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross":   cv2.MORPH_CROSS,
}

# ── Panel-view dropdown options (supports up to 20 pipeline steps) ────────
VIEW_OPTIONS = [
    "rgb_raw",       "rgb_blur",      "rgb_hsv_full",  "rgb_hsv_H",    "rgb_hsv_S",
    "rgb_hsv_V",     "rgb_m1",        "rgb_m2",        "rgb_hsv_mask", "rgb_bgsub",
    "rgb_mask_pre",
    *[f"rgb_step{i}" for i in range(1, 21)],
    "rgb_mask",      "rgb_post_blur", "rgb_det",
    "ir_raw",        "ir_gray",       "ir_blur",        "ir_clahe",
    "ir_bgsub",      "ir_thresh",     "ir_mask_pre",
    *[f"ir_step{i}"  for i in range(1, 21)],
    "ir_mask",       "ir_post_blur",  "ir_det",
    "combined",
]

# ── All-Masks flowchart layout ────────────────────────────────────────────
FC_W, FC_H     = 144, 108   # thumbnail size per node
STEPS_PER_ROW  = 7          # max steps before wrapping to a new row
STEP_ROW_BASE  = 2          # first row used for RGB step nodes (IR uses 1)
YOLO_COL       = 7          # column reserved for the YOLO bypass
