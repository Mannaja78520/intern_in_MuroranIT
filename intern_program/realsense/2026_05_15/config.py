"""Configuration constants for the RealSense Cable Video Analyzer.

All tunable constants live here so they can be edited in one place without
touching application logic.
"""
import cv2

# -- File-system paths -----------------------------------------------------
RECORDINGS_DIR     = "/home/mannaja/intern_in_MuroranIT/intern_program/videos/realsense/recordings"
YOLO_MODEL_PATH    = "/home/mannaja/intern_in_MuroranIT/intern_program/models/drone_t_hook.pt"
SCREENSHOTS_DIR    = "/home/mannaja/intern_in_MuroranIT/intern_program/pictures/screenshots"

# -- Main panel display size -----------------------------------------------
DISPLAY_W, DISPLAY_H = 320, 240

# -- IR colour-map options -------------------------------------------------
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

# -- Morphological operations ----------------------------------------------
MORPH_OPS = {
    "Close":    cv2.MORPH_CLOSE,
    "Open":     cv2.MORPH_OPEN,
    "Dilate":   cv2.MORPH_DILATE,
    "Erode":    cv2.MORPH_ERODE,
    "Gradient": cv2.MORPH_GRADIENT,
    "TopHat":   cv2.MORPH_TOPHAT,
    "BlackHat": cv2.MORPH_BLACKHAT,
}

# -- Extra (non-morph) processing operations -------------------------------
EXTRA_OPS = [
    # -- Blur ----------------------------------------------------------
    "GaussBlur",          # KX = kernel size
    "MedianBlur",         # KX = kernel size
    "BilateralBlur",      # KX = diameter, KY = sigma
    # -- Threshold -----------------------------------------------------
    "Thresh_Binary",      # T = threshold
    "Thresh_Otsu",        # (auto)
    "Thresh_Adaptive",    # KX = block size, KY = C
    # -- Illumination correction ----------------------------------------
    "HistEq",             # global histogram equalisation (no params)
    "CLAHE",              # KX = clip limit, KY = tile grid size
    "Gamma",              # KX = gammax10  (KX=10 -> gamma=1.0, KX=5 -> gamma=0.5)
    "Normalize",          # stretch to 0-255 (no params)
    "Retinex",            # single-scale retinex, KX = sigma
    # -- Edge / gradient -----------------------------------------------
    "Sharpen",            # unsharp mask, KX = kernel size
    "Laplacian",          # Laplacian edge, KX = kernel size (1/3/5)
    "Sobel",              # gradient magnitude, KX = kernel size
    "Canny",              # KX = low threshold, KY = high threshold
    # -- Combine with other step ----------------------------------------
    "AND_prev",           # AND current mask with previous step snapshot
    "OR_prev",            # OR  current mask with previous step snapshot
    "XOR_prev",           # XOR current mask with previous step snapshot
    "AND_input",          # AND current mask with pipeline input (mask_pre)
    "OR_input",           # OR  current mask with pipeline input
    "XOR_input",          # XOR current mask with pipeline input
    # -- Misc ----------------------------------------------------------
    "Invert",
    "FillHoles",
]

ALL_PROC_OPS = list(MORPH_OPS.keys()) + EXTRA_OPS

# Op groups for the cascading "Group -> Op" picker in step cards. Pick a
# group first, then narrow to the actual operation. Shows ~5 ops per
# group instead of 30 in a flat list.
OP_GROUPS = {
    "Morphology":   ["Close", "Open", "Dilate", "Erode",
                     "Gradient", "TopHat", "BlackHat"],
    "Blur":         ["GaussBlur", "MedianBlur", "BilateralBlur"],
    "Threshold":    ["Thresh_Binary", "Thresh_Otsu", "Thresh_Adaptive"],
    "Illumination": ["HistEq", "CLAHE", "Gamma", "Normalize", "Retinex"],
    "Edge":         ["Sharpen", "Laplacian", "Sobel", "Canny"],
    "Bitwise":      ["AND_prev", "OR_prev",  "XOR_prev",
                     "AND_input", "OR_input", "XOR_input"],
    "Misc":         ["Invert", "FillHoles"],
}

def group_for_op(op_name):
    """Return the group that contains `op_name`, or 'Misc' as fallback."""
    for g, ops in OP_GROUPS.items():
        if op_name in ops:
            return g
    return "Misc"


# Op groups that semantically operate on the RAW source IMAGE (the
# blurred RGB frame or the IR gray frame) rather than on a binary
# MASK. Pipeline steps with one of these ops run as PRE-MORPH steps
# BEFORE HSV / IR thresholding — they modify the source image, then
# detection runs on the modified image, then the rest of the steps
# (morph / threshold / bitwise / misc) operate on the resulting mask.
PRE_MORPH_OP_GROUPS = {"Blur", "Illumination", "Edge"}


def is_pre_morph_op(op_name):
    """True if `op_name` belongs to a group that consumes the source
    image (not the binary mask)."""
    return group_for_op(op_name) in PRE_MORPH_OP_GROUPS


# Per-op visible parameters. Each value is the subset of
# {N, Dir, KX, KY, T} that this op actually uses, plus per-op labels
# so the user sees what the kernel/threshold actually means.
#   "label" - heading shown above the param row
#   "kx_lbl"/"ky_lbl"/"t_lbl"/"n_lbl"/"dir_lbl" - per-row labels
OP_PARAMS = {
    # -- Morphology - uses kernel + iterations + direction --
    "Close":     {"params": ["N", "Dir", "KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "n_lbl":  "Iter.",
                  "desc":   "Close small holes inside white regions "
                            "(dilate -> erode). Bigger Kernel/Iter -> "
                            "fills bigger gaps."},
    "Open":      {"params": ["N", "Dir", "KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "n_lbl":  "Iter.",
                  "desc":   "Remove tiny noise specks (erode -> dilate)."
                            " Bigger Kernel -> wipes bigger specks."},
    "Dilate":    {"params": ["N", "Dir", "KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "n_lbl":  "Iter.",
                  "desc":   "Grow white regions outwards. "
                            "Iter ^ -> grow more."},
    "Erode":     {"params": ["N", "Dir", "KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "n_lbl":  "Iter.",
                  "desc":   "Shrink white regions inwards. "
                            "Iter ^ -> shrink more."},
    "Gradient":  {"params": ["KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "desc":   "Outline of white regions (dilate - erode)."},
    "TopHat":    {"params": ["KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "desc":   "Bright spots smaller than the kernel "
                            "(input - Open)."},
    "BlackHat":  {"params": ["KX", "KY"],
                  "kx_lbl": "Kernel X", "ky_lbl": "Kernel Y",
                  "desc":   "Dark spots smaller than the kernel "
                            "(Close - input)."},
    # -- Blur --
    "GaussBlur":     {"params": ["KX"],
                      "kx_lbl": "Kernel size (odd)",
                      "desc":   "Gaussian blur. Bigger kernel -> softer."},
    "MedianBlur":    {"params": ["KX"],
                      "kx_lbl": "Kernel size (odd)",
                      "desc":   "Median blur - strong salt-&-pepper "
                                "noise removal, preserves edges."},
    "BilateralBlur": {"params": ["KX", "KY"],
                      "kx_lbl": "Diameter", "ky_lbl": "Sigma",
                      "desc":   "Edge-preserving blur. Higher Sigma -> "
                                "smoother in flat areas."},
    # -- Threshold --
    "Thresh_Binary":   {"params": ["T"],     "t_lbl":  "Threshold",
                        "desc":   "Pixels > T -> 255, else 0."},
    "Thresh_Otsu":     {"params": [],
                        "desc":   "Auto-pick T via Otsu's method "
                                  "(bimodal histogram split)."},
    "Thresh_Adaptive": {"params": ["KX", "KY"],
                        "kx_lbl": "Block size", "ky_lbl": "C constant",
                        "desc":   "Per-pixel threshold = local mean - C "
                                  "over block. Robust to lighting."},
    # -- Illumination --
    "HistEq":     {"params": [],
                   "desc":   "Stretch pixel histogram across 0-255 "
                             "(global contrast boost)."},
    "CLAHE":      {"params": ["KX", "KY"],
                   "kx_lbl": "Clip limit", "ky_lbl": "Tile grid",
                   "desc":   "Local histogram equalisation per tile. "
                             "Higher Clip -> stronger contrast."},
    "Gamma":      {"params": ["KX"],
                   "kx_lbl": "Gamma x 10",
                   "desc":   "Power-law brightness curve. KX=10 -> gamma=1.0 "
                             "(no change); <10 brightens, >10 darkens."},
    "Normalize":  {"params": [],
                   "desc":   "Linear stretch min/max of pixels to 0-255."},
    "Retinex":    {"params": ["KX"],
                   "kx_lbl": "Sigma",
                   "desc":   "Single-scale retinex - illumination "
                             "removal. Bigger Sigma -> broader correction."},
    # -- Edge --
    "Sharpen":    {"params": ["KX"], "kx_lbl": "Kernel size (odd)",
                   "desc":   "Unsharp mask - sharpens fine detail."},
    "Laplacian":  {"params": ["KX"], "kx_lbl": "Kernel size (1/3/5)",
                   "desc":   "Second-derivative edge magnitude."},
    "Sobel":      {"params": ["KX"], "kx_lbl": "Kernel size",
                   "desc":   "Sobel gradient magnitude (X+Y)."},
    "Canny":      {"params": ["KX", "KY"],
                   "kx_lbl": "Low threshold", "ky_lbl": "High threshold",
                   "desc":   "Canny edge detector. Pixels above High = "
                             "edge; pixels above Low = edge if connected."},
    # -- Bitwise (no params) --
    "AND_prev":   {"params": [],
                   "desc":   "Bitwise-AND with the PREVIOUS step's mask."},
    "OR_prev":    {"params": [],
                   "desc":   "Bitwise-OR with the PREVIOUS step's mask."},
    "XOR_prev":   {"params": [],
                   "desc":   "Bitwise-XOR with the PREVIOUS step's mask."},
    "AND_input":  {"params": [],
                   "desc":   "Bitwise-AND with the pipeline INPUT mask."},
    "OR_input":   {"params": [],
                   "desc":   "Bitwise-OR with the pipeline INPUT mask."},
    "XOR_input":  {"params": [],
                   "desc":   "Bitwise-XOR with the pipeline INPUT mask."},
    # -- Misc --
    "Invert":     {"params": [],
                   "desc":   "Flip mask: white <-> black."},
    "FillHoles":  {"params": [],
                   "desc":   "Fill enclosed black regions inside white. "
                             "Useful after Canny / Threshold."},
}

def params_for_op(op_name):
    """Return param spec dict for op, or sane fallback."""
    return OP_PARAMS.get(op_name, {"params": ["N", "Dir", "KX", "KY", "T"]})

KERNEL_SHAPES = {
    "Rect":    cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross":   cv2.MORPH_CROSS,
}

# -- Panel-view dropdown options (supports up to 20 pipeline steps) --------
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

# -- All-Masks flowchart layout --------------------------------------------
FC_W, FC_H     = 144, 108   # thumbnail size per node
STEPS_PER_ROW  = 7          # max steps before wrapping to a new row
STEP_ROW_BASE  = 3          # first row used for RGB step nodes (IR uses 1)
YOLO_COL       = 7          # column reserved for the YOLO bypass
