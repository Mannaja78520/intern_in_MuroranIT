"""Mixin: export the whole detection pipeline.

`_export_pipeline` writes two files next to a user-chosen path:

  * ``<name>.py``  - a SELF-CONTAINED script. It needs only
                     opencv-python + numpy (no part of this program)
                     and exposes a ``detect(bgr_image, ir_image=None)``
                     function plus a CLI runner, so the tuned pipeline
                     can be reused in any other Python code.
  * ``<name>.png`` - a flowchart diagram of the same pipeline.

The exported script faithfully reproduces:
  - RGB detection in every mode (hue / achroma / value / lab / hsv_lab)
  - IR grayscale threshold detection (+ optional CLAHE)
  - the pre-morph (image-conditioning) step stage
  - the morph / extra-op step pipeline incl. AND/OR/XOR combine
  - every user branch (rgb / ir / custom) and its step pipeline

YOLO inference, MOG2 background subtraction and the OVERLAY combine
(visualisation-only / model- or video-dependent) are intentionally
left out of the standalone script and noted as comments there.
"""
import os
import json
import tkinter as tk

from config import RGB_DETECT_MODES, OP_PARAMS, YOLO_MODEL_PATH


def _fmt_step_params(st):
    """Compact human-readable parameter string for one step dict, e.g.
    'kx=5 ky=5 iter=2' — used in the PNG flowchart + settings report."""
    spec = OP_PARAMS.get(st.get("op", ""), {})
    ps   = spec.get("params", [])
    out  = []
    if "N"  in ps:
        out.append("iter=%d" % st.get("n", 1))
    if "KX" in ps:
        out.append("kx=%d" % st.get("kx", 3))
    if "KY" in ps:
        out.append("ky=%d" % st.get("ky", 3))
    if "T"  in ps:
        out.append("t=%d" % st.get("t", 0))
    if "Dir" in ps:
        d = st.get("dir", "")
        if d and d not in ("Both", "XY"):
            out.append("dir=%s" % d)
    ks = st.get("kshape")
    if ks and ks != "Rect":
        out.append("shape=%s" % ks)
    return " ".join(out)


def _step_line(idx, st):
    """One '3. GaussBlur  kx=5  (on V)' report line for a step dict."""
    if st.get("comb_en"):
        head = "combine %s  with %s" % (st.get("comb_op", "AND"),
                                        st.get("comb_src", "?"))
        params = ""
    else:
        head   = st.get("op", "?")
        params = _fmt_step_params(st)
    tgt = st.get("pm_target")
    suffix = ("  (on %s)" % tgt) if tgt and tgt != "BGR" else ""
    return "  %2d. %-16s %s%s" % (idx, head, params, suffix)


# ======================================================================
#  Op library — shared verbatim by BOTH exports:
#    * the self-contained <name>.py        (CONFIG baked in)
#    * the reusable <name>_lib.py library  (CONFIG loaded from JSON)
#  Pure functions: opencv + numpy only, no tkinter, no this program.
# ======================================================================
_OPS_SOURCE = '''
# ----------------------------------------------------------------------
#  Op library (faithful copy of the analyzer's processing operations)
# ----------------------------------------------------------------------
MORPH_OPS = {
    "Close":    cv2.MORPH_CLOSE,    "Open":     cv2.MORPH_OPEN,
    "Dilate":   cv2.MORPH_DILATE,   "Erode":    cv2.MORPH_ERODE,
    "Gradient": cv2.MORPH_GRADIENT, "TopHat":   cv2.MORPH_TOPHAT,
    "BlackHat": cv2.MORPH_BLACKHAT,
}
KERNEL_SHAPES = {
    "Rect":    cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross":   cv2.MORPH_CROSS,
}
RGB_DETECT_MODES = ''' + repr(dict(RGB_DETECT_MODES)) + '''


def _odd(v):
    v = int(v)
    return v if v % 2 == 1 else v + 1


def _fill_holes(m):
    flood  = m.copy()
    h, w   = m.shape[:2]
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    return cv2.bitwise_or(m, cv2.bitwise_not(flood))


def apply_op(mask, op_name, n, kx, ky, d, thresh,
             prev=None, inp=None, kshape=cv2.MORPH_RECT):
    """Apply one morphology / extra op to a (mask or image) array."""
    n  = max(1, int(n)); kx = max(1, int(kx)); ky = max(1, int(ky))
    if op_name in MORPH_OPS:
        if d == "X":   sk = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        elif d == "Y": sk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
        else:          sk = cv2.getStructuringElement(kshape,         (kx, ky))
        return cv2.morphologyEx(mask, MORPH_OPS[op_name], sk, iterations=n)
    elif op_name == "GaussBlur":
        return cv2.GaussianBlur(mask, (_odd(max(1, kx)),) * 2, 0)
    elif op_name == "MedianBlur":
        return cv2.medianBlur(mask, _odd(max(1, kx)))
    elif op_name == "BilateralBlur":
        return cv2.bilateralFilter(mask, kx, float(ky), float(ky))
    elif op_name == "Thresh_Binary":
        _, m = cv2.threshold(mask, max(0, min(255, int(thresh))),
                             255, cv2.THRESH_BINARY)
        return m
    elif op_name == "Thresh_Otsu":
        _, m = cv2.threshold(mask, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return m
    elif op_name == "Thresh_Adaptive":
        return cv2.adaptiveThreshold(mask, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY, _odd(max(3, kx)), ky)
    elif op_name == "HistEq":
        return cv2.equalizeHist(mask)
    elif op_name == "CLAHE":
        cl = cv2.createCLAHE(clipLimit=float(max(1, kx)),
                             tileGridSize=(max(1, ky),) * 2)
        return cl.apply(mask)
    elif op_name == "Gamma":
        gamma = max(0.1, kx / 10.0)
        lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                        for i in range(256)], dtype=np.uint8)
        return lut[mask]
    elif op_name == "Normalize":
        lo, hi = int(mask.min()), int(mask.max())
        if hi > lo:
            return cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return mask
    elif op_name == "Retinex":
        sigma = float(max(1, kx))
        blur  = cv2.GaussianBlur(mask.astype(np.float32) + 1,
                                 (_odd(int(sigma) * 3 | 1),) * 2, sigma)
        r     = np.log1p(mask.astype(np.float32)) - np.log(blur)
        return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
    elif op_name == "Sharpen":
        k   = _odd(max(1, kx))
        blr = cv2.GaussianBlur(mask, (k, k), 0)
        return cv2.addWeighted(mask, 2.0, blr, -1.0, 0)
    elif op_name == "Laplacian":
        ksize = kx if kx in (1, 3, 5) else 3
        lap = cv2.Laplacian(mask, cv2.CV_16S, ksize=ksize)
        return cv2.convertScaleAbs(lap)
    elif op_name == "Sobel":
        ksize = kx if kx in (1, 3, 5, 7) else 3
        sx  = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=ksize)
        sy  = cv2.Sobel(mask, cv2.CV_16S, 0, 1, ksize=ksize)
        mag = cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32))
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
    elif op_name == "Canny":
        return cv2.Canny(mask, max(1, kx), max(1, ky))
    elif op_name in ("AND_prev", "OR_prev", "XOR_prev"):
        ref = prev if prev is not None else mask
        if op_name == "AND_prev": return cv2.bitwise_and(mask, ref)
        if op_name == "OR_prev":  return cv2.bitwise_or(mask, ref)
        return cv2.bitwise_xor(mask, ref)
    elif op_name in ("AND_input", "OR_input", "XOR_input"):
        ref = inp if inp is not None else mask
        if op_name == "AND_input": return cv2.bitwise_and(mask, ref)
        if op_name == "OR_input":  return cv2.bitwise_or(mask, ref)
        return cv2.bitwise_xor(mask, ref)
    elif op_name == "Invert":
        return cv2.bitwise_not(mask)
    elif op_name == "FillHoles":
        return _fill_holes(mask)
    return mask


def apply_op_image(img, op_name, n, kx, ky, d, thresh,
                   kshape=cv2.MORPH_RECT):
    """Apply a PRE-MORPH op to a BGR / grayscale IMAGE (keeps colour)."""
    is_bgr = (img.ndim == 3 and img.shape[2] == 3)
    if op_name in ("GaussBlur", "MedianBlur", "BilateralBlur"):
        return apply_op(img, op_name, n, kx, ky, d, thresh, kshape=kshape)
    if op_name in ("HistEq", "CLAHE", "Gamma", "Normalize", "Retinex"):
        if not is_bgr:
            return apply_op(img, op_name, n, kx, ky, d, thresh, kshape=kshape)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = apply_op(lab[:, :, 0], op_name, n, kx, ky, d,
                                thresh, kshape=kshape)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if op_name == "Sharpen":
        if not is_bgr:
            return apply_op(img, op_name, n, kx, ky, d, thresh, kshape=kshape)
        return cv2.merge([apply_op(img[:, :, i], op_name, n, kx, ky, d,
                                   thresh, kshape=kshape) for i in range(3)])
    if op_name in ("Laplacian", "Sobel", "Canny"):
        if not is_bgr:
            return apply_op(img, op_name, n, kx, ky, d, thresh, kshape=kshape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out  = apply_op(gray, op_name, n, kx, ky, d, thresh, kshape=kshape)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if op_name in MORPH_OPS:
        return apply_op(img, op_name, n, kx, ky, d, thresh, kshape=kshape)
    return img


def apply_pm_targeted(img, target, op_name, n, kx, ky, d, thresh,
                      kshape=cv2.MORPH_RECT):
    """Apply a PM op to the whole image OR a single HSV channel."""
    if target in ("H", "S", "V") and img.ndim == 3 and img.shape[2] == 3:
        ci   = {"H": 0, "S": 1, "V": 2}[target]
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ch2  = apply_op_image(hsv[:, :, ci], op_name, n, kx, ky, d,
                              thresh, kshape=kshape)
        if ch2.ndim == 3:
            ch2 = cv2.cvtColor(ch2, cv2.COLOR_BGR2GRAY)
        if ch2.shape != hsv[:, :, ci].shape:
            ch2 = cv2.resize(ch2, (hsv.shape[1], hsv.shape[0]))
        hsv[:, :, ci] = ch2
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return apply_op_image(img, op_name, n, kx, ky, d, thresh, kshape=kshape)


def _kshape_of(step, default_shape):
    return KERNEL_SHAPES.get(step.get("kshape", ""), default_shape)


# ----------------------------------------------------------------------
#  Pre-morph + step pipeline runners
# ----------------------------------------------------------------------
def run_pre_morph(image, pre_steps, default_shape):
    """Apply the pre-morph (image-conditioning) steps in order."""
    running = image
    for st in pre_steps:
        if not st.get("en"):
            continue
        running = apply_pm_targeted(
            running, st.get("pm_target", "BGR"), st.get("op", "GaussBlur"),
            max(1, st.get("n", 1)), max(1, st.get("kx", 3)),
            max(1, st.get("ky", 3)), st.get("dir", "XY"),
            st.get("t", 0), kshape=_kshape_of(st, default_shape))
    return running


def _coerce_mask(img, like):
    if img is None:
        return like
    m = img
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape != like.shape:
        m = cv2.resize(m, (like.shape[1], like.shape[0]))
    if m.dtype != like.dtype:
        m = m.astype(like.dtype)
    return m


def _resolve_src(snaps, src, mask_pre, views):
    """Resolve a combine-source name to an array."""
    if src == "mask_pre":
        return mask_pre
    if src == "prev":
        return snaps[-1] if snaps else mask_pre
    if src.startswith("step_"):
        try:
            i = int(src.split("_")[1]) - 1
            return snaps[i] if 0 <= i < len(snaps) else mask_pre
        except (ValueError, IndexError):
            return mask_pre
    if src in views:
        return views[src]
    return mask_pre   # cross-pipeline ref not available -> input mask


def run_steps(mask_pre, steps, default_shape, views, view_prefix=None):
    """Run a morph / extra-op step pipeline. Returns (final, snaps).

    `snaps` stores REFERENCES (no per-step .copy()): every op returns a
    fresh array and nothing is mutated in place, so copying each step
    just to keep a snapshot only wasted time on long pipelines.
    """
    running = mask_pre
    snaps   = []
    for idx, st in enumerate(steps):
        if not st.get("en"):
            snaps.append(running)
            if view_prefix is not None:
                views["%s_step%d" % (view_prefix, idx + 1)] = running
            continue
        if st.get("comb_en"):
            ref = _coerce_mask(
                _resolve_src(snaps, st.get("comb_src", "mask_pre"),
                             mask_pre, views), running)
            cop = st.get("comb_op", "AND")
            if cop == "AND":
                running = cv2.bitwise_and(running, ref)
            elif cop == "OR":
                running = cv2.bitwise_or(running, ref)
            elif cop == "XOR":
                running = cv2.bitwise_xor(running, ref)
            # cop == "OVERLAY": visualisation only -> mask unchanged.
        else:
            prev = snaps[-1] if snaps else None
            running = apply_op(
                running, st.get("op", "Dilate"),
                max(1, st.get("n", 1)), max(1, st.get("kx", 3)),
                max(1, st.get("ky", 3)), st.get("dir", "Both"),
                st.get("t", 0), prev=prev, inp=mask_pre,
                kshape=_kshape_of(st, default_shape))
        snaps.append(running)
        if view_prefix is not None:
            views["%s_step%d" % (view_prefix, idx + 1)] = running
    return running, snaps


# ----------------------------------------------------------------------
#  Detection
# ----------------------------------------------------------------------
def rgb_detect(bgr, P, mode_key):
    """RGB binary detection in the selected mode."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    smin, smax = P["S_min"], P["S_max"]
    vmin, vmax = P["V_min"], P["V_max"]
    h1l, h1h   = P["H1_low"], P["H1_high"]
    h2l, h2h   = P["H2_low"], P["H2_high"]

    def labm():
        return cv2.inRange(lab,
                           (P.get("L_min", 0), P["A_min"], P["B_min"]),
                           (P.get("L_max", 255), P["A_max"], P["B_max"]))
    if mode_key == "achroma":
        return cv2.inRange(hsv, (0, smin, vmin), (180, smax, vmax))
    if mode_key == "value":
        return cv2.inRange(hsv, (0, 0, vmin), (180, 255, vmax))
    if mode_key == "lab":
        return labm()
    if mode_key == "hsv_lab":
        a = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        b = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        return cv2.bitwise_and(cv2.bitwise_or(a, b), labm())
    # default: hue
    a = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
    b = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
    return cv2.bitwise_or(a, b)


def branch_rgb_detect(bgr, bd):
    """Branch RGB detection: AND of the channels named by 'channels'."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h1 = cv2.inRange(hsv[:, :, 0], bd.get("h1_lo", 0),   bd.get("h1_hi", 10))
    h2 = cv2.inRange(hsv[:, :, 0], bd.get("h2_lo", 170), bd.get("h2_hi", 180))
    s  = cv2.inRange(hsv[:, :, 1], bd.get("s_lo", 0),    bd.get("s_hi", 255))
    v  = cv2.inRange(hsv[:, :, 2], bd.get("v_lo", 0),    bd.get("v_hi", 255))
    a  = cv2.inRange(lab[:, :, 1], bd.get("a_lo", 150),  bd.get("a_hi", 255))
    b  = cv2.inRange(lab[:, :, 2], bd.get("b_lo", 0),    bd.get("b_hi", 255))
    ch = bd.get("channels", "HSV")
    to_and = []
    if "H" in ch or ch == "full":
        to_and.append(cv2.bitwise_or(h1, h2))
    if "S" in ch or ch == "full":
        to_and.append(s)
    if "V" in ch or ch == "full":
        to_and.append(v)
    if "a" in ch:
        to_and.append(a)
    if "b" in ch:
        to_and.append(b)
    if not to_and:
        return np.zeros(bgr.shape[:2], np.uint8)
    det = to_and[0]
    for m in to_and[1:]:
        det = cv2.bitwise_and(det, m)
    return det


def find_objects(mask, min_area):
    """Return [(x, y, w, h), ...] for blobs at least `min_area` px."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            boxes.append(cv2.boundingRect(c))
    return boxes


def adjust_box(x1, y1, x2, y2, w, h, scale=1.0, pad=0):
    """Grow / shrink a YOLO box: scale (%) around its centre, then add
    `pad` pixels on every side (negative trims). Clamped to w x h.
    Lets the exported pipeline make YOLO boxes bigger (margin to
    subtract) or smaller (focus tightly inside the box)."""
    if scale != 1.0 or pad:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw = (x2 - x1) * scale + 2 * pad
        bh = (y2 - y1) * scale + 2 * pad
        x1, x2 = cx - bw / 2.0, cx + bw / 2.0
        y1, y2 = cy - bh / 2.0, cy + bh / 2.0
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2
'''


# ======================================================================
#  CONFIG-driven tail for the self-contained <name>.py export. CONFIG
#  is a module-global baked into the file; detect() reads it directly.
# ======================================================================
_RUNTIME_TAIL = '''

# ----------------------------------------------------------------------
#  YOLO object detection (optional add-on stage)
#
#  The 'ultralytics' package is imported LAZILY inside _get_yolo_model,
#  so a config with YOLO disabled never needs it installed. When the
#  analyzer had "Use YOLO" ticked, detect() runs this automatically.
# ----------------------------------------------------------------------
_YOLO_MODEL = None


def _get_yolo_model():
    """Lazy-load the YOLO model (imports ultralytics only when called)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        path = CONFIG.get("export_yolo", {}).get("model_path", "yolo.pt")
        _YOLO_MODEL = YOLO(path)
    return _YOLO_MODEL


def run_yolo(bgr, conf=None):
    """Run YOLO on a BGR image.

    Returns (boxes, mask):
      boxes - list of dicts {x, y, w, h, cls, name, conf}
      mask  - uint8 image, white inside every detected box
    Needs `pip install ultralytics` and the model weights file
    (path is baked into CONFIG['export_yolo']['model_path']).
    """
    if conf is None:
        conf = CONFIG.get("export_yolo", {}).get("conf", 0.5)
    # YOLO box size tuning — scale % + pixel pad, from the tuned config.
    _P     = CONFIG.get("params", {})
    _scale = _P.get("YOLO_Box_Scale", 100) / 100.0
    _pad   = int(_P.get("YOLO_Box_Pad", 0))
    model = _get_yolo_model()
    boxes = []
    mask  = np.zeros(bgr.shape[:2], np.uint8)
    h, w  = bgr.shape[:2]
    for r in model(bgr, conf=conf, verbose=False):
        names = getattr(r, "names", {}) or {}
        for b in getattr(r, "boxes", []):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            x1, y1, x2, y2 = adjust_box(x1, y1, x2, y2, w, h,
                                        _scale, _pad)
            cid = int(b.cls[0])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                          "cls": cid, "name": names.get(cid, str(cid)),
                          "conf": float(b.conf[0])})
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return boxes, mask


# ----------------------------------------------------------------------
#  Top-level pipeline
# ----------------------------------------------------------------------
def detect(bgr_image, ir_image=None, with_yolo=None):
    """Run the exported pipeline on one BGR image.

    `ir_image` is optional — when omitted the IR pipeline runs on the
    grayscale of the RGB image (same as the analyzer's RGB-only mode).

    `with_yolo` - None = follow the exported config; True/False forces
    the YOLO stage on/off.

    Returns a dict of binary masks: 'rgb_mask', 'ir_mask', 'combined',
    plus 'up_<branch>' for every user branch. When YOLO runs it also
    adds 'yolo_mask' (uint8) and 'yolo_boxes' (list of dicts).
    """
    P     = CONFIG["params"]
    S     = CONFIG.get("strings", {})
    F     = CONFIG.get("flags", {})
    I     = CONFIG.get("ints", {})
    shape = KERNEL_SHAPES.get(S.get("kernel_shape", "Rect"), cv2.MORPH_RECT)
    mode  = RGB_DETECT_MODES.get(S.get("rgb_detect_mode", ""), "hue")

    views = {}

    # -- RGB pipeline --------------------------------------------------
    rgb_src   = run_pre_morph(bgr_image, CONFIG.get("rgb_pre_pipeline", []),
                              shape)
    rgb_pre   = rgb_detect(rgb_src, P, mode)
    views["rgb_mask_pre"] = rgb_pre
    rgb_mask, _ = run_steps(rgb_pre, CONFIG.get("rgb_pipeline", []),
                            shape, views, view_prefix="rgb")
    views["rgb_mask"] = rgb_mask
    views["rgb_det"]  = rgb_mask

    # -- IR pipeline ---------------------------------------------------
    if ir_image is None:
        ir_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    elif ir_image.ndim == 3:
        ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_image
    if F.get("use_clahe_ir"):
        cl = cv2.createCLAHE(
            clipLimit=float(max(1, I.get("clahe_ir_clip", 2))),
            tileGridSize=(max(1, I.get("clahe_ir_tile", 8)),) * 2)
        ir_gray = cl.apply(ir_gray)
    ir_src    = run_pre_morph(ir_gray, CONFIG.get("ir_pre_pipeline", []),
                              shape)
    _, ir_pre = cv2.threshold(ir_src, int(P["IR_thresh"]), 255,
                              cv2.THRESH_BINARY)
    views["ir_mask_pre"] = ir_pre
    ir_mask, _ = run_steps(ir_pre, CONFIG.get("ir_pipeline", []),
                           shape, views, view_prefix="ir")
    views["ir_mask"] = ir_mask
    views["ir_det"]  = ir_mask

    combined = cv2.bitwise_or(rgb_mask, _coerce_mask(ir_mask, rgb_mask))
    views["combined"] = combined

    out = {"rgb_mask": rgb_mask, "ir_mask": ir_mask, "combined": combined}

    # -- User branches -------------------------------------------------
    for bd in CONFIG.get("user_pipelines", []):
        nm = (bd.get("name") or "branch").strip()
        bt = bd.get("type", "rgb")
        if bt == "ir":
            src  = run_pre_morph(ir_gray, bd.get("pre_steps", []), shape)
            det  = cv2.inRange(src, bd.get("ir_lo", 80),
                                    bd.get("ir_hi", 255))
        elif bt == "custom":
            ref  = views.get(bd.get("source", ""))
            det  = (_coerce_mask(ref, rgb_mask) if ref is not None
                    else np.zeros(rgb_mask.shape[:2], np.uint8))
        else:  # rgb
            src  = run_pre_morph(bgr_image, bd.get("pre_steps", []), shape)
            det  = branch_rgb_detect(src, bd)
        views["up_%s_det" % nm] = det
        fin, _ = run_steps(det, bd.get("steps", []), shape, views,
                           view_prefix="up_%s" % nm)
        views["up_%s" % nm]      = fin
        views["up_%s_mask" % nm] = fin
        out["up_%s" % nm]        = fin

    # -- optional YOLO stage ------------------------------------------
    if with_yolo is None:
        with_yolo = CONFIG.get("export_yolo", {}).get("enabled", False)
    if with_yolo:
        try:
            boxes, ymask = run_yolo(bgr_image)
            out["yolo_boxes"] = boxes
            out["yolo_mask"]  = ymask
        except Exception as e:
            print("YOLO stage skipped (%s)" % e)

    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img = cv2.imread(src)
    if img is None:
        print("Could not read image:", src)
        sys.exit(1)
    ir = None
    if len(sys.argv) > 2:
        ir = cv2.imread(sys.argv[2])
    result = detect(img, ir)
    base = os.path.splitext(os.path.basename(src))[0]
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            out_name = "%s_%s.png" % (base, name)
            cv2.imwrite(out_name, value)
            print("wrote", out_name)
    # Bounding boxes of the combined mask (Min_area filter).
    boxes = find_objects(result["combined"],
                         CONFIG["params"].get("Min_area", 100))
    print("objects (combined, area >= Min_area):", len(boxes))
    for (x, y, w, h) in boxes:
        print("  box x=%d y=%d w=%d h=%d" % (x, y, w, h))
    # YOLO detections, when the exported config had YOLO enabled.
    if "yolo_boxes" in result:
        print("YOLO detections:", len(result["yolo_boxes"]))
        for d in result["yolo_boxes"]:
            print("  %s  conf=%.2f  x=%d y=%d w=%d h=%d"
                  % (d["name"], d["conf"], d["x"], d["y"],
                     d["w"], d["h"]))
'''

# The self-contained <name>.py = shared ops + the CONFIG-baked tail.
_EXPORT_RUNTIME = _OPS_SOURCE + _RUNTIME_TAIL


# ======================================================================
#  Library tail for the reusable <name>_lib.py.
#
#  Unlike the self-contained script, the library bakes in NOTHING: the
#  caller hands it a JSON config (the exported <name>_config.json or
#  any analyzer-saved config). This is what makes the pipeline reusable
#  from another program with just two lines:
#
#      from <name>_lib import load_pipeline
#      pipe = load_pipeline("<name>_config.json")
#      mask = pipe.final_mask(cv2.imread("photo.jpg"))
# ======================================================================
_LIB_TAIL = '''

# ----------------------------------------------------------------------
#  YOLO object detection (optional add-on stage). 'ultralytics' is
#  imported lazily, so a YOLO-off config never needs it installed.
# ----------------------------------------------------------------------
_YOLO_MODELS = {}


def _get_yolo_model(model_path):
    """Lazy-load (and cache) a YOLO model by weights path."""
    if model_path not in _YOLO_MODELS:
        from ultralytics import YOLO
        _YOLO_MODELS[model_path] = YOLO(model_path)
    return _YOLO_MODELS[model_path]


def run_yolo(bgr, model_path="yolo.pt", conf=0.5,
             box_scale=1.0, box_pad=0):
    """Run YOLO on a BGR image.

    Returns (boxes, mask):
      boxes - list of dicts {x, y, w, h, cls, name, conf}
      mask  - uint8 image, white inside every detected box
    `box_scale` (%/100) and `box_pad` (px) grow / shrink each box.
    """
    model = _get_yolo_model(model_path)
    boxes = []
    mask  = np.zeros(bgr.shape[:2], np.uint8)
    h, w  = bgr.shape[:2]
    for r in model(bgr, conf=conf, verbose=False):
        names = getattr(r, "names", {}) or {}
        for b in getattr(r, "boxes", []):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            x1, y1, x2, y2 = adjust_box(x1, y1, x2, y2, w, h,
                                        box_scale, box_pad)
            cid = int(b.cls[0])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                          "cls": cid, "name": names.get(cid, str(cid)),
                          "conf": float(b.conf[0])})
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return boxes, mask


# ----------------------------------------------------------------------
#  Core pipeline — runs the whole config and returns every named mask.
# ----------------------------------------------------------------------
def _run_pipeline(bgr_image, ir_image, CONFIG, with_yolo):
    """Run the JSON-config pipeline. Returns (out, views).

    `out`   - dict of result masks: rgb_mask / ir_mask / combined and
              up_<branch> for each user branch (+ yolo_* when YOLO runs).
    `views` - every intermediate array, keyed by name.
    """
    P     = CONFIG.get("params", {})
    S     = CONFIG.get("strings", {})
    F     = CONFIG.get("flags", {})
    I     = CONFIG.get("ints", {})
    shape = KERNEL_SHAPES.get(S.get("kernel_shape", "Rect"), cv2.MORPH_RECT)
    mode  = RGB_DETECT_MODES.get(S.get("rgb_detect_mode", ""), "hue")

    views = {}

    # -- RGB pipeline --------------------------------------------------
    rgb_src = run_pre_morph(bgr_image, CONFIG.get("rgb_pre_pipeline", []),
                            shape)
    rgb_pre = rgb_detect(rgb_src, P, mode)
    views["rgb_mask_pre"] = rgb_pre
    rgb_mask, _ = run_steps(rgb_pre, CONFIG.get("rgb_pipeline", []),
                            shape, views, view_prefix="rgb")
    views["rgb_mask"] = rgb_mask
    views["rgb_det"]  = rgb_mask

    # -- IR pipeline ---------------------------------------------------
    if ir_image is None:
        ir_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    elif ir_image.ndim == 3:
        ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_image
    if F.get("use_clahe_ir"):
        cl = cv2.createCLAHE(
            clipLimit=float(max(1, I.get("clahe_ir_clip", 2))),
            tileGridSize=(max(1, I.get("clahe_ir_tile", 8)),) * 2)
        ir_gray = cl.apply(ir_gray)
    ir_src    = run_pre_morph(ir_gray, CONFIG.get("ir_pre_pipeline", []),
                              shape)
    _, ir_pre = cv2.threshold(ir_src, int(P.get("IR_thresh", 128)), 255,
                              cv2.THRESH_BINARY)
    views["ir_mask_pre"] = ir_pre
    ir_mask, _ = run_steps(ir_pre, CONFIG.get("ir_pipeline", []),
                           shape, views, view_prefix="ir")
    views["ir_mask"] = ir_mask
    views["ir_det"]  = ir_mask

    combined = cv2.bitwise_or(rgb_mask, _coerce_mask(ir_mask, rgb_mask))
    views["combined"] = combined

    out = {"rgb_mask": rgb_mask, "ir_mask": ir_mask, "combined": combined}

    # -- User branches -------------------------------------------------
    for bd in CONFIG.get("user_pipelines", []):
        nm = (bd.get("name") or "branch").strip()
        bt = bd.get("type", "rgb")
        if bt == "ir":
            src = run_pre_morph(ir_gray, bd.get("pre_steps", []), shape)
            det = cv2.inRange(src, bd.get("ir_lo", 80), bd.get("ir_hi", 255))
        elif bt == "custom":
            ref = views.get(bd.get("source", ""))
            det = (_coerce_mask(ref, rgb_mask) if ref is not None
                   else np.zeros(rgb_mask.shape[:2], np.uint8))
        else:  # rgb
            src = run_pre_morph(bgr_image, bd.get("pre_steps", []), shape)
            det = branch_rgb_detect(src, bd)
        views["up_%s_det" % nm] = det
        fin, _ = run_steps(det, bd.get("steps", []), shape, views,
                           view_prefix="up_%s" % nm)
        views["up_%s" % nm]      = fin
        views["up_%s_mask" % nm] = fin
        out["up_%s" % nm]        = fin

    # -- optional YOLO stage ------------------------------------------
    if with_yolo is None:
        with_yolo = CONFIG.get("export_yolo", {}).get("enabled", False)
    if with_yolo:
        try:
            yb = CONFIG.get("export_yolo", {})
            boxes, ymask = run_yolo(bgr_image,
                                    yb.get("model_path", "yolo.pt"),
                                    yb.get("conf", 0.5),
                                    P.get("YOLO_Box_Scale", 100) / 100.0,
                                    int(P.get("YOLO_Box_Pad", 0)))
            out["yolo_boxes"] = boxes
            out["yolo_mask"]  = ymask
            views["yolo_mask"] = ymask
        except Exception as e:
            print("YOLO stage skipped (%s)" % e)

    return out, views


# ----------------------------------------------------------------------
#  Pipeline — the easy-to-use object. Build it from a JSON config, then
#  ask it for exactly the output your program wants.
# ----------------------------------------------------------------------
class Pipeline:
    """Reusable detector built from an analyzer JSON config.

    Output helpers (pick whichever your program needs):
      detect()       - dict of EVERY mask, exactly like the analyzer
      final_mask()   - ONLY the final combined binary mask  (uint8)
      mask_overlay() - the final mask painted over the image background
      mask_points()  - centre (x, y) point of every detected blob
      yolo_points()  - YOLO detections as centre points + names
    """

    def __init__(self, config):
        # `config` may be a path to a .json file or an already-loaded dict.
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config or {}

    @classmethod
    def from_json(cls, path):
        """Load a Pipeline from a JSON config file path."""
        return cls(path)

    # -- full output -------------------------------------------------
    def detect(self, bgr_image, ir_image=None, with_yolo=None):
        """Run the pipeline. Returns a dict of binary masks:
        rgb_mask / ir_mask / combined, plus up_<branch> for each branch
        (and yolo_mask + yolo_boxes when YOLO is on)."""
        out, _ = _run_pipeline(bgr_image, ir_image, self.config, with_yolo)
        return out

    def views(self, bgr_image, ir_image=None, with_yolo=None):
        """Like detect() but returns EVERY intermediate array too."""
        _, views = _run_pipeline(bgr_image, ir_image, self.config,
                                 with_yolo)
        return views

    # -- only the mask -----------------------------------------------
    def final_mask(self, bgr_image, ir_image=None, key="combined"):
        """ONLY the final mask (uint8, white = detected). `key` picks
        which one: 'combined' (default), 'rgb_mask', 'ir_mask' or
        'up_<branch>'."""
        return self.detect(bgr_image, ir_image).get(key)

    # -- mask + image background -------------------------------------
    def mask_overlay(self, bgr_image, ir_image=None, key="combined",
                     color=(0, 0, 255), alpha=0.5):
        """The final mask painted (semi-transparent) over the original
        image — handy to SEE what was detected. BGR colour, default red."""
        m = self.final_mask(bgr_image, ir_image, key)
        if m is None:
            return bgr_image.copy()
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        paint = np.zeros_like(bgr_image)
        paint[m > 0] = color
        return cv2.addWeighted(bgr_image, 1.0, paint, float(alpha), 0)

    # -- mask points -------------------------------------------------
    def mask_points(self, bgr_image, ir_image=None, key="combined",
                    min_area=None):
        """Centre point of every detected blob in the final mask.
        Returns [{x, y, w, h, area}, ...] — `x, y` is the blob centre."""
        m = self.final_mask(bgr_image, ir_image, key)
        if m is None:
            return []
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if min_area is None:
            min_area = self.config.get("params", {}).get("Min_area", 100)
        pts = []
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            pts.append({"x": x + w // 2, "y": y + h // 2,
                        "w": w, "h": h, "area": int(area)})
        return pts

    # -- YOLO points -------------------------------------------------
    def yolo_points(self, bgr_image, conf=None):
        """YOLO detections as centre points:
        [{x, y, name, conf, w, h}, ...].  Needs `pip install ultralytics`
        and the weights file named in the config."""
        yb = self.config.get("export_yolo", {})
        _P = self.config.get("params", {})
        boxes, _ = run_yolo(bgr_image, yb.get("model_path", "yolo.pt"),
                            yb.get("conf", 0.5) if conf is None else conf,
                            _P.get("YOLO_Box_Scale", 100) / 100.0,
                            int(_P.get("YOLO_Box_Pad", 0)))
        return [{"x": b["x"] + b["w"] // 2, "y": b["y"] + b["h"] // 2,
                 "w": b["w"], "h": b["h"],
                 "name": b["name"], "conf": b["conf"]} for b in boxes]


def load_pipeline(config="config.json"):
    """Load a ready-to-use Pipeline from a JSON config file (or dict).

        pipe = load_pipeline("my_config.json")
        mask = pipe.final_mask(cv2.imread("photo.jpg"))
    """
    return Pipeline(config)


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    img_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not os.path.exists(cfg_path):
        print("config not found:", cfg_path)
        print("usage: python THIS_LIB.py config.json [photo.jpg]")
        sys.exit(1)
    pipe = load_pipeline(cfg_path)
    if img_path and os.path.exists(img_path):
        img  = cv2.imread(img_path)
        mask = pipe.final_mask(img)
        cv2.imwrite("lib_final_mask.png", mask)
        cv2.imwrite("lib_overlay.png", pipe.mask_overlay(img))
        pts = pipe.mask_points(img)
        print("final mask  -> lib_final_mask.png")
        print("overlay     -> lib_overlay.png")
        print("mask points -> %d blob(s):" % len(pts))
        for p in pts:
            print("   x=%d y=%d area=%d" % (p["x"], p["y"], p["area"]))
    else:
        print("Pipeline loaded OK from", cfg_path)
        print("pass an image to test:  python THIS_LIB.py %s photo.jpg"
              % cfg_path)
'''

# The reusable <name>_lib.py = shared ops + the JSON-driven library tail.
_EXPORT_LIB = _OPS_SOURCE + _LIB_TAIL


# ======================================================================
#  Example file templates — each of the 7 examples is written as its
#  own standalone .py inside the exported "<name>_examples/" folder.
#  @PLACEHOLDERS@ are filled in by _export_build_example_files.
# ======================================================================
_EX_HEADER = '''"""@TITLE@

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python @FNAME@ @RUNARGS@
"""
import os
import sys

import cv2
import numpy as np

# Make the exported pipeline importable whether it sits next to this
# file or one folder up, then import it. The exported file is a
# self-contained library: every tuned setting lives in CONFIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from @MOD@ import detect, run_yolo, find_objects, CONFIG


def demo_image():
    """A throw-away test image (red box on white) so the example runs
    even before you have a real photo."""
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(img, (110, 80), (210, 160), (0, 0, 200), -1)
    return img


'''

_EX1_BODY = '''def main():
    """COMMAND-LINE ARGUMENTS.

    The exported pipeline file is runnable on its own:
        python @MOD@.py photo.jpg
        python @MOD@.py photo.jpg ir.png
    argv[1] = RGB image, argv[2] = optional IR image. Here we read
    those arguments ourselves and run the pipeline."""
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    ir_path  = sys.argv[2] if len(sys.argv) > 2 else None
    if img_path and os.path.exists(img_path):
        bgr = cv2.imread(img_path)
    else:
        print("no image given - using a generated test image")
        bgr = demo_image()
    ir = cv2.imread(ir_path) if ir_path and os.path.exists(ir_path) else None
    masks = detect(bgr, ir)
    cv2.imwrite("ex1_combined.png", masks["combined"])
    print("example 1: wrote ex1_combined.png")


if __name__ == "__main__":
    main()
'''

_EX2_BODY = '''def main():
    """Process a VIDEO file frame by frame and save a mask video."""
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("usage: python @FNAME@ clip.mp4")
        return
    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("ex2_mask.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h), isColor=False)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        masks = detect(frame)              # <-- run the pipeline
        out.write(masks["combined"])
        n += 1
    cap.release()
    out.release()
    print("example 2: processed %d frames -> ex2_mask.mp4" % n)


if __name__ == "__main__":
    main()
'''

_EX3_BODY = '''def main():
    """Feed in ONE IMAGE - the simplest use. Save every mask."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p and os.path.exists(p):
        bgr = cv2.imread(p)
    else:
        print("no image given - using a generated test image")
        bgr = demo_image()
    masks = detect(bgr)                    # dict {name: array}
    for name, value in masks.items():
        if not isinstance(value, np.ndarray):
            continue                       # skip non-image keys
        cv2.imwrite("ex3_%s.png" % name, value)
        print("example 3: wrote ex3_%s.png" % name)


if __name__ == "__main__":
    main()
'''

_EX4_BODY = '''def main():
    """Use ONLY ONE pipeline / one mask.

    detect() returns rgb_mask, ir_mask, combined and every branch. If
    you only need one, just take that key and ignore the rest."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    rgb_only = detect(bgr)["rgb_mask"]     # <-- the only thing we want
    cv2.imwrite("ex4_rgb_only.png", rgb_only)
    print("example 4: rgb_mask has %d detected pixels"
          % int((rgb_only > 0).sum()))


if __name__ == "__main__":
    main()
'''

_EX5_BODY = '''def main():
    """Combine a BRANCH with the main RGB mask using OR.
@BRANCHNOTE@
    Two ways to do it:
    A) INSIDE the analyzer GUI: add a step to the main RGB pipeline,
       set Combine = OR, source = up_@BRANCH@. Then the OR is baked in
       and detect()['rgb_mask'] already contains it.
    B) OUTSIDE, on the returned masks - shown here."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    masks  = detect(bgr)
    branch = masks.get("up_@BRANCH@")
    if branch is None:
        print("example 5: no branch 'up_@BRANCH@' in this export - "
              "add one in the analyzer and re-export.")
        return
    merged = cv2.bitwise_or(masks["rgb_mask"], branch)   # <-- the OR
    cv2.imwrite("ex5_merged.png", merged)
    print("example 5: wrote ex5_merged.png  (rgb_mask OR up_@BRANCH@)")


if __name__ == "__main__":
    main()
'''

_EX6_BODY = '''def main():
    """EVERYTHING: detect, find objects, draw boxes, save."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    masks = detect(bgr)
    # Min_area comes from the exported CONFIG (the value you tuned).
    min_area = CONFIG["params"].get("Min_area", 100)
    boxes = find_objects(masks["combined"], min_area)
    annotated = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("ex6_annotated.png", annotated)
    cv2.imwrite("ex6_combined.png", masks["combined"])
    print("example 6: found %d object(s) -> ex6_annotated.png"
          % len(boxes))


if __name__ == "__main__":
    main()
'''

_EX7_BODY = '''def main():
    """YOLO object detection.

    When the exported config had 'Use YOLO' ticked, detect() ALREADY
    runs YOLO - the result then has 'yolo_boxes' and 'yolo_mask'. You
    can also call run_yolo() directly, as shown here.
    Needs:  pip install ultralytics   + the model weights file."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    if not CONFIG.get("export_yolo", {}).get("enabled"):
        print("example 7: this export had YOLO OFF - tick 'Use YOLO' "
              "in the analyzer and re-export to use it.")
        return
    try:
        boxes, ymask = run_yolo(bgr)
    except Exception as e:
        print("example 7: YOLO needs 'ultralytics' + weights -", e)
        return
    annotated = bgr.copy()
    for d in boxes:
        cv2.rectangle(annotated, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 0), 2)
        cv2.putText(annotated, "%s %.2f" % (d["name"], d["conf"]),
                    (d["x"], max(12, d["y"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("ex7_yolo.png", annotated)
    print("example 7: YOLO found %d object(s) -> ex7_yolo.png"
          % len(boxes))


if __name__ == "__main__":
    main()
'''

# (filename, title, run-args hint, body source)
_EXAMPLE_PARTS = [
    ("example_1_commandline.py", "Example 1 - command-line arguments",
     "[photo.jpg] [ir.png]", _EX1_BODY),
    ("example_2_video.py",       "Example 2 - process a video file",
     "clip.mp4", _EX2_BODY),
    ("example_3_image.py",       "Example 3 - run on one image",
     "[photo.jpg]", _EX3_BODY),
    ("example_4_one_pipeline.py", "Example 4 - use only one mask",
     "[photo.jpg]", _EX4_BODY),
    ("example_5_branch_or.py",   "Example 5 - branch + OR with main RGB",
     "[photo.jpg]", _EX5_BODY),
    ("example_6_full.py",        "Example 6 - everything together",
     "[photo.jpg]", _EX6_BODY),
    ("example_7_yolo.py",        "Example 7 - YOLO object detection",
     "[photo.jpg]", _EX7_BODY),
]


# What the export dialog can write. (key, label, detail) — order shown.
_EXPORT_CHOICES = [
    ("py",       "Python script  (<name>.py)",
     "Self-contained: tuned settings baked in. Runs on its own."),
    ("lib",      "Reusable library  (<name>_lib.py)",
     "Import it in your own program; reads the config from JSON."),
    ("json",     "Config file  (<name>_config.json)",
     "The tuned settings only. The library loads this."),
    ("png",      "Flowchart image  (<name>.png)",
     "Diagram of the pipeline + a full settings report."),
    ("txt",      "Settings report  (<name>.txt)",
     "The same report text as in the PNG, as a plain text file."),
    ("readme",   "README  (<name>_README.md)",
     "How to use the library + JSON in your own code."),
    ("examples", "Worked examples folder  (<name>_examples/)",
     "7 ready-to-run example scripts."),
]


class ExportMixin:
    """Adds the 'Export pipeline' feature: pick exactly which of the
    standalone script / reusable library / config JSON / flowchart PNG /
    text report / README / examples folder to write."""

    # ------------------------------------------------------------------
    def _export_options_dialog(self):
        """Modal dialog of tick-boxes — one per exportable output.
        Returns a {key: bool} dict, or None if the user cancelled."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Export pipeline — choose what to write")
        dlg.configure(bg="#1e1e22")
        dlg.transient(self.root)
        dlg.resizable(False, False)

        tk.Label(dlg, text="Tick the files you want to export:",
                 bg="#1e1e22", fg="#e6e6e6",
                 font=("DejaVu Sans", 10, "bold")
                 ).pack(anchor="w", padx=14, pady=(12, 6))

        # Sensible default: everything except the examples folder.
        vars_ = {}
        for key, label, detail in _EXPORT_CHOICES:
            v = tk.BooleanVar(value=(key != "examples"))
            vars_[key] = v
            row = tk.Frame(dlg, bg="#1e1e22")
            row.pack(fill="x", padx=14, anchor="w")
            tk.Checkbutton(row, text=label, variable=v,
                           bg="#1e1e22", fg="#e6e6e6",
                           selectcolor="#333", activebackground="#1e1e22",
                           activeforeground="#ffffff",
                           font=("DejaVu Sans", 9, "bold")
                           ).pack(anchor="w")
            tk.Label(row, text="    " + detail, bg="#1e1e22",
                     fg="#9a9aa2", font=("DejaVu Sans", 8)
                     ).pack(anchor="w")

        # Quick presets so the common cases are one click.
        def _set(keys):
            for k, v in vars_.items():
                v.set(k in keys)

        pre = tk.Frame(dlg, bg="#1e1e22")
        pre.pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(pre, text="Presets:", bg="#1e1e22", fg="#9a9aa2",
                 font=("DejaVu Sans", 8)).pack(side="left")
        for txt, keys in [
            ("All", [k for k, _, _ in _EXPORT_CHOICES]),
            ("Library + JSON + README", ["lib", "json", "readme"]),
            ("PNG only", ["png"]),
            ("Script only", ["py"]),
        ]:
            tk.Button(pre, text=txt, font=("DejaVu Sans", 7),
                      bg="#2a2a30", fg="#dcdcdc",
                      command=lambda k=keys: _set(k)
                      ).pack(side="left", padx=2)

        result = {}

        def _ok():
            result.update({k: v.get() for k, v in vars_.items()})
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn = tk.Frame(dlg, bg="#1e1e22")
        btn.pack(fill="x", padx=14, pady=12)
        tk.Button(btn, text="Export", bg="#225522", fg="white",
                  font=("DejaVu Sans", 9, "bold"), command=_ok
                  ).pack(side="right", padx=4)
        tk.Button(btn, text="Cancel", bg="#553322", fg="white",
                  font=("DejaVu Sans", 9), command=_cancel
                  ).pack(side="right")

        dlg.update_idletasks()
        dlg.grab_set()
        self.root.wait_window(dlg)
        if not result or not any(result.values()):
            return None
        return result

    # ------------------------------------------------------------------
    def _export_pipeline(self):
        """Pick a base name, ask which outputs to write, then write the
        ticked ones: <name>.py / <name>_lib.py / <name>_config.json /
        <name>.png / <name>.txt / <name>_README.md / <name>_examples/."""
        from tkinter import filedialog
        sel = self._export_options_dialog()
        if sel is None:
            self.lbl_status.config(text="Export cancelled.", fg="#c8c8c8")
            return
        path = filedialog.asksaveasfilename(
            title="Export pipeline — base name",
            defaultextension=".py",
            filetypes=[("Python script", "*.py"), ("All files", "*")],
            initialfile="exported_pipeline.py")
        if not path:
            return
        if path.lower().endswith(".py"):
            path = path[:-3]
        base      = os.path.basename(path)
        py_path   = path + ".py"
        lib_file  = base + "_lib.py"
        lib_path  = path + "_lib.py"
        json_path = path + "_config.json"
        png_path  = path + ".png"
        txt_path  = path + ".txt"
        readme_path = path + "_README.md"
        ex_dir    = path + "_examples"
        mod_file  = os.path.basename(py_path)
        try:
            cfg = self._collect_config()
            cfg["export_yolo"] = self._export_yolo_block(cfg)
            written = []
            if sel.get("py"):
                with open(py_path, "w") as f:
                    f.write(self._export_build_py(cfg, mod_file))
                written.append(os.path.basename(py_path))
            if sel.get("lib"):
                with open(lib_path, "w") as f:
                    f.write(self._export_build_lib(lib_file))
                written.append(os.path.basename(lib_path))
            if sel.get("json"):
                with open(json_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                written.append(os.path.basename(json_path))
            if sel.get("png"):
                self._export_build_png(cfg, png_path)
                written.append(os.path.basename(png_path))
            if sel.get("txt"):
                with open(txt_path, "w") as f:
                    f.write("\n".join(self._settings_report(cfg)) + "\n")
                written.append(os.path.basename(txt_path))
            if sel.get("readme"):
                with open(readme_path, "w") as f:
                    f.write(self._export_build_readme(cfg, base, sel))
                written.append(os.path.basename(readme_path))
            if sel.get("examples"):
                n = self._export_write_example_files(cfg, mod_file, ex_dir)
                written.append("%s/ (%d examples)"
                               % (os.path.basename(ex_dir), n))
            if not written:
                self.lbl_status.config(text="Nothing selected to export.",
                                       fg="#c8c8c8")
                return
            self.lbl_status.config(
                text="Exported: " + ", ".join(written), fg="#88ff88")
        except Exception as e:
            self.lbl_status.config(
                text="Export failed: %s" % e, fg="red")

    # ------------------------------------------------------------------
    def _export_build_lib(self, lib_file="exported_pipeline_lib.py"):
        """Compose the reusable library source. It bakes in NOTHING —
        the caller passes a JSON config at run time."""
        mod = (lib_file[:-3] if lib_file.lower().endswith(".py")
               else lib_file)
        header = (
            '"""Reusable RealSense Cable detection pipeline (library).\n'
            "\n"
            "Auto-generated by the RealSense Cable Video Analyzer.\n"
            "Self-contained: needs only opencv-python + numpy.\n"
            "\n"
            "This file bakes in NO settings. Hand it a JSON config\n"
            "(exported alongside as <name>_config.json, or any config\n"
            "the analyzer saved) and it reproduces that exact pipeline:\n"
            "\n"
            "    import cv2\n"
            "    from %s import load_pipeline\n" % mod +
            "    pipe = load_pipeline('exported_pipeline_config.json')\n"
            "    mask = pipe.final_mask(cv2.imread('photo.jpg'))\n"
            "\n"
            "See the exported <name>_README.md for the full guide.\n"
            '"""\n'
            "import os\n"
            "import json\n"
            "import cv2\n"
            "import numpy as np\n")
        return header + _EXPORT_LIB.replace("THIS_LIB.py", lib_file)

    # ------------------------------------------------------------------
    def _export_build_readme(self, cfg, base, sel):
        """Compose the README.md that guides a user through importing
        the library + JSON config into their own program."""
        lib_mod  = base + "_lib"
        json_nm  = base + "_config.json"
        branches = [b.get("name", "branch")
                    for b in cfg.get("user_pipelines", [])]
        yolo_on  = cfg.get("export_yolo", {}).get("enabled")

        L = []
        L.append("# %s — exported detection pipeline\n" % base)
        L.append("Auto-generated by the RealSense Cable Video Analyzer. "
                 "It reproduces the\nexact pipeline you tuned in the GUI "
                 "so you can reuse it in any program.\n")

        L.append("## Files in this export\n")
        files = [
            ("py",       "`%s.py`" % base,
             "stand-alone script — settings baked in, runs on its own"),
            ("lib",      "`%s.py`" % lib_mod,
             "**the reusable library** — import this in your code"),
            ("json",     "`%s`" % json_nm,
             "the tuned settings — the library loads this"),
            ("png",      "`%s.png`" % base,
             "flowchart diagram + settings report"),
            ("txt",      "`%s.txt`" % base,
             "the settings report as plain text"),
            ("examples", "`%s_examples/`" % base,
             "7 ready-to-run example scripts"),
        ]
        for key, name, desc in files:
            if sel.get(key):
                L.append("- %s — %s" % (name, desc))
        L.append("")

        L.append("## Use it in your own program\n")
        L.append("Copy `%s.py` and `%s` next to your code, then:\n"
                 % (lib_mod, json_nm))
        L.append("```python")
        L.append("import cv2")
        L.append("from %s import load_pipeline" % lib_mod)
        L.append("")
        L.append("# 1. load the library with the exported JSON config")
        L.append("pipe = load_pipeline(\"%s\")" % json_nm)
        L.append("")
        L.append("# 2. give it an image")
        L.append("img = cv2.imread(\"photo.jpg\")")
        L.append("")
        L.append("# 3. get the final mask — exactly what the analyzer shows")
        L.append("mask = pipe.final_mask(img)")
        L.append("cv2.imwrite(\"mask.png\", mask)")
        L.append("```\n")

        L.append("## Choosing what you get back\n")
        L.append("`pipe` gives you the output your program needs — "
                 "pick one:\n")
        L.append("| You want | Call | Returns |")
        L.append("|---|---|---|")
        L.append("| every mask (like the analyzer) | "
                 "`pipe.detect(img)` | dict of masks |")
        L.append("| **only the mask** | "
                 "`pipe.final_mask(img)` | uint8 binary mask |")
        L.append("| **mask + image background** | "
                 "`pipe.mask_overlay(img)` | BGR image, mask painted on |")
        L.append("| **mask points** (blob centres) | "
                 "`pipe.mask_points(img)` | `[{x, y, w, h, area}, ...]` |")
        L.append("| **YOLO points** | "
                 "`pipe.yolo_points(img)` | `[{x, y, name, conf}, ...]` |")
        L.append("")
        L.append("All of them also take an optional IR image: "
                 "`pipe.final_mask(img, ir_img)`.\n")

        L.append("### `final_mask` — pick which mask\n")
        L.append("`final_mask(img, key=...)` selects the mask: "
                 "`\"combined\"` (default),\n"
                 "`\"rgb_mask\"`, `\"ir_mask\"`, or a branch "
                 "`\"up_<branch>\"`.\n")

        if branches:
            L.append("## Branches in this config\n")
            L.append("This config has %d user branch(es). Each one is "
                     "also a mask:\n" % len(branches))
            for nm in branches:
                L.append("- `%s` -> `pipe.detect(img)[\"up_%s\"]`"
                         % (nm, nm))
            L.append("")
        else:
            L.append("## Branches\n")
            L.append("This config has no user branches — only "
                     "`rgb_mask`, `ir_mask` and `combined`.\n")

        L.append("## YOLO\n")
        if yolo_on:
            L.append("YOLO is **ON** in this config. `pipe.detect(img)` "
                     "also returns\n`yolo_mask` + `yolo_boxes`, and "
                     "`pipe.yolo_points(img)` gives centre points.\n")
            L.append("Needs `pip install ultralytics` and the weights "
                     "file:\n`%s`\n"
                     % cfg.get("export_yolo", {}).get("model_path", "?"))
        else:
            L.append("YOLO is **off** in this config. Tick *Use YOLO* in "
                     "the analyzer\nand re-export to include it.\n")

        L.append("## Notes\n")
        L.append("- Needs only `opencv-python` + `numpy` "
                 "(plus `ultralytics` if YOLO is used).")
        L.append("- MOG2 background subtraction and the OVERLAY combine "
                 "are not\n  reproduced (they need video history / are "
                 "visualisation-only).")
        L.append("- Re-tune in the analyzer and re-export to update the "
                 "JSON — your\n  code does not change, only `%s`.\n"
                 % json_nm)
        return "\n".join(L)

    # ------------------------------------------------------------------
    def _export_build_example_files(self, cfg, mod_file):
        """Build the 7 standalone example scripts. Returns a dict
        {filename: source}."""
        mod = (mod_file[:-3] if mod_file.lower().endswith(".py")
               else mod_file)
        branches = [b.get("name", "branch")
                    for b in cfg.get("user_pipelines", [])]
        branch = branches[0] if branches else "yourbranch"
        bnote  = ("" if branches else
                  "    NOTE: this export has no branches yet, so "
                  "'up_yourbranch'\n"
                  "    below is only a placeholder name.\n")
        files = {}
        for fname, title, runargs, body in _EXAMPLE_PARTS:
            src = (_EX_HEADER + body)
            src = (src.replace("@TITLE@", title)
                      .replace("@FNAME@", fname)
                      .replace("@RUNARGS@", runargs)
                      .replace("@BRANCHNOTE@", bnote)
                      .replace("@BRANCH@", branch)
                      .replace("@MOD@", mod))
            files[fname] = src
        files["README.txt"] = (
            "Examples for the exported RealSense Cable pipeline.\n"
            "\n"
            "Each file is standalone - run it directly:\n"
            "    python example_3_image.py  photo.jpg\n"
            "\n"
            "They import the pipeline from '%s.py', which they look for\n"
            "next to themselves or one folder up. Keep that file with\n"
            "this folder (or in its parent).\n" % mod)
        return files

    def _export_write_example_files(self, cfg, mod_file, ex_dir):
        """Write the example folder. Also drops a copy of the pipeline
        .py inside it so the folder is self-contained. Returns the
        number of example scripts written."""
        os.makedirs(ex_dir, exist_ok=True)
        files = self._export_build_example_files(cfg, mod_file)
        for fname, src in files.items():
            with open(os.path.join(ex_dir, fname), "w") as f:
                f.write(src)
        # self-contained copy of the pipeline next to the examples
        with open(os.path.join(ex_dir, mod_file), "w") as f:
            f.write(self._export_build_py(cfg, mod_file))
        return sum(1 for k in files if k.startswith("example_"))

    # ------------------------------------------------------------------
    def _export_yolo_block(self, cfg):
        """Resolve the YOLO settings to embed in the exported config.
        YOLO counts as enabled when the 'Use YOLO' flag is on OR any
        pipeline step has a per-step YOLO add-on enabled."""
        enabled = bool(cfg.get("flags", {}).get("use_yolo"))

        def _scan(steps):
            for s in steps or []:
                if (s.get("yolo") or {}).get("yolo_en"):
                    return True
            return False

        if not enabled:
            for key in ("rgb_pipeline", "ir_pipeline",
                        "rgb_pre_pipeline", "ir_pre_pipeline"):
                if _scan(cfg.get(key)):
                    enabled = True
                    break
        if not enabled:
            for bd in cfg.get("user_pipelines", []):
                if _scan(bd.get("steps")) or _scan(bd.get("pre_steps")):
                    enabled = True
                    break
        try:
            path = self._load_pref("yolo_model_path", YOLO_MODEL_PATH)
        except Exception:
            path = YOLO_MODEL_PATH
        conf = cfg.get("params", {}).get("YOLO_Conf", 50)
        return {"enabled": enabled, "model_path": path,
                "conf": float(conf) / 100.0}

    # ------------------------------------------------------------------
    def _export_build_py(self, cfg, mod_file="exported_pipeline.py"):
        """Compose the self-contained Python script source."""
        mod = mod_file[:-3] if mod_file.lower().endswith(".py") else mod_file
        header = (
            '"""Exported RealSense Cable detection pipeline.\n'
            "\n"
            "Auto-generated by the RealSense Cable Video Analyzer.\n"
            "Self-contained: needs only opencv-python + numpy.\n"
            "\n"
            "Usage as a library:\n"
            "    from %s import detect\n" % mod +
            "    import cv2\n"
            "    masks = detect(cv2.imread('photo.jpg'))\n"
            "    cv2.imwrite('mask.png', masks['combined'])\n"
            "\n"
            "Usage from the command line:\n"
            "    python %s photo.jpg [ir.png]\n" % mod_file +
            "\n"
            "See the %s_examples/ folder for 7 worked examples.\n" % mod +
            "\n"
            "YOLO: when the analyzer had 'Use YOLO' ticked, detect()\n"
            "runs YOLO too (adds 'yolo_mask' + 'yolo_boxes'). It needs\n"
            "`pip install ultralytics` and the weights file; ultralytics\n"
            "is imported lazily so a YOLO-off config never needs it.\n"
            "\n"
            "NOT reproduced: MOG2 background subtraction (needs video\n"
            "history) and the OVERLAY combine (visualisation-only).\n"
            '"""\n'
            "import os\n"
            "import json\n"
            "import cv2\n"
            "import numpy as np\n"
            "\n")
        cfg_json = json.dumps(cfg, indent=1)
        cfg_block = ("# ---- Tuned configuration (snapshot of the GUI) ----\n"
                     "CONFIG = json.loads(r'''\n" + cfg_json + "\n''')\n")
        return header + cfg_block + _EXPORT_RUNTIME

    # ------------------------------------------------------------------
    def _settings_report(self, cfg):
        """Return a list of report lines describing EXACTLY what the
        pipeline does — every step's op + kernel sizes / iterations /
        thresholds, and the detection ranges."""
        P = cfg.get("params", {})
        S = cfg.get("strings", {})
        F = cfg.get("flags", {})
        I = cfg.get("ints", {})
        mode_key = RGB_DETECT_MODES.get(S.get("rgb_detect_mode", ""), "hue")
        L = ["SETTINGS DETAIL  -  what each stage does"]

        L.append("")
        L.append("RGB detection mode: %s  (%s)"
                 % (S.get("rgb_detect_mode", "?"), mode_key))
        L.append("  Hue   H1 %s..%s   H2 %s..%s"
                 % (P.get("H1_low"), P.get("H1_high"),
                    P.get("H2_low"), P.get("H2_high")))
        L.append("  Sat   S %s..%s      Val V %s..%s"
                 % (P.get("S_min"), P.get("S_max"),
                    P.get("V_min"), P.get("V_max")))
        L.append("  LAB   L %s..%s  a* %s..%s  b* %s..%s   (lab / hsv_lab modes)"
                 % (P.get("L_min", 0), P.get("L_max", 255),
                    P.get("A_min"), P.get("A_max"),
                    P.get("B_min"), P.get("B_max")))

        def _dump(title, pre, main):
            L.append("")
            L.append(title)
            pon = [s for s in pre if s.get("en")]
            L.append("  pre-morph (conditions the image before detect):")
            if pon:
                for i, s in enumerate(pon, 1):
                    L.append(_step_line(i, s))
            else:
                L.append("     (none)")
            mon = [s for s in main if s.get("en")]
            L.append("  pipeline steps (run on the binary mask):")
            if mon:
                for i, s in enumerate(mon, 1):
                    L.append(_step_line(i, s))
            else:
                L.append("     (none)")

        _dump("RGB pipeline", cfg.get("rgb_pre_pipeline", []),
              cfg.get("rgb_pipeline", []))

        L.append("")
        L.append("IR detection: threshold t=%s  (pixels brighter -> white)"
                 % P.get("IR_thresh"))
        if F.get("use_clahe_ir"):
            L.append("  CLAHE on: clip=%s  tile=%s"
                     % (I.get("clahe_ir_clip"), I.get("clahe_ir_tile")))
        _dump("IR pipeline", cfg.get("ir_pre_pipeline", []),
              cfg.get("ir_pipeline", []))

        for bd in cfg.get("user_pipelines", []):
            nm = bd.get("name", "branch")
            bt = bd.get("type", "rgb")
            L.append("")
            head = "Branch '%s'  type=%s" % (nm, bt)
            if bt == "rgb":
                head += "  channels=%s" % bd.get("channels", "HSV")
            elif bt == "ir":
                head += "  threshold %s..%s" % (bd.get("ir_lo"),
                                                bd.get("ir_hi"))
            else:
                head += "  source=%s" % bd.get("source", "?")
            _dump(head, bd.get("pre_steps", []), bd.get("steps", []))

        L.append("")
        yb = cfg.get("export_yolo", {})
        if yb.get("enabled"):
            L.append("YOLO: ON   model=%s   conf=%.2f"
                     % (os.path.basename(yb.get("model_path", "?")),
                        yb.get("conf", 0.5)))
            L.append("  detect() also returns yolo_mask + yolo_boxes.")
        else:
            L.append("YOLO: off  (tick 'Use YOLO' before export to "
                     "include it)")
        L.append("")
        L.append("Object filter: Min_area = %s px" % P.get("Min_area"))
        return L

    def _export_build_png(self, cfg, png_path):
        """Draw a flowchart diagram + a settings report to `png_path`."""
        from PIL import Image, ImageDraw

        S = cfg.get("strings", {})
        P = cfg.get("params", {})
        I = cfg.get("ints", {})
        mode_key = RGB_DETECT_MODES.get(S.get("rgb_detect_mode", ""), "hue")

        def _trunc(s, n):
            s = str(s)
            return s if len(s) <= n else s[:n - 1] + "."

        def _step_boxes(steps):
            """[(line1, line2), ...] — one box per ENABLED step, with
            its op name and parameters so the diagram shows the sizes."""
            out = []
            for s in steps:
                if not s.get("en"):
                    continue
                if s.get("comb_en"):
                    out.append((s.get("comb_op", "AND"),
                                _trunc(s.get("comb_src", ""), 20)))
                else:
                    out.append((s.get("op", "?"),
                                _trunc(_fmt_step_params(s), 20)))
            return out

        # Build the list of pipeline "rows": (title, [(l1,l2)], colour).
        rows = []
        rb = [("RGB raw", "")]
        rb += _step_boxes(cfg.get("rgb_pre_pipeline", []))
        rb.append(("detect", mode_key))
        rb += _step_boxes(cfg.get("rgb_pipeline", []))
        rb.append(("rgb_mask", ""))
        rows.append(("RGB pipeline", rb, (90, 150, 230)))

        ib = [("IR raw", "")]
        if cfg.get("flags", {}).get("use_clahe_ir"):
            ib.append(("CLAHE", "clip=%s tile=%s"
                       % (I.get("clahe_ir_clip"), I.get("clahe_ir_tile"))))
        ib += _step_boxes(cfg.get("ir_pre_pipeline", []))
        ib.append(("threshold", "t=%s" % P.get("IR_thresh")))
        ib += _step_boxes(cfg.get("ir_pipeline", []))
        ib.append(("ir_mask", ""))
        rows.append(("IR pipeline", ib, (210, 120, 90)))

        rows.append(("Fuse", [("rgb_mask", ""), ("OR", ""),
                              ("ir_mask", ""), ("combined", "")],
                     (140, 140, 140)))

        for bd in cfg.get("user_pipelines", []):
            nm = bd.get("name", "branch")
            bt = bd.get("type", "rgb")
            bx = [("%s source" % bt, "")]
            bx += _step_boxes(bd.get("pre_steps", []))
            if bt == "custom":
                bx.append(("source", _trunc(bd.get("source", ""), 15)))
            elif bt == "ir":
                bx.append(("detect", "ir %s..%s"
                           % (bd.get("ir_lo"), bd.get("ir_hi"))))
            else:
                bx.append(("detect", bd.get("channels", "HSV")))
            bx += _step_boxes(bd.get("steps", []))
            bx.append(("up_%s" % nm, ""))
            rows.append(("branch: %s" % nm, bx, (170, 110, 200)))

        # -- Layout ----------------------------------------------------
        # A pipeline row with many step boxes WRAPS: at most
        # MAX_PER_LINE boxes per visual line, the rest flow onto a new
        # line below — so a long pipeline never makes the PNG absurdly
        # wide.
        MAX_PER_LINE = 7
        bw, bh   = 142, 50
        gap_x    = 28
        gap_y    = 28
        title_w  = 140
        margin   = 24

        # Split each logical row into visual lines of <= MAX_PER_LINE
        # boxes. Tuple: (title, boxes_chunk, colour, is_cont, cont_after)
        vis = []
        for title, boxes, colour in rows:
            if not boxes:
                vis.append((title, [], colour, False, False))
                continue
            chunks = [boxes[i:i + MAX_PER_LINE]
                      for i in range(0, len(boxes), MAX_PER_LINE)]
            for ci, ch in enumerate(chunks):
                vis.append((title if ci == 0 else "", ch, colour,
                            ci > 0, ci < len(chunks) - 1))

        max_cols = min(MAX_PER_LINE,
                       max((len(b) for _, b, _ in rows), default=1))
        report   = self._settings_report(cfg)
        rep_h    = len(report) * 15 + 30
        width    = max(margin * 2 + title_w + max_cols * bw
                       + (max_cols - 1) * gap_x,
                       margin * 2 + 560)
        flow_h   = len(vis) * bh + (len(vis) - 1) * gap_y
        height   = margin * 2 + 30 + flow_h + rep_h

        img  = Image.new("RGB", (width, height), (28, 28, 32))
        draw = ImageDraw.Draw(img)
        draw.text((margin, 6), "Exported detection pipeline  -  "
                  "RealSense Cable Video Analyzer", fill=(225, 225, 225))

        y = margin + 24
        for title, boxes, colour, is_cont, cont_after in vis:
            if title:
                draw.text((margin, y + bh // 2 - 6), title,
                          fill=(230, 230, 230))
            elif is_cont:
                draw.text((margin + 12, y + bh // 2 - 6), "...cont",
                          fill=(140, 140, 150))
            x = margin + title_w
            for i, (l1, l2) in enumerate(boxes):
                draw.rectangle([x, y, x + bw, y + bh],
                               fill=colour, outline=(235, 235, 235))
                if l2:
                    draw.text((x + 6, y + 9), _trunc(l1, 21),
                              fill=(15, 15, 15))
                    draw.text((x + 6, y + 27), _trunc(l2, 21),
                              fill=(45, 45, 55))
                else:
                    draw.text((x + 6, y + bh // 2 - 6), _trunc(l1, 21),
                              fill=(15, 15, 15))
                last = (i == len(boxes) - 1)
                if not last:
                    ax, ay = x + bw, y + bh // 2
                    draw.line([ax, ay, ax + gap_x, ay],
                              fill=(235, 235, 235), width=2)
                    draw.polygon([(ax + gap_x, ay),
                                  (ax + gap_x - 7, ay - 4),
                                  (ax + gap_x - 7, ay + 4)],
                                 fill=(235, 235, 235))
                elif cont_after:
                    # Row wraps here — a down-arrow shows the pipeline
                    # continues on the next visual line.
                    cx, cy = x + bw // 2, y + bh
                    draw.line([cx, cy, cx, cy + gap_y - 6],
                              fill=(235, 235, 235), width=2)
                    draw.polygon([(cx, cy + gap_y - 6),
                                  (cx - 4, cy + gap_y - 13),
                                  (cx + 4, cy + gap_y - 13)],
                                 fill=(235, 235, 235))
                x += bw + gap_x
            y += bh + gap_y

        # -- Settings report (text block under the flowchart) ----------
        y += 6
        draw.line([margin, y, width - margin, y], fill=(80, 80, 90), width=1)
        y += 10
        for line in report:
            bold = (not line.startswith(" ")) and line.strip() != ""
            draw.text((margin, y), line,
                      fill=(255, 235, 170) if bold else (200, 200, 205))
            y += 15

        img.save(png_path)
