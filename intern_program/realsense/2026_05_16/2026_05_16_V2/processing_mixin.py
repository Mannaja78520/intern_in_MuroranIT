"""Image-processing pipeline for the RealSense Cable Video Analyzer.

Extracted from realsense_video_analyzer.py so the main file stays
navigable. Holds the single large _process() method (HSV / IR
detection, PM + Morph pipelines, Combine, YOLO, branch pipelines and
view emission) as a mixin, plus the torch / YOLO availability probe.
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from PIL import Image, ImageTk

from config import (
    RECORDINGS_DIR, YOLO_MODEL_PATH, SCREENSHOTS_DIR,
    DISPLAY_W, DISPLAY_H,
    IR_COLORMAPS, MORPH_OPS, EXTRA_OPS, ALL_PROC_OPS, KERNEL_SHAPES,
    VIEW_OPTIONS,
    FC_W, FC_H,
    is_pre_morph_op,
)

# GPU detection for YOLO. ultralytics auto-picks CUDA when available,
# but we surface a flag here so we can explicitly pass device='0' /
# 'cpu' to inference calls.
_TORCH_DEVICE = "cpu"
try:
    import torch  # noqa: F401 - only used for cuda probe
    if torch.cuda.is_available():
        _TORCH_DEVICE = "0"  # first CUDA device; ultralytics accepts str
except Exception:
    pass

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except Exception:
    # Always bind _YOLO so other modules can import it unconditionally;
    # callers gate real use behind _YOLO_AVAILABLE.
    _YOLO = None
    _YOLO_AVAILABLE = False


class ProcessingMixin:
    # Auto-detected device ("0" = CUDA, "cpu"). Exposed so the UI can
    # tell whether a GPU is available.
    GPU_AVAILABLE = (_TORCH_DEVICE != "cpu")

    def _yolo_device(self):
        """Resolve the YOLO inference device from the user's
        Performance setting. 'Auto' uses whatever was detected;
        'GPU' falls back to CPU when no CUDA device exists."""
        sel = "Auto"
        try:
            sel = self.set_yolo_device.get()
        except Exception:
            pass
        if sel == "CPU":
            return "cpu"
        if sel == "GPU":
            return "0" if _TORCH_DEVICE != "cpu" else "cpu"
        return _TORCH_DEVICE          # "Auto"

    def _process(self, f_rgb, f_ir):
        # Performance: optionally downscale the input frames before
        # the whole pipeline runs. Everything downstream (detection,
        # PM/morph stages, YOLO, views) then works on smaller arrays;
        # _put upscales for display. Lower quality, much faster.
        _sc = getattr(self, "_proc_scale", 1.0)
        if _sc and _sc < 0.999:
            try:
                f_rgb = cv2.resize(f_rgb, None, fx=_sc, fy=_sc,
                                   interpolation=cv2.INTER_AREA)
                if f_ir is not None:
                    f_ir = cv2.resize(f_ir, None, fx=_sc, fy=_sc,
                                      interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        h1l  = self.sv["H1_low"].get();   h1h  = self.sv["H1_high"].get()
        h2l  = self.sv["H2_low"].get();   h2h  = self.sv["H2_high"].get()
        smin = self.sv["S_min"].get();    smax = self.sv["S_max"].get()
        vmin = self.sv["V_min"].get();    vmax = self.sv["V_max"].get()
        irt       = self.sv["IR_thresh"].get()
        mna       = self.sv["Min_area"].get()
        xy_kshape = KERNEL_SHAPES.get(self.kernel_shape_var.get(), cv2.MORPH_RECT)
        conf_thr  = self.sv["YOLO_Conf"].get() / 100.0

        # view_lookup is referenced by the pre-morph stage below (so
        # pre-morph step results can be used as OVERLAY / combine
        # sources later in the morph stage). Initialise it early.
        view_lookup = {}

        # Whether the All-Masks flowchart window is open. Several view
        # products (HSV colormap / paired tiles) are consumed ONLY by
        # that window and are not panel-selectable, so their (costly)
        # generation is skipped while it is closed.
        _all_masks_open = bool(getattr(self, "all_masks_win", None)
                               and self.all_masks_win.winfo_exists())

        # Pre-allocate step-snapshot lists. Pre-morph and morph
        # pipelines are now SEPARATE lists, so each has its own
        # snapshot list.
        snap_rgb_steps     = [None] * len(self.rgb_pipeline)
        snap_ir_steps      = [None] * len(self.ir_pipeline)
        snap_rgb_pre_steps = [None] * len(self.rgb_pre_pipeline)
        snap_ir_pre_steps  = [None] * len(self.ir_pre_pipeline)

        # RGB / IR pre-blur, IR CLAHE and "blur position" radios have
        # been removed — the user adds those ops as PM (Pre-morph)
        # steps now. work_rgb / ir_gray start raw and the PM stage
        # below transforms them in user-defined order.
        work_rgb = f_rgb

        # RGB HSV detection — placeholder values; real detection runs
        # after the pre-morph stage modifies `work_rgb`.
        hsv = cv2.cvtColor(work_rgb, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        m2  = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        red_mask = cv2.bitwise_or(m1, m2)
        snap_rgb_hsv_mask = red_mask.copy()

        # BG-sub feed picker (shared helper). Each side picks its own
        # source via bgsub_{rgb,ir}_src. When the source changes we
        # rebuild MOG2 so the trained model isn't stale against a
        # different-shape input.
        def _bgsub_feed(src_name):
            if src_name == "rgb":
                return f_rgb
            if src_name == "ir":
                return (f_ir if f_ir.ndim == 3
                        else cv2.cvtColor(f_ir, cv2.COLOR_GRAY2BGR))
            if src_name == "H":
                return hsv[:, :, 0]
            if src_name == "S":
                return hsv[:, :, 1]
            if src_name == "V":
                return hsv[:, :, 2]
            return f_rgb

        snap_fg_rgb = None
        if self.use_bgsub.get():
            _rgb_src = self.bgsub_rgb_src.get()
            # Rebuild MOG2 when the chosen source feed changes shape.
            if (self.backSub_rgb is None
                    or self._backSub_rgb_src_sig != _rgb_src):
                self.backSub_rgb = cv2.createBackgroundSubtractorMOG2(
                    history=self.sv["BG_hist"].get(),
                    varThreshold=self.sv["BG_var"].get(),
                    detectShadows=False)
                self._backSub_rgb_src_sig = _rgb_src
            try:
                snap_fg_rgb = self.backSub_rgb.apply(_bgsub_feed(_rgb_src))
                red_mask    = cv2.bitwise_and(red_mask, snap_fg_rgb)
            except Exception:
                snap_fg_rgb = None
        rgb_mask = red_mask.copy()
        snap_rgb_mask_pre = rgb_mask.copy()

        # IR pre-process — gray + optional CLAHE + optional BG-sub,
        # then threshold. The dedicated CLAHE knob is independent
        # from the PM-step CLAHE (both stack when enabled).
        snap_ir_gray = (cv2.cvtColor(f_ir, cv2.COLOR_BGR2GRAY)
                        if f_ir.ndim == 3 else f_ir.copy())
        ir_gray = snap_ir_gray.copy()
        if self.use_clahe_ir.get():
            try:
                _cl = cv2.createCLAHE(
                    clipLimit=float(max(1, self.clahe_ir_clip.get())),
                    tileGridSize=(max(1, self.clahe_ir_tile.get()),) * 2)
                ir_gray = _cl.apply(ir_gray)
            except Exception:
                pass
        # ir_clahe view now reflects the post-CLAHE gray (or raw gray
        # when CLAHE is disabled). ir_blur stays as the legacy alias.
        snap_ir_blur  = snap_ir_gray.copy()
        snap_ir_clahe = ir_gray.copy()

        if self.use_bgsub.get():
            _ir_src = self.bgsub_ir_src.get()
            if (self.backSub_ir is None
                    or self._backSub_ir_src_sig != _ir_src):
                self.backSub_ir = cv2.createBackgroundSubtractorMOG2(
                    history=self.sv["BG_hist"].get(),
                    varThreshold=self.sv["BG_var"].get(),
                    detectShadows=False)
                self._backSub_ir_src_sig = _ir_src
            try:
                _feed = _bgsub_feed(_ir_src)
                # When the IR side is fed an HSV channel (single-chan)
                # the resulting foreground mask is still 1-channel and
                # can be used as `ir_fg`; otherwise we need to combine
                # the BG-mask with the gray frame so threshold can run.
                _fg = self.backSub_ir.apply(_feed)
                ir_fg = cv2.bitwise_and(ir_gray, _fg)
            except Exception:
                ir_fg = ir_gray
        else:
            ir_fg = ir_gray
        snap_ir_fg = ir_fg.copy()
        _, ir_bin = cv2.threshold(ir_fg, irt, 255, cv2.THRESH_BINARY)
        ir_mask          = ir_bin.copy()
        snap_ir_thresh   = ir_bin.copy()
        snap_ir_mask_pre = ir_mask.copy()

        # Processing pipelines (RGB and IR independent)
        def _odd(v): return v if v % 2 == 1 else v + 1

        def _fill_holes(m):
            flood  = m.copy()
            h, w   = m.shape[:2]
            ffmask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ffmask, (0, 0), 255)
            return cv2.bitwise_or(m, cv2.bitwise_not(flood))

        def _apply_op_image(img, op_name, n, kx, ky, d, thresh,
                            prev=None, inp=None, kshape=None):
            """Apply a PRE-MORPH op (Blur / Illumination / Edge) to a
            BGR or grayscale IMAGE — keeps colour information instead
            of treating the input as a binary mask. `kshape` forwards
            the per-step Kernel-shape override down into _apply_op so
            morph ops respect the step's selection. Routes per-op:
              * Blur ops (Gauss/Median/Bilateral): cv2 handles multi-
                channel natively, so just reuse _apply_op.
              * Illumination ops (HistEq/CLAHE/Gamma/Normalize/Retinex):
                for BGR, convert to LAB, run on L channel, convert back
                so brightness/contrast change without colour shift.
                For grayscale just call _apply_op directly.
              * Edge ops (Sharpen/Laplacian/Sobel/Canny): Sharpen is
                applied per-channel to preserve colour; the others go
                BGR -> gray -> op -> gray-broadcast-to-BGR."""
            is_bgr = (img.ndim == 3 and img.shape[2] == 3)

            if op_name in ("GaussBlur", "MedianBlur", "BilateralBlur"):
                return _apply_op(img, op_name, n, kx, ky, d, thresh,
                                 prev=prev, inp=inp, kshape=kshape)

            if op_name in ("HistEq", "CLAHE", "Gamma",
                           "Normalize", "Retinex"):
                if not is_bgr:
                    return _apply_op(img, op_name, n, kx, ky, d, thresh,
                                     prev=prev, inp=inp, kshape=kshape)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l = lab[:, :, 0]
                l_new = _apply_op(l, op_name, n, kx, ky, d, thresh,
                                  prev=prev, inp=inp, kshape=kshape)
                lab[:, :, 0] = l_new
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if op_name == "Sharpen":
                if not is_bgr:
                    return _apply_op(img, op_name, n, kx, ky, d, thresh,
                                     prev=prev, inp=inp, kshape=kshape)
                chans = [_apply_op(img[:, :, i], op_name, n, kx, ky, d,
                                    thresh, prev=prev, inp=inp,
                                    kshape=kshape)
                         for i in range(3)]
                return cv2.merge(chans)

            if op_name in ("Laplacian", "Sobel", "Canny"):
                if not is_bgr:
                    return _apply_op(img, op_name, n, kx, ky, d, thresh,
                                     prev=prev, inp=inp, kshape=kshape)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                out = _apply_op(gray, op_name, n, kx, ky, d, thresh,
                                prev=prev, inp=inp, kshape=kshape)
                return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

            # Morph ops on an image (rare, but supported) go through
            # _apply_op which already handles `kshape`.
            if op_name in MORPH_OPS:
                return _apply_op(img, op_name, n, kx, ky, d, thresh,
                                 prev=prev, inp=inp, kshape=kshape)

            # Unknown / unsupported: pass through.
            return img

        def _step_kshape(step_tup):
            """Resolve a step's per-step Kernel-shape into a cv2 MORPH_*
            constant. Falls back to the global `xy_kshape` when the
            step has no kshape state yet (e.g. freshly loaded from an
            old config)."""
            _st = (getattr(self, "_kshape_state", {}) or {}
                   ).get(id(step_tup))
            if _st is None:
                return xy_kshape
            try:
                return KERNEL_SHAPES.get(_st["kshape"].get(), xy_kshape)
            except Exception:
                return xy_kshape

        def _apply_pm_targeted(img, target, op_name, n, kx, ky, d,
                               thresh, kshape=None):
            """Apply a PM op to the whole image OR to a single HSV
            channel of it. `target` is 'BGR' (whole image), or
            'H'/'S'/'V'. For a per-channel target on a 3-channel BGR
            image the op runs only on that channel and the image is
            returned in the same BGR layout, so downstream HSV / IR
            detection still works. A non-BGR (grayscale IR) image
            ignores the target and runs whole-image."""
            if (target in ("H", "S", "V")
                    and img.ndim == 3 and img.shape[2] == 3):
                _ci  = {"H": 0, "S": 1, "V": 2}[target]
                _hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                _ch  = _hsv[:, :, _ci]
                _ch2 = _apply_op_image(_ch, op_name, n, kx, ky, d,
                                       thresh, kshape=kshape)
                if _ch2.ndim == 3:
                    _ch2 = cv2.cvtColor(_ch2, cv2.COLOR_BGR2GRAY)
                if _ch2.shape != _ch.shape:
                    _ch2 = cv2.resize(_ch2,
                                      (_ch.shape[1], _ch.shape[0]))
                _hsv[:, :, _ci] = _ch2
                return cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)
            return _apply_op_image(img, op_name, n, kx, ky, d, thresh,
                                   kshape=kshape)

        def _run_pre_morph_stage(image, pre_pipeline, snap_slots,
                                  view_prefix):
            """Apply every step in the dedicated pre-morph pipeline
            `pre_pipeline` to `image` in user order. `snap_slots` is
            a list pre-sized to len(pre_pipeline); per-step images go
            into their matching index. Returns the cumulative image.
            Each step's Target ('BGR'/'H'/'S'/'V') picks whether the
            op runs on the whole image or a single HSV channel.
            Combine / YOLO add-ons are intentionally ignored — the
            pre-morph stage is single-pass image-to-image."""
            running = image
            for _idx, _step_tup in enumerate(pre_pipeline):
                (en, op, n, dr, kx, ky, th, *_) = _step_tup
                if en.get():
                    _tgt = self._pm_target_for(_step_tup)
                    running = _apply_pm_targeted(
                        running, _tgt, op.get(),
                        max(1, n.get()), max(1, kx.get()),
                        max(1, ky.get()), dr.get(), th.get(),
                        kshape=_step_kshape(_step_tup))
                snap_slots[_idx] = running.copy()
                view_lookup[f"{view_prefix}_pre_step{_idx+1}"] = running
            return running

        def _apply_op(mask, op_name, n, kx, ky, d, thresh,
                      prev=None, inp=None, kshape=None):
            # `kshape` overrides the XY structuring-element shape for
            # this single call (one of cv2.MORPH_RECT / ELLIPSE / CROSS).
            # None falls back to the global Pre-process selection.
            _xy_sk = kshape if kshape is not None else xy_kshape
            if op_name in MORPH_OPS:
                if d == "X":   sk = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
                elif d == "Y": sk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
                else:          sk = cv2.getStructuringElement(_xy_sk,         (kx, ky))
                return cv2.morphologyEx(mask, MORPH_OPS[op_name], sk, iterations=n)
            elif op_name == "GaussBlur":
                return cv2.GaussianBlur(mask, (_odd(max(1, kx)),) * 2, 0)
            elif op_name == "MedianBlur":
                return cv2.medianBlur(mask, _odd(max(1, kx)))
            elif op_name == "BilateralBlur":
                return cv2.bilateralFilter(mask, kx, float(ky), float(ky))
            elif op_name == "Thresh_Binary":
                _, m = cv2.threshold(mask, max(0, min(255, thresh)),
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
            elif op_name == "AND_prev":
                ref = prev if prev is not None else mask
                return cv2.bitwise_and(mask, ref)
            elif op_name == "OR_prev":
                ref = prev if prev is not None else mask
                return cv2.bitwise_or(mask, ref)
            elif op_name == "XOR_prev":
                ref = prev if prev is not None else mask
                return cv2.bitwise_xor(mask, ref)
            elif op_name == "AND_input":
                ref = inp if inp is not None else mask
                return cv2.bitwise_and(mask, ref)
            elif op_name == "OR_input":
                ref = inp if inp is not None else mask
                return cv2.bitwise_or(mask, ref)
            elif op_name == "XOR_input":
                ref = inp if inp is not None else mask
                return cv2.bitwise_xor(mask, ref)
            elif op_name == "Invert":
                return cv2.bitwise_not(mask)
            elif op_name == "FillHoles":
                return _fill_holes(mask)
            return mask

        # NOTE: view_lookup was initialised earlier (top of _process)
        # so the pre-morph stage can register its per-step images.
        # Progressively populated with fully-qualified view names ->
        # uint8 mask (single-channel) OR BGR image for pre-morph
        # slots. _resolve_src + _coerce_mask handle either.

        def _coerce_mask(img, like):
            """Return img as a single-channel uint8 mask matching like.shape."""
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

        def _resolve_src(snap_steps, src_str, mask_pre):
            if src_str == "mask_pre":
                return mask_pre
            if src_str == "prev":
                return snap_steps[-1] if snap_steps else mask_pre
            if src_str.startswith("step_"):
                try:
                    si = int(src_str.split("_")[1]) - 1
                    return snap_steps[si] if si < len(snap_steps) else mask_pre
                except (ValueError, IndexError):
                    return mask_pre
            # Full view-name reference (rgb_step4, up_branch1, ir_thresh_det...)
            if src_str in view_lookup:
                return _coerce_mask(view_lookup[src_str], mask_pre)
            return mask_pre

        # BGR view lookup for OVERLAY combine (paint mask on raw image).
        # Populated EARLY (before the rgb/ir/branch pipelines run) so
        # OVERLAY's base can be the raw frame OR any HSV channel
        # visualisation OR a pre-blurred / pre-CLAHE'd source.
        # HSV full-colour visualisation: full saturation, but keep
        # the ORIGINAL value (brightness) channel so the image shows
        # the actual scene texture instead of collapsing every patch
        # of similar hue into a flat colour block.
        _hsv_pure = np.dstack([
            hsv[:, :, 0],
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
            hsv[:, :, 2],
        ])
        _early_hsv_full = cv2.cvtColor(_hsv_pure, cv2.COLOR_HSV2BGR)
        # HSV raw: the uint8 HSV array displayed AS-IS (no inverse
        # conversion), so the pixel colour literally encodes the raw
        # channel values (B<-H, G<-S, R<-V). Handy for diagnostics.
        _early_hsv_raw = hsv.astype(np.uint8)
        # Per-channel views show the RAW channel value as grayscale —
        # so the displayed pixel intensity == the channel value the
        # threshold (T / Hmin / Hmax / etc.) acts on. H is rescaled
        # 0..179 -> 0..255 so the full range fills the viewable range.
        _early_h_scaled = (hsv[:, :, 0].astype(np.float32) / 179 * 255
                            ).clip(0, 255).astype(np.uint8)
        _early_hsv_H = cv2.cvtColor(_early_h_scaled, cv2.COLOR_GRAY2BGR)
        _early_hsv_S = cv2.cvtColor(hsv[:, :, 1],   cv2.COLOR_GRAY2BGR)
        _early_hsv_V = cv2.cvtColor(hsv[:, :, 2],   cv2.COLOR_GRAY2BGR)
        # Colormap + paired views (H uses HSV cmap, S/V use JET) are
        # consumed ONLY by the All-Masks flowchart pair tiles and are
        # not panel-selectable — skip applyColorMap / vstack entirely
        # while that window is closed.
        _early_hsv_H_cmap = _early_hsv_S_cmap = _early_hsv_V_cmap = None
        _early_hsv_H_pair = _early_hsv_S_pair = _early_hsv_V_pair = None

        def _hsv_pair(ch_bgr, cmap_bgr):
            return np.vstack([ch_bgr, cmap_bgr])
        if _all_masks_open:
            _early_hsv_H_cmap = cv2.applyColorMap(_early_h_scaled,
                                                  cv2.COLORMAP_HSV)
            _early_hsv_S_cmap = cv2.applyColorMap(hsv[:, :, 1],
                                                  cv2.COLORMAP_JET)
            _early_hsv_V_cmap = cv2.applyColorMap(hsv[:, :, 2],
                                                  cv2.COLORMAP_JET)
            _early_hsv_H_pair = _hsv_pair(_early_hsv_H, _early_hsv_H_cmap)
            _early_hsv_S_pair = _hsv_pair(_early_hsv_S, _early_hsv_S_cmap)
            _early_hsv_V_pair = _hsv_pair(_early_hsv_V, _early_hsv_V_cmap)
        _ir_bgr = (f_ir if f_ir.ndim == 3
                   else cv2.cvtColor(f_ir, cv2.COLOR_GRAY2BGR))

        bgr_views = {
            "rgb_raw":         f_rgb,
            "rgb_blur":        work_rgb,
            "rgb_hsv_full":    _early_hsv_full,
            "rgb_hsv_raw":     _early_hsv_raw,
            "rgb_hsv_H":       _early_hsv_H,
            "rgb_hsv_S":       _early_hsv_S,
            "rgb_hsv_V":       _early_hsv_V,
            "rgb_hsv_H_cmap":  _early_hsv_H_cmap,
            "rgb_hsv_S_cmap":  _early_hsv_S_cmap,
            "rgb_hsv_V_cmap":  _early_hsv_V_cmap,
            "rgb_hsv_H_pair":  _early_hsv_H_pair,
            "rgb_hsv_S_pair":  _early_hsv_S_pair,
            "rgb_hsv_V_pair":  _early_hsv_V_pair,
            "ir_raw":          _ir_bgr,
            "ir_gray":         cv2.cvtColor(snap_ir_gray, cv2.COLOR_GRAY2BGR),
        }

        # ── YOLO helpers (defined EARLY so per-step YOLO add-ons
        # can run inference DURING pipeline processing — needed for
        # the "focus" / "subtract" modes that filter the running
        # mask with the YOLO box union). The helpers also support
        # the post-loop view-emission code (just calls them with
        # the same cache so each source view is inferred once). ──
        _step_yolo_cache = {}

        def _resolve_bgr_view(name):
            """Best-effort BGR lookup for a view name."""
            v = bgr_views.get(name)
            if v is None:
                try:
                    v = views.get(name)        # noqa: F821 (late-bound)
                except NameError:
                    v = None
            if v is None:
                v2 = view_lookup.get(name)
                if v2 is not None:
                    v = (v2 if v2.ndim == 3
                         else cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR))
            if v is None:
                v = f_rgb
            if v.ndim == 2:
                v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            return v

        def _yolo_class_ids(yst):
            """Resolve a per-step YOLO class selection into a frozenset
            of ticked class ids, or None (= keep every class). The UI
            stores a {class_id: BooleanVar} tick-box dict; an empty
            dict means the step card was never built so no filtering
            applies."""
            try:
                classes = yst.get("yolo_classes") or {}
            except Exception:
                return None
            if not isinstance(classes, dict) or not classes:
                return None
            try:
                return frozenset(int(cid) for cid, v in classes.items()
                                 if bool(v.get()))
            except Exception:
                return None

        def _run_yolo_for(src_name, class_ids=None):
            """Run YOLO on the named view (cached). Returns
            (boxed_bgr, mask_uint8, boxes_list). Box edges and label
            text are drawn in the user-chosen YOLO colour so they
            match the mask thumbnail outlines. `class_ids` (a frozenset
            or None) restricts which detections are kept — the cache
            key includes it so per-step class filters don't collide."""
            _ck = (src_name, class_ids)
            if _ck in _step_yolo_cache:
                return _step_yolo_cache[_ck]
            base = _resolve_bgr_view(src_name).copy()
            mask = np.zeros(base.shape[:2], dtype=np.uint8)
            boxes = []
            if self.yolo_model is None:
                _step_yolo_cache[_ck] = (base, mask, boxes)
                return base, mask, boxes
            try:
                _conf = self.sv["YOLO_Conf"].get() / 100.0
            except Exception:
                _conf = 0.5
            _box_color = self._yolo_mask_bgr()
            try:
                res = self.yolo_model(base, verbose=False, conf=_conf, device=self._yolo_device())[0]
                for box in res.boxes:
                    cid = int(box.cls[0])
                    v = self.yolo_class_enabled.get(cid)
                    if v is not None and not v.get():
                        continue
                    if class_ids is not None and cid not in class_ids:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lbl = (f"{res.names[cid]} "
                           f"{float(box.conf[0]):.2f}")
                    cv2.rectangle(base, (x1, y1), (x2, y2),
                                  _box_color, 2)
                    cv2.putText(base, lbl, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                _box_color, 1)
                    cv2.rectangle(mask, (x1, y1), (x2, y2),
                                  255, thickness=-1)
                    boxes.append((x1, y1, x2, y2))
            except Exception as _e:
                print(f"[per-step yolo error] {_e}")
            _step_yolo_cache[_ck] = (base, mask, boxes)
            return base, mask, boxes

        def _yolo_effective_src(yst):
            """Pick the right view name to feed YOLO based on the
            per-step input kind. 'image' uses the raw-view picker,
            'mask' uses the mask-view picker. Falls back to rgb_raw."""
            try:
                kind = yst.get("yolo_input_kind",
                               tk.StringVar(value="image")).get()
            except Exception:
                kind = "image"
            if kind == "mask":
                src_var = yst.get("yolo_mask_src") or yst.get("yolo_src")
            else:
                src_var = yst.get("yolo_src")
            return (src_var.get() if src_var is not None else "rgb_raw") \
                   or "rgb_raw"

        def _apply_yolo_mode(running_mask, step_tuple):
            """If this step's per-step YOLO is enabled with mode
            'focus' or 'subtract', AND/AND-NOT the running mask
            with the YOLO box union and return the modified mask.
            Otherwise return the running mask unchanged."""
            try:
                yst = (getattr(self, "_yolo_state", {}) or {}).get(
                    id(step_tuple))
                if yst is None:
                    return running_mask
                # A "yolo"-type step IS a YOLO step — treat it as
                # enabled regardless of the (now-implied) Enable box.
                _kind = self._step_kind_for(step_tuple)
                if not yst["yolo_en"].get() and _kind != "yolo":
                    return running_mask
                mode = yst["yolo_mode"].get()
                if mode not in ("focus", "subtract"):
                    return running_mask
                src = _yolo_effective_src(yst)
                _, ymask, _ = _run_yolo_for(src, _yolo_class_ids(yst))
                # Resize / coerce mask to match running_mask shape.
                if ymask.shape != running_mask.shape:
                    ymask = cv2.resize(ymask,
                                       (running_mask.shape[1],
                                        running_mask.shape[0]))
                if mode == "focus":
                    return cv2.bitwise_and(running_mask, ymask)
                else:  # "subtract"
                    return cv2.bitwise_and(running_mask,
                                           cv2.bitwise_not(ymask))
            except Exception as _e:
                print(f"[yolo apply mode] {_e}")
            return running_mask

        def _bgr_for_overlay(src_name, gray_fallback):
            """Return a 3-channel BGR base image for OVERLAY combine.
            "none" returns a black canvas same shape as gray_fallback."""
            if src_name in ("none", "", None):
                h, w = gray_fallback.shape[:2]
                return np.zeros((h, w, 3), dtype=np.uint8)
            v = bgr_views.get(src_name)
            if v is None:
                try:
                    v = views.get(src_name)
                except NameError:
                    v = None
            if v is None:
                v = gray_fallback
            if v.ndim == 2:
                v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            return v

        def _paint_overlay(base_bgr, mask_gray, color=(0, 0, 255), alpha=0.4):
            """Composite a binary mask onto a BGR image as a pale tinted layer."""
            if base_bgr.shape[:2] != mask_gray.shape[:2]:
                base_bgr = cv2.resize(base_bgr, (mask_gray.shape[1], mask_gray.shape[0]))
            
            # Create a color plane
            overlay = base_bgr.copy()
            paint = np.full_like(overlay, color, dtype=np.uint8)
            
            # Apply transparency only where the mask is active
            sel = mask_gray > 0
            overlay[sel] = cv2.addWeighted(overlay, 1 - alpha, paint, alpha, 0)[sel]
            return overlay

        def _paint_overlay_two(base_bgr, m1_gray, c1, m2_gray=None, c2=(255, 255, 0), alpha=0.4):
            """
            Modified: Applies both masks with alpha transparency. 
            Where they overlap, the colors will blend, revealing the intersection.
            """
            # Apply Mask 1
            out = _paint_overlay(base_bgr, m1_gray, c1, alpha)
            
            # Apply Mask 2 on top of the result of Mask 1
            if m2_gray is not None:
                # We use the same logic: where m2 is active, blend its color 
                # into the image that already contains the tint from m1.
                if out.shape[:2] != m2_gray.shape[:2]:
                    m2_gray = cv2.resize(m2_gray, (out.shape[1], out.shape[0]))
                
                paint2 = np.full_like(out, c2, dtype=np.uint8)
                sel2 = m2_gray > 0
                # Blending here ensures the 'overlap' area gets both color contributions
                out[sel2] = cv2.addWeighted(out, 1 - alpha, paint2, alpha, 0)[sel2]
                
            return out
        # Per-step overlay-color/2nd-mask state lives on the host class
        # (PipelineUIMixin populates it when the step UI is built).
        from pipeline_ui_mixin import OVERLAY_COLORS as _OV_COLORS

        def _run_step(running, snap_steps, op_var, n_var, kx_var, ky_var,
                      dir_var, thresh_var, comb_en_var, comb_op_var, comb_src_var,
                      mask_pre, branch_sink=None, branch_key=None,
                      overlay_sink=None, overlay_state=None,
                      kshape=None):
            # OVERLAY can never produce BGR running, but defensively
            # coerce in case some other code path produced a 3-channel.
            if running.ndim == 3:
                running = cv2.cvtColor(running, cv2.COLOR_BGR2GRAY)
            _prev = snap_steps[-1] if snap_steps else None
            if comb_en_var.get():
                # COMBINE step: result = running <op> source. Morph and
                # Combine are mutually exclusive step types.
                src_name = comb_src_var.get()
                ref_raw  = _resolve_src(snap_steps, src_name, mask_pre)
                ref      = _coerce_mask(ref_raw, running)
                if branch_sink is not None and branch_key is not None:
                    branch_sink[branch_key] = ref.copy()
                cop = comb_op_var.get()
                if cop == "AND":     return cv2.bitwise_and(running, ref)
                if cop == "OR":      return cv2.bitwise_or(running,  ref)
                if cop == "XOR":     return cv2.bitwise_xor(running, ref)
                if cop == "OVERLAY":
                    # OVERLAY is a VISUALISATION-only combine: the
                    # running mask passes through UNCHANGED, while a
                    # separate painted image is stored for display.
                    # C1 paints Mask 1 (running), C2 paints Mask 2.
                    if overlay_sink is not None and branch_key is not None:
                        try:
                            _c_running = _OV_COLORS["red"]    # Mask 1 -> C1
                            _c_m2      = _OV_COLORS["cyan"]   # Mask 2 -> C2
                            _m2        = None
                            _base_src  = "rgb_raw"
                            if overlay_state is not None:
                                _c_running = _OV_COLORS.get(
                                    overlay_state["color1"].get(), _c_running)
                                _c_m2      = _OV_COLORS.get(
                                    overlay_state["color2"].get(), _c_m2)
                                _base_src  = (overlay_state["base_src"].get()
                                              or "rgb_raw")
                                _m2_src = overlay_state["mask2_src"].get()
                                if _m2_src and _m2_src != "none":
                                    _m2_raw = _resolve_src(
                                        snap_steps, _m2_src, mask_pre)
                                    _m2 = _coerce_mask(_m2_raw, running)
                            base = _bgr_for_overlay(_base_src, running)
                            # Paint Mask 1 first (C1), then Mask 2 (C2)
                            # on top so the 2nd mask is the topmost layer.
                            overlay_sink[branch_key] = _paint_overlay_two(
                                base, running, _c_running, _m2, _c_m2)
                        except Exception:
                            pass
                    return running
                # Unknown op -> fall through to AND.
                return cv2.bitwise_and(running, ref)
            # MORPH step: apply Op to the running mask.
            return _apply_op(running, op_var.get(),
                             max(1, n_var.get()), max(1, kx_var.get()),
                             max(1, ky_var.get()), dir_var.get(),
                             thresh_var.get(), prev=_prev, inp=mask_pre,
                             kshape=kshape)

        # Seed view_lookup with pre-pipeline masks so rgb/ir pipeline
        # combine sources can reference them by full name.
        view_lookup.update({
            "rgb_m1":         m1,
            "rgb_m2":         m2,
            "rgb_hsv_mask":   snap_rgb_hsv_mask,
            "rgb_mask_pre":   snap_rgb_mask_pre,
            "ir_thresh":      snap_ir_thresh,
            "ir_mask_pre":    snap_ir_mask_pre,
        })
        if snap_fg_rgb is not None:
            view_lookup["rgb_bgsub"] = snap_fg_rgb

        # Helper that runs every user pipeline once and stores its
        # result + per-step intermediates in view_lookup. Called in two
        # passes: a "pre-pass" before rgb/ir (so RGB/IR steps can pick
        # up_<name> as a combine source), and a "post-pass" afterwards
        # (so user pipelines that pick rgb_step/ir_step as their combine
        # source get those values). Branch images are also captured.
        up_branch_store  = {}   # {("name", step_idx): branch_img}
        up_overlay_store = {}   # {("name", step_idx): overlay_bgr}
        rgb_overlay_imgs = {}   # {step_idx: overlay_bgr}
        ir_overlay_imgs  = {}   # {step_idx: overlay_bgr}
        up_step_snaps   = {}   # {"name": [step1_img, step2_img, ...]}
        def _run_user_pipelines():
            for _up in self.user_pipelines:
                try:
                    _nm   = _up["name"].get().strip() or "branch"
                    _src  = _up["source"].get()
                    _type = _up.get("type", tk.StringVar(value="rgb")).get()

                    # Per-branch PM (Pre-morph) pipeline — same idea
                    # as the main RGB/IR PM pipelines. Each step is a
                    # 10-tuple; only the morph fields (en/op/n/dr/kx/
                    # ky/th) are used. The PM stage runs on the raw
                    # branch SOURCE image (BGR for rgb/custom, gray
                    # for ir) and the modified image is what feeds
                    # HSV / IR detection below.
                    _pre_list = _up.get("pre_steps", []) or []
                    _pre_snap_slots = [None] * len(_pre_list)
                    if _type == "ir":
                        # Apply this branch's dedicated IR CLAHE first
                        # (independent from the main pipeline's CLAHE),
                        # then feed the PM stage.
                        _ir_in = snap_ir_gray
                        if (_up.get("use_clahe_ir")
                                and _up["use_clahe_ir"].get()):
                            try:
                                _cl = cv2.createCLAHE(
                                    clipLimit=float(max(
                                        1, _up["clahe_ir_clip"].get())),
                                    tileGridSize=(max(
                                        1, _up["clahe_ir_tile"].get()),) * 2)
                                _ir_in = _cl.apply(snap_ir_gray)
                            except Exception:
                                pass
                        _ir_pre = _run_pre_morph_stage(
                            _ir_in, _pre_list,
                            _pre_snap_slots, f"up_{_nm}")
                        _src_rgb_pre = f_rgb
                        _hsv_pre = hsv
                    else:
                        _src_rgb_pre = _run_pre_morph_stage(
                            f_rgb, _pre_list,
                            _pre_snap_slots, f"up_{_nm}")
                        _hsv_pre = cv2.cvtColor(_src_rgb_pre,
                                                cv2.COLOR_BGR2HSV)
                        _ir_pre  = snap_ir_gray
                    # Stash PM step snapshots on the branch so the
                    # flowchart / view picker can read them later.
                    _up["_pre_snap"] = _pre_snap_slots
                    # Publish the source-image pipeline so the branch
                    # flowchart can show what actually feeds detection.
                    # For rgb-type branches the source is BGR f_rgb;
                    # for ir-type branches it's the grayscale IR frame.
                    if _type == "ir":
                        view_lookup[f"up_{_nm}_src_raw"]  = snap_ir_gray
                        view_lookup[f"up_{_nm}_src_blur"] = _ir_pre
                    else:
                        view_lookup[f"up_{_nm}_src_raw"]  = f_rgb
                        view_lookup[f"up_{_nm}_src_blur"] = _src_rgb_pre
                        # Per-branch HSV channel grayscales taken from
                        # the PM-processed source — so the channel
                        # thumbnail in the flowchart pair tile reflects
                        # any per-channel PM step (e.g. a Blur on H).
                        try:
                            _hp_h = (_hsv_pre[:, :, 0].astype(np.float32)
                                     / 179 * 255).clip(0, 255).astype(np.uint8)
                            view_lookup[f"up_{_nm}_hsv_H"] = _hp_h
                            view_lookup[f"up_{_nm}_hsv_S"] = _hsv_pre[:, :, 1]
                            view_lookup[f"up_{_nm}_hsv_V"] = _hsv_pre[:, :, 2]
                        except Exception:
                            pass

                    # -- Per-branch channel detection ----------------
                    if _type == "rgb":
                        try:
                            _bh1 = cv2.inRange(_hsv_pre[:, :, 0],
                                               _up["h1_lo"].get(),
                                               _up["h1_hi"].get())
                            _bh2 = cv2.inRange(_hsv_pre[:, :, 0],
                                               _up["h2_lo"].get(),
                                               _up["h2_hi"].get())
                            _bs  = cv2.inRange(_hsv_pre[:, :, 1],
                                               _up["s_lo"].get(),
                                               _up["s_hi"].get())
                            _bv  = cv2.inRange(_hsv_pre[:, :, 2],
                                               _up["v_lo"].get(),
                                               _up["v_hi"].get())
                            view_lookup[f"up_{_nm}_h1_mask"] = _bh1
                            view_lookup[f"up_{_nm}_h2_mask"] = _bh2
                            view_lookup[f"up_{_nm}_s_mask"]  = _bs
                            view_lookup[f"up_{_nm}_v_mask"]  = _bv
                        except Exception:
                            _bh1 = _bh2 = _bs = _bv = None

                        # Channel mode controls which channels participate
                        # in the AND combination; H mode means H1 OR H2.
                        _ch_mode = _up.get("channels",
                                           tk.StringVar(value="HSV")).get()
                        _h_comb = None
                        if "H" in _ch_mode or _ch_mode == "full":
                            if _bh1 is not None:
                                _h_comb = _bh1
                            if _bh2 is not None:
                                _h_comb = (_bh2 if _h_comb is None
                                           else cv2.bitwise_or(_h_comb, _bh2))
                        _to_and = []
                        if _h_comb is not None:
                            _to_and.append(_h_comb)
                        if ("S" in _ch_mode or _ch_mode == "full") and _bs is not None:
                            _to_and.append(_bs)
                        if ("V" in _ch_mode or _ch_mode == "full") and _bv is not None:
                            _to_and.append(_bv)
                        if _to_and:
                            _det = _to_and[0]
                            for _m in _to_and[1:]:
                                if _m.shape != _det.shape:
                                    _m = cv2.resize(_m,
                                                    (_det.shape[1],
                                                     _det.shape[0]))
                                _det = cv2.bitwise_and(_det, _m)
                        else:
                            _det = np.zeros_like(rgb_mask)
                        view_lookup[f"up_{_nm}_det"] = _det

                    elif _type == "ir":
                        try:
                            _bir = cv2.inRange(_ir_pre,
                                               _up["ir_lo"].get(),
                                               _up["ir_hi"].get())
                            if _bir.shape != rgb_mask.shape:
                                _bir = cv2.resize(_bir,
                                                  (rgb_mask.shape[1],
                                                   rgb_mask.shape[0]))
                            view_lookup[f"up_{_nm}_ir_mask"] = _bir
                            view_lookup[f"up_{_nm}_det"]     = _bir
                        except Exception:
                            view_lookup[f"up_{_nm}_det"] = np.zeros_like(rgb_mask)

                    else:  # custom
                        # No detection - pass-through. _det = source view
                        # (or blank if source not yet computed).
                        _ref = view_lookup.get(_src)
                        if _ref is None:
                            _ref = np.zeros_like(rgb_mask)
                        view_lookup[f"up_{_nm}_det"] = _coerce_mask(
                            _ref, rgb_mask)

                    # Per-branch BG subtractor - each branch owns its
                    # OWN MOG2 instance with editable history/varth.
                    # Channel can be the full BGR frame ("rgb"/"ir") or
                    # a single HSV channel ("H"/"S"/"V") so you can
                    # subtract drift in only the saturation (or only
                    # hue, etc.) without colour or brightness noise.
                    if _up.get("use_bgsub") and _up["use_bgsub"].get():
                        _bg_src = _up["bgsub_src"].get()
                        _hist   = max(10, _up["bgsub_history"].get())
                        _varth  = max(1, _up["bgsub_varth"].get())
                        # Channel is part of the signature: switching
                        # H<->V means the trained model is incompatible
                        # so we rebuild MOG2.
                        _sig    = (_hist, _varth, _bg_src)
                        if (_up.get("_backSub") is None
                                or _up.get("_backSub_sig") != _sig):
                            try:
                                _up["_backSub"] = (
                                    cv2.createBackgroundSubtractorMOG2(
                                        history=_hist,
                                        varThreshold=_varth,
                                        detectShadows=False))
                                _up["_backSub_sig"] = _sig
                            except Exception:
                                _up["_backSub"] = None
                        _bs = _up.get("_backSub")
                        if _bs is not None:
                            # Pick the input frame for MOG2:
                            #   "rgb" -> full BGR frame
                            #   "ir"  -> IR frame as BGR
                            #   "H"   -> hsv[:, :, 0]   (uint8)
                            #   "S"   -> hsv[:, :, 1]
                            #   "V"   -> hsv[:, :, 2]
                            if _bg_src == "rgb":
                                _base = f_rgb
                            elif _bg_src == "ir":
                                _base = (f_ir if f_ir.ndim == 3
                                         else cv2.cvtColor(f_ir,
                                              cv2.COLOR_GRAY2BGR))
                            elif _bg_src == "H":
                                _base = hsv[:, :, 0]
                            elif _bg_src == "S":
                                _base = hsv[:, :, 1]
                            elif _bg_src == "V":
                                _base = hsv[:, :, 2]
                            else:
                                _base = f_rgb
                            try:
                                _fg_branch = _bs.apply(_base)
                                _fg_g = _coerce_mask(_fg_branch, rgb_mask)
                                _cur  = view_lookup.get(f"up_{_nm}_det")
                                if _cur is not None:
                                    view_lookup[f"up_{_nm}_det"] = \
                                        cv2.bitwise_and(_cur, _fg_g)
                                view_lookup[f"up_{_nm}_bgsub"] = _fg_g
                                # Source image with BG removed —
                                # appears in the branch's source-image
                                # pipeline next to the FG-mask thumb.
                                try:
                                    if _type == "ir":
                                        _src_full = (cv2.cvtColor(
                                            _ir_pre,
                                            cv2.COLOR_GRAY2BGR)
                                            if _ir_pre.ndim == 2
                                            else _ir_pre)
                                    else:
                                        _src_full = _src_rgb_pre
                                    if _fg_g.shape != _src_full.shape[:2]:
                                        _fg_resized = cv2.resize(
                                            _fg_g,
                                            (_src_full.shape[1],
                                             _src_full.shape[0]))
                                    else:
                                        _fg_resized = _fg_g
                                    view_lookup[f"up_{_nm}_src_bgsub"] = \
                                        cv2.bitwise_and(
                                            _src_full, _src_full,
                                            mask=_fg_resized)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                    # Resolve actual source for pipeline steps.
                    _ref = view_lookup.get(_src)
                    if _ref is None:
                        _ref = np.zeros_like(rgb_mask)
                    _src_g = _coerce_mask(_ref, rgb_mask)
                    _running   = _src_g.copy()
                    _snaps     = []
                    for _u_idx, _ust in enumerate(_up["steps"]):
                        if _ust[0].get():
                            # YOLO-type steps skip morph/combine — the
                            # step is purely a YOLO focus/subtract.
                            if self._step_kind_for(_ust) != "yolo":
                                _u_ov_st = self._overlay_state_for(_ust)
                                _running = _run_step(
                                    _running, _snaps,
                                    _ust[1], _ust[2], _ust[4], _ust[5],
                                    _ust[3], _ust[6],
                                    _ust[7], _ust[8], _ust[9],
                                    _src_g,                 # mask_pre = source
                                    branch_sink=up_branch_store,
                                    branch_key=(_nm, _u_idx),
                                    overlay_sink=up_overlay_store,
                                    overlay_state=_u_ov_st,
                                    kshape=_step_kshape(_ust),
                                )
                            _running = _apply_yolo_mode(_running, _ust)
                        _snaps.append(_running.copy())
                        view_lookup[f"up_{_nm}_step{_u_idx+1}"] = _running
                    view_lookup[f"up_{_nm}"] = _running
                    up_step_snaps[_nm] = _snaps
                except Exception as _e:
                    print(f"[user-pipeline pre-pass error] {_e}")

        # Pre-pass: rgb/ir steps can reference user-pipeline outputs.
        _run_user_pipelines()

        # --- RGB & IR PRE-MORPH stages -------------------------------
        # Pre-morph steps run on the raw IMAGE; the modified image is
        # then sent through HSV / threshold detection to produce the
        # binary mask the morph stage consumes.
        rgb_branch_imgs = {}
        ir_branch_imgs  = {}

        _work_rgb_pre = _run_pre_morph_stage(
            work_rgb, self.rgb_pre_pipeline, snap_rgb_pre_steps, "rgb")
        _ir_fg_pre = _run_pre_morph_stage(
            ir_fg, self.ir_pre_pipeline, snap_ir_pre_steps, "ir")

        # Re-run HSV / threshold detection on the post-pre-morph image
        # (overwrites the placeholder values computed near the top).
        hsv = cv2.cvtColor(_work_rgb_pre, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        m2  = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        red_mask = cv2.bitwise_or(m1, m2)
        snap_rgb_hsv_mask = red_mask.copy()
        if self.use_bgsub.get() and self.backSub_rgb and snap_fg_rgb is not None:
            red_mask = cv2.bitwise_and(red_mask, snap_fg_rgb)
        rgb_mask = red_mask.copy()
        snap_rgb_mask_pre = rgb_mask.copy()

        _, ir_bin = cv2.threshold(_ir_fg_pre, irt, 255,
                                   cv2.THRESH_BINARY)
        ir_mask          = ir_bin.copy()
        snap_ir_thresh   = ir_bin.copy()
        snap_ir_mask_pre = ir_mask.copy()

        # Refresh view_lookup seeds with the post-pre-morph detection
        # so combine sources referencing rgb_m1 / rgb_hsv_mask /
        # rgb_mask_pre / ir_thresh / ir_mask_pre by name get the
        # up-to-date values.
        view_lookup.update({
            "rgb_m1":         m1,
            "rgb_m2":         m2,
            "rgb_hsv_mask":   snap_rgb_hsv_mask,
            "rgb_mask_pre":   snap_rgb_mask_pre,
            "ir_thresh":      snap_ir_thresh,
            "ir_mask_pre":    snap_ir_mask_pre,
        })

        # --- RGB MORPH stage ---------------------------------------
        # The morph pipeline now accepts ANY op (Edge / Blur /
        # Illumination ops on a single-channel mask still work
        # — e.g. Sharpen makes the mask outline crisper). No more
        # auto-routing of ops out of this pipeline.
        for _idx, _step_tup in enumerate(self.rgb_pipeline):
            (en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var,
             comb_en_var, comb_op_var, comb_src_var) = _step_tup
            if en_var.get():
                # YOLO-type steps skip morph/combine entirely — the
                # step is purely a YOLO focus / subtract / box_only.
                if self._step_kind_for(_step_tup) != "yolo":
                    _ov_st = self._overlay_state_for(_step_tup)
                    rgb_mask = _run_step(rgb_mask, snap_rgb_steps, op_var, n_var, kx_var,
                                         ky_var, dir_var, thresh_var, comb_en_var,
                                         comb_op_var, comb_src_var, snap_rgb_mask_pre,
                                         branch_sink=rgb_branch_imgs, branch_key=_idx,
                                         overlay_sink=rgb_overlay_imgs,
                                         overlay_state=_ov_st,
                                         kshape=_step_kshape(_step_tup))
                # YOLO mode: focus / subtract filters the running mask
                # by the YOLO box union (no-op if mode == box_only).
                rgb_mask = _apply_yolo_mode(rgb_mask, _step_tup)
            snap_rgb_steps[_idx] = rgb_mask.copy()
            view_lookup[f"rgb_step{_idx+1}"] = rgb_mask
        view_lookup["rgb_mask"] = rgb_mask

        # --- IR MORPH stage ----------------------------------------
        for _idx, _step_tup in enumerate(self.ir_pipeline):
            (en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var,
             comb_en_var, comb_op_var, comb_src_var) = _step_tup
            if en_var.get():
                if self._step_kind_for(_step_tup) != "yolo":
                    _ov_st = self._overlay_state_for(_step_tup)
                    ir_mask = _run_step(ir_mask, snap_ir_steps, op_var, n_var, kx_var,
                                        ky_var, dir_var, thresh_var, comb_en_var,
                                        comb_op_var, comb_src_var, snap_ir_mask_pre,
                                        branch_sink=ir_branch_imgs, branch_key=_idx,
                                        overlay_sink=ir_overlay_imgs,
                                        overlay_state=_ov_st,
                                        kshape=_step_kshape(_step_tup))
                ir_mask = _apply_yolo_mode(ir_mask, _step_tup)
            snap_ir_steps[_idx] = ir_mask.copy()
            view_lookup[f"ir_step{_idx+1}"] = ir_mask
        view_lookup["ir_mask"] = ir_mask

        # Post-morph blur was removed — to blur the mask after morph
        # ops, add a Blur step to the END of the Morph pipeline.
        snap_rgb_post_blur = rgb_mask.copy()
        snap_ir_post_blur  = ir_mask.copy()

        # RGB detection overlay
        rgb_det = f_rgb.copy()
        cnts, _ = cv2.findContours(rgb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_cable_px = int(np.sum(rgb_mask > 0))
        if self.show_boxes.get():
            for c in cnts:
                if cv2.contourArea(c) > mna:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(rgb_det, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(rgb_det, "cable", (x, max(0, y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if self.show_overlay.get():
            ov = np.zeros_like(rgb_det)
            ov[rgb_mask > 0] = list(self._rgb_mask_bgr())
            rgb_det = cv2.addWeighted(rgb_det, 0.65, ov, 0.35, 0)

        # IR false-colour display
        cmap_id = IR_COLORMAPS.get(self.ir_cmap_var.get(), None)
        ir_display_base = (cv2.applyColorMap(ir_gray, cmap_id)
                           if cmap_id is not None
                           else cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR))

        # IR detection overlay
        ir_det      = ir_display_base.copy()
        ir_cnts, _  = cv2.findContours(ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ir_cable_px = int(np.sum(ir_mask > 0))
        if self.show_boxes.get():
            for c in ir_cnts:
                if cv2.contourArea(c) > mna:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(ir_det, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(ir_det, "cable", (x, max(0, y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if self.show_overlay.get():
            ov_ir = np.zeros_like(ir_det)
            ov_ir[ir_mask > 0] = list(self._ir_mask_bgr())
            ir_det = cv2.addWeighted(ir_det, 0.65, ov_ir, 0.35, 0)

        # YOLO - multi-class with per-class enable filter
        yolo_rgb_n = yolo_ir_n = 0
        # Build per-class boolean mask once per frame.
        def _class_enabled(cid):
            v = self.yolo_class_enabled.get(int(cid))
            return True if v is None else bool(v.get())

        # Per-class binary masks (built so branches can use them via
        # views like "yolo_class_<id>_rgb" / _ir).
        yolo_class_masks_rgb = {}
        yolo_class_masks_ir  = {}

        if self.use_yolo.get() and self.yolo_model:
            _ydet_color = self._yolo_mask_bgr()
            res_rgb = self.yolo_model(f_rgb, verbose=False, conf=conf_thr, device=self._yolo_device())[0]
            for box in res_rgb.boxes:
                _cid = int(box.cls[0])
                if not _class_enabled(_cid):
                    continue
                yolo_rgb_n += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_rgb.names[_cid]} {float(box.conf[0]):.2f}"
                cv2.rectangle(rgb_det, (x1, y1), (x2, y2), _ydet_color, 2)
                cv2.putText(rgb_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ydet_color, 1)
                # Per-class binary mask (filled box).
                m = yolo_class_masks_rgb.setdefault(
                    _cid, np.zeros(f_rgb.shape[:2], dtype=np.uint8))
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, thickness=-1)
            ir_bgr = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            res_ir = self.yolo_model(ir_bgr, verbose=False, conf=conf_thr, device=self._yolo_device())[0]
            for box in res_ir.boxes:
                _cid = int(box.cls[0])
                if not _class_enabled(_cid):
                    continue
                yolo_ir_n += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_ir.names[_cid]} {float(box.conf[0]):.2f}"
                cv2.rectangle(ir_det, (x1, y1), (x2, y2), _ydet_color, 2)
                cv2.putText(ir_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ydet_color, 1)
                m = yolo_class_masks_ir.setdefault(
                    _cid, np.zeros(ir_bgr.shape[:2], dtype=np.uint8))
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, thickness=-1)

        # -- Build views dict ----------------------------------------------
        def _mask_bgr(m, color):
            out = np.zeros((*m.shape, 3), dtype=np.uint8)
            out[m > 0] = color
            return out
        def _g2b(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        _blank = np.zeros((*f_rgb.shape[:2], 3), dtype=np.uint8)

        # HSV full-colour viz: full saturation but keep the real V
        # channel so structure is visible (otherwise every region of
        # similar hue collapses to one flat colour block).
        _hsv_pure = np.dstack([
            hsv[:, :, 0],
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
            hsv[:, :, 2],
        ])
        _rgb_hsv_full = cv2.cvtColor(_hsv_pure, cv2.COLOR_HSV2BGR)
        _rgb_hsv_raw  = hsv.astype(np.uint8)
        # Grayscale visualisations of the H/S/V channels — the
        # displayed pixel intensity equals the channel's actual
        # value, so thresholding sliders match what you see.
        _h_scaled  = (hsv[:, :, 0].astype(np.float32) / 179 * 255
                      ).clip(0, 255).astype(np.uint8)
        _rgb_hsv_H = cv2.cvtColor(_h_scaled,    cv2.COLOR_GRAY2BGR)
        _rgb_hsv_S = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        _rgb_hsv_V = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)
        # Colormap + paired views — All-Masks-only (see above); skip
        # the applyColorMap / resize / vstack work while it is closed.
        _rgb_hsv_H_cmap = _rgb_hsv_S_cmap = _rgb_hsv_V_cmap = None
        _rgb_hsv_H_pair = _rgb_hsv_S_pair = _rgb_hsv_V_pair = None

        def _hsv_pair(ch_bgr, cmap_bgr):
            h, w = ch_bgr.shape[:2]
            half = max(1, h // 2)
            a = cv2.resize(ch_bgr,   (w, half), interpolation=cv2.INTER_AREA)
            b = cv2.resize(cmap_bgr, (w, half), interpolation=cv2.INTER_AREA)
            return np.vstack([a, b])
        if _all_masks_open:
            _rgb_hsv_H_cmap = cv2.applyColorMap(_h_scaled,    cv2.COLORMAP_HSV)
            _rgb_hsv_S_cmap = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_JET)
            _rgb_hsv_V_cmap = cv2.applyColorMap(hsv[:, :, 2], cv2.COLORMAP_JET)
            _rgb_hsv_H_pair = _hsv_pair(_rgb_hsv_H, _rgb_hsv_H_cmap)
            _rgb_hsv_S_pair = _hsv_pair(_rgb_hsv_S, _rgb_hsv_S_cmap)
            _rgb_hsv_V_pair = _hsv_pair(_rgb_hsv_V, _rgb_hsv_V_cmap)

        # User-chosen pipeline mask colours (live; refreshed every frame
        # so editing the hex updates all downstream views immediately).
        _RGB_MC = list(self._rgb_mask_bgr())
        _IR_MC  = list(self._ir_mask_bgr())

        views = {
            "rgb_raw":         f_rgb,
            "rgb_blur":        work_rgb,
            "rgb_hsv_full":    _rgb_hsv_full,
            "rgb_hsv_raw":     _rgb_hsv_raw,
            "rgb_hsv_H":       _rgb_hsv_H,
            "rgb_hsv_S":       _rgb_hsv_S,
            "rgb_hsv_V":       _rgb_hsv_V,
            "rgb_hsv_H_cmap":  _rgb_hsv_H_cmap,
            "rgb_hsv_S_cmap":  _rgb_hsv_S_cmap,
            "rgb_hsv_V_cmap":  _rgb_hsv_V_cmap,
            "rgb_hsv_H_pair":  _rgb_hsv_H_pair,
            "rgb_hsv_S_pair":  _rgb_hsv_S_pair,
            "rgb_hsv_V_pair":  _rgb_hsv_V_pair,
            "rgb_m1":       _mask_bgr(m1,                  _RGB_MC),
            "rgb_m2":       _mask_bgr(m2,                  _RGB_MC),
            "rgb_hsv_mask": _mask_bgr(snap_rgb_hsv_mask,   _RGB_MC),
            "rgb_bgsub":    _g2b(snap_fg_rgb) if snap_fg_rgb is not None else _blank,
            "rgb_mask_pre": _mask_bgr(snap_rgb_mask_pre,   _RGB_MC),
            "rgb_mask":     _mask_bgr(rgb_mask,            _RGB_MC),
            "rgb_post_blur":_g2b(snap_rgb_post_blur),
            "rgb_det":      rgb_det,
            "ir_raw":       ir_display_base,
            "ir_gray":      _g2b(snap_ir_gray),
            "ir_blur":      _g2b(snap_ir_blur),
            "ir_clahe":     _g2b(snap_ir_clahe),
            "ir_bgsub":     _g2b(snap_ir_fg),
            "ir_thresh":    _mask_bgr(snap_ir_thresh,      _IR_MC),
            "ir_mask_pre":  _mask_bgr(snap_ir_mask_pre,    _IR_MC),
            "ir_mask":      _mask_bgr(ir_mask,             _IR_MC),
            "ir_post_blur": _g2b(snap_ir_post_blur),
            "ir_det":       ir_det,
        }
        # -- YOLO per-class masks (binary box masks, useful as
        # combine sources or as a YOLO-branch input) -----------------
        if self.yolo_model:
            try:
                _names = (self.yolo_model.names
                          if isinstance(self.yolo_model.names, dict)
                          else dict(enumerate(self.yolo_model.names)))
            except Exception:
                _names = {}
            for _cid, _cname in _names.items():
                _cid = int(_cid)
                _slug = str(_cname).strip().replace(" ", "_") or f"c{_cid}"
                _mr = yolo_class_masks_rgb.get(_cid)
                _mi = yolo_class_masks_ir.get(_cid)
                if _mr is None:
                    _mr = np.zeros(f_rgb.shape[:2], dtype=np.uint8)
                if _mi is None:
                    _mi = np.zeros(ir_gray.shape[:2], dtype=np.uint8)
                views[f"yolo_{_slug}_rgb"] = _mask_bgr(_mr, list(self._yolo_mask_bgr()))
                views[f"yolo_{_slug}_ir"]  = _mask_bgr(_mi, list(self._yolo_mask_bgr()))
                view_lookup[f"yolo_{_slug}_rgb"] = _mr
                view_lookup[f"yolo_{_slug}_ir"]  = _mi
            # "Any class" union mask (intended for YOLO branch type
            # default) - bitwise OR across all per-class masks.
            if yolo_class_masks_rgb:
                _any_rgb = next(iter(yolo_class_masks_rgb.values())).copy()
                for _m in list(yolo_class_masks_rgb.values())[1:]:
                    _any_rgb = cv2.bitwise_or(_any_rgb, _m)
            else:
                _any_rgb = np.zeros(f_rgb.shape[:2], dtype=np.uint8)
            if yolo_class_masks_ir:
                _any_ir = next(iter(yolo_class_masks_ir.values())).copy()
                for _m in list(yolo_class_masks_ir.values())[1:]:
                    _any_ir = cv2.bitwise_or(_any_ir, _m)
            else:
                _any_ir = np.zeros(ir_gray.shape[:2], dtype=np.uint8)
            views["yolo_any_rgb"]      = _mask_bgr(_any_rgb, list(self._yolo_mask_bgr()))
            views["yolo_any_ir"]       = _mask_bgr(_any_ir,  list(self._yolo_mask_bgr()))
            view_lookup["yolo_any_rgb"] = _any_rgb
            view_lookup["yolo_any_ir"]  = _any_ir

        # Per-step YOLO add-on emit. The helpers
        # (_resolve_bgr_view, _run_yolo_for, _step_yolo_cache) were
        # defined earlier so they could be used at step time for
        # focus / subtract modes — re-using the same cache here
        # avoids duplicate inference on the same source view.

        def _emit_step_yolo(step_tuple, vid_prefix):
            """
            Modified: 
            - YOLO Mask: Pipeline mask + hollow boxes.
            - YOLO Box (output): YOLO detections + pale tinted pipeline mask overlay.
            """
            try:
                yst = (getattr(self, "_yolo_state", {}) or {}).get(id(step_tuple))
                if yst is None:
                    return
                if (not yst["yolo_en"].get()
                        and self._step_kind_for(step_tuple) != "yolo"):
                    return

                src = _yolo_effective_src(yst)
                boxed_bgr, mask_bin, boxes = _run_yolo_for(
                    src, _yolo_class_ids(yst))

                # 1. Get the current pipeline mask for this step
                step_mask_gray = view_lookup.get(vid_prefix, np.zeros_like(mask_bin))
                
                # 2. Identify colors
                pipe_color = _RGB_MC if "rgb" in vid_prefix else _IR_MC
                yolo_color = self._yolo_mask_bgr()

                # --- PART A: YOLO Box Output (Tinted Overlay) ---
                # Start with the BGR image that already has YOLO boxes/labels
                boxed_output = boxed_bgr.copy()
                
                # Create a solid color layer for the mask
                overlay_layer = np.zeros_like(boxed_output)
                overlay_layer[step_mask_gray > 0] = pipe_color
                
                # Blend the color layer into the boxed image (0.35 alpha for "pale" effect)
                boxed_output = cv2.addWeighted(boxed_output, 1.0, overlay_layer, 0.35, 0)

                # --- PART B: YOLO Mask Visual ---
                viz_mask = np.zeros((*mask_bin.shape, 3), dtype=np.uint8)
                viz_mask[step_mask_gray > 0] = pipe_color
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(viz_mask, (x1, y1), (x2, y2), yolo_color, 2)

                # 3. Save the raw source view (no boxes)
                _raw_src = _resolve_bgr_view(src)
                if _raw_src is not None:
                    views[f"{vid_prefix}_yolo_raw"] = _raw_src.copy()
                
                # 4. Update the dictionary
                views[f"{vid_prefix}_yolo"] = boxed_output  # Now with pale overlay
                views[f"{vid_prefix}_yolo_mask"] = viz_mask
                view_lookup[f"{vid_prefix}_yolo_mask"] = mask_bin
                
            except Exception as _e:
                print(f"[step-yolo emit] {_e}")

        for _i, _st in enumerate(self.rgb_pipeline):
            _emit_step_yolo(_st, f"rgb_step{_i+1}")
        for _i, _st in enumerate(self.ir_pipeline):
            _emit_step_yolo(_st, f"ir_step{_i+1}")
        for _up in self.user_pipelines:
            _nm = _up["name"].get().strip() or "branch"
            for _i, _st in enumerate(_up["steps"]):
                _emit_step_yolo(_st, f"up_{_nm}_step{_i+1}")

        # Dynamic step views — pre-morph slots hold BGR images, morph
        # slots hold single-channel masks. Display each appropriately.
        def _step_view(snap, color):
            if snap is None:
                return None
            if snap.ndim == 3:
                return snap
            if snap.ndim == 2:
                return _mask_bgr(snap, color)
            return None
        for _i, _s in enumerate(snap_rgb_steps):
            _v = _step_view(_s, _RGB_MC)
            if _v is not None:
                views[f"rgb_step{_i+1}"] = _v
        for _i, _s in enumerate(snap_ir_steps):
            _v = _step_view(_s, _IR_MC)
            if _v is not None:
                views[f"ir_step{_i+1}"] = _v
        # Pre-morph step views — published under `<chan>_pre_step{N}`
        # so the All-Masks canvas / picker can show / reference them.
        for _i, _s in enumerate(snap_rgb_pre_steps):
            _v = _step_view(_s, _RGB_MC)
            if _v is not None:
                views[f"rgb_pre_step{_i+1}"] = _v
        for _i, _s in enumerate(snap_ir_pre_steps):
            _v = _step_view(_s, _IR_MC)
            if _v is not None:
                views[f"ir_pre_step{_i+1}"] = _v
        # Branch images for combine-enabled steps (the "before-combine" preview)
        for _i, _br in rgb_branch_imgs.items():
            views[f"rgb_step{_i+1}_branch"] = _mask_bgr(_br, [255, 128, 255])
        for _i, _br in ir_branch_imgs.items():
            views[f"ir_step{_i+1}_branch"] = _mask_bgr(_br, [255, 128, 128])
        # OVERLAY combine views - these are ALREADY BGR images
        # (raw + mask painted on top), so DO NOT pass them through
        # _mask_bgr (which expects a single-channel mask).
        for _i, _ov in rgb_overlay_imgs.items():
            views[f"rgb_step{_i+1}_overlay"] = _ov
        for _i, _ov in ir_overlay_imgs.items():
            views[f"ir_step{_i+1}_overlay"] = _ov

        # -- Build the composite "RAW + MASK -> OVERLAY" view per OVERLAY step --
        from cv2 import FONT_HERSHEY_SIMPLEX as _F

        def _to_bgr(im):
            if im is None:
                return None
            if im.ndim == 2:
                return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            return im

        def _centre_text(canvas, text, scale, thickness):
            (tw, th), _ = cv2.getTextSize(text, _F, scale, thickness)
            x = max(0, (canvas.shape[1] - tw) // 2)
            y = (canvas.shape[0] + th) // 2
            cv2.putText(canvas, text, (x, y),
                        _F, scale, (255, 150, 255),
                        thickness, cv2.LINE_AA)

        def _glyph(text, tile_h, band_w):
            """Operator band sized in proportion to the tile so the
            composite stays sharp at any source resolution. The font
            scale is auto-shrunk so the rendered text never spills
            past the band — '->' was the worst case at full resolution
            because it was wider than the band and bled into the
            next tile."""
            band = np.zeros((tile_h, band_w, 3), dtype=np.uint8)
            txt = "+" if text == "+" else "->"
            scale = max(0.6, tile_h / 80.0)
            thick = max(1, tile_h // 40)
            (tw, _th), _ = cv2.getTextSize(txt, _F, scale, thick)
            # Clamp text width to ~80 % of band so the glyph has a
            # visible margin on both sides.
            limit = band_w * 0.8
            if tw > limit:
                shrink = limit / tw
                scale = max(0.4, scale * shrink)
                thick = max(1, int(thick * shrink))
            _centre_text(band, txt, scale, thick)
            return band

        def _make_composite(operands, overlay_bgr):
            """Build a composite [op1][+][op2][+]...[->][overlay] at the
            FULL source resolution so the zoom window shows crisp
            pixels (the flowchart thumbnail still gets shrunk down by
            the All-Masks update loop). Tile size is taken from the
            overlay image; smaller operands are upsized to match.
            Returns (composite_image, tile_images)."""
            ovl = _to_bgr(overlay_bgr)
            th, tw = ovl.shape[:2]
            band_w = max(48, tw // 3)  # operator band ~= 1/3 tile width

            tile_images = []
            for op in operands:
                t = _to_bgr(op)
                if t.shape[:2] != (th, tw):
                    t = cv2.resize(t, (tw, th))
                tile_images.append(t)
            tile_images.append(ovl)

            tiles = []
            for i, t in enumerate(tile_images):
                if i:
                    # Last separator is "->", earlier ones are "+".
                    sep = "->" if i == len(tile_images) - 1 else "+"
                    tiles.append(_glyph(sep, th, band_w))
                tiles.append(t)
            return np.hstack(tiles), tile_images

        def _resolve_raw_view(src_name):
            """Best BGR image for `src_name` (used as the "raw" tile)."""
            if src_name in views and views[src_name].ndim == 3:
                return views[src_name]
            v = bgr_views.get(src_name)
            if v is not None:
                return v
            v2 = view_lookup.get(src_name)
            if v2 is not None:
                return _to_bgr(v2)
            return _blank

        def _ov_state(step_tup):
            try:
                return self._overlay_state.get(id(step_tup))
            except Exception:
                return None

        def _build_overlay_operands(step_tup, mask1_gray):
            """Return the ordered list of operand images (BGR or gray)
            for the composite, following per-step state. Mask 1 is
            ALWAYS the first element; Mask 2 / Base appear only when
            their src != 'none'."""
            ops = [mask1_gray]
            st = _ov_state(step_tup)
            if st is not None:
                _m2 = st["mask2_src"].get()
                if _m2 not in ("", "none"):
                    _m2_img = view_lookup.get(_m2)
                    if _m2_img is None:
                        try:
                            _m2_img = views.get(_m2)
                        except NameError:
                            _m2_img = None
                    if _m2_img is not None:
                        ops.append(_m2_img)
                _b = st["base_src"].get()
                if _b not in ("", "none"):
                    ops.append(_resolve_raw_view(_b))
            return ops

        # Per-composite tile layout (filled below) so the All-Masks
        # right-click handler can offer a "zoom this tile" menu.
        if not hasattr(self, "_composite_tile_views"):
            self._composite_tile_views = {}
        self._composite_tile_views.clear()

        def _publish_composite(vid_prefix, ops, overlay, step_tup):
            _cv, _tiles = _make_composite(ops, overlay)
            views[f"{vid_prefix}_composite"] = _cv
            # Tile labels mirror the operand order set up by
            # _build_overlay_operands (mask1, optional mask2, optional
            # base) followed by the overlay result.
            _labels = ["Mask 1"]
            st = _ov_state(step_tup)
            if st is not None:
                if st["mask2_src"].get() not in ("", "none"):
                    _labels.append("Mask 2")
                if st["base_src"].get() not in ("", "none"):
                    _labels.append("Base")
            _labels.append("Result")
            tile_entries = []
            for _ti, (_im, _lab) in enumerate(zip(_tiles, _labels)):
                _tv = f"{vid_prefix}_composite_tile{_ti+1}"
                views[_tv] = _im
                tile_entries.append((_tv, _lab))
            # Also expose `_result` as an alias of the overlay output
            # (== <vid>_overlay) so menu code can use a stable name.
            self._composite_tile_views[
                f"{vid_prefix}_composite"] = tile_entries

        # RGB pipeline composites
        for _i, _ov in rgb_overlay_imgs.items():
            try:
                _step_tup = self.rgb_pipeline[_i]
                _msk = snap_rgb_steps[_i] if _i < len(snap_rgb_steps) else _ov
                _ops = _build_overlay_operands(_step_tup, _msk)
                _publish_composite(f"rgb_step{_i+1}", _ops, _ov, _step_tup)
            except Exception:
                pass
        # IR pipeline composites
        for _i, _ov in ir_overlay_imgs.items():
            try:
                _step_tup = self.ir_pipeline[_i]
                _msk = snap_ir_steps[_i] if _i < len(snap_ir_steps) else _ov
                _ops = _build_overlay_operands(_step_tup, _msk)
                _publish_composite(f"ir_step{_i+1}", _ops, _ov, _step_tup)
            except Exception:
                pass


        # -- User pipelines (parallel branches) - POST-pass ----------------
        # Re-run after rgb/ir so any user-pipeline step that combines
        # with rgb_step{N} / ir_step{N} now sees those values too.
        _run_user_pipelines()

        _palette = {
            "cyan":    [255, 255,   0],
            "yellow":  [  0, 255, 255],
            "magenta": [255,   0, 255],
            "green":   [  0, 255,   0],
            "orange":  [  0, 165, 255],
            "red":     [  0,   0, 255],
            "white":   [255, 255, 255],
        }
        def _resolve_branch_color(s):
            """Branch colour can be either a named-palette key OR a
            #RRGGBB hex string. Falls back to soft amber on parse
            failure so it stays visible."""
            s = (s or "").strip()
            if s in _palette:
                return _palette[s]
            return list(self._hex_to_bgr(s, fallback=(0, 200, 255)))
        # Export user-pipeline results into the views dict for display.
        for up in self.user_pipelines:
            _nm = up["name"].get().strip() or "branch"
            _clr = _resolve_branch_color(up["color"].get())
            _final = view_lookup.get(f"up_{_nm}")
            if _final is None:
                continue
            # Always emit the coloured-mask version (default rendering).
            _mask_bgr_view = _mask_bgr(_final, _clr)
            views[f"up_{_nm}_mask"] = _mask_bgr_view
            # Overlay outputs: paint the mask on top of the rgb_raw or
            # ir_raw image so the user can see WHAT was detected on the
            # original frame.
            try:
                _ovr_rgb = cv2.bitwise_and(f_rgb, f_rgb,
                                           mask=_coerce_mask(_final, rgb_mask))
                views[f"up_{_nm}_overlay_rgb"] = _ovr_rgb
            except Exception:
                pass
            try:
                # HSV overlay: paint the mask on the rgb_hsv_full visual
                # (so the user sees which HSV-coloured pixels passed).
                _hsv_base = views.get("rgb_hsv_full", f_rgb)
                _ovr_hsv = cv2.bitwise_and(_hsv_base, _hsv_base,
                                           mask=_coerce_mask(_final, rgb_mask))
                views[f"up_{_nm}_overlay_hsv"] = _ovr_hsv
            except Exception:
                pass
            try:
                _ir_bgr = (f_ir if f_ir.ndim == 3
                           else cv2.cvtColor(f_ir, cv2.COLOR_GRAY2BGR))
                _mask_for_ir = _coerce_mask(_final, snap_ir_gray)
                _ovr_ir = cv2.bitwise_and(_ir_bgr, _ir_bgr, mask=_mask_for_ir)
                views[f"up_{_nm}_overlay_ir"] = _ovr_ir
            except Exception:
                pass
            # The default `up_<name>` view follows the user's Output choice.
            _ov = (up.get("overlay").get()
                   if up.get("overlay") else "mask")
            if _ov == "overlay_rgb" and f"up_{_nm}_overlay_rgb" in views:
                views[f"up_{_nm}"] = views[f"up_{_nm}_overlay_rgb"]
            elif _ov == "overlay_hsv" and f"up_{_nm}_overlay_hsv" in views:
                views[f"up_{_nm}"] = views[f"up_{_nm}_overlay_hsv"]
            elif _ov == "overlay_ir" and f"up_{_nm}_overlay_ir" in views:
                views[f"up_{_nm}"] = views[f"up_{_nm}_overlay_ir"]
            else:
                views[f"up_{_nm}"] = _mask_bgr_view
            for _i, _s in enumerate(up_step_snaps.get(_nm, [])):
                views[f"up_{_nm}_step{_i+1}"] = _mask_bgr(_s, _clr)
            # Per-branch detection masks for display & zoom.
            for _suf, _col in (("det",      _clr),
                                ("h1_mask",  [255,   0, 255]),
                                ("h2_mask",  [255, 100, 255]),
                                ("s_mask",   [255, 255,   0]),
                                ("v_mask",   [  0, 255, 255]),
                                ("ir_mask",  [255, 200,   0]),
                                ("bgsub",    [128, 255, 128])):
                _bm = view_lookup.get(f"up_{_nm}_{_suf}")
                if _bm is not None:
                    views[f"up_{_nm}_{_suf}"] = _mask_bgr(_bm, _col)
            # Source-image pipeline views (raw / pre-blur / bg-removed).
            # These are stored in view_lookup as the actual BGR / gray
            # frames; copy them to `views` so the flowchart can show
            # them as full thumbnails (no _mask_bgr colouring).
            for _suf in ("src_raw", "src_blur", "src_bgsub",
                         "hsv_H", "hsv_S", "hsv_V"):
                _v = view_lookup.get(f"up_{_nm}_{_suf}")
                if _v is None:
                    continue
                if _v.ndim == 2:
                    _v = cv2.cvtColor(_v, cv2.COLOR_GRAY2BGR)
                views[f"up_{_nm}_{_suf}"] = _v
            # Branch PM (pre-morph) step views — the branch flowchart
            # has PM step nodes (up_<nm>_pre_step{N}); without these
            # `views` entries they render as a black placeholder.
            # Snapshots are real images (BGR for rgb / gray for ir),
            # so display them directly (not _mask_bgr-coloured).
            for _i, _s in enumerate(up.get("_pre_snap") or []):
                if _s is None:
                    continue
                _pv = _s if _s.ndim == 3 else \
                    cv2.cvtColor(_s, cv2.COLOR_GRAY2BGR)
                views[f"up_{_nm}_pre_step{_i+1}"] = _pv
        for (_nm, _u_idx), _br in up_branch_store.items():
            views[f"up_{_nm}_step{_u_idx+1}_branch"] = \
                _mask_bgr(_br, [255, 128, 255])
        # OVERLAY views for branch combine steps - already BGR.
        for (_nm, _u_idx), _ov in up_overlay_store.items():
            views[f"up_{_nm}_step{_u_idx+1}_overlay"] = _ov
            # Composite "raw + mask -> overlay" for this branch step.
            try:
                _step = None
                _snaps = up_step_snaps.get(_nm, [])
                for _up in self.user_pipelines:
                    if _up["name"].get().strip() == _nm \
                       and 0 <= _u_idx < len(_up["steps"]):
                        _step = _up["steps"][_u_idx]
                        break
                if _step is not None:
                    _msk = (_snaps[_u_idx]
                            if _u_idx < len(_snaps) else _ov)
                    _ops = _build_overlay_operands(_step, _msk)
                    _publish_composite(f"up_{_nm}_step{_u_idx+1}",
                                       _ops, _ov, _step)
            except Exception:
                pass

        # -- Combine view (A AND/OR/XOR B) --------------------------------
        try:
            _ca   = self.combine_a_var.get()
            _cb   = self.combine_b_var.get()
            _cop  = self.combine_op_var.get()
            _ima  = views.get(_ca, _blank)
            _imb  = views.get(_cb, _blank)
            def _to_gray(im):
                return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            _ga = _to_gray(_ima)
            _gb = cv2.resize(_to_gray(_imb), (_ga.shape[1], _ga.shape[0]))
            if _cop == "AND":
                _comb = cv2.bitwise_and(_ga, _gb)
            elif _cop == "OR":
                _comb = cv2.bitwise_or(_ga, _gb)
            else:
                _comb = cv2.bitwise_xor(_ga, _gb)
            views["combined"] = cv2.cvtColor(_comb, cv2.COLOR_GRAY2BGR)
        except Exception:
            views["combined"] = _blank

        # -- Update main panels --------------------------------------------
        for key, lbl in self.panels.items():
            view_name = self.panel_view[key].get()
            img = views.get(view_name, views.get(key, _blank))
            self._put(lbl, img, bgr=True)

        # -- Update All-Masks window ---------------------------------------
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            fc_sz = getattr(self, "_all_masks_img_size", (FC_W, FC_H))
            for view_name, lbl in self.all_masks_labels.items():
                # Aliased keys carry the actual view name after "@".
                #   "<step>__img1@<view>" or "<step>__img2@<view>"     -> small
                #   "<step>__cmp@<view>"                               -> wide (3x FC_W)
                #   "<branch>__input@<view>" / other "@<view>" keys    -> full FC_WxFC_H
                if "__img" in view_name:
                    _vn = view_name.split("@", 1)[1]
                    _sz = (max(28, fc_sz[0] // 3 - 4),
                           max(20, fc_sz[1] // 3))
                elif "__cmp" in view_name:
                    _vn = view_name.split("@", 1)[1]
                    # The "__cmp" alias is followed by the tile count
                    # (2..4) so we know the exact composite width.
                    try:
                        _key_left = view_name.split("@", 1)[0]
                        _cnt_str  = _key_left.rsplit("__cmp", 1)[1]
                        _cnt = int(_cnt_str) if _cnt_str else 4
                    except Exception:
                        _cnt = 4
                    _sz = (_cnt * fc_sz[0] + (_cnt - 1) * 64, fc_sz[1])
                elif "@" in view_name:
                    _vn = view_name.split("@", 1)[1]
                    _sz = fc_sz
                else:
                    _vn = view_name
                    _sz = fc_sz
                # HSV pair views render at full thumbnail width but
                # double height (top = channel, bottom = cmap),
                # matching the 2*FC_H tile reserved by the flowchart.
                if _vn.endswith("_pair"):
                    _sz = (fc_sz[0], fc_sz[1] * 2)
                self._put(lbl, views.get(_vn, _blank), bgr=True, size=_sz)
            # Step-label formatter shared across rgb/ir/user pipelines.
            def _step_lbl(i, en, op_v, cen, cop_v, csr_v):
                if not en.get():
                    return f"S{i+1}: -"
                if cen.get():
                    return f"S{i+1}: +{cop_v.get()}({csr_v.get()})"
                return f"S{i+1}: {op_v.get()}"

            # PM step labels never use combine/overlay/YOLO add-ons,
            # so a tiny dedicated formatter keeps the label short. A
            # non-BGR Target channel is appended so the user can see
            # which channel an image-stage op runs on.
            def _pm_lbl(i, en, op_v, step_tup=None):
                if not en.get():
                    return f"PM{i+1}: -"
                _tg = (self._pm_target_for(step_tup)
                       if step_tup is not None else "BGR")
                _suf = f" [{_tg}]" if _tg and _tg != "BGR" else ""
                return f"PM{i+1}: {op_v.get()}{_suf}"

            # Live-update step node label text
            for _i, (_en, _op, _n, _d, _kx, _ky, _t, _cen, _cop, _csr) in enumerate(self.rgb_pipeline):
                _k = f"rgb_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=_step_lbl(_i, _en, _op, _cen, _cop, _csr))
            for _i, (_en, _op, _n, _d, _kx, _ky, _t, _cen, _cop, _csr) in enumerate(self.ir_pipeline):
                _k = f"ir_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=_step_lbl(_i, _en, _op, _cen, _cop, _csr))
            for _i, _pst in enumerate(self.rgb_pre_pipeline):
                _k = f"rgb_pre_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=_pm_lbl(_i, _pst[0], _pst[1], _pst))
            for _i, _pst in enumerate(self.ir_pre_pipeline):
                _k = f"ir_pre_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=_pm_lbl(_i, _pst[0], _pst[1], _pst))
            for _up in self.user_pipelines:
                _nm = _up["name"].get().strip() or "branch"
                for _i, (_en, _op, _n, _d, _kx, _ky, _t,
                         _cen, _cop, _csr) in enumerate(_up["steps"]):
                    _k = f"up_{_nm}_step{_i+1}"
                    if _k in self.all_masks_step_labels:
                        self.all_masks_step_labels[_k].config(
                            text=_step_lbl(_i, _en, _op, _cen, _cop, _csr))
                for _i, _pst in enumerate(_up.get("pre_steps") or []):
                    _k = f"up_{_nm}_pre_step{_i+1}"
                    if _k in self.all_masks_step_labels:
                        self.all_masks_step_labels[_k].config(
                            text=_pm_lbl(_i, _pst[0], _pst[1], _pst))

        # -- Zoom windows --------------------------------------------------
        # Multiple zoom windows can be open at once; each lives at
        # self._zoom_wins[view_name]. Re-render each one and store its
        # source image so the hover-readout can read native pixel
        # values.
        for _vn, _z in list(getattr(self, "_zoom_wins", {}).items()):
            if not _z["win"].winfo_exists():
                self._zoom_wins.pop(_vn, None)
                continue
            zoom_img = views.get(_vn, _blank)
            _z["src_img"] = zoom_img
            if _vn.endswith("_composite"):
                _zh = 360
                _zw = int(zoom_img.shape[1] * _zh / zoom_img.shape[0]) \
                       if zoom_img.ndim >= 2 and zoom_img.shape[0] > 0 else 1280
                _zw = min(_zw, 1600)
                self._put(_z["label"], zoom_img, bgr=True, size=(_zw, _zh))
            else:
                # Show at native source resolution (clamped to a sane
                # max box) so masks zoom in crisp pixel-for-pixel
                # instead of being force-resized to 640x480.
                if zoom_img.ndim >= 2 and zoom_img.shape[0] > 0:
                    sh, sw = zoom_img.shape[:2]
                else:
                    sh, sw = 480, 640
                max_w, max_h = 1280, 960
                scale = min(max_w / sw, max_h / sh, 1.0)
                # Tiny sources (< 320x240) still get upscaled to ~640x480
                # so the user can actually see them, but use NEAREST so
                # binary mask edges stay sharp.
                if sw < 320 and sh < 240:
                    scale = max(scale, min(640 / sw, 480 / sh))
                tw = max(1, int(sw * scale))
                th = max(1, int(sh * scale))
                self._put(_z["label"], zoom_img,
                          bgr=True, size=(tw, th))

        # -- Recording ----------------------------------------------------
        if self.recording and self.writers:
            vsz = (640, 480)
            rgb_md = _mask_bgr(rgb_mask, list(self._rgb_mask_bgr()))
            ir_md  = _mask_bgr(ir_mask,  list(self._ir_mask_bgr()))
            self.writers["rgb_raw"].write(cv2.resize(f_rgb,          vsz))
            self.writers["rgb_det"].write(cv2.resize(rgb_det,        vsz))
            self.writers["rgb_mask"].write(cv2.resize(rgb_md,        vsz))
            self.writers["ir_raw"].write(cv2.resize(ir_display_base, vsz))
            self.writers["ir_det"].write(cv2.resize(ir_det,          vsz))
            self.writers["ir_mask"].write(cv2.resize(ir_md,          vsz))

        self.lbl_cable.config(
            text=f"RGB cable px:{rgb_cable_px:>6}  IR cable px:{ir_cable_px:>6}  "
                 f"RGB cnts:{len(cnts)}  IR cnts:{len(ir_cnts)}  "
                 f"YOLO RGB:{yolo_rgb_n}  YOLO IR:{yolo_ir_n}")

