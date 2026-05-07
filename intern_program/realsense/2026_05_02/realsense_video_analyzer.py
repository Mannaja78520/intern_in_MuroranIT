import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from PIL import Image, ImageTk

RECORDINGS_DIR = "/home/mannaja/intern_in_MuroranIT/intern_program/videos/realsense/recordings"
DISPLAY_W, DISPLAY_H = 320, 240

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

MORPH_OPS = {
    "Close":    cv2.MORPH_CLOSE,
    "Open":     cv2.MORPH_OPEN,
    "Dilate":   cv2.MORPH_DILATE,
    "Erode":    cv2.MORPH_ERODE,
    "Gradient": cv2.MORPH_GRADIENT,
    "TopHat":   cv2.MORPH_TOPHAT,
    "BlackHat": cv2.MORPH_BLACKHAT,
}

KERNEL_SHAPES = {
    "Rect":    cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross":   cv2.MORPH_CROSS,
}

N_MORPH_STEPS = 5


class VideoAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense - Cable Video Analyzer")

        self.cap_rgb = None
        self.cap_ir  = None
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.after_id = None
        self._updating_slider = False

        self.backSub_rgb = None
        self.backSub_ir  = None

        self.recording = False
        self.writers   = {}
        self.rec_dir   = ""

        self.morph_pipeline = []   # filled in _build_ui

        self._build_ui()
        self._populate_folders()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- folder selector ----
        sel = tk.Frame(self.root)
        sel.pack(fill="x", padx=6, pady=4)
        tk.Label(sel, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_cb  = ttk.Combobox(sel, textvariable=self.folder_var,
                                       width=24, state="readonly")
        self.folder_cb.pack(side="left", padx=4)
        tk.Button(sel, text="Load", command=self._load).pack(side="left")
        self.lbl_status = tk.Label(sel, text="— select a folder and click Load —",
                                   fg="gray")
        self.lbl_status.pack(side="left", padx=8)

        # ---- 2×3 video panels ----
        vf = tk.Frame(self.root)
        vf.pack()
        self.panels = {}
        for key, title, r, c in [
            ("rgb_raw",  "RGB (raw)",     0, 0),
            ("rgb_det",  "RGB Detection", 0, 1),
            ("rgb_mask", "RGB Mask",      0, 2),
            ("ir_raw",   "IR (raw)",      1, 0),
            ("ir_det",   "IR Detection",  1, 1),
            ("ir_mask",  "IR Mask",       1, 2),
        ]:
            frm = tk.Frame(vf, bd=1, relief="sunken")
            frm.grid(row=r, column=c, padx=2, pady=2)
            tk.Label(frm, text=title, font=("Arial", 9, "bold")).pack()
            lbl = tk.Label(frm, bg="black", width=DISPLAY_W, height=DISPLAY_H)
            lbl.pack()
            self.panels[key] = lbl

        # ---- playback controls ----
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill="x", padx=6, pady=2)
        tk.Button(ctrl, text="|<",  width=3, command=self._go_first).pack(side="left")
        tk.Button(ctrl, text="<<",  width=3, command=self._step_back).pack(side="left")
        self.btn_play = tk.Button(ctrl, text="Play", width=7, command=self._toggle_play)
        self.btn_play.pack(side="left")
        tk.Button(ctrl, text=">>",  width=3, command=self._step_forward).pack(side="left")
        tk.Button(ctrl, text=">|",  width=3, command=self._go_last).pack(side="left")
        tk.Label(ctrl, text="Speed:").pack(side="left", padx=(8, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Combobox(ctrl, textvariable=self.speed_var,
                     values=[0.25, 0.5, 1.0, 2.0, 4.0], width=5).pack(side="left")
        self.frame_var = tk.IntVar(value=0)
        self.slider = tk.Scale(ctrl, variable=self.frame_var, from_=0, to=1,
                               orient="horizontal", length=250,
                               showvalue=False, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_pos = tk.Label(ctrl, text="0 / 0", width=12)
        self.lbl_pos.pack(side="left")
        self.btn_rec = tk.Button(ctrl, text="● Rec", width=7,
                                 bg="darkred", fg="white",
                                 font=("Arial", 9, "bold"),
                                 command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=4)

        # ---- detection parameters ----
        pf = tk.LabelFrame(self.root, text="Detection Parameters")
        pf.pack(fill="x", padx=6, pady=4)

        self.sv = {}

        def add_param(parent, name, default, lo, hi, slider_len=100):
            v   = tk.IntVar(value=default)
            row = tk.Frame(parent)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, width=11, anchor="w",
                     font=("Arial", 8)).pack(side="left")
            tk.Spinbox(row, textvariable=v, from_=lo, to=hi,
                       width=5, font=("Arial", 8)).pack(side="left", padx=2)
            tk.Scale(row, variable=v, from_=lo, to=hi,
                     orient="horizontal", length=slider_len,
                     showvalue=False, font=("Arial", 7)).pack(side="left")
            self.sv[name] = v

        top_cols = tk.Frame(pf)
        top_cols.pack(fill="x")

        # ---- Column 1: RGB HSV (full 0-180 hue range) ----
        col1 = tk.LabelFrame(top_cols, text="RGB – HSV Range",
                             font=("Arial", 8, "bold"))
        col1.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col1, text="── Hue range 1 ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col1, "H1_low",  0,   0, 180)
        add_param(col1, "H1_high", 10,  0, 180)
        tk.Label(col1, text="── Hue range 2 ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col1, "H2_low",  160, 0, 180)
        add_param(col1, "H2_high", 180, 0, 180)
        tk.Label(col1, text="── Saturation ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col1, "S_min",   80,  0, 255)
        add_param(col1, "S_max",   255, 0, 255)
        tk.Label(col1, text="── Value (brightness) ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col1, "V_min",   40,  0, 255)
        add_param(col1, "V_max",   255, 0, 255)

        # ---- Column 2: Pre-process (RGB + IR) ----
        col2 = tk.LabelFrame(top_cols, text="Pre-process",
                             font=("Arial", 8, "bold"))
        col2.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col2, text="── RGB ──", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "Blur_K",     0,  0, 21)

        tk.Label(col2, text="── IR ──", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "IR_Blur_K",  0,  0, 21)

        tk.Label(col2, text="── IR CLAHE ──", font=("Arial", 7), fg="gray").pack(anchor="w")
        self.use_clahe = tk.BooleanVar(value=False)
        tk.Checkbutton(col2, text="Use CLAHE", variable=self.use_clahe,
                       font=("Arial", 8)).pack(anchor="w")
        add_param(col2, "CLAHE_Clip", 2,  1, 40)
        add_param(col2, "CLAHE_Grid", 8,  1, 16)

        # kernel shape used when direction = XY
        tk.Label(col2, text="── Kernel shape (XY) ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        ks_row = tk.Frame(col2)
        ks_row.pack(fill="x", pady=2)
        tk.Label(ks_row, text="K Shape:", font=("Arial", 8),
                 width=11, anchor="w").pack(side="left")
        self.kernel_shape_var = tk.StringVar(value="Rect")
        ttk.Combobox(ks_row, textvariable=self.kernel_shape_var,
                     values=list(KERNEL_SHAPES.keys()),
                     width=9, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)
        tk.Label(col2, text="(X/Y steps always use Rect)", font=("Arial", 7),
                 fg="gray").pack(anchor="w")

        # ---- Column 3: BG Sub / IR / Filter ----
        col3 = tk.LabelFrame(top_cols, text="BG Sub / IR / Filter",
                             font=("Arial", 8, "bold"))
        col3.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col3, text="── Background Sub ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col3, "BG_hist", 500, 10, 2000)
        add_param(col3, "BG_var",   50,  1,  200)
        tk.Button(col3, text="Reset BG Sub", font=("Arial", 8),
                  command=self._reset_bg).pack(pady=2)
        tk.Label(col3, text="── IR Threshold ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col3, "IR_thresh", 100, 1, 254)
        tk.Label(col3, text="── Filter ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        add_param(col3, "Min_area",   20, 1, 5000)

        self.use_bgsub    = tk.BooleanVar(value=True)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        for text, var in [
            ("Use BG Subtractor",  self.use_bgsub),
            ("Show Bounding Boxes", self.show_boxes),
            ("Mask Overlay on Raw", self.show_overlay),
        ]:
            tk.Checkbutton(col3, text=text, variable=var,
                           font=("Arial", 8)).pack(anchor="w")

        tk.Label(col3, text="── IR Display ──", font=("Arial", 7),
                 fg="gray").pack(anchor="w")
        cmap_row = tk.Frame(col3)
        cmap_row.pack(fill="x", pady=2)
        tk.Label(cmap_row, text="IR colour:", font=("Arial", 8)).pack(side="left")
        self.ir_cmap_var = tk.StringVar(value="Gray")
        ttk.Combobox(cmap_row, textvariable=self.ir_cmap_var,
                     values=list(IR_COLORMAPS.keys()),
                     width=9, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)

        # ---- Morphology Pipeline (5 steps in series) ----
        mp_frame = tk.LabelFrame(pf,
                                 text="Morphology Pipeline  —  steps applied in order to both RGB & IR",
                                 font=("Arial", 8, "bold"))
        mp_frame.pack(fill="x", padx=3, pady=4)

        # (enabled, op, N, direction, KX, KY)
        # KX = kernel width  (points in X / horizontal)
        # KY = kernel height (points in Y / vertical)
        # direction: "X" → (KX,1), "Y" → (1,KY), "XY" → (KX,KY)
        defaults = [
            (True,  "Close",    1, "XY", 3,  10),
            (False, "Open",     1, "XY", 3,  3),
            (False, "Dilate",   1, "X",  15, 1),
            (False, "Erode",    1, "Y",  1,  15),
            (False, "Gradient", 1, "XY", 3,  3),
        ]
        self.morph_pipeline = []
        for i, (en_def, op_def, n_def, dir_def, kx_def, ky_def) in enumerate(defaults):
            sf = tk.LabelFrame(mp_frame, text=f"Step {i + 1}",
                               font=("Arial", 8))
            sf.pack(side="left", padx=5, pady=3, fill="y")

            en_var  = tk.BooleanVar(value=en_def)
            op_var  = tk.StringVar(value=op_def)
            n_var   = tk.IntVar(value=n_def)
            dir_var = tk.StringVar(value=dir_def)
            kx_var  = tk.IntVar(value=kx_def)
            ky_var  = tk.IntVar(value=ky_def)

            tk.Checkbutton(sf, text="Enable", variable=en_var,
                           font=("Arial", 8)).pack(anchor="w")

            def _row(parent, label, widget_fn):
                r = tk.Frame(parent)
                r.pack(fill="x", pady=1)
                tk.Label(r, text=label, font=("Arial", 8),
                         width=4, anchor="w").pack(side="left")
                widget_fn(r)

            _row(sf, "Op:",
                 lambda r, v=op_var: ttk.Combobox(
                     r, textvariable=v, values=list(MORPH_OPS.keys()),
                     width=9, state="readonly", font=("Arial", 8)).pack(side="left"))

            _row(sf, "N:",
                 lambda r, v=n_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=20,
                     width=4, font=("Arial", 8)).pack(side="left"))

            _row(sf, "Dir:",
                 lambda r, v=dir_var: ttk.Combobox(
                     r, textvariable=v, values=["X", "Y", "XY"],
                     width=4, state="readonly", font=("Arial", 8)).pack(side="left"))

            _row(sf, "KX:",
                 lambda r, v=kx_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=99,
                     width=4, font=("Arial", 8)).pack(side="left"))

            _row(sf, "KY:",
                 lambda r, v=ky_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=99,
                     width=4, font=("Arial", 8)).pack(side="left"))

            self.morph_pipeline.append((en_var, op_var, n_var, dir_var, kx_var, ky_var))

        # ---- status bars ----
        self.lbl_cable = tk.Label(
            self.root,
            text="RGB cable pixels: —   IR cable pixels: —",
            font=("Courier", 8), anchor="w")
        self.lbl_cable.pack(fill="x", padx=6)

        self.lbl_rec_status = tk.Label(self.root, text="",
                                       fg="red", font=("Arial", 8), anchor="w")
        self.lbl_rec_status.pack(fill="x", padx=6)

    # ------------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------------
    def _populate_folders(self):
        if not os.path.exists(RECORDINGS_DIR):
            self.lbl_status.config(
                text=f"Recordings dir not found: {RECORDINGS_DIR}", fg="red")
            return
        folders = sorted(
            [d for d in os.listdir(RECORDINGS_DIR)
             if os.path.isdir(os.path.join(RECORDINGS_DIR, d))],
            reverse=True)
        self.folder_cb["values"] = folders
        if folders:
            self.folder_cb.set(folders[0])

    def _load(self):
        name = self.folder_var.get()
        if not name:
            return
        path  = os.path.join(RECORDINGS_DIR, name)
        rgb_p = os.path.join(path, "rgb.mp4")
        ir_p  = os.path.join(path, "ir.mp4")
        for fp, lbl in [(rgb_p, "rgb.mp4"), (ir_p, "ir.mp4")]:
            if not os.path.exists(fp):
                self.lbl_status.config(text=f"{lbl} not found in {name}!", fg="red")
                return
        self._stop_play()
        if self.cap_rgb: self.cap_rgb.release()
        if self.cap_ir:  self.cap_ir.release()
        self.cap_rgb = cv2.VideoCapture(rgb_p)
        self.cap_ir  = cv2.VideoCapture(ir_p)
        self.total_frames  = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self._updating_slider = True
        self.slider.config(to=max(1, self.total_frames - 1))
        self.frame_var.set(0)
        self._updating_slider = False
        self.lbl_pos.config(text=f"0 / {self.total_frames - 1}")
        self.lbl_status.config(
            text=f"Loaded: {name}  ({self.total_frames} frames)", fg="green")
        self._reset_bg()
        self._show_frame(0)

    # ------------------------------------------------------------------
    # Background subtractor
    # ------------------------------------------------------------------
    def _reset_bg(self):
        hist = self.sv["BG_hist"].get()
        var  = self.sv["BG_var"].get()
        self.backSub_rgb = cv2.createBackgroundSubtractorMOG2(
            history=hist, varThreshold=var, detectShadows=False)
        self.backSub_ir  = cv2.createBackgroundSubtractorMOG2(
            history=hist, varThreshold=var, detectShadows=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _toggle_record(self):
        if self.recording: self._stop_record()
        else:              self._start_record()

    def _start_record(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("analysis_recordings", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps, sz = 30.0, (640, 480)
        self.writers = {
            "rgb_raw":  cv2.VideoWriter(os.path.join(out_dir, "rgb_raw.mp4"),  fourcc, fps, sz),
            "rgb_det":  cv2.VideoWriter(os.path.join(out_dir, "rgb_det.mp4"),  fourcc, fps, sz),
            "rgb_mask": cv2.VideoWriter(os.path.join(out_dir, "rgb_mask.mp4"), fourcc, fps, sz),
            "ir_raw":   cv2.VideoWriter(os.path.join(out_dir, "ir_raw.mp4"),   fourcc, fps, sz),
            "ir_det":   cv2.VideoWriter(os.path.join(out_dir, "ir_det.mp4"),   fourcc, fps, sz),
            "ir_mask":  cv2.VideoWriter(os.path.join(out_dir, "ir_mask.mp4"),  fourcc, fps, sz),
        }
        self.rec_dir   = out_dir
        self.recording = True
        self.btn_rec.config(text="■ Stop", bg="red")
        self.lbl_rec_status.config(text=f"● Recording → {out_dir}/")
        print(f"Recording started: {out_dir}/")

    def _stop_record(self):
        self.recording = False
        for w in self.writers.values(): w.release()
        self.writers = {}
        self.btn_rec.config(text="● Rec", bg="darkred")
        self.lbl_rec_status.config(text=f"Saved → {self.rec_dir}/  (6 videos)")
        print(f"Recording saved: {self.rec_dir}/")

    # ------------------------------------------------------------------
    # Frame display & processing
    # ------------------------------------------------------------------
    def _show_frame(self, idx):
        if not self.cap_rgb: return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.cap_ir.set( cv2.CAP_PROP_POS_FRAMES, idx)
        ret_rgb, f_rgb = self.cap_rgb.read()
        ret_ir,  f_ir  = self.cap_ir.read()
        if ret_rgb and ret_ir:
            self.current_frame = idx
            self._updating_slider = True
            self.frame_var.set(idx)
            self.lbl_pos.config(text=f"{idx} / {self.total_frames - 1}")
            self._updating_slider = False
            self._process(f_rgb, f_ir)

    def _process(self, f_rgb, f_ir):
        # ---- read parameters ----
        h1l  = self.sv["H1_low"].get();   h1h  = self.sv["H1_high"].get()
        h2l  = self.sv["H2_low"].get();   h2h  = self.sv["H2_high"].get()
        smin = self.sv["S_min"].get();    smax = self.sv["S_max"].get()
        vmin = self.sv["V_min"].get();    vmax = self.sv["V_max"].get()
        blur_k    = self.sv["Blur_K"].get()
        ir_blur_k = self.sv["IR_Blur_K"].get()
        irt = self.sv["IR_thresh"].get()
        mna = self.sv["Min_area"].get()
        xy_kshape = KERNEL_SHAPES.get(self.kernel_shape_var.get(), cv2.MORPH_RECT)

        # ---- RGB pre-blur ----
        work_rgb = f_rgb
        if blur_k > 0:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            work_rgb = cv2.GaussianBlur(f_rgb, (k, k), 0)

        # ---- RGB HSV red detection ----
        hsv = cv2.cvtColor(work_rgb, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        m2  = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        red_mask = cv2.bitwise_or(m1, m2)
        if self.use_bgsub.get() and self.backSub_rgb:
            fg = self.backSub_rgb.apply(f_rgb)
            red_mask = cv2.bitwise_and(red_mask, fg)
        rgb_mask = red_mask.copy()

        # ---- IR pre-process ----
        ir_gray = cv2.cvtColor(f_ir, cv2.COLOR_BGR2GRAY) if f_ir.ndim == 3 else f_ir.copy()

        if ir_blur_k > 0:
            k = ir_blur_k if ir_blur_k % 2 == 1 else ir_blur_k + 1
            ir_gray = cv2.GaussianBlur(ir_gray, (k, k), 0)

        if self.use_clahe.get():
            clahe   = cv2.createCLAHE(
                clipLimit=float(self.sv["CLAHE_Clip"].get()),
                tileGridSize=(self.sv["CLAHE_Grid"].get(),
                              self.sv["CLAHE_Grid"].get()))
            ir_gray = clahe.apply(ir_gray)

        # ---- IR background sub + threshold ----
        if self.use_bgsub.get() and self.backSub_ir:
            ir_fg = self.backSub_ir.apply(ir_gray)
        else:
            ir_fg = ir_gray
        _, ir_bin = cv2.threshold(ir_fg, irt, 255, cv2.THRESH_BINARY)
        ir_mask = ir_bin.copy()

        # ---- Morphology pipeline (same steps applied to both RGB and IR) ----
        for en_var, op_var, n_var, dir_var, kx_var, ky_var in self.morph_pipeline:
            if en_var.get():
                op  = MORPH_OPS.get(op_var.get(), cv2.MORPH_CLOSE)
                n   = max(1, n_var.get())
                kx  = max(1, kx_var.get())
                ky  = max(1, ky_var.get())
                direction = dir_var.get()
                # build kernel: X-only = wide line, Y-only = tall line, XY = 2D shape
                if direction == "X":
                    step_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
                elif direction == "Y":
                    step_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
                else:
                    step_k = cv2.getStructuringElement(xy_kshape, (kx, ky))
                rgb_mask = cv2.morphologyEx(rgb_mask, op, step_k, iterations=n)
                ir_mask  = cv2.morphologyEx(ir_mask,  op, step_k, iterations=n)

        # ---- RGB detection overlay ----
        rgb_det = f_rgb.copy()
        cnts, _ = cv2.findContours(rgb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_cable_px = int(np.sum(rgb_mask > 0))
        if self.show_boxes.get():
            for c in cnts:
                if cv2.contourArea(c) > mna:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(rgb_det, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(rgb_det, "cable", (x, max(0, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if self.show_overlay.get():
            ov = np.zeros_like(rgb_det)
            ov[rgb_mask > 0] = [0, 0, 255]
            rgb_det = cv2.addWeighted(rgb_det, 0.65, ov, 0.35, 0)

        # ---- IR false-colour display ----
        cmap_id = IR_COLORMAPS.get(self.ir_cmap_var.get(), None)
        ir_display_base = (cv2.applyColorMap(ir_gray, cmap_id)
                           if cmap_id is not None
                           else cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR))

        # ---- IR detection overlay ----
        ir_det = ir_display_base.copy()
        ir_cnts, _ = cv2.findContours(ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ir_cable_px = int(np.sum(ir_mask > 0))
        if self.show_boxes.get():
            for c in ir_cnts:
                if cv2.contourArea(c) > mna:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(ir_det, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(ir_det, "cable", (x, max(0, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if self.show_overlay.get():
            ov_ir = np.zeros_like(ir_det)
            ov_ir[ir_mask > 0] = [0, 255, 255]
            ir_det = cv2.addWeighted(ir_det, 0.65, ov_ir, 0.35, 0)

        # ---- colourised mask panels ----
        rgb_mask_disp             = np.zeros((*rgb_mask.shape, 3), dtype=np.uint8)
        rgb_mask_disp[rgb_mask > 0] = [0, 0, 255]
        ir_mask_disp              = np.zeros((*ir_mask.shape,  3), dtype=np.uint8)
        ir_mask_disp[ir_mask > 0]   = [0, 255, 255]

        # ---- save to video ----
        if self.recording and self.writers:
            vid_sz = (640, 480)
            self.writers["rgb_raw"].write(cv2.resize(f_rgb,           vid_sz))
            self.writers["rgb_det"].write(cv2.resize(rgb_det,         vid_sz))
            self.writers["rgb_mask"].write(cv2.resize(rgb_mask_disp,  vid_sz))
            self.writers["ir_raw"].write(cv2.resize(ir_display_base,  vid_sz))
            self.writers["ir_det"].write(cv2.resize(ir_det,           vid_sz))
            self.writers["ir_mask"].write(cv2.resize(ir_mask_disp,    vid_sz))

        # ---- update display ----
        self._put(self.panels["rgb_raw"],  f_rgb,           bgr=True)
        self._put(self.panels["rgb_det"],  rgb_det,         bgr=True)
        self._put(self.panels["rgb_mask"], rgb_mask_disp,   bgr=True)
        self._put(self.panels["ir_raw"],   ir_display_base, bgr=True)
        self._put(self.panels["ir_det"],   ir_det,          bgr=True)
        self._put(self.panels["ir_mask"],  ir_mask_disp,    bgr=True)

        self.lbl_cable.config(
            text=f"RGB cable pixels: {rgb_cable_px:>6}   "
                 f"IR cable pixels: {ir_cable_px:>6}   "
                 f"RGB contours: {len(cnts)}   IR contours: {len(ir_cnts)}")

    def _put(self, label, img, bgr=False):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _toggle_play(self):
        if not self.cap_rgb: return
        if self.playing:
            self._stop_play()
        else:
            self.playing = True
            self.btn_play.config(text="Pause")
            self._play_loop()

    def _stop_play(self):
        self.playing = False
        self.btn_play.config(text="Play")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _play_loop(self):
        if not self.playing or not self.cap_rgb: return
        if self.current_frame >= self.total_frames - 1:
            self._stop_play(); return
        ret_rgb, f_rgb = self.cap_rgb.read()
        ret_ir,  f_ir  = self.cap_ir.read()
        if ret_rgb and ret_ir:
            self.current_frame += 1
            self._updating_slider = True
            self.frame_var.set(self.current_frame)
            self.lbl_pos.config(
                text=f"{self.current_frame} / {self.total_frames - 1}")
            self._updating_slider = False
            self._process(f_rgb, f_ir)
        speed = self.speed_var.get()
        delay = max(1, int(33 / speed))
        self.after_id = self.root.after(delay, self._play_loop)

    def _on_slider(self, val):
        if self._updating_slider or not self.cap_rgb: return
        idx = int(float(val))
        if not self.playing and idx != self.current_frame:
            self._show_frame(idx)

    def _step_back(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(self.current_frame - 1)

    def _step_forward(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(self.current_frame + 1)

    def _go_first(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(0)

    def _go_last(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(self.total_frames - 1)

    def _on_close(self):
        self._stop_play()
        if self.recording: self._stop_record()
        if self.cap_rgb:   self.cap_rgb.release()
        if self.cap_ir:    self.cap_ir.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = VideoAnalyzer(root)
    root.mainloop()
