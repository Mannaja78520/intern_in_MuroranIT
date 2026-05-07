import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

RECORDINGS_DIR  = "/home/mannaja/intern_in_MuroranIT/intern_program/videos/realsense/recordings"
YOLO_MODEL_PATH = "/home/mannaja/intern_in_MuroranIT/intern_program/models/realsense_merged.pt"
DISPLAY_W, DISPLAY_H = 320, 240
ALL_MASKS_W, ALL_MASKS_H = 160, 120   # smaller panels in the all-masks window
ALL_MASKS_COLS = 5

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

# Additional mask-processing operations (non-morphological)
EXTRA_OPS = [
    "GaussBlur",       # Gaussian blur on mask  (KX = kernel size)
    "MedianBlur",      # Median blur             (KX = kernel size, must be odd)
    "BilateralBlur",   # Bilateral filter        (KX = diameter, KY = sigma)
    "Thresh_Binary",   # Re-threshold            (T  = threshold value)
    "Thresh_Otsu",     # Otsu auto-threshold
    "Thresh_Adaptive", # Adaptive threshold      (KX = blockSize, KY = C)
    "Invert",          # Bitwise-NOT the mask
    "FillHoles",       # Flood-fill enclosed holes
]

ALL_PROC_OPS = list(MORPH_OPS.keys()) + EXTRA_OPS

KERNEL_SHAPES = {
    "Rect":    cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross":   cv2.MORPH_CROSS,
}

# All selectable views per panel (id shown in the combobox)
VIEW_OPTIONS = [
    "rgb_raw",       "rgb_blur",      "rgb_hsv_full",  "rgb_hsv_H",    "rgb_hsv_S",
    "rgb_hsv_V",     "rgb_m1",        "rgb_m2",        "rgb_hsv_mask", "rgb_bgsub",
    "rgb_mask_pre",  "rgb_step1",     "rgb_step2",     "rgb_step3",    "rgb_step4",
    "rgb_step5",     "rgb_mask",      "rgb_post_blur", "rgb_det",
    "ir_raw",        "ir_gray",       "ir_blur",       "ir_clahe",
    "ir_bgsub",      "ir_thresh",     "ir_mask_pre",
    "ir_step1",      "ir_step2",      "ir_step3",      "ir_step4",     "ir_step5",
    "ir_mask",       "ir_post_blur",  "ir_det",
]


class VideoAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense - Cable Video Analyzer")

        self.cap_rgb = None
        self.cap_ir  = None
        self.total_frames  = 0
        self.current_frame = 0
        self.playing   = False
        self.after_id  = None
        self._updating_slider = False

        self.backSub_rgb = None
        self.backSub_ir  = None

        self.recording = False
        self.writers   = {}
        self.rec_dir   = ""

        self.loop_var = tk.BooleanVar(value=False)

        # All-masks window
        self.all_masks_win         = None
        self.all_masks_labels      = {}   # view_name → tk.Label
        self.all_masks_step_labels = {}   # "rgb_step1"… → text label widget

        # Arrow-key continuous navigation state
        self._arrow_direction    = None   # "Left" | "Right" | None
        self._arrow_loop_id      = None   # after id for the step loop
        self._arrow_release_timer = {}    # sym -> after id for release confirmation

        # Load YOLO
        self.yolo_model = None
        if _YOLO_AVAILABLE and os.path.exists(YOLO_MODEL_PATH):
            try:
                self.yolo_model = _YOLO(YOLO_MODEL_PATH)
                print(f"YOLO loaded — classes: {self.yolo_model.names}")
            except Exception as e:
                print(f"YOLO load failed: {e}")
        elif not _YOLO_AVAILABLE:
            print("ultralytics not installed — YOLO disabled")
        else:
            print(f"Model not found: {YOLO_MODEL_PATH}")

        self.rgb_pipeline = []
        self.ir_pipeline  = []

        self._setup_scroll()
        self._build_ui()
        self._bind_keys()
        self._populate_folders()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Scrollable container
    # ------------------------------------------------------------------
    def _setup_scroll(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(container, highlightthickness=0)
        v_sb = ttk.Scrollbar(container, orient="vertical",
                              command=self._canvas.yview)
        h_sb = ttk.Scrollbar(container, orient="horizontal",
                              command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=v_sb.set,
                               xscrollcommand=h_sb.set)

        h_sb.pack(side="bottom", fill="x")
        v_sb.pack(side="right",  fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # All UI widgets go inside this frame
        self.f = tk.Frame(self._canvas)
        self._canvas_win = self._canvas.create_window(
            (0, 0), window=self.f, anchor="nw")

        self.f.bind("<Configure>",
                    lambda e: self._canvas.configure(
                        scrollregion=self._canvas.bbox("all")))

        # Mouse-wheel scroll (Linux Button-4/5 + Windows/Mac MouseWheel)
        for widget in (self._canvas, self.f):
            widget.bind("<Button-4>",
                        lambda e: self._canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>",
                        lambda e: self._canvas.yview_scroll(1, "units"))
            widget.bind("<MouseWheel>",
                        lambda e: self._canvas.yview_scroll(
                            int(-1 * e.delta / 120), "units"))

    # ------------------------------------------------------------------
    # UI (all widgets parented to self.f — the scrollable inner frame)
    # ------------------------------------------------------------------
    def _build_ui(self):
        p = self.f   # shorthand for parent

        # ---- folder selector ----
        sel = tk.Frame(p)
        sel.pack(fill="x", padx=6, pady=4)
        tk.Label(sel, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_cb  = ttk.Combobox(sel, textvariable=self.folder_var,
                                       width=24, state="readonly")
        self.folder_cb.pack(side="left", padx=4)
        tk.Button(sel, text="Load", command=self._load).pack(side="left")
        self.lbl_status = tk.Label(sel,
                                   text="— select a folder and click Load —",
                                   fg="gray")
        self.lbl_status.pack(side="left", padx=8)

        # ---- 2×3 video panels ----
        vf = tk.Frame(p)
        vf.pack()
        self.panels     = {}
        self.panel_view = {}
        for key, title, r, c, default_view in [
            ("rgb_raw",  "RGB (raw)",     0, 0, "rgb_raw"),
            ("rgb_det",  "RGB Detection", 0, 1, "rgb_det"),
            ("rgb_mask", "RGB Mask",      0, 2, "rgb_mask"),
            ("ir_raw",   "IR (raw)",      1, 0, "ir_raw"),
            ("ir_det",   "IR Detection",  1, 1, "ir_det"),
            ("ir_mask",  "IR Mask",       1, 2, "ir_mask"),
        ]:
            frm = tk.Frame(vf, bd=1, relief="sunken")
            frm.grid(row=r, column=c, padx=2, pady=2)
            # view selector dropdown
            v = tk.StringVar(value=default_view)
            self.panel_view[key] = v
            ttk.Combobox(frm, textvariable=v, values=VIEW_OPTIONS,
                         width=15, state="readonly",
                         font=("Arial", 7)).pack()
            tk.Label(frm, text=title, font=("Arial", 9, "bold")).pack()
            lbl = tk.Label(frm, bg="black", width=DISPLAY_W, height=DISPLAY_H)
            lbl.pack()
            self.panels[key] = lbl

        # ---- playback controls ----
        ctrl = tk.Frame(p)
        ctrl.pack(fill="x", padx=6, pady=2)
        tk.Button(ctrl, text="|<",  width=3, command=self._go_first).pack(side="left")
        tk.Button(ctrl, text="<<",  width=3, command=self._step_back).pack(side="left")
        self.btn_play = tk.Button(ctrl, text="Play", width=7,
                                  command=self._toggle_play)
        self.btn_play.pack(side="left")
        tk.Button(ctrl, text=">>",  width=3, command=self._step_forward).pack(side="left")
        tk.Button(ctrl, text=">|",  width=3, command=self._go_last).pack(side="left")

        tk.Checkbutton(ctrl, text="Loop", variable=self.loop_var,
                       font=("Arial", 9)).pack(side="left", padx=(6, 2))

        tk.Label(ctrl, text="Speed:").pack(side="left", padx=(6, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Combobox(ctrl, textvariable=self.speed_var,
                     values=[0.25, 0.5, 1.0, 2.0, 4.0],
                     width=5).pack(side="left")

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

        self.btn_all_masks = tk.Button(ctrl, text="All Masks", width=9,
                                       command=self._toggle_all_masks_window)
        self.btn_all_masks.pack(side="left", padx=4)

        tk.Label(ctrl, text="Space=Play/Pause  ◄►=Step",
                 font=("Arial", 7), fg="gray").pack(side="left", padx=4)

        # ---- detection parameters ----
        pf = tk.LabelFrame(p, text="Detection Parameters")
        pf.pack(fill="x", padx=6, pady=4)

        self.sv = {}

        def add_param(parent, name, default, lo, hi):
            v   = tk.IntVar(value=default)
            row = tk.Frame(parent)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, width=11, anchor="w",
                     font=("Arial", 8)).pack(side="left")
            tk.Spinbox(row, textvariable=v, from_=lo, to=hi,
                       width=5, font=("Arial", 8)).pack(side="left", padx=2)
            tk.Scale(row, variable=v, from_=lo, to=hi,
                     orient="horizontal", length=100,
                     showvalue=False, font=("Arial", 7)).pack(side="left")
            self.sv[name] = v

        top_cols = tk.Frame(pf)
        top_cols.pack(fill="x")

        # ---- Col 1: RGB HSV ----
        col1 = tk.LabelFrame(top_cols, text="RGB – HSV Range",
                             font=("Arial", 8, "bold"))
        col1.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col1, text="── Hue range 1 ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "H1_low",  0,   0, 180)
        add_param(col1, "H1_high", 10,  0, 180)
        tk.Label(col1, text="── Hue range 2 ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "H2_low",  160, 0, 180)
        add_param(col1, "H2_high", 180, 0, 180)
        tk.Label(col1, text="── Saturation ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "S_min",   80,  0, 255)
        add_param(col1, "S_max",   255, 0, 255)
        tk.Label(col1, text="── Value (brightness) ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "V_min",   40,  0, 255)
        add_param(col1, "V_max",   255, 0, 255)

        # ---- Col 2: Pre-process ----
        col2 = tk.LabelFrame(top_cols, text="Pre-process",
                             font=("Arial", 8, "bold"))
        col2.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col2, text="── RGB ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "Blur_K",    0,  0, 21)
        tk.Label(col2, text="── IR ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "IR_Blur_K", 0,  0, 21)
        tk.Label(col2, text="── IR CLAHE ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.use_clahe = tk.BooleanVar(value=False)
        tk.Checkbutton(col2, text="Use CLAHE", variable=self.use_clahe,
                       font=("Arial", 8)).pack(anchor="w")
        add_param(col2, "CLAHE_Clip", 2,  1, 40)
        add_param(col2, "CLAHE_Grid", 8,  1, 16)
        tk.Label(col2, text="── Kernel shape (XY) ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        ks_row = tk.Frame(col2)
        ks_row.pack(fill="x", pady=2)
        tk.Label(ks_row, text="K Shape:", font=("Arial", 8),
                 width=11, anchor="w").pack(side="left")
        self.kernel_shape_var = tk.StringVar(value="Rect")
        ttk.Combobox(ks_row, textvariable=self.kernel_shape_var,
                     values=list(KERNEL_SHAPES.keys()),
                     width=9, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)
        tk.Label(col2, text="(X/Y steps always use Rect)",
                 font=("Arial", 7), fg="gray").pack(anchor="w")

        tk.Label(col2, text="── RGB Blur Position ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.blur_pos_rgb_var = tk.StringVar(value="Before HSV")
        for txt, val in [("Before HSV  (blur RGB image)", "Before HSV"),
                         ("After Morph (blur RGB mask)",  "After Morph")]:
            tk.Radiobutton(col2, text=txt, variable=self.blur_pos_rgb_var,
                           value=val, font=("Arial", 7)).pack(anchor="w")

        tk.Label(col2, text="── IR Blur Position ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.blur_pos_ir_var = tk.StringVar(value="Before CLAHE")
        for txt, val in [("Before CLAHE (blur IR image)", "Before CLAHE"),
                         ("After Morph  (blur IR mask)",  "After Morph")]:
            tk.Radiobutton(col2, text=txt, variable=self.blur_pos_ir_var,
                           value=val, font=("Arial", 7)).pack(anchor="w")

        # ---- Col 3: BG Sub / IR / YOLO / Filter ----
        col3 = tk.LabelFrame(top_cols, text="BG Sub / IR / YOLO / Filter",
                             font=("Arial", 8, "bold"))
        col3.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col3, text="── Background Sub ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "BG_hist", 500, 10, 2000)
        add_param(col3, "BG_var",   50,  1,  200)
        tk.Button(col3, text="Reset BG Sub", font=("Arial", 8),
                  command=self._reset_bg).pack(pady=2)
        tk.Label(col3, text="── IR Threshold ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "IR_thresh", 100, 1, 254)
        tk.Label(col3, text="── Filter ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "Min_area",   20, 1, 5000)

        self.use_bgsub    = tk.BooleanVar(value=True)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        for text, var in [
            ("Use BG Subtractor",   self.use_bgsub),
            ("Show Bounding Boxes", self.show_boxes),
            ("Mask Overlay on Raw", self.show_overlay),
        ]:
            tk.Checkbutton(col3, text=text, variable=var,
                           font=("Arial", 8)).pack(anchor="w")

        yolo_status = ("loaded" if self.yolo_model else
                       "no ultralytics" if not _YOLO_AVAILABLE else "model not found")
        tk.Label(col3, text=f"── YOLO  ({yolo_status}) ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.use_yolo = tk.BooleanVar(value=self.yolo_model is not None)
        tk.Checkbutton(col3, text="Use YOLO", variable=self.use_yolo,
                       font=("Arial", 8),
                       state="normal" if self.yolo_model else "disabled").pack(anchor="w")
        add_param(col3, "YOLO_Conf", 50, 1, 99)

        tk.Label(col3, text="── IR Display ──",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        cmap_row = tk.Frame(col3)
        cmap_row.pack(fill="x", pady=2)
        tk.Label(cmap_row, text="IR colour:", font=("Arial", 8)).pack(side="left")
        self.ir_cmap_var = tk.StringVar(value="Gray")
        ttk.Combobox(cmap_row, textvariable=self.ir_cmap_var,
                     values=list(IR_COLORMAPS.keys()),
                     width=9, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)

        # ---- Processing Pipelines (separate for RGB & IR) ----
        pipelines_outer = tk.LabelFrame(
            pf,
            text="Processing Pipelines  —  RGB & IR have independent steps",
            font=("Arial", 8, "bold"))
        pipelines_outer.pack(fill="x", padx=3, pady=4)

        tk.Label(pipelines_outer,
                 text="Morph ops → N/Dir/KX/KY   |   GaussBlur/MedianBlur → KX   "
                      "|   BilateralBlur → KX(d)/KY(σ)   |   Thresh_* → T   |   Invert/FillHoles → no params",
                 font=("Arial", 7), fg="gray").pack(anchor="w", padx=3)

        rgb_pip_frame = tk.LabelFrame(pipelines_outer, text="RGB Processing Steps",
                                      font=("Arial", 8, "bold"), fg="#4488ff")
        rgb_pip_frame.pack(fill="x", padx=3, pady=2)

        ir_pip_frame = tk.LabelFrame(pipelines_outer, text="IR Processing Steps",
                                     font=("Arial", 8, "bold"), fg="#44cc66")
        ir_pip_frame.pack(fill="x", padx=3, pady=2)

        rgb_defaults = [
            (True,  "Close",    1, "XY", 3,  10, 128),
            (False, "Open",     1, "XY", 3,   3, 128),
            (False, "Dilate",   1, "X",  15,  1, 128),
            (False, "Erode",    1, "Y",   1, 15, 128),
            (False, "Gradient", 1, "XY", 3,   3, 128),
        ]
        ir_defaults = [
            (True,  "Close",    1, "XY", 3,   3, 128),
            (False, "Open",     1, "XY", 3,   3, 128),
            (False, "Dilate",   1, "X",   7,  1, 128),
            (False, "Erode",    1, "Y",   1,  7, 128),
            (False, "Gradient", 1, "XY", 3,   3, 128),
        ]
        self._build_pipeline_steps(rgb_pip_frame, self.rgb_pipeline, rgb_defaults)
        self._build_pipeline_steps(ir_pip_frame,  self.ir_pipeline,  ir_defaults)

        # ---- status bars ----
        self.lbl_cable = tk.Label(p,
                                  text="RGB cable pixels: —   IR cable pixels: —",
                                  font=("Courier", 8), anchor="w")
        self.lbl_cable.pack(fill="x", padx=6)

        self.lbl_rec_status = tk.Label(p, text="",
                                       fg="red", font=("Arial", 8), anchor="w")
        self.lbl_rec_status.pack(fill="x", padx=6)

    # ------------------------------------------------------------------
    # Pipeline step builder (shared by RGB and IR sections)
    # ------------------------------------------------------------------
    def _build_pipeline_steps(self, parent, pipeline_list, defaults):
        for i, (en_def, op_def, n_def, dir_def, kx_def, ky_def, t_def) in enumerate(defaults):
            sf = tk.LabelFrame(parent, text=f"Step {i + 1}", font=("Arial", 8))
            sf.pack(side="left", padx=5, pady=3, fill="y")

            en_var     = tk.BooleanVar(value=en_def)
            op_var     = tk.StringVar(value=op_def)
            n_var      = tk.IntVar(value=n_def)
            dir_var    = tk.StringVar(value=dir_def)
            kx_var     = tk.IntVar(value=kx_def)
            ky_var     = tk.IntVar(value=ky_def)
            thresh_var = tk.IntVar(value=t_def)

            tk.Checkbutton(sf, text="Enable", variable=en_var,
                           font=("Arial", 8)).pack(anchor="w")

            def _row(p, lbl, wfn):
                r = tk.Frame(p)
                r.pack(fill="x", pady=1)
                tk.Label(r, text=lbl, font=("Arial", 8),
                         width=4, anchor="w").pack(side="left")
                wfn(r)

            _row(sf, "Op:",
                 lambda r, v=op_var: ttk.Combobox(
                     r, textvariable=v, values=ALL_PROC_OPS,
                     width=13, state="readonly",
                     font=("Arial", 8)).pack(side="left"))
            _row(sf, "N:",
                 lambda r, v=n_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=20,
                     width=4, font=("Arial", 8)).pack(side="left"))
            _row(sf, "Dir:",
                 lambda r, v=dir_var: ttk.Combobox(
                     r, textvariable=v, values=["X", "Y", "XY"],
                     width=4, state="readonly",
                     font=("Arial", 8)).pack(side="left"))
            _row(sf, "KX:",
                 lambda r, v=kx_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=99,
                     width=4, font=("Arial", 8)).pack(side="left"))
            _row(sf, "KY:",
                 lambda r, v=ky_var: tk.Spinbox(
                     r, textvariable=v, from_=1, to=99,
                     width=4, font=("Arial", 8)).pack(side="left"))
            _row(sf, "T:",
                 lambda r, v=thresh_var: tk.Spinbox(
                     r, textvariable=v, from_=0, to=255,
                     width=4, font=("Arial", 8)).pack(side="left"))

            pipeline_list.append(
                (en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var))

    # ------------------------------------------------------------------
    # Keyboard bindings  (arrow keys use a loop + release-confirmation
    # so holding and releasing stops navigation immediately)
    # ------------------------------------------------------------------
    def _bind_keys(self):
        # bind_all fires regardless of which widget has focus (needed because
        # the scrollable tk.Canvas grabs focus and its class bindings intercept
        # key events before they propagate to the root window binding)
        self.root.bind_all("<KeyPress>",   self._on_key)
        self.root.bind_all("<KeyRelease>", self._on_key_release)
        # Override Scale's built-in arrow handling so it doesn't move the slider
        self.slider.bind("<KeyPress>",   self._on_slider_keypress)
        self.slider.bind("<KeyRelease>", self._on_key_release)

    # ---- event handlers ----
    _NAV_KEYS = {"space", "Left", "Right"}

    def _on_key(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, tk.Scale):
            # Scale has its own instance binding — skip to avoid double invocation
            return
        if isinstance(focused, (tk.Spinbox, tk.Entry, tk.Text)):
            # For nav keys, defocus the field so future keys also work,
            # then handle the key. All other keys type normally in the field.
            if event.keysym in self._NAV_KEYS:
                self.root.focus_set()
                self._handle_key(event.keysym)
            return
        self._handle_key(event.keysym)

    def _on_slider_keypress(self, event):
        self._handle_key(event.keysym)
        return "break"   # prevent Scale from moving its own value

    def _on_key_release(self, event):
        sym = event.keysym
        if sym not in ("Left", "Right"):
            return
        # Delay confirmation by 20 ms so we can cancel it if this is
        # an X11 auto-repeat (which fires KeyRelease + KeyPress in rapid succession)
        if sym in self._arrow_release_timer:
            self.root.after_cancel(self._arrow_release_timer[sym])
        self._arrow_release_timer[sym] = self.root.after(
            20, lambda s=sym: self._confirm_arrow_release(s))

    def _handle_key(self, sym):
        if sym == "space":
            self._toggle_play()
        elif sym in ("Left", "Right"):
            # Cancel pending release (this press is part of auto-repeat)
            if sym in self._arrow_release_timer:
                self.root.after_cancel(self._arrow_release_timer.pop(sym))
            # Start the navigation loop only if not already going this direction
            if self._arrow_direction != sym:
                self._start_arrow_nav(sym)

    # ---- continuous navigation loop ----
    def _start_arrow_nav(self, direction):
        self._stop_arrow_nav()            # cancel any existing loop
        self._arrow_direction = direction
        self._arrow_do_step()             # execute first step immediately

    def _stop_arrow_nav(self):
        self._arrow_direction = None
        if self._arrow_loop_id:
            self.root.after_cancel(self._arrow_loop_id)
            self._arrow_loop_id = None

    def _confirm_arrow_release(self, sym):
        self._arrow_release_timer.pop(sym, None)
        if self._arrow_direction == sym:
            self._stop_arrow_nav()        # real release confirmed → stop now

    def _arrow_do_step(self):
        if not self._arrow_direction:
            return
        if self._arrow_direction == "Left":
            self._step_back()
        else:
            self._step_forward()
        # Re-schedule next step while key is still held
        if self._arrow_direction:
            self._arrow_loop_id = self.root.after(80, self._arrow_do_step)

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
        out_dir   = os.path.join("analysis_recordings", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
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
        h1l  = self.sv["H1_low"].get();   h1h  = self.sv["H1_high"].get()
        h2l  = self.sv["H2_low"].get();   h2h  = self.sv["H2_high"].get()
        smin = self.sv["S_min"].get();    smax = self.sv["S_max"].get()
        vmin = self.sv["V_min"].get();    vmax = self.sv["V_max"].get()
        blur_k    = self.sv["Blur_K"].get()
        ir_blur_k = self.sv["IR_Blur_K"].get()
        irt       = self.sv["IR_thresh"].get()
        mna       = self.sv["Min_area"].get()
        xy_kshape = KERNEL_SHAPES.get(self.kernel_shape_var.get(), cv2.MORPH_RECT)
        conf_thr  = self.sv["YOLO_Conf"].get() / 100.0

        # RGB pre-blur (only when position is "Before HSV")
        work_rgb = f_rgb
        if self.blur_pos_rgb_var.get() == "Before HSV" and blur_k > 0:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            work_rgb = cv2.GaussianBlur(f_rgb, (k, k), 0)

        # RGB HSV detection
        hsv = cv2.cvtColor(work_rgb, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        m2  = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        red_mask = cv2.bitwise_or(m1, m2)
        snap_rgb_hsv_mask = red_mask.copy()   # pure HSV mask before BG sub
        snap_fg_rgb = None
        if self.use_bgsub.get() and self.backSub_rgb:
            snap_fg_rgb = self.backSub_rgb.apply(f_rgb)
            red_mask = cv2.bitwise_and(red_mask, snap_fg_rgb)
        rgb_mask = red_mask.copy()
        snap_rgb_mask_pre = rgb_mask.copy()   # before morphology

        # IR pre-process
        snap_ir_gray = (cv2.cvtColor(f_ir, cv2.COLOR_BGR2GRAY)
                        if f_ir.ndim == 3 else f_ir.copy())  # raw grayscale
        ir_gray = snap_ir_gray.copy()
        # IR pre-blur (only when position is "Before CLAHE")
        if self.blur_pos_ir_var.get() == "Before CLAHE" and ir_blur_k > 0:
            k = ir_blur_k if ir_blur_k % 2 == 1 else ir_blur_k + 1
            ir_gray = cv2.GaussianBlur(ir_gray, (k, k), 0)
        snap_ir_blur = ir_gray.copy()         # after blur, before CLAHE
        if self.use_clahe.get():
            clahe   = cv2.createCLAHE(
                clipLimit=float(self.sv["CLAHE_Clip"].get()),
                tileGridSize=(self.sv["CLAHE_Grid"].get(),
                              self.sv["CLAHE_Grid"].get()))
            ir_gray = clahe.apply(ir_gray)
        snap_ir_clahe = ir_gray.copy()        # after blur + CLAHE

        # IR background sub + threshold
        if self.use_bgsub.get() and self.backSub_ir:
            ir_fg = self.backSub_ir.apply(ir_gray)
        else:
            ir_fg = ir_gray
        snap_ir_fg = ir_fg.copy()             # BG-sub foreground (or raw)
        _, ir_bin = cv2.threshold(ir_fg, irt, 255, cv2.THRESH_BINARY)
        ir_mask        = ir_bin.copy()
        snap_ir_thresh = ir_bin.copy()        # threshold result before morph
        snap_ir_mask_pre = ir_mask.copy()     # before morphology

        # Processing pipelines (separate for RGB & IR)
        def _odd(v): return v if v % 2 == 1 else v + 1
        def _fill_holes(m):
            flood  = m.copy()
            h, w   = m.shape[:2]
            ffmask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ffmask, (0, 0), 255)
            return cv2.bitwise_or(m, cv2.bitwise_not(flood))

        def _apply_op(mask, op_name, n, kx, ky, d, thresh):
            if op_name in MORPH_OPS:
                if d == "X":   sk = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
                elif d == "Y": sk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
                else:          sk = cv2.getStructuringElement(xy_kshape, (kx, ky))
                return cv2.morphologyEx(mask, MORPH_OPS[op_name], sk, iterations=n)
            elif op_name == "GaussBlur":
                return cv2.GaussianBlur(mask, (_odd(max(1, kx)),) * 2, 0)
            elif op_name == "MedianBlur":
                return cv2.medianBlur(mask, _odd(max(1, kx)))
            elif op_name == "BilateralBlur":
                return cv2.bilateralFilter(mask, kx, float(ky), float(ky))
            elif op_name == "Thresh_Binary":
                _, m = cv2.threshold(mask, max(0, min(255, thresh)), 255, cv2.THRESH_BINARY)
                return m
            elif op_name == "Thresh_Otsu":
                _, m = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return m
            elif op_name == "Thresh_Adaptive":
                return cv2.adaptiveThreshold(mask, 255,
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, _odd(max(3, kx)), ky)
            elif op_name == "Invert":
                return cv2.bitwise_not(mask)
            elif op_name == "FillHoles":
                return _fill_holes(mask)
            return mask

        # RGB pipeline — saves snap after every step
        snap_rgb_steps = []
        for en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var in self.rgb_pipeline:
            if en_var.get():
                rgb_mask = _apply_op(rgb_mask, op_var.get(),
                                     max(1, n_var.get()), max(1, kx_var.get()),
                                     max(1, ky_var.get()), dir_var.get(), thresh_var.get())
            snap_rgb_steps.append(rgb_mask.copy())

        # IR pipeline — saves snap after every step
        snap_ir_steps = []
        for en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var in self.ir_pipeline:
            if en_var.get():
                ir_mask = _apply_op(ir_mask, op_var.get(),
                                    max(1, n_var.get()), max(1, kx_var.get()),
                                    max(1, ky_var.get()), dir_var.get(), thresh_var.get())
            snap_ir_steps.append(ir_mask.copy())

        # Post-morph blur (applied to mask when position = "After Morph")
        snap_rgb_post_blur = rgb_mask.copy()
        if self.blur_pos_rgb_var.get() == "After Morph" and blur_k > 0:
            k = _odd(max(1, blur_k))
            snap_rgb_post_blur = cv2.GaussianBlur(rgb_mask, (k, k), 0)
            _, rgb_mask = cv2.threshold(snap_rgb_post_blur, 127, 255, cv2.THRESH_BINARY)

        snap_ir_post_blur = ir_mask.copy()
        if self.blur_pos_ir_var.get() == "After Morph" and ir_blur_k > 0:
            k = _odd(max(1, ir_blur_k))
            snap_ir_post_blur = cv2.GaussianBlur(ir_mask, (k, k), 0)
            _, ir_mask = cv2.threshold(snap_ir_post_blur, 127, 255, cv2.THRESH_BINARY)

        # RGB detection overlay (HSV — green boxes)
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
            ov[rgb_mask > 0] = [0, 0, 255]
            rgb_det = cv2.addWeighted(rgb_det, 0.65, ov, 0.35, 0)

        # IR false-colour display
        cmap_id = IR_COLORMAPS.get(self.ir_cmap_var.get(), None)
        ir_display_base = (cv2.applyColorMap(ir_gray, cmap_id)
                           if cmap_id is not None
                           else cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR))

        # IR detection overlay (threshold — green boxes)
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
            ov_ir[ir_mask > 0] = [0, 255, 255]
            ir_det = cv2.addWeighted(ir_det, 0.65, ov_ir, 0.35, 0)

        # YOLO detection (orange boxes, drawn on top)
        yolo_rgb_n = 0
        yolo_ir_n  = 0
        if self.use_yolo.get() and self.yolo_model:
            res_rgb = self.yolo_model(f_rgb, verbose=False, conf=conf_thr)[0]
            yolo_rgb_n = len(res_rgb.boxes)
            for box in res_rgb.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_rgb.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                cv2.rectangle(rgb_det, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(rgb_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

            ir_bgr = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            res_ir = self.yolo_model(ir_bgr, verbose=False, conf=conf_thr)[0]
            yolo_ir_n = len(res_ir.boxes)
            for box in res_ir.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_ir.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                cv2.rectangle(ir_det, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(ir_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        # Colourised mask panels
        def _mask_bgr(m, color):
            out = np.zeros((*m.shape, 3), dtype=np.uint8)
            out[m > 0] = color
            return out
        def _g2b(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        rgb_mask_disp = _mask_bgr(rgb_mask, [0, 0, 255])
        ir_mask_disp  = _mask_bgr(ir_mask,  [0, 255, 255])

        # Save to video if recording
        if self.recording and self.writers:
            vsz = (640, 480)
            self.writers["rgb_raw"].write(cv2.resize(f_rgb,           vsz))
            self.writers["rgb_det"].write(cv2.resize(rgb_det,         vsz))
            self.writers["rgb_mask"].write(cv2.resize(rgb_mask_disp,  vsz))
            self.writers["ir_raw"].write(cv2.resize(ir_display_base,  vsz))
            self.writers["ir_det"].write(cv2.resize(ir_det,           vsz))
            self.writers["ir_mask"].write(cv2.resize(ir_mask_disp,    vsz))

        # All selectable views (BGR images, same spatial size as input frames)
        _blank = np.zeros((*f_rgb.shape[:2], 3), dtype=np.uint8)

        # HSV visualisations -----------------------------------------------
        # Full-colour hue view: H=original, S=255, V=255 → pure hue colour
        _hsv_pure = np.dstack([
            hsv[:, :, 0],
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
        ])
        _rgb_hsv_full = cv2.cvtColor(_hsv_pure, cv2.COLOR_HSV2BGR)
        # Hue heatmap: scale 0-179 → 0-255, apply HSV colormap so hue maps
        # to its natural colour (red→yellow→green→cyan→blue→magenta→red)
        _h_scaled = (hsv[:, :, 0].astype(np.float32) / 179 * 255).clip(0, 255).astype(np.uint8)
        _rgb_hsv_H = cv2.applyColorMap(_h_scaled, cv2.COLORMAP_HSV)
        # Saturation and Value as JET heatmap (cold=low, hot=high)
        _rgb_hsv_S = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_JET)
        _rgb_hsv_V = cv2.applyColorMap(hsv[:, :, 2], cv2.COLORMAP_JET)
        # ------------------------------------------------------------------

        views = {
            "rgb_raw":      f_rgb,
            "rgb_blur":     work_rgb,
            "rgb_hsv_full": _rgb_hsv_full,
            "rgb_hsv_H":    _rgb_hsv_H,
            "rgb_hsv_S":    _rgb_hsv_S,
            "rgb_hsv_V":    _rgb_hsv_V,
            "rgb_m1":       _mask_bgr(m1,                  [0, 0, 255]),
            "rgb_m2":       _mask_bgr(m2,                  [0, 0, 255]),
            "rgb_hsv_mask": _mask_bgr(snap_rgb_hsv_mask,   [0, 0, 255]),
            "rgb_bgsub":    _g2b(snap_fg_rgb) if snap_fg_rgb is not None else _blank,
            "rgb_mask_pre": _mask_bgr(snap_rgb_mask_pre,   [0, 0, 255]),
            "rgb_mask":     rgb_mask_disp,
            "rgb_post_blur":_g2b(snap_rgb_post_blur),
            "rgb_det":      rgb_det,
            "ir_raw":       ir_display_base,
            "ir_gray":      _g2b(snap_ir_gray),
            "ir_blur":      _g2b(snap_ir_blur),
            "ir_clahe":     _g2b(snap_ir_clahe),
            "ir_bgsub":     _g2b(snap_ir_fg),
            "ir_thresh":    _mask_bgr(snap_ir_thresh,      [0, 255, 255]),
            "ir_mask_pre":  _mask_bgr(snap_ir_mask_pre,    [0, 255, 255]),
            "ir_mask":      ir_mask_disp,
            "ir_post_blur": _g2b(snap_ir_post_blur),
            "ir_det":       ir_det,
        }
        # Pipeline step intermediate views
        for _i in range(5):
            views[f"rgb_step{_i+1}"] = (_mask_bgr(snap_rgb_steps[_i], [0, 0, 255])
                                         if _i < len(snap_rgb_steps) else _blank)
            views[f"ir_step{_i+1}"]  = (_mask_bgr(snap_ir_steps[_i],  [0, 255, 255])
                                         if _i < len(snap_ir_steps) else _blank)

        # Update each panel using its selected view
        for key, lbl in self.panels.items():
            view_name = self.panel_view[key].get()
            img = views.get(view_name, views.get(key, _blank))
            self._put(lbl, img, bgr=True)

        # Update all-masks window if open
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            fc_sz = getattr(self, "_all_masks_img_size", (ALL_MASKS_W, ALL_MASKS_H))
            for view_name, lbl in self.all_masks_labels.items():
                self._put(lbl, views.get(view_name, _blank), bgr=True, size=fc_sz)

        # Update step node labels in All Masks flowchart
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            for _i, (_en, _op, *_) in enumerate(self.rgb_pipeline):
                _k = f"rgb_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=f"S{_i+1}: {_op.get() if _en.get() else '—'}")
            for _i, (_en, _op, *_) in enumerate(self.ir_pipeline):
                _k = f"ir_step{_i+1}"
                if _k in self.all_masks_step_labels:
                    self.all_masks_step_labels[_k].config(
                        text=f"S{_i+1}: {_op.get() if _en.get() else '—'}")

        self.lbl_cable.config(
            text=f"RGB cable px:{rgb_cable_px:>6}  IR cable px:{ir_cable_px:>6}  "
                 f"RGB cnts:{len(cnts)}  IR cnts:{len(ir_cnts)}  "
                 f"YOLO RGB:{yolo_rgb_n}  YOLO IR:{yolo_ir_n}")

    def _put(self, label, img, bgr=False, size=(DISPLAY_W, DISPLAY_H)):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = cv2.resize(img, size)
        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # ------------------------------------------------------------------
    # All-masks window
    # ------------------------------------------------------------------
    def _toggle_all_masks_window(self):
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            self.all_masks_win.destroy()
            self.all_masks_win         = None
            self.all_masks_labels      = {}
            self.all_masks_step_labels = {}
            self.btn_all_masks.config(relief="raised")
        else:
            self._open_all_masks_window()
            self.btn_all_masks.config(relief="sunken")

    def _open_all_masks_window(self):
        # ── layout constants ──────────────────────────────────────────────
        FC_W, FC_H = 128, 96          # image size inside each node
        CW  = FC_W + 4                # node frame width
        CH  = FC_H + 22               # node frame height (+ label row)
        HS  = CW  + 44                # horizontal step between nodes
        VS  = CH  + 44                # vertical step between rows
        MX, MY   = 20, 20
        SECT_H   = 30                 # section-header bar height
        SECT_GAP = 55                 # gap between RGB and IR sections
        self._all_masks_img_size = (FC_W, FC_H)

        # ── pipeline node specs: (view_id, col, row, label) ──────────────
        # "_yolo_*"  → text-only YOLO box (no image)
        # Cols 7-11 = RGB step nodes;  mask/post_blur/det shifted right to 12-13
        rgb_nodes = [
            ("rgb_raw",       0, 0, " 1  Input"),
            ("rgb_blur",      1, 0, " 2  Blur*"),
            ("rgb_hsv_full",  2, 0, " 3  HSV colour"),
            ("rgb_hsv_H",     3, 0, " 4  Hue"),
            ("rgb_hsv_S",     4, 0, " 5  Saturation"),
            ("rgb_hsv_V",     5, 0, " 6  Value"),
            ("rgb_m1",        2, 1, " 7  Hue mask 1"),
            ("rgb_m2",        3, 1, " 8  Hue mask 2"),
            ("rgb_hsv_mask",  4, 1, " 9  HSV mask"),
            ("rgb_bgsub",     5, 1, "10  BG sub"),
            ("rgb_mask_pre",  6, 1, "11  Pre-pipeline"),
            ("rgb_step1",     7, 1, "S1: ?"),
            ("rgb_step2",     8, 1, "S2: ?"),
            ("rgb_step3",     9, 1, "S3: ?"),
            ("rgb_step4",    10, 1, "S4: ?"),
            ("rgb_step5",    11, 1, "S5: ?"),
            ("rgb_mask",     12, 1, "12  Post-pipeline"),
            ("rgb_post_blur",13, 1, "13  Post-blur*"),
            ("_yolo_rgb",     0, 2, "YOLO"),
            ("rgb_det",      13, 2, "14  Detection"),
        ]
        rgb_edges = [
            ("rgb_raw",      "rgb_blur"),
            ("rgb_blur",     "rgb_hsv_full"),
            ("rgb_hsv_full", "rgb_hsv_H"),
            ("rgb_hsv_H",    "rgb_hsv_S"),
            ("rgb_hsv_S",    "rgb_hsv_V"),
            ("rgb_hsv_full", "rgb_m1"),
            ("rgb_m1",       "rgb_m2"),
            ("rgb_m2",       "rgb_hsv_mask"),
            ("rgb_hsv_mask", "rgb_bgsub"),
            ("rgb_bgsub",    "rgb_mask_pre"),
            ("rgb_mask_pre", "rgb_step1"),
            ("rgb_step1",    "rgb_step2"),
            ("rgb_step2",    "rgb_step3"),
            ("rgb_step3",    "rgb_step4"),
            ("rgb_step4",    "rgb_step5"),
            ("rgb_step5",    "rgb_mask"),
            ("rgb_mask",     "rgb_post_blur"),
            ("rgb_post_blur","rgb_det"),
            ("rgb_raw",      "_yolo_rgb"),
            ("_yolo_rgb",    "rgb_det"),
        ]

        # Cols 6-10 = IR step nodes; ir_mask/post_blur/det shifted to 11-12
        ir_nodes = [
            ("ir_raw",        0, 0, "15  IR input"),
            ("ir_gray",       1, 0, "16  Grayscale"),
            ("ir_blur",       2, 0, "17  Blur*"),
            ("ir_clahe",      3, 0, "18  CLAHE"),
            ("ir_bgsub",      3, 1, "19  BG sub"),
            ("ir_thresh",     4, 1, "20  Threshold"),
            ("ir_mask_pre",   5, 1, "21  Pre-pipeline"),
            ("ir_step1",      6, 1, "S1: ?"),
            ("ir_step2",      7, 1, "S2: ?"),
            ("ir_step3",      8, 1, "S3: ?"),
            ("ir_step4",      9, 1, "S4: ?"),
            ("ir_step5",     10, 1, "S5: ?"),
            ("ir_mask",      11, 1, "22  Post-pipeline"),
            ("ir_post_blur", 12, 1, "23  Post-blur*"),
            ("_yolo_ir",      0, 2, "YOLO"),
            ("ir_det",       12, 2, "24  Detection"),
        ]
        ir_edges = [
            ("ir_raw",       "ir_gray"),
            ("ir_gray",      "ir_blur"),
            ("ir_blur",      "ir_clahe"),
            ("ir_clahe",     "ir_bgsub"),
            ("ir_bgsub",     "ir_thresh"),
            ("ir_thresh",    "ir_mask_pre"),
            ("ir_mask_pre",  "ir_step1"),
            ("ir_step1",     "ir_step2"),
            ("ir_step2",     "ir_step3"),
            ("ir_step3",     "ir_step4"),
            ("ir_step4",     "ir_step5"),
            ("ir_step5",     "ir_mask"),
            ("ir_mask",      "ir_post_blur"),
            ("ir_post_blur", "ir_det"),
            ("ir_raw",       "_yolo_ir"),
            ("_yolo_ir",     "ir_det"),
        ]

        # ── canvas sizing ─────────────────────────────────────────────────
        max_col   = max(c for _, c, _, _ in rgb_nodes + ir_nodes)  # 8
        canvas_w  = MX + (max_col + 1) * HS + MX
        rgb_y0    = MY + SECT_H
        ir_y0     = rgb_y0 + 3 * VS + SECT_GAP + SECT_H
        canvas_h  = ir_y0 + 3 * VS + MY

        # ── window + scrollable canvas ────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Pipeline Flow — All Masks   (* = blur position selectable)")
        win.protocol("WM_DELETE_WINDOW", self._toggle_all_masks_window)
        self.all_masks_win         = win
        self.all_masks_labels      = {}
        self.all_masks_step_labels = {}

        container = tk.Frame(win)
        container.pack(fill="both", expand=True)
        fc = tk.Canvas(container,
                       width=min(canvas_w, 1400), height=min(canvas_h, 720),
                       scrollregion=(0, 0, canvas_w, canvas_h),
                       highlightthickness=0, bg="#111111")
        v_sb = ttk.Scrollbar(container, orient="vertical",   command=fc.yview)
        h_sb = ttk.Scrollbar(container, orient="horizontal", command=fc.xview)
        fc.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        h_sb.pack(side="bottom", fill="x")
        v_sb.pack(side="right",  fill="y")
        fc.pack(side="left", fill="both", expand=True)
        fc.bind("<Button-4>",   lambda _e: fc.yview_scroll(-1, "units"))
        fc.bind("<Button-5>",   lambda _e: fc.yview_scroll(1,  "units"))
        fc.bind("<MouseWheel>", lambda _e: fc.yview_scroll(int(-1*_e.delta/120), "units"))

        # ── helpers ───────────────────────────────────────────────────────
        def px(col, row, y0):    return MX + col * HS,       y0 + row * VS
        def right(c, r, y0):     x, y = px(c, r, y0); return x + CW, y + CH // 2
        def left(c, r, y0):      x, y = px(c, r, y0); return x,      y + CH // 2
        def bottom(c, r, y0):    x, y = px(c, r, y0); return x + CW // 2, y + CH
        def top(c, r, y0):       x, y = px(c, r, y0); return x + CW // 2, y

        def draw_section(nodes, edges, y0, title, hdr_bg, node_bg, arr_clr, yolo_clr):
            # header bar
            fc.create_rectangle(0, y0 - SECT_H, canvas_w, y0,
                                 fill=hdr_bg, outline="")
            fc.create_text(canvas_w // 2, y0 - SECT_H // 2,
                           text=title, font=("Arial", 11, "bold"),
                           fill="white", anchor="center")

            pos = {vid: (c, r) for vid, c, r, _ in nodes}

            # edges (drawn first, under nodes)
            for f_id, t_id in edges:
                if f_id not in pos or t_id not in pos:
                    continue
                fc_pos, fr = pos[f_id]
                tc_pos, tr = pos[t_id]
                is_yolo = f_id.startswith("_yolo") or t_id.startswith("_yolo")
                clr  = yolo_clr if is_yolo else arr_clr
                w    = 3        if is_yolo else 2
                arrs = (10, 12, 4)

                if fc_pos == tc_pos and fr < tr:
                    # straight ↓
                    fc.create_line(*bottom(fc_pos, fr, y0), *top(tc_pos, tr, y0),
                                   arrow=tk.LAST, width=w, fill=clr, arrowshape=arrs)
                elif fr == tr and fc_pos < tc_pos:
                    # straight →
                    fc.create_line(*right(fc_pos, fr, y0), *left(tc_pos, tr, y0),
                                   arrow=tk.LAST, width=w, fill=clr, arrowshape=arrs)
                else:
                    # bent: from bottom of source → down to target row → right
                    bx, by = bottom(fc_pos, fr, y0)
                    ex, ey = left(tc_pos,  tr, y0)
                    fc.create_line(bx, by, bx, ey, ex, ey,
                                   arrow=tk.LAST, width=w, fill=clr,
                                   smooth=False, arrowshape=arrs)

            # nodes
            for vid, col, row, label in nodes:
                nx, ny = px(col, row, y0)
                if vid.startswith("_yolo"):
                    fc.create_rectangle(nx, ny, nx + CW, ny + CH,
                                        fill="#2a1500", outline=yolo_clr, width=2)
                    fc.create_text(nx + CW // 2, ny + CH // 2 - 7,
                                   text="YOLO", font=("Arial", 10, "bold"),
                                   fill=yolo_clr)
                    fc.create_text(nx + CW // 2, ny + CH // 2 + 9,
                                   text=label, font=("Arial", 7), fill="#aaaaaa")
                else:
                    is_step = vid.startswith("rgb_step") or vid.startswith("ir_step")
                    step_bg = "#1a1a2e" if is_step else node_bg
                    frm = tk.Frame(fc, bg=step_bg,
                                   highlightbackground="#4466aa" if is_step else "#555555",
                                   highlightthickness=1)
                    fc.create_window(nx, ny, window=frm, anchor="nw",
                                     width=CW, height=CH)
                    txt_lbl = tk.Label(frm, text=label, font=("Arial", 7, "bold"),
                                       bg=step_bg, fg="#aaccff" if is_step else "#dddddd",
                                       anchor="w")
                    txt_lbl.pack(fill="x", padx=2)
                    if is_step:
                        self.all_masks_step_labels[vid] = txt_lbl
                    img_lbl = tk.Label(frm, bg="#000000",
                                       width=FC_W, height=FC_H)
                    img_lbl.pack(padx=1, pady=(0, 1))
                    self.all_masks_labels[vid] = img_lbl

        draw_section(rgb_nodes, rgb_edges, rgb_y0,
                     "━━━━━━━━━━━━━━━  RGB Pipeline  ━━━━━━━━━━━━━━━",
                     "#2b4f7a", "#1a2535", "#888888", "#ff8800")
        draw_section(ir_nodes, ir_edges, ir_y0,
                     "━━━━━━━━━━━━━━━━  IR Pipeline  ━━━━━━━━━━━━━━━━",
                     "#1a5c2a", "#141f14", "#888888", "#ff8800")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _toggle_play(self):
        if not self.cap_rgb: return
        if self.playing: self._stop_play()
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
            if self.loop_var.get():
                self.cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cap_ir.set( cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
            else:
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
        delay = max(1, int(33 / self.speed_var.get()))
        self.after_id = self.root.after(delay, self._play_loop)

    def _on_slider(self, val):
        if self._updating_slider or not self.cap_rgb: return
        idx = int(float(val))
        if not self.playing and idx != self.current_frame:
            self._show_frame(idx)

    def _step_back(self):
        if not self.cap_rgb: return
        if self.playing: self._stop_play()
        self._show_frame(self.current_frame - 1)

    def _step_forward(self):
        if not self.cap_rgb: return
        if self.playing: self._stop_play()
        self._show_frame(self.current_frame + 1)

    def _go_first(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(0)

    def _go_last(self):
        if self.cap_rgb: self._stop_play(); self._show_frame(self.total_frames - 1)

    def _on_close(self):
        self._stop_play()
        self._stop_arrow_nav()
        if self.recording: self._stop_record()
        if self.cap_rgb:   self.cap_rgb.release()
        if self.cap_ir:    self.cap_ir.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = VideoAnalyzer(root)
    root.mainloop()
