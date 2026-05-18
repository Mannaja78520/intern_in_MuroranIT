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

from config import (
    RECORDINGS_DIR, YOLO_MODEL_PATH, SCREENSHOTS_DIR,
    DISPLAY_W, DISPLAY_H,
    IR_COLORMAPS, MORPH_OPS, EXTRA_OPS, ALL_PROC_OPS, KERNEL_SHAPES,
    VIEW_OPTIONS,
    FC_W, FC_H,
)
from pipeline_ui_mixin    import PipelineUIMixin
from flowchart_mixin       import FlowchartMixin
from user_pipelines_mixin  import UserPipelinesMixin


class VideoAnalyzer(PipelineUIMixin, FlowchartMixin, UserPipelinesMixin):
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

        # Cached frames for live re-processing when paused
        self._last_f_rgb = None
        self._last_f_ir  = None

        self.backSub_rgb = None
        self.backSub_ir  = None

        self.recording = False
        self.writers   = {}
        self.rec_dir   = ""

        self.loop_var = tk.BooleanVar(value=False)

        # All-Masks window state
        self.all_masks_win         = None
        self.all_masks_labels      = {}   # view_name -> image Label
        self.all_masks_step_labels = {}   # "rgb_step1"... -> text Label

        # Zoom window state
        # Zoom windows — multiple at once. Keyed by view_name; each
        # entry is a dict with the Toplevel, image label, hover readout,
        # and the most-recent source ndarray for pixel-value lookup.
        self._zoom_wins = {}
        # Legacy single-window slots (still set so older code paths
        # don't blow up; they alias the most-recently opened window).
        self._zoom_win   = None
        self._zoom_label = None
        self._zoom_view  = None

        # Arrow-key continuous navigation
        self._arrow_direction     = None
        self._arrow_loop_id       = None
        self._arrow_release_timer = {}

        # Persisted user prefs (must come before YOLO so the previously
        # loaded model path can be restored at startup).
        self._prefs_path = os.path.join(os.path.expanduser("~"),
                                        ".realsense_analyzer_prefs.json")

        # YOLO
        self.yolo_model = None
        # Per-class enable map: { class_id: tk.BooleanVar }. Built
        # AFTER the model loads so we know how many classes there are.
        self.yolo_class_enabled = {}
        _saved_yolo = self._load_pref("yolo_model_path", YOLO_MODEL_PATH)
        if _YOLO_AVAILABLE and _saved_yolo and os.path.exists(_saved_yolo):
            try:
                self.yolo_model = _YOLO(_saved_yolo)
                print(f"YOLO loaded ({_saved_yolo}) - classes: "
                      f"{self.yolo_model.names}")
            except Exception as e:
                print(f"YOLO load failed: {e}")
        elif not _YOLO_AVAILABLE:
            print("ultralytics not installed - YOLO disabled")
        else:
            print(f"Model not found: {_saved_yolo}")

        self.rgb_pipeline = []
        self.ir_pipeline  = []

        # User-defined parallel pipelines. Each entry is a dict:
        #   { "name":      tk.StringVar,        # output name -> view "up_<name>"
        #     "source":    tk.StringVar,        # any view name (built-in or up_*)
        #     "color":     tk.StringVar,        # one of "red"/"yellow"/...
        #     "steps":     [10-tuple, ...],     # same shape as rgb_pipeline
        #     "frame":     tk.LabelFrame,       # outer frame in the UI
        #     "steps_row": tk.Frame }           # row that holds step cards
        self.user_pipelines = []

        self.screenshot_dir_var = tk.StringVar(
            value=self._load_pref("screenshot_dir", SCREENSHOTS_DIR))
        # Recording save dir - same UX as screenshot dir.
        self.recording_dir_var = tk.StringVar(
            value=self._load_pref("recording_dir",
                                  os.path.join(os.getcwd(),
                                               "analysis_recordings")))

        self._setup_scroll()
        self._build_ui()
        self._bind_keys()
        self._populate_folders()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Scrollable main container
    # ------------------------------------------------------------------
    def _setup_scroll(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(container, highlightthickness=0)
        v_sb = ttk.Scrollbar(container, orient="vertical",   command=self._canvas.yview)
        h_sb = ttk.Scrollbar(container, orient="horizontal", command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)

        h_sb.pack(side="bottom", fill="x")
        v_sb.pack(side="right",  fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.f = tk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self.f, anchor="nw")
        self.f.bind("<Configure>",
                    lambda e: self._canvas.configure(
                        scrollregion=self._canvas.bbox("all")))

        # Application-wide wheel binding: scroll the main config canvas
        # whenever the pointer sits over it or any of its descendants.
        # This is needed because Combobox, Spinbox, LabelFrame, etc. inside
        # the panel swallow wheel events by default.
        def _wheel_units(e):
            if hasattr(e, "delta") and e.delta:
                return int(-1 * e.delta / 120)
            return -1 if getattr(e, "num", 0) == 4 else 1

        def _on_wheel(e):
            # Only scroll if the event widget is inside the config canvas
            # (so the All-Masks window keeps its own scrolling).
            w = e.widget
            try:
                while w is not None:
                    if w is self._canvas or w is self.f:
                        self._canvas.yview_scroll(_wheel_units(e), "units")
                        return "break"
                    w = w.master
            except Exception:
                pass
            return None

        # bind_all so every descendant receives wheel events, including
        # native ttk widgets that normally steal them.
        self.root.bind_all("<MouseWheel>", _on_wheel, add="+")
        self.root.bind_all("<Button-4>",   _on_wheel, add="+")
        self.root.bind_all("<Button-5>",   _on_wheel, add="+")

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        p = self.f

        # ---- folder selector ----
        sel = tk.Frame(p)
        sel.pack(fill="x", padx=6, pady=4)
        tk.Label(sel, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_cb  = ttk.Combobox(sel, textvariable=self.folder_var,
                                       width=24, state="readonly")
        self.folder_cb.pack(side="left", padx=4)
        tk.Button(sel, text="Load", command=self._load).pack(side="left")
        self.lbl_status = tk.Label(sel, text="- select a folder and click Load -",
                                   fg="gray")
        self.lbl_status.pack(side="left", padx=8)

        # ---- 2x3 video panels ----
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
            v = tk.StringVar(value=default_view)
            self.panel_view[key] = v
            _vc = ttk.Combobox(frm, textvariable=v,
                               values=self._all_view_names(),
                               width=18, state="readonly",
                               font=("Arial", 8))
            _vc.pack()
            _vc.bind("<Button-1>",
                     lambda e, c=_vc: c.configure(values=self._all_view_names()))
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

        self.btn_rec = tk.Button(ctrl, text="* Rec", width=7,
                                 bg="darkred", fg="white",
                                 font=("Arial", 9, "bold"),
                                 command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=4)
        tk.Button(ctrl, text="Save->", width=6,
                  font=("Arial", 8),
                  command=self._pick_recording_dir).pack(side="left")
        tk.Label(ctrl, textvariable=self.recording_dir_var,
                 font=("Arial", 7), fg="#888888",
                 width=18, anchor="w").pack(side="left", padx=2)
        self.btn_all_masks = tk.Button(ctrl, text="All Masks", width=9,
                                       command=self._toggle_all_masks_window)
        self.btn_all_masks.pack(side="left", padx=4)
        tk.Button(ctrl, text="Apply F5", width=8,
                  bg="#1a3a1a", fg="white", font=("Arial", 9, "bold"),
                  command=self._apply_all).pack(side="left", padx=4)
        tk.Button(ctrl, text="[C] F12", width=7,
                  bg="#1a1a3a", fg="white", font=("Arial", 9),
                  command=self._take_screenshot).pack(side="left", padx=4)
        tk.Button(ctrl, text="Save->", width=6,
                  font=("Arial", 8),
                  command=self._pick_screenshot_dir).pack(side="left")
        tk.Label(ctrl, textvariable=self.screenshot_dir_var,
                 font=("Arial", 7), fg="#888888",
                 width=24, anchor="w").pack(side="left", padx=2)
        tk.Label(ctrl, text="Space=Play/Pause  <>=Step",
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

        col1 = tk.LabelFrame(top_cols, text="RGB - HSV Range",
                             font=("Arial", 8, "bold"))
        col1.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col1, text="-- Hue range 1 --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "H1_low",  0,   0, 180)
        add_param(col1, "H1_high", 10,  0, 180)
        tk.Label(col1, text="-- Hue range 2 --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "H2_low",  160, 0, 180)
        add_param(col1, "H2_high", 180, 0, 180)
        tk.Label(col1, text="-- Saturation --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "S_min",   80,  0, 255)
        add_param(col1, "S_max",   255, 0, 255)
        tk.Label(col1, text="-- Value (brightness) --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col1, "V_min",   40,  0, 255)
        add_param(col1, "V_max",   255, 0, 255)

        col2 = tk.LabelFrame(top_cols, text="Pre-process",
                             font=("Arial", 8, "bold"))
        col2.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col2, text="-- RGB --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "Blur_K",    0, 0, 21)
        tk.Label(col2, text="-- IR --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col2, "IR_Blur_K", 0, 0, 21)
        tk.Label(col2, text="-- IR CLAHE --", font=("Arial", 7), fg="gray").pack(anchor="w")
        self.use_clahe = tk.BooleanVar(value=False)
        tk.Checkbutton(col2, text="Use CLAHE", variable=self.use_clahe,
                       font=("Arial", 8)).pack(anchor="w")
        add_param(col2, "CLAHE_Clip", 2, 1, 40)
        add_param(col2, "CLAHE_Grid", 8, 1, 16)
        tk.Label(col2, text="-- Kernel shape (XY) --", font=("Arial", 7), fg="gray").pack(anchor="w")
        ks_row = tk.Frame(col2)
        ks_row.pack(fill="x", pady=2)
        tk.Label(ks_row, text="K Shape:", font=("Arial", 8),
                 width=11, anchor="w").pack(side="left")
        self.kernel_shape_var = tk.StringVar(value="Rect")
        ttk.Combobox(ks_row, textvariable=self.kernel_shape_var,
                     values=list(KERNEL_SHAPES.keys()),
                     width=9, state="readonly", font=("Arial", 8)).pack(side="left", padx=2)
        tk.Label(col2, text="(X/Y steps always use Rect)",
                 font=("Arial", 7), fg="gray").pack(anchor="w")

        tk.Label(col2, text="-- RGB Blur Position --",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.blur_pos_rgb_var = tk.StringVar(value="Before HSV")
        for txt, val in [("Before HSV  (blur RGB image)", "Before HSV"),
                         ("After Morph (blur RGB mask)",  "After Morph")]:
            tk.Radiobutton(col2, text=txt, variable=self.blur_pos_rgb_var,
                           value=val, font=("Arial", 7)).pack(anchor="w")

        tk.Label(col2, text="-- IR Blur Position --",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.blur_pos_ir_var = tk.StringVar(value="Before CLAHE")
        for txt, val in [("Before CLAHE (blur IR image)", "Before CLAHE"),
                         ("After Morph  (blur IR mask)",  "After Morph")]:
            tk.Radiobutton(col2, text=txt, variable=self.blur_pos_ir_var,
                           value=val, font=("Arial", 7)).pack(anchor="w")

        col3 = tk.LabelFrame(top_cols, text="BG Sub / IR / YOLO / Filter",
                             font=("Arial", 8, "bold"))
        col3.pack(side="left", padx=3, pady=3, fill="y")
        tk.Label(col3, text="-- Background Sub --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "BG_hist", 500, 10, 2000)
        add_param(col3, "BG_var",   50,  1,  200)
        tk.Button(col3, text="Reset BG Sub", font=("Arial", 8),
                  command=self._reset_bg).pack(pady=2)
        tk.Label(col3, text="-- IR Threshold --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "IR_thresh", 100, 1, 254)
        tk.Label(col3, text="-- Filter --", font=("Arial", 7), fg="gray").pack(anchor="w")
        add_param(col3, "Min_area",   20, 1, 5000)

        self.use_bgsub    = tk.BooleanVar(value=True)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        for text, var in [("Use BG Subtractor",   self.use_bgsub),
                          ("Show Bounding Boxes", self.show_boxes),
                          ("Mask Overlay on Raw", self.show_overlay)]:
            tk.Checkbutton(col3, text=text, variable=var, font=("Arial", 8)).pack(anchor="w")

        yolo_status = ("loaded" if self.yolo_model else
                       "no ultralytics" if not _YOLO_AVAILABLE else "model not found")
        tk.Label(col3, text=f"-- YOLO  ({yolo_status}) --",
                 font=("Arial", 7), fg="gray").pack(anchor="w")
        self.use_yolo = tk.BooleanVar(value=self.yolo_model is not None)
        tk.Checkbutton(col3, text="Use YOLO", variable=self.use_yolo,
                       font=("Arial", 8)).pack(anchor="w")
        add_param(col3, "YOLO_Conf", 50, 1, 99)

        # Load-model row - button on its own line, path BELOW so a
        # long path can't push the next panel off-screen.
        tk.Button(col3, text="Load YOLO model...",
                  font=("Arial", 8),
                  command=self._pick_yolo_model_file
                  ).pack(anchor="w", pady=(2, 0))
        _saved_path = (self._load_pref("yolo_model_path",
                                       YOLO_MODEL_PATH)
                       if self.yolo_model else "(none loaded)")
        self.yolo_model_path_var = tk.StringVar(value=_saved_path)
        tk.Label(col3, textvariable=self.yolo_model_path_var,
                 font=("Arial", 7), fg="#888",
                 anchor="w", justify="left",
                 wraplength=220
                 ).pack(anchor="w", padx=2)

        # YOLO class filter - rebuilt every time a new model loads.
        # Wrapped in a fixed-height scrollable canvas so a model with
        # many classes can't push other panels (IR Display, etc.)
        # off-screen or overlap them.
        self.yolo_class_panel = tk.LabelFrame(
            col3, text="YOLO classes",
            font=("Arial", 7, "bold"), fg="#cccccc")
        self.yolo_class_panel.pack(fill="x", pady=2)

        self._yolo_classes_canvas = tk.Canvas(
            self.yolo_class_panel, height=140, highlightthickness=0)
        _yolo_sb = ttk.Scrollbar(
            self.yolo_class_panel, orient="vertical",
            command=self._yolo_classes_canvas.yview)
        self._yolo_classes_canvas.configure(yscrollcommand=_yolo_sb.set)
        _yolo_sb.pack(side="right", fill="y")
        self._yolo_classes_canvas.pack(side="left", fill="both", expand=True)
        self._yolo_classes_inner = tk.Frame(self._yolo_classes_canvas)
        self._yolo_classes_canvas.create_window(
            (0, 0), window=self._yolo_classes_inner, anchor="nw")
        self._yolo_classes_inner.bind(
            "<Configure>",
            lambda e: self._yolo_classes_canvas.configure(
                scrollregion=self._yolo_classes_canvas.bbox("all")))
        self._rebuild_yolo_class_panel()

        tk.Label(col3, text="-- IR Display --", font=("Arial", 7), fg="gray").pack(anchor="w")
        cmap_row = tk.Frame(col3)
        cmap_row.pack(fill="x", pady=2)
        tk.Label(cmap_row, text="IR colour:", font=("Arial", 8)).pack(side="left")
        self.ir_cmap_var = tk.StringVar(value="Gray")
        ttk.Combobox(cmap_row, textvariable=self.ir_cmap_var,
                     values=list(IR_COLORMAPS.keys()),
                     width=9, state="readonly", font=("Arial", 8)).pack(side="left", padx=2)

        # ---- Processing Pipelines (dynamic add/remove per channel) ----
        pipelines_outer = tk.LabelFrame(
            pf,
            text="Processing Pipelines  -  RGB & IR independent  |  + Add / x Remove steps",
            font=("Arial", 8, "bold"))
        pipelines_outer.pack(fill="x", padx=3, pady=4)
        tk.Label(pipelines_outer,
                 text="Morph -> N/Dir/KX/KY  |  GaussBlur/MedianBlur -> KX  "
                      "|  BilateralBlur -> KX(d)/KY(sigma)  |  Thresh_* -> T  |  Invert/FillHoles -> no params",
                 font=("Arial", 7), fg="gray").pack(anchor="w", padx=3)

        self.rgb_pip_frame = tk.LabelFrame(pipelines_outer, text="RGB Processing Steps",
                                           font=("Arial", 8, "bold"), fg="#4488ff")
        self.rgb_pip_frame.pack(fill="x", padx=3, pady=2)

        self.ir_pip_frame = tk.LabelFrame(pipelines_outer, text="IR Processing Steps",
                                          font=("Arial", 8, "bold"), fg="#44cc66")
        self.ir_pip_frame.pack(fill="x", padx=3, pady=2)

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
        self._create_default_steps(self.rgb_pipeline, rgb_defaults)
        self._create_default_steps(self.ir_pipeline,  ir_defaults)
        self._rebuild_pipeline_ui(self.rgb_pip_frame, self.rgb_pipeline)
        self._rebuild_pipeline_ui(self.ir_pip_frame,  self.ir_pipeline)

        # ---- Combine AND / OR / XOR of any two views ----
        cf = tk.LabelFrame(pf, text="Combine  (AND / OR / XOR any two views -> 'combined')",
                           font=("Arial", 8, "bold"))
        cf.pack(fill="x", padx=3, pady=4)
        crow = tk.Frame(cf)
        crow.pack(fill="x", pady=3, padx=4)
        tk.Label(crow, text="A:", font=("Arial", 9, "bold"), fg="#4488ff").pack(side="left")
        self.combine_a_var = tk.StringVar(value="rgb_mask")
        cb_a = ttk.Combobox(crow, textvariable=self.combine_a_var,
                            values=self._all_view_names(),
                            width=20, state="readonly", font=("Arial", 8))
        cb_a.pack(side="left", padx=2)
        cb_a.bind("<Button-1>",
                  lambda e, c=cb_a: c.configure(values=self._all_view_names()))
        self.combine_op_var = tk.StringVar(value="AND")
        ttk.Combobox(crow, textvariable=self.combine_op_var, values=["AND", "OR", "XOR"],
                     width=5, state="readonly", font=("Arial", 9, "bold")).pack(side="left", padx=6)
        tk.Label(crow, text="B:", font=("Arial", 9, "bold"), fg="#44cc66").pack(side="left")
        self.combine_b_var = tk.StringVar(value="ir_mask")
        cb_b = ttk.Combobox(crow, textvariable=self.combine_b_var,
                            values=self._all_view_names(),
                            width=20, state="readonly", font=("Arial", 8))
        cb_b.pack(side="left", padx=2)
        cb_b.bind("<Button-1>",
                  lambda e, c=cb_b: c.configure(values=self._all_view_names()))
        tk.Label(crow, text="-> select 'combined' in any panel or zoom",
                 font=("Arial", 7), fg="gray").pack(side="left", padx=8)

        # ---- User Pipelines (parallel branches) -------------------------
        upf = tk.LabelFrame(p, text="User Pipelines  (parallel branches - outputs as up_<name>)",
                            font=("Arial", 8, "bold"), fg="#cc88ff")
        upf.pack(fill="x", padx=3, pady=3)
        top_row = tk.Frame(upf)
        top_row.pack(fill="x", padx=4, pady=2)
        tk.Button(top_row, text="+ New Pipeline", bg="#3b1a4a", fg="white",
                  font=("Arial", 8, "bold"),
                  command=self._on_add_user_pipeline).pack(side="left")
        tk.Label(top_row,
                 text="  Each pipeline takes any view as input, runs its own steps, "
                      "and exposes its result as up_<name>. up_* views are usable "
                      "as panel views, combine sources, and other pipelines' inputs.",
                 font=("Arial", 7), fg="gray", wraplength=900,
                 justify="left").pack(side="left")
        self.user_pipelines_host = tk.Frame(upf)
        self.user_pipelines_host.pack(fill="x", padx=4, pady=2)

        # ---- status bars ----
        self.lbl_cable = tk.Label(p, text="RGB cable pixels: -   IR cable pixels: -",
                                  font=("Courier", 8), anchor="w")
        self.lbl_cable.pack(fill="x", padx=6)
        self.lbl_rec_status = tk.Label(p, text="", fg="red",
                                       font=("Arial", 8), anchor="w")
        self.lbl_rec_status.pack(fill="x", padx=6)

    def _refresh(self, *_):
        """Re-process the cached frame immediately when paused."""
        if not self.playing and self._last_f_rgb is not None:
            self._process(self._last_f_rgb, self._last_f_ir)

    def _apply_all(self, *_):
        """Apply button / F5 hotkey: rebuild the All-Masks canvas
        (so newly added branches / step changes show up without
        manually closing it) and reprocess the current frame."""
        self._rebuild_all_masks_if_open()
        self._refresh()

    # ------------------------------------------------------------------
    # Screenshot (F12 or button)
    # ------------------------------------------------------------------
    # -- User-prefs persistence (screenshot dir etc.) ------------------
    def _load_pref(self, key, default=None):
        try:
            import json
            with open(self._prefs_path, "r") as f:
                return json.load(f).get(key, default)
        except Exception:
            return default

    def _save_pref(self, key, value):
        try:
            import json
            try:
                with open(self._prefs_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            data[key] = value
            with open(self._prefs_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _pick_screenshot_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.screenshot_dir_var.get(),
                                    title="Choose screenshot save folder")
        if d:
            self.screenshot_dir_var.set(d)
            self._save_pref("screenshot_dir", d)

    def _pick_recording_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.recording_dir_var.get(),
                                    title="Choose recording save folder")
        if d:
            self.recording_dir_var.set(d)
            self._save_pref("recording_dir", d)

    # -- YOLO model loader -----------------------------------------
    def _pick_yolo_model_file(self):
        """Open a file dialog and (re)load the YOLO model from disk."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Choose a YOLO model (.pt / .onnx)",
            filetypes=[("YOLO weights", "*.pt *.onnx *.engine"),
                       ("All files", "*")])
        if not path:
            return
        if not _YOLO_AVAILABLE:
            self.lbl_status.config(
                text="ultralytics not installed - cannot load YOLO",
                fg="red")
            return
        try:
            self.yolo_model = _YOLO(path)
            self.yolo_model_path_var.set(path)
            self._save_pref("yolo_model_path", path)
            print(f"YOLO loaded: {path}")
            print(f"  classes: {self.yolo_model.names}")
            self.use_yolo.set(True)
            # Wipe per-class state for previous model and rebuild UI.
            self.yolo_class_enabled = {}
            self._rebuild_yolo_class_panel()
            self._refresh()
        except Exception as e:
            self.lbl_status.config(text=f"YOLO load failed: {e}",
                                   fg="red")

    def _rebuild_yolo_class_panel(self):
        """Refill the per-class checkbox panel from the current model.
        Checkboxes go into the bounded scrollable inner frame so the
        panel never grows past its 140-px-high canvas viewport."""
        host = getattr(self, "_yolo_classes_inner", None)
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        if not self.yolo_model:
            tk.Label(host, text="(no model loaded)",
                     font=("Arial", 7, "italic"), fg="#888",
                     anchor="w").pack(anchor="w", padx=4)
            self.yolo_class_panel.config(text="YOLO classes")
            return
        try:
            _names = self.yolo_model.names
            if isinstance(_names, dict):
                _items = sorted(_names.items(), key=lambda x: int(x[0]))
            else:
                _items = list(enumerate(_names))
        except Exception:
            _items = []
        self.yolo_class_panel.config(text=f"YOLO classes ({len(_items)})")
        for _cid, _cname in _items:
            _cid = int(_cid)
            _v = self.yolo_class_enabled.get(_cid)
            if _v is None:
                _v = tk.BooleanVar(value=True)
                self.yolo_class_enabled[_cid] = _v
            tk.Checkbutton(host,
                           text=f"{_cid}: {_cname}",
                           variable=_v,
                           font=("Arial", 7),
                           anchor="w"
                           ).pack(anchor="w", padx=4)

    def _take_screenshot(self):
        import subprocess
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.screenshot_dir_var.get() or SCREENSHOTS_DIR
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            save_dir = SCREENSHOTS_DIR
            os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"screenshot_{ts}.png")
        self.root.update_idletasks()
        try:
            from PIL import ImageGrab
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            x1   = x + self.root.winfo_width()
            y1   = y + self.root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x1, y1)).save(fname)
            self.lbl_status.config(text=f"Screenshot: {fname}", fg="green")
            return
        except Exception:
            pass
        # Fallback: scrot (Linux)
        try:
            subprocess.Popen(["scrot", fname])
            self.lbl_status.config(text=f"Screenshot (scrot): {fname}", fg="green")
        except Exception as e:
            self.lbl_status.config(text=f"Screenshot failed: {e}", fg="red")

    # ------------------------------------------------------------------
    # Keyboard bindings
    # ------------------------------------------------------------------
    def _bind_keys(self):
        self.root.bind_all("<KeyPress>",   self._on_key)
        self.root.bind_all("<KeyRelease>", self._on_key_release)
        # Multiple keysyms cover Print Screen across Linux/X11 layouts.
        for _sym in ("<F12>", "<Print>", "<Snapshot>", "<Sys_Req>"):
            try:
                self.root.bind_all(_sym, lambda e: self._take_screenshot())
            except tk.TclError:
                pass
        # F5 = Apply (rebuild All-Masks + reprocess current frame).
        self.root.bind_all("<F5>", lambda e: self._apply_all())
        self.slider.bind("<KeyPress>",     self._on_slider_keypress)
        self.slider.bind("<KeyRelease>",   self._on_key_release)

    _NAV_KEYS = {"space", "Left", "Right"}

    def _on_key(self, event):
        # Print Screen must work even when a Combobox dropdown holds focus.
        if event.keysym in ("Print", "Snapshot", "Sys_Req", "F12"):
            self._take_screenshot()
            return
        # Combobox popdown is a transient tkinter-internal widget;
        # focus_get() can raise KeyError or other errors when it has focus.
        try:
            focused = self.root.focus_get()
        except Exception:
            return
        if isinstance(focused, tk.Scale):
            return
        if isinstance(focused, (tk.Spinbox, tk.Entry, tk.Text)):
            if event.keysym in self._NAV_KEYS:
                self.root.focus_set()
                self._handle_key(event.keysym)
            return
        self._handle_key(event.keysym)

    def _on_slider_keypress(self, event):
        self._handle_key(event.keysym)
        return "break"

    def _on_key_release(self, event):
        sym = event.keysym
        if sym not in ("Left", "Right"):
            return
        if sym in self._arrow_release_timer:
            self.root.after_cancel(self._arrow_release_timer[sym])
        self._arrow_release_timer[sym] = self.root.after(
            20, lambda s=sym: self._confirm_arrow_release(s))

    def _handle_key(self, sym):
        if sym == "space":
            self._toggle_play()
        elif sym in ("Left", "Right"):
            if sym in self._arrow_release_timer:
                self.root.after_cancel(self._arrow_release_timer.pop(sym))
            if self._arrow_direction != sym:
                self._start_arrow_nav(sym)

    def _start_arrow_nav(self, direction):
        self._stop_arrow_nav()
        self._arrow_direction = direction
        self._arrow_do_step()

    def _stop_arrow_nav(self):
        self._arrow_direction = None
        if self._arrow_loop_id:
            self.root.after_cancel(self._arrow_loop_id)
            self._arrow_loop_id = None

    def _confirm_arrow_release(self, sym):
        self._arrow_release_timer.pop(sym, None)
        if self._arrow_direction == sym:
            self._stop_arrow_nav()

    def _arrow_do_step(self):
        if not self._arrow_direction:
            return
        if self._arrow_direction == "Left":
            self._step_back()
        else:
            self._step_forward()
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
        base_dir  = (self.recording_dir_var.get()
                     or os.path.join(os.getcwd(), "analysis_recordings"))
        out_dir   = os.path.join(base_dir, timestamp)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            base_dir = os.path.join(os.getcwd(), "analysis_recordings")
            out_dir  = os.path.join(base_dir, timestamp)
            os.makedirs(out_dir, exist_ok=True)
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        fps, sz = 30.0, (640, 480)
        self.writers = {
            k: cv2.VideoWriter(os.path.join(out_dir, f"{k}.mp4"), fourcc, fps, sz)
            for k in ("rgb_raw", "rgb_det", "rgb_mask", "ir_raw", "ir_det", "ir_mask")
        }
        self.rec_dir   = out_dir
        self.recording = True
        self.btn_rec.config(text="[] Stop", bg="red")
        self.lbl_rec_status.config(text=f"* Recording -> {out_dir}/")

    def _stop_record(self):
        self.recording = False
        for w in self.writers.values(): w.release()
        self.writers = {}
        self.btn_rec.config(text="* Rec", bg="darkred")
        self.lbl_rec_status.config(text=f"Saved -> {self.rec_dir}/  (6 videos)")

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
            self._last_f_rgb = f_rgb.copy()
            self._last_f_ir  = f_ir.copy()
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

        # RGB pre-blur
        work_rgb = f_rgb
        if self.blur_pos_rgb_var.get() == "Before HSV" and blur_k > 0:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            work_rgb = cv2.GaussianBlur(f_rgb, (k, k), 0)

        # RGB HSV detection
        hsv = cv2.cvtColor(work_rgb, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (h1l, smin, vmin), (h1h, smax, vmax))
        m2  = cv2.inRange(hsv, (h2l, smin, vmin), (h2h, smax, vmax))
        red_mask = cv2.bitwise_or(m1, m2)
        snap_rgb_hsv_mask = red_mask.copy()
        snap_fg_rgb = None
        if self.use_bgsub.get() and self.backSub_rgb:
            snap_fg_rgb = self.backSub_rgb.apply(f_rgb)
            red_mask = cv2.bitwise_and(red_mask, snap_fg_rgb)
        rgb_mask = red_mask.copy()
        snap_rgb_mask_pre = rgb_mask.copy()

        # IR pre-process
        snap_ir_gray = (cv2.cvtColor(f_ir, cv2.COLOR_BGR2GRAY)
                        if f_ir.ndim == 3 else f_ir.copy())
        ir_gray = snap_ir_gray.copy()
        if self.blur_pos_ir_var.get() == "Before CLAHE" and ir_blur_k > 0:
            k = ir_blur_k if ir_blur_k % 2 == 1 else ir_blur_k + 1
            ir_gray = cv2.GaussianBlur(ir_gray, (k, k), 0)
        snap_ir_blur = ir_gray.copy()
        if self.use_clahe.get():
            clahe   = cv2.createCLAHE(
                clipLimit=float(self.sv["CLAHE_Clip"].get()),
                tileGridSize=(self.sv["CLAHE_Grid"].get(),) * 2)
            ir_gray = clahe.apply(ir_gray)
        snap_ir_clahe = ir_gray.copy()

        if self.use_bgsub.get() and self.backSub_ir:
            ir_fg = self.backSub_ir.apply(ir_gray)
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

        def _apply_op(mask, op_name, n, kx, ky, d, thresh, prev=None, inp=None):
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

        # Progressively populated map of fully-qualified view names ->
        # uint8 mask (single-channel). _resolve_src consults this so that
        # any step in any pipeline can combine with any view, including
        # cross-pipeline references like  branch1 + rgb_step4.
        view_lookup = {}

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
        _hsv_pure = np.dstack([
            hsv[:, :, 0],
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
        ])
        _early_hsv_full = cv2.cvtColor(_hsv_pure, cv2.COLOR_HSV2BGR)
        _early_h_scaled = (hsv[:, :, 0].astype(np.float32) / 179 * 255
                            ).clip(0, 255).astype(np.uint8)
        _early_hsv_H = cv2.applyColorMap(_early_h_scaled, cv2.COLORMAP_HSV)
        _early_hsv_S = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_JET)
        _early_hsv_V = cv2.applyColorMap(hsv[:, :, 2], cv2.COLORMAP_JET)
        _ir_bgr = (f_ir if f_ir.ndim == 3
                   else cv2.cvtColor(f_ir, cv2.COLOR_GRAY2BGR))

        bgr_views = {
            "rgb_raw":      f_rgb,
            "rgb_blur":     work_rgb,
            "rgb_hsv_full": _early_hsv_full,
            "rgb_hsv_H":    _early_hsv_H,
            "rgb_hsv_S":    _early_hsv_S,
            "rgb_hsv_V":    _early_hsv_V,
            "ir_raw":       _ir_bgr,
            "ir_gray":      cv2.cvtColor(snap_ir_gray, cv2.COLOR_GRAY2BGR),
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

        def _run_yolo_for(src_name):
            """Run YOLO on the named view (cached). Returns
            (boxed_bgr, mask_uint8). Empty if model missing."""
            if src_name in _step_yolo_cache:
                return _step_yolo_cache[src_name]
            base = _resolve_bgr_view(src_name).copy()
            mask = np.zeros(base.shape[:2], dtype=np.uint8)
            if self.yolo_model is None:
                _step_yolo_cache[src_name] = (base, mask)
                return base, mask
            try:
                _conf = self.sv["YOLO_Conf"].get() / 100.0
            except Exception:
                _conf = 0.5
            try:
                res = self.yolo_model(base, verbose=False, conf=_conf)[0]
                for box in res.boxes:
                    cid = int(box.cls[0])
                    v = self.yolo_class_enabled.get(cid)
                    if v is not None and not v.get():
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lbl = (f"{res.names[cid]} "
                           f"{float(box.conf[0]):.2f}")
                    cv2.rectangle(base, (x1, y1), (x2, y2),
                                  (0, 140, 255), 2)
                    cv2.putText(base, lbl, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 140, 255), 1)
                    cv2.rectangle(mask, (x1, y1), (x2, y2),
                                  255, thickness=-1)
            except Exception as _e:
                print(f"[per-step yolo error] {_e}")
            _step_yolo_cache[src_name] = (base, mask)
            return base, mask

        def _apply_yolo_mode(running_mask, step_tuple):
            """If this step's per-step YOLO is enabled with mode
            'focus' or 'subtract', AND/AND-NOT the running mask
            with the YOLO box union and return the modified mask.
            Otherwise return the running mask unchanged."""
            try:
                yst = (getattr(self, "_yolo_state", {}) or {}).get(
                    id(step_tuple))
                if yst is None or not yst["yolo_en"].get():
                    return running_mask
                mode = yst["yolo_mode"].get()
                if mode not in ("focus", "subtract"):
                    return running_mask
                src = yst["yolo_src"].get() or "rgb_raw"
                _, ymask = _run_yolo_for(src)
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

        def _paint_overlay(base_bgr, mask_gray, color=(0, 0, 255), alpha=0.55):
            """Composite a binary mask onto a BGR image as a tinted layer."""
            if base_bgr.shape[:2] != mask_gray.shape[:2]:
                base_bgr = cv2.resize(base_bgr,
                                      (mask_gray.shape[1], mask_gray.shape[0]))
            out = base_bgr.copy()
            paint = np.full_like(out, color, dtype=np.uint8)
            blended = cv2.addWeighted(out, 1 - alpha, paint, alpha, 0)
            sel = mask_gray > 0
            out[sel] = blended[sel]
            return out

        def _paint_overlay_two(base_bgr, m1_gray, c1,
                               m2_gray=None, c2=(255, 255, 0),
                               alpha=0.55):
            """Paint mask1 (in c1) and optionally mask2 (in c2) on the
            base, blending both with alpha."""
            out = _paint_overlay(base_bgr, m1_gray, c1, alpha)
            if m2_gray is not None:
                out = _paint_overlay(out, m2_gray, c2, alpha)
            return out

        # Per-step overlay-color/2nd-mask state lives on the host class
        # (PipelineUIMixin populates it when the step UI is built).
        from pipeline_ui_mixin import OVERLAY_COLORS as _OV_COLORS

        def _run_step(running, snap_steps, op_var, n_var, kx_var, ky_var,
                      dir_var, thresh_var, comb_en_var, comb_op_var, comb_src_var,
                      mask_pre, branch_sink=None, branch_key=None,
                      overlay_sink=None, overlay_state=None):
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
                             thresh_var.get(), prev=_prev, inp=mask_pre)

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

                    # -- Per-branch channel detection ----------------
                    if _type == "rgb":
                        try:
                            _bh1 = cv2.inRange(hsv[:, :, 0],
                                               _up["h1_lo"].get(),
                                               _up["h1_hi"].get())
                            _bh2 = cv2.inRange(hsv[:, :, 0],
                                               _up["h2_lo"].get(),
                                               _up["h2_hi"].get())
                            _bs  = cv2.inRange(hsv[:, :, 1],
                                               _up["s_lo"].get(),
                                               _up["s_hi"].get())
                            _bv  = cv2.inRange(hsv[:, :, 2],
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
                            _bir = cv2.inRange(snap_ir_gray,
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
                            _u_ov_st = self._overlay_state_for(_ust)
                            _running = _run_step(
                                _running, _snaps,
                                _ust[1], _ust[2], _ust[4], _ust[5],
                                _ust[3], _ust[6],
                                _ust[7], _ust[8], _ust[9],
                                _src_g,                     # mask_pre = source
                                branch_sink=up_branch_store,
                                branch_key=(_nm, _u_idx),
                                overlay_sink=up_overlay_store,
                                overlay_state=_u_ov_st,
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

        rgb_branch_imgs = {}
        snap_rgb_steps = []
        for _idx, _step_tup in enumerate(self.rgb_pipeline):
            (en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var,
             comb_en_var, comb_op_var, comb_src_var) = _step_tup
            if en_var.get():
                _ov_st = self._overlay_state_for(_step_tup)
                rgb_mask = _run_step(rgb_mask, snap_rgb_steps, op_var, n_var, kx_var,
                                     ky_var, dir_var, thresh_var, comb_en_var,
                                     comb_op_var, comb_src_var, snap_rgb_mask_pre,
                                     branch_sink=rgb_branch_imgs, branch_key=_idx,
                                     overlay_sink=rgb_overlay_imgs,
                                     overlay_state=_ov_st)
                # YOLO mode: focus / subtract filters the running mask
                # by the YOLO box union (no-op if mode == box_only).
                rgb_mask = _apply_yolo_mode(rgb_mask, _step_tup)
            snap_rgb_steps.append(rgb_mask.copy())
            view_lookup[f"rgb_step{_idx+1}"] = rgb_mask
        view_lookup["rgb_mask"] = rgb_mask

        ir_branch_imgs = {}
        snap_ir_steps = []
        for _idx, _step_tup in enumerate(self.ir_pipeline):
            (en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var,
             comb_en_var, comb_op_var, comb_src_var) = _step_tup
            if en_var.get():
                _ov_st = self._overlay_state_for(_step_tup)
                ir_mask = _run_step(ir_mask, snap_ir_steps, op_var, n_var, kx_var,
                                    ky_var, dir_var, thresh_var, comb_en_var,
                                    comb_op_var, comb_src_var, snap_ir_mask_pre,
                                    branch_sink=ir_branch_imgs, branch_key=_idx,
                                    overlay_sink=ir_overlay_imgs,
                                    overlay_state=_ov_st)
                ir_mask = _apply_yolo_mode(ir_mask, _step_tup)
            snap_ir_steps.append(ir_mask.copy())
            view_lookup[f"ir_step{_idx+1}"] = ir_mask
        view_lookup["ir_mask"] = ir_mask

        # Post-morph blur
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
            ov[rgb_mask > 0] = [0, 0, 255]
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
            ov_ir[ir_mask > 0] = [0, 255, 255]
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
            res_rgb = self.yolo_model(f_rgb, verbose=False, conf=conf_thr)[0]
            for box in res_rgb.boxes:
                _cid = int(box.cls[0])
                if not _class_enabled(_cid):
                    continue
                yolo_rgb_n += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_rgb.names[_cid]} {float(box.conf[0]):.2f}"
                cv2.rectangle(rgb_det, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(rgb_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
                # Per-class binary mask (filled box).
                m = yolo_class_masks_rgb.setdefault(
                    _cid, np.zeros(f_rgb.shape[:2], dtype=np.uint8))
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, thickness=-1)
            ir_bgr = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            res_ir = self.yolo_model(ir_bgr, verbose=False, conf=conf_thr)[0]
            for box in res_ir.boxes:
                _cid = int(box.cls[0])
                if not _class_enabled(_cid):
                    continue
                yolo_ir_n += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lbl = f"{res_ir.names[_cid]} {float(box.conf[0]):.2f}"
                cv2.rectangle(ir_det, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(ir_det, lbl, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
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

        _hsv_pure = np.dstack([
            hsv[:, :, 0],
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
            np.full(hsv.shape[:2], 255, dtype=np.uint8),
        ])
        _rgb_hsv_full = cv2.cvtColor(_hsv_pure, cv2.COLOR_HSV2BGR)
        _h_scaled = (hsv[:, :, 0].astype(np.float32) / 179 * 255).clip(0, 255).astype(np.uint8)
        _rgb_hsv_H = cv2.applyColorMap(_h_scaled, cv2.COLORMAP_HSV)
        _rgb_hsv_S = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_JET)
        _rgb_hsv_V = cv2.applyColorMap(hsv[:, :, 2], cv2.COLORMAP_JET)

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
            "rgb_mask":     _mask_bgr(rgb_mask,            [0, 0, 255]),
            "rgb_post_blur":_g2b(snap_rgb_post_blur),
            "rgb_det":      rgb_det,
            "ir_raw":       ir_display_base,
            "ir_gray":      _g2b(snap_ir_gray),
            "ir_blur":      _g2b(snap_ir_blur),
            "ir_clahe":     _g2b(snap_ir_clahe),
            "ir_bgsub":     _g2b(snap_ir_fg),
            "ir_thresh":    _mask_bgr(snap_ir_thresh,      [0, 255, 255]),
            "ir_mask_pre":  _mask_bgr(snap_ir_mask_pre,    [0, 255, 255]),
            "ir_mask":      _mask_bgr(ir_mask,             [0, 255, 255]),
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
                views[f"yolo_{_slug}_rgb"] = _mask_bgr(_mr, [0, 140, 255])
                views[f"yolo_{_slug}_ir"]  = _mask_bgr(_mi, [0, 140, 255])
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
            views["yolo_any_rgb"]      = _mask_bgr(_any_rgb, [0, 140, 255])
            views["yolo_any_ir"]       = _mask_bgr(_any_ir,  [0, 140, 255])
            view_lookup["yolo_any_rgb"] = _any_rgb
            view_lookup["yolo_any_ir"]  = _any_ir

        # Per-step YOLO add-on emit. The helpers
        # (_resolve_bgr_view, _run_yolo_for, _step_yolo_cache) were
        # defined earlier so they could be used at step time for
        # focus / subtract modes — re-using the same cache here
        # avoids duplicate inference on the same source view.

        def _emit_step_yolo(step_tuple, vid_prefix):
            """If the step's per-step YOLO is enabled, write its two
            views into the views dict (and the mask into view_lookup)."""
            try:
                yst = (getattr(self, "_yolo_state", {}) or {}).get(
                    id(step_tuple))
                if yst is None or not yst["yolo_en"].get():
                    return
                src = yst["yolo_src"].get() or "rgb_raw"
                boxed, mask = _run_yolo_for(src)
                views[f"{vid_prefix}_yolo"]      = boxed
                views[f"{vid_prefix}_yolo_mask"] = _mask_bgr(
                    mask, [0, 140, 255])
                view_lookup[f"{vid_prefix}_yolo_mask"] = mask
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

        # Dynamic step views (only as many as the current pipeline length)
        for _i, _s in enumerate(snap_rgb_steps):
            views[f"rgb_step{_i+1}"] = _mask_bgr(_s, [0, 0, 255])
        for _i, _s in enumerate(snap_ir_steps):
            views[f"ir_step{_i+1}"] = _mask_bgr(_s, [0, 255, 255])
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
        _OV_GLYPH = 64           # width (px) of each operator band - wide
                                  # enough that "+" / "->" never crowd
                                  # the adjacent sub-images.

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

        def _glyph(text, tile_h):
            band = np.zeros((tile_h, _OV_GLYPH, 3), dtype=np.uint8)
            if text == "+":
                _centre_text(band, "+", 1.6, 4)
            else:
                _centre_text(band, "->", 1.1, 3)
            return band

        def _make_composite(operands, overlay_bgr,
                            tile_w=FC_W, tile_h=FC_H):
            """Build a composite [op1][+][op2][+]...[->][overlay].
            `operands` is a list of BGR/gray images (mask1 first, then
            optional mask2, then optional base). Always >=1 item."""
            tiles = []
            for i, op in enumerate(operands):
                if i:
                    tiles.append(_glyph("+", tile_h))
                tiles.append(cv2.resize(_to_bgr(op), (tile_w, tile_h)))
            tiles.append(_glyph("->", tile_h))
            tiles.append(cv2.resize(_to_bgr(overlay_bgr),
                                    (tile_w, tile_h)))
            return np.hstack(tiles)

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

        # RGB pipeline composites
        for _i, _ov in rgb_overlay_imgs.items():
            try:
                _step_tup = self.rgb_pipeline[_i]
                _msk = snap_rgb_steps[_i] if _i < len(snap_rgb_steps) else _ov
                _ops = _build_overlay_operands(_step_tup, _msk)
                views[f"rgb_step{_i+1}_composite"] = \
                    _make_composite(_ops, _ov)
            except Exception:
                pass
        # IR pipeline composites
        for _i, _ov in ir_overlay_imgs.items():
            try:
                _step_tup = self.ir_pipeline[_i]
                _msk = snap_ir_steps[_i] if _i < len(snap_ir_steps) else _ov
                _ops = _build_overlay_operands(_step_tup, _msk)
                views[f"ir_step{_i+1}_composite"] = \
                    _make_composite(_ops, _ov)
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
        # Export user-pipeline results into the views dict for display.
        for up in self.user_pipelines:
            _nm = up["name"].get().strip() or "branch"
            _clr = _palette.get(up["color"].get(), [255, 200, 0])
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
                    views[f"up_{_nm}_step{_u_idx+1}_composite"] = \
                        _make_composite(_ops, _ov)
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
                self._put(lbl, views.get(_vn, _blank), bgr=True, size=_sz)
            # Step-label formatter shared across rgb/ir/user pipelines.
            def _step_lbl(i, en, op_v, cen, cop_v, csr_v):
                if not en.get():
                    return f"S{i+1}: -"
                if cen.get():
                    return f"S{i+1}: +{cop_v.get()}({csr_v.get()})"
                return f"S{i+1}: {op_v.get()}"

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
            for _up in self.user_pipelines:
                _nm = _up["name"].get().strip() or "branch"
                for _i, (_en, _op, _n, _d, _kx, _ky, _t,
                         _cen, _cop, _csr) in enumerate(_up["steps"]):
                    _k = f"up_{_nm}_step{_i+1}"
                    if _k in self.all_masks_step_labels:
                        self.all_masks_step_labels[_k].config(
                            text=_step_lbl(_i, _en, _op, _cen, _cop, _csr))

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
                self._put(_z["label"], zoom_img, bgr=True, size=(640, 480))

        # -- Recording ----------------------------------------------------
        if self.recording and self.writers:
            vsz = (640, 480)
            rgb_md = _mask_bgr(rgb_mask, [0, 0, 255])
            ir_md  = _mask_bgr(ir_mask,  [0, 255, 255])
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

    def _put(self, label, img, bgr=False, size=(DISPLAY_W, DISPLAY_H)):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = cv2.resize(img, size)
        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

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
            self._last_f_rgb = f_rgb.copy()
            self._last_f_ir  = f_ir.copy()
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
        self._close_zoom()
        if self.cap_rgb:   self.cap_rgb.release()
        if self.cap_ir:    self.cap_ir.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = VideoAnalyzer(root)
    root.mainloop()
