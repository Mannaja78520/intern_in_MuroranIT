import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from PIL import Image, ImageTk

# Multi-core CPU for OpenCV. setNumThreads(-1) -> use all available
# logical cores (OpenCV's TBB / OpenMP parallel sections — most
# filters / morphology / cvtColor / inRange benefit).
try:
    _cpu_n = os.cpu_count() or 1
    cv2.setNumThreads(max(1, _cpu_n))
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    # OpenCL backend (T-API) — if a GPU/iGPU advertises OpenCL,
    # cv2.UMat-backed ops will run on it. Most non-UMat code paths
    # are unaffected, so this is a safe opt-in.
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass
except Exception:
    pass

# GPU detection for YOLO. ultralytics auto-picks CUDA when available,
# but we surface a flag here so we can explicitly pass device='0' /
# 'cpu' to inference calls.
_TORCH_DEVICE = "cpu"
try:
    import torch  # noqa: F401 — only used for cuda probe
    if torch.cuda.is_available():
        _TORCH_DEVICE = "0"  # first CUDA device; ultralytics accepts str
except Exception:
    pass

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
    is_pre_morph_op,
)
from pipeline_ui_mixin    import PipelineUIMixin
from flowchart_mixin       import FlowchartMixin
from user_pipelines_mixin  import UserPipelinesMixin


# -- UI theme (applied via option_add + ttk.Style at app start) ---------
# Two selectable themes: "dark" (default) and "light". Every theme
# colour used anywhere in the app is registered in THEME_COLORS under
# a canonical name with a value per theme. _apply_ui_theme picks the
# palette; _retheme_tree (live switch) walks existing widgets and
# remaps any colour it recognises to the other theme's value.
UI_FONT_FAMILY = "DejaVu Sans"  # friendly, well-hinted on Linux

# canonical name -> {"dark": hex, "light": hex}. Pure-black hexes
# (#000000 / #111111 — thumbnail placeholders) are deliberately left
# OUT so they stay black under both themes.
THEME_COLORS = {
    # -- backgrounds --
    "bg_main":    {"dark": "#1f1f2b", "light": "#ececf0"},
    "bg_panel":   {"dark": "#28283a", "light": "#dcdce4"},
    "bg_input":   {"dark": "#33334a", "light": "#ffffff"},
    "bg_near":    {"dark": "#1a1a2a", "light": "#e6e6ec"},
    "bg_canvas":  {"dark": "#111111", "light": "#e8e8ec"},
    # All-Masks flowchart node-frame fills — light variants are pale
    # tints so the (rethemed dark) node text stays readable on them.
    "node_rgb":   {"dark": "#1a2535", "light": "#dde6f0"},
    "node_ir":    {"dark": "#141f14", "light": "#dfeadf"},
    "node_branch":{"dark": "#1f1130", "light": "#e8def2"},
    "node_comb":  {"dark": "#1a0a2e", "light": "#e2d4f0"},
    "node_step":  {"dark": "#1a1a2e", "light": "#dfdfea"},
    # Action-button fills — kept a medium tone (not pale) under the
    # light theme so the white button text stays readable.
    "btn_green":  {"dark": "#225522", "light": "#3a7a3a"},
    "btn_green2": {"dark": "#1a3a1a", "light": "#2f6630"},
    "btn_green3": {"dark": "#223322", "light": "#356b3a"},
    "btn_blue":   {"dark": "#1a1a3a", "light": "#3a3a8a"},
    "btn_purple": {"dark": "#3b1a4a", "light": "#6a3a82"},
    # -- foregrounds --
    "fg_main":    {"dark": "#f0f0f0", "light": "#1a1a1a"},
    "fg_muted":   {"dark": "#c8c8c8", "light": "#555555"},
    "fg_dim":     {"dark": "#cccccc", "light": "#565656"},
    "fg_dim2":    {"dark": "#dddddd", "light": "#4a4a4a"},
    "fg_dim3":    {"dark": "#b0b0b0", "light": "#606060"},
    "fg_sage":    {"dark": "#aaccaa", "light": "#3a6a3a"},
    "fg_yellow":  {"dark": "#ffdd66", "light": "#8a6a00"},
    "fg_accent":  {"dark": "#aaccff", "light": "#1a4a8a"},
    "fg_blue2":   {"dark": "#88aaff", "light": "#2a4faf"},
    "fg_blue3":   {"dark": "#4488ff", "light": "#1a55cc"},
    "fg_blue4":   {"dark": "#3366aa", "light": "#24507f"},
    "fg_blue5":   {"dark": "#88bbff", "light": "#1f5fb0"},
    "fg_orange3": {"dark": "#ff8800", "light": "#b35e00"},
    "fg_green":   {"dark": "#88dd88", "light": "#2c8a3c"},
    "fg_green2":  {"dark": "#88ff88", "light": "#2f9440"},
    "fg_green3":  {"dark": "#44cc66", "light": "#1f9a44"},
    "fg_orange":  {"dark": "#ffaa66", "light": "#b35e00"},
    "fg_purple":  {"dark": "#cc88ff", "light": "#7a34b0"},
    "fg_purple2": {"dark": "#bb66ff", "light": "#6d28a8"},
    "fg_purple3": {"dark": "#aa88cc", "light": "#6a4790"},
    "fg_purple4": {"dark": "#e0c8ff", "light": "#5a3a86"},
    "fg_red":     {"dark": "#ff4444", "light": "#cc1111"},
}

# Flat reverse index: any known hex (either theme) -> canonical name.
_HEX_TO_CANON = {}
for _cn, _cv in THEME_COLORS.items():
    for _tn in ("dark", "light"):
        _HEX_TO_CANON[_cv[_tn].lower()] = _cn

# Module-level active palette — reassigned by _apply_ui_theme().
UI_BG        = THEME_COLORS["bg_main"]["dark"]
UI_BG_PANEL  = THEME_COLORS["bg_panel"]["dark"]
UI_BG_INPUT  = THEME_COLORS["bg_input"]["dark"]
UI_FG        = THEME_COLORS["fg_main"]["dark"]
UI_FG_MUTED  = THEME_COLORS["fg_muted"]["dark"]
UI_FG_ACCENT = THEME_COLORS["fg_accent"]["dark"]


def _apply_ui_theme(root, theme="dark"):
    """Install the chosen theme ("dark" / "light") via option_add +
    ttk.Style so every tk and ttk widget renders with consistent,
    high-contrast colours. Called from VideoAnalyzer.__init__ and
    again on a live theme switch."""
    global UI_BG, UI_BG_PANEL, UI_BG_INPUT, UI_FG, UI_FG_MUTED, UI_FG_ACCENT
    if theme not in ("dark", "light"):
        theme = "dark"
    UI_BG        = THEME_COLORS["bg_main"][theme]
    UI_BG_PANEL  = THEME_COLORS["bg_panel"][theme]
    UI_BG_INPUT  = THEME_COLORS["bg_input"][theme]
    UI_FG        = THEME_COLORS["fg_main"][theme]
    UI_FG_MUTED  = THEME_COLORS["fg_muted"][theme]
    UI_FG_ACCENT = THEME_COLORS["fg_accent"][theme]
    try:
        root.configure(bg=UI_BG)
    except Exception:
        pass
    # Pin Tk's named default fonts to a friendly sans-serif so any
    # widget that doesn't pass an explicit font= still uses the same
    # readable face as the explicit "DejaVu Sans" call sites elsewhere.
    try:
        from tkinter import font as _tkfont
        for _fname, _sz in (("TkDefaultFont",      9),
                            ("TkTextFont",         9),
                            ("TkMenuFont",         9),
                            ("TkHeadingFont",      9),
                            ("TkCaptionFont",      9),
                            ("TkSmallCaptionFont", 8),
                            ("TkIconFont",         9),
                            ("TkTooltipFont",      9)):
            try:
                _f = _tkfont.nametofont(_fname)
                _f.configure(family=UI_FONT_FAMILY, size=_sz)
            except Exception:
                pass
    except Exception:
        pass
    # tk widgets — option_add fans out to every widget that
    # doesn't explicitly override these attributes.
    _opts = [
        ("*Background",                UI_BG),
        ("*Foreground",                UI_FG),
        ("*Frame.Background",          UI_BG),
        ("*Label.Background",          UI_BG),
        ("*Label.Foreground",          UI_FG),
        ("*LabelFrame.Background",     UI_BG),
        ("*LabelFrame.Foreground",     UI_FG_ACCENT),
        # Drop the default 3-D groove border on every LabelFrame so
        # nested panels don't draw a stack of competing edges. Each
        # panel is identified by its coloured title text alone — much
        # cleaner against the dark background. The Tk widget class is
        # "Labelframe" (single lowercase f) — option_add patterns are
        # class-sensitive, so we list both spellings.
        ("*LabelFrame.borderWidth",    "0"),
        ("*LabelFrame.relief",         "flat"),
        ("*Labelframe.background",     UI_BG),
        ("*Labelframe.foreground",     UI_FG_ACCENT),
        ("*Labelframe.borderWidth",    "0"),
        ("*Labelframe.relief",         "flat"),
        ("*Checkbutton.Background",    UI_BG),
        ("*Checkbutton.Foreground",    UI_FG),
        ("*Checkbutton.selectColor",   UI_BG_INPUT),
        ("*Checkbutton.activeBackground", UI_BG),
        ("*Checkbutton.activeForeground", UI_FG),
        ("*Radiobutton.Background",    UI_BG),
        ("*Radiobutton.Foreground",    UI_FG),
        ("*Radiobutton.selectColor",   UI_BG_INPUT),
        ("*Radiobutton.activeBackground", UI_BG),
        ("*Radiobutton.activeForeground", UI_FG),
        ("*Button.Background",         UI_BG_PANEL),
        ("*Button.Foreground",         UI_FG),
        ("*Button.activeBackground",   UI_BG_INPUT),
        ("*Button.activeForeground",   UI_FG),
        ("*Entry.Background",          UI_BG_INPUT),
        ("*Entry.Foreground",          UI_FG),
        ("*Entry.insertBackground",    UI_FG),
        # Flat inputs — the tinted field colour (bg=UI_BG_INPUT) is
        # the only cue that it's editable. No 3-D sunken edges inside
        # step / PM cards (or anywhere else in the app).
        ("*Entry.borderWidth",         "0"),
        ("*Entry.relief",              "flat"),
        ("*Entry.highlightThickness",  "0"),
        ("*Spinbox.Background",        UI_BG_INPUT),
        ("*Spinbox.Foreground",        UI_FG),
        ("*Spinbox.insertBackground",  UI_FG),
        ("*Spinbox.borderWidth",       "0"),
        ("*Spinbox.relief",            "flat"),
        ("*Spinbox.highlightThickness", "0"),
        ("*Spinbox.buttonBackground",  UI_BG_PANEL),
        ("*Scale.Background",          UI_BG),
        ("*Scale.Foreground",          UI_FG),
        ("*Scale.troughColor",         UI_BG_INPUT),
        ("*Scale.activeBackground",    UI_FG_ACCENT),
        # Sliders keep a visible frame + raised handle so the
        # draggable affordance is obvious; everything else inside
        # step / PM cards stays flat.
        ("*Scale.borderWidth",         "1"),
        ("*Scale.relief",              "ridge"),
        ("*Scale.highlightThickness",  "0"),
        ("*Scale.sliderRelief",        "raised"),
        # Buttons inside step cards (^ v x reorder, + Add Step, etc.)
        # already use bg=#223322/#3b1a4a etc. — strip the default
        # raised relief so they sit flush instead of bulging out.
        ("*Button.borderWidth",        "1"),
        ("*Button.relief",             "flat"),
        ("*Button.highlightThickness", "0"),
        ("*Checkbutton.borderWidth",   "0"),
        ("*Checkbutton.highlightThickness", "0"),
        ("*Radiobutton.borderWidth",   "0"),
        ("*Radiobutton.highlightThickness", "0"),
        ("*Canvas.Background",         UI_BG),
        ("*Toplevel.Background",       UI_BG),
        ("*Menu.Background",           UI_BG_PANEL),
        ("*Menu.Foreground",           UI_FG),
        ("*Menu.activeBackground",     UI_FG_ACCENT),
        ("*Menu.activeForeground",     UI_BG),
    ]
    for k, v in _opts:
        try:
            root.option_add(k, v)
        except Exception:
            pass
    # ttk widgets — these ignore option_add and need a Style.
    try:
        style = ttk.Style(root)
        # 'clam' is the most theme-able built-in.
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox",
                        fieldbackground=UI_BG_INPUT,
                        background=UI_BG_PANEL,
                        foreground=UI_FG,
                        arrowcolor=UI_FG,
                        bordercolor=UI_BG_INPUT,
                        lightcolor=UI_BG_INPUT,
                        darkcolor=UI_BG_INPUT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", UI_BG_INPUT)],
                  foreground=[("readonly", UI_FG)],
                  background=[("readonly", UI_BG_PANEL)])
        style.configure("Vertical.TScrollbar",
                        background=UI_BG_PANEL,
                        troughcolor=UI_BG,
                        bordercolor=UI_BG,
                        arrowcolor=UI_FG)
        style.configure("Horizontal.TScrollbar",
                        background=UI_BG_PANEL,
                        troughcolor=UI_BG,
                        bordercolor=UI_BG,
                        arrowcolor=UI_FG)
        style.configure("TScrollbar",
                        background=UI_BG_PANEL,
                        troughcolor=UI_BG,
                        arrowcolor=UI_FG)
        style.configure("TFrame",     background=UI_BG)
        style.configure("TLabel",     background=UI_BG, foreground=UI_FG)
        style.configure("TButton",    background=UI_BG_PANEL, foreground=UI_FG)
        style.configure("TCheckbutton",
                        background=UI_BG, foreground=UI_FG)
        style.configure("TRadiobutton",
                        background=UI_BG, foreground=UI_FG)
        # -- Notebook (tabbed panels) --
        # Default 'clam' tabs are tiny, low contrast, and indistinguishable
        # from the page below them. Pad them out, give selected/hover
        # states distinct colours, and tint the tab strip so the user
        # can see the active panel at a glance.
        style.configure("TNotebook",
                        background=UI_BG,
                        borderwidth=0,
                        tabmargins=(4, 4, 4, 0))
        style.configure("TNotebook.Tab",
                        background=UI_BG_PANEL,
                        foreground=UI_FG_MUTED,
                        padding=(14, 6),
                        font=("DejaVu Sans", 9, "bold"),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", UI_FG_ACCENT),
                              ("active",   UI_BG_INPUT)],
                  foreground=[("selected", UI_BG),
                              ("active",   UI_FG)],
                  # Selected tab is the larger one — extra padding
                  # makes the active page obvious at a glance instead
                  # of shrinking it (the default 'clam' behaviour).
                  padding=[("selected", (20, 8)),
                           ("active",   (14, 6)),
                           ("",         (14, 6))])
        # Purple-tinted Notebook reserved for the User Pipelines
        # (branches). Each branch tab keeps the same per-branch palette
        # cue we already use elsewhere in the UI.
        style.configure("Branch.TNotebook",
                        background=UI_BG,
                        borderwidth=0,
                        tabmargins=(4, 4, 4, 0))
        style.configure("Branch.TNotebook.Tab",
                        background="#3b1a4a",
                        foreground="#e0c8ff",
                        padding=(14, 6),
                        font=("DejaVu Sans", 9, "bold"),
                        borderwidth=0)
        style.map("Branch.TNotebook.Tab",
                  background=[("selected", "#cc88ff"),
                              ("active",   "#5c2a7a")],
                  foreground=[("selected", "#1a0a2e"),
                              ("active",   "#f0f0f0")],
                  padding=[("selected", (20, 8)),
                           ("active",   (14, 6)),
                           ("",         (14, 6))])
    except Exception:
        pass


class VideoAnalyzer(PipelineUIMixin, FlowchartMixin, UserPipelinesMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense - Cable Video Analyzer")
        # Prefs path is needed early so the saved theme can be read
        # before any widgets are built (re-set harmlessly below).
        self._prefs_path = os.path.join(os.path.expanduser("~"),
                                        ".realsense_analyzer_prefs.json")
        self.ui_theme = self._load_pref("ui_theme", "dark")
        if self.ui_theme not in ("dark", "light"):
            self.ui_theme = "dark"
        _apply_ui_theme(self.root, self.ui_theme)
        # Theme selector var lives on the app (the Settings window
        # binds a combobox to it). settings_win tracks the single
        # Settings Toplevel so it's raised instead of duplicated.
        self.ui_theme_var = tk.StringVar(master=self.root,
                                         value=self.ui_theme)
        self.ui_theme_var.trace_add("write", self._on_theme_change)
        # The All-Masks flowchart window has its OWN theme (default
        # dark — the flowchart is tuned for a dark board) so it can
        # differ from the main program theme.
        self.am_theme = self._load_pref("all_masks_theme", "dark")
        if self.am_theme not in ("dark", "light"):
            self.am_theme = "dark"
        self.am_theme_var = tk.StringVar(master=self.root,
                                         value=self.am_theme)
        self.am_theme_var.trace_add("write", self._on_am_theme_change)
        self.settings_win = None

        # -- UI font tuning (prefs-backed) --------------------------
        # _ui_font_offset bumps every widget's font size; _ui_bold
        # off converts bold text to normal weight. Both applied by
        # _refont_tree (a walk parallel to _retheme_tree).
        self._ui_font_offset = int(self._load_pref("ui_font_offset", 0))
        self._ui_bold = bool(self._load_pref("ui_bold", True))
        self.set_font_offset = tk.IntVar(master=self.root,
                                         value=self._ui_font_offset)
        self.set_ui_bold = tk.BooleanVar(master=self.root,
                                         value=self._ui_bold)
        self.set_font_offset.trace_add("write", self._on_font_setting)
        self.set_ui_bold.trace_add("write", self._on_font_setting)

        # -- Minor layout settings (prefs-backed) -------------------
        # FC_W/FC_H/STEPS_PER_ROW live in the flowchart module and
        # are reassigned there so the All-Masks window picks them up
        # on its next open. DISPLAY_W/H need a restart (the video
        # panels are built once).
        import flowchart_mixin as _fcm
        self._fcm = _fcm
        _fcm.FC_W = int(self._load_pref("fc_w", _fcm.FC_W))
        _fcm.FC_H = int(self._load_pref("fc_h", _fcm.FC_H))
        _fcm.STEPS_PER_ROW = int(self._load_pref("steps_per_row",
                                                 _fcm.STEPS_PER_ROW))
        global DISPLAY_W, DISPLAY_H
        DISPLAY_W = int(self._load_pref("display_w", DISPLAY_W))
        DISPLAY_H = int(self._load_pref("display_h", DISPLAY_H))
        self.set_fc_w      = tk.IntVar(master=self.root, value=_fcm.FC_W)
        self.set_fc_h      = tk.IntVar(master=self.root, value=_fcm.FC_H)
        self.set_steps_row = tk.IntVar(master=self.root,
                                       value=_fcm.STEPS_PER_ROW)
        self.set_disp_w    = tk.IntVar(master=self.root, value=DISPLAY_W)
        self.set_disp_h    = tk.IntVar(master=self.root, value=DISPLAY_H)
        for _v, _k in ((self.set_fc_w, "fc_w"),
                       (self.set_fc_h, "fc_h"),
                       (self.set_steps_row, "steps_per_row"),
                       (self.set_disp_w, "display_w"),
                       (self.set_disp_h, "display_h")):
            _v.trace_add("write",
                         lambda *a, vv=_v, kk=_k:
                             self._on_layout_setting(kk, vv))

        # -- Step cards per row (UI grid wrap) ----------------------
        # How many Morph / PM step cards sit in a row before wrapping.
        # These shadow the PipelineUIMixin class defaults (3 / 4).
        self.STEPS_PER_UI_ROW = int(self._load_pref(
            "steps_per_ui_row", self.STEPS_PER_UI_ROW))
        self.PM_STEPS_PER_UI_ROW = int(self._load_pref(
            "pm_steps_per_ui_row", self.PM_STEPS_PER_UI_ROW))
        self.set_morph_per_row = tk.IntVar(master=self.root,
                                           value=self.STEPS_PER_UI_ROW)
        self.set_pm_per_row    = tk.IntVar(master=self.root,
                                           value=self.PM_STEPS_PER_UI_ROW)
        self.set_morph_per_row.trace_add(
            "write", lambda *a: self._on_steps_per_row(
                "morph", self.set_morph_per_row))
        self.set_pm_per_row.trace_add(
            "write", lambda *a: self._on_steps_per_row(
                "pm", self.set_pm_per_row))

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
        # Per-side BG-sub feed (what gets pushed into MOG2 each frame).
        # Same selector model as branches: "rgb"/"ir" for the full BGR
        # frame, "H"/"S"/"V" for a single HSV channel. We rebuild the
        # MOG2 instance when the source changes (different feed sizes /
        # statistics make the trained model meaningless).
        self.bgsub_rgb_src = tk.StringVar(value="rgb")
        self.bgsub_ir_src  = tk.StringVar(value="ir")
        self._backSub_rgb_src_sig = None
        self._backSub_ir_src_sig  = None

        # Dedicated IR CLAHE (applied on ir_gray BEFORE BG-sub /
        # threshold / PM stage). Kept separate from the PM-step
        # CLAHE op so the user can toggle it without juggling a PM
        # card.
        self.use_clahe_ir   = tk.BooleanVar(value=False)
        self.clahe_ir_clip  = tk.IntVar(value=2)   # clip limit (~1..40)
        self.clahe_ir_tile  = tk.IntVar(value=8)   # tile grid NxN

        self.recording = False
        self.writers   = {}
        self.rec_dir   = ""

        self.loop_var = tk.BooleanVar(value=False)

        # All-Masks window state
        self.all_masks_win         = None
        self.all_masks_labels      = {}   # view_name -> image Label
        self.all_masks_step_labels = {}   # "rgb_step1"... -> text Label
        # Per-section visibility flags keyed by section id
        # ("rgb", "ir", or "up_<branch-name>"). Lazy-populated when
        # the All-Masks window opens; toggling a checkbox triggers
        # a rebuild that skips hidden sections entirely.
        self.all_masks_section_visibility = {}

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

        # Morph pipelines — operate on the BINARY mask after HSV /
        # threshold detection. ANY op is allowed (Morphology, Blur,
        # Illumination, Edge, Threshold, Bitwise, Misc). On a single-
        # channel mask, Edge / Blur ops still work and are useful
        # (e.g. sharpen the mask edges, smooth mask jitter).
        self.rgb_pipeline = []
        self.ir_pipeline  = []
        # Pre-morph pipelines — operate on the SOURCE IMAGE (RGB BGR
        # frame or IR grayscale) BEFORE HSV / threshold detection.
        # Steps here run in user order on the image; the modified
        # image is what feeds the threshold. Same 10-tuple shape as
        # the morph pipeline (the combine slots are unused — the
        # pre-morph stage doesn't run combine ops).
        self.rgb_pre_pipeline = []
        self.ir_pre_pipeline  = []

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
        # Apply the saved theme + font settings to everything the
        # build just created (build sites use dark-tuned literals
        # and explicit bold fonts).
        self._restyle_subtree(self.root)
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

        # ttk.Combobox and tk.Spinbox have their OWN class-level wheel
        # bindings that CHANGE the selected value when the pointer just
        # hovers over them while scrolling. That makes scrolling the
        # panel accidentally edit op / kernel / view fields. Replace
        # those class bindings so the wheel ONLY scrolls the panel —
        # the value changes only on an explicit click / select.
        # (Class bindings run before bind_all, so returning "break"
        # here also stops the bind_all handler firing a second scroll.)
        def _wheel_no_value_change(e):
            _on_wheel(e)
            return "break"
        for _wcls in ("TCombobox", "Spinbox"):
            self.root.bind_class(_wcls, "<MouseWheel>",
                                 _wheel_no_value_change)
            self.root.bind_class(_wcls, "<Button-4>",
                                 _wheel_no_value_change)
            self.root.bind_class(_wcls, "<Button-5>",
                                 _wheel_no_value_change)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        p = self.f

        # ---- folder selector + config save/load ----
        sel = tk.Frame(p)
        sel.pack(fill="x", padx=6, pady=4)
        tk.Label(sel, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_cb  = ttk.Combobox(sel, textvariable=self.folder_var,
                                       width=24, state="readonly")
        self.folder_cb.pack(side="left", padx=4)
        tk.Button(sel, text="Load", command=self._load).pack(side="left")
        # Per-video folder selector — user can point the program at any
        # directory of recordings, not just the default RECORDINGS_DIR.
        self.video_root_var = tk.StringVar(
            value=self._load_pref("video_root_dir", RECORDINGS_DIR))
        tk.Button(sel, text="Browse...",
                  font=("DejaVu Sans", 8),
                  command=self._pick_video_root_dir
                  ).pack(side="left", padx=(6, 2))
        tk.Label(sel, textvariable=self.video_root_var,
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 width=22, anchor="w").pack(side="left")

        # Config save/load (writes/reads a single .json with every
        # tunable — pipelines, branches, sliders, mask colours, etc.).
        tk.Button(sel, text="Apply (F5)", bg="#225522", fg="white",
                  font=("DejaVu Sans", 8, "bold"),
                  command=self._apply_all
                  ).pack(side="right", padx=2)
        tk.Button(sel, text="Settings",
                  font=("DejaVu Sans", 8, "bold"),
                  bg="#1a1a3a", fg="white",
                  command=self._open_settings_window
                  ).pack(side="right", padx=2)
        tk.Button(sel, text="Load config...",
                  font=("DejaVu Sans", 8),
                  command=self._load_config_dialog
                  ).pack(side="right", padx=2)
        tk.Button(sel, text="Save config...",
                  font=("DejaVu Sans", 8),
                  command=self._save_config_dialog
                  ).pack(side="right", padx=2)

        self.lbl_status = tk.Label(sel, text="- select a folder and click Load -",
                                   fg="#c8c8c8")
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
                               font=("DejaVu Sans", 8))
            _vc.pack()
            _vc.bind("<Button-1>",
                     lambda e, c=_vc: c.configure(values=self._all_view_names()))
            tk.Label(frm, text=title, font=("DejaVu Sans", 9, "bold")).pack()
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
                       font=("DejaVu Sans", 9)).pack(side="left", padx=(6, 2))
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

        # ● (U+25CF Black Circle) is in DejaVu Sans so it always
        # renders — unlike 🔴 (U+1F534) which falls back to a "?"
        # glyph on Tk because there's no fallback chain for color
        # emoji on most builds. The darkred bg keeps the visual.
        self.btn_rec = tk.Button(ctrl, text="● Rec", width=8,
                                 bg="darkred", fg="white",
                                 font=("DejaVu Sans", 9, "bold"),
                                 command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=4)
        tk.Button(ctrl, text="Save->", width=6,
                  font=("DejaVu Sans", 8),
                  command=self._pick_recording_dir).pack(side="left")
        tk.Label(ctrl, textvariable=self.recording_dir_var,
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 width=18, anchor="w").pack(side="left", padx=2)
        self.btn_all_masks = tk.Button(ctrl, text="All Masks", width=9,
                                       command=self._toggle_all_masks_window)
        self.btn_all_masks.pack(side="left", padx=4)
        tk.Button(ctrl, text="Apply F5", width=8,
                  bg="#1a3a1a", fg="white", font=("DejaVu Sans", 9, "bold"),
                  command=self._apply_all).pack(side="left", padx=4)
        # 📷 = U+1F4F7 Camera. Same fallback caveat as the record
        # button — colour-only cue if the emoji font isn't installed.
        tk.Button(ctrl, text="◎ F12", width=8,
                  bg="#1a1a3a", fg="white", font=("DejaVu Sans", 9),
                  command=self._take_screenshot).pack(side="left", padx=4)
        tk.Button(ctrl, text="Save->", width=6,
                  font=("DejaVu Sans", 8),
                  command=self._pick_screenshot_dir).pack(side="left")
        tk.Label(ctrl, textvariable=self.screenshot_dir_var,
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 width=24, anchor="w").pack(side="left", padx=2)
        tk.Label(ctrl, text="Space=Play/Pause  <>=Step",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(side="left", padx=4)

        # ---- Pipeline panel ------------------------------------------
        # Top-level Notebook splits the configuration vertically into
        # RGB / IR / Shared lanes. Each side's lane bundles its
        # Detection params, BG-Sub controls, PM Steps and Morph Steps
        # panels — so to tune the RGB or IR side end-to-end the user
        # stays in one tab instead of jumping between Detection
        # Parameters / Processing Pipelines / BG-Sub sub-panels.
        pf = tk.LabelFrame(p, text="  Pipeline  ",
                           font=("DejaVu Sans", 10, "bold"),
                           fg=UI_FG_ACCENT,
                           bd=0, relief="flat", padx=4, pady=2)
        pf.pack(fill="x", padx=8, pady=6)
        self.sv = {}
        # Row frame for every add_param call (kept so blur-type / etc.
        # can later show/hide param rows).
        self._param_rows = {}

        def add_param(parent, name, default, lo, hi):
            v   = tk.IntVar(value=default)
            row = tk.Frame(parent)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, width=11, anchor="w",
                     font=("DejaVu Sans", 8)).pack(side="left")
            tk.Spinbox(row, textvariable=v, from_=lo, to=hi,
                       width=5, font=("DejaVu Sans", 8)).pack(side="left", padx=2)
            tk.Scale(row, variable=v, from_=lo, to=hi,
                     orient="horizontal", length=120,
                     showvalue=False, font=("DejaVu Sans", 9)).pack(side="left")
            self.sv[name] = v
            self._param_rows[name] = row
            return row

        def _hdr(parent, title, color):
            """Section header label inside an RGB / IR / Shared tab."""
            tk.Label(parent, text=title, fg=color,
                     font=("DejaVu Sans", 10, "bold"),
                     anchor="w").pack(fill="x", padx=2, pady=(6, 1))

        # -- Notebook: RGB | IR | Shared -------------------------------
        self.main_nb = ttk.Notebook(pf)
        self.main_nb.pack(fill="x", padx=3, pady=3)
        rgb_tab    = tk.Frame(self.main_nb)
        ir_tab     = tk.Frame(self.main_nb)
        shared_tab = tk.Frame(self.main_nb)
        self.main_nb.add(rgb_tab,    text="RGB")
        self.main_nb.add(ir_tab,     text="IR")
        self.main_nb.add(shared_tab, text="Shared")

        # Shared BG-Sub toggle (used by both rgb_bgsub and ir_bgsub
        # MOG2 instances). Declared before either tab populates so
        # both RGB and IR tabs can bind the same checkbutton var.
        self.use_bgsub = tk.BooleanVar(value=True)
        _bgsrc_opts = ["rgb", "ir", "H", "S", "V"]

        def _bgsub_block(parent, side):
            """Build the BG-Sub control block for the given side
            ('rgb' or 'ir'). The history / varTh / Reset are shared
            (one MOG2 per side, but both fed the same params) so the
            sliders edit the same self.sv['BG_hist'] / 'BG_var' vars
            shown in either tab — changing them in RGB also updates
            the value the IR tab displays."""
            _hdr(parent, f"BG Sub  -  {side.upper()}-side feed", "#88dd88")
            tk.Checkbutton(parent, text="Use BG Subtractor",
                           variable=self.use_bgsub,
                           font=("DejaVu Sans", 8, "bold")).pack(anchor="w")
            src_var = self.bgsub_rgb_src if side == "rgb" else self.bgsub_ir_src
            row = tk.Frame(parent); row.pack(fill="x", pady=1)
            tk.Label(row, text="Source:", font=("DejaVu Sans", 8),
                     width=11, anchor="w").pack(side="left")
            ttk.Combobox(row, textvariable=src_var,
                         values=_bgsrc_opts, width=6,
                         state="readonly",
                         font=("DejaVu Sans", 8)).pack(side="left", padx=2)
            tk.Label(parent,
                     text=("rgb / ir = full BGR frame. "
                           "H / S / V = single HSV channel."),
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
                     wraplength=420, justify="left"
                     ).pack(anchor="w", padx=2, pady=(0, 2))
            # History / VarTh sliders. First side that lays them out
            # registers the IntVars in self.sv; the second side reuses
            # the same vars (referenced via add_param's IntVar, which
            # is created fresh — so we manually wire the existing var
            # the second time around).
            if "BG_hist" not in self.sv:
                add_param(parent, "BG_hist", 500, 10, 2000)
                add_param(parent, "BG_var",   50,  1,  200)
            else:
                for nm, lo, hi in (("BG_hist", 10, 2000),
                                   ("BG_var",   1,  200)):
                    v = self.sv[nm]
                    r = tk.Frame(parent); r.pack(fill="x", pady=1)
                    tk.Label(r, text=nm, width=11, anchor="w",
                             font=("DejaVu Sans", 8)).pack(side="left")
                    tk.Spinbox(r, textvariable=v, from_=lo, to=hi,
                               width=5,
                               font=("DejaVu Sans", 8)).pack(side="left", padx=2)
                    tk.Scale(r, variable=v, from_=lo, to=hi,
                             orient="horizontal", length=120,
                             showvalue=False,
                             font=("DejaVu Sans", 9)).pack(side="left")
            tk.Button(parent, text="Reset BG Sub",
                      font=("DejaVu Sans", 8),
                      command=self._reset_bg).pack(anchor="w", pady=2)

        # ====================== RGB tab =============================
        _hdr(rgb_tab, "Detection  -  HSV", "#aaccff")
        tk.Label(rgb_tab, text="-- Hue range 1 --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(rgb_tab, "H1_low",  0,   0, 180)
        add_param(rgb_tab, "H1_high", 10,  0, 180)
        tk.Label(rgb_tab, text="-- Hue range 2 --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(rgb_tab, "H2_low",  160, 0, 180)
        add_param(rgb_tab, "H2_high", 180, 0, 180)
        tk.Label(rgb_tab, text="-- Saturation --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(rgb_tab, "S_min",   80,  0, 255)
        add_param(rgb_tab, "S_max",   255, 0, 255)
        tk.Label(rgb_tab, text="-- Value (brightness) --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(rgb_tab, "V_min",   40,  0, 255)
        add_param(rgb_tab, "V_max",   255, 0, 255)
        self.rgb_mask_color_hex = tk.StringVar(
            value=self._load_pref("rgb_mask_color", "#ff0000"))
        self._make_color_picker_row(
            rgb_tab, "Mask colour:", self.rgb_mask_color_hex,
            "rgb_mask_color", "Pick RGB mask colour")

        _bgsub_block(rgb_tab, "rgb")

        # PM Steps frame (image-stage, before HSV). Re-parented from
        # the old Processing Pipelines outer LabelFrame so the entire
        # RGB lane lives inside the RGB tab now.
        self.rgb_pre_pip_frame = tk.LabelFrame(
            rgb_tab,
            text="  [PM]  RGB Pre-morph Steps  (on image, BEFORE HSV)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#88dd88",
            bd=0, relief="flat", padx=3, pady=2)
        self.rgb_pre_pip_frame.pack(fill="x", padx=3, pady=(8, 2))
        # Morph Steps frame (mask-stage, after HSV).
        self.rgb_pip_frame = tk.LabelFrame(
            rgb_tab,
            text="  [Morph]  RGB Morph Steps  (on mask, AFTER HSV)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#4488ff",
            bd=0, relief="flat", padx=3, pady=2)
        self.rgb_pip_frame.pack(fill="x", padx=3, pady=2)

        # ====================== IR tab ==============================
        _hdr(ir_tab, "Detection  -  IR threshold", "#44cc66")
        tk.Label(ir_tab, text="-- IR Threshold --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(ir_tab, "IR_thresh", 100, 1, 254)
        self.ir_mask_color_hex = tk.StringVar(
            value=self._load_pref("ir_mask_color", "#ffff00"))
        self._make_color_picker_row(
            ir_tab, "IR mask colour:", self.ir_mask_color_hex,
            "ir_mask_color", "Pick IR mask colour")

        # Dedicated IR CLAHE — applied to ir_gray BEFORE BG-sub /
        # threshold so the IR mask sees contrast-enhanced input.
        # Independent from any PM-step CLAHE (which also runs, after
        # this); the two stack if both are enabled.
        tk.Label(ir_tab, text="-- IR CLAHE --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        tk.Checkbutton(ir_tab, text="Use IR CLAHE",
                       variable=self.use_clahe_ir,
                       font=("DejaVu Sans", 8)).pack(anchor="w")
        _clahe_row1 = tk.Frame(ir_tab); _clahe_row1.pack(fill="x", pady=1)
        tk.Label(_clahe_row1, text="Clip limit", width=11, anchor="w",
                 font=("DejaVu Sans", 8)).pack(side="left")
        tk.Spinbox(_clahe_row1, textvariable=self.clahe_ir_clip,
                   from_=1, to=40, width=5,
                   font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Scale(_clahe_row1, variable=self.clahe_ir_clip, from_=1, to=40,
                 orient="horizontal", length=120,
                 showvalue=False, font=("DejaVu Sans", 9)).pack(side="left")
        _clahe_row2 = tk.Frame(ir_tab); _clahe_row2.pack(fill="x", pady=1)
        tk.Label(_clahe_row2, text="Tile grid", width=11, anchor="w",
                 font=("DejaVu Sans", 8)).pack(side="left")
        tk.Spinbox(_clahe_row2, textvariable=self.clahe_ir_tile,
                   from_=1, to=32, width=5,
                   font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Scale(_clahe_row2, variable=self.clahe_ir_tile, from_=1, to=32,
                 orient="horizontal", length=120,
                 showvalue=False, font=("DejaVu Sans", 9)).pack(side="left")

        tk.Label(ir_tab, text="-- IR Display --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        cmap_row = tk.Frame(ir_tab); cmap_row.pack(fill="x", pady=2)
        tk.Label(cmap_row, text="IR colour:",
                 font=("DejaVu Sans", 8)).pack(side="left")
        self.ir_cmap_var = tk.StringVar(value="Gray")
        ttk.Combobox(cmap_row, textvariable=self.ir_cmap_var,
                     values=list(IR_COLORMAPS.keys()),
                     width=9, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)

        _bgsub_block(ir_tab, "ir")

        # IR PM + Morph steps frames.
        self.ir_pre_pip_frame = tk.LabelFrame(
            ir_tab,
            text="  [PM]  IR Pre-morph Steps  (on gray, BEFORE threshold)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#88dd88",
            bd=0, relief="flat", padx=3, pady=2)
        self.ir_pre_pip_frame.pack(fill="x", padx=3, pady=(8, 2))
        self.ir_pip_frame = tk.LabelFrame(
            ir_tab,
            text="  [Morph]  IR Morph Steps  (on mask, AFTER threshold)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#44cc66",
            bd=0, relief="flat", padx=3, pady=2)
        self.ir_pip_frame.pack(fill="x", padx=3, pady=2)

        # ====================== Shared tab ==========================
        # Stuff that isn't tied to one side: YOLO, post-filter knobs,
        # the global Kernel-shape default for steps that don't carry
        # their own per-step shape.
        # Appearance / folders / model live in the dedicated Settings
        # window (toolbar "Settings" button) — see _open_settings_window.
        _hdr(shared_tab, "YOLO", "#ffaa66")
        yolo_status = ("loaded" if self.yolo_model else
                       "no ultralytics" if not _YOLO_AVAILABLE else "model not found")
        _dev_lbl = ("GPU" if _TORCH_DEVICE != "cpu" else "CPU")
        tk.Label(shared_tab,
                 text=f"({yolo_status}, {_dev_lbl})",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        self.use_yolo = tk.BooleanVar(value=self.yolo_model is not None)
        tk.Checkbutton(shared_tab, text="Use YOLO", variable=self.use_yolo,
                       font=("DejaVu Sans", 8)).pack(anchor="w")
        add_param(shared_tab, "YOLO_Conf", 50, 1, 99)
        self.yolo_mask_color_hex = tk.StringVar(
            value=self._load_pref("yolo_mask_color", "#00ffff"))
        self._make_color_picker_row(
            shared_tab, "YOLO colour:", self.yolo_mask_color_hex,
            "yolo_mask_color", "Pick YOLO colour")
        tk.Button(shared_tab, text="Load YOLO model...",
                  font=("DejaVu Sans", 8),
                  command=self._pick_yolo_model_file
                  ).pack(anchor="w", pady=(2, 0))
        _saved_path = (self._load_pref("yolo_model_path",
                                       YOLO_MODEL_PATH)
                       if self.yolo_model else "(none loaded)")
        self.yolo_model_path_var = tk.StringVar(value=_saved_path)
        tk.Label(shared_tab, textvariable=self.yolo_model_path_var,
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 anchor="w", justify="left",
                 wraplength=420
                 ).pack(anchor="w", padx=2)

        self.yolo_class_panel = tk.LabelFrame(
            shared_tab, text="  YOLO classes  ",
            font=("DejaVu Sans", 9, "bold"), fg="#ffaa66",
            bd=0, relief="flat", padx=3, pady=2)
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

        _hdr(shared_tab, "Filter", "#ffdd66")
        tk.Label(shared_tab, text="-- Min contour area --",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(anchor="w")
        add_param(shared_tab, "Min_area", 20, 1, 5000)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        for text, var in [("Show Bounding Boxes", self.show_boxes),
                          ("Mask Overlay on Raw", self.show_overlay)]:
            tk.Checkbutton(shared_tab, text=text, variable=var,
                           font=("DejaVu Sans", 8)).pack(anchor="w")

        _hdr(shared_tab, "Pre-process (default kernel shape)", "#ffaa66")
        tk.Label(shared_tab,
                 text="Default K shape for XY steps that don't override it.",
                 font=("DejaVu Sans", 9, "italic"), fg="#aaccaa",
                 anchor="w", justify="left").pack(fill="x", padx=2)
        ks_row = tk.Frame(shared_tab); ks_row.pack(fill="x", pady=2)
        tk.Label(ks_row, text="K Shape:", font=("DejaVu Sans", 8),
                 width=11, anchor="w").pack(side="left")
        self.kernel_shape_var = tk.StringVar(value="Rect")
        ttk.Combobox(ks_row, textvariable=self.kernel_shape_var,
                     values=list(KERNEL_SHAPES.keys()),
                     width=9, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Label(shared_tab,
                 text=("Rect = fast / axis-aligned   |   "
                       "Ellipse = isotropic   |   Cross = thin lines.   "
                       "X/Y-only steps always use Rect."),
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 anchor="w", justify="left",
                 wraplength=420).pack(fill="x", padx=2, pady=(0, 4))

        rgb_defaults = [
            (True,  "Close",    1, "XY", 3,  10, 128),
            (False, "Open",     1, "XY", 3,   3, 128),
        ]
        ir_defaults = [
            (True,  "Close",    1, "XY", 3,   3, 128),
            (False, "Open",     1, "XY", 3,   3, 128),
        ]
        self._create_default_steps(self.rgb_pipeline, rgb_defaults)
        self._create_default_steps(self.ir_pipeline,  ir_defaults)
        self._rebuild_pipeline_ui(self.rgb_pip_frame, self.rgb_pipeline)
        self._rebuild_pipeline_ui(self.ir_pip_frame,  self.ir_pipeline)
        # Pre-morph pipelines start empty — the user adds the steps
        # they actually need (blur kernel sizes / illumination ops).
        self._rebuild_pre_morph_pipeline_ui(
            self.rgb_pre_pip_frame, self.rgb_pre_pipeline)
        self._rebuild_pre_morph_pipeline_ui(
            self.ir_pre_pip_frame,  self.ir_pre_pipeline)

        # ---- Combine + User Pipelines as additional tabs ---------------
        # Both move into the same main_nb Notebook so RGB / IR / Shared
        # / Combine / Pipelines all live behind one tab strip — no more
        # vertical sprawl below the Pipeline panel.
        combine_tab    = tk.Frame(self.main_nb)
        userpipes_tab  = tk.Frame(self.main_nb)
        self.main_nb.add(combine_tab,   text="Combine")
        self.main_nb.add(userpipes_tab, text="New Pipeline")

        # -- Combine tab content ---------------------------------------
        _hdr(combine_tab, "Combine  (AND / OR / XOR any two views -> 'combined')",
             "#ffdd66")
        crow = tk.Frame(combine_tab); crow.pack(fill="x", pady=3, padx=4)
        tk.Label(crow, text="A:", font=("DejaVu Sans", 9, "bold"),
                 fg="#4488ff").pack(side="left")
        self.combine_a_var = tk.StringVar(value="rgb_mask")
        cb_a = ttk.Combobox(crow, textvariable=self.combine_a_var,
                            values=self._all_view_names(),
                            width=20, state="readonly",
                            font=("DejaVu Sans", 8))
        cb_a.pack(side="left", padx=2)
        cb_a.bind("<Button-1>",
                  lambda e, c=cb_a: c.configure(values=self._all_view_names()))
        self.combine_op_var = tk.StringVar(value="AND")
        ttk.Combobox(crow, textvariable=self.combine_op_var,
                     values=["AND", "OR", "XOR"],
                     width=5, state="readonly",
                     font=("DejaVu Sans", 9, "bold")
                     ).pack(side="left", padx=6)
        tk.Label(crow, text="B:", font=("DejaVu Sans", 9, "bold"),
                 fg="#44cc66").pack(side="left")
        self.combine_b_var = tk.StringVar(value="ir_mask")
        cb_b = ttk.Combobox(crow, textvariable=self.combine_b_var,
                            values=self._all_view_names(),
                            width=20, state="readonly",
                            font=("DejaVu Sans", 8))
        cb_b.pack(side="left", padx=2)
        cb_b.bind("<Button-1>",
                  lambda e, c=cb_b: c.configure(values=self._all_view_names()))
        tk.Label(combine_tab,
                 text="-> select 'combined' in any panel or zoom",
                 font=("DejaVu Sans", 9), fg="#c8c8c8"
                 ).pack(anchor="w", padx=8, pady=(2, 0))

        # -- New Pipeline (User Pipelines / branches) tab content ------
        _hdr(userpipes_tab,
             "User Pipelines  (parallel branches - outputs as up_<name>)",
             "#cc88ff")
        top_row = tk.Frame(userpipes_tab)
        top_row.pack(fill="x", padx=4, pady=2)
        tk.Button(top_row, text="+ New Pipeline", bg="#3b1a4a", fg="white",
                  font=("DejaVu Sans", 8, "bold"),
                  command=self._on_add_user_pipeline).pack(side="left")
        tk.Label(top_row,
                 text=("  Each pipeline takes any view as input, runs "
                       "its own steps, and exposes its result as "
                       "up_<name>. up_* views are usable as panel "
                       "views, combine sources, and other pipelines' "
                       "inputs."),
                 font=("DejaVu Sans", 9), fg="#c8c8c8", wraplength=900,
                 justify="left").pack(side="left")
        # Branches each live in their own Notebook tab so the panel
        # stays compact even when many branches exist. The tab label
        # mirrors the branch name var (live). The purple-tinted style
        # matches the rest of the user-pipelines palette so the tab
        # strip is instantly recognisable.
        self.user_pipelines_host = ttk.Notebook(userpipes_tab,
                                                style="Branch.TNotebook")
        self.user_pipelines_host.pack(fill="both", expand=True,
                                      padx=4, pady=2)

        # ---- status bars ----
        self.lbl_cable = tk.Label(p, text="RGB cable pixels: -   IR cable pixels: -",
                                  font=("Courier", 8), anchor="w")
        self.lbl_cable.pack(fill="x", padx=6)
        self.lbl_rec_status = tk.Label(p, text="", fg="red",
                                       font=("DejaVu Sans", 8), anchor="w")
        self.lbl_rec_status.pack(fill="x", padx=6)

    def _refresh(self, *_):
        """Re-process the cached frame immediately when paused.
        Suppressed during config-load so trace bursts don't fire
        _process on half-applied state."""
        if getattr(self, "_loading_config", False):
            return
        if not self.playing and self._last_f_rgb is not None:
            self._process(self._last_f_rgb, self._last_f_ir)

    def _apply_all(self, *_):
        """Apply button / F5 hotkey: rebuild the All-Masks canvas
        (so newly added branches / step changes show up without
        manually closing it) and reprocess the current frame."""
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _confirm_path(self, path, title="Confirm selection",
                      action_label="Use this path"):
        """Show a modal Yes/No popup so the user can deliberately
        commit (or back out of) a path picked by a file/dir dialog.
        Fixes the 'system dialog auto-selected the previous folder
        because the OK button was under my click' issue: the user
        always sees the picked path and presses OK or Cancel."""
        if not path:
            return False
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)
        tk.Label(win, text=action_label + ":",
                 font=("DejaVu Sans", 9, "bold")
                 ).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(win, text=path,
                 font=("DejaVu Sans", 9), fg="#3366aa",
                 wraplength=520, justify="left", anchor="w"
                 ).pack(fill="x", padx=14, pady=(0, 12))
        result = {"ok": False}
        btns = tk.Frame(win)
        btns.pack(fill="x", padx=10, pady=(0, 12))
        def _ok():
            result["ok"] = True
            win.destroy()
        ok_btn = tk.Button(btns, text="OK", width=10,
                           command=_ok,
                           bg="#225522", fg="white",
                           font=("DejaVu Sans", 9, "bold"))
        ok_btn.pack(side="right", padx=4)
        tk.Button(btns, text="Cancel", width=10,
                  command=win.destroy).pack(side="right", padx=4)
        # Brief delay before OK becomes clickable — so a stray click
        # carried over from the system dialog can't auto-confirm.
        ok_btn.config(state="disabled")
        win.after(450, lambda: ok_btn.config(state="normal"))
        win.bind("<Return>", lambda *_a: _ok()
                 if str(ok_btn["state"]) == "normal" else None)
        win.bind("<Escape>", lambda *_a: win.destroy())
        try:
            win.wait_window()
        except Exception:
            pass
        return result["ok"]

    @staticmethod
    def _apply_blur(img, blur_type, kx, ky=0):
        """Run the selected blur (Gaussian / Median / Bilateral) on
        `img`. 'None' bypasses entirely. Parameter conventions
        match the Ops list:
          GaussBlur, MedianBlur  -> kx is the (odd) kernel size,
          BilateralBlur          -> kx = diameter, ky = sigma."""
        if blur_type == "None" or kx <= 0:
            return img
        def _odd(n):
            return n if n % 2 == 1 else n + 1
        if blur_type == "MedianBlur":
            k = _odd(max(1, int(kx)))
            return cv2.medianBlur(img, k)
        if blur_type == "BilateralBlur":
            d = max(1, int(kx))
            sig = max(1, int(ky) if ky > 0 else int(kx))
            return cv2.bilateralFilter(img, d, sig, sig)
        # default: GaussBlur
        k = _odd(max(1, int(kx)))
        return cv2.GaussianBlur(img, (k, k), 0)

    # ------------------------------------------------------------------
    # Save / load config (sliders + pipelines + branches + colours)
    # ------------------------------------------------------------------
    def _collect_config(self):
        """Serialise every tunable into a JSON-safe dict."""
        def _step_to_dict(t):
            (en, op, n, dr, kx, ky, th, cen, cop, csr) = t
            d = {
                "en": en.get(), "op": op.get(), "n": n.get(),
                "dir": dr.get(), "kx": kx.get(), "ky": ky.get(),
                "t": th.get(), "comb_en": cen.get(),
                "comb_op": cop.get(), "comb_src": csr.get(),
            }
            ks = (getattr(self, "_kshape_state", {}) or {}).get(id(t))
            if ks is not None:
                try:
                    d["kshape"] = ks["kshape"].get()
                except Exception:
                    pass
            pmt = (getattr(self, "_pm_target_state", {}) or {}).get(id(t))
            if pmt is not None:
                try:
                    d["pm_target"] = pmt["target"].get()
                except Exception:
                    pass
            ov = (getattr(self, "_overlay_state", {}) or {}).get(id(t))
            if ov is not None:
                d["overlay"] = {k: ov[k].get() for k in
                                ("color1", "color2", "mask2_src", "base_src")}
            ys = (getattr(self, "_yolo_state", {}) or {}).get(id(t))
            if ys is not None:
                d["yolo"] = {}
                for k in ("yolo_en", "yolo_input_kind",
                          "yolo_src", "yolo_mask_src", "yolo_mode"):
                    if k in ys:
                        try:
                            d["yolo"][k] = ys[k].get()
                        except Exception:
                            pass
                # Per-step class tick boxes -> {str(class_id): bool}.
                _ycl = ys.get("yolo_classes")
                if isinstance(_ycl, dict) and _ycl:
                    try:
                        d["yolo"]["yolo_classes"] = {
                            str(_cid): bool(_v.get())
                            for _cid, _v in _ycl.items()}
                    except Exception:
                        pass
            stt = (getattr(self, "_step_type_state", {}) or {}).get(id(t))
            if stt is not None:
                try:
                    d["step_kind"] = stt["kind"].get()
                except Exception:
                    pass
            return d

        def _branch_to_dict(rec):
            d = {}
            for k, v in rec.items():
                if k in ("frame", "steps", "pre_steps",
                         "steps_row", "pre_steps_row",
                         "_inline_det_host", "_pre_snap"):
                    continue
                try:
                    d[k] = v.get()
                except Exception:
                    pass
            d["steps"]     = [_step_to_dict(s) for s in rec.get("steps", [])]
            d["pre_steps"] = [_step_to_dict(s) for s in rec.get("pre_steps", [])]
            return d

        cfg = {
            "version": 1,
            "params":  {k: v.get() for k, v in self.sv.items()},
            "flags": {
                "use_bgsub":      self.use_bgsub.get(),
                "show_boxes":     self.show_boxes.get(),
                "show_overlay":   self.show_overlay.get(),
                "use_yolo":       self.use_yolo.get(),
                "loop_var":       self.loop_var.get(),
                "use_clahe_ir":   self.use_clahe_ir.get(),
            },
            "ints": {
                "clahe_ir_clip":  self.clahe_ir_clip.get(),
                "clahe_ir_tile":  self.clahe_ir_tile.get(),
            },
            "strings": {
                "ir_cmap":        self.ir_cmap_var.get(),
                "kernel_shape":   getattr(self, "kernel_shape_var",
                                          tk.StringVar(value="Rect")).get(),
                "rgb_mask_color": self.rgb_mask_color_hex.get(),
                "ir_mask_color":  self.ir_mask_color_hex.get(),
                "yolo_mask_color": self.yolo_mask_color_hex.get(),
                "bgsub_rgb_src":  self.bgsub_rgb_src.get(),
                "bgsub_ir_src":   self.bgsub_ir_src.get(),
            },
            "rgb_pipeline":     [_step_to_dict(s) for s in self.rgb_pipeline],
            "ir_pipeline":      [_step_to_dict(s) for s in self.ir_pipeline],
            "rgb_pre_pipeline": [_step_to_dict(s) for s in self.rgb_pre_pipeline],
            "ir_pre_pipeline":  [_step_to_dict(s) for s in self.ir_pre_pipeline],
            "user_pipelines":   [_branch_to_dict(r) for r in self.user_pipelines],
        }
        return cfg

    def _apply_config(self, cfg):
        """Apply a previously-saved config dict to the current UI."""
        # Suppress _refresh during load: every var.set fires a trace,
        # and each trace ends up calling _process. Loading a 20-step
        # config would otherwise re-run the full pipeline ~100 times.
        self._loading_config = True
        try:
            self._apply_config_inner(cfg)
        finally:
            self._loading_config = False
        # Single refresh + canvas rebuild at the very end.
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _apply_config_inner(self, cfg):
        # Scalars
        for k, val in (cfg.get("params") or {}).items():
            if k in self.sv:
                try:
                    self.sv[k].set(val)
                except Exception:
                    pass
        flags = cfg.get("flags") or {}
        # `use_clahe` from old configs is silently ignored — CLAHE now
        # lives as an IR PM step. Same for the removed pre-blur fields.
        for k in ("use_bgsub", "show_boxes",
                  "show_overlay", "use_yolo", "loop_var",
                  "use_clahe_ir"):
            if k in flags and hasattr(self, k):
                try:
                    getattr(self, k).set(bool(flags[k]))
                except Exception:
                    pass
        ints = cfg.get("ints") or {}
        for src_key, attr in [
            ("clahe_ir_clip", "clahe_ir_clip"),
            ("clahe_ir_tile", "clahe_ir_tile"),
        ]:
            if src_key in ints and hasattr(self, attr):
                try:
                    getattr(self, attr).set(int(ints[src_key]))
                except Exception:
                    pass
        strs = cfg.get("strings") or {}
        for src_key, attr in [
            ("ir_cmap",        "ir_cmap_var"),
            ("kernel_shape",   "kernel_shape_var"),
            ("rgb_mask_color", "rgb_mask_color_hex"),
            ("ir_mask_color",  "ir_mask_color_hex"),
            ("yolo_mask_color", "yolo_mask_color_hex"),
            ("bgsub_rgb_src",  "bgsub_rgb_src"),
            ("bgsub_ir_src",   "bgsub_ir_src"),
        ]:
            if src_key in strs and hasattr(self, attr):
                try:
                    getattr(self, attr).set(strs[src_key])
                except Exception:
                    pass

        # Pipelines — replace contents wholesale.
        def _set_step(t, d):
            (en, op, n, dr, kx, ky, th, cen, cop, csr) = t
            for k, v in [("en", en), ("op", op), ("n", n), ("dir", dr),
                         ("kx", kx), ("ky", ky), ("t", th),
                         ("comb_en", cen), ("comb_op", cop),
                         ("comb_src", csr)]:
                if k in d:
                    try:
                        v.set(d[k])
                    except Exception:
                        pass
            # Eagerly create overlay/yolo/kshape state for this step
            # so we can write the saved values (the lazy-create path
            # means the dicts are empty until the user opens that UI).
            if "kshape" in d:
                ks = self._kshape_state_for(t)
                try:
                    ks["kshape"].set(d["kshape"])
                except Exception:
                    pass
            if "pm_target" in d:
                pmt = self._pm_target_state_for(t)
                try:
                    pmt["target"].set(d["pm_target"])
                except Exception:
                    pass
            ovd = d.get("overlay") or {}
            if ovd:
                ov = self._overlay_state_for(t)
                for k in ("color1", "color2", "mask2_src", "base_src"):
                    if k in ovd:
                        try:
                            ov[k].set(ovd[k])
                        except Exception:
                            pass
            ysd = d.get("yolo") or {}
            if ysd:
                ys = self._yolo_state_for(t)
                for k in ("yolo_en", "yolo_input_kind",
                          "yolo_src", "yolo_mask_src", "yolo_mode"):
                    if k in ysd and k in ys:
                        try:
                            ys[k].set(ysd[k])
                        except Exception:
                            pass
                _ycl = ysd.get("yolo_classes")
                if isinstance(_ycl, dict):
                    for _cid_s, _val in _ycl.items():
                        try:
                            self._yolo_class_var(
                                ys, int(_cid_s)).set(bool(_val))
                        except Exception:
                            pass
            if "step_kind" in d:
                stt = self._step_type_state_for(t)
                try:
                    stt["kind"].set(d["step_kind"])
                except Exception:
                    pass

        def _rebuild_pipeline(pl, frame, step_dicts):
            pl.clear()
            for _d in step_dicts:
                # Build a default-shaped step then load the dict into it.
                self._create_default_steps(pl, [(_d.get("en", False),
                                                 _d.get("op", "Dilate"),
                                                 _d.get("n", 1),
                                                 _d.get("dir", "Both"),
                                                 _d.get("kx", 3),
                                                 _d.get("ky", 3),
                                                 _d.get("t", 0))])
                _set_step(pl[-1], _d)
            self._rebuild_pipeline_ui(frame, pl)

        if "rgb_pipeline" in cfg:
            _rebuild_pipeline(self.rgb_pipeline,
                              self.rgb_pip_frame,
                              cfg["rgb_pipeline"])
        if "ir_pipeline" in cfg:
            _rebuild_pipeline(self.ir_pipeline,
                              self.ir_pip_frame,
                              cfg["ir_pipeline"])

        # Pre-morph pipelines — use the dedicated rebuild helper so
        # the compact card layout is recreated (not the full
        # combine/YOLO morph card).
        def _rebuild_pre_pipeline(pl, frame, step_dicts):
            pl.clear()
            for _d in step_dicts:
                self._create_default_steps(pl, [(_d.get("en", False),
                                                 _d.get("op", "GaussBlur"),
                                                 _d.get("n", 1),
                                                 _d.get("dir", "XY"),
                                                 _d.get("kx", 5),
                                                 _d.get("ky", 5),
                                                 _d.get("t", 0))])
                _set_step(pl[-1], _d)
            self._rebuild_pre_morph_pipeline_ui(frame, pl)

        if "rgb_pre_pipeline" in cfg:
            _rebuild_pre_pipeline(self.rgb_pre_pipeline,
                                  self.rgb_pre_pip_frame,
                                  cfg["rgb_pre_pipeline"])
        if "ir_pre_pipeline" in cfg:
            _rebuild_pre_pipeline(self.ir_pre_pipeline,
                                  self.ir_pre_pip_frame,
                                  cfg["ir_pre_pipeline"])

        # User pipelines — destroy and recreate. Each branch lives in
        # its own Notebook tab now, so we forget() the tab before
        # destroying its frame to keep the ttk state consistent.
        if "user_pipelines" in cfg:
            for rec in list(self.user_pipelines):
                try:
                    if rec.get("frame") is not None:
                        try:
                            self.user_pipelines_host.forget(rec["frame"])
                        except Exception:
                            pass
                        rec["frame"].destroy()
                except Exception:
                    pass
            self.user_pipelines.clear()
            for bd in cfg["user_pipelines"]:
                self._create_branch_from_dict(bd)
        # Outer _apply_config does the final refresh once loading
        # finishes, so we don't burn cycles re-processing per var-set.

    def _create_branch_from_dict(self, bd):
        """Recreate a single user-pipeline branch from its serialised
        dict (the inverse of _branch_to_dict). Mirrors _on_submit_setup
        but seeds the rec vars from `bd` instead of dialog defaults."""
        # Start with a fresh rec via the same code paths that the setup
        # dialog uses — we re-call _new_branch_record (a thin wrapper)
        # so trace_add wiring and the persistent header layout are
        # identical to interactively-created branches.
        rec = self._new_branch_record()
        for k, v in bd.items():
            if k in ("steps", "pre_steps"):
                continue
            if k in rec:
                try:
                    rec[k].set(v)
                except Exception:
                    pass
        self._add_user_pipeline(rec)
        # PM steps first (run before morph), then morph steps.
        for psd in bd.get("pre_steps", []):
            self._add_branch_pre_step(rec, psd)
        for sd in bd.get("steps", []):
            self._add_branch_step(rec, sd)

    def _save_config_dialog(self):
        """Open a save dialog and write the current config as JSON."""
        from tkinter import filedialog
        import json
        path = filedialog.asksaveasfilename(
            title="Save analyzer config",
            defaultextension=".json",
            filetypes=[("JSON config", "*.json"), ("All files", "*")],
            initialfile="analyzer_config.json")
        if not path:
            return
        if not self._confirm_path(path, "Confirm save location",
                                  "Save config to"):
            return
        try:
            with open(path, "w") as f:
                json.dump(self._collect_config(), f, indent=2)
            self._save_pref("last_config_path", path)
            self.lbl_status.config(
                text=f"Config saved to {path}", fg="#88ff88")
        except Exception as e:
            self.lbl_status.config(
                text=f"Save config failed: {e}", fg="red")

    def _load_config_dialog(self):
        """Open a load dialog and apply the chosen config."""
        from tkinter import filedialog
        import json
        path = filedialog.askopenfilename(
            title="Load analyzer config",
            initialfile=self._load_pref("last_config_path", ""),
            filetypes=[("JSON config", "*.json"), ("All files", "*")])
        if not path:
            return
        if not self._confirm_path(path, "Confirm config to load",
                                  "Load config from"):
            return
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            self._apply_config(cfg)
            self._save_pref("last_config_path", path)
            self.lbl_status.config(
                text=f"Config loaded from {path}", fg="#88ff88")
        except Exception as e:
            self.lbl_status.config(
                text=f"Load config failed: {e}", fg="red")

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

    # ------------------------------------------------------------------
    # Theme switching (dark <-> light)
    # ------------------------------------------------------------------
    def _retheme_one(self, w, theme):
        """Remap a single widget's recognised colours to `theme`.
        Only hexes registered in THEME_COLORS are touched, so mask
        colours / black thumbnails / anything custom is left alone.
        Works in both directions (dark<->light)."""
        for opt in ("background", "foreground", "activebackground",
                    "activeforeground", "selectcolor", "insertbackground",
                    "troughcolor", "highlightbackground", "highlightcolor"):
            try:
                cur = str(w.cget(opt))
            except Exception:
                continue
            canon = _HEX_TO_CANON.get(cur.lower())
            if canon is None:
                continue
            try:
                w.configure(**{opt: THEME_COLORS[canon][theme]})
            except Exception:
                pass

    def _retheme_tree(self, widget, theme=None):
        """Recursively retheme `widget` and its descendants to `theme`
        (defaults to the main program theme). Separate Toplevels are
        skipped — call this directly on one to include it."""
        if theme is None:
            theme = getattr(self, "ui_theme", "dark")
        try:
            self._retheme_one(widget, theme)
        except Exception:
            pass
        try:
            children = widget.winfo_children()
        except Exception:
            return
        for ch in children:
            if isinstance(ch, tk.Toplevel):
                continue
            self._retheme_tree(ch, theme)

    def _style_all_masks(self, win):
        """Apply the All-Masks window's own theme + the shared font
        settings to it. Always rethemes (the window may have been
        built under a different ambient theme)."""
        am = getattr(self, "am_theme", None) or getattr(self, "ui_theme",
                                                        "dark")
        self._retheme_tree(win, am)
        if (int(getattr(self, "_ui_font_offset", 0)) != 0
                or not getattr(self, "_ui_bold", True)):
            self._refont_tree(win)

    def _on_theme_change(self, *_a):
        """Apply a live MAIN-program theme switch: re-run option_add /
        ttk styles (so future widgets use the new palette) and walk
        the existing widget tree remapping recognised colours."""
        theme = self.ui_theme_var.get()
        if theme not in ("dark", "light"):
            return
        self.ui_theme = theme
        self._save_pref("ui_theme", theme)
        _apply_ui_theme(self.root, theme)
        self._retheme_tree(self.root, theme)
        # The Settings window follows the main theme; the All-Masks
        # window keeps its own (am_theme). Both are separate Toplevels
        # skipped by the root walk above, so retheme them explicitly.
        if (getattr(self, "settings_win", None)
                and self.settings_win.winfo_exists()):
            self._retheme_tree(self.settings_win, theme)
        if (getattr(self, "all_masks_win", None)
                and self.all_masks_win.winfo_exists()):
            self._style_all_masks(self.all_masks_win)
        try:
            self.lbl_status.config(text=f"Theme: {theme}")
        except Exception:
            pass

    def _on_am_theme_change(self, *_a):
        """Live switch for the All-Masks window's own theme."""
        theme = self.am_theme_var.get()
        if theme not in ("dark", "light"):
            return
        self.am_theme = theme
        self._save_pref("all_masks_theme", theme)
        if (getattr(self, "all_masks_win", None)
                and self.all_masks_win.winfo_exists()):
            self._style_all_masks(self.all_masks_win)
        try:
            self.lbl_status.config(text=f"All-Masks theme: {theme}")
        except Exception:
            pass

    # -- UI font tuning -------------------------------------------------
    def _refont_one(self, w):
        """Apply the font offset / bold setting to one widget. The
        widget's original font is cached on first touch so repeated
        calls stay idempotent (offsets never compound)."""
        try:
            cur = w.cget("font")
        except Exception:
            return
        if not cur:
            return
        orig = getattr(w, "_orig_font", None)
        if orig is None:
            try:
                import tkinter.font as _tkf
                f = _tkf.Font(font=cur)
                orig = (f.actual("family"), int(f.actual("size")),
                        f.actual("weight"), f.actual("slant"))
            except Exception:
                return
            w._orig_font = orig
        fam, size, weight, slant = orig
        base = abs(size) if size else 9
        new_size = max(6, base + int(getattr(self, "_ui_font_offset", 0)))
        styles = [weight if getattr(self, "_ui_bold", True) else "normal"]
        if slant == "italic":
            styles.append("italic")
        try:
            w.configure(font=(fam, new_size, *styles))
        except Exception:
            pass

    def _refont_tree(self, widget):
        """Recursively apply the font settings to a widget subtree
        (separate Toplevels skipped — call directly to include one)."""
        try:
            self._refont_one(widget)
        except Exception:
            pass
        try:
            children = widget.winfo_children()
        except Exception:
            return
        for ch in children:
            if isinstance(ch, tk.Toplevel):
                continue
            self._refont_tree(ch)

    def _restyle_subtree(self, widget):
        """Apply the active theme + font settings to a freshly built
        subtree (used by the pipeline / branch rebuild paths)."""
        if getattr(self, "ui_theme", "dark") != "dark":
            self._retheme_tree(widget)
        if (int(getattr(self, "_ui_font_offset", 0)) != 0
                or not getattr(self, "_ui_bold", True)):
            self._refont_tree(widget)

    def _on_font_setting(self, *_a):
        """Live-apply a font-size / bold change from the Settings UI."""
        try:
            self._ui_font_offset = int(self.set_font_offset.get())
        except Exception:
            self._ui_font_offset = 0
        self._ui_bold = bool(self.set_ui_bold.get())
        self._save_pref("ui_font_offset", self._ui_font_offset)
        self._save_pref("ui_bold", self._ui_bold)
        for _w in (self.root,
                   getattr(self, "settings_win", None),
                   getattr(self, "all_masks_win", None)):
            if _w is not None and _w.winfo_exists():
                self._refont_tree(_w)

    def _on_layout_setting(self, key, var):
        """Persist a minor layout setting and push it into the live
        module globals. FC_W/FC_H/STEPS_PER_ROW take effect the next
        time the All-Masks window opens; DISPLAY_W/H need a restart."""
        try:
            val = int(var.get())
        except Exception:
            return
        if val < 1:
            return
        self._save_pref(key, val)
        if key == "fc_w":
            self._fcm.FC_W = val
        elif key == "fc_h":
            self._fcm.FC_H = val
        elif key == "steps_per_row":
            self._fcm.STEPS_PER_ROW = val
        elif key == "display_w":
            global DISPLAY_W
            DISPLAY_W = val
        elif key == "display_h":
            global DISPLAY_H
            DISPLAY_H = val
        try:
            self.lbl_status.config(text=f"{key} = {val}  (saved)")
        except Exception:
            pass

    def _open_settings_window(self):
        """Open the Settings window — the one place to configure the
        major program settings (appearance, folders, YOLO model).
        Re-uses the existing app variables, so edits here take effect
        exactly like the in-panel controls did."""
        if (getattr(self, "settings_win", None)
                and self.settings_win.winfo_exists()):
            self.settings_win.lift()
            self.settings_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=UI_BG)
        win.resizable(False, False)
        self.settings_win = win

        def _section(title, color):
            tk.Label(win, text=title, fg=color,
                     font=("DejaVu Sans", 11, "bold"), anchor="w"
                     ).pack(fill="x", padx=10, pady=(12, 2))

        def _path_row(label, var, cmd):
            r = tk.Frame(win)
            r.pack(fill="x", padx=16, pady=2)
            tk.Label(r, text=label, width=13, anchor="w",
                     font=("DejaVu Sans", 9)).pack(side="left")
            tk.Button(r, text="Browse...", font=("DejaVu Sans", 8),
                      command=cmd).pack(side="left")
            tk.Label(r, textvariable=var, font=("DejaVu Sans", 8),
                     fg=UI_FG_MUTED, anchor="w", wraplength=340,
                     justify="left").pack(side="left", padx=6)

        def _int_row(label, var, lo, hi, note=""):
            r = tk.Frame(win)
            r.pack(fill="x", padx=16, pady=2)
            tk.Label(r, text=label, width=13, anchor="w",
                     font=("DejaVu Sans", 9)).pack(side="left")
            tk.Spinbox(r, textvariable=var, from_=lo, to=hi, width=6,
                       font=("DejaVu Sans", 9)).pack(side="left")
            if note:
                tk.Label(r, text=note, font=("DejaVu Sans", 8, "italic"),
                         fg=UI_FG_MUTED).pack(side="left", padx=6)

        # -- Appearance --
        _section("Appearance", "#aaccff")
        _ar = tk.Frame(win)
        _ar.pack(fill="x", padx=16, pady=2)
        tk.Label(_ar, text="Theme:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        ttk.Combobox(_ar, textvariable=self.ui_theme_var,
                     values=["dark", "light"], width=10,
                     state="readonly",
                     font=("DejaVu Sans", 9)).pack(side="left")
        tk.Label(_ar, text="(main program — instant)",
                 font=("DejaVu Sans", 8, "italic"),
                 fg=UI_FG_MUTED).pack(side="left", padx=6)
        _ar2 = tk.Frame(win)
        _ar2.pack(fill="x", padx=16, pady=2)
        tk.Label(_ar2, text="All-Masks theme:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        ttk.Combobox(_ar2, textvariable=self.am_theme_var,
                     values=["dark", "light"], width=10,
                     state="readonly",
                     font=("DejaVu Sans", 9)).pack(side="left")
        tk.Label(_ar2, text="(flowchart window — separate)",
                 font=("DejaVu Sans", 8, "italic"),
                 fg=UI_FG_MUTED).pack(side="left", padx=6)

        # -- Text (font size / boldness) --
        _section("Text", "#aaccff")
        _int_row("Font size +/-:", self.set_font_offset, -2, 8,
                 "bump every label up/down")
        _tr = tk.Frame(win)
        _tr.pack(fill="x", padx=16, pady=2)
        tk.Label(_tr, text="Bold text:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        tk.Checkbutton(_tr, text="use bold labels",
                       variable=self.set_ui_bold,
                       font=("DejaVu Sans", 9)).pack(side="left")
        tk.Label(_tr, text="(off = lighter, easier to read)",
                 font=("DejaVu Sans", 8, "italic"),
                 fg=UI_FG_MUTED).pack(side="left", padx=6)

        # -- Step cards per row --
        _section("Step layout", "#44cc66")
        _int_row("Morph / row:", self.set_morph_per_row, 1, 12,
                 "morph cards before wrapping")
        _int_row("PM / row:",    self.set_pm_per_row, 1, 12,
                 "pre-morph cards before wrapping")
        tk.Label(win,
                 text="Max step cards in a row before they wrap "
                      "(applies to RGB / IR and branches).",
                 font=("DejaVu Sans", 8, "italic"), fg=UI_FG_MUTED,
                 wraplength=360, justify="left"
                 ).pack(fill="x", padx=16, pady=(2, 0))

        # -- Folders --
        _section("Folders", "#88dd88")
        _path_row("Video folder:", self.video_root_var,
                  self._pick_video_root_dir)
        _path_row("Recordings:", self.recording_dir_var,
                  self._pick_recording_dir)
        _path_row("Screenshots:", self.screenshot_dir_var,
                  self._pick_screenshot_dir)

        # -- YOLO model --
        _section("YOLO model", "#ffaa66")
        _yr = tk.Frame(win)
        _yr.pack(fill="x", padx=16, pady=2)
        tk.Label(_yr, text="Model file:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        tk.Button(_yr, text="Load model...", font=("DejaVu Sans", 8),
                  command=self._pick_yolo_model_file).pack(side="left")
        tk.Label(_yr, textvariable=self.yolo_model_path_var,
                 font=("DejaVu Sans", 8), fg=UI_FG_MUTED, anchor="w",
                 wraplength=340, justify="left").pack(side="left", padx=6)

        # -- Display & Layout --
        _section("Display & Layout", "#ffdd66")
        _int_row("Panel width:",  self.set_disp_w, 120, 1280,
                 "video panel px  (restart)")
        _int_row("Panel height:", self.set_disp_h, 90, 960,
                 "video panel px  (restart)")
        _int_row("Flowchart W:",  self.set_fc_w, 60, 480,
                 "All-Masks thumb px")
        _int_row("Flowchart H:",  self.set_fc_h, 45, 360,
                 "All-Masks thumb px")
        _int_row("Steps / row:",  self.set_steps_row, 2, 20,
                 "All-Masks wrap")
        tk.Label(win,
                 text="Flowchart values apply next time the "
                      "All-Masks window opens.",
                 font=("DejaVu Sans", 8, "italic"), fg=UI_FG_MUTED,
                 wraplength=360, justify="left"
                 ).pack(fill="x", padx=16, pady=(2, 0))

        tk.Button(win, text="Close", font=("DejaVu Sans", 9),
                  command=win.destroy).pack(pady=12)

        # Match the active theme + font settings.
        self._restyle_subtree(win)

    def _pick_screenshot_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.screenshot_dir_var.get(),
                                    title="Choose screenshot save folder")
        if d and self._confirm_path(d, "Confirm screenshot folder",
                                    "Save screenshots to"):
            self.screenshot_dir_var.set(d)
            self._save_pref("screenshot_dir", d)

    def _pick_recording_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.recording_dir_var.get(),
                                    title="Choose recording save folder")
        if d and self._confirm_path(d, "Confirm recording folder",
                                    "Save recordings to"):
            self.recording_dir_var.set(d)
            self._save_pref("recording_dir", d)

    # -- Mask colour helpers ---------------------------------------
    @staticmethod
    def _hex_to_bgr(hex_str, fallback=(255, 255, 0)):
        """Parse a #RRGGBB / #RGB / RRGGBB hex string into a BGR
        tuple; return `fallback` on failure."""
        hx = (hex_str or "").strip().lstrip("#")
        if len(hx) == 3:
            hx = "".join(c * 2 for c in hx)
        try:
            r = int(hx[0:2], 16)
            g = int(hx[2:4], 16)
            b = int(hx[4:6], 16)
            return (b, g, r)
        except (ValueError, IndexError):
            return fallback

    def _yolo_mask_bgr(self):
        """User-chosen YOLO colour (used for both box edges and mask
        outlines). Cyan fallback."""
        return self._hex_to_bgr(self.yolo_mask_color_hex.get(),
                                fallback=(255, 255, 0))

    def _rgb_mask_bgr(self):
        """User-chosen RGB-pipeline mask colour. Red fallback."""
        return self._hex_to_bgr(
            getattr(self, "rgb_mask_color_hex",
                    tk.StringVar(value="#ff0000")).get(),
            fallback=(0, 0, 255))

    def _ir_mask_bgr(self):
        """User-chosen IR-pipeline mask colour. Yellow fallback."""
        return self._hex_to_bgr(
            getattr(self, "ir_mask_color_hex",
                    tk.StringVar(value="#ffff00")).get(),
            fallback=(0, 255, 255))

    def _make_color_picker_row(self, parent, label_text, string_var,
                               pref_key, picker_title):
        """Create a [Label][swatch][hex entry][Pick...] row, wire the
        swatch to live-update on var changes, and persist edits."""
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(2, 0))
        tk.Label(row, text=label_text,
                 font=("DejaVu Sans", 8)).pack(side="left")
        try:
            _bg_init = string_var.get()
        except Exception:
            _bg_init = "#ffffff"
        swatch = tk.Label(row, text="    ", width=3,
                          relief="solid", bd=1, bg=_bg_init)
        swatch.pack(side="left", padx=(4, 4))
        tk.Entry(row, textvariable=string_var, width=9,
                 font=("DejaVu Sans", 8)).pack(side="left")
        tk.Button(row, text="Pick...", font=("DejaVu Sans", 8),
                  command=lambda: self._pick_color_for(
                      string_var, picker_title)
                  ).pack(side="left", padx=(4, 0))

        def _on_change(*_a):
            hx = string_var.get().strip()
            if not hx.startswith("#"):
                hx = "#" + hx
            try:
                swatch.config(bg=hx)
                if pref_key:
                    self._save_pref(pref_key, hx)
            except tk.TclError:
                pass  # invalid hex, swatch left as-is
        string_var.trace_add("write", _on_change)
        return row, swatch

    def _pick_color_for(self, string_var, title):
        """Open a colour-wheel picker. Click anywhere on the wheel
        to set Hue (angle) and Saturation (radius); the Value (brightness)
        slider sits beneath. A row of famous-colour presets is at the
        bottom. Writes the chosen #RRGGBB back to `string_var`."""
        import colorsys

        cur = (string_var.get() or "#ffffff").strip()
        bgr_init = self._hex_to_bgr(cur, fallback=(255, 255, 255))
        b0, g0, r0 = bgr_init
        h0, s0, v0 = colorsys.rgb_to_hsv(r0 / 255, g0 / 255, b0 / 255)

        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)

        WHEEL = 220                # diameter
        R     = WHEEL // 2         # radius
        v_var   = tk.IntVar(value=int(round(v0 * 100)))
        hex_var = tk.StringVar(
            value=cur if cur.startswith("#") else "#" + cur)
        # Tracks the user-picked H/S in normalised [0,1].
        hs_state = {"h": float(h0), "s": float(s0)}

        # -- Build the colour-wheel image (BGR via cv2, max V) -----
        _yy, _xx = np.mgrid[0:WHEEL, 0:WHEEL].astype(np.float32)
        _dx = _xx - R
        _dy = _yy - R
        _dist = np.sqrt(_dx * _dx + _dy * _dy)
        _ang  = (np.arctan2(-_dy, _dx) + 2 * np.pi) % (2 * np.pi)
        _H    = (_ang / (2 * np.pi) * 179).astype(np.uint8)   # 0..179
        _S    = np.clip(_dist / R * 255, 0, 255).astype(np.uint8)
        _V    = np.full_like(_H, 255)
        _hsv  = np.dstack([_H, _S, _V])
        _bgr  = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)
        _alpha = (_dist <= R).astype(np.uint8) * 255
        # Outside the disc: paint white so it blends with the dialog.
        _bgr[_dist > R] = (255, 255, 255)
        _wheel_full = _bgr  # kept full-bright; V slider darkens preview only

        def _wheel_with_value(v_pct):
            """Apply Value (brightness) to the wheel pixels for display."""
            v = max(0, min(100, v_pct)) / 100.0
            out = (_wheel_full.astype(np.float32) * v).clip(0, 255
                                                            ).astype(np.uint8)
            out[_dist > R] = (255, 255, 255)  # keep outside white
            return out

        # Convert numpy BGR -> PhotoImage via PIL (already imported).
        from PIL import Image as _Image, ImageTk as _ImageTk
        _wheel_photo = {"img": None}
        def _make_wheel_photo(v_pct):
            arr = cv2.cvtColor(_wheel_with_value(v_pct), cv2.COLOR_BGR2RGB)
            pim = _Image.fromarray(arr)
            ph  = _ImageTk.PhotoImage(pim)
            _wheel_photo["img"] = ph        # keep ref; gc would blank it
            return ph

        # -- Layout -----------------------------------------------
        preview = tk.Label(win, text="", width=22, height=3,
                           relief="solid", bd=1)
        preview.pack(padx=10, pady=(10, 6), fill="x")

        canv = tk.Canvas(win, width=WHEEL, height=WHEEL,
                         highlightthickness=0)
        canv.pack(padx=10, pady=(0, 4))
        _wheel_id = canv.create_image(0, 0, anchor="nw",
                                      image=_make_wheel_photo(v_var.get()))
        # Crosshair marker showing the selected H/S.
        _xh1 = canv.create_line(0, 0, 0, 0, fill="#000", width=2)
        _xh2 = canv.create_line(0, 0, 0, 0, fill="#000", width=2)
        _xh3 = canv.create_line(0, 0, 0, 0, fill="#fff", width=1)
        _xh4 = canv.create_line(0, 0, 0, 0, fill="#fff", width=1)

        def _move_marker(h, s):
            ang = h * 2 * np.pi
            r   = max(0.0, min(1.0, s)) * R
            cx  = R + r * np.cos(ang)
            cy  = R - r * np.sin(ang)
            canv.coords(_xh1, cx - 8, cy, cx + 8, cy)
            canv.coords(_xh2, cx, cy - 8, cx, cy + 8)
            canv.coords(_xh3, cx - 8, cy, cx + 8, cy)
            canv.coords(_xh4, cx, cy - 8, cx, cy + 8)

        _suspend = {"hex": False}

        def _push_hex_from_state():
            hx = "#{:02x}{:02x}{:02x}".format(
                *(int(c * 255) for c in colorsys.hsv_to_rgb(
                    hs_state["h"], hs_state["s"], v_var.get() / 100.0)))
            _suspend["hex"] = True
            hex_var.set(hx)
            _suspend["hex"] = False
            try:
                preview.config(bg=hx)
            except tk.TclError:
                pass

        def _on_wheel_click(event):
            dx = event.x - R
            dy = -(event.y - R)
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > R:
                # Snap to wheel edge for clicks outside the disc.
                scale = R / max(dist, 1e-6)
                dx *= scale
                dy *= scale
                dist = R
            ang = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
            hs_state["h"] = float(ang / (2 * np.pi))
            hs_state["s"] = float(dist / R)
            _move_marker(hs_state["h"], hs_state["s"])
            _push_hex_from_state()
        canv.bind("<Button-1>",        _on_wheel_click)
        canv.bind("<B1-Motion>",       _on_wheel_click)

        # Value slider
        val_row = tk.Frame(win)
        val_row.pack(fill="x", padx=10, pady=2)
        tk.Label(val_row, text="V", width=2,
                 font=("DejaVu Sans", 9, "bold")).pack(side="left")
        def _on_v(*_a):
            canv.itemconfig(_wheel_id,
                            image=_make_wheel_photo(v_var.get()))
            _push_hex_from_state()
        tk.Scale(val_row, from_=0, to=100, orient="horizontal",
                 variable=v_var, length=180, showvalue=True,
                 command=lambda *_a: _on_v()
                 ).pack(side="left", fill="x", expand=True)

        # Hex entry — round-trips through hex too, so users can paste.
        hex_row = tk.Frame(win)
        hex_row.pack(fill="x", padx=10, pady=(2, 6))
        tk.Label(hex_row, text="Hex:",
                 font=("DejaVu Sans", 9)).pack(side="left")
        tk.Entry(hex_row, textvariable=hex_var, width=10,
                 font=("DejaVu Sans", 9)).pack(side="left", padx=(4, 0))

        def _on_hex_change(*_a):
            if _suspend["hex"]:
                return
            hx = hex_var.get().strip()
            if not hx.startswith("#"):
                hx = "#" + hx
            try:
                preview.config(bg=hx)
            except tk.TclError:
                return
            b, g, r = self._hex_to_bgr(hx, fallback=(0, 0, 0))
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            hs_state["h"] = float(h)
            hs_state["s"] = float(s)
            v_var.set(int(round(v * 100)))
            canv.itemconfig(_wheel_id,
                            image=_make_wheel_photo(v_var.get()))
            _move_marker(hs_state["h"], hs_state["s"])
        hex_var.trace_add("write", _on_hex_change)

        # Famous-colour presets
        tk.Label(win, text="Presets", font=("DejaVu Sans", 8, "bold"),
                 fg="#b0b0b0").pack(anchor="w", padx=10)
        presets = [
            ("red",     "#ff0000"),
            ("orange",  "#ffa500"),
            ("yellow",  "#ffff00"),
            ("green",   "#00ff00"),
            ("cyan",    "#00ffff"),
            ("blue",    "#0000ff"),
            ("magenta", "#ff00ff"),
            ("pink",    "#ff80c0"),
            ("white",   "#ffffff"),
            ("black",   "#000000"),
        ]
        pr_frm = tk.Frame(win)
        pr_frm.pack(padx=10, pady=(2, 6))
        for _i, (_nm, _hx) in enumerate(presets):
            tk.Button(pr_frm, text="  ", bg=_hx, width=3,
                      relief="raised", bd=1,
                      command=lambda h=_hx: hex_var.set(h)
                      ).grid(row=_i // 5, column=_i % 5, padx=2, pady=2)

        # OK / Cancel
        btns = tk.Frame(win)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        result = {"hex": None}
        def _ok():
            result["hex"] = hex_var.get().strip()
            if result["hex"] and not result["hex"].startswith("#"):
                result["hex"] = "#" + result["hex"]
            win.destroy()
        tk.Button(btns, text="OK", width=8,
                  command=_ok).pack(side="right", padx=2)
        tk.Button(btns, text="Cancel", width=8,
                  command=win.destroy).pack(side="right", padx=2)

        # Initial paint
        _move_marker(hs_state["h"], hs_state["s"])
        _push_hex_from_state()
        try:
            win.wait_window()
        except Exception:
            pass
        if result["hex"]:
            string_var.set(result["hex"])

    def _pick_yolo_mask_color(self):  # legacy alias
        self._pick_color_for(self.yolo_mask_color_hex, "Pick YOLO colour")

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
        if not self._confirm_path(path, "Confirm YOLO model",
                                  "Load YOLO model"):
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
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
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
                           font=("DejaVu Sans", 9),
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
    def _video_root_dir(self):
        """Active recordings root — user-selectable via Browse,
        falls back to the compile-time RECORDINGS_DIR."""
        root = getattr(self, "video_root_var", None)
        return root.get() if root else RECORDINGS_DIR

    def _pick_video_root_dir(self):
        """Open a directory chooser so the user can point at a custom
        recordings folder (videos elsewhere on disk)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self._video_root_dir(),
                                    title="Pick recordings root")
        if not d:
            return
        if not self._confirm_path(d, "Confirm recordings root",
                                  "Read recordings from"):
            return
        self.video_root_var.set(d)
        self._save_pref("video_root_dir", d)
        self._populate_folders()

    def _populate_folders(self):
        root = self._video_root_dir()
        if not os.path.exists(root):
            self.lbl_status.config(
                text=f"Recordings dir not found: {root}", fg="red")
            return
        folders = sorted(
            [d for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))],
            reverse=True)
        self.folder_cb["values"] = folders
        if folders:
            self.folder_cb.set(folders[0])

    def _load(self):
        name = self.folder_var.get()
        if not name:
            return
        path  = os.path.join(self._video_root_dir(), name)
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
        # Mark the current source signatures as fresh so the next
        # frame's source check doesn't immediately tear them down.
        self._backSub_rgb_src_sig = self.bgsub_rgb_src.get()
        self._backSub_ir_src_sig  = self.bgsub_ir_src.get()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _toggle_record(self):
        if self.recording: self._stop_record()
        else:              self._start_record()

    def _start_record(self):
        # Subfolder name = the LOADED VIDEO FOLDER name (e.g. if the
        # user loaded "test1", recordings go to <rec_dir>/test1/).
        # If the same folder name already exists in the recording
        # destination, append a "_N" counter so we never overwrite a
        # previous recording. If no folder is loaded, fall back to a
        # timestamp-named folder.
        base_dir = (self.recording_dir_var.get()
                    or os.path.join(os.getcwd(), "analysis_recordings"))
        loaded_name = (self.folder_var.get() or "").strip()
        if not loaded_name:
            loaded_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitise — strip path separators just in case.
        loaded_name = (loaded_name.replace("/", "_")
                                    .replace("\\", "_")
                                    .replace(" ", "_"))

        candidate = loaded_name
        out_dir   = os.path.join(base_dir, candidate)
        suffix    = 1
        while os.path.isdir(out_dir):
            suffix += 1
            candidate = f"{loaded_name}_{suffix}"
            out_dir   = os.path.join(base_dir, candidate)

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            # Fallback: if base_dir is unwritable, drop into cwd.
            base_dir = os.path.join(os.getcwd(), "analysis_recordings")
            candidate = loaded_name
            out_dir   = os.path.join(base_dir, candidate)
            suffix    = 1
            while os.path.isdir(out_dir):
                suffix += 1
                candidate = f"{loaded_name}_{suffix}"
                out_dir   = os.path.join(base_dir, candidate)
            os.makedirs(out_dir, exist_ok=True)

        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        fps, sz = 30.0, (640, 480)
        self.writers = {
            k: cv2.VideoWriter(os.path.join(out_dir, f"{k}.mp4"), fourcc, fps, sz)
            for k in ("rgb_raw", "rgb_det", "rgb_mask", "ir_raw", "ir_det", "ir_mask")
        }
        self.rec_dir   = out_dir
        self.recording = True
        self.btn_rec.config(text="■ Stop", bg="red")
        self.lbl_rec_status.config(text=f"● Recording -> {out_dir}/")

    def _stop_record(self):
        self.recording = False
        for w in self.writers.values(): w.release()
        self.writers = {}
        self.btn_rec.config(text="● Rec", bg="darkred")
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
        irt       = self.sv["IR_thresh"].get()
        mna       = self.sv["Min_area"].get()
        xy_kshape = KERNEL_SHAPES.get(self.kernel_shape_var.get(), cv2.MORPH_RECT)
        conf_thr  = self.sv["YOLO_Conf"].get() / 100.0

        # view_lookup is referenced by the pre-morph stage below (so
        # pre-morph step results can be used as OVERLAY / combine
        # sources later in the morph stage). Initialise it early.
        view_lookup = {}

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
        # Colormap variants — easier to eyeball channel structure;
        # H uses HSV cmap (hue->hue), S/V use JET (low->blue, high->red).
        _early_hsv_H_cmap = cv2.applyColorMap(_early_h_scaled, cv2.COLORMAP_HSV)
        _early_hsv_S_cmap = cv2.applyColorMap(hsv[:, :, 1],    cv2.COLORMAP_JET)
        _early_hsv_V_cmap = cv2.applyColorMap(hsv[:, :, 2],    cv2.COLORMAP_JET)
        # Paired views: channel grayscale on top, colormap below.
        # Each half stays at the full source resolution so the
        # All-Masks pair tile shows both images at the same physical
        # size as any other (single-image) thumbnail. The combined
        # image is therefore 2H tall — the flowchart gives pair tiles
        # twice the vertical space accordingly.
        def _hsv_pair(ch_bgr, cmap_bgr):
            return np.vstack([ch_bgr, cmap_bgr])
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
                res = self.yolo_model(base, verbose=False, conf=_conf, device=_TORCH_DEVICE)[0]
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
            res_rgb = self.yolo_model(f_rgb, verbose=False, conf=conf_thr, device=_TORCH_DEVICE)[0]
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
            res_ir = self.yolo_model(ir_bgr, verbose=False, conf=conf_thr, device=_TORCH_DEVICE)[0]
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
        # Colormap variants alongside the grayscale ones.
        _rgb_hsv_H_cmap = cv2.applyColorMap(_h_scaled,    cv2.COLORMAP_HSV)
        _rgb_hsv_S_cmap = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_JET)
        _rgb_hsv_V_cmap = cv2.applyColorMap(hsv[:, :, 2], cv2.COLORMAP_JET)
        # Paired view: channel grayscale (top half) + colormap (bottom
        # half) in ONE thumbnail. Each half is resized to half-height
        # so the combined image's aspect matches the source — the
        # thumbnail resize in the All-Masks loop won't squish it.
        def _hsv_pair(ch_bgr, cmap_bgr):
            h, w = ch_bgr.shape[:2]
            half = max(1, h // 2)
            a = cv2.resize(ch_bgr,   (w, half), interpolation=cv2.INTER_AREA)
            b = cv2.resize(cmap_bgr, (w, half), interpolation=cv2.INTER_AREA)
            return np.vstack([a, b])
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

    def _put(self, label, img, bgr=False, size=(DISPLAY_W, DISPLAY_H)):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != size[0] or img.shape[0] != size[1]:
            # Pick interpolation by direction:
            #   downscale  -> INTER_AREA (best anti-alias, no moire)
            #   upscale    -> INTER_NEAREST (binary mask pixels stay crisp;
            #                 natural images look slightly blocky but
            #                 sharp — better than INTER_LINEAR for masks)
            if size[0] * size[1] < img.shape[1] * img.shape[0]:
                _interp = cv2.INTER_AREA
            else:
                _interp = cv2.INTER_NEAREST
            img = cv2.resize(img, size, interpolation=_interp)
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
