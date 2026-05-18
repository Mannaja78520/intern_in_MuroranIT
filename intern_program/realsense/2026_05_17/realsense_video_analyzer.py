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

from config import (
    RECORDINGS_DIR, YOLO_MODEL_PATH, SCREENSHOTS_DIR,
    DISPLAY_W, DISPLAY_H,
    IR_COLORMAPS, MORPH_OPS, EXTRA_OPS, ALL_PROC_OPS, KERNEL_SHAPES,
    VIEW_OPTIONS,
    FC_W, FC_H,
    is_pre_morph_op,
    RGB_DETECT_MODES,
)
from pipeline_ui_mixin    import PipelineUIMixin
from flowchart_mixin       import FlowchartMixin
from user_pipelines_mixin  import UserPipelinesMixin
# Image processing + config-IO live in their own mixin modules so
# this file stays navigable. The torch / YOLO availability probe is
# owned by processing_mixin; re-import the flags it produced.
from processing_mixin import (ProcessingMixin, _TORCH_DEVICE,
                               _YOLO, _YOLO_AVAILABLE)
from config_io_mixin  import ConfigIOMixin
from export_mixin     import ExportMixin


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


class VideoAnalyzer(ProcessingMixin, ConfigIOMixin, ExportMixin,
                    PipelineUIMixin, FlowchartMixin, UserPipelinesMixin):
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

        # -- Performance (prefs-backed) -----------------------------
        # CPU threads -> OpenCV's parallel backend (filters / morph /
        # cvtColor / inRange). YOLO device -> CPU vs CUDA GPU.
        self._cpu_count = max(1, os.cpu_count() or 1)
        _threads = int(self._load_pref("cpu_threads", self._cpu_count))
        _threads = max(1, min(_threads, self._cpu_count))
        try:
            cv2.setNumThreads(_threads)
        except Exception:
            pass
        self._cpu_threads_cur = _threads
        self.set_cpu_threads = tk.IntVar(master=self.root, value=_threads)
        self.set_cpu_threads.trace_add("write", self._on_cpu_threads)
        _ydev = self._load_pref("yolo_device", "Auto")
        if _ydev not in ("Auto", "CPU", "GPU"):
            _ydev = "Auto"
        self.set_yolo_device = tk.StringVar(master=self.root, value=_ydev)
        self.set_yolo_device.trace_add("write", self._on_yolo_device)
        # Process scale: downscale frames before _process (faster,
        # lower quality). Frame skip: process every Nth frame during
        # playback so it keeps up with real time.
        _psc = self._load_pref("proc_scale", 100)
        try:
            _psc = max(25, min(100, int(_psc)))
        except Exception:
            _psc = 100
        self._proc_scale = _psc / 100.0
        self.set_proc_scale = tk.StringVar(master=self.root,
                                           value=f"{_psc}%")
        self.set_proc_scale.trace_add("write", self._on_proc_scale)
        _fsk = self._load_pref("frame_skip", 1)
        try:
            _fsk = max(1, min(10, int(_fsk)))
        except Exception:
            _fsk = 1
        self._frame_skip = _fsk
        self.set_frame_skip = tk.IntVar(master=self.root, value=_fsk)
        self.set_frame_skip.trace_add("write", self._on_frame_skip)

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
        # Default "auto": the BG-sub feed follows the active detection
        # mode (see _bgsub_auto_rgb) so it models the same signal the
        # detector thresholds without the user wiring it by hand.
        self.bgsub_rgb_src = tk.StringVar(value="auto")
        self.bgsub_ir_src  = tk.StringVar(value="auto")
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
        self._wire_live_refresh()
        self._bind_keys()
        self._populate_folders()
        self._apply_source_layout()
        # Apply the saved theme + font settings to everything the
        # build just created (build sites use dark-tuned literals
        # and explicit bold fonts).
        self._restyle_subtree(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._fit_initial_geometry()

    def _fit_initial_geometry(self):
        """Open the window at a size that fits on the screen. The
        config panels can be taller/wider than the display — the
        scrollable canvas handles the overflow. Without this the
        window opens at the full content size and spills off-screen,
        so only the top of the UI is visible."""
        try:
            self.root.update_idletasks()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            req_w = self.f.winfo_reqwidth()  + 24   # + scrollbar
            req_h = self.f.winfo_reqheight() + 24
            w = max(640, min(req_w, sw - 80))
            h = max(480, min(req_h, sh - 120))
            self.root.geometry(f"{w}x{h}+40+30")
            self.root.minsize(640, 480)
        except Exception:
            pass

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
        # -- Folder loader -- Browse + Load both work on the recording
        # folder picked here, so they sit together. The single-file
        # loaders (Image / RGB / IR) are a separate group further right
        # because they ignore the folder entirely.
        tk.Label(sel, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_cb  = ttk.Combobox(sel, textvariable=self.folder_var,
                                       width=20, state="readonly")
        self.folder_cb.pack(side="left", padx=4)
        tk.Button(sel, text="Load", command=self._load).pack(side="left")
        # Per-video folder selector — user can point the program at any
        # directory of recordings, not just the default RECORDINGS_DIR.
        self.video_root_var = tk.StringVar(
            value=self._load_pref("video_root_dir", RECORDINGS_DIR))
        tk.Button(sel, text="Browse...",
                  font=("DejaVu Sans", 8),
                  command=self._pick_video_root_dir
                  ).pack(side="left", padx=(4, 2))
        tk.Label(sel, textvariable=self.video_root_var,
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 width=18, anchor="w").pack(side="left")

        # -- Single-file loaders -- pick just an image, or just an RGB
        # / IR video file. These ignore the Folder / Browse selection;
        # a missing stream is mirrored so the pipeline still runs.
        # Loading one keeps any already-loaded other side, so two
        # unrelated files can be paired too.
        ttk.Separator(sel, orient="vertical").pack(side="left", fill="y",
                                                   padx=6, pady=2)
        # Single-file loaders collapsed into one menu so the toolbar
        # stays short — Image / RGB video / IR video.
        _load_mb = tk.Menubutton(sel, text="Load file ▾",
                                 font=("DejaVu Sans", 8),
                                 relief="raised", bd=1, padx=4)
        _load_menu = tk.Menu(_load_mb, tearoff=0)
        _load_menu.add_command(label="Load Image...",
                               command=self._load_image)
        _load_menu.add_command(label="Load RGB video...",
                               command=lambda: self._load_single("rgb"))
        _load_menu.add_command(label="Load IR video...",
                               command=lambda: self._load_single("ir"))
        _load_mb.config(menu=_load_menu)
        _load_mb.pack(side="left", padx=(0, 0))
        # Live indicator of which streams are active.
        self.src_label_var = tk.StringVar(value="no source")
        tk.Label(sel, textvariable=self.src_label_var,
                 font=("DejaVu Sans", 8, "bold"), fg="#88ccff"
                 ).pack(side="left", padx=(6, 0))
        # Use selector — pick which stream(s) to analyse. When a folder
        # has BOTH rgb.mp4 and ir.mp4, this switches between using both,
        # RGB only, or IR only WITHOUT reloading. The unused side is
        # mirrored (RGB-only feeds the IR side a grayscale copy; IR-only
        # mirrors to the RGB side) and the 6 panels / IR tab relabel to
        # match the choice.
        tk.Label(sel, text="Use:", font=("DejaVu Sans", 8)
                 ).pack(side="left", padx=(8, 0))
        self.src_use = tk.StringVar(value="Both")
        ttk.Combobox(sel, textvariable=self.src_use,
                     values=["Both", "RGB only", "IR only"],
                     width=9, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        self.src_use.trace_add("write", self._on_src_use_change)

        # Right side: only the primary action (Apply) and Settings stay
        # as direct buttons. Config load/save + pipeline export are
        # collapsed into one "File" menu so the toolbar isn't crammed.
        tk.Button(sel, text="Apply (F5)", bg="#225522", fg="white",
                  font=("DejaVu Sans", 8, "bold"),
                  command=self._apply_all
                  ).pack(side="right", padx=2)
        _file_mb = tk.Menubutton(sel, text="File ▾",
                                 font=("DejaVu Sans", 8, "bold"),
                                 bg="#3a2a1a", fg="white",
                                 relief="raised", bd=1, padx=4)
        _file_menu = tk.Menu(_file_mb, tearoff=0)
        _file_menu.add_command(label="Load config...",
                               command=self._load_config_dialog)
        _file_menu.add_command(label="Save config...",
                               command=self._save_config_dialog)
        _file_menu.add_separator()
        _file_menu.add_command(label="Export pipeline  (.py + .png)...",
                               command=self._export_pipeline)
        _file_mb.config(menu=_file_menu)
        _file_mb.pack(side="right", padx=2)
        tk.Button(sel, text="Settings",
                  font=("DejaVu Sans", 8, "bold"),
                  bg="#1a1a3a", fg="white",
                  command=self._open_settings_window
                  ).pack(side="right", padx=2)

        self.lbl_status = tk.Label(sel, text="- select a folder and click Load -",
                                   fg="#c8c8c8")
        self.lbl_status.pack(side="left", padx=8)

        # ---- 2x3 video panels ----
        vf = tk.Frame(p)
        vf.pack()
        self.panels       = {}
        self.panel_view   = {}
        self.panel_titles = {}   # key -> title Label (relabelled by mode)
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
            _title_lbl = tk.Label(frm, text=title,
                                  font=("DejaVu Sans", 9, "bold"))
            _title_lbl.pack()
            self.panel_titles[key] = _title_lbl
            # Two cascading dropdowns (pipeline group -> step/view)
            # instead of one long flat list. Step entries are built
            # from the LIVE pipeline lengths so there is no fixed
            # 20-step cap — adding more steps just shows more entries.
            self._make_cascading_picker(
                frm, v, [],
                label_from="Pipe:", label_step="View:",
                include_this_pipe=False)
            lbl = tk.Label(frm, bg="black")
            lbl.pack()
            # Seed with a black placeholder image so the label sizes in
            # PIXELS (DISPLAY_W x DISPLAY_H) right away. A tk.Label with
            # no image treats width/height as TEXT units (chars/lines),
            # so width=320,height=240 would make a giant panel and the
            # window would open far bigger than the screen.
            _ph = ImageTk.PhotoImage(
                Image.new("RGB", (DISPLAY_W, DISPLAY_H), "black"))
            lbl.imgtk = _ph
            lbl.configure(image=_ph)
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

        def add_param(parent, name, default, lo, hi, label=None):
            v   = tk.IntVar(value=default)
            row = tk.Frame(parent)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label or name, width=11, anchor="w",
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
            """Section header label inside an RGB / IR / Shared tab.
            Returns the label so callers can relabel it later."""
            lbl = tk.Label(parent, text=title, fg=color,
                           font=("DejaVu Sans", 10, "bold"),
                           anchor="w")
            lbl.pack(fill="x", padx=2, pady=(6, 1))
            return lbl

        # -- Notebook: RGB | IR | Shared -------------------------------
        self.main_nb = ttk.Notebook(pf)
        self.main_nb.pack(fill="x", padx=3, pady=3)
        rgb_tab    = tk.Frame(self.main_nb)
        ir_tab     = tk.Frame(self.main_nb)
        shared_tab = tk.Frame(self.main_nb)
        self.main_nb.add(rgb_tab,    text="RGB")
        self.main_nb.add(ir_tab,     text="IR")
        self.main_nb.add(shared_tab, text="Shared")

        # Per-side BG-Sub toggles. RGB and IR each have their own
        # enable flag and their own history / varTh params, so one
        # side can run BG subtraction while the other is off, each
        # tuned independently.
        self.use_bgsub_rgb = tk.BooleanVar(value=True)
        self.use_bgsub_ir  = tk.BooleanVar(value=True)
        # Feed options: full BGR frames, single HSV channels, or single
        # LAB channels (L = lightness, a = red<->green, b = yellow<->blue).
        _bgsrc_opts = ["auto", "rgb", "ir", "H", "S", "V", "L", "a", "b"]

        def _bgsub_block(parent, side):
            """Build the BG-Sub control block for the given side
            ('rgb' or 'ir'). The enable flag and the history / varTh
            params are independent per side — turning one off or
            retuning it does not affect the other."""
            use_var = (self.use_bgsub_rgb if side == "rgb"
                       else self.use_bgsub_ir)
            _bg_hdr = _hdr(parent,
                           f"BG Sub  -  {side.upper()}-side feed",
                           "#88dd88")
            _bg_chk = tk.Checkbutton(
                parent, text=f"Use BG Subtractor ({side.upper()})",
                variable=use_var, font=("DejaVu Sans", 8, "bold"))
            _bg_chk.pack(anchor="w")
            if side == "ir":
                # Stored so the IR tab can relabel them to 'Gray' when
                # only an RGB source is loaded.
                self._ir_bgsub_hdr = _bg_hdr
                self._ir_bgsub_chk = _bg_chk
            src_var = self.bgsub_rgb_src if side == "rgb" else self.bgsub_ir_src
            row = tk.Frame(parent); row.pack(fill="x", pady=1)
            tk.Label(row, text="Source:", font=("DejaVu Sans", 8),
                     width=11, anchor="w").pack(side="left")
            ttk.Combobox(row, textvariable=src_var,
                         values=_bgsrc_opts, width=6,
                         state="readonly",
                         font=("DejaVu Sans", 8)).pack(side="left", padx=2)
            tk.Label(parent,
                     text=("auto = follow the detection mode.   "
                           "rgb / ir = full BGR frame.   "
                           "H / S / V = HSV channel.   "
                           "L / a / b = LAB channel."),
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
                     wraplength=420, justify="left"
                     ).pack(anchor="w", padx=2, pady=(0, 2))
            # Per-side history / varTh params (BG_hist_<side> /
            # BG_var_<side>); the label keeps the short generic name.
            add_param(parent, f"BG_hist_{side}", 500, 10, 2000,
                      label="BG_hist")
            add_param(parent, f"BG_var_{side}",   50,  1,  200,
                      label="BG_var")
            tk.Button(parent, text="Reset BG Sub",
                      font=("DejaVu Sans", 8),
                      command=self._reset_bg).pack(anchor="w", pady=2)

        # ====================== RGB tab =============================
        _hdr(rgb_tab, "Detection", "#aaccff")
        # Detection mode — decides how the binary mask is built from
        # the HSV image. "Color" matches the Hue ranges (red etc.);
        # "Achromatic" / "Brightness" ignore Hue so white / gray /
        # black objects (which have no stable hue) can be detected.
        # The panel adapts to the chosen mode: rows a mode does not use
        # are hidden, and the description below is rewritten — see
        # _apply_rgb_mode_ui.
        _dm_row = tk.Frame(rgb_tab); _dm_row.pack(fill="x", pady=(2, 1))
        tk.Label(_dm_row, text="Mode", width=11, anchor="w",
                 font=("DejaVu Sans", 8)).pack(side="left")
        self.rgb_detect_mode = tk.StringVar(value="Color (HSV hue)")
        ttk.Combobox(_dm_row, textvariable=self.rgb_detect_mode,
                     values=list(RGB_DETECT_MODES.keys()),
                     width=21, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        # Mode description — text rewritten by _apply_rgb_mode_ui.
        self._rgb_mode_desc = tk.Label(
            rgb_tab, text="", font=("DejaVu Sans", 8, "italic"),
            fg="#c8c8c8", wraplength=420, justify="left")
        self._rgb_mode_desc.pack(anchor="w", padx=2, pady=(0, 2))

        # Detection-parameter widgets. Each entry is (widget,
        # pack-kwargs, {modes that show it}). _apply_rgb_mode_ui
        # re-packs only the widgets the active mode needs so the panel
        # changes with the mode. Section labels share their row's set.
        def _sec_lbl(text):
            return tk.Label(rgb_tab, text=text,
                            font=("DejaVu Sans", 9), fg="#c8c8c8")
        _hue1_lbl = _sec_lbl("-- Hue range 1 --")
        _h1l_row  = add_param(rgb_tab, "H1_low",  0,   0, 180)
        _h1h_row  = add_param(rgb_tab, "H1_high", 10,  0, 180)
        _hue2_lbl = _sec_lbl("-- Hue range 2 --")
        _h2l_row  = add_param(rgb_tab, "H2_low",  160, 0, 180)
        _h2h_row  = add_param(rgb_tab, "H2_high", 180, 0, 180)
        _sat_lbl  = _sec_lbl("-- Saturation --")
        _smin_row = add_param(rgb_tab, "S_min",   80,  0, 255)
        _smax_row = add_param(rgb_tab, "S_max",   255, 0, 255)
        _val_lbl  = _sec_lbl("-- Value (brightness) --")
        _vmin_row = add_param(rgb_tab, "V_min",   40,  0, 255)
        _vmax_row = add_param(rgb_tab, "V_max",   255, 0, 255)
        # LAB a*/b* rows — used by the 'LAB' and 'HSV hue + LAB' modes.
        # a* is red<->green (128 neutral; >128 = red), b* is
        # yellow<->blue. For red: raise A_min (~150). Defaults keep b*
        # wide so any red shade passes.
        _alab_lbl = _sec_lbl("-- LAB  a* (red <-> green) --")
        _amin_row = add_param(rgb_tab, "A_min", 150, 0, 255)
        _amax_row = add_param(rgb_tab, "A_max", 255, 0, 255)
        _blab_lbl = _sec_lbl("-- LAB  b* (yellow <-> blue) --")
        _bmin_row = add_param(rgb_tab, "B_min",   0, 0, 255)
        _bmax_row = add_param(rgb_tab, "B_max", 255, 0, 255)
        # Zero-height anchor: re-shown rows are packed *before* this so
        # they keep their place above the colour picker / BG-sub block.
        self._rgb_detect_anchor = tk.Frame(rgb_tab, height=0)
        self._rgb_detect_anchor.pack(fill="x")
        _ALL  = {"hue", "achroma", "value", "hsv_lab"}  # Value rows
        _HUE  = {"hue", "hsv_lab"}                      # Hue rows
        _HSAT = {"hue", "achroma", "hsv_lab"}           # Saturation rows
        _LABM = {"lab", "hsv_lab"}                      # LAB a*/b* rows
        _lbl_kw = {"anchor": "w"}
        _row_kw = {"fill": "x", "pady": 1}
        self._rgb_detect_widgets = [
            (_hue1_lbl, _lbl_kw, _HUE),  (_h1l_row,  _row_kw, _HUE),
            (_h1h_row,  _row_kw, _HUE),
            (_hue2_lbl, _lbl_kw, _HUE),  (_h2l_row,  _row_kw, _HUE),
            (_h2h_row,  _row_kw, _HUE),
            (_sat_lbl,  _lbl_kw, _HSAT), (_smin_row, _row_kw, _HSAT),
            (_smax_row, _row_kw, _HSAT),
            (_val_lbl,  _lbl_kw, _ALL),  (_vmin_row, _row_kw, _ALL),
            (_vmax_row, _row_kw, _ALL),
            (_alab_lbl, _lbl_kw, _LABM), (_amin_row, _row_kw, _LABM),
            (_amax_row, _row_kw, _LABM),
            (_blab_lbl, _lbl_kw, _LABM), (_bmin_row, _row_kw, _LABM),
            (_bmax_row, _row_kw, _LABM),
        ]
        self._apply_rgb_mode_ui()
        self.rgb_detect_mode.trace_add(
            "write",
            lambda *a: (self._apply_rgb_mode_ui(), self._refresh()))
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
        # The IR tab is reused as the grayscale-analysis tab when only
        # an RGB source is loaded — the SAME settings apply, so only
        # the wording changes. Every "IR ..." label here is registered
        # in self._ir_relabel as (widget, ir_text, gray_text);
        # _apply_source_layout swaps them when the mode changes.
        self._ir_tab = ir_tab
        self._ir_relabel = []

        def _ir_lbl(widget, gray_text):
            """Register a widget so its text flips to `gray_text` in
            RGB-only mode. The current text is kept as the IR-mode
            text."""
            try:
                self._ir_relabel.append(
                    (widget, widget.cget("text"), gray_text))
            except Exception:
                pass
            return widget

        self._ir_detect_hdr = _hdr(ir_tab, "Detection  -  IR threshold",
                                   "#44cc66")
        _ir_lbl(self._ir_detect_hdr, "Detection  -  Gray threshold")
        _ir_thr_lbl = tk.Label(ir_tab, text="-- IR Threshold --",
                               font=("DejaVu Sans", 9), fg="#c8c8c8")
        _ir_thr_lbl.pack(anchor="w")
        _ir_lbl(_ir_thr_lbl, "-- Gray Threshold --")
        _ir_thr_row = add_param(ir_tab, "IR_thresh", 100, 1, 254,
                                label="IR thresh")
        _ir_lbl(_ir_thr_row.winfo_children()[0], "Gray thresh")
        self.ir_mask_color_hex = tk.StringVar(
            value=self._load_pref("ir_mask_color", "#ffff00"))
        _ir_cp_row, _ = self._make_color_picker_row(
            ir_tab, "IR mask colour:", self.ir_mask_color_hex,
            "ir_mask_color", "Pick IR mask colour")
        _ir_lbl(_ir_cp_row.winfo_children()[0], "Gray mask colour:")

        # Dedicated IR CLAHE — applied to ir_gray BEFORE BG-sub /
        # threshold so the IR mask sees contrast-enhanced input.
        # Independent from any PM-step CLAHE (which also runs, after
        # this); the two stack if both are enabled.
        _ir_clahe_lbl = tk.Label(ir_tab, text="-- IR CLAHE --",
                                 font=("DejaVu Sans", 9), fg="#c8c8c8")
        _ir_clahe_lbl.pack(anchor="w")
        _ir_lbl(_ir_clahe_lbl, "-- Gray CLAHE --")
        _ir_clahe_chk = tk.Checkbutton(ir_tab, text="Use IR CLAHE",
                                       variable=self.use_clahe_ir,
                                       font=("DejaVu Sans", 8))
        _ir_clahe_chk.pack(anchor="w")
        _ir_lbl(_ir_clahe_chk, "Use Gray CLAHE")
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

        _ir_disp_lbl = tk.Label(ir_tab, text="-- IR Display --",
                                font=("DejaVu Sans", 9), fg="#c8c8c8")
        _ir_disp_lbl.pack(anchor="w")
        _ir_lbl(_ir_disp_lbl, "-- Gray Display --")
        cmap_row = tk.Frame(ir_tab); cmap_row.pack(fill="x", pady=2)
        _ir_cmap_lbl = tk.Label(cmap_row, text="IR colour:",
                                font=("DejaVu Sans", 8))
        _ir_cmap_lbl.pack(side="left")
        _ir_lbl(_ir_cmap_lbl, "Gray colour:")
        self.ir_cmap_var = tk.StringVar(value="Gray")
        ttk.Combobox(cmap_row, textvariable=self.ir_cmap_var,
                     values=list(IR_COLORMAPS.keys()),
                     width=9, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)

        _bgsub_block(ir_tab, "ir")
        _ir_lbl(self._ir_bgsub_hdr, "BG Sub  -  Gray-side feed")
        _ir_lbl(self._ir_bgsub_chk, "Use BG Subtractor (Gray)")

        # IR PM + Morph steps frames.
        self.ir_pre_pip_frame = tk.LabelFrame(
            ir_tab,
            text="  [PM]  IR Pre-morph Steps  (on gray, BEFORE threshold)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#88dd88",
            bd=0, relief="flat", padx=3, pady=2)
        self.ir_pre_pip_frame.pack(fill="x", padx=3, pady=(8, 2))
        _ir_lbl(self.ir_pre_pip_frame,
                "  [PM]  Gray Pre-morph Steps  (BEFORE threshold)  ")
        self.ir_pip_frame = tk.LabelFrame(
            ir_tab,
            text="  [Morph]  IR Morph Steps  (on mask, AFTER threshold)  ",
            font=("DejaVu Sans", 10, "bold"), fg="#44cc66",
            bd=0, relief="flat", padx=3, pady=2)
        self.ir_pip_frame.pack(fill="x", padx=3, pady=2)
        _ir_lbl(self.ir_pip_frame,
                "  [Morph]  Gray Morph Steps  (AFTER threshold)  ")

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

    def _apply_rgb_mode_ui(self, *_):
        """Adapt the RGB Detection panel to the selected mode: show
        only the parameter rows that mode uses and rewrite the mode
        description. Hidden rows keep their values — they are just not
        displayed. Called on startup, on mode change, and after a
        config load sets the mode."""
        from config import RGB_DETECT_MODES
        mode = RGB_DETECT_MODES.get(self.rgb_detect_mode.get(), "hue")
        _descs = {
            "hue": ("Color  -  detects saturated colours by Hue. Tune "
                    "Hue 1/2 to the target colour and S/V to its "
                    "strength and brightness. White / gray / black "
                    "have no hue - pick another mode for those."),
            "achroma": ("Achromatic  -  Hue is ignored. Lower S_max "
                        "(~0-40) so only desaturated pixels pass, then "
                        "set the V band: high = white, mid = gray, "
                        "low = black."),
            "value": ("Brightness  -  Hue and Saturation are ignored. "
                      "Only the V band is kept; raise V_min to isolate "
                      "bright / white objects."),
            "lab": ("LAB a*b*  -  HSV ignored. a* is red<->green "
                    "(>128 = red), b* is yellow<->blue. White / gray / "
                    "black all sit near 128, so a coloured object is "
                    "found on ANY-brightness background. For red, raise "
                    "A_min (~150+)."),
            "hsv_lab": ("HSV + LAB  -  runs the Hue detector AND the "
                        "LAB a*/b* detector and keeps only pixels in "
                        "BOTH. Most selective - rejects a bright white "
                        "floor that leaks past the Hue test alone."),
        }
        try:
            self._rgb_mode_desc.config(text=_descs.get(mode, ""))
        except Exception:
            pass
        for w, kw, modes in getattr(self, "_rgb_detect_widgets", []):
            try:
                w.pack_forget()
                if mode in modes:
                    w.pack(before=self._rgb_detect_anchor, **kw)
            except Exception:
                pass

    def _wire_live_refresh(self):
        """Make every main detection control re-process the current
        frame the moment it changes. _process re-reads these vars each
        frame during playback, but with a paused video — and above all
        in IMAGE mode, where there is only one frame and no playback
        loop — nothing would update without this. Covers all add_param
        sliders/spinboxes (HSV, IR threshold, BG, YOLO, Min area) plus
        the IR CLAHE, BG-sub and display controls."""
        live = list(self.sv.values())
        for _name in ("use_clahe_ir", "clahe_ir_clip", "clahe_ir_tile",
                      "use_bgsub_rgb", "use_bgsub_ir",
                      "bgsub_rgb_src", "bgsub_ir_src",
                      "ir_cmap_var", "kernel_shape_var"):
            v = getattr(self, _name, None)
            if v is not None:
                live.append(v)
        for v in live:
            try:
                v.trace_add("write", lambda *a: self._refresh())
            except Exception:
                pass

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
        if theme == getattr(self, "ui_theme", None):
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
        if theme == getattr(self, "am_theme", None):
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
    @staticmethod
    def _parse_font_spec(cur):
        """Parse a font spec into (family, size, weight, slant) WITHOUT
        a Tk round-trip. Every widget in this app is created with an
        explicit ("Family", size, *styles) tuple, so the spec is either
        a tuple/list or a brace/space string Tk hands back from
        cget("font") — both parse locally. Only genuine *named* fonts
        (e.g. "TkDefaultFont") need the slow tkinter.font.Font lookup,
        and those are rare here. Returns None if it can't be parsed."""
        try:
            items = None
            if isinstance(cur, (tuple, list)):
                items = list(cur)
            else:
                s = str(cur).strip()
                if not s:
                    return None
                if s.startswith("{"):                # "{DejaVu Sans} 8 bold"
                    end = s.find("}")
                    items = [s[1:end]] + s[end + 1:].split()
                else:                                # "DejaVu Sans 8 bold"
                    parts = s.split()
                    size_idx = next(
                        (i for i, p in enumerate(parts)
                         if p.lstrip("-").isdigit()), None)
                    if size_idx is None:
                        items = None                 # named font -> fall back
                    else:
                        items = ([" ".join(parts[:size_idx])]
                                 + parts[size_idx:])
            if items:
                fam    = str(items[0])
                size   = int(items[1]) if len(items) > 1 else 9
                styles = [str(x).lower() for x in items[2:]]
                return (fam, size,
                        "bold" if "bold" in styles else "normal",
                        "italic" if "italic" in styles else "roman")
        except Exception:
            pass
        # Fallback: real Tk lookup (named fonts only).
        try:
            import tkinter.font as _tkf
            f = _tkf.Font(font=cur)
            return (f.actual("family"), int(f.actual("size")),
                    f.actual("weight"), f.actual("slant"))
        except Exception:
            return None

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
            orig = self._parse_font_spec(cur)
            if orig is None:
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
            _off = int(self.set_font_offset.get())
        except Exception:
            _off = 0
        _bold = bool(self.set_ui_bold.get())
        # Idempotent: tk.Spinbox fires its write trace on creation, so
        # skip the (expensive) whole-tree re-font when nothing changed.
        if (_off == getattr(self, "_ui_font_offset", 0)
                and _bold == getattr(self, "_ui_bold", True)):
            return
        self._ui_font_offset = _off
        self._ui_bold = _bold
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
        global DISPLAY_W, DISPLAY_H
        try:
            val = int(var.get())
        except Exception:
            return
        if val < 1:
            return
        # Idempotent (tk.Spinbox fires its write trace on creation).
        _cur = {"fc_w": self._fcm.FC_W, "fc_h": self._fcm.FC_H,
                "steps_per_row": self._fcm.STEPS_PER_ROW,
                "display_w": DISPLAY_W, "display_h": DISPLAY_H}.get(key)
        if val == _cur:
            return
        self._save_pref(key, val)
        if key == "fc_w":
            self._fcm.FC_W = val
        elif key == "fc_h":
            self._fcm.FC_H = val
        elif key == "steps_per_row":
            self._fcm.STEPS_PER_ROW = val
        elif key == "display_w":
            DISPLAY_W = val
        elif key == "display_h":
            DISPLAY_H = val
        try:
            self.lbl_status.config(text=f"{key} = {val}  (saved)")
        except Exception:
            pass

    # -- Performance -----------------------------------------------------
    def _on_cpu_threads(self, *_a):
        """Apply the OpenCV worker-thread count live."""
        try:
            n = int(self.set_cpu_threads.get())
        except Exception:
            return
        n = max(1, min(n, getattr(self, "_cpu_count", n)))
        if n == getattr(self, "_cpu_threads_cur", None):
            return
        self._cpu_threads_cur = n
        try:
            cv2.setNumThreads(n)
        except Exception:
            pass
        self._save_pref("cpu_threads", n)
        try:
            self.lbl_status.config(text=f"CPU threads: {n}")
        except Exception:
            pass

    def _on_yolo_device(self, *_a):
        """Persist the YOLO device choice (read live by _yolo_device)."""
        dev = self.set_yolo_device.get()
        if dev not in ("Auto", "CPU", "GPU"):
            return
        self._save_pref("yolo_device", dev)
        try:
            self.lbl_status.config(text=f"YOLO device: {dev}")
        except Exception:
            pass

    def _on_proc_scale(self, *_a):
        """Process-resolution scale: downscale frames before the
        pipeline runs. Lower = faster, lower output quality."""
        raw = self.set_proc_scale.get()
        try:
            pct = max(25, min(100, int(str(raw).rstrip("%").strip())))
        except Exception:
            return
        sc = pct / 100.0
        if abs(sc - getattr(self, "_proc_scale", 1.0)) < 1e-6:
            return
        self._proc_scale = sc
        self._save_pref("proc_scale", pct)
        # Re-process the current frame so the change is visible at once.
        self._refresh()
        try:
            self.lbl_status.config(text=f"Process scale: {pct}%")
        except Exception:
            pass

    def _on_frame_skip(self, *_a):
        """Frame-skip: process every Nth frame during playback."""
        try:
            n = max(1, min(10, int(self.set_frame_skip.get())))
        except Exception:
            return
        if n == getattr(self, "_frame_skip", 1):
            return
        self._frame_skip = n
        self._save_pref("frame_skip", n)
        try:
            self.lbl_status.config(
                text=(f"Frame skip: every {n} frames" if n > 1
                      else "Frame skip: off"))
        except Exception:
            pass

    def _set_max_performance(self):
        """One-click: all CPU cores + GPU (when a CUDA device exists)."""
        try:
            self.set_cpu_threads.set(getattr(self, "_cpu_count", 1))
        except Exception:
            pass
        self.set_yolo_device.set("GPU" if getattr(self, "GPU_AVAILABLE",
                                                  False) else "CPU")
        try:
            self.lbl_status.config(text="Performance: maxed")
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

        # -- Performance --
        _section("Performance", "#44cc66")
        _int_row("CPU threads:", self.set_cpu_threads, 1,
                 getattr(self, "_cpu_count", 8),
                 f"OpenCV workers (max {getattr(self, '_cpu_count', '?')})")
        _pr = tk.Frame(win)
        _pr.pack(fill="x", padx=16, pady=2)
        tk.Label(_pr, text="YOLO device:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        ttk.Combobox(_pr, textvariable=self.set_yolo_device,
                     values=["Auto", "CPU", "GPU"], width=8,
                     state="readonly",
                     font=("DejaVu Sans", 9)).pack(side="left")
        _gpu_txt = ("GPU detected" if getattr(self, "GPU_AVAILABLE", False)
                    else "no CUDA GPU - CPU only")
        tk.Label(_pr, text=f"({_gpu_txt})",
                 font=("DejaVu Sans", 8, "italic"),
                 fg=UI_FG_MUTED).pack(side="left", padx=6)
        # Process scale — downscale frames before the pipeline runs.
        _ps = tk.Frame(win)
        _ps.pack(fill="x", padx=16, pady=2)
        tk.Label(_ps, text="Process scale:", width=13, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        ttk.Combobox(_ps, textvariable=self.set_proc_scale,
                     values=["100%", "75%", "50%", "33%", "25%"],
                     width=8, state="readonly",
                     font=("DejaVu Sans", 9)).pack(side="left")
        tk.Label(_ps, text="(smaller = faster, lower quality)",
                 font=("DejaVu Sans", 8, "italic"),
                 fg=UI_FG_MUTED).pack(side="left", padx=6)
        _int_row("Frame skip:", self.set_frame_skip, 1, 10,
                 "process every Nth frame on playback")
        tk.Button(win, text="Use max performance",
                  font=("DejaVu Sans", 9, "bold"),
                  bg="#225522", fg="white",
                  command=self._set_max_performance
                  ).pack(padx=16, pady=(4, 0), anchor="w")
        tk.Label(win,
                 text="More CPU threads speed up the image pipeline; "
                      "GPU greatly speeds up YOLO when available.",
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

    # -- Source helpers ----------------------------------------------------
    def _primary_cap(self):
        """Capture used for seeking / frame-count. RGB if loaded, else
        IR. None when nothing is loaded."""
        return self.cap_rgb if self.cap_rgb is not None else self.cap_ir

    def _src_name(self):
        """Human label for which streams are active."""
        if self.cap_rgb is not None and self.cap_ir is not None:
            return "RGB + IR"
        if self.cap_rgb is not None:
            return "RGB only"
        if self.cap_ir is not None:
            return "IR only"
        return "no source"

    def _effective_src(self):
        """Which stream(s) to actually process — 'rgb', 'ir' or
        'both'. Combines the user's Use selector with what is loaded:
        if the chosen side is not loaded it falls back to whatever is.
        This is what drives reading, panel labels and the IR tab —
        NOT raw capture presence — so the Use selector takes effect
        even when both rgb.mp4 and ir.mp4 are loaded."""
        rgb = getattr(self, "cap_rgb", None) is not None
        ir  = getattr(self, "cap_ir",  None) is not None
        sel = "Both"
        try:
            sel = self.src_use.get()
        except Exception:
            pass
        if not rgb and not ir:
            # Nothing loaded — honour the selector intent so the panels
            # / IR tab reflect the choice before a video is opened.
            return {"RGB only": "rgb", "IR only": "ir"}.get(sel, "both")
        if sel == "RGB only" and rgb:
            return "rgb"
        if sel == "IR only" and ir:
            return "ir"
        if rgb and ir:
            return "both"
        return "rgb" if rgb else "ir"

    def _on_src_use_change(self, *_):
        """React to the Use selector: relabel panels / IR tab, then
        re-read the current frame so the switch is visible at once
        (the cached frame was captured under the old selection)."""
        self._apply_source_layout()
        if self._primary_cap() is not None and not self.playing:
            self._show_frame(self.current_frame)
        else:
            self._refresh()

    def _apply_source_layout(self):
        """Relabel the six panels (and the IR tab) to match the active
        source mode:
          RGB + IR  - panels keep 'RGB ...' / 'IR ...'.
          RGB only  - the IR side analyses a grayscale copy of the RGB
                      frame, so its panels become 'Gray ...' and the IR
                      tab / detection header read 'Gray'.
          IR only   - the RGB/HSV pipeline runs on the IR frame as a
                      second classifier, so the top panels become
                      'IR HSV ...' (IR has no colour to detect).
        Driven by _effective_src so the Use selector takes effect.
        Called on startup, on load, and when the Use selector changes."""
        rgb_on = getattr(self, "cap_rgb", None) is not None
        ir_on  = getattr(self, "cap_ir",  None) is not None
        mode = self._effective_src()
        # Panel / IR-tab labels always follow the selector; the source
        # indicator only shows a source name once something is loaded.
        if rgb_on or ir_on:
            self.src_label_var.set({"rgb":  "RGB only",
                                    "ir":   "IR only",
                                    "both": "RGB + IR"}[mode])
        titles = {
            "both": {"rgb_raw": "RGB (raw)",
                     "rgb_det": "RGB Detection",
                     "rgb_mask": "RGB Mask",
                     "ir_raw":  "IR (raw)",
                     "ir_det":  "IR Detection",
                     "ir_mask": "IR Mask"},
            "rgb":  {"rgb_raw": "RGB (raw)",
                     "rgb_det": "RGB Detection",
                     "rgb_mask": "RGB Mask",
                     "ir_raw":  "Gray (raw)",
                     "ir_det":  "Gray Detection",
                     "ir_mask": "Gray Mask"},
            "ir":   {"rgb_raw": "IR HSV (raw)",
                     "rgb_det": "IR HSV Detection",
                     "rgb_mask": "IR HSV Mask",
                     "ir_raw":  "IR (raw)",
                     "ir_det":  "IR Detection",
                     "ir_mask": "IR Mask"},
        }[mode]
        for key, lbl in getattr(self, "panel_titles", {}).items():
            try:
                lbl.config(text=titles.get(key, key))
            except Exception:
                pass
        # IR-tab wording: every registered 'IR ...' label flips to
        # 'Gray ...' when the IR side is really analysing a grayscale
        # copy of the RGB frame (RGB-only mode). The IR notebook tab
        # follows too.
        _gray = (mode == "rgb")
        for w, ir_t, gray_t in getattr(self, "_ir_relabel", []):
            try:
                w.config(text=gray_t if _gray else ir_t)
            except Exception:
                pass
        try:
            self.main_nb.tab(self._ir_tab,
                             text="Gray" if _gray else "IR")
        except Exception:
            pass

    def _read_pair(self):
        """Read one frame for processing, honouring the Use selector
        (_effective_src). A side that is disabled by the selector — or
        simply not loaded — is mirrored from the other side so
        _process always receives two BGR frames. A loaded-but-disabled
        capture is still grabbed (no decode) so both stay frame-synced
        for when the selector switches back. Returns (ok, f_rgb, f_ir)."""
        eff = self._effective_src()
        f_rgb = f_ir = None
        if self.cap_rgb is not None:
            if eff in ("rgb", "both"):
                ok, fr = self.cap_rgb.read()
                if ok and fr is not None:
                    f_rgb = fr
            else:
                self.cap_rgb.grab()          # keep synced, skip decode
        if self.cap_ir is not None:
            if eff in ("ir", "both"):
                ok, fi = self.cap_ir.read()
                if ok and fi is not None:
                    f_ir = fi
            else:
                self.cap_ir.grab()           # keep synced, skip decode
        if f_rgb is None and f_ir is None:
            return False, None, None
        if f_rgb is None:
            # IR-only: feed the RGB/HSV side a copy of the IR frame.
            f_rgb = f_ir.copy()
        if f_ir is None:
            # RGB-only: the IR side analyses a true grayscale version
            # of the RGB frame, so its raw / mask panels are genuinely
            # grayscale instead of the colour RGB frame mirrored over.
            _g = cv2.cvtColor(f_rgb, cv2.COLOR_BGR2GRAY)
            f_ir = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
        return True, f_rgb, f_ir

    def _seek_all(self, idx):
        """Seek every loaded capture to frame `idx`."""
        for cap in (self.cap_rgb, self.cap_ir):
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def _grab_all(self):
        """Grab (cheap, no decode) one frame on every loaded capture."""
        ok = True
        for cap in (self.cap_rgb, self.cap_ir):
            if cap is not None:
                ok = cap.grab() and ok
        return ok

    def _finalize_caps(self, label):
        """Common post-open work: frame count, slider, status, source
        indicator and first-frame display. Called after cap_rgb /
        cap_ir have been (re)assigned by _load or _load_single."""
        # Drop any capture that failed to open.
        if self.cap_rgb is not None and not self.cap_rgb.isOpened():
            self.cap_rgb.release(); self.cap_rgb = None
        if self.cap_ir is not None and not self.cap_ir.isOpened():
            self.cap_ir.release(); self.cap_ir = None
        if self._primary_cap() is None:
            self.src_label_var.set("no source")
            self.lbl_status.config(
                text=f"Could not open video(s): {label}", fg="red")
            return
        # Frame count = shortest loaded clip so neither side overruns.
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT))
                  for c in (self.cap_rgb, self.cap_ir) if c is not None]
        self.total_frames  = max(1, min(counts) if counts else 1)
        self.current_frame = 0
        self._updating_slider = True
        try:
            self.slider.config(to=max(1, self.total_frames - 1))
            self.frame_var.set(0)
        except Exception:
            pass
        self._updating_slider = False
        self.lbl_pos.config(text=f"0 / {self.total_frames - 1}")
        # Default the Use selector to match what was actually loaded —
        # both files -> "Both", a single file -> that side. The user
        # can then switch it without reloading.
        if self.cap_rgb is not None and self.cap_ir is not None:
            self.src_use.set("Both")
        elif self.cap_rgb is not None:
            self.src_use.set("RGB only")
        else:
            self.src_use.set("IR only")
        self._apply_source_layout()
        self.lbl_status.config(
            text=f"Loaded: {label}  [{self._src_name()}]  "
                 f"({self.total_frames} frames)", fg="green")
        self._reset_bg()
        self._show_frame(0)

    # Video file extensions recognised when scanning a folder.
    _VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def _open_pair(self, rgb_path, ir_path, label):
        """(Re)open the RGB / IR captures from explicit file paths
        (either may be None) and finalise. Shared by _load and the
        folder-video picker."""
        self._stop_play()
        if self.cap_rgb: self.cap_rgb.release()
        if self.cap_ir:  self.cap_ir.release()
        self.cap_rgb = cv2.VideoCapture(rgb_path) if rgb_path else None
        self.cap_ir  = cv2.VideoCapture(ir_path)  if ir_path  else None
        self._finalize_caps(label)

    def _load(self):
        """Load a recording folder. Uses rgb.mp4 / ir.mp4 when present;
        otherwise lists the folder's video files and lets the user
        assign which is the RGB / IR stream — so a folder whose clips
        are named e.g. 'video 1.mp4' can still be loaded."""
        name = self.folder_var.get()
        if not name:
            return
        path  = os.path.join(self._video_root_dir(), name)
        rgb_p = os.path.join(path, "rgb.mp4")
        ir_p  = os.path.join(path, "ir.mp4")
        have_rgb = os.path.exists(rgb_p)
        have_ir  = os.path.exists(ir_p)
        if have_rgb or have_ir:
            # Standard layout — load the canonically-named file(s).
            self._open_pair(rgb_p if have_rgb else None,
                            ir_p  if have_ir  else None, name)
            return
        # Non-standard names — scan for video files and let the user
        # pick which is RGB / IR.
        try:
            vids = sorted(
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in self._VIDEO_EXTS)
        except Exception:
            vids = []
        if not vids:
            self.lbl_status.config(
                text=f"No video files in {name}!", fg="red")
            return
        self._pick_folder_videos(path, name, vids)

    def _pick_folder_videos(self, folder_path, folder_name, videos):
        """Modal picker: assign which of the folder's video files are
        the RGB and IR streams (either can be '(none)'). Defaults are
        guessed from the file names."""
        NONE = "(none)"
        opts = [NONE] + videos

        def _match(kws):
            for v in videos:
                lv = v.lower()
                if any(k in lv for k in kws):
                    return v
            return None
        rgb_def = _match(("rgb", "color", "colour")) or videos[0]
        ir_def  = _match(("ir", "infra", "thermal", "depth"))
        if ir_def == rgb_def:
            ir_def = None

        win = tk.Toplevel(self.root)
        win.title(f"Select videos - {folder_name}")
        win.transient(self.root)
        win.resizable(False, False)
        tk.Label(win,
                 text=(f"'{folder_name}' has no rgb.mp4 / ir.mp4.\n"
                       "Choose which file is the RGB and the IR "
                       "stream (either may be left as (none)):"),
                 justify="left", font=("DejaVu Sans", 9)
                 ).pack(anchor="w", padx=12, pady=(12, 8))
        rgb_var = tk.StringVar(value=rgb_def or NONE)
        ir_var  = tk.StringVar(value=ir_def  or NONE)
        for lbl, var in (("RGB video:", rgb_var), ("IR video:", ir_var)):
            r = tk.Frame(win); r.pack(fill="x", padx=12, pady=3)
            tk.Label(r, text=lbl, width=10, anchor="w",
                     font=("DejaVu Sans", 9)).pack(side="left")
            ttk.Combobox(r, textvariable=var, values=opts,
                         width=34, state="readonly",
                         font=("DejaVu Sans", 9)).pack(side="left")
        btns = tk.Frame(win); btns.pack(fill="x", padx=12, pady=(10, 12))

        def _ok():
            rv, iv = rgb_var.get(), ir_var.get()
            rp = (os.path.join(folder_path, rv)
                  if rv and rv != NONE else None)
            ip = (os.path.join(folder_path, iv)
                  if iv and iv != NONE else None)
            if rp is None and ip is None:
                self.lbl_status.config(
                    text="Pick at least one video.", fg="red")
                return
            win.destroy()
            self._open_pair(rp, ip, folder_name)

        tk.Button(btns, text="Load", command=_ok,
                  bg="#225522", fg="white",
                  font=("DejaVu Sans", 9, "bold")).pack(side="right")
        tk.Button(btns, text="Cancel", command=win.destroy,
                  font=("DejaVu Sans", 9)).pack(side="right", padx=6)
        try:
            self._restyle_subtree(win)
        except Exception:
            pass
        win.grab_set()

    def _load_single(self, side):
        """Load a single video file into one side ('rgb' or 'ir').
        Any capture already loaded on the other side is kept, so two
        unrelated clips can be paired."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=f"Choose an {side.upper()} video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            self.lbl_status.config(
                text=f"Could not open video: {path}", fg="red")
            return
        self._stop_play()
        if side == "rgb":
            if self.cap_rgb: self.cap_rgb.release()
            self.cap_rgb = cap
        else:
            if self.cap_ir: self.cap_ir.release()
            self.cap_ir = cap
        self._finalize_caps(os.path.basename(path))

    def _load_image(self):
        """Analyse a single still image instead of a video pair.
        The picture is fed through the same pipeline as one static
        frame (RGB pipeline; the IR side sees a gray copy). Playback
        controls stay inert — there is only one frame."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Choose an image to analyse",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp "
                                   "*.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.lbl_status.config(
                text=f"Could not read image: {path}", fg="red")
            return
        self._stop_play()
        if self.cap_rgb:
            self.cap_rgb.release()
        if self.cap_ir:
            self.cap_ir.release()
        # No capture in image mode -> playback / slider stay disabled.
        self.cap_rgb = None
        self.cap_ir  = None
        self.total_frames  = 1
        self.current_frame = 0
        self._updating_slider = True
        try:
            self.slider.config(to=1)
            self.frame_var.set(0)
        except Exception:
            pass
        self._updating_slider = False
        self._last_f_rgb = img.copy()
        self._last_f_ir  = img.copy()
        self._reset_bg()
        try:
            self.lbl_pos.config(text="image")
        except Exception:
            pass
        self.src_label_var.set("Image")
        self._apply_source_layout()
        self.lbl_status.config(
            text=f"Loaded image: {os.path.basename(path)}  "
                 f"({img.shape[1]}x{img.shape[0]})", fg="green")
        self._process(img, img.copy())

    # ------------------------------------------------------------------
    # Background subtractor
    # ------------------------------------------------------------------
    def _reset_bg(self):
        # Each side rebuilds from its own history / varTh params.
        self.backSub_rgb = cv2.createBackgroundSubtractorMOG2(
            history=self.sv["BG_hist_rgb"].get(),
            varThreshold=self.sv["BG_var_rgb"].get(),
            detectShadows=False)
        self.backSub_ir  = cv2.createBackgroundSubtractorMOG2(
            history=self.sv["BG_hist_ir"].get(),
            varThreshold=self.sv["BG_var_ir"].get(),
            detectShadows=False)
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
        # The writer KEYS stay fixed (processing_mixin writes by them),
        # but the FILE NAMES follow the selected source so the saved
        # videos match the on-screen panel names: RGB-only writes the
        # bottom row as gray_*, IR-only writes the top row as ir_hsv_*.
        _keys  = ("rgb_raw", "rgb_det", "rgb_mask",
                  "ir_raw",  "ir_det",  "ir_mask")
        _fname = {
            "both": {k: k for k in _keys},
            "rgb":  {"rgb_raw": "rgb_raw",  "rgb_det": "rgb_det",
                     "rgb_mask": "rgb_mask",
                     "ir_raw": "gray_raw",  "ir_det": "gray_det",
                     "ir_mask": "gray_mask"},
            "ir":   {"rgb_raw": "ir_hsv_raw", "rgb_det": "ir_hsv_det",
                     "rgb_mask": "ir_hsv_mask",
                     "ir_raw": "ir_raw",    "ir_det": "ir_det",
                     "ir_mask": "ir_mask"},
        }[self._effective_src()]
        self.writers = {
            k: cv2.VideoWriter(
                os.path.join(out_dir, f"{_fname[k]}.mp4"),
                fourcc, fps, sz)
            for k in _keys
        }
        self.rec_dir   = out_dir
        self.recording = True
        self.btn_rec.config(text="■ Stop", bg="red")
        self.lbl_rec_status.config(
            text=f"● Recording [{self.src_label_var.get()}] "
                 f"-> {out_dir}/")

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
        if self._primary_cap() is None: return
        idx = max(0, min(idx, self.total_frames - 1))
        self._seek_all(idx)
        ok, f_rgb, f_ir = self._read_pair()
        if ok:
            self.current_frame = idx
            self._last_f_rgb = f_rgb.copy()
            self._last_f_ir  = f_ir.copy()
            self._updating_slider = True
            self.frame_var.set(idx)
            self.lbl_pos.config(text=f"{idx} / {self.total_frames - 1}")
            self._updating_slider = False
            self._process(f_rgb, f_ir)

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
        if self._primary_cap() is None: return
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
        if not self.playing or self._primary_cap() is None: return
        if self.current_frame >= self.total_frames - 1:
            if self.loop_var.get():
                self._seek_all(0)
                self.current_frame = 0
            else:
                self._stop_play(); return
        # Frame skip: grab (cheap, no decode-to-array) the in-between
        # frames so playback advances N frames per _process — keeps
        # it real-time when the pipeline is slower than the frame rate.
        _skip = max(1, int(getattr(self, "_frame_skip", 1)))
        for _ in range(_skip - 1):
            if self.current_frame >= self.total_frames - 1:
                break
            if self._grab_all():
                self.current_frame += 1
        ok, f_rgb, f_ir = self._read_pair()
        if ok:
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
        if self._updating_slider or self._primary_cap() is None: return
        idx = int(float(val))
        if not self.playing and idx != self.current_frame:
            self._show_frame(idx)

    def _step_back(self):
        if self._primary_cap() is None: return
        if self.playing: self._stop_play()
        self._show_frame(self.current_frame - 1)

    def _step_forward(self):
        if self._primary_cap() is None: return
        if self.playing: self._stop_play()
        self._show_frame(self.current_frame + 1)

    def _go_first(self):
        if self._primary_cap() is not None:
            self._stop_play(); self._show_frame(0)

    def _go_last(self):
        if self._primary_cap() is not None:
            self._stop_play(); self._show_frame(self.total_frames - 1)

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

