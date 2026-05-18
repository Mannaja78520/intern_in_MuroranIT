"""Stress test for the 2026_05_18 RealSense Cable Video Analyzer.

It pushes every recently-added feature at full size:

  1. MAXIMUM config — 20 morph + 20 pre-morph steps on the main RGB /
     IR pipelines and on 3 user branches (rgb / ir / custom).
  2. Combine tab — exercises the new cascading A/B picker, the
     OVERLAY op and the "none" default, plus Mask 1 / Mask 2 colours,
     and the save/load round-trip of those settings.
  3. Export selection — builds EVERY output the export dialog offers:
     <name>.py / <name>_lib.py / <name>_config.json / <name>.png /
     <name>.txt / <name>_README.md / <name>_examples/.
  4. The reusable library — loads it from the exported JSON and runs
     detect / final_mask / mask_overlay / mask_points / (yolo path).
  5. PNG row-wrap — a 20-step pipeline must wrap at 7 boxes/line so
     the PNG stays a sane width instead of growing without bound.
  6. YOLO box scale / pad — grow or shrink YOLO boxes, verified on a
     real RealSense frame; YOLO + mask images are exported into a
     detection/ folder (auto-versioned detection_V2, _V3 ...).

Run:  conda run -n intern_muroranIT_py312 python stress_test/stress_test.py
  or:  python stress_test/stress_test.py
"""
import os
import sys
import ast
import json
import subprocess
import importlib.util

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # the 2026_05_18 folder
sys.path.insert(0, PARENT)
os.environ.setdefault("DISPLAY", ":1")

import tkinter as tk
import numpy as np
import cv2
from realsense_video_analyzer import VideoAnalyzer

MAX = 20

MORPH_CYCLE = ["Close", "Open", "Dilate", "Erode", "Gradient",
               "TopHat", "BlackHat", "GaussBlur", "MedianBlur",
               "Sharpen", "Invert", "FillHoles", "Thresh_Otsu"]
PM_CYCLE    = ["GaussBlur", "MedianBlur", "BilateralBlur", "CLAHE",
               "Gamma", "Sharpen", "Normalize", "HistEq"]

ok = 0


def chk(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name)
    if cond:
        ok += 1
    else:
        raise SystemExit("FAILED: " + name)


def morph_step(i):
    """One morph step. Every 5th is a Combine (OR) step so the
    per-step combine path is exercised at size too."""
    if i % 5 == 4:
        return {"en": True, "op": "Dilate", "n": 1, "dir": "Both",
                "kx": 3, "ky": 3, "t": 0,
                "comb_en": True, "comb_op": "OR", "comb_src": "mask_pre"}
    return {"en": True, "op": MORPH_CYCLE[i % len(MORPH_CYCLE)],
            "n": 1, "dir": "Both", "kx": 3, "ky": 3, "t": 120}


def pm_step(i):
    return {"en": True, "op": PM_CYCLE[i % len(PM_CYCLE)],
            "n": 1, "dir": "XY", "kx": 5, "ky": 5, "t": 0}


def branch(name, btype, **extra):
    bd = {"name": name, "type": btype,
          "steps":     [morph_step(i) for i in range(MAX)],
          "pre_steps": [pm_step(i)    for i in range(MAX)]}
    bd.update(extra)
    return bd


print("=" * 64)
print(" STRESS TEST — 2026_05_18 analyzer  (combine + export + library)")
print("=" * 64)

# ----------------------------------------------------------------------
# 1) Build the analyzer + the MAXIMUM config.
# ----------------------------------------------------------------------
root = tk.Tk()
root.withdraw()
app = VideoAnalyzer(root)
root.update_idletasks()

cfg = app._collect_config()
cfg["rgb_pipeline"]     = [morph_step(i) for i in range(MAX)]
cfg["ir_pipeline"]      = [morph_step(i) for i in range(MAX)]
cfg["rgb_pre_pipeline"] = [pm_step(i)    for i in range(MAX)]
cfg["ir_pre_pipeline"]  = [pm_step(i)    for i in range(MAX)]
cfg["user_pipelines"]   = [
    branch("maxrgb",    "rgb", channels="HSVab"),
    branch("maxir",     "ir"),
    branch("maxcustom", "custom", source="rgb_mask"),
]
cfg["export_yolo"] = app._export_yolo_block(cfg)

chk("main RGB pipeline has 20 morph steps",
    len(cfg["rgb_pipeline"]) == MAX)
chk("main IR pipeline has 20 morph steps",
    len(cfg["ir_pipeline"]) == MAX)
chk("main RGB pre-morph has 20 PM steps",
    len(cfg["rgb_pre_pipeline"]) == MAX)
chk("main IR pre-morph has 20 PM steps",
    len(cfg["ir_pre_pipeline"]) == MAX)
chk("3 user branches built", len(cfg["user_pipelines"]) == 3)

# ----------------------------------------------------------------------
# 2) Combine tab — defaults, OVERLAY op, colours, save/load round-trip.
# ----------------------------------------------------------------------
from config import VIEW_OPTIONS

chk("Combine tab A/B default to 'none'",
    app.combine_a_var.get() == "none"
    and app.combine_b_var.get() == "none")
chk("Combine tab op defaults to 'none'",
    app.combine_op_var.get() == "none")
chk("Combine has Mask 1 + Mask 2 colour vars",
    hasattr(app, "combine_c1_var") and hasattr(app, "combine_c2_var"))
chk("'combined_overlay' is a selectable panel view",
    "combined_overlay" in VIEW_OPTIONS
    and "combined_overlay" in app._all_view_names())

# Configure the combine like a user would: raw OR mask, OVERLAY op.
app.combine_a_var.set("rgb_mask")
app.combine_b_var.set("ir_raw")          # one operand is a RAW image
app.combine_op_var.set("OVERLAY")
app.combine_c1_var.set("cyan")
app.combine_c2_var.set("orange")

saved = app._collect_config()
strs  = saved.get("strings", {})
chk("combine settings are saved into the config",
    strs.get("combine_a") == "rgb_mask"
    and strs.get("combine_b") == "ir_raw"
    and strs.get("combine_op") == "OVERLAY"
    and strs.get("combine_c1") == "cyan"
    and strs.get("combine_c2") == "orange")

# Round-trip: reset the vars, then reload and confirm they come back.
app.combine_a_var.set("none")
app.combine_b_var.set("none")
app.combine_op_var.set("none")
app._apply_config(saved)
chk("combine settings restored by load",
    app.combine_a_var.get() == "rgb_mask"
    and app.combine_b_var.get() == "ir_raw"
    and app.combine_op_var.get() == "OVERLAY"
    and app.combine_c1_var.get() == "cyan"
    and app.combine_c2_var.get() == "orange")

# ----------------------------------------------------------------------
# 2b) Step add / move / remove must be FAST — incremental, not a full
#     rebuild of every card. Times the ops on a big pipeline and proves
#     they stay quick and beat a full rebuild.
# ----------------------------------------------------------------------
import time

_pl = app.rgb_pipeline
_pf = app.rgb_pip_frame
_pl.clear()
app._rebuild_pipeline_ui(_pf, _pl)


def _timed(fn):
    root.update_idletasks()
    t0 = time.perf_counter()
    fn()
    root.update_idletasks()
    return time.perf_counter() - t0


# Add 5 steps; time the 5th add (small pipeline).
for _ in range(4):
    app._on_add_step(_pf, _pl)
t_add_small = _timed(lambda: app._on_add_step(_pf, _pl))     # -> 5 steps
# Grow to 25 steps; time an add on the big pipeline.
for _ in range(19):
    app._on_add_step(_pf, _pl)
t_add_big = _timed(lambda: app._on_add_step(_pf, _pl))       # -> 25 steps
# Move + remove a middle card on the 25-step pipeline.
t_move   = _timed(lambda: app._on_move_step(_pf, _pl,
                                            _pf._cards[12], +1))
t_remove = _timed(lambda: app._on_remove_step(_pf, _pl,
                                              _pf._cards[12]))
# A full rebuild = what every single edit used to cost.
t_rebuild = _timed(lambda: app._rebuild_pipeline_ui(_pf, _pl))

print("  -- step-edit timings (24-25 step pipeline) --")
print("     incremental add  (5 steps) : %6.1f ms" % (t_add_small * 1e3))
print("     incremental add (25 steps) : %6.1f ms" % (t_add_big   * 1e3))
print("     incremental move           : %6.1f ms" % (t_move      * 1e3))
print("     incremental remove         : %6.1f ms" % (t_remove    * 1e3))
print("     FULL rebuild (old per-edit): %6.1f ms" % (t_rebuild   * 1e3))

chk("incremental add stays FLAT as the pipeline grows "
    "(5 steps %.0f ms ~ 25 steps %.0f ms)"
    % (t_add_small * 1e3, t_add_big * 1e3),
    t_add_big < t_add_small * 2 + 0.10)
chk("incremental add is far cheaper than a full rebuild "
    "(%.0f ms vs %.0f ms)" % (t_add_big * 1e3, t_rebuild * 1e3),
    t_add_big * 4 < t_rebuild)
chk("incremental move is fast (%.1f ms)" % (t_move * 1e3),
    t_move < 0.4)
chk("incremental remove is fast (%.1f ms)" % (t_remove * 1e3),
    t_remove < 0.4)
chk("move/remove beat a full rebuild (%.0f/%.0f ms vs %.0f ms)"
    % (t_move * 1e3, t_remove * 1e3, t_rebuild * 1e3),
    t_move < t_rebuild and t_remove < t_rebuild)

_pl.clear()
app._rebuild_pipeline_ui(_pf, _pl)

# ----------------------------------------------------------------------
# 3) Export — build EVERY output the export dialog offers.
# ----------------------------------------------------------------------
base      = "exported_pipeline"
py_path   = os.path.join(HERE, base + ".py")
lib_file  = base + "_lib.py"
lib_path  = os.path.join(HERE, lib_file)
json_path = os.path.join(HERE, base + "_config.json")
png_path  = os.path.join(HERE, base + ".png")
txt_path  = os.path.join(HERE, base + ".txt")
readme    = os.path.join(HERE, base + "_README.md")

ecfg = app._collect_config()
ecfg["rgb_pipeline"]     = cfg["rgb_pipeline"]
ecfg["ir_pipeline"]      = cfg["ir_pipeline"]
ecfg["rgb_pre_pipeline"] = cfg["rgb_pre_pipeline"]
ecfg["ir_pre_pipeline"]  = cfg["ir_pre_pipeline"]
ecfg["user_pipelines"]   = cfg["user_pipelines"]
ecfg["export_yolo"]      = app._export_yolo_block(ecfg)

with open(py_path, "w") as f:
    f.write(app._export_build_py(ecfg, base + ".py"))
with open(lib_path, "w") as f:
    f.write(app._export_build_lib(lib_file))
with open(json_path, "w") as f:
    json.dump(ecfg, f, indent=2)
app._export_build_png(ecfg, png_path)
with open(txt_path, "w") as f:
    f.write("\n".join(app._settings_report(ecfg)) + "\n")
sel = {k: True for k, _, _ in __import__("export_mixin")._EXPORT_CHOICES}
with open(readme, "w") as f:
    f.write(app._export_build_readme(ecfg, base, sel))
n_examples = app._export_write_example_files(ecfg, base + ".py", HERE)
root.destroy()

for fn in (base + ".py", lib_file, base + "_config.json",
           base + ".png", base + ".txt", base + "_README.md"):
    chk("export wrote %s" % fn,
        os.path.getsize(os.path.join(HERE, fn)) > 0)
chk("export wrote 7 example scripts", n_examples == 7)

# Generated python must be valid python.
ast.parse(open(py_path).read())
ast.parse(open(lib_path).read())
chk("exported .py + _lib.py are valid Python", True)

# ----------------------------------------------------------------------
# 4) PNG row-wrap — 20-step pipeline must wrap, not grow without bound.
# ----------------------------------------------------------------------
from PIL import Image
pw, ph = Image.open(png_path).size
# 20 morph + 20 PM + endpoints ~= 40+ boxes on the RGB row. Un-wrapped
# that would be > 6000 px wide; wrapped at 7/line it stays well under.
chk("PNG wrapped long rows (width %d px is bounded)" % pw, pw < 1600)
chk("PNG has real height (%d px)" % ph, ph > 200)

# ----------------------------------------------------------------------
# 5) The reusable library — load from JSON, run every output mode.
# ----------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("stress_lib", lib_path)
lib  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
chk("library exposes load_pipeline + Pipeline",
    hasattr(lib, "load_pipeline") and hasattr(lib, "Pipeline"))

pipe = lib.load_pipeline(json_path)

img = np.full((240, 320, 3), 255, np.uint8)
cv2.rectangle(img, (110, 80), (210, 160), (0, 0, 200), -1)
ir  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

masks = pipe.detect(img, ir)
chk("library detect() runs the full 20-step config",
    masks["combined"].dtype == np.uint8)
for nm in ("maxrgb", "maxir", "maxcustom"):
    chk("library produced branch up_%s" % nm, ("up_%s" % nm) in masks)

fm = pipe.final_mask(img, ir)
chk("library final_mask() -> single binary mask",
    fm.ndim == 2 and fm.dtype == np.uint8)

ov = pipe.mask_overlay(img, ir)
chk("library mask_overlay() -> BGR image",
    ov.ndim == 3 and ov.shape[2] == 3)

pts = pipe.mask_points(img, ir)
chk("library mask_points() -> list of points", isinstance(pts, list))

# Branch mask is selectable by key, too.
bm = pipe.final_mask(img, ir, key="up_maxrgb")
chk("library final_mask(key='up_maxrgb') works", bm is not None)

# ----------------------------------------------------------------------
# 6) Standalone script — runs end to end as a subprocess.
# ----------------------------------------------------------------------
test_img = os.path.join("/tmp", "_stress_test_img.png")
cv2.imwrite(test_img, img)
r = subprocess.run([sys.executable, py_path, test_img],
                   cwd="/tmp", capture_output=True, text=True)
chk("standalone exported .py runs (%s)"
    % (r.stdout.strip().splitlines() or ["no output"])[-1],
    r.returncode == 0)

# ----------------------------------------------------------------------
# 7) Run every exported example script.
# ----------------------------------------------------------------------
for fn in sorted(f for f in os.listdir(HERE)
                 if f.startswith("example_") and f.endswith(".py")):
    r = subprocess.run([sys.executable, os.path.join(HERE, fn)],
                       cwd="/tmp", capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    chk("ran %s  ->  %s" % (fn, last), r.returncode == 0)

# ----------------------------------------------------------------------
# 8) YOLO box scale / pad + a REAL RealSense frame. Results are written
#    into a detection/ folder inside stress_test/ — auto-versioned to
#    detection_V2, detection_V3 ... so earlier runs are never clobbered.
# ----------------------------------------------------------------------
REAL_IMG = ("/home/mannaja/intern_in_MuroranIT/intern_program/pictures/"
            "realsense/20260427_160836/rgb/frame_000640.png")


def _next_detection_dir():
    base = os.path.join(HERE, "detection")
    if not os.path.exists(base):
        return base
    i = 2
    while os.path.exists("%s_V%d" % (base, i)):
        i += 1
    return "%s_V%d" % (base, i)


det_dir = _next_detection_dir()
os.makedirs(det_dir)
print("detection results -> %s" % det_dir)

# 8a) The box-resize maths is deterministic — verify it head-on.
ab = lib.adjust_box
bx = ab(100, 100, 200, 200, 640, 480, 1.5, 0)
chk("adjust_box: scale 150% enlarges a 100px box to 150px",
    (bx[2] - bx[0]) == 150 and (bx[3] - bx[1]) == 150)
bx = ab(100, 100, 200, 200, 640, 480, 0.5, 0)
chk("adjust_box: scale 50% shrinks a 100px box to 50px",
    (bx[2] - bx[0]) == 50 and (bx[3] - bx[1]) == 50)
bx = ab(100, 100, 200, 200, 640, 480, 1.0, 20)
chk("adjust_box: +20px pad widens a 100px box to 140px",
    (bx[2] - bx[0]) == 140 and (bx[3] - bx[1]) == 140)
bx = ab(0, 0, 50, 50, 640, 480, 4.0, 0)
chk("adjust_box: an over-large box is clamped inside the image",
    bx[0] >= 0 and bx[1] >= 0 and bx[2] <= 640 and bx[3] <= 480)

# 8b) Run the YOLO pipeline on the real RealSense frame and export it.
if not os.path.exists(REAL_IMG):
    print("SKIP - real test image not found:", REAL_IMG)
else:
    real = cv2.imread(REAL_IMG)
    chk("real RealSense frame loaded (%dx%d)"
        % (real.shape[1], real.shape[0]), real is not None)

    def _save(name, arr):
        cv2.imwrite(os.path.join(det_dir, name), arr)

    def _annotate(img, boxes):
        out = img.copy()
        for d in boxes:
            cv2.rectangle(out, (d["x"], d["y"]),
                          (d["x"] + d["w"], d["y"] + d["h"]),
                          (0, 255, 0), 2)
            cv2.putText(out, "%s %.2f" % (d["name"], d["conf"]),
                        (d["x"], max(12, d["y"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out

    # Normal box size — YOLO + every mask output.
    res    = pipe.detect(real)
    boxes0 = res.get("yolo_boxes", [])
    chk("YOLO ran on the real image (%d detection(s))" % len(boxes0),
        isinstance(boxes0, list))
    _save("01_input.png",          real)
    _save("02_yolo_detected.png",  _annotate(real, boxes0))
    if "yolo_mask" in res:
        _save("03_yolo_mask.png",  res["yolo_mask"])
    _save("04_combined_mask.png",  res["combined"])
    _save("05_final_mask.png",     pipe.final_mask(real))
    _save("06_mask_overlay.png",   pipe.mask_overlay(real))
    chk("detection + mask images written into %s/"
        % os.path.basename(det_dir), len(os.listdir(det_dir)) >= 5)

    fm = pipe.final_mask(real)
    chk("mask on the real image is a binary uint8 array",
        fm.ndim == 2 and fm.dtype == np.uint8)

    # Bigger YOLO boxes (scale 180%) via a tweaked exported config.
    big_json = os.path.join(det_dir, "config_box180.json")
    bcfg = json.load(open(json_path))
    bcfg["params"]["YOLO_Box_Scale"] = 180
    json.dump(bcfg, open(big_json, "w"), indent=2)
    bres   = lib.load_pipeline(big_json).detect(real)
    boxes1 = bres.get("yolo_boxes", [])
    _save("07_yolo_detected_box180.png", _annotate(real, boxes1))
    if "yolo_mask" in bres:
        _save("08_yolo_mask_box180.png", bres["yolo_mask"])
    if boxes0 and boxes1:
        a0 = sum(d["w"] * d["h"] for d in boxes0)
        a1 = sum(d["w"] * d["h"] for d in boxes1)
        chk("YOLO_Box_Scale 180%% enlarges the boxes (area %d -> %d)"
            % (a0, a1), a1 > a0)
    else:
        print("note: model found no objects in this frame — the live "
              "box-scale size comparison is skipped (the resize maths "
              "is already verified in step 8a).")

# -- tidy up /tmp scratch files ---------------------------------------
for fn in os.listdir("/tmp"):
    if (fn.startswith("ex") or fn.startswith("lib_")
            or fn.startswith("_stress_test")) and \
            fn.endswith((".png", ".mp4")):
        try:
            os.remove(os.path.join("/tmp", fn))
        except OSError:
            pass

print()
print("=" * 64)
print(" ALL %d CHECKS PASSED" % ok)
print("=" * 64)
