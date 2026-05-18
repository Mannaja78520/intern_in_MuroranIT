"""Maximum / stress test for the pipeline-export feature.

The program has NO hard step cap (step views are dynamic), so this
test uses 20 as the "maximum" and builds the heaviest realistic config:

  * main RGB pipeline  - 20 morph steps + 20 pre-morph (PM) steps
  * main IR  pipeline  - 20 morph steps + 20 pre-morph (PM) steps
  * 3 user branches    - each 20 morph steps + 20 pre-morph steps
    (branch types: rgb, ir, custom)

It EXPORTS that maximum config into this folder (pipeline .py + .png +
the 7 example scripts - the program building the examples too) and
RUNS the exported pipeline + every example to prove it all survives
at full size.

Run:  conda run -n intern_muroranIT_py312 python maximum_test/maximum_test.py
"""
import os
import sys
import subprocess
import importlib.util

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # the 2026_05_17 folder
sys.path.insert(0, PARENT)
os.environ.setdefault("DISPLAY", ":1")

import tkinter as tk
import numpy as np
import cv2
from realsense_video_analyzer import VideoAnalyzer

MAX = 20

# Cycle through many ops so the test exercises a wide range of code.
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
    """One of the 20 morph steps. Every 5th step is a Combine (OR)
    step so the combine path is exercised too."""
    if i % 5 == 4:
        return {"en": True, "op": "Dilate", "n": 1, "dir": "Both",
                "kx": 3, "ky": 3, "t": 0,
                "comb_en": True, "comb_op": "OR", "comb_src": "mask_pre"}
    return {"en": True, "op": MORPH_CYCLE[i % len(MORPH_CYCLE)],
            "n": 1, "dir": "Both", "kx": 3, "ky": 3, "t": 120}


def pm_step(i):
    """One of the 20 pre-morph (image-conditioning) steps."""
    return {"en": True, "op": PM_CYCLE[i % len(PM_CYCLE)],
            "n": 1, "dir": "XY", "kx": 5, "ky": 5, "t": 0}


def branch(name, btype, **extra):
    bd = {"name": name, "type": btype,
          "steps":     [morph_step(i) for i in range(MAX)],
          "pre_steps": [pm_step(i)    for i in range(MAX)]}
    bd.update(extra)
    return bd


# -- 1) build the analyzer, then construct the MAXIMUM config ---------
# The config dict is built directly (instead of clicking 200 steps into
# the live UI) — the export reads the dict, so this is the same result.
root = tk.Tk()
root.withdraw()
app = VideoAnalyzer(root)
root.update_idletasks()

cfg = app._collect_config()                  # real defaults (all params)
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

# -- 2) verify everything is at maximum size --------------------------
chk("main RGB pipeline has 20 morph steps",
    len(cfg["rgb_pipeline"]) == MAX)
chk("main IR pipeline has 20 morph steps",
    len(cfg["ir_pipeline"]) == MAX)
chk("main RGB pre-morph has 20 PM steps",
    len(cfg["rgb_pre_pipeline"]) == MAX)
chk("main IR pre-morph has 20 PM steps",
    len(cfg["ir_pre_pipeline"]) == MAX)
chk("3 user branches built", len(cfg["user_pipelines"]) == 3)
for b in cfg["user_pipelines"]:
    chk("branch '%s' has 20 morph + 20 PM steps" % b["name"],
        len(b["steps"]) == MAX and len(b["pre_steps"]) == MAX)

# -- 3) export the maximum config into THIS folder --------------------
py_path  = os.path.join(HERE, "exported_pipeline.py")
png_path = os.path.join(HERE, "exported_pipeline.png")
with open(py_path, "w") as f:
    f.write(app._export_build_py(cfg, "exported_pipeline.py"))
app._export_build_png(cfg, png_path)
n_examples = app._export_write_example_files(
    cfg, "exported_pipeline.py", HERE)        # program builds examples
root.destroy()
print("exported maximum pipeline + %d example scripts into maximum_test/"
      % n_examples)

for fn in ["exported_pipeline.py", "exported_pipeline.png", "README.txt",
           "example_1_commandline.py", "example_2_video.py",
           "example_3_image.py", "example_4_one_pipeline.py",
           "example_5_branch_or.py", "example_6_full.py",
           "example_7_yolo.py"]:
    chk("maximum_test/%s written" % fn,
        os.path.getsize(os.path.join(HERE, fn)) > 0)

# -- 4) the exported MAXIMUM pipeline runs ----------------------------
spec = importlib.util.spec_from_file_location("exported_pipeline", py_path)
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
chk("exported pipeline imports", hasattr(pl, "detect"))
chk("CONFIG carries all 20 RGB morph steps",
    len(pl.CONFIG["rgb_pipeline"]) == MAX)
chk("CONFIG carries all 20 RGB PM steps",
    len(pl.CONFIG["rgb_pre_pipeline"]) == MAX)

img = np.full((240, 320, 3), 255, np.uint8)
cv2.rectangle(img, (110, 80), (210, 160), (0, 0, 200), -1)
res = pl.detect(img, with_yolo=False)
chk("detect() runs the full 20-step pipeline",
    res["combined"].dtype == np.uint8 and res["combined"].ndim == 2)
for nm in ("maxrgb", "maxir", "maxcustom"):
    chk("branch up_%s produced" % nm, ("up_%s" % nm) in res)

# -- 5) run every example script as a real subprocess -----------------
for fn in sorted(f for f in os.listdir(HERE)
                 if f.startswith("example_") and f.endswith(".py")):
    r = subprocess.run([sys.executable, os.path.join(HERE, fn)],
                       cwd="/tmp", capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    chk("ran %s  ->  %s" % (fn, last), r.returncode == 0)

for fn in os.listdir("/tmp"):
    if fn.startswith("ex") and (fn.endswith(".png") or fn.endswith(".mp4")):
        try:
            os.remove(os.path.join("/tmp", fn))
        except OSError:
            pass

print()
print("ALL %d CHECKS PASSED  (maximum 20-step config)" % ok)
