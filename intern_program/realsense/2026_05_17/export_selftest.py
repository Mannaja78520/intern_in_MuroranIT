"""Self-test for the pipeline-export feature.

Run:  conda run -n intern_muroranIT_py312 python export_selftest.py

It (1) builds the analyzer, (2) loads analyzer_config.json if present,
(3) writes the export into the  example/  folder:
      example/exported_pipeline.py     - the standalone pipeline
      example/exported_pipeline.png    - flowchart + settings report
      example/example_1_commandline.py ... example_7_yolo.py
      example/README.txt
and (4) RUNS every one of the 7 example scripts to prove they work.
"""
import os
import sys
import json
import shutil
import subprocess
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DISPLAY", ":1")

import tkinter as tk
import numpy as np
import cv2
from realsense_video_analyzer import VideoAnalyzer

ok = 0


def chk(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name)
    if cond:
        ok += 1
    else:
        raise SystemExit("FAILED: " + name)


# -- 1) build the analyzer --------------------------------------------
root = tk.Tk()
root.withdraw()
app = VideoAnalyzer(root)
root.update_idletasks()

# -- 2) load the tuned config so the export reflects real settings ----
cfg_path = os.path.join(HERE, "analyzer_config.json")
if os.path.exists(cfg_path):
    try:
        with open(cfg_path) as f:
            app._apply_config(json.load(f))
        print("loaded", os.path.basename(cfg_path))
    except Exception as e:
        print("could not load config (%s) - using defaults" % e)

# -- 3) write the export into the example/ folder ---------------------
EX_DIR = os.path.join(HERE, "example")
if os.path.isdir(EX_DIR):
    shutil.rmtree(EX_DIR)
os.makedirs(EX_DIR, exist_ok=True)

cfg = app._collect_config()
cfg["export_yolo"] = app._export_yolo_block(cfg)
app._export_build_png(cfg, os.path.join(EX_DIR, "exported_pipeline.png"))
n_examples = app._export_write_example_files(
    cfg, "exported_pipeline.py", EX_DIR)
root.destroy()

# drop any stale root-level export files from earlier runs
for stale in ("exported_pipeline.py", "exported_pipeline.png",
              "exported_pipeline_examples.py"):
    p = os.path.join(HERE, stale)
    if os.path.exists(p):
        os.remove(p)

print("wrote example/ with %d example scripts + pipeline + png"
      % n_examples)

# -- 4) check the files exist -----------------------------------------
expect = ["exported_pipeline.py", "exported_pipeline.png", "README.txt",
          "example_1_commandline.py", "example_2_video.py",
          "example_3_image.py", "example_4_one_pipeline.py",
          "example_5_branch_or.py", "example_6_full.py",
          "example_7_yolo.py"]
for fn in expect:
    chk("example/%s exists" % fn,
        os.path.getsize(os.path.join(EX_DIR, fn)) > 0)
chk("wrote all 7 example scripts", n_examples == 7)

# -- the exported pipeline imports + runs -----------------------------
spec = importlib.util.spec_from_file_location(
    "exported_pipeline", os.path.join(EX_DIR, "exported_pipeline.py"))
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
chk("exported pipeline imports cleanly", hasattr(pl, "detect"))
chk("CONFIG has export_yolo block", "export_yolo" in pl.CONFIG)
print("YOLO enabled in export:", pl.CONFIG["export_yolo"]["enabled"])

# -- 5) RUN every example script as a real subprocess -----------------
for fn in sorted(f for f in expect if f.startswith("example_")):
    r = subprocess.run([sys.executable, os.path.join(EX_DIR, fn)],
                       cwd="/tmp", capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    chk("ran %s  ->  %s" % (fn, last), r.returncode == 0)

# tidy the throw-away example outputs
for fn in os.listdir("/tmp"):
    if fn.startswith("ex") and (fn.endswith(".png") or fn.endswith(".mp4")):
        try:
            os.remove(os.path.join("/tmp", fn))
        except OSError:
            pass

print()
print("ALL %d CHECKS PASSED" % ok)
