# Change summary — 17 May & 18 May 2026

RealSense Cable Video Analyzer. Folders compared:
`2026_05_16_V2` → `2026_05_17` → `2026_05_18`.

---

## 17 May  (16_V2 → 17)

### Pipeline export — NEW
- New `export_mixin.py` + `export_selftest.py`: the GUI can export its
  tuned pipeline as a self-contained, runnable file.

### Main app restructured (`realsense_video_analyzer.py`, +794 / −154)
- Refactored into a `VideoAnalyzer` class built from mixins
  (Processing / ConfigIO / Export / Flowchart / PipelineUI / UserPipelines).
- New: initial window-geometry fit, `add_param` slider helper,
  section labels, live-refresh wiring.
- Source handling: RGB/IR pairing, multi-capture seek/grab,
  open-pair, folder-video picker with keyword matching.

### Detection (`processing_mixin.py`, +170 / −19)
- Multiple RGB detect modes: `_rgb_detect`, `_rgb_detect_key`,
  `_lab_mask` (LAB colour detection), `_stamp_mode_tag`.
- Background-subtraction auto-source: `_bgsub_auto_rgb`,
  `_bgsub_auto_branch`.

### Branches (`user_pipelines_mixin.py`, +156 / −36)
- Serialised branch step add (`_add_branch_pre_step`,
  `_add_branch_step`) and branch rename
  (`_commit_branch_rename`, `_branch_view_rewrite`).
- `config.py` +42, `config_io_mixin.py` +43, `flowchart_mixin.py` +23.

---

## 18 May  (17 → 18)  +  cable-detection work

### Export feature greatly expanded (`export_mixin.py`, +678 / −30)
- New `Pipeline` class: `from_json`, `detect`, `views`, `final_mask`,
  `mask_overlay`, `mask_points`, `yolo_points`; plus `load_pipeline`,
  `run_yolo`, `adjust_box`.
- Export options dialog + exported-library / README builder.
- New `stress_test/` harness: 7 example scripts, `stress_test.py`
  (52 checks), exported-pipeline sample files.

### Quick save / load (`config_io_mixin.py`, +108 / −12)
- Ctrl+S / Ctrl+O quick save/load: `_quick_save_config`,
  `_quick_load_config`, `_default_config_path`.
- `_short_path` shortens long paths in the status bar so the
  window no longer widens and shifts the 6 video panels.

### Step-card UX (`pipeline_ui_mixin.py`, +446 / −245)
- Lazy build of the Combine / OVERLAY / YOLO sub-sections
  (`_build_combine`, `_build_yolo`, `_on_yolo_en`) — "Add step" is
  fast now.
- Step move / remove / relayout (`_on_move_step`, `_on_remove_step`,
  `_relayout_step_cards`, `_step_card_index`).
- Fixed the `bad window path name` Tkinter crash — stale trace
  callbacks now cleaned on card destroy (`_cleanup_card_traces`).

### Per-step YOLO box size (`processing_mixin.py`, +143 / −33)
- `_yolo_box_adjust` takes a per-step scale% / pad override;
  `_yolo_box_adj` + `_run_yolo_for(box_adj=...)`.
- Each YOLO step card (main + branch) gets its own Box scale% / pad.

### LAB detection + branch channels
- LAB mode is now L/a*/b* (L lightness made adjustable; was a*/b*
  only). All descriptions updated; old configs remapped for
  backward compatibility.
- Branch channel picker shows only the selected channels
  (`_refresh_branch_channels`).

### Recording
- Saves the analyzer config JSON into the recording folder.
- Records all 6 panels combined into one `combined_6up.mp4`.

### Cable detection — NEW (18–19 May)
Task: detect the red cable linking the drone to the T-hook across
the 17 `videos/realsense/recordings/*/rgb.mp4` clips.
- `cable_detect_batch.py` (v1): HSV+LAB red mask + morphology;
  YOLO (`drone_t_hook.pt`) boxes the drone & T-hook; red kept only
  inside the padded box region; writes annotated + mask videos +
  `cable_detect_config.json` (analyzer-loadable).
- `cable_detect_v2.py`: multi-step — MOG2 bg-sub + grayscale
  frame-diff motion, YOLO erases drone/T-hook boxes from the motion
  mask, cable = red ∪ motion-reddish, plus a T-hook focus panel with
  a wrap "curl" score. Writes a 3×2 step-grid + mask video.
- All 17 recordings processed by both.

### Known limits (cable detection)
- Far / small cable (drone far away) is 1–3 px wide — a physical
  pixel limit no colour/motion rule recovers.
- Static red marker tape on background bars is genuinely as red as
  the cable and can't be separated by colour.
- Recommended next step: train a `yolo11-seg` instance-segmentation
  model on polygon-labelled cable frames.
