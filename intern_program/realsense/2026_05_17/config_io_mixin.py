"""Config save / load + user-prefs persistence.

Extracted from realsense_video_analyzer.py as a mixin: serialises
every tunable (sliders, pipelines, branches, colours) to JSON and
restores it, plus the small ~/.realsense_analyzer_prefs.json helpers.
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


class ConfigIOMixin:
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
                "use_bgsub_rgb":  self.use_bgsub_rgb.get(),
                "use_bgsub_ir":   self.use_bgsub_ir.get(),
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
                "rgb_detect_mode": getattr(
                    self, "rgb_detect_mode",
                    tk.StringVar(value="Color (HSV hue)")).get(),
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
        _params = cfg.get("params") or {}
        for k, val in _params.items():
            if k in self.sv:
                try:
                    self.sv[k].set(val)
                except Exception:
                    pass
        # Backward compat: old configs stored one shared BG_hist /
        # BG_var; the params are per-side now (BG_hist_rgb / _ir ...),
        # so fan an old shared value out to both sides.
        for _old, _news in (("BG_hist", ("BG_hist_rgb", "BG_hist_ir")),
                            ("BG_var",  ("BG_var_rgb",  "BG_var_ir"))):
            if _old in _params:
                for _n in _news:
                    if _n in self.sv:
                        try:
                            self.sv[_n].set(_params[_old])
                        except Exception:
                            pass
        flags = cfg.get("flags") or {}
        # `use_clahe` from old configs is silently ignored — CLAHE now
        # lives as an IR PM step. Same for the removed pre-blur fields.
        for k in ("use_bgsub_rgb", "use_bgsub_ir", "show_boxes",
                  "show_overlay", "use_yolo", "loop_var",
                  "use_clahe_ir"):
            if k in flags and hasattr(self, k):
                try:
                    getattr(self, k).set(bool(flags[k]))
                except Exception:
                    pass
        # Backward compat: old configs had one shared `use_bgsub`
        # flag — apply it to both sides when the per-side flags are
        # absent.
        if ("use_bgsub" in flags
                and "use_bgsub_rgb" not in flags
                and "use_bgsub_ir" not in flags):
            for _attr in ("use_bgsub_rgb", "use_bgsub_ir"):
                if hasattr(self, _attr):
                    try:
                        getattr(self, _attr).set(bool(flags["use_bgsub"]))
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
            ("rgb_detect_mode", "rgb_detect_mode"),
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
        # PM steps first (run before morph), then morph steps. The
        # per-step UI rebuild is deferred (defer_ui=True) and done ONCE
        # at the end — appending N steps is then O(n), not O(n^2).
        for psd in bd.get("pre_steps", []):
            self._add_branch_pre_step(rec, psd, defer_ui=True)
        for sd in bd.get("steps", []):
            self._add_branch_step(rec, sd, defer_ui=True)
        if rec.get("pre_steps_row") is not None:
            self._rebuild_pre_morph_pipeline_ui(
                rec["pre_steps_row"], rec["pre_steps"])
        if rec.get("steps_row") is not None:
            self._rebuild_pipeline_ui(rec["steps_row"], rec["steps"])

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

