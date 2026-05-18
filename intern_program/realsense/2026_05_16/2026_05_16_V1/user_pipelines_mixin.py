"""Mixin: user-defined parallel pipelines (branches).

Each branch has a TYPE picked at creation time:

  * ``rgb``    - same H1/H2 + S + V detection structure as the main
                 RGB pipeline, with a CHANNELS dropdown that selects
                 which channels participate in the AND combination
                 (H, S, V, HS, HV, SV, HSV, full).
  * ``ir``     - IR threshold detection (lo/hi).
  * ``custom`` - pass-through: pick any view in the program (e.g.
                 ``rgb_step3``, ``ir_clahe``, ``up_branch1``) as the
                 source. No channel detection.

Detection result is exposed as ``up_<name>_det`` and is the default
input to the branch's pipeline steps.

Host class must already include PipelineUIMixin and FlowchartMixin and
provide:
    self.root, self.user_pipelines, self.user_pipelines_host
    self._refresh, self._rebuild_all_masks_if_open
"""
import tkinter as tk
from tkinter import ttk

from config import VIEW_OPTIONS

# OpenCV's hue is 0-179, but the user wants the slider to go to 180 so
# the upper edge is clearly inclusive of every red. cv2.inRange's upper
# bound is inclusive, and a value of 180 still selects every legal H.
H_MAX = 180

CHANNEL_MODES = ["H", "S", "V", "HS", "HV", "SV", "HSV", "full"]
TYPE_OPTIONS  = ["rgb", "ir", "custom"]


class UserPipelinesMixin:
    """Adds user-defined parallel pipeline UI methods."""

    # ------------------------------------------------------------------
    # Public list of selectable source views (built-ins + per-branch
    # masks + every other user-pipeline output).
    # ------------------------------------------------------------------
    def _all_view_names(self):
        ups = []
        # YOLO per-class views (model classes + the "any" union mask)
        try:
            ym = getattr(self, "yolo_model", None)
            if ym is not None:
                _names = (ym.names if isinstance(ym.names, dict)
                          else dict(enumerate(ym.names)))
                for _cid, _cname in _names.items():
                    _slug = str(_cname).strip().replace(" ", "_") \
                            or f"c{int(_cid)}"
                    ups.append(f"yolo_{_slug}_rgb")
                    ups.append(f"yolo_{_slug}_ir")
                ups += ["yolo_any_rgb", "yolo_any_ir"]
        except Exception:
            pass
        for p in self.user_pipelines:
            nm = p["name"].get().strip()
            if not nm:
                continue
            ups.append(f"up_{nm}")
            ups += [f"up_{nm}_det",
                    f"up_{nm}_mask",
                    f"up_{nm}_overlay_rgb",
                    f"up_{nm}_overlay_hsv",
                    f"up_{nm}_overlay_ir",
                    f"up_{nm}_bgsub",
                    f"up_{nm}_h1_mask", f"up_{nm}_h2_mask",
                    f"up_{nm}_s_mask",  f"up_{nm}_v_mask",
                    f"up_{nm}_ir_mask"]
            for i in range(len(p["steps"])):
                ups.append(f"up_{nm}_step{i+1}")
            # Branch PM (pre-morph) step views run on the source
            # image before detection — published as `up_<name>_pre_step{N}`.
            for i in range(len(p.get("pre_steps") or [])):
                ups.append(f"up_{nm}_pre_step{i+1}")
        return list(VIEW_OPTIONS) + ups

    # ------------------------------------------------------------------
    # Add / remove a branch
    # ------------------------------------------------------------------
    def _new_branch_record(self, name=None):
        """Build the StringVar/IntVar dict for a new branch (used by
        both the interactive setup dialog and the config-load path)."""
        idx = len(self.user_pipelines) + 1
        nm  = name or f"branch{idx}"
        return {
            "name":     tk.StringVar(value=nm),
            "type":     tk.StringVar(value="rgb"),
            "source":   tk.StringVar(value=f"up_{nm}_det"),
            "color":    tk.StringVar(value="#00ffff"),
            "channels": tk.StringVar(value="HSV"),
            # H1 / H2 / S / V mirror the main RGB pipeline params.
            "h1_lo":    tk.IntVar(value=0),
            "h1_hi":    tk.IntVar(value=10),
            "h2_lo":    tk.IntVar(value=170),
            "h2_hi":    tk.IntVar(value=180),
            "s_lo":     tk.IntVar(value=120),
            "s_hi":     tk.IntVar(value=255),
            "v_lo":     tk.IntVar(value=70),
            "v_hi":     tk.IntVar(value=255),
            "ir_lo":    tk.IntVar(value=80),
            "ir_hi":    tk.IntVar(value=255),
            # PM (Pre-morph) pipeline — list of 10-tuples (same shape
            # as `steps`). Each step is an op applied to the branch's
            # SOURCE image (BGR for rgb/custom branches, grayscale for
            # ir branches) BEFORE HSV / IR detection. Replaces the old
            # single-knob pre_blur_type / pre_blur_k / pre_blur_ky.
            "pre_steps":      [],
            "pre_steps_row":  None,
            "use_bgsub":      tk.BooleanVar(value=False),
            "bgsub_src":      tk.StringVar(value="rgb"),
            "bgsub_history":  tk.IntVar(value=500),
            "bgsub_varth":    tk.IntVar(value=16),
            "_backSub":       None,
            "_backSub_sig":   None,
            # Dedicated IR CLAHE for ir-type branches. Applied to the
            # branch's IR gray feed before PM steps + threshold so the
            # branch detection sees contrast-enhanced input.
            "use_clahe_ir":   tk.BooleanVar(value=False),
            "clahe_ir_clip":  tk.IntVar(value=2),
            "clahe_ir_tile":  tk.IntVar(value=8),
            "overlay":  tk.StringVar(value="mask"),
            "steps":    [],
            "frame":    None,
            "steps_row": None,
        }

    def _on_add_user_pipeline(self):
        """Open a 'New Branch' setup dialog. The user picks the type
        (rgb / ir / custom), tunes the detection or source, then clicks
        Submit. Only after Submit is the branch actually added."""
        rec = self._new_branch_record()
        self._open_branch_setup_dialog(rec, on_submit=self._submit_new_branch)

    def _add_user_pipeline(self, rec):
        """Programmatic branch-add (skips the setup dialog). Used by
        the config loader."""
        self._submit_new_branch(rec)

    def _add_branch_pre_step(self, rec, step_dict):
        """Append a PM (Pre-morph) step to `rec`'s branch from a
        serialised step dict. PM steps only use the morph fields
        (en/op/n/dir/kx/ky/t) plus the per-step kshape and Target
        channel; combine / overlay / yolo are unused for image-stage
        ops."""
        self._create_default_steps(rec["pre_steps"], [
            (step_dict.get("en", False),
             step_dict.get("op", "GaussBlur"),
             step_dict.get("n", 1),
             step_dict.get("dir", "XY"),
             step_dict.get("kx", 5),
             step_dict.get("ky", 5),
             step_dict.get("t", 0))])
        if "kshape" in step_dict:
            ks = self._kshape_state_for(rec["pre_steps"][-1])
            try:
                ks["kshape"].set(step_dict["kshape"])
            except Exception:
                pass
        if "pm_target" in step_dict:
            pmt = self._pm_target_state_for(rec["pre_steps"][-1])
            try:
                pmt["target"].set(step_dict["pm_target"])
            except Exception:
                pass
        if rec.get("pre_steps_row") is not None:
            self._rebuild_pre_morph_pipeline_ui(
                rec["pre_steps_row"], rec["pre_steps"])

    def _add_branch_step(self, rec, step_dict):
        """Append a single step to `rec`'s branch from a serialised
        step dict (the inverse of _step_to_dict)."""
        self._create_default_steps(rec["steps"], [
            (step_dict.get("en", False),
             step_dict.get("op", "Dilate"),
             step_dict.get("n", 1),
             step_dict.get("dir", "Both"),
             step_dict.get("kx", 3),
             step_dict.get("ky", 3),
             step_dict.get("t", 0))])
        st = rec["steps"][-1]
        (en, op, n, dr, kx, ky, th, cen, cop, csr) = st
        if "comb_en" in step_dict:
            cen.set(step_dict["comb_en"])
        if "comb_op" in step_dict:
            cop.set(step_dict["comb_op"])
        if "comb_src" in step_dict:
            csr.set(step_dict["comb_src"])
        if "kshape" in step_dict:
            ks = self._kshape_state_for(st)
            try:
                ks["kshape"].set(step_dict["kshape"])
            except Exception:
                pass
        ov = (getattr(self, "_overlay_state", {}) or {}).get(id(st))
        for k, v in (step_dict.get("overlay") or {}).items():
            if ov is not None and k in ov:
                try:
                    ov[k].set(v)
                except Exception:
                    pass
        ys = (getattr(self, "_yolo_state", {}) or {}).get(id(st))
        for k, v in (step_dict.get("yolo") or {}).items():
            if ys is not None and k in ys and k != "yolo_classes":
                try:
                    ys[k].set(v)
                except Exception:
                    pass
        if ys is not None:
            _ycl = (step_dict.get("yolo") or {}).get("yolo_classes")
            if isinstance(_ycl, dict):
                for _cid_s, _val in _ycl.items():
                    try:
                        self._yolo_class_var(
                            ys, int(_cid_s)).set(bool(_val))
                    except Exception:
                        pass
        if "step_kind" in step_dict:
            stt = self._step_type_state_for(st)
            try:
                stt["kind"].set(step_dict["step_kind"])
            except Exception:
                pass
        # Refresh the branch's step UI so the new step card appears.
        if rec.get("steps_row") is not None:
            self._rebuild_pipeline_ui(rec["steps_row"], rec["steps"])

    def _submit_new_branch(self, rec):
        """Called by the setup dialog's Submit button.
        The new branch's UI card is built immediately so the user
        sees that the add worked, but the All-Masks flowchart and
        the pipeline reprocess are NOT triggered here — adding a
        branch is potentially expensive (every step rebuilds), so
        we wait for the user to press Apply (F5) or hit play."""
        self.user_pipelines.append(rec)
        # Trace every config var so EDITS auto-refresh (slider
        # tweaks should feel live), but not the initial add itself.
        # NOTE: rec["name"] is also not traced — typing in the name
        # field would otherwise rebuild on every keystroke.
        for _key in ("type", "channels",
                     "h1_lo", "h1_hi", "h2_lo", "h2_hi",
                     "s_lo",  "s_hi",  "v_lo",  "v_hi",
                     "ir_lo", "ir_hi",
                     "use_bgsub", "bgsub_src",
                     "bgsub_history", "bgsub_varth",
                     "use_clahe_ir", "clahe_ir_clip", "clahe_ir_tile",
                     "overlay",
                     "source", "color"):
            rec[_key].trace_add("write", lambda *a: self._refresh())
        self._build_user_pipeline_frame(rec)

    def _on_remove_user_pipeline(self, rec):
        try:
            self.user_pipelines.remove(rec)
            if rec["frame"] is not None:
                # Forget the Notebook tab first so the ttk widget
                # state stays consistent, then destroy the frame.
                try:
                    self.user_pipelines_host.forget(rec["frame"])
                except Exception:
                    pass
                rec["frame"].destroy()
        except ValueError:
            pass
        # Defer the All-Masks rebuild + reprocess until Apply (F5)
        # so removing a branch doesn't trigger an expensive redraw.

    # ------------------------------------------------------------------
    # 'New Branch' setup dialog (modal)
    # ------------------------------------------------------------------
    def _open_branch_setup_dialog(self, rec, on_submit):
        dlg = tk.Toplevel(self.root)
        dlg.title("New Branch - pick type and detection settings")
        dlg.transient(self.root)
        dlg.grab_set()

        top = tk.Frame(dlg, padx=12, pady=10)
        top.pack(fill="both", expand=True)

        # -- Name + Colour --------------------------------------------
        row1 = tk.Frame(top)
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="Name:",
                 font=("DejaVu Sans", 9, "bold")).pack(side="left")
        tk.Entry(row1, textvariable=rec["name"], width=14,
                 font=("DejaVu Sans", 9)).pack(side="left", padx=4)
        tk.Label(row1, text="  Colour:",
                 font=("DejaVu Sans", 9)).pack(side="left")
        _br_swatch_d = tk.Label(row1, text="    ", width=3,
                                relief="solid", bd=1,
                                bg=rec["color"].get())
        _br_swatch_d.pack(side="left", padx=(4, 2))
        tk.Entry(row1, textvariable=rec["color"], width=9,
                 font=("DejaVu Sans", 9)).pack(side="left", padx=2)
        tk.Button(row1, text="Pick...", font=("DejaVu Sans", 9),
                  command=lambda v=rec["color"]: self._pick_color_for(
                      v, "Pick branch colour")
                  ).pack(side="left", padx=2)
        def _br_sync_swatch_d(*_a, _s=_br_swatch_d, _v=rec["color"]):
            hx = _v.get().strip()
            if hx and not hx.startswith("#"):
                hx = "#" + hx
            try:
                _s.config(bg=hx)
            except tk.TclError:
                pass
        rec["color"].trace_add("write", _br_sync_swatch_d)

        # -- Type radio buttons ---------------------------------------
        type_lf = tk.LabelFrame(top, text="Branch type",
                                font=("DejaVu Sans", 9, "bold"),
                                fg="#cc88ff")
        type_lf.pack(fill="x", pady=4)
        for label, val, hint in [
            ("RGB / HSV detection",
             "rgb",
             "H1/H2 + S + V (same as main RGB pipeline). "
             "Pick which channels to AND together."),
            ("IR threshold",
             "ir",
             "Greyscale IR threshold (lo / hi)."),
            ("Custom (any frame)",
             "custom",
             "Use any view (e.g. rgb_step3, ir_clahe, up_branch1) "
             "directly as the source. No detection."),
        ]:
            row = tk.Frame(type_lf)
            row.pack(fill="x", padx=4, pady=1)
            tk.Radiobutton(row, text=label, variable=rec["type"],
                           value=val,
                           font=("DejaVu Sans", 9, "bold"),
                           command=lambda: _refresh_panels()
                           ).pack(side="left")
            tk.Label(row, text=hint,
                     font=("DejaVu Sans", 9), fg="#c8c8c8",
                     wraplength=420, justify="left"
                     ).pack(side="left", padx=4)

        # -- RGB / HSV detection panel --------------------------------
        rgb_lf = tk.LabelFrame(top, text="RGB / HSV detection",
                               font=("DejaVu Sans", 9, "bold"))

        # Channel-mode dropdown.
        chrow = tk.Frame(rgb_lf)
        chrow.pack(fill="x", padx=6, pady=3)
        tk.Label(chrow, text="Channels to use:",
                 font=("DejaVu Sans", 9, "bold")).pack(side="left")
        ttk.Combobox(chrow, textvariable=rec["channels"],
                     values=CHANNEL_MODES,
                     width=8, state="readonly",
                     font=("DejaVu Sans", 9)).pack(side="left", padx=4)
        tk.Label(chrow,
                 text=("H = (H1 OR H2).  S, V each individual.  "
                       "Combo (e.g. HS) = AND of those.  "
                       "full = (H1 OR H2) AND S AND V."),
                 font=("DejaVu Sans", 9), fg="#c8c8c8",
                 wraplength=480, justify="left"
                 ).pack(side="left", padx=4)

        def _h_row(parent, label, lo_var, hi_var, lo_max):
            r = tk.Frame(parent)
            r.pack(fill="x", padx=6, pady=1)
            tk.Label(r, text=label, font=("DejaVu Sans", 9),
                     width=10, anchor="w").pack(side="left")
            tk.Label(r, text="lo", fg="#c8c8c8",
                     font=("DejaVu Sans", 8)).pack(side="left")
            tk.Spinbox(r, textvariable=lo_var, from_=0, to=lo_max,
                       width=5, font=("DejaVu Sans", 9)
                       ).pack(side="left", padx=2)
            tk.Scale(r, variable=lo_var, from_=0, to=lo_max,
                     orient="horizontal", length=140,
                     showvalue=False).pack(side="left")
            tk.Label(r, text="hi", fg="#c8c8c8",
                     font=("DejaVu Sans", 8)).pack(side="left")
            tk.Spinbox(r, textvariable=hi_var, from_=0, to=lo_max,
                       width=5, font=("DejaVu Sans", 9)
                       ).pack(side="left", padx=2)
            tk.Scale(r, variable=hi_var, from_=0, to=lo_max,
                     orient="horizontal", length=140,
                     showvalue=False).pack(side="left")

        _h_row(rgb_lf, "H1",   rec["h1_lo"], rec["h1_hi"], H_MAX)
        _h_row(rgb_lf, "H2",   rec["h2_lo"], rec["h2_hi"], H_MAX)
        _h_row(rgb_lf, "S",    rec["s_lo"],  rec["s_hi"],  255)
        _h_row(rgb_lf, "V",    rec["v_lo"],  rec["v_hi"],  255)

        # -- IR detection panel ---------------------------------------
        ir_lf = tk.LabelFrame(top, text="IR threshold detection",
                              font=("DejaVu Sans", 9, "bold"))
        _h_row(ir_lf, "IR",   rec["ir_lo"], rec["ir_hi"], 255)

        # -- Custom-source panel --------------------------------------
        cust_lf = tk.LabelFrame(top, text="Custom source",
                                font=("DejaVu Sans", 9, "bold"))
        tk.Label(cust_lf,
                 text="Pick the source view via two cascading dropdowns:",
                 font=("DejaVu Sans", 8), fg="#c8c8c8"
                 ).pack(anchor="w", padx=6, pady=(4, 2))
        # Cascading From: / Step: picker — same UX as the combine and
        # Mask 2 pickers so behaviour is consistent everywhere.
        self._make_cascading_picker(
            cust_lf, rec["source"],
            pipeline_list=[],          # no in-pipeline shortcuts here
            label_from="From:", label_step="Step:",
            allow_none=False)

        def _refresh_panels():
            t = rec["type"].get()
            for w in (rgb_lf, ir_lf, cust_lf):
                w.pack_forget()
            if t == "rgb":
                rgb_lf.pack(fill="x", pady=4)
            elif t == "ir":
                ir_lf.pack(fill="x", pady=4)
            else:
                cust_lf.pack(fill="x", pady=4)
        _refresh_panels()

        # -- Buttons --------------------------------------------------
        btn = tk.Frame(top)
        btn.pack(fill="x", pady=(8, 0))

        def _cancel():
            dlg.destroy()

        def _submit():
            # If type=rgb, default the source to the branch's own _det
            # so the pipeline starts from the channel-detection mask.
            t = rec["type"].get()
            nm = rec["name"].get().strip() or "branch"
            if t == "rgb":
                rec["source"].set(f"up_{nm}_det")
            elif t == "ir":
                rec["source"].set(f"up_{nm}_det")
            # custom: rec["source"] is whatever the user picked
            dlg.destroy()
            on_submit(rec)

        tk.Button(btn, text="[OK] Submit", bg="#225522", fg="white",
                  font=("DejaVu Sans", 9, "bold"),
                  width=12, command=_submit).pack(side="right", padx=4)
        tk.Button(btn, text="Cancel",
                  font=("DejaVu Sans", 9),
                  width=10, command=_cancel).pack(side="right")

    # ------------------------------------------------------------------
    # Build one branch's permanent UI block (after Submit)
    # ------------------------------------------------------------------
    def _build_user_pipeline_frame(self, rec):
        # Each branch is a tab in the User-Pipelines Notebook. The
        # tab label tracks the branch's name var (so renaming the
        # branch renames its tab live). The inner LabelFrame keeps
        # the type indicator visible inside the tab body.
        outer = tk.LabelFrame(self.user_pipelines_host,
                              text=f"  [{rec['type'].get()}]  ",
                              font=("DejaVu Sans", 9, "bold"),
                              fg="#cc88ff",
                              bd=0, relief="flat", padx=4, pady=3)
        rec["frame"] = outer
        _tab_title = rec["name"].get().strip() or f"branch{len(self.user_pipelines)}"
        self.user_pipelines_host.add(outer, text=_tab_title)
        # Show the new tab immediately so the user sees the branch
        # they just created instead of staying on the previous tab.
        try:
            self.user_pipelines_host.select(outer)
        except Exception:
            pass

        # Live tab-title rename when the user edits the Name entry.
        def _retitle_tab(*_a, _o=outer, _r=rec):
            try:
                nm = _r["name"].get().strip() or "branch"
                self.user_pipelines_host.tab(_o, text=nm)
            except Exception:
                pass
        rec["name"].trace_add("write", _retitle_tab)
        # Refresh the bracketed type indicator on type changes.
        def _retitle_type(*_a, _o=outer, _r=rec):
            try:
                _o.config(text=f"[{_r['type'].get()}]")
            except Exception:
                pass
        rec["type"].trace_add("write", _retitle_type)

        hdr = tk.Frame(outer)
        hdr.pack(fill="x", padx=3, pady=2)

        tk.Label(hdr, text="Name:",
                 font=("DejaVu Sans", 8)).pack(side="left")
        tk.Entry(hdr, textvariable=rec["name"], width=12,
                 font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Label(hdr, text=" -> up_<name>",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(side="left")

        # Source view (read-only display in the header — full
        # cascading picker lives in its own row below).
        tk.Label(hdr, text="  Source:",
                 font=("DejaVu Sans", 8)).pack(side="left")
        tk.Label(hdr, textvariable=rec["source"],
                 font=("DejaVu Sans", 8, "bold"), fg="#aaccff",
                 width=22, anchor="w").pack(side="left", padx=2)

        tk.Label(hdr, text=" Colour:",
                 font=("DejaVu Sans", 8)).pack(side="left", padx=(8, 0))
        _br_swatch = tk.Label(hdr, text="    ", width=3,
                              relief="solid", bd=1,
                              bg=rec["color"].get())
        _br_swatch.pack(side="left", padx=(2, 2))
        tk.Entry(hdr, textvariable=rec["color"], width=9,
                 font=("DejaVu Sans", 8)).pack(side="left", padx=1)
        tk.Button(hdr, text="Pick...", font=("DejaVu Sans", 8),
                  command=lambda v=rec["color"]: self._pick_color_for(
                      v, "Pick branch colour")
                  ).pack(side="left", padx=1)
        def _br_sync_swatch(*_a, _s=_br_swatch, _v=rec["color"]):
            hx = _v.get().strip()
            if hx and not hx.startswith("#"):
                hx = "#" + hx
            try:
                _s.config(bg=hx)
            except tk.TclError:
                pass
        rec["color"].trace_add("write", _br_sync_swatch)

        tk.Button(hdr, text="[edit] Edit detection",
                  font=("DejaVu Sans", 8),
                  command=lambda r=rec: self._edit_branch_detection(r)
                  ).pack(side="right", padx=4)
        tk.Button(hdr, text="x Delete", fg="#ff4444",
                  font=("DejaVu Sans", 8, "bold"),
                  command=lambda r=rec: self._on_remove_user_pipeline(r)
                  ).pack(side="right", padx=4)

        # -- Inline Source picker (cascading From -> Step), so the
        # source can be re-pointed without opening the modal. --
        src_lf = tk.LabelFrame(outer, text="Source view (pick From -> Step)",
                               font=("DejaVu Sans", 9, "italic"), fg="#aaccff")
        src_lf.pack(fill="x", padx=3, pady=(0, 2))
        self._make_cascading_picker(
            src_lf, rec["source"],
            pipeline_list=[],
            label_from="From:", label_step="Step:",
            allow_none=False)

        # -- Per-branch options row (BG subtractor + Overlay output) --
        opt_row = tk.Frame(outer)
        opt_row.pack(fill="x", padx=6, pady=(0, 2))
        tk.Checkbutton(opt_row, text="Use BG sub",
                       variable=rec["use_bgsub"],
                       font=("DejaVu Sans", 8)).pack(side="left")
        # BG-sub channel: "rgb"/"ir" use the full BGR frame, while
        # "H"/"S"/"V" feed JUST that single HSV channel into MOG2 so
        # you can subtract (say) saturation drift without colour or
        # brightness noise interfering.
        ttk.Combobox(opt_row, textvariable=rec["bgsub_src"],
                     values=["rgb", "ir", "H", "S", "V"],
                     width=5, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Label(opt_row, text="hist:",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(side="left", padx=(4, 0))
        tk.Spinbox(opt_row, textvariable=rec["bgsub_history"],
                   from_=10, to=10000, increment=50,
                   width=5, font=("DejaVu Sans", 8)).pack(side="left", padx=1)
        tk.Label(opt_row, text="varTh:",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(side="left", padx=(4, 0))
        tk.Spinbox(opt_row, textvariable=rec["bgsub_varth"],
                   from_=1, to=200, increment=1,
                   width=4, font=("DejaVu Sans", 8)).pack(side="left", padx=1)
        tk.Button(opt_row, text="Reset BG", font=("DejaVu Sans", 9),
                  command=lambda r=rec: self._reset_branch_bgsub(r)
                  ).pack(side="left", padx=(2, 0))
        tk.Label(opt_row, text="   Output:",
                 font=("DejaVu Sans", 8)).pack(side="left", padx=(12, 0))
        ttk.Combobox(opt_row, textvariable=rec["overlay"],
                     values=["mask", "overlay_rgb",
                             "overlay_hsv", "overlay_ir"],
                     width=14, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Label(opt_row,
                 text="(mask -> coloured binary,  overlay_rgb/hsv/ir -> "
                      "paint mask onto the chosen base image)",
                 font=("DejaVu Sans", 9), fg="#c8c8c8"
                 ).pack(side="left", padx=4)

        # Compact detection summary row (read-only display).
        det_summary = tk.Label(outer, text=self._branch_summary(rec),
                               font=("DejaVu Sans", 9, "italic"),
                               fg="#aa88cc", anchor="w", justify="left")
        det_summary.pack(fill="x", padx=6)
        rec["det_summary"] = det_summary
        # Refresh summary text whenever any config var changes.
        for _key in ("type", "channels",
                     "h1_lo", "h1_hi", "h2_lo", "h2_hi",
                     "s_lo",  "s_hi",  "v_lo",  "v_hi",
                     "ir_lo", "ir_hi",
                     "source",
                     "use_bgsub", "bgsub_src", "overlay"):
            rec[_key].trace_add(
                "write",
                lambda *a, r=rec, lbl=det_summary:
                    lbl.config(text=self._branch_summary(r)))

        # -- Inline detection sliders (type-aware) ---------------------
        # RGB-type branches see HSV sliders, IR-type branches see the
        # IR threshold slider, custom branches show nothing here.
        det_inline = tk.LabelFrame(outer, text="Detection (live)",
                                   font=("DejaVu Sans", 8, "bold"),
                                   fg="#aa88cc")
        det_inline.pack(fill="x", padx=3, pady=2)
        rec["det_inline_frame"] = det_inline

        def _slider_row(parent, label, var, lo_max=255):
            """Compact slider row matching the Edit-detection modal."""
            r = tk.Frame(parent)
            r.pack(fill="x", padx=4, pady=0)
            tk.Label(r, text=label, font=("DejaVu Sans", 8),
                     width=8, anchor="w").pack(side="left")
            tk.Spinbox(r, textvariable=var, from_=0, to=lo_max,
                       width=5, font=("DejaVu Sans", 8)
                       ).pack(side="left", padx=2)
            # Fixed length (matches the Edit-detection modal sliders)
            # - no expand=True, so the slider doesn't stretch to fill
            # the whole branch width.
            tk.Scale(r, variable=var, from_=0, to=lo_max,
                     orient="horizontal", length=140,
                     showvalue=False).pack(side="left", padx=2)
            return r

        # Pre-blur Type/K/Sigma rows were removed — use the branch's
        # PM (Pre-morph) Steps panel below instead. A PM Blur step
        # gives the same behaviour with full ordering control, and
        # any other PM op (CLAHE, Gamma, Sharpen, etc.) is now
        # available too.
        tk.Label(det_inline,
                 text="Pre-blur moved to PM Steps  ->  scroll down "
                      "to 'Branch PM Steps' and click + Add PM Step.",
                 font=("DejaVu Sans", 9, "italic"), fg="#aaccaa",
                 anchor="w", justify="left", wraplength=460
                 ).pack(fill="x", padx=4, pady=(2, 2))

        # Build the HSV section (rgb-type)
        hsv_frame = tk.Frame(det_inline)
        # Channel-mode dropdown
        ch_row = tk.Frame(hsv_frame)
        ch_row.pack(fill="x", padx=4, pady=2)
        tk.Label(ch_row, text="Channels:", font=("DejaVu Sans", 8, "bold"),
                 width=10, anchor="w").pack(side="left")
        ttk.Combobox(ch_row, textvariable=rec["channels"],
                     values=["H", "S", "V", "HS", "HV", "SV", "HSV", "full"],
                     width=8, state="readonly",
                     font=("DejaVu Sans", 8)).pack(side="left")
        # Build slider rows but keep references so we can show/hide
        # them based on the Channels mode (H mode -> only H rows,
        # S mode -> only S rows, etc.).
        h1_lo_r = _slider_row(hsv_frame, "H1 lo", rec["h1_lo"], lo_max=180)
        h1_hi_r = _slider_row(hsv_frame, "H1 hi", rec["h1_hi"], lo_max=180)
        h2_lo_r = _slider_row(hsv_frame, "H2 lo", rec["h2_lo"], lo_max=180)
        h2_hi_r = _slider_row(hsv_frame, "H2 hi", rec["h2_hi"], lo_max=180)
        s_lo_r  = _slider_row(hsv_frame, "S min", rec["s_lo"])
        s_hi_r  = _slider_row(hsv_frame, "S max", rec["s_hi"])
        v_lo_r  = _slider_row(hsv_frame, "V min", rec["v_lo"])
        v_hi_r  = _slider_row(hsv_frame, "V max", rec["v_hi"])

        h_rows = (h1_lo_r, h1_hi_r, h2_lo_r, h2_hi_r)
        s_rows = (s_lo_r, s_hi_r)
        v_rows = (v_lo_r, v_hi_r)

        def _refresh_channel_visibility(*_):
            mode = rec["channels"].get() or "HSV"
            show_h = ("H" in mode) or mode == "full"
            show_s = ("S" in mode) or mode == "full"
            show_v = ("V" in mode) or mode == "full"
            for r in h_rows:
                if show_h:
                    r.pack(fill="x", padx=4, pady=0)
                else:
                    r.pack_forget()
            for r in s_rows:
                if show_s:
                    r.pack(fill="x", padx=4, pady=0)
                else:
                    r.pack_forget()
            for r in v_rows:
                if show_v:
                    r.pack(fill="x", padx=4, pady=0)
                else:
                    r.pack_forget()
        rec["channels"].trace_add("write", _refresh_channel_visibility)
        _refresh_channel_visibility()

        # Build the IR section (ir-type)
        ir_frame = tk.Frame(det_inline)
        _slider_row(ir_frame, "IR lo", rec["ir_lo"])
        _slider_row(ir_frame, "IR hi", rec["ir_hi"])

        # Dedicated IR CLAHE — applied to the branch's IR gray feed
        # BEFORE PM steps and threshold (so the branch detection sees
        # contrast-enhanced input). Stacks with any PM-step CLAHE.
        tk.Label(ir_frame, text="-- IR CLAHE --",
                 font=("DejaVu Sans", 8, "bold"), fg="#aa88cc",
                 anchor="w").pack(fill="x", padx=4, pady=(4, 0))
        tk.Checkbutton(ir_frame, text="Use IR CLAHE",
                       variable=rec["use_clahe_ir"],
                       font=("DejaVu Sans", 8)).pack(anchor="w", padx=4)
        _bc_row1 = tk.Frame(ir_frame); _bc_row1.pack(fill="x", padx=4, pady=0)
        tk.Label(_bc_row1, text="Clip",
                 font=("DejaVu Sans", 8), width=8, anchor="w").pack(side="left")
        tk.Spinbox(_bc_row1, textvariable=rec["clahe_ir_clip"],
                   from_=1, to=40, width=5,
                   font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Scale(_bc_row1, variable=rec["clahe_ir_clip"], from_=1, to=40,
                 orient="horizontal", length=120,
                 showvalue=False).pack(side="left", padx=2)
        _bc_row2 = tk.Frame(ir_frame); _bc_row2.pack(fill="x", padx=4, pady=0)
        tk.Label(_bc_row2, text="Tile",
                 font=("DejaVu Sans", 8), width=8, anchor="w").pack(side="left")
        tk.Spinbox(_bc_row2, textvariable=rec["clahe_ir_tile"],
                   from_=1, to=32, width=5,
                   font=("DejaVu Sans", 8)).pack(side="left", padx=2)
        tk.Scale(_bc_row2, variable=rec["clahe_ir_tile"], from_=1, to=32,
                 orient="horizontal", length=120,
                 showvalue=False).pack(side="left", padx=2)

        # Custom-type placeholder
        custom_frame = tk.Frame(det_inline)
        tk.Label(custom_frame,
                 text="Custom branch - no detection sliders. "
                      "The Source view is used directly as the input.",
                 font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
                 wraplength=480, justify="left"
                 ).pack(anchor="w", padx=6, pady=4)

        def _refresh_inline_det(*_):
            t = rec["type"].get()
            for fr in (hsv_frame, ir_frame, custom_frame):
                fr.pack_forget()
            if t == "rgb":
                hsv_frame.pack(fill="x")
            elif t == "ir":
                ir_frame.pack(fill="x")
            else:
                custom_frame.pack(fill="x")
        rec["type"].trace_add("write", _refresh_inline_det)
        _refresh_inline_det()

        # -- Branch PM (Pre-morph) Steps — image-stage pipeline ----
        # Runs in user order on the branch's SOURCE image before
        # HSV / IR detection. Same compact card UI as the main RGB
        # / IR PM panels. Use this for Blur, CLAHE, Gamma, Sharpen,
        # etc. The morph step cards below run on the binary mask.
        pre_steps_lf = tk.LabelFrame(
            outer, text="Branch PM Steps  (on source image, BEFORE detection)",
            font=("DejaVu Sans", 9, "bold"), fg="#88dd88")
        pre_steps_lf.pack(fill="x", padx=3, pady=(2, 1))
        rec["pre_steps_row"] = pre_steps_lf
        self._rebuild_pre_morph_pipeline_ui(pre_steps_lf, rec["pre_steps"])

        steps_lf = tk.LabelFrame(
            outer, text="Branch Morph Steps  (on mask, AFTER detection)",
            font=("DejaVu Sans", 9, "bold"), fg="#4488ff")
        steps_lf.pack(fill="x", padx=3, pady=(0, 2))
        rec["steps_row"] = steps_lf
        self._rebuild_pipeline_ui(steps_lf, rec["steps"])
        # A branch added live needs its freshly built widgets matched
        # to the active theme + font settings.
        if hasattr(self, "_restyle_subtree"):
            self._restyle_subtree(outer)

    def _branch_summary(self, rec):
        t = rec["type"].get()
        bg = (f"  bgsub({rec['bgsub_src'].get()}"
              f", h={rec['bgsub_history'].get()}"
              f", vT={rec['bgsub_varth'].get()})"
              if rec.get("use_bgsub") and rec["use_bgsub"].get()
              else "")
        ov = (f"  out={rec['overlay'].get()}"
              if rec.get("overlay") and rec["overlay"].get() != "mask"
              else "")
        _pre_n = len(rec.get("pre_steps") or [])
        pb = f"  pm={_pre_n}" if _pre_n else ""
        suf = pb + bg + ov
        if t == "rgb":
            ch = rec["channels"].get()
            return (f"  type=RGB  channels={ch}{suf}  "
                    f"H1=[{rec['h1_lo'].get()}..{rec['h1_hi'].get()}]  "
                    f"H2=[{rec['h2_lo'].get()}..{rec['h2_hi'].get()}]  "
                    f"S=[{rec['s_lo'].get()}..{rec['s_hi'].get()}]  "
                    f"V=[{rec['v_lo'].get()}..{rec['v_hi'].get()}]  "
                    f"-> up_{rec['name'].get()}_det")
        if t == "ir":
            return (f"  type=IR{suf}  "
                    f"thresh=[{rec['ir_lo'].get()}..{rec['ir_hi'].get()}]  "
                    f"-> up_{rec['name'].get()}_det")
        return (f"  type=Custom{suf}  source={rec['source'].get()}  "
                f"(no detection)")

    def _reset_branch_bgsub(self, rec):
        """Drop the branch's MOG2 instance so it's rebuilt next frame
        with the current params (and freshly-trained from this frame)."""
        rec["_backSub"]     = None
        rec["_backSub_sig"] = None
        self._refresh()

    def _edit_branch_detection(self, rec):
        """Re-open the setup dialog for an existing branch."""
        self._open_branch_setup_dialog(
            rec,
            on_submit=lambda r: (self._rebuild_all_masks_if_open(),
                                 self._refresh()))
