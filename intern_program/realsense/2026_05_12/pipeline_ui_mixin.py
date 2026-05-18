"""Mixin: dynamic pipeline-step UI (the row of "Step N" cards in the side panel).

Concerns: building/destroying step frames, add/remove/move buttons, default
step seeding. All methods operate on a `pipeline_list` of 10-tuples of tk
variables stored on the host class.
"""
import tkinter as tk
from tkinter import ttk

from config import ALL_PROC_OPS, OP_GROUPS, group_for_op, params_for_op

# Selectable OVERLAY paint colours - name -> BGR tuple (used by _process).
OVERLAY_COLORS = {
    "red":     (  0,   0, 255),
    "green":   (  0, 255,   0),
    "blue":    (255,   0,   0),
    "cyan":    (255, 255,   0),
    "magenta": (255,   0, 255),
    "yellow":  (  0, 255, 255),
    "orange":  (  0, 165, 255),
    "purple":  (128,   0, 128),
    "white":   (255, 255, 255),
}


class PipelineUIMixin:
    """Adds dynamic pipeline-step UI methods to the host class.

    Step tuple layout (10 vars):
        (en, op, n, dir, kx, ky, thresh, comb_en, comb_op, comb_src)

    Host class must provide:
        self._refresh()
        self._rebuild_all_masks_if_open()
    """

    # ------------------------------------------------------------------
    # Reusable cascading source-view picker (From: + Step:)
    # ------------------------------------------------------------------
    def _make_cascading_picker(self, parent, target_var, pipeline_list,
                               label_from="From:", label_step="Step:",
                               allow_none=False):
        """Build two cascading dropdowns. The user picks a pipeline
        GROUP first (this-pipe, Raw, RGB, IR, branch:<name>, optionally
        none), then narrows to a specific step or view. The composed
        view name is written to `target_var`.

        Returns (src_row, view_row) so the caller can pack/forget them.
        """
        grp_var  = tk.StringVar(value="(this pipe)")
        view_var = tk.StringVar(value=target_var.get() or "prev")

        src_row = tk.Frame(parent)
        src_row.pack(fill="x", pady=1)
        tk.Label(src_row, text=label_from, font=("Arial", 8),
                 width=8, anchor="w").pack(side="left")
        cb_grp = ttk.Combobox(src_row, textvariable=grp_var,
                              width=14, state="readonly",
                              font=("Arial", 8))
        cb_grp.pack(side="left", padx=1)

        view_row = tk.Frame(parent)
        view_row.pack(fill="x", pady=1)
        tk.Label(view_row, text=label_step, font=("Arial", 8),
                 width=8, anchor="w").pack(side="left")
        cb_view = ttk.Combobox(view_row, textvariable=view_var,
                               width=22, state="readonly",
                               font=("Arial", 8))
        cb_view.pack(side="left")

        def _groups():
            g = []
            if allow_none:
                g.append("none")
            g += ["(this pipe)", "Raw inputs", "RGB pipeline", "IR pipeline"]
            for up in self.user_pipelines:
                nm = up["name"].get().strip()
                if nm:
                    g.append(f"branch:{nm}")
            return g

        def _views_for(grp):
            if grp == "none":
                return ["none"]
            if grp == "(this pipe)":
                return (["mask_pre", "prev"]
                        + [f"step_{i}"
                           for i in range(1, len(pipeline_list) + 1)])
            if grp == "Raw inputs":
                return ["rgb_raw", "rgb_blur",
                        "rgb_hsv_full", "rgb_hsv_H",
                        "rgb_hsv_S",    "rgb_hsv_V",
                        "ir_raw", "ir_gray", "ir_blur", "ir_clahe"]
            if grp == "RGB pipeline":
                n = len(getattr(self, "rgb_pipeline", []))
                return (["rgb_m1", "rgb_m2", "rgb_hsv_mask",
                         "rgb_bgsub", "rgb_mask_pre"]
                        + [f"rgb_step{i}" for i in range(1, n + 1)]
                        + ["rgb_mask", "rgb_post_blur"])
            if grp == "IR pipeline":
                n = len(getattr(self, "ir_pipeline", []))
                return (["ir_thresh", "ir_mask_pre"]
                        + [f"ir_step{i}" for i in range(1, n + 1)]
                        + ["ir_mask", "ir_post_blur"])
            if grp.startswith("branch:"):
                bnm = grp[len("branch:"):]
                base = [f"up_{bnm}", f"up_{bnm}_det",
                        f"up_{bnm}_mask",
                        f"up_{bnm}_overlay_rgb",
                        f"up_{bnm}_overlay_hsv",
                        f"up_{bnm}_overlay_ir",
                        f"up_{bnm}_bgsub",
                        f"up_{bnm}_h1_mask", f"up_{bnm}_h2_mask",
                        f"up_{bnm}_s_mask",  f"up_{bnm}_v_mask",
                        f"up_{bnm}_ir_mask"]
                for up in self.user_pipelines:
                    if up["name"].get().strip() == bnm:
                        base += [f"up_{bnm}_step{i}"
                                 for i in range(1, len(up["steps"]) + 1)]
                        break
                return base
            return []

        def _populate_from_existing():
            v = target_var.get()
            if allow_none and (v == "none" or v == ""):
                grp_var.set("none")
                view_var.set("none")
                return
            if v in ("mask_pre", "prev") or v.startswith("step_"):
                grp_var.set("(this pipe)")
            elif v in ("rgb_raw", "rgb_blur", "rgb_hsv_full",
                       "rgb_hsv_H", "rgb_hsv_S", "rgb_hsv_V",
                       "ir_raw", "ir_gray", "ir_blur", "ir_clahe"):
                grp_var.set("Raw inputs")
            elif v.startswith("rgb_"):
                grp_var.set("RGB pipeline")
            elif v.startswith("ir_"):
                grp_var.set("IR pipeline")
            elif v.startswith("up_"):
                bare = v[len("up_"):]
                best = ""
                for up in self.user_pipelines:
                    nm = up["name"].get().strip()
                    if nm and (bare == nm or bare.startswith(nm + "_")) \
                       and len(nm) > len(best):
                        best = nm
                grp_var.set(f"branch:{best}" if best else "(this pipe)")
            else:
                grp_var.set("(this pipe)")
            view_var.set(v if v else _views_for(grp_var.get())[0])

        def _refresh_grp_options(*_):
            cb_grp.configure(values=_groups())

        def _refresh_view_options(*_):
            opts = _views_for(grp_var.get())
            cb_view.configure(values=opts)
            if view_var.get() not in opts and opts:
                view_var.set(opts[0])

        def _commit(*_):
            target_var.set(view_var.get())

        cb_grp.bind("<Button-1>",   lambda e: _refresh_grp_options())
        cb_view.bind("<Button-1>",  lambda e: _refresh_view_options())
        grp_var.trace_add("write",  _refresh_view_options)
        view_var.trace_add("write", _commit)

        _populate_from_existing()
        _refresh_grp_options()
        _refresh_view_options()
        return src_row, view_row

    # ------------------------------------------------------------------
    # Per-step OVERLAY extras (color1, color2, optional 2nd mask source)
    # ------------------------------------------------------------------
    def _yolo_state_for(self, step_tuple):
        """Per-step YOLO-add-on state. Lazy-create.

        Fields:
          yolo_en   - toggle YOLO inference on this step
          yolo_src  - RAW source view to feed YOLO
          yolo_mode - "box_only" (just draws boxes, no effect on mask),
                      "focus"    (running mask = running AND boxes),
                      "subtract" (running mask = running AND NOT boxes)
        """
        if not hasattr(self, "_yolo_state"):
            self._yolo_state = {}
        key = id(step_tuple)
        st = self._yolo_state.get(key)
        if st is None:
            st = {
                "yolo_en":   tk.BooleanVar(value=False),
                "yolo_src":  tk.StringVar(value="rgb_raw"),
                "yolo_mode": tk.StringVar(value="box_only"),
            }
            # yolo_en toggles whether the step gets a YOLO-output
            # sub-thumbnail in the flowchart, so rebuild the
            # All-Masks window when it changes. yolo_src/yolo_mode
            # only affect processing, not the flowchart layout.
            st["yolo_en"].trace_add(
                "write",
                lambda *a: (self._rebuild_all_masks_if_open(),
                            self._refresh()))
            for k, v in st.items():
                if k != "yolo_en":
                    v.trace_add("write", lambda *a: self._refresh())
            self._yolo_state[key] = st
        return st

    def _overlay_state_for(self, step_tuple):
        """Return the per-step OVERLAY state dict (lazy-create).

        Fields:
          color1    - paint colour for Mask 1 (running mask)
          color2    - paint colour for Mask 2
          mask2_src - optional 2nd-mask view ("none" disables)
          base_src  - BGR image painted on ("none" = black bg)
        """
        if not hasattr(self, "_overlay_state"):
            self._overlay_state = {}
        key = id(step_tuple)
        st = self._overlay_state.get(key)
        if st is None:
            st = {
                "color1":    tk.StringVar(value="red"),    # Mask 1
                "color2":    tk.StringVar(value="cyan"),   # Mask 2
                "mask2_src": tk.StringVar(value="none"),
                "base_src":  tk.StringVar(value="rgb_raw"),
            }
            # Toggling mask2/base between "none" and a view changes the
            # composite tile count -> rebuild the All-Masks window so
            # the thumbnail width matches the new count.
            for _name, _v in st.items():
                if _name in ("mask2_src", "base_src"):
                    _v.trace_add(
                        "write",
                        lambda *a: (self._rebuild_all_masks_if_open(),
                                    self._refresh()))
                else:
                    _v.trace_add("write", lambda *a: self._refresh())
            self._overlay_state[key] = st
        return st

    def _overlay_count_for_vid(self, vid):
        """How many tiles the OVERLAY composite has for this step
        (2 = Mask 1 + overlay, 3 = + Mask 2 OR Base, 4 = all four)."""
        state = None
        try:
            if vid.startswith("rgb_step"):
                _i = int(vid[len("rgb_step"):]) - 1
                if 0 <= _i < len(self.rgb_pipeline):
                    state = getattr(self, "_overlay_state", {}).get(
                        id(self.rgb_pipeline[_i]))
            elif vid.startswith("ir_step"):
                _i = int(vid[len("ir_step"):]) - 1
                if 0 <= _i < len(self.ir_pipeline):
                    state = getattr(self, "_overlay_state", {}).get(
                        id(self.ir_pipeline[_i]))
            elif vid.startswith("up_") and "_step" in vid:
                _bare = vid[len("up_"):]
                _nm, _, _stxt = _bare.rpartition("_step")
                _ix = int(_stxt) - 1
                for up in getattr(self, "user_pipelines", []):
                    if (up["name"].get().strip() == _nm
                            and 0 <= _ix < len(up["steps"])):
                        state = getattr(self, "_overlay_state", {}).get(
                            id(up["steps"][_ix]))
                        break
        except Exception:
            pass
        if state is None:
            return 4   # safe default = full layout
        n = 1   # Mask 1 always
        if state["mask2_src"].get() not in ("", "none"):
            n += 1
        if state["base_src"].get() not in ("", "none"):
            n += 1
        return n + 1   # + final overlay tile

    # ------------------------------------------------------------------
    # Default seeding
    # ------------------------------------------------------------------
    def _attach_step_traces(self, step_tuple):
        """Trace every step var so editing any field (Enable, Op,
        N, Dir, KX, KY, T, combine_en/op/src) immediately re-runs the
        pipeline and updates the All-Masks flowchart labels."""
        # Combine vars also need a flowchart rebuild because they change
        # whether the step shows the image1/+op/image2 sub-row.
        en, op, n, dir_, kx, ky, th, cen, cop, csr = step_tuple
        for v in (en, op, n, dir_, kx, ky, th):
            v.trace_add("write", lambda *a: self._refresh())
        for v in (cen, cop, csr):
            v.trace_add("write",
                        lambda *a: (self._rebuild_all_masks_if_open(),
                                    self._refresh()))

    def _create_default_steps(self, pipeline_list, defaults):
        """Append default steps to `pipeline_list` from a list of 7-tuples."""
        for en_d, op_d, n_d, dir_d, kx_d, ky_d, t_d in defaults:
            tup = (
                tk.BooleanVar(value=en_d),
                tk.StringVar(value=op_d),
                tk.IntVar(value=n_d),
                tk.StringVar(value=dir_d),
                tk.IntVar(value=kx_d),
                tk.IntVar(value=ky_d),
                tk.IntVar(value=t_d),
                tk.BooleanVar(value=False),     # combine_en
                tk.StringVar(value="AND"),      # combine_op
                tk.StringVar(value="mask_pre"), # combine_src
            )
            pipeline_list.append(tup)
            self._attach_step_traces(tup)

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------
    # How many step cards before we wrap to a new row.
    STEPS_PER_UI_ROW = 5

    def _grid_step_card(self, sf, i):
        """Place an existing step card `sf` at grid cell (i // 5, i % 5)."""
        sf.grid(row=i // self.STEPS_PER_UI_ROW,
                column=i % self.STEPS_PER_UI_ROW,
                padx=5, pady=3, sticky="n")

    def _place_add_btn(self, parent_frame, pipeline_list):
        """(Re)place the '+ Add Step' button right after the last card."""
        # Remove old button if it exists.
        old = getattr(parent_frame, "_add_btn", None)
        if old is not None and str(old) != "":
            try:
                old.destroy()
            except Exception:
                pass
        i = len(pipeline_list)
        btn = tk.Button(parent_frame, text="+ Add Step",
                        font=("Arial", 8), bg="#223322", fg="white",
                        command=lambda pl=pipeline_list, pf=parent_frame:
                            self._on_add_step(pf, pl))
        btn.grid(row=i // self.STEPS_PER_UI_ROW,
                 column=i % self.STEPS_PER_UI_ROW,
                 padx=8, pady=10, sticky="n")
        parent_frame._add_btn = btn

    def _rebuild_pipeline_ui(self, parent_frame, pipeline_list):
        """Full rebuild - used on remove / move where indexes shift."""
        for w in parent_frame.winfo_children():
            w.destroy()
        parent_frame._add_btn = None
        for i in range(len(pipeline_list)):
            sf = self._add_step_frame(parent_frame, pipeline_list, i)
            self._grid_step_card(sf, i)
        self._place_add_btn(parent_frame, pipeline_list)

    def _add_step_frame(self, parent_frame, pipeline_list, idx):
        en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var, \
            comb_en_var, comb_op_var, comb_src_var = pipeline_list[idx]

        sf = tk.LabelFrame(parent_frame, text=f"Step {idx + 1}", font=("Arial", 8))
        # Caller decides placement (grid). Do NOT pack here.

        # Reorder / remove buttons
        btn_row = tk.Frame(sf)
        btn_row.pack(anchor="ne", padx=1, pady=1)
        tk.Button(btn_row, text="^", font=("Arial", 7), width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_move_step(pf, pl, i, -1)).pack(side="left")
        tk.Button(btn_row, text="v", font=("Arial", 7), width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_move_step(pf, pl, i, +1)).pack(side="left")
        tk.Button(btn_row, text="x", font=("Arial", 8, "bold"), fg="#ff4444",
                  width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_remove_step(pf, pl, i)).pack(side="left")

        tk.Checkbutton(sf, text="Enable", variable=en_var,
                       font=("Arial", 8)).pack(anchor="w")

        # -- Step TYPE radio: Morph (apply Op) vs Combine (bitwise) ----
        # comb_en_var doubles as the type flag:
        #   False -> Morph step (Op + kernel applied to running mask)
        #   True  -> Combine step (running <AND/OR/XOR> source, no Op)
        type_row = tk.Frame(sf)
        type_row.pack(fill="x", pady=1)
        tk.Label(type_row, text="Type:", font=("Arial", 8, "bold"),
                 width=5, anchor="w").pack(side="left")
        # We store True/False in comb_en_var via two radio buttons.
        morph_rb = tk.Radiobutton(type_row, text="Morph", value=False,
                                  variable=comb_en_var,
                                  font=("Arial", 8), fg="#aaccff")
        morph_rb.pack(side="left")
        comb_rb  = tk.Radiobutton(type_row, text="Combine", value=True,
                                  variable=comb_en_var,
                                  font=("Arial", 8, "bold"), fg="#bb66ff")
        comb_rb.pack(side="left")

        def _row(p, lbl, wfn):
            r = tk.Frame(p)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, font=("Arial", 8),
                     width=4, anchor="w").pack(side="left")
            wfn(r)

        # -- Morph fields: Op (Group -> Op), N, Dir, KX, KY, T ---------
        morph_fr = tk.Frame(sf)
        morph_fr.pack(fill="x")

        # Cascading op picker: pick the GROUP first (Morphology /
        # Blur / Threshold / Illumination / Edge / Bitwise / Misc),
        # then narrow to the actual op. Drives op_var on change.
        grp_op_var = tk.StringVar(value=group_for_op(op_var.get()))

        def _grp_row(r):
            tk.Label(r, text="Grp:", font=("Arial", 8),
                     width=4, anchor="w").pack(side="left")
            cb_g = ttk.Combobox(r, textvariable=grp_op_var,
                                values=list(OP_GROUPS.keys()),
                                width=14, state="readonly",
                                font=("Arial", 8))
            cb_g.pack(side="left")

        def _op_row(r, v=op_var):
            tk.Label(r, text="Op:", font=("Arial", 8),
                     width=4, anchor="w").pack(side="left")
            cb_o = ttk.Combobox(r, textvariable=v,
                                values=OP_GROUPS[grp_op_var.get()],
                                width=14, state="readonly",
                                font=("Arial", 8))
            cb_o.pack(side="left")

            def _on_grp_change(*_):
                ops = OP_GROUPS.get(grp_op_var.get(), ALL_PROC_OPS)
                cb_o.configure(values=ops)
                if v.get() not in ops and ops:
                    v.set(ops[0])
            grp_op_var.trace_add("write", _on_grp_change)

        # Use _row helper for "Grp:" and "Op:".
        gr_row = tk.Frame(morph_fr); gr_row.pack(fill="x", pady=1)
        _grp_row(gr_row)
        op_r   = tk.Frame(morph_fr); op_r.pack(fill="x", pady=1)
        _op_row(op_r)
        # Param rows are created up-front but only shown for the
        # ops that actually use them. The labels also update per op
        # (e.g. KX -> "Kernel size" for GaussBlur, "Low threshold"
        # for Canny, "Diameter" for BilateralBlur, etc.).
        param_rows = {}

        def _param_row(key, default_label, var, kind="spin",
                       lo=1, hi=99, opts=None):
            r = tk.Frame(morph_fr)
            lbl = tk.Label(r, text=default_label + ":",
                           font=("Arial", 8), width=14, anchor="w")
            lbl.pack(side="left")
            if kind == "combo":
                ttk.Combobox(r, textvariable=var, values=opts,
                             width=5, state="readonly",
                             font=("Arial", 8)).pack(side="left")
            else:
                tk.Spinbox(r, textvariable=var, from_=lo, to=hi,
                           width=6, font=("Arial", 8)
                           ).pack(side="left")
            param_rows[key] = (r, lbl)
            return r

        _param_row("N",   "N",            n_var,     lo=1, hi=20)
        _param_row("Dir", "Direction",    dir_var,   kind="combo",
                   opts=["X", "Y", "XY"])
        _param_row("KX",  "KX",           kx_var,    lo=1, hi=999)
        _param_row("KY",  "KY",           ky_var,    lo=1, hi=999)
        _param_row("T",   "Threshold",    thresh_var, lo=0, hi=255)

        # One-line description of what THIS op actually does. Updates
        # whenever op_var changes.
        op_desc_lbl = tk.Label(morph_fr, text="",
                               font=("Arial", 7, "italic"),
                               fg="#aaccaa", anchor="w",
                               wraplength=240, justify="left")
        op_desc_lbl.pack(fill="x", padx=4, pady=(2, 2))

        def _refresh_op_params(*_):
            spec = params_for_op(op_var.get())
            visible = set(spec.get("params", []))
            label_map = {
                "N":  spec.get("n_lbl",  "N (iterations)"),
                "Dir": spec.get("dir_lbl", "Direction"),
                "KX": spec.get("kx_lbl", "KX"),
                "KY": spec.get("ky_lbl", "KY"),
                "T":  spec.get("t_lbl",  "Threshold"),
            }
            # Show the op description (or hide if none).
            _desc = spec.get("desc", "")
            if _desc:
                op_desc_lbl.config(text="(i) " + _desc)
            else:
                op_desc_lbl.config(text="")
            for key, (row, lbl) in param_rows.items():
                if key in visible:
                    lbl.config(text=label_map[key] + ":")
                    row.pack(fill="x", pady=1)
                else:
                    row.pack_forget()
            if not visible:
                # Inform the user that this op has no parameters.
                if "_no_params_lbl" not in param_rows:
                    nl = tk.Label(morph_fr,
                                  text="(this op has no parameters)",
                                  font=("Arial", 7, "italic"), fg="#888")
                    nl.pack(anchor="w", padx=4)
                    param_rows["_no_params_lbl"] = (nl, nl)
                else:
                    param_rows["_no_params_lbl"][0].pack(anchor="w", padx=4)
            else:
                if "_no_params_lbl" in param_rows:
                    param_rows["_no_params_lbl"][0].pack_forget()
        op_var.trace_add("write", _refresh_op_params)
        _refresh_op_params()

        # -- Combine fields: bitwise op + source view ------------------
        combine_fr = tk.Frame(sf)
        combine_fr.pack(fill="x")
        cop_row = tk.Frame(combine_fr)
        cop_row.pack(fill="x", pady=1)
        tk.Label(cop_row, text="Op:", font=("Arial", 8),
                 width=4, anchor="w").pack(side="left")
        ttk.Combobox(cop_row, textvariable=comb_op_var,
                     values=["AND", "OR", "XOR", "OVERLAY"],
                     width=8, state="readonly",
                     font=("Arial", 8, "bold")).pack(side="left", padx=2)
        tk.Label(combine_fr,
                 text="OVERLAY = paint running mask on top of src "
                      "(useful with rgb_raw / ir_raw / hsv_full)",
                 font=("Arial", 6, "italic"), fg="#888"
                 ).pack(anchor="w", padx=3)

        # -- Combine source: TWO cascading dropdowns (From + Step). -
        src_row, view_row = self._make_cascading_picker(
            combine_fr, comb_src_var, pipeline_list,
            label_from="From:", label_step="Step:",
            allow_none=False)
        src_caption = tk.Label(combine_fr,
                               text="result = running <op> src",
                               font=("Arial", 6, "italic"),
                               fg="#bb66ff")
        src_caption.pack(anchor="w", padx=3)

        # -- OVERLAY-only extras -----------------------------------
        # Mask 1 is always the running mask (output of the previous
        # step) painted in C2.  Mask 2 is an optional second mask
        # painted in C1 ("none" disables it).  Base is the BGR image
        # painted on ("none" = black background).
        # The composite thumbnail adapts: 2 / 3 / 4 tiles depending
        # on which operands are active.
        ov_state = self._overlay_state_for(pipeline_list[idx])
        ov_fr = tk.Frame(combine_fr)

        def _base_options():
            return ["none", "rgb_raw", "rgb_blur",
                    "rgb_hsv_full", "rgb_hsv_H",
                    "rgb_hsv_S", "rgb_hsv_V",
                    "ir_raw", "ir_gray"]

        # Base view (BGR image painted on top of, or "none" = black)
        base_row = tk.Frame(ov_fr); base_row.pack(fill="x", pady=1)
        tk.Label(base_row, text="Base:", font=("Arial", 8, "bold"),
                 width=8, anchor="w").pack(side="left")
        cb_base = ttk.Combobox(base_row, textvariable=ov_state["base_src"],
                               values=_base_options(), width=18,
                               state="readonly", font=("Arial", 8))
        cb_base.pack(side="left")
        cb_base.bind("<Button-1>",
                     lambda e, c=cb_base: c.configure(values=_base_options()))

        # Mask 2 - cascading From/Step picker (same UX as the regular
        # combine source). "none" is a valid selection (top-level
        # group) which disables the 2nd mask.
        tk.Label(ov_fr, text="Mask 2  v  pick From -> Step:",
                 font=("Arial", 7, "bold"), fg="#cc88ff",
                 anchor="w").pack(fill="x", padx=3, pady=(4, 0))
        m2_src_row, m2_view_row = self._make_cascading_picker(
            ov_fr, ov_state["mask2_src"], pipeline_list,
            label_from="  From:", label_step="  Step:",
            allow_none=True)

        # C1 = colour for Mask 1 (the running mask from the previous step)
        col1_row = tk.Frame(ov_fr); col1_row.pack(fill="x", pady=1)
        tk.Label(col1_row, text="C1:", font=("Arial", 8, "bold"),
                 width=8, anchor="w").pack(side="left")
        ttk.Combobox(col1_row, textvariable=ov_state["color1"],
                     values=list(OVERLAY_COLORS.keys()),
                     width=10, state="readonly",
                     font=("Arial", 8)).pack(side="left")
        tk.Label(col1_row, text=" colour of Mask 1 (running)",
                 font=("Arial", 7), fg="#888"
                 ).pack(side="left", padx=2)

        # C2 = colour for Mask 2
        col2_row = tk.Frame(ov_fr); col2_row.pack(fill="x", pady=1)
        tk.Label(col2_row, text="C2:", font=("Arial", 8, "bold"),
                 width=8, anchor="w").pack(side="left")
        ttk.Combobox(col2_row, textvariable=ov_state["color2"],
                     values=list(OVERLAY_COLORS.keys()),
                     width=10, state="readonly",
                     font=("Arial", 8)).pack(side="left")
        tk.Label(col2_row, text=" colour of Mask 2",
                 font=("Arial", 7), fg="#888"
                 ).pack(side="left", padx=2)

        tk.Label(ov_fr,
                 text=("Composite shows: Mask 1 [+ Mask 2] [+ Base] -> "
                       "overlay. Mask 2 = none and Base = none are both "
                       "valid (compose only what's enabled)."),
                 font=("Arial", 6, "italic"), fg="#888",
                 wraplength=260, justify="left"
                 ).pack(anchor="w", padx=3)

        # -- YOLO add-on (optional, per step) -------------------------
        # Pinned to the bottom of the step card BEFORE the Morph /
        # Combine show/hide logic runs, so toggling Morph<->Combine
        # never swaps the order of these two regions.
        yolo_st = self._yolo_state_for(pipeline_list[idx])
        yolo_fr = tk.LabelFrame(sf, text="+ YOLO box detection",
                                font=("Arial", 7, "italic"),
                                fg="#ffaa66")
        yolo_fr.pack(side="bottom", fill="x", pady=(4, 1))

        # Show only the relevant block based on the radio state, AND
        # hide the From/Step pickers + caption when op == OVERLAY
        # (OVERLAY uses its own Base / Mask 2 selectors instead, so
        # the From/Step row would be confusing duplication).
        def _refresh_visibility(*_):
            # `before=yolo_fr` keeps the Morph / Combine block above
            # the YOLO block; without it pack_forget()/pack() flips
            # the ordering when the user toggles type.
            if comb_en_var.get():
                morph_fr.pack_forget()
                combine_fr.pack(fill="x", before=yolo_fr)
                if comb_op_var.get() == "OVERLAY":
                    src_row.pack_forget()
                    view_row.pack_forget()
                    src_caption.pack_forget()
                    ov_fr.pack(fill="x")
                else:
                    src_row.pack(fill="x", pady=1)
                    view_row.pack(fill="x", pady=1)
                    src_caption.pack(anchor="w", padx=3)
                    ov_fr.pack_forget()
            else:
                combine_fr.pack_forget()
                ov_fr.pack_forget()
                morph_fr.pack(fill="x", before=yolo_fr)
        comb_en_var.trace_add("write", _refresh_visibility)
        comb_op_var.trace_add("write", _refresh_visibility)
        _refresh_visibility()

        yolo_top = tk.Frame(yolo_fr); yolo_top.pack(fill="x", padx=2, pady=1)
        tk.Checkbutton(yolo_top, text="Enable",
                       variable=yolo_st["yolo_en"],
                       font=("Arial", 7, "bold"), fg="#ffaa66"
                       ).pack(side="left")
        tk.Label(yolo_top, text=" runs on:",
                 font=("Arial", 7), fg="#888").pack(side="left")
        # Cascading From / Step picker so the user can point YOLO at
        # any RAW frame in the program (rgb_raw, ir_raw, rgb_step3 ...).
        self._make_cascading_picker(
            yolo_fr, yolo_st["yolo_src"], pipeline_list,
            label_from="  From:", label_step="  Step:",
            allow_none=False)
        # Mode dropdown: how the YOLO box union affects the running mask.
        mode_row = tk.Frame(yolo_fr); mode_row.pack(fill="x", padx=2, pady=1)
        tk.Label(mode_row, text="  Mode:", font=("Arial", 7),
                 width=8, anchor="w").pack(side="left")
        ttk.Combobox(mode_row, textvariable=yolo_st["yolo_mode"],
                     values=["box_only", "focus", "subtract"],
                     width=10, state="readonly",
                     font=("Arial", 7)).pack(side="left")
        tk.Label(yolo_fr,
                 text=("box_only = just draw boxes  |  "
                       "focus = keep mask only inside boxes  |  "
                       "subtract = remove mask pixels inside boxes"),
                 font=("Arial", 6, "italic"), fg="#888",
                 wraplength=240, justify="left"
                 ).pack(anchor="w", padx=4)
        return sf

    # ------------------------------------------------------------------
    # Add / move / remove
    # ------------------------------------------------------------------
    def _on_add_step(self, parent_frame, pipeline_list):
        # Fast path: append-only. Existing cards keep their indexes,
        # so we just place ONE new card and re-position the "+ Add"
        # button instead of destroying / recreating every widget.
        tup = (
            tk.BooleanVar(value=False),
            tk.StringVar(value="Close"),
            tk.IntVar(value=1),
            tk.StringVar(value="XY"),
            tk.IntVar(value=3),
            tk.IntVar(value=3),
            tk.IntVar(value=128),
            tk.BooleanVar(value=False),
            tk.StringVar(value="AND"),
            tk.StringVar(value="mask_pre"),
        )
        pipeline_list.append(tup)
        self._attach_step_traces(tup)
        new_idx = len(pipeline_list) - 1
        sf = self._add_step_frame(parent_frame, pipeline_list, new_idx)
        self._grid_step_card(sf, new_idx)
        self._place_add_btn(parent_frame, pipeline_list)
        # Don't rebuild the All-Masks window on every add - it's
        # expensive. The user can hit Apply (F5) or toggle a step var
        # to refresh. The flowchart's edge list updates automatically
        # when the window is next opened/rebuilt.
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _on_move_step(self, parent_frame, pipeline_list, idx, direction):
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(pipeline_list):
            return
        pipeline_list[idx], pipeline_list[new_idx] = \
            pipeline_list[new_idx], pipeline_list[idx]
        # Only rebuild the step-card UI (so the new order is visible);
        # do NOT auto-refresh _process / All-Masks. The user presses
        # Apply (F5) when they want the pipeline to re-run.
        self._rebuild_pipeline_ui(parent_frame, pipeline_list)

    def _on_remove_step(self, parent_frame, pipeline_list, idx):
        if not pipeline_list:
            return
        del pipeline_list[idx]
        self._rebuild_pipeline_ui(parent_frame, pipeline_list)
        self._rebuild_all_masks_if_open()
        self._refresh()
