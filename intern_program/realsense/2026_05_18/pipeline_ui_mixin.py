"""Mixin: dynamic pipeline-step UI (the row of "Step N" cards in the side panel).

Concerns: building/destroying step frames, add/remove/move buttons, default
step seeding. All methods operate on a `pipeline_list` of 10-tuples of tk
variables stored on the host class.
"""
import tkinter as tk
from tkinter import ttk

from config import (
    ALL_PROC_OPS, OP_GROUPS, group_for_op, params_for_op,
    is_pre_morph_op, MORPH_OPS,
)

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
                               allow_none=False, include_this_pipe=True):
        """Build two cascading dropdowns. The user picks a pipeline
        GROUP first (this-pipe, Raw, RGB, IR, branch:<name>, optionally
        none), then narrows to a specific step or view. The composed
        view name is written to `target_var`.

        `include_this_pipe` - when False the "(this pipe)" group is
        omitted (used by the main 6-panel selectors, which have no
        owning pipeline). The step lists are always built from the
        LIVE pipeline lengths, so there is no fixed step cap.

        Returns (src_row, view_row) so the caller can pack/forget them.
        """
        grp_var  = tk.StringVar(value="(this pipe)")
        view_var = tk.StringVar(value=target_var.get() or "prev")

        src_row = tk.Frame(parent)
        src_row.pack(fill="x", pady=1)
        tk.Label(src_row, text=label_from, font=("DejaVu Sans", 8),
                 width=8, anchor="w").pack(side="left")
        cb_grp = ttk.Combobox(src_row, textvariable=grp_var,
                              width=14, state="readonly",
                              font=("DejaVu Sans", 8))
        cb_grp.pack(side="left", padx=1)

        view_row = tk.Frame(parent)
        view_row.pack(fill="x", pady=1)
        tk.Label(view_row, text=label_step, font=("DejaVu Sans", 8),
                 width=8, anchor="w").pack(side="left")
        cb_view = ttk.Combobox(view_row, textvariable=view_var,
                               width=22, state="readonly",
                               font=("DejaVu Sans", 8))
        cb_view.pack(side="left")

        def _groups():
            g = []
            if allow_none:
                g.append("none")
            if include_this_pipe:
                g.append("(this pipe)")
            g += ["Raw inputs", "RGB pipeline", "IR pipeline"]
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
                        "ir_raw", "ir_gray", "ir_blur", "ir_clahe",
                        "combined"]
            if grp == "RGB pipeline":
                n = len(getattr(self, "rgb_pipeline", []))
                return (["rgb_m1", "rgb_m2", "rgb_hsv_mask",
                         "rgb_bgsub", "rgb_mask_pre"]
                        + [f"rgb_step{i}" for i in range(1, n + 1)]
                        + ["rgb_mask", "rgb_post_blur", "rgb_det"])
            if grp == "IR pipeline":
                n = len(getattr(self, "ir_pipeline", []))
                return (["ir_thresh", "ir_mask_pre"]
                        + [f"ir_step{i}" for i in range(1, n + 1)]
                        + ["ir_mask", "ir_post_blur", "ir_det"])
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
                        f"up_{bnm}_ir_mask",
                        f"up_{bnm}_src_raw", f"up_{bnm}_src_blur"]
                for up in self.user_pipelines:
                    if up["name"].get().strip() == bnm:
                        base += [f"up_{bnm}_pre_step{i}"
                                 for i in range(1,
                                                len(up.get("pre_steps") or [])
                                                + 1)]
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
            _dflt_grp = "(this pipe)" if include_this_pipe else "Raw inputs"
            if (v in ("mask_pre", "prev") or v.startswith("step_")) \
                    and include_this_pipe:
                grp_var.set("(this pipe)")
            elif v in ("rgb_raw", "rgb_blur", "rgb_hsv_full",
                       "rgb_hsv_H", "rgb_hsv_S", "rgb_hsv_V",
                       "ir_raw", "ir_gray", "ir_blur", "ir_clahe",
                       "combined"):
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
                grp_var.set(f"branch:{best}" if best else _dflt_grp)
            else:
                grp_var.set(_dflt_grp)
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

        def _external_sync(*_):
            # When target_var is changed from OUTSIDE (e.g. a config
            # load), re-seed the two dropdowns to match. _commit always
            # sets target_var == view_var, so a genuine external change
            # is the only case where they differ — no loop, no flag.
            if target_var.get() != view_var.get():
                _populate_from_existing()
                _refresh_view_options()

        cb_grp.bind("<Button-1>",   lambda e: _refresh_grp_options())
        cb_view.bind("<Button-1>",  lambda e: _refresh_view_options())
        grp_var.trace_add("write",  _refresh_view_options)
        view_var.trace_add("write", _commit)
        target_var.trace_add("write", _external_sync)

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
          yolo_en         - toggle YOLO inference on this step
          yolo_input_kind - "image" (feed raw BGR view) or "mask"
                            (feed a binary mask view, useful when
                            YOLO is trained on shape silhouettes or
                            you want to detect inside an already-
                            isolated region)
          yolo_src        - RAW source view (when kind == "image")
          yolo_mask_src   - mask view name (when kind == "mask")
          yolo_mode       - "box_only" / "focus" / "subtract":
                            how the detected boxes affect the
                            running pipeline mask
          yolo_box_scale  - per-step box scale % (100 = unchanged,
                            >100 grows the box, <100 shrinks it)
          yolo_box_pad    - per-step pixel pad added on every side
                            (negative trims). Overrides the shared
                            YOLO box size for THIS step only.
        """
        if not hasattr(self, "_yolo_state"):
            self._yolo_state = {}
        key = id(step_tuple)
        st = self._yolo_state.get(key)
        if st is None:
            st = {
                "yolo_en":          tk.BooleanVar(value=False),
                "yolo_input_kind":  tk.StringVar(value="image"),
                "yolo_src":         tk.StringVar(value="rgb_raw"),
                "yolo_mask_src":    tk.StringVar(value="rgb_mask"),
                "yolo_mode":        tk.StringVar(value="box_only"),
                # Per-step YOLO box size: scale % around the box
                # centre + pixel pad on every side. Defaults
                # (100 / 0) leave the raw YOLO box unchanged.
                "yolo_box_scale":   tk.IntVar(value=100),
                "yolo_box_pad":     tk.IntVar(value=0),
                # Per-step class filter: a {class_id: BooleanVar} dict
                # of tick boxes — same idea as the shared YOLO-classes
                # panel, but local to this step. Lazy-filled when the
                # step card is built (so it works whenever the model
                # loads). Empty dict = no filtering (keep all classes).
                "yolo_classes":     {},
            }
            # ALL per-step YOLO settings (yolo_en, yolo_input_kind,
            # yolo_src, yolo_mask_src, yolo_mode) are deliberately
            # NOT auto-refreshed. Each YOLO change forces an
            # inference per step, which is expensive — the user
            # commits with Apply (F5) or by playing the video.
            # Local UI swaps (image-picker <-> mask-picker) are
            # handled by _refresh_yolo_input_visibility in
            # _add_step_frame, so the step card still updates
            # instantly when the kind is toggled.
            self._yolo_state[key] = st
        return st

    def _yolo_class_var(self, yolo_st, cid):
        """Lazy-create a per-step per-class enable BooleanVar (default
        ticked). `yolo_st` is a _yolo_state entry; `cid` an int class
        id. Tick boxes built from these let a step keep only chosen
        YOLO classes (e.g. only drone, only t-hook)."""
        classes = yolo_st.setdefault("yolo_classes", {})
        v = classes.get(cid)
        if v is None:
            v = tk.BooleanVar(value=True)
            classes[cid] = v
        return v

    def _kshape_state_for(self, step_tuple):
        """Per-step Kernel-shape state (Rect / Ellipse / Cross).
        Used only when Direction == 'XY' on a morphology op; ignored
        for X/Y-only directions (which always use Rect) and for non-
        morph ops. Stored as side state — same lazy-create pattern
        as overlay/yolo state — so the 10-tuple shape stays intact.
        """
        if not hasattr(self, "_kshape_state"):
            self._kshape_state = {}
        key = id(step_tuple)
        st = self._kshape_state.get(key)
        if st is None:
            st = {"kshape": tk.StringVar(value="Rect")}
            # Slider-level live refresh: a kernel-shape flip is cheap
            # to re-apply, so update the current frame immediately.
            st["kshape"].trace_add("write",
                                    lambda *a: self._refresh())
            self._kshape_state[key] = st
        return st

    def _pm_target_state_for(self, step_tuple):
        """Per-PM-step Target channel state (lazy-create).

        A PM (Pre-morph) step normally runs on the whole source image
        ("BGR"). Picking "H", "S" or "V" instead makes the op run on
        ONLY that HSV channel of the source — e.g. a Blur on just the
        Hue channel — and the modified channel is written back before
        detection. For grayscale (IR) sources the target is ignored.
        Stored as side state — same lazy-create pattern as
        kshape/overlay/yolo — so the 10-tuple shape stays intact.
        """
        if not hasattr(self, "_pm_target_state"):
            self._pm_target_state = {}
        key = id(step_tuple)
        st = self._pm_target_state.get(key)
        if st is None:
            st = {"target": tk.StringVar(value="BGR")}
            # Re-applying a PM step against a different channel is
            # cheap, so refresh the current frame immediately.
            st["target"].trace_add("write", lambda *a: self._refresh())
            self._pm_target_state[key] = st
        return st

    def _pm_target_for(self, step_tuple):
        """Resolve a PM step's Target channel ('BGR'/'H'/'S'/'V').
        Falls back to 'BGR' when the step has no target state yet
        (e.g. freshly loaded from an older config)."""
        st = (getattr(self, "_pm_target_state", {}) or {}
              ).get(id(step_tuple))
        if st is None:
            return "BGR"
        try:
            return st["target"].get() or "BGR"
        except Exception:
            return "BGR"

    def _step_type_state_for(self, step_tuple):
        """Per-step Type state: 'morph' / 'combine' / 'yolo'.

        The legacy `comb_en` boolean (tuple index 7) still tells
        morph from combine for serialisation & processing; this
        side-state adds the third 'yolo' kind (a step that ONLY
        runs YOLO box detection — focus / subtract / box_only)
        without changing the 10-tuple shape. Lazy-create, seeded
        from comb_en so older configs still load correctly.
        """
        if not hasattr(self, "_step_type_state"):
            self._step_type_state = {}
        key = id(step_tuple)
        st = self._step_type_state.get(key)
        if st is None:
            try:
                init = "combine" if step_tuple[7].get() else "morph"
            except Exception:
                init = "morph"
            st = {"kind": tk.StringVar(value=init)}
            self._step_type_state[key] = st
        return st

    def _step_kind_for(self, step_tuple):
        """Resolve a step's Type ('morph' / 'combine' / 'yolo').
        Falls back to the legacy comb_en boolean when no type state
        exists yet (e.g. freshly loaded from an older config)."""
        st = (getattr(self, "_step_type_state", {}) or {}
              ).get(id(step_tuple))
        if st is None:
            try:
                return "combine" if step_tuple[7].get() else "morph"
            except Exception:
                return "morph"
        try:
            return st["kind"].get() or "morph"
        except Exception:
            return "morph"

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
            # All overlay state vars: live refresh only. The composite
            # tile count can change when mask2_src / base_src flip
            # between "none" and a view, but the All-Masks canvas
            # rebuild is deferred to Apply (F5) so add/move/delete
            # rebuilds don't transitively rebuild the canvas via the
            # cascading picker re-populating these vars.
            for _v in st.values():
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
        """Trace step vars that should live-update (slider/param
        tweaks). Topology-level changes — adding a step, removing
        one, swapping order, or flipping Morph <-> Combine — are
        deliberately NOT auto-applied here; they wait for the user
        to press Apply (F5) or play the video so heavy reprocessing
        doesn't fire on every keypress."""
        en, op, n, dir_, kx, ky, th, cen, cop, csr = step_tuple
        # Param-level vars: live refresh — these are the sliders the
        # user is actively tweaking.
        for v in (en, op, n, dir_, kx, ky, th):
            v.trace_add("write", lambda *a: self._refresh())
        # Combine-level vars (cen=Morph/Combine toggle, cop=AND/OR/XOR
        # /OVERLAY, csr=combine source): NO auto-refresh. The card UI
        # still updates instantly via the local _refresh_visibility
        # callback in _add_step_frame; only the pipeline reprocess +
        # All-Masks canvas rebuild are deferred.

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
    # How many step cards before we wrap to a new row. Morph cards
    # carry the full Type / Op / KX / KY / Combine / OVERLAY / YOLO
    # control set, so 3-per-row keeps each card readable. PM cards
    # only need Op / Iter / Dir / KX / KY / Thresh / K-Shape and are
    # noticeably narrower — 4-per-row uses the available width
    # better without squeezing the labels.
    STEPS_PER_UI_ROW    = 3   # Morph step cards
    PM_STEPS_PER_UI_ROW = 4   # PM (Pre-morph) step cards

    def _grid_step_card(self, sf, i, per_row=None):
        """Place an existing step card `sf` at grid cell
        (i // per_row, i % per_row).  When `per_row` is None the
        Morph-stage default (`STEPS_PER_UI_ROW`) is used so existing
        call sites keep their behaviour."""
        if per_row is None:
            per_row = self.STEPS_PER_UI_ROW
        sf.grid(row=i // per_row,
                column=i % per_row,
                padx=5, pady=3, sticky="n")

    def _place_add_btn(self, parent_frame, pipeline_list):
        """(Re)place the '+ Add Step' button right after the last
        Morph step card."""
        # Remove old button if it exists.
        old = getattr(parent_frame, "_add_btn", None)
        if old is not None and str(old) != "":
            try:
                old.destroy()
            except Exception:
                pass
        i = len(pipeline_list)
        btn = tk.Button(parent_frame, text="+ Add Step",
                        font=("DejaVu Sans", 8), bg="#223322", fg="white",
                        command=lambda pl=pipeline_list, pf=parent_frame:
                            self._on_add_step(pf, pl))
        btn.grid(row=i // self.STEPS_PER_UI_ROW,
                 column=i % self.STEPS_PER_UI_ROW,
                 padx=8, pady=10, sticky="n")
        parent_frame._add_btn = btn

    def _place_pm_add_btn(self, parent_frame, pipeline_list):
        """(Re)place the '+ Add PM Step' button after the last PM card."""
        old = getattr(parent_frame, "_add_btn", None)
        if old is not None:
            try:
                old.destroy()
            except Exception:
                pass
        i = len(pipeline_list)
        btn = tk.Button(parent_frame, text="+ Add PM Step",
                        font=("DejaVu Sans", 9), bg="#225522", fg="white",
                        command=lambda pl=pipeline_list, pf=parent_frame:
                            self._on_add_pre_morph_step(pf, pl))
        btn.grid(row=i // self.PM_STEPS_PER_UI_ROW,
                 column=i % self.PM_STEPS_PER_UI_ROW,
                 padx=8, pady=10, sticky="n")
        parent_frame._add_btn = btn

    # ------------------------------------------------------------------
    # Fast incremental step-card management.
    #
    # Every pipeline panel keeps `parent_frame._cards` — the list of
    # step-card frames, parallel to its `pipeline_list`. add / move /
    # remove edit that list and re-GRID the existing cards instead of
    # destroying and rebuilding every card (a step card is expensive:
    # comboboxes, cascading pickers, traces). This turns each edit
    # from "rebuild N heavy cards" into "build at most 1 card".
    # ------------------------------------------------------------------
    def _step_card_index(self, parent_frame, sf):
        """Current index of step card `sf` in its panel, or -1.

        The ^/v/x buttons capture their card frame (not a fixed index),
        so the right step is always found even after other cards have
        been moved or removed."""
        try:
            return list(getattr(parent_frame, "_cards", [])).index(sf)
        except ValueError:
            return -1

    def _relayout_step_cards(self, parent_frame, per_row, title_prefix):
        """Re-grid the EXISTING step cards in their current order and
        refresh each 'Step N' title. No widget is destroyed or rebuilt
        — the fast path used by move / remove."""
        for i, sf in enumerate(getattr(parent_frame, "_cards", [])):
            self._grid_step_card(sf, i, per_row=per_row)
            try:
                sf.config(text="%s %d" % (title_prefix, i + 1))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Pre-morph pipeline (image-stage) — compact step card with the
    # same 10-tuple shape so it can share the runtime op-apply paths,
    # but no Combine / OVERLAY / YOLO sub-sections (those concepts
    # only make sense on a binary mask). Move up/down/delete and
    # +Add work the same way.
    # ------------------------------------------------------------------
    def _add_pre_morph_step_frame(self, parent_frame, pipeline_list, idx):
        en, op, n, dr, kx, ky, th, *_ = pipeline_list[idx]
        # Step / PM-Step cards keep a visible border so each card is
        # an obvious unit even though the global LabelFrame default
        # is bd=0 / relief=flat for the surrounding section panels.
        # Same title font as the Morph step card (bold, size 9) so
        # PM and Morph cards read consistently.
        sf = tk.LabelFrame(parent_frame,
                           text=f"PM Step {idx + 1}",
                           font=("DejaVu Sans", 9, "bold"),
                           fg="#88dd88",
                           bd=2, relief="groove", padx=3, pady=2)

        btn_row = tk.Frame(sf)
        btn_row.pack(anchor="ne", padx=1, pady=1)
        tk.Button(btn_row, text="^", font=("DejaVu Sans", 9), width=2,
                  relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_move_pre_morph_step(pf, pl, s, -1)
                  ).pack(side="left")
        tk.Button(btn_row, text="v", font=("DejaVu Sans", 9), width=2,
                  relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_move_pre_morph_step(pf, pl, s, +1)
                  ).pack(side="left")
        tk.Button(btn_row, text="x", font=("DejaVu Sans", 9, "bold"),
                  fg="#ff4444", width=2, relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_remove_pre_morph_step(pf, pl, s)
                  ).pack(side="left")

        tk.Checkbutton(sf, text="Enable", variable=en,
                       font=("DejaVu Sans", 9)).pack(anchor="w")

        # Cascading Group -> Op picker — compact single row.
        grp_var = tk.StringVar(value=group_for_op(op.get()))
        gr_row = tk.Frame(sf); gr_row.pack(fill="x", pady=1)
        tk.Label(gr_row, text="Grp:", font=("DejaVu Sans", 9),
                 width=4, anchor="w").pack(side="left")
        cb_g = ttk.Combobox(gr_row, textvariable=grp_var,
                            values=list(OP_GROUPS.keys()),
                            width=11, state="readonly",
                            font=("DejaVu Sans", 9))
        cb_g.pack(side="left", padx=(0, 4))
        tk.Label(gr_row, text="Op:", font=("DejaVu Sans", 9),
                 anchor="w").pack(side="left")
        cb_o = ttk.Combobox(gr_row, textvariable=op,
                            values=OP_GROUPS[grp_var.get()],
                            width=11, state="readonly",
                            font=("DejaVu Sans", 9))
        cb_o.pack(side="left", padx=(2, 0))

        def _on_grp(*_):
            ops = OP_GROUPS.get(grp_var.get(), ALL_PROC_OPS)
            cb_o.configure(values=ops)
            if op.get() not in ops and ops:
                op.set(ops[0])
        grp_var.trace_add("write", _on_grp)

        ks_state = self._kshape_state_for(pipeline_list[idx])

        # Three compact param rows.
        #   Row A: Iter | Dir | K Shape
        #   Row B: KX   | KY
        #   Row C: Thresh
        param_row_a = tk.Frame(sf); param_row_a.pack(fill="x", pady=1)
        param_row_b = tk.Frame(sf); param_row_b.pack(fill="x", pady=1)
        param_row_c = tk.Frame(sf); param_row_c.pack(fill="x", pady=1)

        def _cell(parent, label_text, var, kind="spin",
                  lo=1, hi=99, opts=None, width=6, lbl_w=None):
            """One labelled control. lbl_w=None auto-sizes the label
            (needed for KX/KY/Thresh whose text is rewritten per-op)."""
            cell = tk.Frame(parent)
            if lbl_w is not None:
                lbl = tk.Label(cell, text=label_text + ":",
                               font=("DejaVu Sans", 9),
                               width=lbl_w, anchor="w")
            else:
                lbl = tk.Label(cell, text=label_text + ":",
                               font=("DejaVu Sans", 9), anchor="w")
            lbl.pack(side="left", padx=(0, 4))
            if kind == "combo":
                ttk.Combobox(cell, textvariable=var, values=opts,
                             width=width, state="readonly",
                             font=("DejaVu Sans", 9)).pack(side="left")
            else:
                tk.Spinbox(cell, textvariable=var, from_=lo, to=hi,
                           width=width, font=("DejaVu Sans", 9)
                           ).pack(side="left")
            return cell, lbl

        n_cell,   n_lbl   = _cell(param_row_a, "Iter", n,
                                  lo=1, hi=20, width=4, lbl_w=5)
        dir_cell, dir_lbl = _cell(param_row_a, "Dir", dr,
                                  kind="combo", opts=["X", "Y", "XY"],
                                  width=4, lbl_w=4)
        ks_cell,  ks_lbl  = _cell(param_row_a, "K Shape", ks_state["kshape"],
                                  kind="combo",
                                  opts=["Rect", "Ellipse", "Cross"],
                                  width=8, lbl_w=8)
        kx_cell,  kx_lbl  = _cell(param_row_b, "KX", kx,
                                  lo=1, hi=999, width=6, lbl_w=None)
        ky_cell,  ky_lbl  = _cell(param_row_b, "KY", ky,
                                  lo=1, hi=999, width=6, lbl_w=None)
        t_cell,   t_lbl   = _cell(param_row_c, "Thresh", th,
                                  lo=0, hi=255, width=6, lbl_w=None)
        param_cells = {
            "N":  (n_cell,  n_lbl),
            "Dir":(dir_cell, dir_lbl),
            "KX": (kx_cell, kx_lbl),
            "KY": (ky_cell, ky_lbl),
            "T":  (t_cell,  t_lbl),
        }

        # -- Target channel -------------------------------------------
        # A PM step runs on the whole source ("BGR") by default; pick
        # H / S / V to run it on ONLY that HSV channel of the source
        # (e.g. a Blur on just Hue). Ignored for grayscale IR sources.
        pm_tgt_state = self._pm_target_state_for(pipeline_list[idx])
        tgt_cell, tgt_lbl = _cell(param_row_c, "On", pm_tgt_state["target"],
                                  kind="combo", opts=["BGR", "H", "S", "V"],
                                  width=5, lbl_w=3)
        tgt_cell.pack(side="left", padx=(8, 6))

        op_desc_lbl = tk.Label(sf, text="",
                               font=("DejaVu Sans", 9, "italic"),
                               fg="#aaccaa", anchor="w",
                               wraplength=240, justify="left")
        op_desc_lbl.pack(fill="x", padx=4, pady=(2, 2))

        def _refresh_params(*_):
            spec = params_for_op(op.get())
            visible = set(spec.get("params", []))
            label_map = {
                "N":  spec.get("n_lbl",  "Iter"),
                "Dir": spec.get("dir_lbl", "Dir"),
                "KX": spec.get("kx_lbl", "KX"),
                "KY": spec.get("ky_lbl", "KY"),
                "T":  spec.get("t_lbl",  "Thresh"),
            }
            _desc = spec.get("desc", "")
            op_desc_lbl.config(text=("(i) " + _desc) if _desc else "")
            show_kx = "KX" in visible
            show_ky = "KY" in visible
            if op.get() in MORPH_OPS:
                d = dr.get()
                if d == "X":
                    show_ky = False
                elif d == "Y":
                    show_kx = False
            cell_visible = {
                "N":  "N"  in visible,
                "Dir":"Dir" in visible,
                "KX": show_kx,
                "KY": show_ky,
                "T":  "T"  in visible,
            }
            for key, (cell, lbl) in param_cells.items():
                if cell_visible[key]:
                    lbl.config(text=label_map[key] + ":")
                    cell.pack(side="left", padx=(0, 6))
                else:
                    cell.pack_forget()
            if op.get() in MORPH_OPS and dr.get() == "XY":
                ks_cell.pack(side="left", padx=(0, 6))
            else:
                ks_cell.pack_forget()
        op.trace_add("write", _refresh_params)
        dr.trace_add("write", _refresh_params)
        _refresh_params()
        return sf

    def _rebuild_pre_morph_pipeline_ui(self, parent_frame, pipeline_list):
        with self._suppress_refresh():
            for w in parent_frame.winfo_children():
                w.destroy()
            parent_frame._add_btn = None
            parent_frame._cards = []
            for i in range(len(pipeline_list)):
                sf = self._add_pre_morph_step_frame(parent_frame,
                                                     pipeline_list, i)
                parent_frame._cards.append(sf)
                # PM cards are narrower than Morph cards, so they pack
                # 4-per-row instead of 3.
                self._grid_step_card(sf, i,
                                     per_row=self.PM_STEPS_PER_UI_ROW)
            self._place_pm_add_btn(parent_frame, pipeline_list)
        # Freshly built cards use dark-tuned literals + explicit bold
        # fonts — apply the active theme + font settings.
        if hasattr(self, "_restyle_subtree"):
            self._restyle_subtree(parent_frame)

    def _on_add_pre_morph_step(self, parent_frame, pipeline_list):
        # Incremental: build ONLY the new card, then re-place the add
        # button. No destroy/rebuild of the existing cards.
        with self._suppress_refresh():
            tup = (
                tk.BooleanVar(value=False),
                tk.StringVar(value="GaussBlur"),
                tk.IntVar(value=1),
                tk.StringVar(value="XY"),
                tk.IntVar(value=5),
                tk.IntVar(value=5),
                tk.IntVar(value=128),
                tk.BooleanVar(value=False),     # combine_en (unused)
                tk.StringVar(value="AND"),      # combine_op (unused)
                tk.StringVar(value="mask_pre"), # combine_src (unused)
            )
            pipeline_list.append(tup)
            self._attach_step_traces(tup)
            if not hasattr(parent_frame, "_cards"):
                parent_frame._cards = []
            new_idx = len(pipeline_list) - 1
            sf = self._add_pre_morph_step_frame(parent_frame,
                                                pipeline_list, new_idx)
            parent_frame._cards.append(sf)
            self._grid_step_card(sf, new_idx,
                                 per_row=self.PM_STEPS_PER_UI_ROW)
            self._place_pm_add_btn(parent_frame, pipeline_list)
            if hasattr(self, "_restyle_subtree"):
                self._restyle_subtree(sf)

    def _on_move_pre_morph_step(self, parent_frame, pipeline_list,
                                 sf, direction):
        # Incremental: swap two entries + re-grid; nothing is rebuilt.
        idx = self._step_card_index(parent_frame, sf)
        if idx < 0:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(pipeline_list):
            return
        with self._suppress_refresh():
            pipeline_list[idx], pipeline_list[new_idx] = \
                pipeline_list[new_idx], pipeline_list[idx]
            cards = parent_frame._cards
            cards[idx], cards[new_idx] = cards[new_idx], cards[idx]
            self._relayout_step_cards(parent_frame,
                                      self.PM_STEPS_PER_UI_ROW, "PM Step")
            self._place_pm_add_btn(parent_frame, pipeline_list)

    def _on_remove_pre_morph_step(self, parent_frame, pipeline_list, sf):
        # Incremental: drop ONLY this card, re-grid the rest. Hidden at
        # once; the heavy widget destroy runs off the critical path.
        idx = self._step_card_index(parent_frame, sf)
        if idx < 0 or not pipeline_list:
            return
        with self._suppress_refresh():
            del pipeline_list[idx]
            parent_frame._cards.pop(idx)
            sf.grid_forget()
            self._relayout_step_cards(parent_frame,
                                      self.PM_STEPS_PER_UI_ROW, "PM Step")
            self._place_pm_add_btn(parent_frame, pipeline_list)
        self.root.after(10, sf.destroy)

    def _rebuild_pipeline_ui(self, parent_frame, pipeline_list):
        """Full rebuild — used on config load / steps-per-row change.
        Interactive add / move / remove use the fast incremental path
        (_on_add_step / _on_move_step / _on_remove_step) instead."""
        for w in parent_frame.winfo_children():
            w.destroy()
        parent_frame._add_btn = None
        parent_frame._cards = []
        for i in range(len(pipeline_list)):
            sf = self._add_step_frame(parent_frame, pipeline_list, i)
            parent_frame._cards.append(sf)
            self._grid_step_card(sf, i)
        self._place_add_btn(parent_frame, pipeline_list)
        # Freshly built cards use dark-tuned literals + explicit bold
        # fonts — apply the active theme + font settings.
        if hasattr(self, "_restyle_subtree"):
            self._restyle_subtree(parent_frame)

    def _add_step_frame(self, parent_frame, pipeline_list, idx):
        en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var, \
            comb_en_var, comb_op_var, comb_src_var = pipeline_list[idx]

        # Step cards keep a visible border so each card reads as a
        # discrete unit (the global LabelFrame default is flat / no
        # border for the surrounding section panels).
        sf = tk.LabelFrame(parent_frame, text=f"Step {idx + 1}",
                           font=("DejaVu Sans", 9, "bold"), fg="#aaccff",
                           bd=2, relief="groove", padx=3, pady=2)
        # Caller decides placement (grid). Do NOT pack here.

        # Reorder / remove buttons
        btn_row = tk.Frame(sf)
        btn_row.pack(anchor="ne", padx=1, pady=1)
        tk.Button(btn_row, text="^", font=("DejaVu Sans", 9), width=2, relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_move_step(pf, pl, s, -1)).pack(side="left")
        tk.Button(btn_row, text="v", font=("DejaVu Sans", 9), width=2, relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_move_step(pf, pl, s, +1)).pack(side="left")
        tk.Button(btn_row, text="x", font=("DejaVu Sans", 8, "bold"), fg="#ff4444",
                  width=2, relief="flat",
                  command=lambda pl=pipeline_list, s=sf, pf=parent_frame:
                      self._on_remove_step(pf, pl, s)).pack(side="left")

        tk.Checkbutton(sf, text="Enable", variable=en_var,
                       font=("DejaVu Sans", 8)).pack(anchor="w")

        # -- Step TYPE radio: Morph / Combine / YOLO -------------------
        # The three radios drive a side-state StringVar `kind_var`
        # ("morph" / "combine" / "yolo"). The legacy comb_en_var
        # boolean is kept in sync (True only for "combine") so
        # serialisation & the morph/combine processing path are
        # unchanged. "yolo" -> the step ONLY runs YOLO box detection
        # (focus / subtract / box_only); morph & combine are skipped.
        kind_var = self._step_type_state_for(pipeline_list[idx])["kind"]
        type_row = tk.Frame(sf)
        type_row.pack(fill="x", pady=1)
        tk.Label(type_row, text="Type:", font=("DejaVu Sans", 8, "bold"),
                 width=5, anchor="w").pack(side="left")
        morph_rb = tk.Radiobutton(type_row, text="Morph", value="morph",
                                  variable=kind_var,
                                  font=("DejaVu Sans", 8), fg="#aaccff")
        morph_rb.pack(side="left")
        comb_rb  = tk.Radiobutton(type_row, text="Combine", value="combine",
                                  variable=kind_var,
                                  font=("DejaVu Sans", 8, "bold"), fg="#bb66ff")
        comb_rb.pack(side="left")
        yolo_rb  = tk.Radiobutton(type_row, text="YOLO", value="yolo",
                                  variable=kind_var,
                                  font=("DejaVu Sans", 8, "bold"), fg="#ffaa66")
        yolo_rb.pack(side="left")

        def _row(p, lbl, wfn):
            r = tk.Frame(p)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, font=("DejaVu Sans", 8),
                     width=4, anchor="w").pack(side="left")
            wfn(r)

        # -- Morph fields: Op (Group -> Op), N, Dir, KX, KY, T ---------
        morph_fr = tk.Frame(sf)
        morph_fr.pack(fill="x")

        # Cascading op picker: pick the GROUP first (Morphology /
        # Blur / Threshold / Illumination / Edge / Bitwise / Misc),
        # then narrow to the actual op. Drives op_var on change.
        # Grp + Op live on a SINGLE row to keep the card compact.
        grp_op_var = tk.StringVar(value=group_for_op(op_var.get()))
        gr_row = tk.Frame(morph_fr); gr_row.pack(fill="x", pady=1)
        tk.Label(gr_row, text="Grp:", font=("DejaVu Sans", 8),
                 width=4, anchor="w").pack(side="left")
        cb_g = ttk.Combobox(gr_row, textvariable=grp_op_var,
                            values=list(OP_GROUPS.keys()),
                            width=11, state="readonly",
                            font=("DejaVu Sans", 8))
        cb_g.pack(side="left", padx=(0, 4))
        tk.Label(gr_row, text="Op:", font=("DejaVu Sans", 8),
                 anchor="w").pack(side="left")
        cb_o = ttk.Combobox(gr_row, textvariable=op_var,
                            values=OP_GROUPS[grp_op_var.get()],
                            width=11, state="readonly",
                            font=("DejaVu Sans", 8))
        cb_o.pack(side="left", padx=(2, 0))

        def _on_grp_change(*_):
            ops = OP_GROUPS.get(grp_op_var.get(), ALL_PROC_OPS)
            cb_o.configure(values=ops)
            if op_var.get() not in ops and ops:
                op_var.set(ops[0])
        grp_op_var.trace_add("write", _on_grp_change)

        # Per-step kernel shape (Rect / Ellipse / Cross). State must
        # exist before the cells reference it. Visible only when the
        # op is Morphology AND Direction is XY — X/Y-only steps
        # always use a Rect line kernel and non-morph ops ignore the
        # shape entirely.
        ks_state = self._kshape_state_for(pipeline_list[idx])

        # Three compact param rows replace the old 5-row stack.
        #   Row A: Iter | Dir | K Shape
        #   Row B: KX   | KY
        #   Row C: Thresh
        # Each cell is its own little frame so we can pack_forget
        # individual cells without losing the row above/below.
        param_row_a = tk.Frame(morph_fr); param_row_a.pack(fill="x", pady=1)
        param_row_b = tk.Frame(morph_fr); param_row_b.pack(fill="x", pady=1)
        param_row_c = tk.Frame(morph_fr); param_row_c.pack(fill="x", pady=1)

        def _cell(parent, label_text, var, kind="spin",
                  lo=1, hi=99, opts=None, width=6, lbl_w=None):
            """One labelled control inside a param row.
            `lbl_w=None` lets the label auto-size — important for the
            KX / KY / Thresh cells whose label text is re-written by
            `_refresh_op_params` to op-specific names like
            'Kernel size (odd)' or 'Low threshold' that don't fit in a
            fixed character width. A small right-padx between label
            and input keeps the spinbox/combo from butting up against
            the text."""
            cell = tk.Frame(parent)
            if lbl_w is not None:
                lbl = tk.Label(cell, text=label_text + ":",
                               font=("DejaVu Sans", 8),
                               width=lbl_w, anchor="w")
            else:
                lbl = tk.Label(cell, text=label_text + ":",
                               font=("DejaVu Sans", 8), anchor="w")
            lbl.pack(side="left", padx=(0, 4))
            if kind == "combo":
                ttk.Combobox(cell, textvariable=var, values=opts,
                             width=width, state="readonly",
                             font=("DejaVu Sans", 8)).pack(side="left")
            else:
                tk.Spinbox(cell, textvariable=var, from_=lo, to=hi,
                           width=width, font=("DejaVu Sans", 8)
                           ).pack(side="left")
            return cell, lbl

        # Static-label cells (Iter / Dir / K Shape) keep a fixed-width
        # label so they line up across Row A. Dynamic-label cells
        # (KX / KY / Thresh) auto-size — `_refresh_op_params` rewrites
        # their text to op-specific names that vary in length.
        n_cell,   n_lbl   = _cell(param_row_a, "Iter", n_var,
                                  lo=1, hi=20, width=4, lbl_w=5)
        dir_cell, dir_lbl = _cell(param_row_a, "Dir", dir_var,
                                  kind="combo", opts=["X", "Y", "XY"],
                                  width=4, lbl_w=4)
        ks_cell,  ks_lbl  = _cell(param_row_a, "K Shape", ks_state["kshape"],
                                  kind="combo",
                                  opts=["Rect", "Ellipse", "Cross"],
                                  width=8, lbl_w=8)
        kx_cell,  kx_lbl  = _cell(param_row_b, "KX", kx_var,
                                  lo=1, hi=999, width=6, lbl_w=None)
        ky_cell,  ky_lbl  = _cell(param_row_b, "KY", ky_var,
                                  lo=1, hi=999, width=6, lbl_w=None)
        t_cell,   t_lbl   = _cell(param_row_c, "Thresh", thresh_var,
                                  lo=0, hi=255, width=6, lbl_w=None)
        param_cells = {
            "N":  (n_cell,  n_lbl),
            "Dir":(dir_cell, dir_lbl),
            "KX": (kx_cell, kx_lbl),
            "KY": (ky_cell, ky_lbl),
            "T":  (t_cell,  t_lbl),
        }
        # Backwards-compatible alias — older code paths still reference
        # `param_rows` (e.g. the "(this op has no parameters)" hint).
        param_rows = {}

        # Morph-stage hint — reminds users that any op placed in this
        # pipeline runs on the binary MASK (not the source image).
        # If they want image-stage processing they should add a card
        # in the dedicated Pre-morph Steps section instead.
        stage_lbl = tk.Label(morph_fr,
                             text="[morph]  runs on BINARY mask "
                                  "(use Pre-morph Steps for image-stage)",
                             font=("DejaVu Sans", 9, "bold"),
                             fg="#88aaff",
                             anchor="w")
        stage_lbl.pack(fill="x", padx=4, pady=(2, 0))

        # One-line description of what THIS op actually does. Updates
        # whenever op_var changes.
        op_desc_lbl = tk.Label(morph_fr, text="",
                               font=("DejaVu Sans", 9, "italic"),
                               fg="#aaccaa", anchor="w",
                               wraplength=240, justify="left")
        op_desc_lbl.pack(fill="x", padx=4, pady=(2, 2))

        def _refresh_op_params(*_):
            # op_var / dir_var are persistent step-tuple variables, so
            # this trace outlives the card it was registered in. After
            # a card rebuild (remove / move step) the old callback
            # still fires but `op_desc_lbl` (and every other widget it
            # touches) is destroyed — bail out instead of crashing.
            try:
                if not op_desc_lbl.winfo_exists():
                    return
            except Exception:
                return
            spec = params_for_op(op_var.get())
            visible = set(spec.get("params", []))
            label_map = {
                "N":  spec.get("n_lbl",  "Iter"),
                "Dir": spec.get("dir_lbl", "Dir"),
                "KX": spec.get("kx_lbl", "KX"),
                "KY": spec.get("ky_lbl", "KY"),
                "T":  spec.get("t_lbl",  "Thresh"),
            }
            _desc = spec.get("desc", "")
            op_desc_lbl.config(text=("(i) " + _desc) if _desc else "")

            # Direction-driven KX/KY filtering for Morph ops: only show
            # the kernel dimensions the op actually uses for the current
            # direction (X -> just KX, Y -> just KY, XY -> both).
            show_kx = "KX" in visible
            show_ky = "KY" in visible
            if op_var.get() in MORPH_OPS:
                d = dir_var.get()
                if d == "X":
                    show_ky = False
                elif d == "Y":
                    show_kx = False

            cell_visible = {
                "N":  "N"  in visible,
                "Dir":"Dir" in visible,
                "KX": show_kx,
                "KY": show_ky,
                "T":  "T"  in visible,
            }
            for key, (cell, lbl) in param_cells.items():
                if cell_visible[key]:
                    lbl.config(text=label_map[key] + ":")
                    cell.pack(side="left", padx=(0, 6))
                else:
                    cell.pack_forget()

            # K Shape cell — only meaningful for Morphology ops with
            # Direction = XY. (X/Y-only steps always use a Rect line
            # kernel; non-morph ops ignore kernel shape entirely.)
            if op_var.get() in MORPH_OPS and dir_var.get() == "XY":
                ks_cell.pack(side="left", padx=(0, 6))
            else:
                ks_cell.pack_forget()

            # "(this op has no parameters)" hint when the op uses none
            # of the param-row controls. Lazy-created on first need.
            if not visible:
                if "_no_params_lbl" not in param_rows:
                    nl = tk.Label(morph_fr,
                                  text="(this op has no parameters)",
                                  font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8")
                    nl.pack(anchor="w", padx=4)
                    param_rows["_no_params_lbl"] = (nl, nl)
                else:
                    param_rows["_no_params_lbl"][0].pack(anchor="w", padx=4)
            else:
                if "_no_params_lbl" in param_rows:
                    param_rows["_no_params_lbl"][0].pack_forget()
        op_var.trace_add("write", _refresh_op_params)
        dir_var.trace_add("write", _refresh_op_params)
        _refresh_op_params()

        # -- Lazy sub-sections: Combine / OVERLAY / YOLO ---------------
        # Building the Combine, OVERLAY and YOLO blocks is expensive —
        # each one carries cascading From/Step pickers. A freshly
        # added step is always "morph" with YOLO off, so those bodies
        # are deferred (built once, on first need) instead of eagerly
        # on every "Add step". `W` holds the lazily-created widgets;
        # `_card_traces` collects (var, trace_id) pairs removed when
        # the card is destroyed.
        W = {}
        _card_traces = []

        # combine_fr / yolo_fr: cheap empty containers created now so
        # _refresh_visibility and `before=yolo_fr` always have a real
        # widget to act on. Their CONTENTS are filled on demand below.
        combine_fr = tk.Frame(sf)          # not packed — morph default
        yolo_st = self._yolo_state_for(pipeline_list[idx])
        yolo_fr = tk.LabelFrame(sf, text="+ YOLO box detection",
                                font=("DejaVu Sans", 9, "italic"),
                                fg="#ffaa66")
        yolo_fr.pack(side="bottom", fill="x", pady=(4, 1))

        def _build_combine():
            """Populate combine_fr the first time the step becomes a
            Combine step. Idempotent."""
            if W.get("combine_built") or not combine_fr.winfo_exists():
                return
            W["combine_built"] = True
            cop_row = tk.Frame(combine_fr)
            cop_row.pack(fill="x", pady=1)
            tk.Label(cop_row, text="Op:", font=("DejaVu Sans", 8),
                     width=4, anchor="w").pack(side="left")
            ttk.Combobox(cop_row, textvariable=comb_op_var,
                         values=["AND", "OR", "XOR", "OVERLAY"],
                         width=8, state="readonly",
                         font=("DejaVu Sans", 8, "bold")
                         ).pack(side="left", padx=2)
            tk.Label(combine_fr,
                     text="OVERLAY = paint running mask on top of src "
                          "(useful with rgb_raw / ir_raw / hsv_full)",
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8"
                     ).pack(anchor="w", padx=3)

            # -- Combine source: TWO cascading dropdowns (From + Step).
            src_row, view_row = self._make_cascading_picker(
                combine_fr, comb_src_var, pipeline_list,
                label_from="From:", label_step="Step:",
                allow_none=False)
            src_caption = tk.Label(combine_fr,
                                   text="result = running <op> src",
                                   font=("DejaVu Sans", 9, "italic"),
                                   fg="#bb66ff")
            src_caption.pack(anchor="w", padx=3)
            W["src_row"] = src_row
            W["view_row"] = view_row
            W["src_caption"] = src_caption

            # -- OVERLAY-only extras --------------------------------
            # Mask 1 is always the running mask (output of the
            # previous step) painted in C2. Mask 2 is an optional
            # second mask painted in C1 ("none" disables it). Base is
            # the BGR image painted on ("none" = black background).
            ov_state = self._overlay_state_for(pipeline_list[idx])
            ov_fr = tk.Frame(combine_fr)
            W["ov_fr"] = ov_fr

            def _base_options():
                return ["none", "rgb_raw", "rgb_blur",
                        "rgb_hsv_full", "rgb_hsv_H",
                        "rgb_hsv_S", "rgb_hsv_V",
                        "ir_raw", "ir_gray"]

            # Base view (BGR image painted on top of, or "none" = black)
            base_row = tk.Frame(ov_fr); base_row.pack(fill="x", pady=1)
            tk.Label(base_row, text="Base:", font=("DejaVu Sans", 8, "bold"),
                     width=8, anchor="w").pack(side="left")
            cb_base = ttk.Combobox(base_row, textvariable=ov_state["base_src"],
                                   values=_base_options(), width=18,
                                   state="readonly", font=("DejaVu Sans", 8))
            cb_base.pack(side="left")
            cb_base.bind("<Button-1>",
                         lambda e, c=cb_base: c.configure(values=_base_options()))

            # Mask 2 - cascading From/Step picker. "none" disables it.
            tk.Label(ov_fr, text="Mask 2  v  pick From -> Step:",
                     font=("DejaVu Sans", 9, "bold"), fg="#cc88ff",
                     anchor="w").pack(fill="x", padx=3, pady=(4, 0))
            self._make_cascading_picker(
                ov_fr, ov_state["mask2_src"], pipeline_list,
                label_from="  From:", label_step="  Step:",
                allow_none=True)

            # C1 = colour for Mask 1 (the running mask from prev step)
            col1_row = tk.Frame(ov_fr); col1_row.pack(fill="x", pady=1)
            tk.Label(col1_row, text="C1:", font=("DejaVu Sans", 8, "bold"),
                     width=8, anchor="w").pack(side="left")
            ttk.Combobox(col1_row, textvariable=ov_state["color1"],
                         values=list(OVERLAY_COLORS.keys()),
                         width=10, state="readonly",
                         font=("DejaVu Sans", 8)).pack(side="left")
            tk.Label(col1_row, text=" colour of Mask 1 (running)",
                     font=("DejaVu Sans", 9), fg="#c8c8c8"
                     ).pack(side="left", padx=2)

            # C2 = colour for Mask 2
            col2_row = tk.Frame(ov_fr); col2_row.pack(fill="x", pady=1)
            tk.Label(col2_row, text="C2:", font=("DejaVu Sans", 8, "bold"),
                     width=8, anchor="w").pack(side="left")
            ttk.Combobox(col2_row, textvariable=ov_state["color2"],
                         values=list(OVERLAY_COLORS.keys()),
                         width=10, state="readonly",
                         font=("DejaVu Sans", 8)).pack(side="left")
            tk.Label(col2_row, text=" colour of Mask 2",
                     font=("DejaVu Sans", 9), fg="#c8c8c8"
                     ).pack(side="left", padx=2)

            tk.Label(ov_fr,
                     text=("Composite shows: Mask 1 [+ Mask 2] [+ Base] -> "
                           "overlay. Mask 2 = none and Base = none are both "
                           "valid (compose only what's enabled)."),
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
                     wraplength=260, justify="left"
                     ).pack(anchor="w", padx=3)
            # Lazily-built widgets miss the one-shot restyle the caller
            # runs after _add_step_frame — restyle this subtree now.
            if hasattr(self, "_restyle_subtree"):
                self._restyle_subtree(combine_fr)

        # Show only the relevant block based on the radio state, AND
        # hide the From/Step pickers + caption when op == OVERLAY
        # (OVERLAY uses its own Base / Mask 2 selectors instead).
        def _refresh_visibility(*_):
            # Guard: this callback is bound to PERSISTENT step vars
            # (kind_var / comb_op_var live in the pipeline tuple, not
            # the card). After a card rebuild the old card's widgets
            # are gone but its trace is still registered — bail out
            # instead of touching a dead `.!frame3` path name.
            if not yolo_fr.winfo_exists():
                return
            k = kind_var.get()
            if k == "yolo":
                # YOLO-only step: hide morph & combine — the YOLO
                # block (always pinned at the bottom) is the whole step.
                morph_fr.pack_forget()
                combine_fr.pack_forget()
            elif k == "combine":
                _build_combine()           # build on first need
                morph_fr.pack_forget()
                # `before=yolo_fr` keeps the Combine block above YOLO.
                combine_fr.pack(fill="x", before=yolo_fr)
                if comb_op_var.get() == "OVERLAY":
                    W["src_row"].pack_forget()
                    W["view_row"].pack_forget()
                    W["src_caption"].pack_forget()
                    W["ov_fr"].pack(fill="x")
                else:
                    W["src_row"].pack(fill="x", pady=1)
                    W["view_row"].pack(fill="x", pady=1)
                    W["src_caption"].pack(anchor="w", padx=3)
                    W["ov_fr"].pack_forget()
            else:  # "morph"
                combine_fr.pack_forget()
                morph_fr.pack(fill="x", before=yolo_fr)

        def _sync_kind(*_):
            # Keep the legacy comb_en boolean in lock-step with the
            # 3-way kind so save/load + the morph/combine processing
            # path need no change. Selecting "yolo" auto-enables the
            # step's YOLO add-on (the block IS the step now).
            k = kind_var.get()
            want_combine = (k == "combine")
            try:
                if bool(comb_en_var.get()) != want_combine:
                    comb_en_var.set(want_combine)
            except Exception:
                pass
            if k == "yolo":
                try:
                    yolo_st["yolo_en"].set(True)
                except Exception:
                    pass
            _refresh_visibility()
        # Capture trace IDs so they can be removed when this card is
        # destroyed — otherwise every rebuild leaves a dead trace on
        # these persistent vars and they pile up forever.
        _card_traces.append((kind_var, kind_var.trace_add("write", _sync_kind)))
        _card_traces.append(
            (comb_op_var, comb_op_var.trace_add("write", _refresh_visibility)))

        # -- YOLO add-on: header eager, body lazy ---------------------
        # Only the Enable checkbox is built up-front; the rest of the
        # block (input radios, From/Step pickers, mode, class filter)
        # is built the first time YOLO is enabled or the step Type
        # becomes "yolo".
        yolo_top = tk.Frame(yolo_fr); yolo_top.pack(fill="x", padx=2, pady=1)
        tk.Checkbutton(yolo_top, text="Enable",
                       variable=yolo_st["yolo_en"],
                       font=("DejaVu Sans", 9, "bold"), fg="#ffaa66"
                       ).pack(side="left")
        tk.Label(yolo_top, text=" runs on:",
                 font=("DejaVu Sans", 9), fg="#c8c8c8").pack(side="left")

        def _refresh_yolo_input_visibility(*_a):
            # Same stale-trace guard as _refresh_visibility: yolo_st's
            # input-kind var outlives the card it was built for.
            if not yolo_fr.winfo_exists() or not W.get("yolo_built"):
                return
            if yolo_st["yolo_input_kind"].get() == "mask":
                W["img_pick_fr"].pack_forget()
                W["mask_pick_fr"].pack(fill="x", before=W["yolo_mode_row"])
            else:
                W["mask_pick_fr"].pack_forget()
                W["img_pick_fr"].pack(fill="x", before=W["yolo_mode_row"])

        def _build_yolo(*_):
            """Populate yolo_fr the first time YOLO is enabled / the
            step Type becomes 'yolo'. Idempotent."""
            if W.get("yolo_built") or not yolo_fr.winfo_exists():
                return
            W["yolo_built"] = True

            # Input-kind radio: feed YOLO a raw IMAGE view or a MASK view.
            kind_row = tk.Frame(yolo_fr); kind_row.pack(fill="x", padx=2, pady=1)
            tk.Label(kind_row, text="  Input:", font=("DejaVu Sans", 9),
                     width=8, anchor="w").pack(side="left")
            tk.Radiobutton(kind_row, text="image",
                           variable=yolo_st["yolo_input_kind"],
                           value="image",
                           font=("DejaVu Sans", 9), fg="#ffaa66"
                           ).pack(side="left")
            tk.Radiobutton(kind_row, text="mask",
                           variable=yolo_st["yolo_input_kind"],
                           value="mask",
                           font=("DejaVu Sans", 9), fg="#ffaa66"
                           ).pack(side="left")

            # Image-source picker (RAW BGR views). Visible when Input == image.
            img_pick_fr = tk.Frame(yolo_fr)
            self._make_cascading_picker(
                img_pick_fr, yolo_st["yolo_src"], pipeline_list,
                label_from="  From:", label_step="  Step:",
                allow_none=False)
            # Mask-source picker (binary mask views). Visible when Input == mask.
            mask_pick_fr = tk.Frame(yolo_fr)
            self._make_cascading_picker(
                mask_pick_fr, yolo_st["yolo_mask_src"], pipeline_list,
                label_from="  M-from:", label_step="  M-step:",
                allow_none=False)
            W["img_pick_fr"] = img_pick_fr
            W["mask_pick_fr"] = mask_pick_fr

            # Mode dropdown: how the YOLO box union affects the running mask.
            yolo_fr_mode_row = tk.Frame(yolo_fr)
            yolo_fr_mode_row.pack(fill="x", padx=2, pady=1)
            W["yolo_mode_row"] = yolo_fr_mode_row
            tk.Label(yolo_fr_mode_row, text="  Mode:", font=("DejaVu Sans", 9),
                     width=8, anchor="w").pack(side="left")
            ttk.Combobox(yolo_fr_mode_row, textvariable=yolo_st["yolo_mode"],
                         values=["box_only", "focus", "subtract"],
                         width=10, state="readonly",
                         font=("DejaVu Sans", 9)).pack(side="left")

            # Per-step YOLO box size — scale % + pixel pad applied to
            # every detected box for THIS step only (overrides the
            # shared YOLO box size). Bigger = leave a margin to
            # subtract; smaller = focus tightly inside the box.
            box_row = tk.Frame(yolo_fr); box_row.pack(fill="x", padx=2, pady=1)
            tk.Label(box_row, text="  Box:", font=("DejaVu Sans", 9),
                     width=8, anchor="w").pack(side="left")
            tk.Label(box_row, text="scale%", font=("DejaVu Sans", 9),
                     fg="#ffaa66").pack(side="left", padx=(0, 2))
            tk.Spinbox(box_row, textvariable=yolo_st["yolo_box_scale"],
                       from_=20, to=300, width=5,
                       font=("DejaVu Sans", 9)).pack(side="left")
            tk.Label(box_row, text="pad px", font=("DejaVu Sans", 9),
                     fg="#ffaa66").pack(side="left", padx=(8, 2))
            tk.Spinbox(box_row, textvariable=yolo_st["yolo_box_pad"],
                       from_=-200, to=200, width=5,
                       font=("DejaVu Sans", 9)).pack(side="left")

            # Per-step class filter — a tick box per model class.
            yolo_cls_row = tk.Frame(yolo_fr)
            yolo_cls_row.pack(fill="x", padx=2, pady=1)
            tk.Label(yolo_cls_row, text="  Detect:", font=("DejaVu Sans", 9),
                     width=8, anchor="nw").pack(side="left", anchor="n")
            _cls_box = tk.Frame(yolo_cls_row)
            _cls_box.pack(side="left", fill="x")
            _model = getattr(self, "yolo_model", None)
            if _model is not None:
                try:
                    _names = _model.names
                    _cls_items = (sorted(_names.items(), key=lambda x: int(x[0]))
                                  if isinstance(_names, dict)
                                  else list(enumerate(_names)))
                except Exception:
                    _cls_items = []
                for _cid, _cname in _cls_items:
                    _cv = self._yolo_class_var(yolo_st, int(_cid))
                    tk.Checkbutton(_cls_box, text=f"{_cid}: {_cname}",
                                   variable=_cv, font=("DejaVu Sans", 9),
                                   fg="#ffaa66", anchor="w"
                                   ).pack(anchor="w")
            else:
                tk.Label(_cls_box, text="(no model loaded)",
                         font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8"
                         ).pack(anchor="w")

            tk.Label(yolo_fr,
                     text=("box_only = just draw boxes  |  "
                           "focus = keep mask only inside boxes  |  "
                           "subtract = remove mask pixels inside boxes  |  "
                           "Detect = tick the classes this step keeps  |  "
                           "Box scale%/pad resize each box for this step  |  "
                           "Input=mask runs YOLO on the chosen mask view"),
                     font=("DejaVu Sans", 9, "italic"), fg="#c8c8c8",
                     wraplength=240, justify="left"
                     ).pack(anchor="w", padx=4)
            _card_traces.append(
                (yolo_st["yolo_input_kind"],
                 yolo_st["yolo_input_kind"].trace_add(
                     "write", _refresh_yolo_input_visibility)))
            _refresh_yolo_input_visibility()
            if hasattr(self, "_restyle_subtree"):
                self._restyle_subtree(yolo_fr)

        def _on_yolo_en(*_):
            if yolo_fr.winfo_exists() and yolo_st["yolo_en"].get():
                _build_yolo()
        _card_traces.append(
            (yolo_st["yolo_en"],
             yolo_st["yolo_en"].trace_add("write", _on_yolo_en)))

        # Initial state: build whatever the loaded config already needs.
        _refresh_visibility()
        if yolo_st["yolo_en"].get() or kind_var.get() == "yolo":
            _build_yolo()

        # Drop this card's traces from the persistent step vars when
        # the card frame is destroyed (rebuild / remove step). The
        # `is sf` check filters out <Destroy> events bubbling up from
        # child widgets.
        def _cleanup_card_traces(event, _sf=sf, _traces=_card_traces):
            if event.widget is not _sf:
                return
            for _var, _tid in _traces:
                try:
                    _var.trace_remove("write", _tid)
                except Exception:
                    pass
        sf.bind("<Destroy>", _cleanup_card_traces, add="+")
        return sf

    # ------------------------------------------------------------------
    # Add / move / remove
    # ------------------------------------------------------------------
    def _suppress_refresh(self):
        """Context manager that flips _loading_config so every trace
        that ends up calling _refresh during topology changes
        (var-set storms when building / rebuilding step cards) is a
        no-op. Caller commits with Apply (F5) or playback."""
        analyzer = self
        class _Ctx:
            def __enter__(self_inner):
                self_inner._prev = getattr(analyzer, "_loading_config", False)
                analyzer._loading_config = True
                return self_inner
            def __exit__(self_inner, *exc):
                analyzer._loading_config = self_inner._prev
                return False
        return _Ctx()

    def _on_add_step(self, parent_frame, pipeline_list):
        # Topology change — refresh suppressed throughout. _refresh()
        # short-circuits on self._loading_config, so any incidental
        # trace firing during card construction is harmless.
        with self._suppress_refresh():
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
            if not hasattr(parent_frame, "_cards"):
                parent_frame._cards = []
            new_idx = len(pipeline_list) - 1
            sf = self._add_step_frame(parent_frame, pipeline_list, new_idx)
            parent_frame._cards.append(sf)
            self._grid_step_card(sf, new_idx)
            self._place_add_btn(parent_frame, pipeline_list)
            if hasattr(self, "_restyle_subtree"):
                self._restyle_subtree(sf)

    def _on_move_step(self, parent_frame, pipeline_list, sf, direction):
        # Incremental: swap two entries + re-grid; nothing is rebuilt.
        idx = self._step_card_index(parent_frame, sf)
        if idx < 0:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(pipeline_list):
            return
        with self._suppress_refresh():
            pipeline_list[idx], pipeline_list[new_idx] = \
                pipeline_list[new_idx], pipeline_list[idx]
            cards = parent_frame._cards
            cards[idx], cards[new_idx] = cards[new_idx], cards[idx]
            self._relayout_step_cards(parent_frame,
                                      self.STEPS_PER_UI_ROW, "Step")
            self._place_add_btn(parent_frame, pipeline_list)

    def _on_remove_step(self, parent_frame, pipeline_list, sf):
        # Incremental: drop ONLY this card, re-grid the rest. The card
        # is hidden immediately (grid_forget) and its ~100 child widgets
        # destroyed off the critical path so the UI updates at once.
        idx = self._step_card_index(parent_frame, sf)
        if idx < 0 or not pipeline_list:
            return
        with self._suppress_refresh():
            del pipeline_list[idx]
            parent_frame._cards.pop(idx)
            sf.grid_forget()
            self._relayout_step_cards(parent_frame,
                                      self.STEPS_PER_UI_ROW, "Step")
            self._place_add_btn(parent_frame, pipeline_list)
        self.root.after(10, sf.destroy)

    # ------------------------------------------------------------------
    # Steps-per-row config (Settings window spinboxes)
    # ------------------------------------------------------------------
    def _on_steps_per_row(self, which, var):
        """Change how many step cards wrap per row. `which` is
        'morph' or 'pm'. Re-grids every pipeline (RGB / IR main +
        all branches) so the new layout takes effect immediately."""
        try:
            n = int(var.get())
        except Exception:
            return
        if n < 1 or n > 12:
            return
        # Idempotent: tk.Spinbox fires its variable's write trace on
        # creation, so bail out when nothing actually changed (else
        # merely opening the Settings window rebuilds every pipeline).
        cur = (self.PM_STEPS_PER_UI_ROW if which == "pm"
               else self.STEPS_PER_UI_ROW)
        if n == cur:
            return
        if which == "pm":
            self.PM_STEPS_PER_UI_ROW = n
            self._save_pref("pm_steps_per_ui_row", n)
        else:
            self.STEPS_PER_UI_ROW = n
            self._save_pref("steps_per_ui_row", n)
        # Re-grid all step panels with the new wrap width.
        with self._suppress_refresh():
            self._rebuild_pre_morph_pipeline_ui(
                self.rgb_pre_pip_frame, self.rgb_pre_pipeline)
            self._rebuild_pre_morph_pipeline_ui(
                self.ir_pre_pip_frame, self.ir_pre_pipeline)
            self._rebuild_pipeline_ui(self.rgb_pip_frame, self.rgb_pipeline)
            self._rebuild_pipeline_ui(self.ir_pip_frame, self.ir_pipeline)
            for _rec in getattr(self, "user_pipelines", []):
                if _rec.get("pre_steps_row") is not None:
                    self._rebuild_pre_morph_pipeline_ui(
                        _rec["pre_steps_row"], _rec["pre_steps"])
                if _rec.get("steps_row") is not None:
                    self._rebuild_pipeline_ui(
                        _rec["steps_row"], _rec["steps"])
