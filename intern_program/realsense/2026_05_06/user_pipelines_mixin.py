"""Mixin: user-defined parallel pipelines (branches).

Each branch has a TYPE picked at creation time:

  • ``rgb``    — same H1/H2 + S + V detection structure as the main
                 RGB pipeline, with a CHANNELS dropdown that selects
                 which channels participate in the AND combination
                 (H, S, V, HS, HV, SV, HSV, full).
  • ``ir``     — IR threshold detection (lo/hi).
  • ``custom`` — pass-through: pick any view in the program (e.g.
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
        for p in self.user_pipelines:
            nm = p["name"].get().strip()
            if not nm:
                continue
            ups.append(f"up_{nm}")
            ups += [f"up_{nm}_det",
                    f"up_{nm}_h1_mask", f"up_{nm}_h2_mask",
                    f"up_{nm}_s_mask",  f"up_{nm}_v_mask",
                    f"up_{nm}_ir_mask"]
            for i in range(len(p["steps"])):
                ups.append(f"up_{nm}_step{i+1}")
        return list(VIEW_OPTIONS) + ups

    # ------------------------------------------------------------------
    # Add / remove a branch
    # ------------------------------------------------------------------
    def _on_add_user_pipeline(self):
        """Open a 'New Branch' setup dialog. The user picks the type
        (rgb / ir / custom), tunes the detection or source, then clicks
        Submit. Only after Submit is the branch actually added."""
        idx = len(self.user_pipelines) + 1
        nm  = f"branch{idx}"

        # Working IntVars / StringVars / BooleanVars for the dialog.
        # If the user cancels, none of these become a real branch.
        rec = {
            "name":     tk.StringVar(value=nm),
            "type":     tk.StringVar(value="rgb"),
            "source":   tk.StringVar(value=f"up_{nm}_det"),
            "color":    tk.StringVar(value="cyan"),
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
            # IR threshold mirrors main IR pipeline.
            "ir_lo":    tk.IntVar(value=80),
            "ir_hi":    tk.IntVar(value=255),
            "steps":    [],
            "frame":    None,
            "steps_row": None,
        }

        self._open_branch_setup_dialog(rec, on_submit=self._submit_new_branch)

    def _submit_new_branch(self, rec):
        """Called by the setup dialog's Submit button."""
        self.user_pipelines.append(rec)
        # Trace every config var so changes auto-refresh.
        for _key in ("type", "channels",
                     "h1_lo", "h1_hi", "h2_lo", "h2_hi",
                     "s_lo",  "s_hi",  "v_lo",  "v_hi",
                     "ir_lo", "ir_hi",
                     "source", "color"):
            rec[_key].trace_add("write", lambda *a: self._refresh())
        rec["name"].trace_add(
            "write",
            lambda *a: (self._rebuild_all_masks_if_open(), self._refresh()))
        self._build_user_pipeline_frame(rec)
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _on_remove_user_pipeline(self, rec):
        try:
            self.user_pipelines.remove(rec)
            if rec["frame"] is not None:
                rec["frame"].destroy()
        except ValueError:
            pass
        self._rebuild_all_masks_if_open()
        self._refresh()

    # ------------------------------------------------------------------
    # 'New Branch' setup dialog (modal)
    # ------------------------------------------------------------------
    def _open_branch_setup_dialog(self, rec, on_submit):
        dlg = tk.Toplevel(self.root)
        dlg.title("New Branch — pick type and detection settings")
        dlg.transient(self.root)
        dlg.grab_set()

        top = tk.Frame(dlg, padx=12, pady=10)
        top.pack(fill="both", expand=True)

        # ── Name + Colour ────────────────────────────────────────────
        row1 = tk.Frame(top)
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="Name:",
                 font=("Arial", 9, "bold")).pack(side="left")
        tk.Entry(row1, textvariable=rec["name"], width=14,
                 font=("Arial", 9)).pack(side="left", padx=4)
        tk.Label(row1, text="  Colour:",
                 font=("Arial", 9)).pack(side="left")
        ttk.Combobox(row1, textvariable=rec["color"],
                     values=["cyan", "yellow", "magenta", "green",
                             "orange", "red", "white"],
                     width=10, state="readonly",
                     font=("Arial", 9)).pack(side="left", padx=4)

        # ── Type radio buttons ───────────────────────────────────────
        type_lf = tk.LabelFrame(top, text="Branch type",
                                font=("Arial", 9, "bold"),
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
                           font=("Arial", 9, "bold"),
                           command=lambda: _refresh_panels()
                           ).pack(side="left")
            tk.Label(row, text=hint,
                     font=("Arial", 7), fg="#888",
                     wraplength=420, justify="left"
                     ).pack(side="left", padx=4)

        # ── RGB / HSV detection panel ────────────────────────────────
        rgb_lf = tk.LabelFrame(top, text="RGB / HSV detection",
                               font=("Arial", 9, "bold"))

        # Channel-mode dropdown.
        chrow = tk.Frame(rgb_lf)
        chrow.pack(fill="x", padx=6, pady=3)
        tk.Label(chrow, text="Channels to use:",
                 font=("Arial", 9, "bold")).pack(side="left")
        ttk.Combobox(chrow, textvariable=rec["channels"],
                     values=CHANNEL_MODES,
                     width=8, state="readonly",
                     font=("Arial", 9)).pack(side="left", padx=4)
        tk.Label(chrow,
                 text=("H = (H1 OR H2).  S, V each individual.  "
                       "Combo (e.g. HS) = AND of those.  "
                       "full = (H1 OR H2) AND S AND V."),
                 font=("Arial", 7), fg="#888",
                 wraplength=480, justify="left"
                 ).pack(side="left", padx=4)

        def _h_row(parent, label, lo_var, hi_var, lo_max):
            r = tk.Frame(parent)
            r.pack(fill="x", padx=6, pady=1)
            tk.Label(r, text=label, font=("Arial", 9),
                     width=10, anchor="w").pack(side="left")
            tk.Label(r, text="lo", fg="#888",
                     font=("Arial", 8)).pack(side="left")
            tk.Spinbox(r, textvariable=lo_var, from_=0, to=lo_max,
                       width=5, font=("Arial", 9)
                       ).pack(side="left", padx=2)
            tk.Scale(r, variable=lo_var, from_=0, to=lo_max,
                     orient="horizontal", length=140,
                     showvalue=False).pack(side="left")
            tk.Label(r, text="hi", fg="#888",
                     font=("Arial", 8)).pack(side="left")
            tk.Spinbox(r, textvariable=hi_var, from_=0, to=lo_max,
                       width=5, font=("Arial", 9)
                       ).pack(side="left", padx=2)
            tk.Scale(r, variable=hi_var, from_=0, to=lo_max,
                     orient="horizontal", length=140,
                     showvalue=False).pack(side="left")

        _h_row(rgb_lf, "H1",   rec["h1_lo"], rec["h1_hi"], H_MAX)
        _h_row(rgb_lf, "H2",   rec["h2_lo"], rec["h2_hi"], H_MAX)
        _h_row(rgb_lf, "S",    rec["s_lo"],  rec["s_hi"],  255)
        _h_row(rgb_lf, "V",    rec["v_lo"],  rec["v_hi"],  255)

        # ── IR detection panel ───────────────────────────────────────
        ir_lf = tk.LabelFrame(top, text="IR threshold detection",
                              font=("Arial", 9, "bold"))
        _h_row(ir_lf, "IR",   rec["ir_lo"], rec["ir_hi"], 255)

        # ── Custom-source panel ──────────────────────────────────────
        cust_lf = tk.LabelFrame(top, text="Custom source",
                                font=("Arial", 9, "bold"))
        crow = tk.Frame(cust_lf)
        crow.pack(fill="x", padx=6, pady=4)
        tk.Label(crow, text="Source view:",
                 font=("Arial", 9, "bold")).pack(side="left")
        cust_cb = ttk.Combobox(crow, textvariable=rec["source"],
                               values=self._all_view_names(),
                               width=24, state="readonly",
                               font=("Arial", 9))
        cust_cb.pack(side="left", padx=4)
        cust_cb.bind("<Button-1>",
                     lambda e, c=cust_cb:
                         c.configure(values=self._all_view_names()))

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

        # ── Buttons ──────────────────────────────────────────────────
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

        tk.Button(btn, text="✓ Submit", bg="#225522", fg="white",
                  font=("Arial", 9, "bold"),
                  width=12, command=_submit).pack(side="right", padx=4)
        tk.Button(btn, text="Cancel",
                  font=("Arial", 9),
                  width=10, command=_cancel).pack(side="right")

    # ------------------------------------------------------------------
    # Build one branch's permanent UI block (after Submit)
    # ------------------------------------------------------------------
    def _build_user_pipeline_frame(self, rec):
        outer = tk.LabelFrame(self.user_pipelines_host,
                              text=f"Pipeline {len(self.user_pipelines)} "
                                   f"[{rec['type'].get()}]",
                              font=("Arial", 8, "bold"),
                              fg="#cc88ff", bd=2)
        outer.pack(fill="x", padx=2, pady=3)
        rec["frame"] = outer

        hdr = tk.Frame(outer)
        hdr.pack(fill="x", padx=3, pady=2)

        tk.Label(hdr, text="Name:",
                 font=("Arial", 8)).pack(side="left")
        tk.Entry(hdr, textvariable=rec["name"], width=12,
                 font=("Arial", 8)).pack(side="left", padx=2)
        tk.Label(hdr, text=" → up_<name>",
                 font=("Arial", 7), fg="#888").pack(side="left")

        tk.Label(hdr, text="   Source:",
                 font=("Arial", 8)).pack(side="left")
        src_cb = ttk.Combobox(hdr, textvariable=rec["source"],
                              values=self._all_view_names(),
                              width=20, state="readonly",
                              font=("Arial", 8))
        src_cb.pack(side="left", padx=2)
        src_cb.bind("<Button-1>",
                    lambda e, cb=src_cb:
                        cb.configure(values=self._all_view_names()))

        tk.Label(hdr, text=" Colour:",
                 font=("Arial", 8)).pack(side="left", padx=(8, 0))
        ttk.Combobox(hdr, textvariable=rec["color"],
                     values=["cyan", "yellow", "magenta", "green",
                             "orange", "red", "white"],
                     width=8, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)

        tk.Button(hdr, text="✎ Edit detection",
                  font=("Arial", 8),
                  command=lambda r=rec: self._edit_branch_detection(r)
                  ).pack(side="right", padx=4)
        tk.Button(hdr, text="× Delete", fg="#ff4444",
                  font=("Arial", 8, "bold"),
                  command=lambda r=rec: self._on_remove_user_pipeline(r)
                  ).pack(side="right", padx=4)

        # Compact detection summary row (read-only display).
        det_summary = tk.Label(outer, text=self._branch_summary(rec),
                               font=("Arial", 7, "italic"),
                               fg="#aa88cc", anchor="w", justify="left")
        det_summary.pack(fill="x", padx=6)
        rec["det_summary"] = det_summary
        # Refresh summary text whenever any config var changes.
        for _key in ("type", "channels",
                     "h1_lo", "h1_hi", "h2_lo", "h2_hi",
                     "s_lo",  "s_hi",  "v_lo",  "v_hi",
                     "ir_lo", "ir_hi", "source"):
            rec[_key].trace_add(
                "write",
                lambda *a, r=rec, lbl=det_summary:
                    lbl.config(text=self._branch_summary(r)))

        steps_row = tk.Frame(outer)
        steps_row.pack(fill="x", padx=3, pady=2)
        rec["steps_row"] = steps_row
        self._rebuild_pipeline_ui(steps_row, rec["steps"])

    def _branch_summary(self, rec):
        t = rec["type"].get()
        if t == "rgb":
            ch = rec["channels"].get()
            return (f"  type=RGB  channels={ch}  "
                    f"H1=[{rec['h1_lo'].get()}..{rec['h1_hi'].get()}]  "
                    f"H2=[{rec['h2_lo'].get()}..{rec['h2_hi'].get()}]  "
                    f"S=[{rec['s_lo'].get()}..{rec['s_hi'].get()}]  "
                    f"V=[{rec['v_lo'].get()}..{rec['v_hi'].get()}]  "
                    f"→ up_{rec['name'].get()}_det")
        if t == "ir":
            return (f"  type=IR  "
                    f"thresh=[{rec['ir_lo'].get()}..{rec['ir_hi'].get()}]  "
                    f"→ up_{rec['name'].get()}_det")
        return f"  type=Custom  source={rec['source'].get()}  (no detection)"

    def _edit_branch_detection(self, rec):
        """Re-open the setup dialog for an existing branch."""
        self._open_branch_setup_dialog(
            rec,
            on_submit=lambda r: (self._rebuild_all_masks_if_open(),
                                 self._refresh()))
