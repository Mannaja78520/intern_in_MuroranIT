"""Mixin: user-defined parallel pipelines.

A user pipeline is an independent processing chain that takes any view
(built-in like rgb_hsv_H, rgb_mask, ir_thresh, or another pipeline's
output) as input, runs its own ordered list of steps, and exposes its
final mask as a new view named ``up_<name>``.

Multiple pipelines run in parallel and their outputs can be combined via
the existing combine UI or referenced as the source of yet another
user pipeline — letting the user chain HSV-channel branches, IR
branches, etc. and merge them however they like.

Host class must already include PipelineUIMixin (for _add_step_frame /
_create_default_steps tuple shape) and provide:
    self.root, self.user_pipelines, self.user_pipelines_host
    self._refresh, self._rebuild_all_masks_if_open
"""
import tkinter as tk
from tkinter import ttk

from config import VIEW_OPTIONS


class UserPipelinesMixin:
    """Adds user-defined parallel pipeline UI methods."""

    # ------------------------------------------------------------------
    # Public list of selectable source views (built-ins + per-channel
    # masks + every other user-pipeline output).
    # ------------------------------------------------------------------
    def _all_view_names(self):
        ups = []
        for p in self.user_pipelines:
            nm = p["name"].get().strip()
            if not nm:
                continue
            ups.append(f"up_{nm}")
            # Per-branch detection masks: combined detection + each
            # individual channel result. All available as sources /
            # combine refs to any pipeline.
            ups += [f"up_{nm}_det",
                    f"up_{nm}_h1_mask", f"up_{nm}_h2_mask",
                    f"up_{nm}_s_mask",
                    f"up_{nm}_v_mask",
                    f"up_{nm}_ir_mask"]
            # Expose every step as a view so it can be picked as a
            # combine source from any other pipeline.
            for i in range(len(p["steps"])):
                ups.append(f"up_{nm}_step{i+1}")
        return list(VIEW_OPTIONS) + ups

    # ------------------------------------------------------------------
    # Add / remove
    # ------------------------------------------------------------------
    def _on_add_user_pipeline(self):
        idx  = len(self.user_pipelines) + 1
        nm   = f"branch{idx}"
        # Detection panel: each channel has an enable checkbox + two
        # threshold spinboxes/sliders. H gets two ranges (H1, H2) so
        # wrap-around colours like red work.
        rec  = {
            "name":   tk.StringVar(value=nm),
            "source": tk.StringVar(value=f"up_{nm}_det"),
            "color":  tk.StringVar(value="cyan"),
            # Per-channel enable flags (which channels participate).
            "en_h1":  tk.BooleanVar(value=True),
            "en_h2":  tk.BooleanVar(value=True),
            "en_s":   tk.BooleanVar(value=True),
            "en_v":   tk.BooleanVar(value=False),
            "en_ir":  tk.BooleanVar(value=False),
            # H1 — first hue range (default red-low).
            "h1_lo":  tk.IntVar(value=0),
            "h1_hi":  tk.IntVar(value=10),
            # H2 — second hue range (default red-high, wrap-around).
            "h2_lo":  tk.IntVar(value=170),
            "h2_hi":  tk.IntVar(value=179),
            "s_lo":   tk.IntVar(value=120),
            "s_hi":   tk.IntVar(value=255),
            "v_lo":   tk.IntVar(value=70),
            "v_hi":   tk.IntVar(value=255),
            "ir_lo":  tk.IntVar(value=80),
            "ir_hi":  tk.IntVar(value=255),
            "steps":  [],
            "frame":  None,
            "steps_row": None,
        }
        self.user_pipelines.append(rec)
        # Trace every detection / config var so any change immediately
        # re-processes and updates panels / flowchart.
        for _key in ("en_h1", "en_h2", "en_s", "en_v", "en_ir",
                     "h1_lo", "h1_hi", "h2_lo", "h2_hi",
                     "s_lo",  "s_hi",  "v_lo",  "v_hi",
                     "ir_lo", "ir_hi", "source", "color"):
            rec[_key].trace_add("write", lambda *a: self._refresh())
        rec["name"].trace_add("write",
                              lambda *a: (self._rebuild_all_masks_if_open(),
                                          self._refresh()))
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
    # Build one pipeline's UI block (header + step row)
    # ------------------------------------------------------------------
    def _build_user_pipeline_frame(self, rec):
        outer = tk.LabelFrame(self.user_pipelines_host,
                              text=f"Pipeline {len(self.user_pipelines)}",
                              font=("Arial", 8, "bold"),
                              fg="#cc88ff", bd=2)
        outer.pack(fill="x", padx=2, pady=3)
        rec["frame"] = outer

        hdr = tk.Frame(outer)
        hdr.pack(fill="x", padx=3, pady=2)

        tk.Label(hdr, text="Name:",
                 font=("Arial", 8)).pack(side="left")
        name_e = tk.Entry(hdr, textvariable=rec["name"], width=12,
                          font=("Arial", 8))
        name_e.pack(side="left", padx=2)

        tk.Label(hdr, text=" → up_<name>",
                 font=("Arial", 7), fg="#888").pack(side="left")

        tk.Label(hdr, text="   Source:",
                 font=("Arial", 8)).pack(side="left")
        src_cb = ttk.Combobox(hdr, textvariable=rec["source"],
                              values=self._all_view_names(),
                              width=18, state="readonly", font=("Arial", 8))
        src_cb.pack(side="left", padx=2)
        # Refresh source-list each time it's opened so newly created
        # user pipelines show up.
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

        tk.Button(hdr, text="× Delete", fg="#ff4444",
                  font=("Arial", 8, "bold"),
                  command=lambda r=rec: self._on_remove_user_pipeline(r)
                  ).pack(side="right", padx=4)

        # Per-branch detection panel. Selected channels are AND'd
        # together (H1 OR H2 first, then AND with S, V, IR as enabled).
        # Result is exposed as  up_<name>_det  and is the default
        # source for this branch's pipeline.
        det = tk.LabelFrame(
            outer,
            text=("Detection channels — selected channels are AND'd "
                  "(H1 OR H2 inside).  Result → up_<name>_det"),
            font=("Arial", 7, "italic"), fg="#aa88cc")
        det.pack(fill="x", padx=3, pady=(0, 3))

        def _ch_row(parent, label, en_var, lo_var, hi_var, lo_max=255):
            r = tk.Frame(parent)
            r.pack(fill="x", padx=4, pady=0)
            tk.Checkbutton(r, text=label, variable=en_var,
                           font=("Arial", 7, "bold"),
                           width=12, anchor="w").pack(side="left")
            tk.Label(r, text="lo", font=("Arial", 7),
                     fg="#888").pack(side="left")
            tk.Spinbox(r, textvariable=lo_var, from_=0, to=lo_max,
                       width=4, font=("Arial", 7)).pack(side="left", padx=1)
            tk.Scale(r, variable=lo_var, from_=0, to=lo_max,
                     orient="horizontal", length=70, showvalue=False,
                     font=("Arial", 7)).pack(side="left")
            tk.Label(r, text="hi", font=("Arial", 7),
                     fg="#888").pack(side="left")
            tk.Spinbox(r, textvariable=hi_var, from_=0, to=lo_max,
                       width=4, font=("Arial", 7)).pack(side="left", padx=1)
            tk.Scale(r, variable=hi_var, from_=0, to=lo_max,
                     orient="horizontal", length=70, showvalue=False,
                     font=("Arial", 7)).pack(side="left")

        _ch_row(det, "☑ H1",  rec["en_h1"],
                rec["h1_lo"], rec["h1_hi"], 179)
        _ch_row(det, "☑ H2",  rec["en_h2"],
                rec["h2_lo"], rec["h2_hi"], 179)
        _ch_row(det, "☑ S",   rec["en_s"],
                rec["s_lo"],  rec["s_hi"])
        _ch_row(det, "☑ V",   rec["en_v"],
                rec["v_lo"],  rec["v_hi"])
        _ch_row(det, "☑ IR",  rec["en_ir"],
                rec["ir_lo"], rec["ir_hi"])
        tk.Label(det,
                 text=("Examples:  H1+H2 → red ring (OR).  "
                       "H1+S+V → coloured object (AND).  "
                       "S only → saturation mask.  "
                       "All four → (H1 OR H2) AND S AND V."),
                 font=("Arial", 7), fg="#888888",
                 wraplength=900, justify="left").pack(anchor="w", padx=5)

        steps_row = tk.Frame(outer)
        steps_row.pack(fill="x", padx=3, pady=2)
        rec["steps_row"] = steps_row
        self._rebuild_pipeline_ui(steps_row, rec["steps"])
