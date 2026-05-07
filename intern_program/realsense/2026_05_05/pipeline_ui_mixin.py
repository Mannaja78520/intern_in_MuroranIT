"""Mixin: dynamic pipeline-step UI (the row of "Step N" cards in the side panel).

Concerns: building/destroying step frames, add/remove/move buttons, default
step seeding. All methods operate on a `pipeline_list` of 10-tuples of tk
variables stored on the host class.
"""
import tkinter as tk
from tkinter import ttk

from config import ALL_PROC_OPS


class PipelineUIMixin:
    """Adds dynamic pipeline-step UI methods to the host class.

    Step tuple layout (10 vars):
        (en, op, n, dir, kx, ky, thresh, comb_en, comb_op, comb_src)

    Host class must provide:
        self._refresh()
        self._rebuild_all_masks_if_open()
    """

    # ------------------------------------------------------------------
    # Default seeding
    # ------------------------------------------------------------------
    def _attach_step_traces(self, step_tuple):
        """Trace every step var so editing any field (Enable, Op,
        N, Dir, KX, KY, T, combine_en/op/src) immediately re-runs the
        pipeline and updates the All-Masks flowchart labels."""
        # Combine vars also need a flowchart rebuild because they change
        # whether the step shows the image1/⊕op/image2 sub-row.
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
    def _rebuild_pipeline_ui(self, parent_frame, pipeline_list):
        """Destroy and recreate all step widgets inside parent_frame."""
        for w in parent_frame.winfo_children():
            w.destroy()
        for i in range(len(pipeline_list)):
            self._add_step_frame(parent_frame, pipeline_list, i)
        tk.Button(parent_frame, text="+ Add Step",
                  font=("Arial", 8), bg="#223322", fg="white",
                  command=lambda pl=pipeline_list, pf=parent_frame:
                      self._on_add_step(pf, pl)).pack(side="left", padx=8, pady=10)

    def _add_step_frame(self, parent_frame, pipeline_list, idx):
        en_var, op_var, n_var, dir_var, kx_var, ky_var, thresh_var, \
            comb_en_var, comb_op_var, comb_src_var = pipeline_list[idx]

        sf = tk.LabelFrame(parent_frame, text=f"Step {idx + 1}", font=("Arial", 8))
        sf.pack(side="left", padx=5, pady=3, fill="y")

        # Reorder / remove buttons
        btn_row = tk.Frame(sf)
        btn_row.pack(anchor="ne", padx=1, pady=1)
        tk.Button(btn_row, text="↑", font=("Arial", 7), width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_move_step(pf, pl, i, -1)).pack(side="left")
        tk.Button(btn_row, text="↓", font=("Arial", 7), width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_move_step(pf, pl, i, +1)).pack(side="left")
        tk.Button(btn_row, text="×", font=("Arial", 8, "bold"), fg="#ff4444",
                  width=2, relief="flat",
                  command=lambda pl=pipeline_list, i=idx, pf=parent_frame:
                      self._on_remove_step(pf, pl, i)).pack(side="left")

        tk.Checkbutton(sf, text="Enable", variable=en_var,
                       font=("Arial", 8)).pack(anchor="w")

        def _row(p, lbl, wfn):
            r = tk.Frame(p)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, font=("Arial", 8),
                     width=4, anchor="w").pack(side="left")
            wfn(r)

        _row(sf, "Op:",
             lambda r, v=op_var: ttk.Combobox(
                 r, textvariable=v, values=ALL_PROC_OPS,
                 width=18, state="readonly", font=("Arial", 8)).pack(side="left"))
        _row(sf, "N:",
             lambda r, v=n_var: tk.Spinbox(
                 r, textvariable=v, from_=1, to=20,
                 width=4, font=("Arial", 8)).pack(side="left"))
        _row(sf, "Dir:",
             lambda r, v=dir_var: ttk.Combobox(
                 r, textvariable=v, values=["X", "Y", "XY"],
                 width=4, state="readonly", font=("Arial", 8)).pack(side="left"))
        _row(sf, "KX:",
             lambda r, v=kx_var: tk.Spinbox(
                 r, textvariable=v, from_=1, to=99,
                 width=4, font=("Arial", 8)).pack(side="left"))
        _row(sf, "KY:",
             lambda r, v=ky_var: tk.Spinbox(
                 r, textvariable=v, from_=1, to=99,
                 width=4, font=("Arial", 8)).pack(side="left"))
        _row(sf, "T:",
             lambda r, v=thresh_var: tk.Spinbox(
                 r, textvariable=v, from_=0, to=255,
                 width=4, font=("Arial", 8)).pack(side="left"))

        # ── Combine mode row ──────────────────────────────────────────
        tk.Frame(sf, height=1, bg="#444444").pack(fill="x", padx=2, pady=2)
        c_row = tk.Frame(sf)
        c_row.pack(fill="x", pady=1)
        tk.Checkbutton(c_row, text="⊕Comb", variable=comb_en_var,
                       font=("Arial", 7, "bold"), fg="#bb66ff").pack(side="left")
        ttk.Combobox(c_row, textvariable=comb_op_var,
                     values=["AND", "OR", "XOR"],
                     width=6, state="readonly",
                     font=("Arial", 8)).pack(side="left", padx=2)
        # Combine source: legacy in-pipeline refs + every available view
        # (rgb_*, ir_*, rgb_h_mask, ir_thresh_det, up_*, etc.) so steps
        # in any pipeline can mix with steps in any other pipeline.
        def _src_options():
            legacy = ["mask_pre", "prev"] + [f"step_{i}" for i in range(1, 21)]
            try:
                views = self._all_view_names()
            except Exception:
                views = []
            return legacy + views
        cb_src = ttk.Combobox(sf, textvariable=comb_src_var,
                              values=_src_options(), width=20,
                              state="readonly", font=("Arial", 8))
        cb_src.pack(anchor="w", padx=3, pady=1)
        # Refresh option list each time the dropdown is opened so newly
        # created user-pipeline outputs appear without restarting.
        cb_src.bind("<Button-1>",
                    lambda e, c=cb_src: c.configure(values=_src_options()))
        tk.Label(sf, text="op(src) ⊕ running   (src may be any view)",
                 font=("Arial", 6), fg="#888888").pack(anchor="w", padx=3)

    # ------------------------------------------------------------------
    # Add / move / remove
    # ------------------------------------------------------------------
    def _on_add_step(self, parent_frame, pipeline_list):
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
        self._rebuild_pipeline_ui(parent_frame, pipeline_list)
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _on_move_step(self, parent_frame, pipeline_list, idx, direction):
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(pipeline_list):
            return
        pipeline_list[idx], pipeline_list[new_idx] = \
            pipeline_list[new_idx], pipeline_list[idx]
        self._rebuild_pipeline_ui(parent_frame, pipeline_list)
        self._rebuild_all_masks_if_open()
        self._refresh()

    def _on_remove_step(self, parent_frame, pipeline_list, idx):
        if not pipeline_list:
            return
        del pipeline_list[idx]
        self._rebuild_pipeline_ui(parent_frame, pipeline_list)
        self._rebuild_all_masks_if_open()
        self._refresh()
