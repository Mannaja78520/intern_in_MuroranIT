"""Mixin: All-Masks flowchart window + Zoom window.

The All-Masks window lays out every intermediate mask in the RGB and IR
pipelines as a flowchart of thumbnails. Clicking any thumbnail opens a
640x480 zoom window that updates live.

Host-class requirements:
    self.root, self.rgb_pipeline, self.ir_pipeline, self.btn_all_masks
    self.all_masks_win, self.all_masks_labels, self.all_masks_step_labels
    self._zoom_win, self._zoom_label, self._zoom_view
    self._last_f_rgb, self._refresh()
"""
import tkinter as tk
from tkinter import ttk

from config import FC_W, FC_H, STEPS_PER_ROW, STEP_ROW_BASE, YOLO_COL


class FlowchartMixin:
    """All-Masks flowchart + zoom window methods."""

    # ------------------------------------------------------------------
    # All-Masks window: open / close / rebuild
    # ------------------------------------------------------------------
    def _toggle_all_masks_window(self):
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            self.all_masks_win.destroy()
            self.all_masks_win         = None
            self.all_masks_labels      = {}
            self.all_masks_step_labels = {}
            self.btn_all_masks.config(relief="raised")
        else:
            self._open_all_masks_window()
            self.btn_all_masks.config(relief="sunken")

    def _rebuild_all_masks_if_open(self):
        """Reopen the All Masks window in-place when pipeline steps change."""
        if not (self.all_masks_win and self.all_masks_win.winfo_exists()):
            return
        try:
            x, y = self.all_masks_win.winfo_x(), self.all_masks_win.winfo_y()
        except Exception:
            x = y = 100
        self.all_masks_win.destroy()
        self.all_masks_win         = None
        self.all_masks_labels      = {}
        self.all_masks_step_labels = {}
        self._open_all_masks_window()
        self.all_masks_win.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # All-Masks window: build
    # ------------------------------------------------------------------
    def _open_all_masks_window(self):
        # -- Per-section visibility map (lazy seed for new branches) --
        vis_map = self.all_masks_section_visibility
        vis_map.setdefault("rgb", tk.BooleanVar(value=True))
        vis_map.setdefault("ir",  tk.BooleanVar(value=True))
        for _up in self.user_pipelines:
            _up_nm = _up["name"].get().strip() or "branch"
            vis_map.setdefault(f"up_{_up_nm}",
                               tk.BooleanVar(value=True))

        show_rgb = vis_map["rgb"].get()
        show_ir  = vis_map["ir"].get()

        # -- Layout constants ------------------------------------------
        CW  = FC_W + 8              # node frame width (extra room
                                     # so the bumped 9-pt labels never
                                     # truncate inside the frame)
        CH  = FC_H + 30             # node frame height
        HS  = CW + 48               # horizontal step
        VS  = CH + 48               # vertical step
        MX, MY   = 24, 56          # MY enlarged: leaves clear band above the
                                   # header for the YOLO bypass line.
        SECT_H   = 34
        SECT_GAP = 80              # extra gap between sections so the IR
                                   # YOLO bypass has room above its header.
        YOLO_BAND = 18             # vertical pixels reserved above each
                                   # header for the YOLO routing line.
        self._all_masks_img_size = (FC_W, FC_H)

        n_rgb = len(self.rgb_pipeline)
        n_ir  = len(self.ir_pipeline)
        # PM (pre-morph) step counts. Each PM step gets its own
        # thumbnail tile inserted between the input-section row and
        # the morph-step rows so the user can see the image-stage
        # progression. 0 PM steps -> original compact layout.
        # PM thumbnails start at col 3 (just past HSV colour / Blur),
        # leaving the first three columns free of overlap, so each
        # PM row fits (STEPS_PER_ROW - 3) tiles.
        PM_START_COL = 3
        PM_ROW_WIDTH = STEPS_PER_ROW - PM_START_COL
        n_rgb_pre = len(getattr(self, "rgb_pre_pipeline", []) or [])
        n_ir_pre  = len(getattr(self, "ir_pre_pipeline",  []) or [])
        _rgb_pm_rows = ((n_rgb_pre + PM_ROW_WIDTH - 1) // PM_ROW_WIDTH
                        if n_rgb_pre else 0)
        _ir_pm_rows  = ((n_ir_pre  + PM_ROW_WIDTH - 1) // PM_ROW_WIDTH
                        if n_ir_pre  else 0)

        # OVERLAY-aware step positioning. An OVERLAY step occupies 3
        # grid columns visually (since its composite thumbnail is 3x
        # FC_W wide), so the next step in the same row must start 3
        # cells later. Returns a list of (col, row) per step index.
        def _overlay_w_for(step_tuple):
            """Number of grid COLUMNS occupied by an OVERLAY step's
            wide composite. ceil(node_w / HS) plus a safety buffer so
            even the 3-tile composite never near-touches the next
            step's edge."""
            try:
                if step_tuple[7].get() and step_tuple[8].get() == "OVERLAY":
                    state = getattr(self, "_overlay_state", {}).get(
                        id(step_tuple))
                    n = 1
                    if state is not None:
                        if state["mask2_src"].get() not in ("", "none"):
                            n += 1
                        if state["base_src"].get() not in ("", "none"):
                            n += 1
                    count = n + 1
                    OV_GLYPH = 64
                    # +24 buffer: keeps a comfortable visual gap before
                    # the next step's column so the composite never
                    # appears to nudge into it.
                    node_w = count * FC_W + (count - 1) * OV_GLYPH + 8 + 24
                    import math
                    return max(1, math.ceil(node_w / HS))
            except Exception:
                pass
            return 1

        def _step_positions(pipeline_steps, base_row):
            positions = []
            col = 0
            row = base_row
            for j, st in enumerate(pipeline_steps):
                w = _overlay_w_for(st)
                if col + w > STEPS_PER_ROW:
                    col = 0
                    row += 1
                positions.append((col, row))
                col += w
                if col >= STEPS_PER_ROW:
                    col = 0
                    row += 1
            return positions

        def _last_row_after(pipeline_steps, base_row):
            """The row INDEX where post-pipeline nodes should sit."""
            poss = _step_positions(pipeline_steps, base_row)
            return (poss[-1][1] + 1) if poss else base_row

        # Morph-step row bases shift down by the PM-row count so the
        # new PM thumbnail row(s) slot in just above the morph rows.
        # RGB step rows start one row below the HSV/m1/.../mask_pre
        # band (which now lives on `_rgb_after_pm` since the separate
        # cmap row was merged into the channel tiles).
        _rgb_step_base = (STEP_ROW_BASE - 1) + _rgb_pm_rows
        _ir_step_base  = 1 + _ir_pm_rows
        rgb_positions = _step_positions(self.rgb_pipeline, _rgb_step_base)
        ir_positions  = _step_positions(self.ir_pipeline,  _ir_step_base)
        rgb_out = _last_row_after(self.rgb_pipeline, _rgb_step_base)
        ir_out  = _last_row_after(self.ir_pipeline,  _ir_step_base)

        # PM thumbnail positions: a dedicated band of rows starting
        # right after row 0 (the input section). PM tiles begin at
        # PM_START_COL so the first three columns (under rgb_raw /
        # rgb_blur / rgb_hsv_full) stay open. rgb_pre_step1 at
        # (3, 1), rgb_pre_step2 at (4, 1), ... wrapping when col 7
        # would be reached.
        def _pm_positions(count):
            return [(PM_START_COL + i % PM_ROW_WIDTH,
                     1 + i // PM_ROW_WIDTH)
                    for i in range(count)]
        rgb_pm_positions = _pm_positions(n_rgb_pre)
        ir_pm_positions  = _pm_positions(n_ir_pre)
        # First post-PM row (used to shift existing input-section
        # rows: row 1 -> row 1+pm_rows, row 2 -> row 2+pm_rows).
        _rgb_after_pm = 1 + _rgb_pm_rows
        _ir_after_pm  = 1 + _ir_pm_rows

        # -- HSV pair-tile metadata -----------------------------------
        # Every "_pair" node stacks TWO independent full-size
        # thumbnails (top + bottom). pair_meta maps a node id to
        # (top_view, bottom_view, top_label, bottom_label) so the
        # renderer can wire + caption each half. Main RGB pairs stack
        # the channel grayscale over its colormap; branch pairs (added
        # in the user-pipeline loop below) stack the channel over its
        # detection mask.
        pair_meta = {
            "rgb_hsv_H_pair": ("rgb_hsv_H", "rgb_hsv_H_cmap",
                               "Hue channel", "Hue colormap"),
            "rgb_hsv_S_pair": ("rgb_hsv_S", "rgb_hsv_S_cmap",
                               "Sat channel", "Sat colormap"),
            "rgb_hsv_V_pair": ("rgb_hsv_V", "rgb_hsv_V_cmap",
                               "Val channel", "Val colormap"),
        }

        # -- Detection-mode-aware node titles -------------------------
        # The m1 / m2 / detection-mask node titles change with the RGB
        # detection mode so the flowchart shows which method is live.
        _dk = self._rgb_detect_key()
        if _dk == "achroma":
            _m1_t, _m2_t, _dm_t = (" 7  S+V mask", " 8  (unused)",
                                   " 9  Achromatic mask")
        elif _dk == "value":
            _m1_t, _m2_t, _dm_t = (" 7  V mask", " 8  (unused)",
                                   " 9  Brightness mask")
        elif _dk == "lab":
            _m1_t, _m2_t, _dm_t = (" 7  LAB L/a*/b* mask", " 8  (unused)",
                                   " 9  LAB mask")
        elif _dk == "hsv_lab":
            _m1_t, _m2_t, _dm_t = (" 7  Hue mask", " 8  LAB L/a*/b* mask",
                                   " 9  HSV & LAB mask")
        else:
            _m1_t, _m2_t, _dm_t = (" 7  Hue mask 1", " 8  Hue mask 2",
                                   " 9  HSV mask")

        # -- Node lists ------------------------------------------------
        # HSV per-channel tiles are PAIRED with their colormap, so each
        # tile shows the channel grayscale on top and the colormap on
        # the bottom. That collapses the old H / S / V + H_cmap / S_cmap
        # / V_cmap row pair into 3 tiles on row 0.
        rgb_nodes = [
            ("rgb_raw",        0, 0, " 1  Input"),
            ("rgb_blur",       1, 0, " 2  Blur*"),
            ("rgb_hsv_full",   2, 0, " 3  HSV colour"),
            ("rgb_hsv_H_pair", 3, 0, " 4  Hue + cmap"),
            ("rgb_hsv_S_pair", 4, 0, " 5  Sat + cmap"),
            ("rgb_hsv_V_pair", 5, 0, " 6  Val + cmap"),
            ("rgb_hsv_raw",    6, 0, " 3r HSV raw"),
            *[(f"rgb_pre_step{i+1}",
               rgb_pm_positions[i][0],
               rgb_pm_positions[i][1],
               f"PM{i+1}: ?")
              for i in range(n_rgb_pre)],
            ("rgb_m1",        2, _rgb_after_pm, _m1_t),
            ("rgb_m2",        3, _rgb_after_pm, _m2_t),
            ("rgb_hsv_mask",  4, _rgb_after_pm, _dm_t),
            ("rgb_bgsub",     5, _rgb_after_pm, "10  BG sub"),
            ("rgb_mask_pre",  6, _rgb_after_pm, "11  Pre-pipeline"),
            *[(f"rgb_step{i+1}",
               rgb_positions[i][0],
               rgb_positions[i][1],
               f"S{i+1}: ?")
              for i in range(n_rgb)],
            ("rgb_mask",      0, rgb_out, "12  Post-pipeline"),
            ("rgb_post_blur", 1, rgb_out, "13  Post-blur*"),
            ("_yolo_rgb",     YOLO_COL, 0,       "YOLO"),
            ("rgb_det",       YOLO_COL, rgb_out, "14  Detection"),
        ]

        ir_nodes = [
            ("ir_raw",        0, 0, "15  IR input"),
            ("ir_gray",       1, 0, "16  Grayscale"),
            ("ir_blur",       2, 0, "17  Blur*"),
            ("ir_clahe",      3, 0, "18  CLAHE"),
            ("ir_bgsub",      4, 0, "19  BG sub"),
            ("ir_thresh",     5, 0, "20  Threshold"),
            ("ir_mask_pre",   6, 0, "21  Pre-pipeline"),
            *[(f"ir_pre_step{i+1}",
               ir_pm_positions[i][0],
               ir_pm_positions[i][1],
               f"PM{i+1}: ?")
              for i in range(n_ir_pre)],
            *[(f"ir_step{i+1}",
               ir_positions[i][0],
               ir_positions[i][1],
               f"S{i+1}: ?")
              for i in range(n_ir)],
            ("ir_mask",       0, ir_out, "22  Post-pipeline"),
            ("ir_post_blur",  1, ir_out, "23  Post-blur*"),
            ("_yolo_ir",      YOLO_COL, 0,      "YOLO"),
            ("ir_det",        YOLO_COL, ir_out, "24  Detection"),
        ]

        # -- Edges -----------------------------------------------------
        def _step_edges(prefix, n):
            edges = []
            pre = f"{prefix}_mask_pre"
            if n > 0:
                edges.append((pre, f"{prefix}_step1"))
                for i in range(n - 1):
                    edges.append((f"{prefix}_step{i+1}", f"{prefix}_step{i+2}"))
                edges.append((f"{prefix}_step{n}", f"{prefix}_mask"))
            else:
                edges.append((pre, f"{prefix}_mask"))
            return edges

        def _pm_edges(prefix, n, src_node, dst_node):
            """Chain src -> pre_step1 -> ... -> pre_stepN -> dst.
            When n == 0, falls through to a direct src -> dst edge so
            the surrounding flow stays connected."""
            if n == 0:
                return [(src_node, dst_node)]
            edges = [(src_node, f"{prefix}_pre_step1")]
            for i in range(n - 1):
                edges.append((f"{prefix}_pre_step{i+1}",
                              f"{prefix}_pre_step{i+2}"))
            edges.append((f"{prefix}_pre_step{n}", dst_node))
            return edges

        rgb_edges = [
            ("rgb_raw",      "rgb_blur"),
            *_pm_edges("rgb", n_rgb_pre, "rgb_blur", "rgb_hsv_full"),
            ("rgb_hsv_full", "rgb_hsv_raw"),
            ("rgb_hsv_full", "rgb_hsv_H_pair"),
            ("rgb_hsv_H_pair", "rgb_hsv_S_pair"),
            ("rgb_hsv_S_pair", "rgb_hsv_V_pair"),
            ("rgb_hsv_full", "rgb_m1"),
            ("rgb_hsv_full", "rgb_m2"),
            ("rgb_m1",       "rgb_m2",         "OR"),
            ("rgb_m2",       "rgb_hsv_mask"),
            ("rgb_hsv_mask", "rgb_bgsub", "AND"),
            ("rgb_bgsub",    "rgb_mask_pre"),
            *_step_edges("rgb", n_rgb),
            ("rgb_mask",     "rgb_post_blur"),
            ("rgb_post_blur","rgb_det"),
            ("rgb_raw",      "_yolo_rgb"),
            ("_yolo_rgb",    "rgb_det"),
        ]

        ir_edges = [
            ("ir_raw",       "ir_gray"),
            ("ir_gray",      "ir_blur"),
            ("ir_blur",      "ir_clahe"),
            ("ir_clahe",     "ir_bgsub"),
            *_pm_edges("ir", n_ir_pre, "ir_bgsub", "ir_thresh"),
            ("ir_thresh",    "ir_mask_pre"),
            *_step_edges("ir", n_ir),
            ("ir_mask",      "ir_post_blur"),
            ("ir_post_blur", "ir_det"),
            ("ir_raw",       "_yolo_ir"),
            ("_yolo_ir",     "ir_det"),
        ]

        # Combine edges: arrows from a source mask -> step when combine ON
        def _combine_edges(prefix, pipeline):
            cedges = []
            for i, step in enumerate(pipeline):
                _en, _op, _n, _d, _kx, _ky, _t, _cen, _cop, _csr = step
                if not _cen.get():
                    continue
                src = _csr.get()
                if src == "mask_pre":
                    src_node = f"{prefix}_mask_pre"
                elif src == "prev":
                    src_node = f"{prefix}_step{i}" if i > 0 else f"{prefix}_mask_pre"
                elif src.startswith("step_"):
                    try:
                        si = int(src.split("_")[1])
                        src_node = f"{prefix}_step{si}"
                    except (ValueError, IndexError):
                        src_node = f"{prefix}_mask_pre"
                else:
                    src_node = f"{prefix}_mask_pre"
                cedges.append((src_node, f"{prefix}_step{i+1}",
                               f"+{_cop.get()}", "combine"))
            return cedges

        rgb_edges += _combine_edges("rgb", self.rgb_pipeline)
        ir_edges  += _combine_edges("ir",  self.ir_pipeline)

        combine_active = set()
        for _i, _s in enumerate(self.rgb_pipeline):
            if _s[7].get():
                combine_active.add(f"rgb_step{_i+1}")
        for _i, _s in enumerate(self.ir_pipeline):
            if _s[7].get():
                combine_active.add(f"ir_step{_i+1}")
        # Also mark combine-active user-pipeline steps.
        user_pipes = list(getattr(self, "user_pipelines", []))
        for up in user_pipes:
            _nm = up["name"].get().strip() or "branch"
            for _i, _s in enumerate(up["steps"]):
                if _s[7].get():
                    combine_active.add(f"up_{_nm}_step{_i+1}")

        # -- Build per-user-pipeline section descriptors --------------
        # Each entry: (nodes, edges, title, hdr_bg, node_bg, n_steps, color)
        up_sections = []
        for up in user_pipes:
            _nm  = up["name"].get().strip() or "branch"
            _src = up["source"].get()
            _n   = len(up["steps"])
            _input_id   = f"up_{_nm}__input"
            _output_id  = f"up_{_nm}"

            # -- Channel-origin & per-channel-mask thumbnails ------
            # Layout per branch — EVERYTHING sits on row 0 alongside
            # the source tiles:
            #   Row 0 : input | src raw | [src blur] | [src bg-sub]
            #           | H pair | S pair | V pair | [BG-sub FG] | det
            # Each "pair" tile stacks the channel grayscale (top) over
            # its detection mask (bottom) — same paired-thumbnail
            # layout the main RGB pipeline uses for channel + colormap.
            # IR branches get a single IR-frame / IR-mask pair.
            _info_nodes = []
            _info_max_row = 0
            _info_max_col = 0
            try:
                _btype = up.get("type").get() if up.get("type") else ""
                _ch_mode = (up.get("channels").get()
                            if up.get("channels") else "")
                _show_bg = (up.get("use_bgsub")
                            and up["use_bgsub"].get())
                # Branch pre-blur is now driven by the PM step list,
                # not a single knob. Count enabled PM steps so the
                # info-row still shows a "src blur" thumbnail when
                # the branch has any image-stage processing.
                _pre_n = len(up.get("pre_steps") or [])
                _pre_k = _pre_n   # alias kept for the conditional below

                # row-0 cursor (source pipeline + channel pairs +
                # bgsub_fg + det). Row 0 col 0 holds the `_input_id`,
                # so the source tiles start at col 1.
                _ic_state = {"col": 1, "max_row": 0}

                def _push_info_at(suffix, label, col, row):
                    _info_nodes.append(
                        (f"_info_{_nm}_{suffix}", col, row, label))
                    if row > _ic_state["max_row"]:
                        _ic_state["max_row"] = row

                def _push_row0(suffix, label):
                    _push_info_at(suffix, label, _ic_state["col"], 0)
                    _ic_state["col"] += 1

                # Source-image pipeline along row 0.
                if _btype in ("rgb", "ir"):
                    _push_row0("src_raw", "src raw")
                    if _pre_k > 0:
                        _push_row0("src_blur",  "src blur")
                    if _show_bg:
                        _push_row0("src_bgsub", "src bg-sub")

                # Per-channel detection pairs — channel grayscale +
                # detection mask stacked into ONE tile, all on row 0.
                if _btype == "rgb":
                    for ch in ("H", "S", "V"):
                        if ch in _ch_mode or _ch_mode == "full":
                            _lc   = ch.lower()
                            _pvid = f"{_lc}_pair"
                            _push_row0(_pvid, f"{ch} channel / {ch} det")
                            _mask_view = (f"up_{_nm}_h1_mask"
                                          if ch == "H"
                                          else f"up_{_nm}_{_lc}_mask")
                            pair_meta[f"_info_{_nm}_{_pvid}"] = (
                                f"up_{_nm}_hsv_{ch}", _mask_view,
                                f"{ch} channel (PM)", f"{ch} mask (det)")
                elif _btype == "ir":
                    _push_row0("ir_pair", "IR frame / IR det")
                    pair_meta[f"_info_{_nm}_ir_pair"] = (
                        "ir_raw", f"up_{_nm}_ir_mask",
                        "IR frame", "IR mask (det)")

                # BG-sub FG + combined detection thumbnail go on row 0
                # after the channel pairs.
                if _show_bg:
                    _push_row0("bgsub", "BG-sub FG")
                _push_row0("det", "branch_det")
                _info_max_row = _ic_state["max_row"]
                _info_max_col = _ic_state["col"] - 1
            except Exception:
                _info_nodes = []
                _info_max_row = 0
                _info_max_col = 0

            # Branch PM step thumbnails: one dedicated row band right
            # after the info nodes and before the morph step rows.
            _branch_pre_n = len(up.get("pre_steps") or [])
            _branch_pm_rows = ((_branch_pre_n + STEPS_PER_ROW - 1)
                               // STEPS_PER_ROW) if _branch_pre_n else 0
            _branch_pm_base = _info_max_row + 1
            up_pm_nodes = [
                (f"up_{_nm}_pre_step{i+1}",
                 i % STEPS_PER_ROW,
                 _branch_pm_base + i // STEPS_PER_ROW,
                 f"PM{i+1}: ?")
                for i in range(_branch_pre_n)
            ]

            # Step nodes start one row AFTER the last PM row (or the
            # last info row when the branch has no PM steps).
            _step_base = _branch_pm_base + _branch_pm_rows
            up_positions = _step_positions(up["steps"], _step_base)
            _out_row_up  = ((up_positions[-1][1] + 1)
                            if up_positions else _step_base)

            nodes_up = [
                (_input_id, 0, 0, f"src <- {_src}"),
                *_info_nodes,
                *up_pm_nodes,
                *[(f"up_{_nm}_step{i+1}",
                   up_positions[i][0],
                   up_positions[i][1],
                   f"S{i+1}: ?")
                  for i in range(_n)],
                # Output sits bottom-right (col YOLO_COL) like the
                # main RGB/IR pipelines' Detection node.
                (_output_id, YOLO_COL, _out_row_up,
                 f"out -> up_{_nm}"),
            ]
            edges_up = []
            # PM (Pre-morph) stage runs on the branch's SOURCE IMAGE
            # — NOT on the detection mask — so chain it from the
            # `src raw` tile into the `branch_det` tile. The modified
            # source image is what feeds HSV / IR detection. When
            # those info tiles are absent (custom branches) fall back
            # to the morph input so the flow stays connected.
            _have_src_raw = any(n[0] == f"_info_{_nm}_src_raw"
                                for n in _info_nodes)
            _have_det     = any(n[0] == f"_info_{_nm}_det"
                                for n in _info_nodes)
            _pm_src_node = (f"_info_{_nm}_src_raw"
                            if _have_src_raw else _input_id)
            _pm_dst_node = (f"_info_{_nm}_det"
                            if _have_det else _input_id)
            if _branch_pre_n > 0:
                edges_up.append((_pm_src_node, f"up_{_nm}_pre_step1"))
                for i in range(_branch_pre_n - 1):
                    edges_up.append((f"up_{_nm}_pre_step{i+1}",
                                     f"up_{_nm}_pre_step{i+2}"))
                edges_up.append((f"up_{_nm}_pre_step{_branch_pre_n}",
                                 _pm_dst_node))
            # Morph chain: runs on the branch's SOURCE view (the
            # detection mask by default), independent of the PM
            # image stage above.
            if _n > 0:
                edges_up.append((_input_id, f"up_{_nm}_step1"))
                for i in range(_n - 1):
                    edges_up.append((f"up_{_nm}_step{i+1}",
                                     f"up_{_nm}_step{i+2}"))
                edges_up.append((f"up_{_nm}_step{_n}", _output_id))
            else:
                edges_up.append((_input_id, _output_id))
            # Combine edges within the user pipeline (in-pipeline shortcuts).
            for i, st in enumerate(up["steps"]):
                _en, _op, _, _, _, _, _, _cen, _cop, _csr = st
                if not _cen.get():
                    continue
                src_str = _csr.get()
                if src_str == "mask_pre":
                    src_node = _input_id
                elif src_str == "prev":
                    src_node = (f"up_{_nm}_step{i}"
                                if i > 0 else _input_id)
                elif src_str.startswith("step_"):
                    try:
                        si = int(src_str.split("_")[1])
                        src_node = f"up_{_nm}_step{si}"
                    except (ValueError, IndexError):
                        src_node = _input_id
                else:
                    # Full view-name reference (cross-section). Drawn as
                    # an annotation on the step rather than an arrow.
                    continue
                edges_up.append((src_node, f"up_{_nm}_step{i+1}",
                                 f"+{_cop.get()}", "combine"))
            up_sections.append({
                "name":  _nm,
                "nodes": nodes_up,
                "edges": edges_up,
                "out_row": _out_row_up,
                "title": f"---------  Pipeline: {_nm}  (src <- {_src})  ---------",
                "hdr_bg":  "#5c2a7a",   # purple for user pipelines
                "node_bg": "#1f1130",
            })

        # -- Canvas sizing ---------------------------------------------
        # Combine-active rows are taller; OVERLAY-active rows are taller
        # still. Count both per-section so the next row never overlaps
        # the bottom of any node above it.
        comb_extra = (FC_H // 3) + 70

        def _is_overlay_step_q(vid):
            try:
                if vid.startswith("rgb_step"):
                    _i = int(vid[len("rgb_step"):]) - 1
                    return (0 <= _i < len(self.rgb_pipeline)
                            and self.rgb_pipeline[_i][7].get()
                            and self.rgb_pipeline[_i][8].get() == "OVERLAY")
                if vid.startswith("ir_step"):
                    _i = int(vid[len("ir_step"):]) - 1
                    return (0 <= _i < len(self.ir_pipeline)
                            and self.ir_pipeline[_i][7].get()
                            and self.ir_pipeline[_i][8].get() == "OVERLAY")
                if vid.startswith("up_") and "_step" in vid:
                    _bare = vid[len("up_"):]
                    _nmx, _, _st = _bare.rpartition("_step")
                    _ix = int(_st) - 1
                    for _up in getattr(self, "user_pipelines", []):
                        if _up["name"].get().strip() == _nmx \
                           and 0 <= _ix < len(_up["steps"]):
                            return (_up["steps"][_ix][7].get()
                                    and _up["steps"][_ix][8].get() == "OVERLAY")
            except Exception:
                pass
            return False

        def _section_height(nodes, out_row):
            """Total y-extent of a section. AND/OR/XOR combine rows
            and YOLO-enabled rows each add their own extra height."""
            comb_rows = {r for vid, _, r, _ in nodes
                         if (combine_active and vid in combine_active
                             and not _is_overlay_step_q(vid))}
            yolo_rows = set()
            for vid, _, r, _ in nodes:
                try:
                    step = None
                    if vid.startswith("rgb_step"):
                        _i = int(vid[len("rgb_step"):]) - 1
                        if 0 <= _i < len(self.rgb_pipeline):
                            step = self.rgb_pipeline[_i]
                    elif vid.startswith("ir_step"):
                        _i = int(vid[len("ir_step"):]) - 1
                        if 0 <= _i < len(self.ir_pipeline):
                            step = self.ir_pipeline[_i]
                    elif vid.startswith("up_") and "_step" in vid:
                        _bare = vid[len("up_"):]
                        _nm, _, _st = _bare.rpartition("_step")
                        _ix = int(_st) - 1
                        for up in getattr(self, "user_pipelines", []):
                            if (up["name"].get().strip() == _nm
                                    and 0 <= _ix < len(up["steps"])):
                                step = up["steps"][_ix]
                                break
                    if step is not None:
                        yst = (getattr(self, "_yolo_state", {}) or {}
                               ).get(id(step))
                        if yst and yst["yolo_en"].get():
                            yolo_rows.add(r)
                except Exception:
                    pass
            yolo_extra = 3 * (FC_H + 18)
            # HSV pair tiles stack two full-size images plus a caption
            # above each, so any row with one needs an extra FC_H of
            # vertical room plus space for the two half-captions.
            pair_rows = {r for vid, _, r, _ in nodes
                         if vid.endswith("_pair")}
            pair_extra = FC_H + 38
            return ((out_row + 1) * VS
                    + len(comb_rows) * comb_extra
                    + len(yolo_rows) * yolo_extra
                    + len(pair_rows) * pair_extra)

        # Canvas width tracks the widest section. Branch sections now
        # put the source tiles + every channel pair + det on row 0, so
        # they can run past the YOLO_COL the RGB/IR pipelines use.
        _max_col = YOLO_COL
        for _nd in rgb_nodes + ir_nodes:
            _max_col = max(_max_col, _nd[1])
        for _sec in up_sections:
            for _nd in _sec["nodes"]:
                _max_col = max(_max_col, _nd[1])
        canvas_w = MX + (_max_col + 1) * HS + MX
        # Visibility-aware y-cursor: sections whose checkbox is off
        # are skipped entirely (zero height contribution).
        _cursor_y = MY + SECT_H
        rgb_y0 = _cursor_y
        if show_rgb:
            _cursor_y += _section_height(rgb_nodes, rgb_out) \
                          + SECT_GAP + SECT_H
        ir_y0 = _cursor_y
        if show_ir:
            _cursor_y += _section_height(ir_nodes, ir_out) \
                          + SECT_GAP + SECT_H
        up_y0s = []
        up_visible = []
        for sec in up_sections:
            _show = vis_map.get(f"up_{sec['name']}",
                                tk.BooleanVar(value=True)).get()
            up_visible.append(_show)
            up_y0s.append(_cursor_y)
            if _show:
                _cursor_y += _section_height(sec["nodes"], sec["out_row"]) \
                              + SECT_GAP + SECT_H
        # When NO section is visible, leave a small placeholder area
        # so the canvas isn't zero-height.
        if _cursor_y <= MY + SECT_H:
            _cursor_y = MY + SECT_H + 60
        canvas_h = _cursor_y + MY

        # -- Toplevel + scrollable canvas ------------------------------
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            self.all_masks_win.destroy()

        win = tk.Toplevel(self.root)
        win.title("Pipeline Flow - All Masks   (* = blur position selectable)")
        win.protocol("WM_DELETE_WINDOW", self._toggle_all_masks_window)
        self.all_masks_win         = win
        self.all_masks_labels      = {}
        self.all_masks_step_labels = {}

        # -- Toolbar: per-pipeline visibility checkboxes -------------
        # Lets the user pick which pipelines to render. Each toggle
        # rebuilds the window so hidden sections reclaim their space.
        toolbar = tk.Frame(win, bg="#1a1a2a")
        toolbar.pack(fill="x", side="top")
        tk.Label(toolbar, text="Show:",
                 font=("DejaVu Sans", 9, "bold"),
                 bg="#1a1a2a", fg="#dddddd"
                 ).pack(side="left", padx=(6, 4), pady=4)

        def _on_section_toggle(*_a):
            self._rebuild_all_masks_if_open()

        def _add_toggle(parent, label, var, fg):
            cb = tk.Checkbutton(parent, text=label, variable=var,
                                font=("DejaVu Sans", 9, "bold"),
                                bg="#1a1a2a", fg=fg,
                                selectcolor="#1a1a2a",
                                activebackground="#1a1a2a",
                                activeforeground=fg,
                                command=_on_section_toggle)
            cb.pack(side="left", padx=4)

        _add_toggle(toolbar, "RGB", vis_map["rgb"], "#88bbff")
        _add_toggle(toolbar, "IR",  vis_map["ir"],  "#88dd88")
        for _up in self.user_pipelines:
            _up_nm = _up["name"].get().strip() or "branch"
            _key = f"up_{_up_nm}"
            if _key not in vis_map:
                vis_map[_key] = tk.BooleanVar(value=True)
            _add_toggle(toolbar, _up_nm, vis_map[_key], "#cc88ff")
        # "All / None" quick buttons.
        def _set_all(value):
            for _v in vis_map.values():
                _v.set(value)
            self._rebuild_all_masks_if_open()
        tk.Button(toolbar, text="All", font=("DejaVu Sans", 8),
                  command=lambda: _set_all(True)
                  ).pack(side="right", padx=4, pady=2)
        tk.Button(toolbar, text="None", font=("DejaVu Sans", 8),
                  command=lambda: _set_all(False)
                  ).pack(side="right", padx=2, pady=2)

        container = tk.Frame(win)
        container.pack(fill="both", expand=True)
        fc = tk.Canvas(container,
                       width=min(canvas_w, 1400), height=min(canvas_h, 820),
                       scrollregion=(0, 0, canvas_w, canvas_h),
                       highlightthickness=0, bg="#111111")
        v_sb = ttk.Scrollbar(container, orient="vertical",   command=fc.yview)
        h_sb = ttk.Scrollbar(container, orient="horizontal", command=fc.xview)
        fc.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        h_sb.pack(side="bottom", fill="x")
        v_sb.pack(side="right",  fill="y")
        fc.pack(side="left", fill="both", expand=True)
        fc.bind("<Button-4>",   lambda e: fc.yview_scroll(-1, "units"))
        fc.bind("<Button-5>",   lambda e: fc.yview_scroll(1,  "units"))
        fc.bind("<MouseWheel>", lambda e: fc.yview_scroll(int(-1*e.delta/120), "units"))

        # -- Drawing helpers -------------------------------------------
        def _draw_edge_label(canvas, x, y, text):
            clr = ("#44ff44" if text == "OR"
                   else "#ff7744" if text == "AND"
                   else "#bb66ff" if text.startswith("+")
                   else "#ffff44")
            canvas.create_oval(x - 20, y - 10, x + 20, y + 10,
                               fill="#222222", outline=clr, width=2)
            canvas.create_text(x, y, text=text, fill=clr,
                               font=("DejaVu Sans", 8, "bold"))

        # Per-section row_y mapping. Filled by draw_section before any
        # px/right/left/top/bottom helper is called.
        section_row_y = {}

        def px(col, row, y0):
            ry = section_row_y.get(row, row * VS)
            return MX + col * HS, y0 + ry
        def right(c, r, y0):   x, y = px(c, r, y0); return x + CW, y + CH // 2
        def left(c, r, y0):    x, y = px(c, r, y0); return x,      y + CH // 2
        def bottom(c, r, y0):  x, y = px(c, r, y0); return x + CW // 2, y + CH
        def top(c, r, y0):     x, y = px(c, r, y0); return x + CW // 2, y

        def _is_overlay_step(vid):
            """True if this step's combine op is OVERLAY (so the node
            needs an extra full-size overlay thumbnail)."""
            try:
                if vid.startswith("rgb_step"):
                    _i = int(vid[len("rgb_step"):]) - 1
                    return (0 <= _i < len(self.rgb_pipeline)
                            and self.rgb_pipeline[_i][7].get()
                            and self.rgb_pipeline[_i][8].get() == "OVERLAY")
                if vid.startswith("ir_step"):
                    _i = int(vid[len("ir_step"):]) - 1
                    return (0 <= _i < len(self.ir_pipeline)
                            and self.ir_pipeline[_i][7].get()
                            and self.ir_pipeline[_i][8].get() == "OVERLAY")
                if vid.startswith("up_") and "_step" in vid:
                    _bare = vid[len("up_"):]
                    _nmx, _, _st = _bare.rpartition("_step")
                    _ix = int(_st) - 1
                    for _up in getattr(self, "user_pipelines", []):
                        if _up["name"].get().strip() == _nmx \
                           and 0 <= _ix < len(_up["steps"]):
                            return (_up["steps"][_ix][7].get()
                                    and _up["steps"][_ix][8].get() == "OVERLAY")
            except Exception:
                pass
            return False

        def _has_yolo_for_vid(vid):
            """True if the step at `vid` has YOLO add-on enabled."""
            try:
                step = None
                if vid.startswith("rgb_step"):
                    _i = int(vid[len("rgb_step"):]) - 1
                    if 0 <= _i < len(self.rgb_pipeline):
                        step = self.rgb_pipeline[_i]
                elif vid.startswith("ir_step"):
                    _i = int(vid[len("ir_step"):]) - 1
                    if 0 <= _i < len(self.ir_pipeline):
                        step = self.ir_pipeline[_i]
                elif vid.startswith("up_") and "_step" in vid:
                    _bare = vid[len("up_"):]
                    _nm, _, _st = _bare.rpartition("_step")
                    _ix = int(_st) - 1
                    for up in getattr(self, "user_pipelines", []):
                        if (up["name"].get().strip() == _nm
                                and 0 <= _ix < len(up["steps"])):
                            step = up["steps"][_ix]
                            break
                if step is not None:
                    yst = (getattr(self, "_yolo_state", {}) or {}
                           ).get(id(step))
                    return bool(yst and yst["yolo_en"].get())
            except Exception:
                pass
            return False

        def _compute_row_y(nodes, comb_active):
            """Compute the y-offset (relative to y0) of each row,
            stretching rows that contain combine-active or YOLO-
            enabled step nodes so their taller frames don't overlap
            the next row. OVERLAY nodes are normal-height (just
            wider) — they only contribute the YOLO extra if YOLO is
            also on. HSV pair tiles add their own pair_extra so the
            stacked-image tile doesn't bleed into the next row."""
            comb_extra = (FC_H // 3) + 70
            yolo_extra = 3 * (FC_H + 18)
            pair_extra = FC_H + 38
            max_row = max((r for _, _, r, _ in nodes), default=0)
            comb_rows = {r for vid, _, r, _ in nodes
                         if (comb_active and vid in comb_active
                             and not _is_overlay_step(vid))}
            yolo_rows = {r for vid, _, r, _ in nodes
                         if _has_yolo_for_vid(vid)}
            pair_rows = {r for vid, _, r, _ in nodes
                         if vid.endswith("_pair")}
            row_y = {0: 0}
            cur = 0
            for r in range(1, max_row + 2):
                step_h = VS
                if (r - 1) in comb_rows:
                    step_h += comb_extra
                if (r - 1) in yolo_rows:
                    step_h += yolo_extra
                if (r - 1) in pair_rows:
                    step_h += pair_extra
                cur += step_h
                row_y[r] = cur
            return row_y

        def draw_section(nodes, edges, y0, title, hdr_bg, node_bg,
                         arr_clr, yolo_clr, comb_active=None):
            section_row_y.clear()
            section_row_y.update(_compute_row_y(nodes, comb_active))
            fc.create_rectangle(0, y0 - SECT_H, canvas_w, y0,
                                fill=hdr_bg, outline="")
            fc.create_text(canvas_w // 2, y0 - 6,
                           text=title, font=("DejaVu Sans", 13, "bold"),
                           fill="white", anchor="s")

            pos = {vid: (c, r) for vid, c, r, _ in nodes}

            for edge in edges:
                f_id, t_id = edge[0], edge[1]
                edge_lbl   = edge[2] if len(edge) > 2 else None
                edge_type  = edge[3] if len(edge) > 3 else None
                if f_id not in pos or t_id not in pos:
                    continue
                fc_c, fr = pos[f_id]
                tc_c, tr = pos[t_id]
                is_yolo    = f_id.startswith("_yolo") or t_id.startswith("_yolo")
                is_combine = edge_type == "combine"
                clr  = yolo_clr if is_yolo else "#bb66ff" if is_combine else arr_clr
                lw   = 3        if is_yolo else 2
                arrs = (10, 12, 4)

                if fc_c == tc_c and fr < tr:
                    fc.create_line(*bottom(fc_c, fr, y0), *top(tc_c, tr, y0),
                                   arrow=tk.LAST, width=lw, fill=clr,
                                   arrowshape=arrs)
                    if edge_lbl:
                        mx = bottom(fc_c, fr, y0)[0]
                        my = (bottom(fc_c, fr, y0)[1] + top(tc_c, tr, y0)[1]) // 2
                        _draw_edge_label(fc, mx, my, edge_lbl)
                elif fr == tr and is_yolo and fc_c != tc_c:
                    # Route YOLO connection above the section header so it
                    # never crosses the header band or the title text.
                    route_y = y0 - SECT_H - YOLO_BAND
                    sx, sy = top(fc_c, fr, y0)
                    ex, ey = top(tc_c, tr, y0)
                    fc.create_line(sx, sy, sx, route_y, ex, route_y, ex, ey,
                                   arrow=tk.LAST, width=lw, fill=clr,
                                   arrowshape=arrs)
                elif fr == tr:
                    fc.create_line(*right(fc_c, fr, y0), *left(tc_c, tr, y0),
                                   arrow=tk.LAST, width=lw, fill=clr,
                                   arrowshape=arrs)
                    if edge_lbl:
                        mx = (right(fc_c, fr, y0)[0] + left(tc_c, tr, y0)[0]) // 2
                        my = right(fc_c, fr, y0)[1]
                        _draw_edge_label(fc, mx, my, edge_lbl)
                elif fr > tr:
                    # Upward edge (e.g. PM step on row 1 -> rgb_hsv_full
                    # on row 0). Start at the TOP of the source so the
                    # arrow body doesn't run behind the source node.
                    bx, by = top(fc_c, fr, y0)
                    ex, ey = left(tc_c, tr, y0)
                    fc.create_line(bx, by, bx, ey, ex, ey,
                                   arrow=tk.LAST, width=lw, fill=clr,
                                   smooth=False, arrowshape=arrs)
                    if edge_lbl:
                        _draw_edge_label(fc, (bx + ex) // 2, ey, edge_lbl)
                else:
                    bx, by = bottom(fc_c, fr, y0)
                    ex, ey = left(tc_c, tr, y0)
                    fc.create_line(bx, by, bx, ey, ex, ey,
                                   arrow=tk.LAST, width=lw, fill=clr,
                                   smooth=False, arrowshape=arrs)
                    if edge_lbl:
                        _draw_edge_label(fc, (bx + ex) // 2, ey, edge_lbl)

            for vid, col, row, label in nodes:
                nx, ny = px(col, row, y0)
                if vid.startswith("_yolo"):
                    fc.create_rectangle(nx, ny, nx + CW, ny + CH,
                                        fill="#2a1500", outline=yolo_clr, width=2)
                    fc.create_text(nx + CW // 2, ny + CH // 2 - 7,
                                   text="YOLO", font=("DejaVu Sans", 11, "bold"),
                                   fill=yolo_clr)
                    fc.create_text(nx + CW // 2, ny + CH // 2 + 9,
                                   text=label, font=("DejaVu Sans", 9), fill="#aaaaaa")
                else:
                    is_step = (vid.startswith("rgb_step")
                               or vid.startswith("ir_step")
                               or (vid.startswith("up_") and "_step" in vid
                                   and not vid.endswith("__input")))
                    is_comb = is_step and comb_active and vid in comb_active
                    # Detect OVERLAY-mode combine steps so we can show
                    # an extra thumbnail beneath the main mask.
                    is_overlay = False
                    # Detect per-step YOLO add-on so we can show the
                    # boxed-image thumbnail beneath the main mask
                    # (separate output, never replaces the mask).
                    has_yolo = False
                    _step_tup_for_node = None
                    if is_step:
                        if vid.startswith("rgb_step"):
                            _i = int(vid[len("rgb_step"):]) - 1
                            if 0 <= _i < len(self.rgb_pipeline):
                                _step_tup_for_node = self.rgb_pipeline[_i]
                                if is_comb:
                                    is_overlay = (
                                        self.rgb_pipeline[_i][8].get() == "OVERLAY")
                        elif vid.startswith("ir_step"):
                            _i = int(vid[len("ir_step"):]) - 1
                            if 0 <= _i < len(self.ir_pipeline):
                                _step_tup_for_node = self.ir_pipeline[_i]
                                if is_comb:
                                    is_overlay = (
                                        self.ir_pipeline[_i][8].get() == "OVERLAY")
                        elif vid.startswith("up_") and "_step" in vid:
                            _bare = vid[len("up_"):]
                            _nmx, _, _st = _bare.rpartition("_step")
                            try:
                                _ix = int(_st) - 1
                            except ValueError:
                                _ix = -1
                            for _up in getattr(self, "user_pipelines", []):
                                if _up["name"].get().strip() == _nmx \
                                   and 0 <= _ix < len(_up["steps"]):
                                    _step_tup_for_node = _up["steps"][_ix]
                                    if is_comb:
                                        is_overlay = (
                                            _up["steps"][_ix][8].get() == "OVERLAY")
                                    break
                    # YOLO-enabled detection — read per-step state.
                    if _step_tup_for_node is not None:
                        try:
                            _yst = (getattr(self, "_yolo_state", {}) or {}
                                    ).get(id(_step_tup_for_node))
                            if _yst is not None and _yst["yolo_en"].get():
                                has_yolo = True
                        except Exception:
                            pass
                    bg = "#1a0a2e" if is_comb else "#1a1a2e" if is_step else node_bg
                    hl = "#9944cc" if is_comb else "#4466aa"  if is_step else "#555555"
                    frm = tk.Frame(fc, bg=bg, highlightbackground=hl,
                                   highlightthickness=1)
                    # Heights:
                    #   plain step  -> CH
                    #   combine     -> CH + small_row + caption + slack
                    #   overlay     -> combine height + full overlay thumb
                    _SMALL_H = max(28, FC_H // 3)
                    _comb_extra = _SMALL_H + 70             # small row + caption
                    # OVERLAY: single composite thumbnail whose width
                    # adapts to the tile count for this step:
                    #   2 tiles = Mask 1 -> overlay
                    #   3 tiles = Mask 1 + (Mask 2 or Base) -> overlay
                    #   4 tiles = Mask 1 + Mask 2 + Base -> overlay
                    _OV_GLYPH = 64
                    # Extra height when YOLO is enabled on this step
                    # — fits THREE sub-thumbnails (raw / boxed / mask)
                    # each with its own caption.
                    _yolo_extra = (3 * (FC_H + 18)) if has_yolo else 0
                    # HSV pair tile: stacked full-size channel + cmap
                    # (or channel + detection mask for branch pairs).
                    # 2 * FC_H tall plus a per-half caption above each.
                    is_pair = vid.endswith("_pair")
                    _pair_gap = 38   # room for the two half-captions
                    if is_overlay:
                        _ov_count = self._overlay_count_for_vid(vid)
                        _OV_W = (_ov_count * FC_W
                                 + (_ov_count - 1) * _OV_GLYPH)
                        node_h = CH + _yolo_extra
                        node_w = _OV_W + 8
                    elif is_comb:
                        _ov_count = 0
                        node_h = CH + _comb_extra + _yolo_extra
                        node_w = CW
                    elif is_pair:
                        _ov_count = 0
                        node_h = CH + FC_H + _pair_gap
                        node_w = CW
                    else:
                        _ov_count = 0
                        node_h = CH + _yolo_extra
                        node_w = CW
                    fc.create_window(nx, ny, window=frm, anchor="nw",
                                     width=node_w, height=node_h)
                    is_pm_step = (vid.startswith("rgb_pre_step")
                                  or vid.startswith("ir_pre_step")
                                  or (vid.startswith("up_")
                                      and "_pre_step" in vid))
                    # PM colour wins over the generic step colour
                    # because branch PM ids look like "_step" ids to
                    # the is_step substring check above.
                    txt_lbl = tk.Label(frm, text=label,
                                       font=("DejaVu Sans", 9, "bold"), bg=bg,
                                       fg=("#cc88ff" if is_comb
                                           else "#88dd88" if is_pm_step
                                           else "#aaccff" if is_step
                                           else "#dddddd"),
                                       anchor="w")
                    txt_lbl.pack(fill="x", padx=2)
                    if is_step or is_pm_step:
                        self.all_masks_step_labels[vid] = txt_lbl
                    # Lock the inner frame to its canvas-allocated
                    # pixel size so the input thumbnail (which has no
                    # text label content beyond the title) renders the
                    # same size as every other step thumbnail.
                    frm.pack_propagate(False)

                    def _make_thumb(parent, w=FC_W, h=FC_H):
                        """Create a placeholder-filled label of pixel size wxh."""
                        lbl = tk.Label(parent, bg="#000000", cursor="hand2")
                        try:
                            _ph = tk.PhotoImage(width=w, height=h)
                            _ph.put("#000000", to=(0, 0, w, h))
                            lbl.imgtk = _ph
                            lbl.configure(image=_ph)
                        except Exception:
                            lbl.configure(width=w, height=h)
                        return lbl

                    if is_overlay:
                        img_lbl = _make_thumb(frm, w=_OV_W, h=FC_H)
                        img_lbl.pack(padx=1, pady=(0, 1))
                    elif is_pair:
                        # TWO independent full-size thumbnails inside
                        # the same node frame: top + bottom, each with
                        # its own caption so the user can tell which
                        # half is which. Each is clickable on its own
                        # (registered separately in all_masks_labels)
                        # so the update loop renders each at native
                        # size — no combined synthesis, no squish.
                        _pm = pair_meta.get(vid)
                        _ptop_l = _pm[2] if _pm else "channel"
                        _pbot_l = _pm[3] if _pm else "colormap"
                        tk.Label(frm, text=_ptop_l,
                                 font=("DejaVu Sans", 8, "italic"),
                                 bg=bg, fg="#cccccc", anchor="w"
                                 ).pack(fill="x", padx=2)
                        img_lbl  = _make_thumb(frm)        # top
                        img_lbl.pack(padx=1, pady=(0, 1))
                        tk.Label(frm, text=_pbot_l,
                                 font=("DejaVu Sans", 8, "italic"),
                                 bg=bg, fg="#cccccc", anchor="w"
                                 ).pack(fill="x", padx=2)
                        img_lbl2 = _make_thumb(frm)        # bottom
                        img_lbl2.pack(padx=1, pady=(0, 1))
                    else:
                        img_lbl = _make_thumb(frm)
                        img_lbl.pack(padx=1, pady=(0, 1))
                    if vid.endswith("__input"):
                        _up_name = vid[len("up_"):-len("__input")]
                        _src_view = "rgb_raw"
                        for _up in getattr(self, "user_pipelines", []):
                            if _up["name"].get().strip() == _up_name:
                                _src_view = _up["source"].get()
                                break
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_src_view: self._open_zoom(v))
                        self.all_masks_labels[f"{vid}@{_src_view}"] = img_lbl
                    elif is_pair:
                        # Pair node: top + bottom thumbnails, each
                        # registered independently (via an @-alias so
                        # branch pairs that share a channel view don't
                        # clobber each other) at full FC_W x FC_H.
                        _pm = pair_meta.get(vid)
                        if _pm:
                            _top_v, _bot_v = _pm[0], _pm[1]
                        else:
                            _b = vid[: -len("_pair")]   # rgb_hsv_H
                            _top_v, _bot_v = _b, f"{_b}_cmap"
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_top_v: self._open_zoom(v))
                        img_lbl2.bind("<Button-1>",
                                      lambda e, v=_bot_v: self._open_zoom(v))
                        self.all_masks_labels[
                            f"{vid}__ptop@{_top_v}"] = img_lbl
                        self.all_masks_labels[
                            f"{vid}__pbot@{_bot_v}"] = img_lbl2
                    elif vid.startswith("_info_"):
                        # Channel-info / per-channel-mask / bgsub-fg
                        # thumbnail. Map vid -> view name and register
                        # an @-aliased label.
                        _bare = vid[len("_info_"):]
                        # _bare format: "<branch>_<chan>_<kind>"
                        # where kind in {"src", "mask"} and chan in
                        # {"h", "s", "v", "ir", "bgsub"}.
                        _kind = _bare.rsplit("_", 1)[-1]
                        _rest = _bare[: -(len(_kind) + 1)]
                        # Branch name is everything before the channel.
                        _chan_keys = ("h", "s", "v", "ir")
                        _ch = None
                        _bnm = _rest
                        for _ck in _chan_keys:
                            if _rest.endswith("_" + _ck):
                                _ch = _ck
                                _bnm = _rest[: -(len(_ck) + 1)]
                                break
                        # Source-image pipeline thumbnails (branch).
                        if vid.endswith("_src_raw"):
                            _bnm = _bare[: -len("_src_raw")]
                            _view = f"up_{_bnm}_src_raw"
                            _label = "Source (raw)"
                        elif vid.endswith("_src_blur"):
                            _bnm = _bare[: -len("_src_blur")]
                            _view = f"up_{_bnm}_src_blur"
                            _label = "Source (pre-blur)"
                        elif vid.endswith("_src_bgsub"):
                            _bnm = _bare[: -len("_src_bgsub")]
                            _view = f"up_{_bnm}_src_bgsub"
                            _label = "Source (BG removed)"
                        # bgsub / branch-det special cases (no chan).
                        elif vid.endswith("_bgsub"):
                            _bnm = _bare[: -len("_bgsub")]
                            _view = f"up_{_bnm}_bgsub"
                            _label = "BG-sub FG"
                        elif vid.endswith("_det"):
                            _bnm = _bare[: -len("_det")]
                            _view = f"up_{_bnm}_det"
                            _label = "branch_det"
                        elif _ch in ("h", "s", "v") and _kind == "src":
                            _view = f"rgb_hsv_{_ch.upper()}"
                            _label = f"{_ch.upper()} channel"
                        elif _ch in ("h", "s", "v") and _kind == "mask":
                            # Use h1_mask for H detection (the detection
                            # already AND-merges H1+H2 into _det; we
                            # just show H1 here as the primary).
                            _suffix = ("h1_mask" if _ch == "h"
                                       else f"{_ch}_mask")
                            _view = f"up_{_bnm}_{_suffix}"
                            _label = f"{_ch.upper()} mask"
                        elif _ch == "ir" and _kind == "src":
                            _view = "ir_raw"
                            _label = "IR frame"
                        elif _ch == "ir" and _kind == "mask":
                            _view = f"up_{_bnm}_ir_mask"
                            _label = "IR mask"
                        else:
                            _view = "rgb_raw"
                            _label = vid
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_view: self._open_zoom(v))
                        self.all_masks_labels[f"{vid}@{_view}"] = img_lbl
                    elif is_overlay:
                        # OVERLAY: register under a count-aware alias
                        # so the update loop knows the composite width.
                        # Left-click  -> zoom big composite AND every
                        #                individual tile in its own window
                        # Right-click -> popup menu for picking ONE tile
                        _comp_view = f"{vid}_composite"
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_comp_view:
                                         self._open_composite_zoom_all(v))
                        img_lbl.bind("<Button-3>",
                                     lambda e, v=_comp_view:
                                         self._open_composite_tile_menu(e, v))
                        self.all_masks_labels[
                            f"{vid}__cmp{_ov_count}@{_comp_view}"
                        ] = img_lbl
                    else:
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=vid: self._open_zoom(v))
                        self.all_masks_labels[vid] = img_lbl

                    # YOLO sub-thumbnails — separate from the running
                    # mask. Three views are emitted per step when YOLO
                    # is enabled:
                    #   raw   = the source image WITHOUT boxes
                    #   box   = the source image WITH boxes drawn on it
                    #   mask  = the binary union of YOLO box masks
                    # (Boxes can't be drawn on the binary mask, hence
                    # the box view sits alongside it like OVERLAY.)
                    if has_yolo:
                        for _suffix, _caption in (
                                ("yolo_raw",  "v YOLO source (raw)"),
                                ("yolo",      "v YOLO box (output)"),
                                ("yolo_mask", "v YOLO mask (output)")):
                            tk.Label(frm, text=_caption,
                                     font=("DejaVu Sans", 9, "italic"),
                                     bg=bg, fg="#ffaa66"
                                     ).pack(anchor="w", padx=3)
                            _yt_lbl = _make_thumb(frm)
                            _yt_lbl.pack(padx=1, pady=(0, 1))
                            _yview = f"{vid}_{_suffix}"
                            _yt_lbl.bind(
                                "<Button-1>",
                                lambda e, v=_yview: self._open_zoom(v))
                            self.all_masks_labels[
                                f"{vid}__{_suffix}@{_yview}"] = _yt_lbl

                    if is_comb:
                        # Resolve the step's owning pipeline + index up
                        # front so both OVERLAY and AND/OR/XOR branches
                        # can use _csr / _img1_view.
                        _step = None
                        _channel = None
                        _idx = -1
                        if vid.startswith("rgb_step"):
                            _idx = int(vid[len("rgb_step"):]) - 1
                            _step = self.rgb_pipeline[_idx]
                            _channel = "rgb"
                        elif vid.startswith("ir_step"):
                            _idx = int(vid[len("ir_step"):]) - 1
                            _step = self.ir_pipeline[_idx]
                            _channel = "ir"
                        elif vid.startswith("up_") and "_step" in vid:
                            _bare = vid[len("up_"):]
                            _nm, _, _stxt = _bare.rpartition("_step")
                            try:
                                _idx = int(_stxt) - 1
                            except ValueError:
                                _idx = 0
                            for _up in getattr(self, "user_pipelines", []):
                                if _up["name"].get().strip() == _nm:
                                    if 0 <= _idx < len(_up["steps"]):
                                        _step = _up["steps"][_idx]
                                    _channel = f"up_{_nm}"
                                    break
                        if _step is None or _channel is None:
                            continue
                        _cop = _step[8].get()
                        _csr = _step[9].get()
                        if _idx > 0:
                            _img1_view = f"{_channel}_step{_idx}"
                        elif _channel.startswith("up_"):
                            _img1_view = f"{_channel}__input"
                        else:
                            _img1_view = f"{_channel}_mask_pre"
                        _branch_view = f"{vid}_branch"

                    # OVERLAY: nothing extra - the single thumbnail
                    # (img_lbl) renders the composite view that the
                    # _process update loop populates.

                    # AND/OR/XOR combine (not OVERLAY): small
                    # img1 +op img2 row beneath the main thumbnail.
                    if is_comb and not is_overlay:
                        cmb_row = tk.Frame(frm, bg=bg)
                        cmb_row.pack(padx=1, pady=(2, 1))
                        _SMALL   = max(36, FC_W // 3 - 4)
                        _SMALL_H = max(28, FC_H // 3)
                        img1 = tk.Label(cmb_row, bg="#000000",
                                        width=_SMALL, height=_SMALL_H,
                                        cursor="hand2")
                        img1.pack(side="left", padx=1)
                        img1.bind("<Button-1>",
                                  lambda e, v=_img1_view:
                                      self._open_zoom(v))
                        self.all_masks_labels[
                            f"{vid}__img1@{_img1_view}"] = img1
                        tk.Label(cmb_row, text=f"+{_cop}",
                                 font=("DejaVu Sans", 8, "bold"),
                                 bg=bg, fg="#bb66ff"
                                 ).pack(side="left", padx=2)
                        img2 = tk.Label(cmb_row, bg="#000000",
                                        width=_SMALL, height=_SMALL_H,
                                        cursor="hand2")
                        img2.pack(side="left", padx=1)
                        img2.bind("<Button-1>",
                                  lambda e, v=_branch_view:
                                      self._open_zoom(v))
                        self.all_masks_labels[
                            f"{vid}__img2@{_branch_view}"] = img2
                        tk.Label(frm,
                                 text=(f"img1 (prev) {_cop} img2 "
                                       f"(src+Op)  ->  result above"),
                                 font=("DejaVu Sans", 9),
                                 bg=bg, fg="#c8c8c8"
                                 ).pack(pady=(0, 1))

        if show_rgb:
            draw_section(rgb_nodes, rgb_edges, rgb_y0,
                         "---------------  RGB Pipeline  ---------------",
                         "#2b4f7a", "#1a2535", "#888888", "#ff8800",
                         comb_active=combine_active)
        if show_ir:
            draw_section(ir_nodes, ir_edges, ir_y0,
                         "----------------  IR Pipeline  ----------------",
                         "#1a5c2a", "#141f14", "#888888", "#ff8800",
                         comb_active=combine_active)
        for sec, y0, _show in zip(up_sections, up_y0s, up_visible):
            if not _show:
                continue
            draw_section(sec["nodes"], sec["edges"], y0,
                         sec["title"],
                         sec["hdr_bg"], sec["node_bg"],
                         "#aa66cc", "#ff8800",
                         comb_active=combine_active)

        # Match the All-Masks window to its OWN theme (am_theme) +
        # the shared font settings — its toolbar, buttons and node
        # frames/labels are ordinary widgets the walk can recolour.
        if hasattr(self, "_style_all_masks"):
            self._style_all_masks(win)

        # Some view products (HSV colormap / pair tiles) are only
        # generated while this window is open — re-process the current
        # frame now so those thumbnails fill in immediately.
        if hasattr(self, "_refresh"):
            self._refresh()

    # ------------------------------------------------------------------
    # Zoom window (click any thumbnail in All Masks)
    # ------------------------------------------------------------------
    def _open_composite_tile_menu(self, event, composite_view):
        """Right-click handler on an OVERLAY composite thumbnail.
        Pops a small menu listing each tile (Mask 1 / Mask 2 / Base /
        Result) plus 'All tiles'. Selecting an entry opens that tile
        in its own zoom window."""
        entries = (getattr(self, "_composite_tile_views", {})
                   .get(composite_view, []))
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="All tiles",
            command=lambda v=composite_view: self._open_zoom(v))
        if entries:
            menu.add_separator()
            for tile_view, label in entries:
                menu.add_command(
                    label=f"{label}",
                    command=lambda v=tile_view: self._open_zoom(v))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _open_composite_zoom_all(self, composite_view):
        """Left-click handler on an OVERLAY composite thumbnail.
        Opens the wide composite zoom window AND a separate zoom
        window for every individual tile in the composite (Mask 1 /
        Mask 2 / Base / Result, as applicable)."""
        # Toggle behaviour: if any of these windows is already open,
        # closing the composite via _open_zoom also closes all tile
        # windows for symmetry — otherwise the user is left with
        # orphan windows after they re-click to "close zoom".
        entries = (getattr(self, "_composite_tile_views", {})
                   .get(composite_view, []))
        wins = getattr(self, "_zoom_wins", {})
        if composite_view in wins:
            # Composite is currently open — close composite + all tiles.
            self._open_zoom(composite_view)
            for tile_view, _label in entries:
                if tile_view in wins:
                    self._open_zoom(tile_view)
            return
        # Otherwise open them all.
        self._open_zoom(composite_view)
        for tile_view, _label in entries:
            if tile_view not in wins:
                self._open_zoom(tile_view)

    def _open_zoom(self, view_name):
        """Open (or focus) a zoom window for `view_name`.
        Multiple zoom windows can be open at once — clicking the
        same thumbnail again toggles its window closed."""
        if not hasattr(self, "_zoom_wins"):
            self._zoom_wins = {}
        existing = self._zoom_wins.get(view_name)
        if existing is not None and existing["win"].winfo_exists():
            # Toggle: clicking the same thumbnail closes the window.
            existing["win"].destroy()
            self._zoom_wins.pop(view_name, None)
            # Update legacy aliases.
            if self._zoom_view == view_name:
                self._zoom_win = self._zoom_label = self._zoom_view = None
            return

        win = tk.Toplevel(self.root)
        win.title(f"Zoom - {view_name}")
        win.resizable(True, True)
        win.protocol(
            "WM_DELETE_WINDOW",
            lambda v=view_name: self._close_zoom(v))
        lbl = tk.Label(win, bg="black")
        lbl.pack(fill="both", expand=True)
        readout = tk.Label(
            win, text="(hover the image to read pixel values)",
            font=("Courier", 9), bg="#111111", fg="#cccccc",
            anchor="w", padx=6, pady=2)
        readout.pack(fill="x", side="bottom")
        entry = {
            "win":     win,
            "label":   lbl,
            "readout": readout,
            "src_img": None,
        }
        self._zoom_wins[view_name] = entry
        lbl.bind("<Motion>",
                 lambda e, v=view_name: self._on_zoom_hover(e, v))
        lbl.bind("<Leave>",
                 lambda e, v=view_name: self._on_zoom_leave(v))
        # Legacy single-window aliases — most-recently-opened window
        # wins. Existing code that pokes self._zoom_label keeps working.
        self._zoom_win   = win
        self._zoom_label = lbl
        self._zoom_view  = view_name

        if self._last_f_rgb is not None:
            self._refresh()

    def _on_zoom_leave(self, view_name=None, _event=None):
        if view_name is None:
            return
        z = (getattr(self, "_zoom_wins", {}) or {}).get(view_name)
        if z and z["readout"].winfo_exists():
            z["readout"].config(
                text="(hover the image to read pixel values)")

    def _on_zoom_hover(self, event, view_name):
        """Translate the cursor's pixel position inside this window's
        zoom label into ORIGINAL image coordinates and read the pixel
        value(s). Shows BGR + HSV for 3-channel views, gray for masks."""
        z = (getattr(self, "_zoom_wins", {}) or {}).get(view_name)
        if z is None or not z["readout"].winfo_exists():
            return
        readout = z["readout"]
        lbl     = z["label"]
        try:
            ph = lbl.imgtk
            disp_w = ph.width()
            disp_h = ph.height()
        except Exception:
            return
        lbl_w = lbl.winfo_width()
        lbl_h = lbl.winfo_height()
        ox = (lbl_w - disp_w) // 2 if lbl_w > disp_w else 0
        oy = (lbl_h - disp_h) // 2 if lbl_h > disp_h else 0
        cx = event.x - ox
        cy = event.y - oy
        if cx < 0 or cy < 0 or cx >= disp_w or cy >= disp_h:
            readout.config(text="(out of image)")
            return
        src = z.get("src_img")
        if src is None:
            readout.config(
                text=f"display x={cx} y={cy}  (no source available)")
            return
        sh, sw = src.shape[:2]
        sx = int(cx * sw / max(1, disp_w))
        sy = int(cy * sh / max(1, disp_h))
        sx = max(0, min(sw - 1, sx))
        sy = max(0, min(sh - 1, sy))
        try:
            # Detect a channel-specific view by exact-suffix match on
            # the view name (so rgb_hsv_S and *_s_mask both resolve
            # cleanly without one accidentally matching the other).
            # If the view IS channel-specific, the readout shows the
            # value from the SOURCE rgb_raw at the cursor — not from
            # the colour-mapped visualisation, which would be wrong.
            low = view_name.lower()
            channel = None      # "H", "S", "V", "IR", or None
            f_raw   = getattr(self, "_last_f_rgb", None)

            # Order matters: most-specific endswith first.
            if low.endswith("_h1_mask") or low.endswith("_h2_mask"):
                channel = "H"
            elif low.endswith("_s_mask"):
                channel = "S"
            elif low.endswith("_v_mask"):
                channel = "V"
            elif low.endswith("_ir_mask"):
                channel = "IR"
            elif (low.endswith("_hsv_h")
                  or low.endswith("_hsv_h_cmap")
                  or low.endswith("_hsv_h_pair")):
                channel = "H"
            elif (low.endswith("_hsv_s")
                  or low.endswith("_hsv_s_cmap")
                  or low.endswith("_hsv_s_pair")):
                channel = "S"
            elif (low.endswith("_hsv_v")
                  or low.endswith("_hsv_v_cmap")
                  or low.endswith("_hsv_v_pair")):
                channel = "V"
            elif low.endswith("_hsv_full"):
                channel = "HSV"

            # Compute channel values from the ORIGINAL RGB frame.
            Hv = Sv = Vv = None
            if (channel is not None and f_raw is not None
                    and f_raw.ndim == 3):
                rh, rw = f_raw.shape[:2]
                if 0 <= sy < rh and 0 <= sx < rw:
                    import cv2 as _cv2
                    hsv_px = _cv2.cvtColor(
                        f_raw[sy:sy+1, sx:sx+1],
                        _cv2.COLOR_BGR2HSV)[0, 0]
                    Hv = int(hsv_px[0])
                    Sv = int(hsv_px[1])
                    Vv = int(hsv_px[2])

            # Helper: also include what the COLORMAP visualisation
            # paints at this pixel (BGR triplet of the rendered image)
            # so the user can verify they're hovering over the right
            # spot. The "source HSV" comes from rgb_raw — this is the
            # value that PRODUCED the colormap, so it's authoritative.
            disp_bgr = ""
            try:
                _px = src[sy, sx]
                if src.ndim == 3:
                    _B, _G, _R = int(_px[0]), int(_px[1]), int(_px[2])
                    disp_bgr = (f"   |  displayed BGR=({_B:3d},"
                                f"{_G:3d},{_R:3d})")
                else:
                    disp_bgr = f"   |  displayed gray={int(_px):3d}"
            except Exception:
                disp_bgr = ""

            if channel == "H" and Hv is not None:
                txt = (f"x={sx:4d} y={sy:4d}   "
                       f"source H = {Hv:3d}   "
                       f"(full source HSV={Hv:3d}/{Sv:3d}/{Vv:3d})"
                       f"{disp_bgr}")
            elif channel == "S" and Sv is not None:
                txt = (f"x={sx:4d} y={sy:4d}   "
                       f"source S = {Sv:3d}   "
                       f"(full source HSV={Hv:3d}/{Sv:3d}/{Vv:3d})"
                       f"{disp_bgr}")
            elif channel == "V" and Vv is not None:
                txt = (f"x={sx:4d} y={sy:4d}   "
                       f"source V = {Vv:3d}   "
                       f"(full source HSV={Hv:3d}/{Sv:3d}/{Vv:3d})"
                       f"{disp_bgr}")
            elif channel == "HSV" and Hv is not None:
                txt = (f"x={sx:4d} y={sy:4d}   "
                       f"source HSV: H={Hv:3d}  S={Sv:3d}  V={Vv:3d}"
                       f"{disp_bgr}")
            elif channel == "IR":
                # Single-channel mask: show its own gray value.
                if src.ndim == 3:
                    val = int(src[sy, sx][0])  # any channel of mask
                else:
                    val = int(src[sy, sx])
                txt = (f"x={sx:4d} y={sy:4d}   "
                       f"IR mask  =>  {val}")
            else:
                # Non-channel view — generic BGR/RGB/HSV (3-channel)
                # or gray (1-channel) readout.
                px = src[sy, sx]
                if src.ndim == 3:
                    B, G, R = int(px[0]), int(px[1]), int(px[2])
                    import cv2 as _cv2
                    hsv = _cv2.cvtColor(src[sy:sy+1, sx:sx+1],
                                        _cv2.COLOR_BGR2HSV)[0, 0]
                    H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
                    txt = (f"x={sx:4d} y={sy:4d}   "
                           f"BGR=({B:3d},{G:3d},{R:3d})   "
                           f"RGB=({R:3d},{G:3d},{B:3d})   "
                           f"HSV=({H:3d},{S:3d},{V:3d})")
                else:
                    gray = int(px)
                    txt = (f"x={sx:4d} y={sy:4d}   "
                           f"gray={gray:3d}   "
                           f"hex=#{gray:02x}{gray:02x}{gray:02x}")
            readout.config(text=txt)
        except Exception as e:
            readout.config(text=f"(read err: {e})")

    def _close_zoom(self, view_name=None):
        """Close one or all zoom windows.
        view_name=None closes every open zoom (used at app shutdown)."""
        wins = getattr(self, "_zoom_wins", {}) or {}
        targets = [view_name] if view_name is not None else list(wins.keys())
        for vn in targets:
            entry = wins.pop(vn, None)
            if entry is not None and entry["win"].winfo_exists():
                entry["win"].destroy()
        if self._zoom_win and self._zoom_win.winfo_exists():
            try:
                self._zoom_win.destroy()
            except Exception:
                pass
        self._zoom_win = self._zoom_label = self._zoom_view = None
