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
        # ── Layout constants ──────────────────────────────────────────
        CW  = FC_W + 4              # node frame width
        CH  = FC_H + 22             # node frame height
        HS  = CW + 44               # horizontal step
        VS  = CH + 44               # vertical step
        MX, MY   = 20, 50          # MY enlarged: leaves clear band above the
                                   # header for the YOLO bypass line.
        SECT_H   = 30
        SECT_GAP = 80              # extra gap between sections so the IR
                                   # YOLO bypass has room above its header.
        YOLO_BAND = 18             # vertical pixels reserved above each
                                   # header for the YOLO routing line.
        self._all_masks_img_size = (FC_W, FC_H)

        n_rgb = len(self.rgb_pipeline)
        n_ir  = len(self.ir_pipeline)

        # OVERLAY-aware step positioning. An OVERLAY step occupies 3
        # grid columns visually (since its composite thumbnail is 3×
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

        rgb_positions = _step_positions(self.rgb_pipeline, STEP_ROW_BASE)
        ir_positions  = _step_positions(self.ir_pipeline,  1)
        rgb_out = _last_row_after(self.rgb_pipeline, STEP_ROW_BASE)
        ir_out  = _last_row_after(self.ir_pipeline,  1)

        # ── Node lists ────────────────────────────────────────────────
        rgb_nodes = [
            ("rgb_raw",       0, 0, " 1  Input"),
            ("rgb_blur",      1, 0, " 2  Blur*"),
            ("rgb_hsv_full",  2, 0, " 3  HSV colour"),
            ("rgb_hsv_H",     3, 0, " 4  Hue"),
            ("rgb_hsv_S",     4, 0, " 5  Saturation"),
            ("rgb_hsv_V",     5, 0, " 6  Value"),
            ("rgb_m1",        2, 1, " 7  Hue mask 1"),
            ("rgb_m2",        3, 1, " 8  Hue mask 2"),
            ("rgb_hsv_mask",  4, 1, " 9  HSV mask"),
            ("rgb_bgsub",     5, 1, "10  BG sub"),
            ("rgb_mask_pre",  6, 1, "11  Pre-pipeline"),
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

        # ── Edges ─────────────────────────────────────────────────────
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

        rgb_edges = [
            ("rgb_raw",      "rgb_blur"),
            ("rgb_blur",     "rgb_hsv_full"),
            ("rgb_hsv_full", "rgb_hsv_H"),
            ("rgb_hsv_H",    "rgb_hsv_S"),
            ("rgb_hsv_S",    "rgb_hsv_V"),
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
            ("ir_bgsub",     "ir_thresh"),
            ("ir_thresh",    "ir_mask_pre"),
            *_step_edges("ir", n_ir),
            ("ir_mask",      "ir_post_blur"),
            ("ir_post_blur", "ir_det"),
            ("ir_raw",       "_yolo_ir"),
            ("_yolo_ir",     "ir_det"),
        ]

        # Combine edges: arrows from a source mask → step when combine ON
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
                               f"⊕{_cop.get()}", "combine"))
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

        # ── Build per-user-pipeline section descriptors ──────────────
        # Each entry: (nodes, edges, title, hdr_bg, node_bg, n_steps, color)
        up_sections = []
        for up in user_pipes:
            _nm  = up["name"].get().strip() or "branch"
            _src = up["source"].get()
            _n   = len(up["steps"])
            up_positions = _step_positions(up["steps"], 1)
            _out_row_up  = ((up_positions[-1][1] + 1)
                            if up_positions else 1)
            _input_id   = f"up_{_nm}__input"
            _output_id  = f"up_{_nm}"
            nodes_up = [
                (_input_id, 0, 0, f"src ← {_src}"),
                *[(f"up_{_nm}_step{i+1}",
                   up_positions[i][0],
                   up_positions[i][1],
                   f"S{i+1}: ?")
                  for i in range(_n)],
                # Output sits bottom-right (col YOLO_COL) like the
                # main RGB/IR pipelines' Detection node.
                (_output_id, YOLO_COL, _out_row_up,
                 f"out → up_{_nm}"),
            ]
            edges_up = []
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
                                 f"⊕{_cop.get()}", "combine"))
            up_sections.append({
                "name":  _nm,
                "nodes": nodes_up,
                "edges": edges_up,
                "out_row": _out_row_up,
                "title": f"━━━━━━━━━  Pipeline: {_nm}  (src ← {_src})  ━━━━━━━━━",
                "hdr_bg":  "#5c2a7a",   # purple for user pipelines
                "node_bg": "#1f1130",
            })

        # ── Canvas sizing ─────────────────────────────────────────────
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
            """Total y-extent of a section. AND/OR/XOR combine rows are
            taller; OVERLAY rows stay normal height (compact)."""
            comb_rows = {r for vid, _, r, _ in nodes
                         if (combine_active and vid in combine_active
                             and not _is_overlay_step_q(vid))}
            return (out_row + 1) * VS + len(comb_rows) * comb_extra

        canvas_w = MX + (YOLO_COL + 1) * HS + MX
        rgb_y0   = MY + SECT_H
        ir_y0    = rgb_y0 + _section_height(rgb_nodes, rgb_out) \
                          + SECT_GAP + SECT_H
        up_y0s = []
        _cursor_y = ir_y0 + _section_height(ir_nodes, ir_out) \
                          + SECT_GAP + SECT_H
        for sec in up_sections:
            up_y0s.append(_cursor_y)
            _cursor_y += _section_height(sec["nodes"], sec["out_row"]) \
                          + SECT_GAP + SECT_H
        canvas_h = (_cursor_y if up_sections
                    else ir_y0 + _section_height(ir_nodes, ir_out)) + MY

        # ── Toplevel + scrollable canvas ──────────────────────────────
        if self.all_masks_win and self.all_masks_win.winfo_exists():
            self.all_masks_win.destroy()

        win = tk.Toplevel(self.root)
        win.title("Pipeline Flow — All Masks   (* = blur position selectable)")
        win.protocol("WM_DELETE_WINDOW", self._toggle_all_masks_window)
        self.all_masks_win         = win
        self.all_masks_labels      = {}
        self.all_masks_step_labels = {}

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

        # ── Drawing helpers ───────────────────────────────────────────
        def _draw_edge_label(canvas, x, y, text):
            clr = ("#44ff44" if text == "OR"
                   else "#ff7744" if text == "AND"
                   else "#bb66ff" if text.startswith("⊕")
                   else "#ffff44")
            canvas.create_oval(x - 20, y - 10, x + 20, y + 10,
                               fill="#222222", outline=clr, width=2)
            canvas.create_text(x, y, text=text, fill=clr,
                               font=("Arial", 8, "bold"))

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

        def _compute_row_y(nodes, comb_active):
            """Compute the y-offset (relative to y0) of each row,
            stretching rows that contain combine-active step nodes so
            their taller frames don't overlap the next row. OVERLAY
            nodes are normal-height (just wider) so they need no extra
            vertical space."""
            comb_extra = (FC_H // 3) + 70
            max_row = max((r for _, _, r, _ in nodes), default=0)
            # AND/OR/XOR combine rows only — exclude OVERLAY rows since
            # OVERLAY nodes are now compact in height.
            comb_rows = {r for vid, _, r, _ in nodes
                         if (comb_active and vid in comb_active
                             and not _is_overlay_step(vid))}
            row_y = {0: 0}
            cur = 0
            for r in range(1, max_row + 2):
                step_h = VS + (comb_extra if (r - 1) in comb_rows else 0)
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
                           text=title, font=("Arial", 11, "bold"),
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
                                   text="YOLO", font=("Arial", 10, "bold"),
                                   fill=yolo_clr)
                    fc.create_text(nx + CW // 2, ny + CH // 2 + 9,
                                   text=label, font=("Arial", 7), fill="#aaaaaa")
                else:
                    is_step = (vid.startswith("rgb_step")
                               or vid.startswith("ir_step")
                               or (vid.startswith("up_") and "_step" in vid
                                   and not vid.endswith("__input")))
                    is_comb = is_step and comb_active and vid in comb_active
                    # Detect OVERLAY-mode combine steps so we can show
                    # an extra thumbnail beneath the main mask.
                    is_overlay = False
                    if is_comb:
                        if vid.startswith("rgb_step"):
                            _i = int(vid[len("rgb_step"):]) - 1
                            if 0 <= _i < len(self.rgb_pipeline):
                                is_overlay = (
                                    self.rgb_pipeline[_i][8].get() == "OVERLAY")
                        elif vid.startswith("ir_step"):
                            _i = int(vid[len("ir_step"):]) - 1
                            if 0 <= _i < len(self.ir_pipeline):
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
                                    is_overlay = (
                                        _up["steps"][_ix][8].get() == "OVERLAY")
                                    break
                    bg = "#1a0a2e" if is_comb else "#1a1a2e" if is_step else node_bg
                    hl = "#9944cc" if is_comb else "#4466aa"  if is_step else "#555555"
                    frm = tk.Frame(fc, bg=bg, highlightbackground=hl,
                                   highlightthickness=1)
                    # Heights:
                    #   plain step  → CH
                    #   combine     → CH + small_row + caption + slack
                    #   overlay     → combine height + full overlay thumb
                    _SMALL_H = max(28, FC_H // 3)
                    _comb_extra = _SMALL_H + 70             # small row + caption
                    # OVERLAY: single composite thumbnail whose width
                    # adapts to the tile count for this step:
                    #   2 tiles = Mask 1 → overlay
                    #   3 tiles = Mask 1 + (Mask 2 or Base) → overlay
                    #   4 tiles = Mask 1 + Mask 2 + Base → overlay
                    _OV_GLYPH = 64
                    if is_overlay:
                        _ov_count = self._overlay_count_for_vid(vid)
                        _OV_W = (_ov_count * FC_W
                                 + (_ov_count - 1) * _OV_GLYPH)
                        node_h = CH
                        node_w = _OV_W + 8
                    elif is_comb:
                        _ov_count = 0
                        node_h = CH + _comb_extra
                        node_w = CW
                    else:
                        _ov_count = 0
                        node_h = CH
                        node_w = CW
                    fc.create_window(nx, ny, window=frm, anchor="nw",
                                     width=node_w, height=node_h)
                    txt_lbl = tk.Label(frm, text=label,
                                       font=("Arial", 7, "bold"), bg=bg,
                                       fg=("#cc88ff" if is_comb
                                           else "#aaccff" if is_step
                                           else "#dddddd"),
                                       anchor="w")
                    txt_lbl.pack(fill="x", padx=2)
                    if is_step:
                        self.all_masks_step_labels[vid] = txt_lbl
                    # Lock the inner frame to its canvas-allocated
                    # pixel size so the input thumbnail (which has no
                    # text label content beyond the title) renders the
                    # same size as every other step thumbnail.
                    frm.pack_propagate(False)

                    def _make_thumb(parent, w=FC_W, h=FC_H):
                        """Create a placeholder-filled label of pixel size w×h."""
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
                    elif is_overlay:
                        # OVERLAY: register under a count-aware alias
                        # so the update loop knows the composite width.
                        # Click → zoom shows the wide composite so the
                        # user can see Mask 1 + Mask 2 + Base + result
                        # all at once.
                        _comp_view = f"{vid}_composite"
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_comp_view:
                                         self._open_zoom(v))
                        self.all_masks_labels[
                            f"{vid}__cmp{_ov_count}@{_comp_view}"
                        ] = img_lbl
                    else:
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=vid: self._open_zoom(v))
                        self.all_masks_labels[vid] = img_lbl

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

                    # OVERLAY: nothing extra — the single thumbnail
                    # (img_lbl) renders the composite view that the
                    # _process update loop populates.

                    # AND/OR/XOR combine (not OVERLAY): small
                    # img1 ⊕op img2 row beneath the main thumbnail.
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
                        tk.Label(cmb_row, text=f"⊕{_cop}",
                                 font=("Arial", 8, "bold"),
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
                                       f"(src⊕Op)  →  result above"),
                                 font=("Arial", 6),
                                 bg=bg, fg="#888888"
                                 ).pack(pady=(0, 1))

        draw_section(rgb_nodes, rgb_edges, rgb_y0,
                     "━━━━━━━━━━━━━━━  RGB Pipeline  ━━━━━━━━━━━━━━━",
                     "#2b4f7a", "#1a2535", "#888888", "#ff8800",
                     comb_active=combine_active)
        draw_section(ir_nodes, ir_edges, ir_y0,
                     "━━━━━━━━━━━━━━━━  IR Pipeline  ━━━━━━━━━━━━━━━━",
                     "#1a5c2a", "#141f14", "#888888", "#ff8800",
                     comb_active=combine_active)
        for sec, y0 in zip(up_sections, up_y0s):
            draw_section(sec["nodes"], sec["edges"], y0,
                         sec["title"],
                         sec["hdr_bg"], sec["node_bg"],
                         "#aa66cc", "#ff8800",
                         comb_active=combine_active)

    # ------------------------------------------------------------------
    # Zoom window (click any thumbnail in All Masks)
    # ------------------------------------------------------------------
    def _open_zoom(self, view_name):
        if self._zoom_win and self._zoom_win.winfo_exists():
            if self._zoom_view == view_name:
                self._zoom_win.destroy()
                self._zoom_win = self._zoom_label = self._zoom_view = None
                return
            self._zoom_view = view_name
            self._zoom_win.title(f"Zoom — {view_name}")
            if self._last_f_rgb is not None:
                self._refresh()
            return
        win = tk.Toplevel(self.root)
        win.title(f"Zoom — {view_name}")
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", self._close_zoom)
        self._zoom_win   = win
        self._zoom_view  = view_name
        self._zoom_label = tk.Label(win, bg="black")
        self._zoom_label.pack(fill="both", expand=True)
        if self._last_f_rgb is not None:
            self._refresh()

    def _close_zoom(self):
        if self._zoom_win and self._zoom_win.winfo_exists():
            self._zoom_win.destroy()
        self._zoom_win = self._zoom_label = self._zoom_view = None
