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

        # Row where post-pipeline nodes appear (after all step rows)
        def _rgb_out_row(n):
            return STEP_ROW_BASE + (n + STEPS_PER_ROW - 1) // STEPS_PER_ROW
        def _ir_out_row(n):
            return 1 + (n + STEPS_PER_ROW - 1) // STEPS_PER_ROW

        rgb_out = _rgb_out_row(n_rgb)
        ir_out  = _ir_out_row(n_ir)

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
               i % STEPS_PER_ROW,
               STEP_ROW_BASE + i // STEPS_PER_ROW,
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
               i % STEPS_PER_ROW,
               1 + i // STEPS_PER_ROW,
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
            _out_row_up = 1 + (_n + STEPS_PER_ROW - 1) // STEPS_PER_ROW
            _input_id   = f"up_{_nm}__input"
            _output_id  = f"up_{_nm}"
            nodes_up = [
                (_input_id, 0, 0, f"src ← {_src}"),
                *[(f"up_{_nm}_step{i+1}",
                   i % STEPS_PER_ROW,
                   1 + i // STEPS_PER_ROW,
                   f"S{i+1}: ?")
                  for i in range(_n)],
                (_output_id, 0, _out_row_up, f"out → up_{_nm}"),
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
        # Combine-active steps are taller; reserve enough vertical room.
        comb_extra = (FC_H // 2 + 32) if combine_active else 0
        canvas_w = MX + (YOLO_COL + 1) * HS + MX
        rgb_y0   = MY + SECT_H
        ir_y0    = rgb_y0 + (rgb_out + 1) * VS + comb_extra \
                          + SECT_GAP + SECT_H
        # Stack user-pipeline sections after IR.
        up_y0s = []
        _cursor_y = ir_y0 + (ir_out + 1) * VS + comb_extra + SECT_GAP + SECT_H
        for sec in up_sections:
            up_y0s.append(_cursor_y)
            _cursor_y += (sec["out_row"] + 1) * VS + comb_extra + SECT_GAP + SECT_H
        canvas_h = (_cursor_y if up_sections
                    else ir_y0 + (ir_out + 1) * VS + comb_extra) + MY

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

        def px(col, row, y0):  return MX + col * HS,       y0 + row * VS
        def right(c, r, y0):   x, y = px(c, r, y0); return x + CW, y + CH // 2
        def left(c, r, y0):    x, y = px(c, r, y0); return x,      y + CH // 2
        def bottom(c, r, y0):  x, y = px(c, r, y0); return x + CW // 2, y + CH
        def top(c, r, y0):     x, y = px(c, r, y0); return x + CW // 2, y

        def draw_section(nodes, edges, y0, title, hdr_bg, node_bg,
                         arr_clr, yolo_clr, comb_active=None):
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
                    bg = "#1a0a2e" if is_comb else "#1a1a2e" if is_step else node_bg
                    hl = "#9944cc" if is_comb else "#4466aa"  if is_step else "#555555"
                    frm = tk.Frame(fc, bg=bg, highlightbackground=hl,
                                   highlightthickness=1)
                    # Combine-active steps get a taller frame with an extra
                    # row showing image1 ⊕op image2 = combined.
                    node_h = (CH + FC_H // 2 + 32) if is_comb else CH
                    fc.create_window(nx, ny, window=frm, anchor="nw",
                                     width=CW, height=node_h)
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
                    img_lbl = tk.Label(frm, bg="#000000",
                                       cursor="hand2")
                    # Pre-set a black PhotoImage of the correct pixel
                    # size so the label uses image (pixel) units from
                    # the very first paint, before _process runs.
                    try:
                        _ph = tk.PhotoImage(width=FC_W, height=FC_H)
                        # Fill it black row-by-row so it's actually opaque.
                        _ph.put("#000000", to=(0, 0, FC_W, FC_H))
                        img_lbl.imgtk = _ph
                        img_lbl.configure(image=_ph)
                    except Exception:
                        img_lbl.configure(width=FC_W, height=FC_H)
                    img_lbl.pack(padx=1, pady=(0, 1))
                    # User-pipeline "input" stub nodes display the source
                    # view; route the image via the @-alias so the update
                    # loop knows which real view to show & zoom into.
                    if vid.endswith("__input"):
                        # vid format: "up_<name>__input"
                        _up_name = vid[len("up_"):-len("__input")]
                        _src_view = "rgb_raw"
                        for _up in getattr(self, "user_pipelines", []):
                            if _up["name"].get().strip() == _up_name:
                                _src_view = _up["source"].get()
                                break
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=_src_view: self._open_zoom(v))
                        self.all_masks_labels[f"{vid}@{_src_view}"] = img_lbl
                    else:
                        img_lbl.bind("<Button-1>",
                                     lambda e, v=vid: self._open_zoom(v))
                        self.all_masks_labels[vid] = img_lbl

                    # Combine-active step: extra explanatory row beneath the
                    # main thumbnail showing  image1 ⊕op image2 = combined.
                    if is_comb:
                        # Resolve the step's owning pipeline + index.
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
                            continue   # cannot render combine row safely
                        _cop = _step[8].get()
                        _csr = _step[9].get()
                        # Resolve the "image1" view: legacy shortcut OR
                        # full view name (cross-pipeline reference).
                        if _csr == "prev":
                            if _channel.startswith("up_"):
                                _img1_view = (f"{_channel}_step{_idx}"
                                              if _idx > 0
                                              else f"{_channel}__input")
                            else:
                                _img1_view = (f"{_channel}_step{_idx}"
                                              if _idx > 0
                                              else f"{_channel}_mask_pre")
                        elif _csr == "mask_pre":
                            _img1_view = (f"{_channel}__input"
                                          if _channel.startswith("up_")
                                          else f"{_channel}_mask_pre")
                        elif _csr.startswith("step_"):
                            try:
                                _img1_view = f"{_channel}_step{int(_csr.split('_')[1])}"
                            except (ValueError, IndexError):
                                _img1_view = (f"{_channel}__input"
                                              if _channel.startswith("up_")
                                              else f"{_channel}_mask_pre")
                        else:
                            # Cross-pipeline / direct view-name reference.
                            _img1_view = _csr
                        _branch_view = f"{vid}_branch"

                        cmb_row = tk.Frame(frm, bg=bg)
                        cmb_row.pack(padx=1, pady=(2, 1))
                        _SMALL = max(28, FC_W // 3 - 4)
                        _SMALL_H = max(20, FC_H // 3)
                        # image1: the running mask BEFORE the combine
                        img1 = tk.Label(cmb_row, bg="#000000",
                                        width=_SMALL, height=_SMALL_H,
                                        cursor="hand2")
                        img1.pack(side="left", padx=1)
                        img1.bind("<Button-1>",
                                  lambda e, v=_img1_view: self._open_zoom(v))
                        self.all_masks_labels[f"{vid}__img1@{_img1_view}"] = img1
                        # operator badge
                        tk.Label(cmb_row, text=f"⊕{_cop}",
                                 font=("Arial", 8, "bold"),
                                 bg=bg, fg="#bb66ff").pack(side="left", padx=2)
                        # image2: the source-after-op (the "branch")
                        img2 = tk.Label(cmb_row, bg="#000000",
                                        width=_SMALL, height=_SMALL_H,
                                        cursor="hand2")
                        img2.pack(side="left", padx=1)
                        img2.bind("<Button-1>",
                                  lambda e, v=_branch_view: self._open_zoom(v))
                        self.all_masks_labels[f"{vid}__img2@{_branch_view}"] = img2
                        # caption
                        tk.Label(frm, text=f"img1 ⊕{_cop} img2  →  combined",
                                 font=("Arial", 6), bg=bg,
                                 fg="#888888").pack(pady=(0, 1))

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
