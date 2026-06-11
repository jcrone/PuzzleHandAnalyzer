#!/usr/bin/env python3
"""
puzzle_report.py - interval heatmaps + one-pager report.

Reads a puzzle_hands.py output folder (a <base>_metrics.json and the matching
<base>_perframe.csv) and writes:

  <base>_heatmap.png    a 2x3 grid - one heatmap per time window, each with
                        an auto-derived "active zone" rectangle showing where
                        the puzzler was actually picking up / placing pieces
                        in that window.

  <base>_report.pdf     a single-page session summary (also written as .png)
  <base>_report.png

Can be run standalone on existing analyses:

    python3 puzzle_report.py path/to/<base>_metrics.json
    python3 puzzle_report.py path/to/<base>_metrics.json --video orig.mp4

If <base>_frame.jpg (a clean background frame) is present next to the JSON,
or --video is given, the heatmaps use it as a backdrop. Otherwise they're
drawn on a neutral background.
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

try:
    import cv2
except ImportError:
    cv2 = None

C_RIGHT = "#3c5af0"
C_LEFT  = "#f0aa28"

N_WINDOWS_DEFAULT = 6
MIN_GRIPS_FOR_ZONE = 5          # below this, no zone box drawn
PCT_LO, PCT_HI = 10, 90         # percentile bounds for the zone rectangle


# ------------------------------------------------------------------- loading
def load_data(json_path):
    """Returns (summary_dict, perframe_cols_dict, base_path_without_suffix)."""
    if not json_path.endswith("_metrics.json"):
        raise ValueError("expected a <name>_metrics.json file, got: "
                         + json_path)
    with open(json_path, encoding="utf-8") as f:
        summary = json.load(f)
    base = json_path[:-len("_metrics.json")]
    csv_path = base + "_perframe.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("missing per-frame data: " + csv_path)
    with open(csv_path, encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [list(map(float, row)) for row in r]
    arr = np.array(rows)
    cols = {h: arr[:, i] for i, h in enumerate(header)}
    return summary, cols, base


def find_bg_frame(base, video_path=None):
    """Returns an RGB numpy array for the heatmap backdrop, or None."""
    clean = base + "_frame.jpg"
    if os.path.exists(clean) and cv2 is not None:
        img = cv2.imread(clean)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if video_path and cv2 is not None and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# ----------------------------------------------------------------- windowing
def compute_windows(cols, n=N_WINDOWS_DEFAULT):
    """Slice the per-frame data into n equal-time windows.

    Each window's 'zone' is the percentile-trimmed bounding box of grip-
    onset and grip-release locations (across both hands) in that window.
    None if the window has fewer than MIN_GRIPS_FOR_ZONE such events.
    """
    N = len(cols["frame"])
    bounds = np.linspace(0, N, n + 1, dtype=int)
    windows = []
    for k in range(n):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if b - a < 2:
            continue
        t_start = float(cols["time_s"][a])
        t_end = float(cols["time_s"][b - 1])
        ex, ey = [], []
        for side in ("right", "left"):
            gr = cols[side + "_gripping"][a:b].astype(bool)
            cx = cols[side + "_x"][a:b]
            cy = cols[side + "_y"][a:b]
            if len(gr) > 1:
                ons = np.where(gr[1:] & ~gr[:-1])[0] + 1
                offs = np.where(~gr[1:] & gr[:-1])[0] + 1
                idx = np.concatenate([ons, offs]) if len(ons) or len(offs) else np.array([], dtype=int)
                if len(idx):
                    ex.extend(cx[idx].tolist())
                    ey.extend(cy[idx].tolist())
        zone = None
        if len(ex) >= MIN_GRIPS_FOR_ZONE:
            xs, ys = np.array(ex), np.array(ey)
            zone = (float(np.percentile(xs, PCT_LO)),
                    float(np.percentile(ys, PCT_LO)),
                    float(np.percentile(xs, PCT_HI)),
                    float(np.percentile(ys, PCT_HI)))
        windows.append({
            "index": k + 1,
            "t_start": t_start,
            "t_end": t_end,
            "grip_events": len(ex),
            "zone": zone,
        })
    return windows


# ----------------------------------------------------------- sort / assemble
def compute_sort_assemble(cols, windows, manual_board):
    """Returns sort-vs-assemble stats keyed to a single session zone.

    Uses the manual board if one was set during analysis, otherwise derives
    a session-wide zone from grip-onset / grip-release locations (same
    percentile-trimmed bounding-box logic as the per-window zones).

    Returns None if no usable zone can be derived (e.g., very few grips).
    """
    if manual_board is not None and len(manual_board) == 4:
        zone = tuple(float(v) for v in manual_board)
        source = "manual"
    else:
        ex, ey = [], []
        for side in ("right", "left"):
            gr = cols[side + "_gripping"].astype(bool)
            cx = cols[side + "_x"]
            cy = cols[side + "_y"]
            if len(gr) > 1:
                ons = np.where(gr[1:] & ~gr[:-1])[0] + 1
                offs = np.where(~gr[1:] & gr[:-1])[0] + 1
                idx = (np.concatenate([ons, offs])
                       if len(ons) or len(offs) else np.array([], dtype=int))
                if len(idx):
                    ex.extend(cx[idx].tolist())
                    ey.extend(cy[idx].tolist())
        if len(ex) < MIN_GRIPS_FOR_ZONE:
            return None
        xs, ys = np.array(ex), np.array(ey)
        zone = (float(np.percentile(xs, PCT_LO)),
                float(np.percentile(ys, PCT_LO)),
                float(np.percentile(xs, PCT_HI)),
                float(np.percentile(ys, PCT_HI)))
        source = "auto"

    x1, y1, x2, y2 = zone
    in_zone = {}
    for side in ("right", "left"):
        cx = cols[side + "_x"]
        cy = cols[side + "_y"]
        in_zone[side] = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

    session = {
        "zone": zone,
        "source": source,
        "right_assemble_pct": round(100 * float(in_zone["right"].mean()), 1),
        "left_assemble_pct":  round(100 * float(in_zone["left"].mean()), 1),
        "combined_assemble_pct": round(
            100 * float((in_zone["right"] | in_zone["left"]).mean()), 1),
    }

    per_window = []
    for win in windows:
        mask = ((cols["time_s"] >= win["t_start"]) &
                (cols["time_s"] <= win["t_end"]))
        if not mask.any():
            continue
        per_window.append({
            "index": win["index"],
            "t_mid_min": (win["t_start"] + win["t_end"]) / 2.0 / 60.0,
            "right_assemble_pct":
                round(100 * float(in_zone["right"][mask].mean()), 1),
            "left_assemble_pct":
                round(100 * float(in_zone["left"][mask].mean()), 1),
            "combined_assemble_pct": round(
                100 * float(
                    (in_zone["right"][mask] | in_zone["left"][mask]).mean()),
                1),
        })
    return {"session": session, "per_window": per_window}


# ------------------------------------------------------------------ heatmaps
def make_heatmaps(cols, windows, bg, manual_board, out_path):
    n = len(windows)
    if n <= 3:
        rows, ncols = 1, n
    else:
        ncols = 3
        rows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols,
                             figsize=(4.6 * ncols, 4.0 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for ax, win in zip(axes, windows):
        mask = ((cols["time_s"] >= win["t_start"]) &
                (cols["time_s"] <= win["t_end"]))
        xs = np.concatenate([cols["right_x"][mask], cols["left_x"][mask]])
        ys = np.concatenate([cols["right_y"][mask], cols["left_y"][mask]])
        if bg is not None:
            ax.imshow(bg, extent=[0, 1, 1, 0], alpha=0.55)
        hist, _, _ = np.histogram2d(xs, ys, bins=24,
                                    range=[[0, 1], [0, 1]])
        hist = np.sqrt(hist)
        masked = np.ma.masked_where(hist == 0, hist)
        ax.imshow(masked.T, extent=[0, 1, 1, 0], origin="upper",
                  cmap="magma", alpha=0.78, aspect="auto",
                  interpolation="bilinear")
        if win["zone"] is not None:
            x1, y1, x2, y2 = win["zone"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="lime",
                                   lw=2.2))
        if manual_board is not None:
            x1, y1, x2, y2 = manual_board
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="cyan",
                                   lw=1.5, ls="--"))
        ax.set_title("Window %d   %.1f - %.1f min\n%d grip events"
                     % (win["index"], win["t_start"] / 60.0,
                        win["t_end"] / 60.0, win["grip_events"]),
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)

    for ax in axes[len(windows):]:
        ax.axis("off")

    fig.suptitle("Where the hands worked - session split into %d windows"
                 % len(windows), fontsize=14, weight="bold", y=1.0)
    fig.text(0.5, 0.01,
             "lime box = auto-detected active zone (grip / release locations "
             "in that window)" +
             ("    cyan dash = manual board" if manual_board else ""),
             ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _competition_lines(summary):
    """Formatted one-pager lines for the competition-metric blocks. Returns
    a (possibly empty) list of strings; tolerates summaries lacking the
    blocks (older runs)."""
    lines = []
    pl = summary.get("placement")
    fp = summary.get("flip_phase") or {}
    ef = summary.get("efficiency") or {}
    if pl:
        line = ("Placing pace:  %.2f pieces/min overall   %.2f pieces/min assembly"
                % (pl.get("overall_pieces_per_min", 0),
                   pl.get("assembly_pieces_per_min", 0)))
        if "percent_complete" in pl:
            line += "   %.0f%% complete" % pl["percent_complete"]
        lines.append(line)
        sp = pl.get("splits") or {}
        if sp:
            def _s(v):
                return "-" if v is None else "%.0fs" % v
            lines.append("Splits:  25%% %s   50%% %s   75%% %s   100%% %s" % (
                _s(sp.get("25pct")), _s(sp.get("50pct")),
                _s(sp.get("75pct")), _s(sp.get("100pct"))))
    if fp:
        if fp.get("confidence") == "unavailable":
            lines.append("Flip/prep phase: not detected (%s)" % fp.get("note", ""))
        else:
            lines.append("Flip/prep: %.0fs   %d manipulations (flipping/sorting)   confidence %s"
                         % (fp.get("duration_s", 0), fp.get("manipulations", 0),
                            fp.get("confidence", "?")))
    if ef:
        lines.append("Productive %.0f%%   Dead time %.0f%%   Stalls: %d (longest %.0fs)"
                     % (ef.get("productive_pct", 0), ef.get("dead_time_pct", 0),
                        ef.get("stall_count", 0), ef.get("longest_stall_s", 0)))
    return lines


# ----------------------------------------------------------------- one pager
def make_one_pager(summary, windows, sa, base):
    fig = plt.figure(figsize=(8.5, 11))
    # split into two gridspecs so the top group can pack tightly
    # while the bottom group gets generous breathing room
    gs_top = GridSpec(4, 1, figure=fig,
                      height_ratios=[0.70, 0.85, 1.70, 1.60],
                      hspace=0.35,
                      left=0.07, right=0.95, top=0.965, bottom=0.55)
    gs_bot = GridSpec(3, 1, figure=fig,
                      height_ratios=[1.85, 1.60, 0.45],
                      hspace=0.95,
                      left=0.07, right=0.95, top=0.46, bottom=0.04)
    gs = [gs_top[0], gs_top[1], gs_top[2], gs_top[3],
          gs_bot[0], gs_bot[1], gs_bot[2]]

    L = summary.get("left", {}) or {}
    R = summary.get("right", {}) or {}
    bim = summary.get("bimanual", {}) or {}
    pace = summary.get("pace", {}) or {}
    fat = summary.get("fatigue", {}) or {}

    # ---- header
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    title = summary.get("puzzle_name") or summary.get("video", "session")
    ax.text(0, 0.78, title, fontsize=20, weight="bold")

    meta_parts = []
    if summary.get("num_pieces"):
        meta_parts.append("%d pieces" % summary["num_pieces"])
    if summary.get("difficulty"):
        meta_parts.append("%s" % summary["difficulty"])
    if summary.get("puzzle_name") and summary.get("video"):
        meta_parts.append("video: %s" % summary["video"])
    if meta_parts:
        ax.text(0, 0.42, "  -  ".join(meta_parts),
                fontsize=11, color="#444444")

    ax.text(0, 0.10,
            "%.1f min   processed @ %.1f fps   %d one-minute segments"
            % (summary["analyzed_seconds"] / 60.0,
               summary["processing_fps"],
               len(summary.get("segments", []))),
            fontsize=10, color="#777777")

    # ---- KPI tiles
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    pct = fat.get("grips_per_min_change_pct")
    pct_txt = ("%+.1f%%" % pct) if pct is not None else "n/a"
    tiles = [
        ("%d" % pace.get("total_grip_cycles", 0), "total grips"),
        ("%.1f/min" % pace.get("combined_grips_per_min", 0),
         "combined pace"),
        ("%s  %.2fx" % (bim.get("dominant_hand", "?").upper(),
                        bim.get("dominance_ratio", 0)),
         "dominant hand"),
        ("%s  %s" % (fat.get("pace_trend", "?"), pct_txt),
         "pace trend"),
    ]
    for i, (val, lbl) in enumerate(tiles):
        x = i / 4.0
        ax.add_patch(Rectangle((x + 0.005, 0.05), 0.24, 0.92,
                               transform=ax.transAxes,
                               facecolor="#f4f5fa",
                               edgecolor="#c9cde0"))
        ax.text(x + 0.125, 0.60, val, transform=ax.transAxes,
                fontsize=14, weight="bold", ha="center")
        ax.text(x + 0.125, 0.20, lbl, transform=ax.transAxes,
                fontsize=9, ha="center", color="#666666")

    # ---- per-hand comparison
    ax = fig.add_subplot(gs[2])
    rows_data = [
        ("Grip cycles",        L.get("grip_cycles", 0),       R.get("grip_cycles", 0)),
        ("Grips / min",        L.get("grips_per_min", 0),     R.get("grips_per_min", 0)),
        ("Moving %",           L.get("moving_pct", 0),        R.get("moving_pct", 0)),
        ("Finger active %",    L.get("finger_active_pct", 0), R.get("finger_active_pct", 0)),
        ("Path length",        L.get("path_length", 0),       R.get("path_length", 0)),
        ("Mean grip hold (s)", L.get("mean_grip_hold_s") or 0,
                               R.get("mean_grip_hold_s") or 0),
    ]
    bar_h = 0.32
    first = True
    for i, (label, lv, rv) in enumerate(rows_data):
        m = max(lv, rv, 1e-6)
        ax.barh(i + bar_h / 2, lv / m, height=bar_h, color=C_LEFT,
                label="Left" if first else None)
        ax.barh(i - bar_h / 2, rv / m, height=bar_h, color=C_RIGHT,
                label="Right" if first else None)
        first = False
        ax.text(-0.02, i, label, ha="right", va="center", fontsize=10)
        ax.text(lv / m + 0.02, i + bar_h / 2,
                ("%g" % round(lv, 1)), va="center", fontsize=9, color="#333333")
        ax.text(rv / m + 0.02, i - bar_h / 2,
                ("%g" % round(rv, 1)), va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.7, len(rows_data) - 0.3)
    ax.invert_yaxis()
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Per-hand comparison (bars normalized per row)",
                 fontsize=12, weight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # ---- pace over time
    ax = fig.add_subplot(gs[3])
    segs = summary.get("segments", [])
    if segs:
        mins = [(sg["t_start"] + sg["t_end"]) / 2 / 60.0 for sg in segs]
        rg = [sg["right"]["grips_per_min"] for sg in segs]
        lg = [sg["left"]["grips_per_min"] for sg in segs]
        cg = [rg[i] + lg[i] for i in range(len(rg))]
        ax.plot(mins, lg, "-", color=C_LEFT, lw=1.5, label="Left")
        ax.plot(mins, rg, "-", color=C_RIGHT, lw=1.5, label="Right")
        ax.plot(mins, cg, "-", color="#222222", lw=1.0, alpha=0.55,
                label="Combined")
        if "grips_per_min_trendline_start" in fat:
            ax.plot([mins[0], mins[-1]],
                    [fat["grips_per_min_trendline_start"],
                     fat["grips_per_min_trendline_end"]],
                    "--", color="#aa3333", lw=1.6,
                    label="Trend (%s)" % pct_txt)
        ax.set_xlabel("Session time (minutes)")
        ax.set_ylabel("Grips / min")
        ax.set_title("Pace over the session", fontsize=12, weight="bold",
                     loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ---- sort vs assemble (uses manual board if set, otherwise auto-zone)
    if sa is not None:
        sa_gs = gs[4].subgridspec(1, 2, wspace=0.35, width_ratios=[1.1, 1.0])
        ax_b = fig.add_subplot(sa_gs[0])
        s = sa["session"]
        bar_rows = [("Right", s["right_assemble_pct"], C_RIGHT),
                    ("Left",  s["left_assemble_pct"],  C_LEFT)]
        first = True
        for i, (lbl, asm, col) in enumerate(bar_rows):
            sort = max(0.0, 100.0 - asm)
            ax_b.barh(i, asm, height=0.5, color=col,
                      label="At board" if first else None)
            ax_b.barh(i, sort, left=asm, height=0.5, color="#bbbbbb",
                      label="Sorting / away" if first else None)
            first = False
            if asm > 8:
                ax_b.text(asm / 2, i, "%.0f%%" % asm, ha="center",
                          va="center", fontsize=10, color="white",
                          weight="bold")
            if sort > 8:
                ax_b.text(asm + sort / 2, i, "%.0f%%" % sort, ha="center",
                          va="center", fontsize=10, color="white",
                          weight="bold")
            ax_b.text(-2, i, lbl, ha="right", va="center",
                      fontsize=10, weight="bold")
        ax_b.set_xlim(0, 100)
        ax_b.set_ylim(-0.7, 1.7)
        ax_b.invert_yaxis()
        ax_b.set_yticks([])
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        src_lbl = ("manual board"
                   if s["source"] == "manual" else "auto-detected")
        # manual title + subtitle so they don't stack on each other
        ax_b.text(0, 1.18, "Sort vs Assemble",
                  transform=ax_b.transAxes,
                  fontsize=12, weight="bold", va="bottom")
        ax_b.text(0, 1.03,
                  "combined %.0f%% at board  -  %s zone"
                  % (s["combined_assemble_pct"], src_lbl),
                  transform=ax_b.transAxes, fontsize=9,
                  color="#666666", style="italic", va="bottom")
        ax_b.legend(loc="upper center", fontsize=9, frameon=False,
                    ncol=2, bbox_to_anchor=(0.5, -0.15))
        for sp in ("top", "right", "left"):
            ax_b.spines[sp].set_visible(False)

        ax_e = fig.add_subplot(sa_gs[1])
        pw = sa["per_window"]
        if pw:
            xw = [w["index"] for w in pw]
            rw = [w["right_assemble_pct"] for w in pw]
            lw = [w["left_assemble_pct"]  for w in pw]
            cw = [w["combined_assemble_pct"] for w in pw]
            ax_e.plot(xw, rw, "o-", color=C_RIGHT, lw=1.5, label="Right")
            ax_e.plot(xw, lw, "o-", color=C_LEFT,  lw=1.5, label="Left")
            ax_e.plot(xw, cw, "o-", color="#222222", lw=1.0, alpha=0.55,
                      label="Either hand")
            ax_e.set_xticks(xw)
            ax_e.set_xlabel("Window")
            ax_e.set_ylabel("% at board")
            ax_e.set_ylim(0, max(100, max(cw) * 1.1))
            # manual title to match left subplot's higher placement
            ax_e.text(0, 1.18, "Per-window evolution",
                      transform=ax_e.transAxes, fontsize=11,
                      weight="bold", va="bottom")
            ax_e.legend(loc="upper center", fontsize=8, frameon=False,
                        ncol=3, bbox_to_anchor=(0.5, -0.30))
            ax_e.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax_e.spines[sp].set_visible(False)
    else:
        ax = fig.add_subplot(gs[4])
        ax.axis("off")
        ax.text(0.5, 0.5, "Sort vs Assemble: not enough grip events "
                "to derive a board zone.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888", style="italic")

    # ---- bimanual breakdown
    ax = fig.add_subplot(gs[5])
    both = float(bim.get("both_moving_pct", 0))
    par  = float(bim.get("parallel_active_pct", 0))
    idle = float(bim.get("both_idle_pct", 0))
    one  = max(0.0, 100.0 - both - par - idle)
    cats = [
        ("Both idle",         idle, "#bbbbbb"),
        ("One hand active",   one,  "#88aacc"),
        ("Parallel active",   par,  "#5577dd"),
        ("Both moving",       both, "#223388"),
    ]
    left = 0
    for name, val, col in cats:
        ax.barh(0, val, left=left, color=col, height=0.5,
                edgecolor="white", lw=1)
        if val > 5:
            ax.text(left + val / 2, 0, "%s\n%.0f%%" % (name, val),
                    ha="center", va="center", fontsize=8, color="white",
                    weight="bold")
        left += val
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Bimanual activity breakdown", fontsize=12,
                 weight="bold", loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # ---- competition metrics (renders in the whitespace band between the
    # top and bottom gridspecs; degrades to nothing for older runs)
    comp = _competition_lines(summary)
    cax = fig.add_axes([0.07, 0.475, 0.88, 0.065])
    cax.axis("off")
    cax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=cax.transAxes,
                            facecolor="#f4f5fa", edgecolor="#c9cde0"))
    cax.text(0.015, 0.86, "Competition metrics", transform=cax.transAxes,
             fontsize=11, weight="bold", va="top", color="#333333")
    if comp:
        cax.text(0.015, 0.58,
                 "\n".join(comp), transform=cax.transAxes,
                 fontsize=9, va="top", color="#444444",
                 linespacing=1.45)
    else:
        cax.text(0.015, 0.58, "n/a (older analysis - rerun to populate)",
                 transform=cax.transAxes, fontsize=9, va="top",
                 color="#888888", style="italic")

    # ---- note
    ax = fig.add_subplot(gs[6])
    ax.axis("off")
    note = fat.get("note", "")
    if note:
        ax.add_patch(Rectangle((0.0, 0.10), 1.0, 0.80, transform=ax.transAxes,
                               facecolor="#fff4dd", edgecolor="#e0b060"))
        ax.text(0.02, 0.5, "Note:  " + note, transform=ax.transAxes,
                fontsize=10, va="center", style="italic", color="#553300")
    if windows:
        with_zone = sum(1 for w in windows if w["zone"] is not None)
        ax.text(0.98, 0.02,
                "see _heatmap.png for the %d-window view "
                "(%d windows with detected active zone)"
                % (len(windows), with_zone),
                transform=ax.transAxes, ha="right", fontsize=8,
                color="#888888", style="italic")

    fig.savefig(base + "_report.pdf", bbox_inches="tight")
    fig.savefig(base + "_report.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- driver
def make_all(metrics_path, video_path=None, n_windows=N_WINDOWS_DEFAULT):
    summary, cols, base = load_data(metrics_path)
    bg = find_bg_frame(base, video_path)
    windows = compute_windows(cols, n=n_windows)
    sa = compute_sort_assemble(cols, windows, summary.get("board_region"))
    make_heatmaps(cols, windows, bg, summary.get("board_region"),
                  base + "_heatmap.png")
    make_one_pager(summary, windows, sa, base)
    return base, windows, sa


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Interval heatmaps + one-pager report from a "
                    "puzzle_hands.py metrics.json file.")
    ap.add_argument("metrics_json", help="path to <base>_metrics.json")
    ap.add_argument("--video", default=None,
                    help="path to original video, used to grab a clean "
                         "background frame for the heatmaps")
    ap.add_argument("--windows", type=int, default=N_WINDOWS_DEFAULT,
                    help="number of time windows (default %d)"
                         % N_WINDOWS_DEFAULT)
    a = ap.parse_args()
    base, _, _ = make_all(a.metrics_json, a.video, a.windows)
    print("wrote:")
    print("  " + base + "_heatmap.png")
    print("  " + base + "_report.pdf")
    print("  " + base + "_report.png")
