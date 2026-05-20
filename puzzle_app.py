#!/usr/bin/env python3
"""
puzzle_app.py - simple desktop app for the puzzle hand analyzer.

Double-click launcher (Start_Windows.bat / Start_Mac.command) opens this.
Pick a video, optionally draw and adjust the puzzle board, press Analyze.
Results are written to a folder next to the video.

This is a thin GUI over puzzle_hands.py - that file does the real work.
"""

import os
import sys
import queue
import threading
import traceback
import subprocess

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    import cv2
    import puzzle_hands
except Exception as exc:  # pragma: no cover
    print("Startup error:", exc)
    print("Run Start_Windows.bat / Start_Mac.command first - it installs "
          "what is needed.")
    sys.exit(1)

PREVIEW_W = 420
HANDLE_R = 9            # corner-handle grab radius
MIN_BOX = 12            # ignore boxes smaller than this (likely an accident)

PREVIEW_CHOICES = ("First 2 minutes", "First 5 minutes",
                   "First 10 minutes", "Full video", "Don't create")
_PREVIEW_MAP = {
    "First 2 minutes":  (True, 120),
    "First 5 minutes":  (True, 300),
    "First 10 minutes": (True, 600),
    "Full video":       (True, 0),
    "Don't create":     (False, 0),
}

DIFFICULTY_CHOICES = ("(not set)", "Easy", "Medium", "Hard", "Expert")


def _sanitize_folder_name(name):
    """Turn a free-text puzzle name into something safe for a folder name."""
    keep = []
    for c in name.strip():
        if c.isalnum() or c in "-_":
            keep.append(c)
        elif c.isspace():
            keep.append("_")
        # anything else (slash, colon, quotes, etc) gets dropped
    out = "".join(keep).strip("_")
    return out or "puzzle"


def _unique_outdir(base_path):
    """Return base_path if unused; else append _2, _3, ... until unused."""
    if not os.path.exists(base_path):
        return base_path
    n = 2
    while True:
        candidate = "%s_%d" % (base_path, n)
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _fmt_time(s):
    s = int(s)
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm %02ds" % (s // 60, s % 60)
    return "%dh %02dm" % (s // 3600, (s % 3600) // 60)


def open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)            # noqa
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass


class QueueWriter:
    """Forwards analyzer print() output into the GUI log (drops the JSON)."""
    def __init__(self, q):
        self.q = q
        self.in_json = False

    def write(self, s):
        if not s.strip():
            return
        if s.strip().startswith("{"):
            self.in_json = True
        if not self.in_json:
            self.q.put(s.rstrip())

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("Puzzle Hand Analyzer")
        root.geometry("520x720")
        root.minsize(480, 420)

        # scrollable container so the window can be smaller than the contents
        self.scroll_canvas = tk.Canvas(root, borderwidth=0,
                                       highlightthickness=0)
        self.scrollbar = tk.Scrollbar(root, orient="vertical",
                                      command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.scroll_canvas)
        self._win_id = self.scroll_canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

        def _on_inner_configure(_event):
            self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all"))
        self.inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            self.scroll_canvas.itemconfig(self._win_id, width=event.width)
        self.scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if getattr(event, "delta", 0):
                self.scroll_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self.scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.scroll_canvas.yview_scroll(1, "units")
        self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", _on_mousewheel)
        self.video = None
        self.disp = (0, 0)              # displayed preview size
        self.preview_img = None
        self.box = None                 # (x1,y1,x2,y2) canvas pixels, sorted
        self.board = None               # normalized 0-1 tuple
        self.drag_mode = None
        self.drag_offset = (0, 0)
        self.create_start = None
        self.dragged = False
        self.q = queue.Queue()

        pad = {"padx": 12, "pady": 4}
        tk.Label(self.inner, text="Puzzle Hand Analyzer",
                 font=("Helvetica", 16, "bold")).pack(**pad)

        tk.Label(self.inner, text="1.  Choose a puzzle video",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(
                     fill="x", **pad)
        self.btn_video = tk.Button(self.inner, text="Choose video file...",
                                   command=self.choose_video)
        self.btn_video.pack(**pad)
        self.lbl_file = tk.Label(self.inner, text="(no video selected)", fg="gray")
        self.lbl_file.pack(**pad)

        tk.Label(self.inner, text="2.  (Optional) Mark the puzzle board",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(
                     fill="x", **pad)
        tk.Label(self.inner, text="Click and drag to draw a box around the puzzle "
                 "board.\nDrag a corner to resize, drag the middle to move. "
                 "Skip if unsure.",
                 fg="gray", justify="left").pack(**pad)
        self.canvas = tk.Canvas(self.inner, width=PREVIEW_W, height=200,
                                bg="#222", highlightthickness=1,
                                highlightbackground="#999", cursor="crosshair")
        self.canvas.pack(**pad)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.lbl_board = tk.Label(self.inner, text="Board: (not set)", fg="gray",
                                  font=("Courier", 9))
        self.lbl_board.pack(**pad)
        self.btn_clear = tk.Button(self.inner, text="Clear board",
                                   command=self.clear_board, state="disabled")
        self.btn_clear.pack(**pad)

        # step 3 - puzzle info (name required; pieces/difficulty optional)
        tk.Label(self.inner, text="3.  Puzzle info  (name required)",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(
                     fill="x", **pad)
        info_frame = tk.Frame(self.inner)
        info_frame.pack(anchor="w", padx=24, pady=2)
        self.puzzle_name = tk.StringVar()
        self.num_pieces = tk.StringVar()
        self.difficulty = tk.StringVar(value=DIFFICULTY_CHOICES[0])
        tk.Label(info_frame, text="Name:", width=10, anchor="w").grid(
            row=0, column=0, sticky="w", pady=2)
        tk.Entry(info_frame, textvariable=self.puzzle_name, width=40).grid(
            row=0, column=1, sticky="w", pady=2)
        tk.Label(info_frame, text="Pieces:", width=10, anchor="w").grid(
            row=1, column=0, sticky="w", pady=2)
        tk.Entry(info_frame, textvariable=self.num_pieces, width=12).grid(
            row=1, column=1, sticky="w", pady=2)
        tk.Label(info_frame, text="Difficulty:", width=10, anchor="w").grid(
            row=2, column=0, sticky="w", pady=2)
        tk.OptionMenu(info_frame, self.difficulty,
                      *DIFFICULTY_CHOICES).grid(row=2, column=1,
                                                sticky="w", pady=2)

        tk.Label(self.inner, text="4.  Options",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(
                     fill="x", **pad)
        self.swap = tk.BooleanVar(value=False)
        self.preview_choice = tk.StringVar(value=PREVIEW_CHOICES[0])
        self.faster = tk.BooleanVar(value=False)
        tk.Checkbutton(self.inner, text="Left/Right labels look reversed - swap them",
                       variable=self.swap).pack(anchor="w", padx=24)
        preview_row = tk.Frame(self.inner)
        preview_row.pack(anchor="w", padx=24, pady=2)
        tk.Label(preview_row, text="Annotated review video:").pack(side="left")
        tk.OptionMenu(preview_row, self.preview_choice,
                      *PREVIEW_CHOICES).pack(side="left", padx=6)
        tk.Checkbutton(self.inner, text="Faster processing (slightly less precise)",
                       variable=self.faster).pack(anchor="w", padx=24)

        tk.Label(self.inner, text="5.  Run",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(
                     fill="x", **pad)
        self.btn_run = tk.Button(self.inner, text="Analyze video",
                                 font=("Helvetica", 12, "bold"),
                                 command=self.run_analysis, state="disabled",
                                 bg="#3c5af0", fg="white")
        self.btn_run.pack(**pad)

        self.log = scrolledtext.ScrolledText(self.inner, height=8, width=74,
                                             state="disabled",
                                             font=("Courier", 9))
        self.log.pack(**pad)

    # ---- video selection / preview ----------------------------------------
    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Choose a puzzle video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                       ("All files", "*.*")])
        if not path:
            return
        self.video = path
        self.lbl_file.config(text=os.path.basename(path), fg="black")
        self.clear_board()
        self.load_preview()
        # pre-fill the puzzle name from the video filename if it's empty,
        # so the required-name validation passes by default
        if not self.puzzle_name.get().strip():
            self.puzzle_name.set(os.path.splitext(os.path.basename(path))[0])
        # reset Run button in case we were in the post-success "Open Results" state
        self.btn_run.config(state="normal", text="Analyze video",
                            bg="#3c5af0", command=self.run_analysis)

    def load_preview(self):
        cap = cv2.VideoCapture(self.video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.1))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Error", "Could not read this video file.")
            return
        h, w = frame.shape[:2]
        dh = int(PREVIEW_W * h / w)
        frame = cv2.resize(frame, (PREVIEW_W, dh))
        self.disp = (PREVIEW_W, dh)
        tmp = os.path.join(os.path.dirname(self.video), ".puzzle_preview.png")
        cv2.imwrite(tmp, frame)
        self.preview_img = tk.PhotoImage(file=tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
        self.canvas.config(width=PREVIEW_W, height=dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.preview_img)
        self.btn_clear.config(state="normal")

    # ---- board interaction ------------------------------------------------
    def _near_corner(self, x, y):
        if not self.box:
            return None
        x1, y1, x2, y2 = self.box
        corners = {"nw": (x1, y1), "ne": (x2, y1),
                   "sw": (x1, y2), "se": (x2, y2)}
        for tag, (cx, cy) in corners.items():
            if abs(x - cx) <= HANDLE_R and abs(y - cy) <= HANDLE_R:
                return tag
        return None

    def _inside_box(self, x, y):
        if not self.box:
            return False
        x1, y1, x2, y2 = self.box
        return x1 < x < x2 and y1 < y < y2

    def _clamp(self, x, y):
        dw, dh = self.disp
        return max(0, min(x, dw)), max(0, min(y, dh))

    def on_press(self, ev):
        if not self.video:
            return
        x, y = ev.x, ev.y
        self.dragged = False
        if self.box:
            c = self._near_corner(x, y)
            if c:
                self.drag_mode = "resize-" + c
                return
            if self._inside_box(x, y):
                self.drag_mode = "move"
                x1, y1, _, _ = self.box
                self.drag_offset = (x - x1, y - y1)
                return
        self.drag_mode = "create"
        self.create_start = (x, y)

    def on_drag(self, ev):
        if not self.drag_mode:
            return
        x, y = self._clamp(ev.x, ev.y)
        if self.drag_mode == "create":
            sx, sy = self.create_start
            if abs(x - sx) > 3 or abs(y - sy) > 3:
                self.dragged = True
                self.box = (sx, sy, x, y)
        elif self.drag_mode == "move" and self.box:
            ox, oy = self.drag_offset
            x1, y1, x2, y2 = self.box
            w, h = x2 - x1, y2 - y1
            dw, dh = self.disp
            nx = max(0, min(dw - w, x - ox))
            ny = max(0, min(dh - h, y - oy))
            self.box = (nx, ny, nx + w, ny + h)
            self.dragged = True
        elif self.drag_mode.startswith("resize-") and self.box:
            x1, y1, x2, y2 = self.box
            c = self.drag_mode.split("-", 1)[1]
            if c == "nw":
                self.box = (x, y, x2, y2)
            elif c == "ne":
                self.box = (x1, y, x, y2)
            elif c == "sw":
                self.box = (x, y1, x2, y)
            elif c == "se":
                self.box = (x1, y1, x, y)
            self.dragged = True
        self._draw_box()

    def on_release(self, ev):
        if not self.drag_mode:
            return
        # a click that never moved leaves any existing box untouched
        if self.drag_mode == "create" and not self.dragged:
            self.drag_mode = None
            return
        if self.box:
            x1, y1, x2, y2 = self.box
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            if (x2 - x1) < MIN_BOX or (y2 - y1) < MIN_BOX:
                self.box = None
            else:
                self.box = (x1, y1, x2, y2)
        self.drag_mode = None
        self._update_board()
        self._draw_box()

    def on_hover(self, ev):
        if self.drag_mode:
            return
        x, y = ev.x, ev.y
        if self.box:
            c = self._near_corner(x, y)
            if c:
                cur = "size_nw_se" if c in ("nw", "se") else "size_ne_sw"
            elif self._inside_box(x, y):
                cur = "fleur"
            else:
                cur = "crosshair"
        else:
            cur = "crosshair"
        try:
            self.canvas.config(cursor=cur)
        except tk.TclError:
            self.canvas.config(cursor="crosshair")     # fallback if OS picky

    def _draw_box(self):
        self.canvas.delete("board")
        if not self.box:
            return
        x1, y1, x2, y2 = self.box
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime",
                                     width=2, tags="board")
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            self.canvas.create_rectangle(cx - 5, cy - 5, cx + 5, cy + 5,
                                         fill="lime", outline="white",
                                         width=1, tags="board")

    def _update_board(self):
        if not self.box:
            self.board = None
            self.lbl_board.config(text="Board: (not set)", fg="gray")
            return
        dw, dh = self.disp
        x1, y1, x2, y2 = self.box
        self.board = (round(x1 / dw, 3), round(y1 / dh, 3),
                      round(x2 / dw, 3), round(y2 / dh, 3))
        self.lbl_board.config(
            text="Board: (%.2f, %.2f) - (%.2f, %.2f)" % self.board,
            fg="black")

    def clear_board(self):
        self.box = None
        self.board = None
        self.dragged = False
        self.drag_mode = None
        self.canvas.delete("board")
        self.lbl_board.config(text="Board: (not set)", fg="gray")

    # ---- run --------------------------------------------------------------
    def run_analysis(self):
        if not self.video:
            return
        puzzle_name = self.puzzle_name.get().strip()
        if not puzzle_name:
            messagebox.showwarning(
                "Puzzle name required",
                "Please enter a name for this puzzle in step 3.\n"
                "It is used to name the results folder.")
            return
        safe = _sanitize_folder_name(puzzle_name)
        desired = os.path.join(os.path.dirname(self.video),
                               safe + "_analysis")
        self.outdir = _unique_outdir(desired)
        self.btn_run.config(state="disabled", text="Analyzing...")
        self.btn_video.config(state="disabled")
        self._log("Starting analysis - a long video can take 10-20 minutes.")
        if self.outdir != desired:
            self._log("(folder %r already exists - using %r instead)"
                      % (os.path.basename(desired),
                         os.path.basename(self.outdir)))
        self._log("Results will be saved to: " + self.outdir)
        self._log("You can leave this window open and come back.\n")
        threading.Thread(target=self.worker, daemon=True).start()
        self.root.after(200, self.poll)

    def worker(self):
        old = sys.stdout
        sys.stdout = QueueWriter(self.q)
        try:
            want_video, preview_seconds = _PREVIEW_MAP[
                self.preview_choice.get()]
            pname = self.puzzle_name.get().strip() or None
            try:
                ppieces = int(self.num_pieces.get().strip())
            except ValueError:
                ppieces = None
            pdiff = self.difficulty.get()
            if pdiff == DIFFICULTY_CHOICES[0]:
                pdiff = None
            puzzle_hands.analyze(
                self.video, 0, None,
                8 if self.faster.get() else 15,
                0.35, 2.5, 0.5, self.board, 60, preview_seconds,
                self.swap.get(), want_video, self.outdir,
                puzzle_name=pname, num_pieces=ppieces,
                difficulty=pdiff)
            self.q.put("__DONE__")
        except Exception:
            self.q.put("__ERROR__" + traceback.format_exc())
        finally:
            sys.stdout = old

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg == "__DONE__":
                    self._finish(True)
                    return
                if msg.startswith("__ERROR__"):
                    self._log("\nSomething went wrong:\n" + msg[9:])
                    self._finish(False)
                    return
                if msg.startswith("__PROGRESS__"):
                    self._on_progress(msg[len("__PROGRESS__"):].strip())
                    continue
                if msg.startswith("__STAGE__"):
                    self._on_stage(msg[len("__STAGE__"):].strip())
                    continue
                self._log(msg)
        except queue.Empty:
            pass
        self.root.after(200, self.poll)

    def _on_progress(self, s):
        try:
            parts = s.split()
            frac = float(parts[0])
            eta_s = int(float(parts[1]))
        except (ValueError, IndexError):
            return
        pct = int(frac * 100)
        if eta_s > 0:
            text = "Analyzing %d%%  -  ~%s left" % (pct, _fmt_time(eta_s))
        else:
            text = "Analyzing %d%%" % pct
        self.btn_run.config(text=text)

    def _on_stage(self, s):
        if s:
            self.btn_run.config(text=s + "...")

    def _finish(self, ok):
        self.btn_video.config(state="normal")
        if ok:
            self._log("\nDone. Results saved to:\n" + self.outdir)
            self.btn_run.config(state="normal",
                                text="✓  Done!  Open Results Folder",
                                bg="#2c8a3a",
                                command=self._open_results)
        else:
            self.btn_run.config(state="normal", text="Analyze video",
                                bg="#3c5af0", command=self.run_analysis)
            messagebox.showerror("Error", "Analysis failed - see the log.")

    def _open_results(self):
        open_folder(self.outdir)

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()