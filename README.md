# Puzzle Hand Analyzer

A tool that watches a speed-puzzling video and measures how the puzzler's
hands move: left vs. right activity, finger work, grip/piece-handling
pace, rhythm, fatigue over the session, and time spent sorting vs.
assembling.

Everything runs on your own computer. Your videos are never uploaded
anywhere.

---

## FOR THE PUZZLER (easy, no typing)

### One-time setup (about 5 minutes, done once)

1. Install Python (the engine this tool runs on):
   - **Windows / Mac:** download from https://www.python.org/downloads/
     and run the installer.
   - **Windows users:** on the first installer screen, tick the box
     **"Add Python to PATH"** before clicking Install. This matters.
   - **Linux:** Python 3 is usually pre-installed. The launcher will
     prompt you to install tkinter (and venv) from your package manager
     if they're missing - it tells you the exact command.
2. Keep all the files in this folder together. Don't separate them.

### Every time you want to analyze a video

1. Open the tool:
   - **Windows:** double-click **Start_Windows.bat**
   - **Mac:** double-click **Start_Mac.command**
     (If Mac says it's from an unidentified developer: right-click the
     file, choose Open, then Open again. Only needed the first time.)
   - **Linux:** in a terminal in this folder, run `./Start_Linux.sh`
     (or double-click it if your file manager is configured to execute
     shell scripts). If the launcher reports that tkinter is missing,
     it prints the exact one-line install command for your distro
     (e.g. `sudo apt install python3-tk python3-venv` on Ubuntu/Debian).
     Run that, then start the launcher again.
   - The first launch installs a few things and is slow. Later launches
     are quick. A small window will open.

2. **Choose a video** - click "Choose video file..." and pick your
   puzzle video.

3. **(Optional) Mark the board** - in the preview picture, **click and
   drag** to draw a box around the puzzle board. Then adjust it:
   - **Drag any corner** to resize the box
   - **Drag the middle** of the box to move it around
   - **Drag outside** to draw a fresh box (replaces the old one)
   - Click **Clear board** to start over

   This unlocks "sorting vs. assembling" stats. If you're not sure, just
   skip it.

4. **Options:**
   - "Left/Right labels look reversed" - leave unticked for now. If the
     results come out with the hands swapped, run it again with this
     ticked.
   - "Annotated review video" - makes a video with the hand tracking
     drawn on, so you can check the analysis looks right. Default is the
     **first 2 minutes** (a quick sanity check, small file). Pick a
     longer length, **Full video**, or **Don't create** as you prefer.
   - "Faster processing" - quicker, slightly less precise.

5. Click **Analyze video**. A long video can take 10-20 minutes. You can
   leave the window open and come back. Progress shows in the box.

6. When it finishes, it offers to open the **results folder**. The
   results are saved next to your video, in a folder ending in
   `_analysis`.

### What you get

- `..._report.pdf` / `..._report.png` - **start here**: a single-page
  session summary with the headline numbers, per-hand comparison, pace
  over time, and bimanual breakdown.
- `..._heatmap.png` - 6 mini heatmaps showing where the hands worked in
  each sixth of the session. A green box on each panel marks the
  auto-detected "active zone" - where pieces were actually picked up
  or placed in that window - so you can see how it tightens as the
  puzzle gets assembled.
- `..._timeline.png` - second-by-second chart of hand movement, finger
  work, and individual grip cycles across the whole session.
- `..._session.png` - per-segment pace, rhythm, and zone time.
- `..._metrics.json` - all the numbers as a single file (send to
  whoever is reviewing the data).
- `..._calibration.jpg` - a single frame with the hands labeled. Check
  this if you ever doubt Left/Right.
- `..._annotated.mp4` - the review video (length set in Options).
- `..._highlight.mp4` - a short (~45s) sped-up highlight reel: each hand
  drawn as a glowing trail over the board. Small and easy to share.
- `..._perframe.csv` - the raw data, openable in Excel.

### If the Left and Right hands are swapped

Open `..._calibration.jpg`. If the labels are on the wrong hands, run the
analysis again with "Left/Right labels look reversed" ticked. Once you
know the right setting for your camera, it stays the same for every
video from that setup.

---

## FOR ADVANCED USERS (command line)

The engine is `puzzle_hands.py` and can be run directly:

    python puzzle_hands.py VIDEO.mp4
    python puzzle_hands.py VIDEO.mp4 --board 0.42,0.20,0.92,0.78
    python puzzle_hands.py VIDEO.mp4 --segment-seconds 60 --no-video
    python puzzle_hands.py VIDEO.mp4 --fps 8          # faster
    python puzzle_hands.py VIDEO.mp4 --swap-hands

Run `python puzzle_hands.py --help` for all options. The board rectangle
is in 0-1 coordinates - read it off the grid in `..._calibration.jpg`.

---

## TROUBLESHOOTING

- **"Python is not installed"** - install it from python.org (Windows:
  tick "Add Python to PATH"), then run the launcher again.
- **The launcher window flashes and closes** - open a terminal /
  command prompt in this folder and run the launcher from there to see
  the error message.
- **It's very slow** - that's normal for long videos. Tick "Faster
  processing", or analyze a shorter clip first.
- **Hands swapped** - see the section above.
- **Best results** come from a steady overhead camera with both hands
  and the table clearly visible.

## REQUIREMENTS

Python 3.10-3.12 recommended. The launcher installs the rest
(mediapipe, opencv, numpy, matplotlib) automatically on first run.
