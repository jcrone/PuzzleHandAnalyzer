# CLAUDE.md — instructions for Claude when working on this project

## What this project is

**Puzzle Hand Analyzer** — a computer-vision tool that turns overhead speed-
puzzling video into per-hand metrics, sort-vs-assemble zone analysis,
six-window heatmaps with auto-detected active zones, and a single-page PDF
report. Distributed as a launcher-script zip via GitHub Releases.

GitHub: https://github.com/jcrone/PuzzleHandAnalyzer (public)

## Files

- `puzzle_app.py` — Tkinter desktop GUI (the main entry point for end users)
- `puzzle_hands.py` — the analyzer (MediaPipe + OpenCV); also a CLI
- `puzzle_report.py` — generates the 6-window heatmaps + one-pager PDF/PNG
- `requirements.txt` — Python deps (mediapipe, opencv-python-headless, numpy, matplotlib)
- `Start_Windows.bat` / `Start_Mac.command` / `Start_Linux.sh` — double-click launchers
- `build_release.sh` — packages the bundle into `dist/PuzzleAnalyzer.zip`
- `HOW_TO_USE.md` / `HOW_TO_USE.pdf` / `README.md` — user docs

## Things NOT to commit

`.gitignore` already covers these — never add them back:

- Source videos (`*.mp4`, `*.mov`, etc.)
- Per-puzzle analysis output folders (`*_analysis/`, `puzzle_output/`)
- `dist/` — built release artifacts (recreated by `build_release.sh`)
- `.venv/`, `__pycache__/`, `.claude/`

---

## RELEASE WORKFLOW — run after every push to `main`

When the user pushes user-visible changes to `main`, follow this procedure
to cut a new release. The end goal: the `Latest` release at
`https://github.com/jcrone/PuzzleHandAnalyzer/releases/latest` always
points at a `PuzzleAnalyzer.zip` containing the just-pushed code.

### Step 1 — pick the next version

```bash
git tag --sort=-v:refname | head -1   # current latest tag, e.g. v0.1.0
```

Choose the next version using semver:

- Bug fix only → patch bump (`v0.1.0` → `v0.1.1`)
- New user-visible feature → minor bump (`v0.1.0` → `v0.2.0`)
- Breaking change or major milestone → major bump (`v0.1.0` → `v1.0.0`)

If the change is ambiguous, ask the user which bump they want before tagging.

### Step 2 — rebuild the bundle

```bash
./build_release.sh
```

This produces `dist/PuzzleAnalyzer.zip` from the current working tree. The
script lists what's inside when it finishes — sanity-check that all the
expected files are there.

### Step 3 — tag and push the tag

```bash
git tag -a v0.X.Y -m "v0.X.Y - one-line summary"
git push origin v0.X.Y
```

### Step 4 — create the GitHub Release with the zip attached

```bash
gh release create v0.X.Y dist/PuzzleAnalyzer.zip \
    --title "Puzzle Hand Analyzer v0.X.Y" \
    --notes "$(cat <<'EOF'
## What's new in v0.X.Y

- <bullet for each user-visible change since the last release>

## Install

1. Download PuzzleAnalyzer.zip below, extract it.
2. Double-click Start_Windows.bat / Start_Mac.command / Start_Linux.sh for your OS.
3. The first launch installs Python deps into a local .venv (one-time, ~2 min).

Requires Python 3.10–3.12 (3.8/3.9 also work). See HOW_TO_USE.md for details.
EOF
)"
```

For the bullet list of what's new: pull from the commits since the previous
tag (`git log v0.X.(Y-1)..HEAD --oneline`) and translate to user-facing
language. Skip internal refactors and tooling-only changes — focus on
things the puzzler will notice.

### Step 5 — report the release URL to the user

The download link the user shares with friends:

```
https://github.com/jcrone/PuzzleHandAnalyzer/releases/latest
```

That always resolves to whichever release is newest, so the link doesn't
need to be updated each time.

---

## Other useful commands

- **Run the GUI**: `python3 puzzle_app.py`
- **Run the CLI analyzer**: `python3 puzzle_hands.py VIDEO.mp4 [--pieces N --difficulty Hard ...]`
- **Regenerate report for an existing run**: `python3 puzzle_report.py path/to/X_metrics.json --video original.mp4`

## Editing rules for this project

- Match the existing code style: short comments only when the *why* is non-obvious
- Don't introduce new dependencies unless asked — the install footprint matters
  for end users
- The Tkinter GUI must stay screen-friendly: scrollable container, no widget
  off-screen on small laptops
