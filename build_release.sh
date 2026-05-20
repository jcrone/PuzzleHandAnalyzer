#!/bin/bash
# Builds dist/PuzzleAnalyzer.zip - the downloadable bundle for end users.
# Includes the Python source, the launcher scripts for each OS, the docs,
# and requirements.txt. Does NOT include any analysis outputs or videos.

set -e
cd "$(dirname "$0")"

NAME="PuzzleAnalyzer"
OUT_DIR="dist"
ZIP_PATH="${OUT_DIR}/${NAME}.zip"

FILES=(
    puzzle_app.py
    puzzle_hands.py
    puzzle_report.py
    requirements.txt
    hand_landmarker.task
    HOW_TO_USE.md
    README.md
    Start_Windows.bat
    Start_Mac.command
    Start_Linux.sh
)
[ -f HOW_TO_USE.pdf ] && FILES+=(HOW_TO_USE.pdf)

mkdir -p "${OUT_DIR}"
rm -f "${ZIP_PATH}"

# Stage everything under a single top-level folder so the zip extracts cleanly
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "${STAGE}/${NAME}"

for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "missing required file: $f" >&2
        exit 1
    fi
    cp "$f" "${STAGE}/${NAME}/"
done

# Make the unix launchers executable inside the zip
chmod +x "${STAGE}/${NAME}/Start_Mac.command" "${STAGE}/${NAME}/Start_Linux.sh"

(cd "${STAGE}" && zip -qr "${OLDPWD}/${ZIP_PATH}" "${NAME}")

echo "Built ${ZIP_PATH}"
unzip -l "${ZIP_PATH}" | tail -n +2
