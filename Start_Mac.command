#!/bin/bash
# macOS launcher for Puzzle Hand Analyzer.
# Uses a local virtualenv (./.venv) to avoid touching system Python.

cd "$(dirname "$0")"

echo "============================================"
echo "  Puzzle Hand Analyzer"
echo "============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed."
    echo "Download and install it from:"
    echo "    https://www.python.org/downloads/"
    echo
    echo "(Use the installer from python.org, not the Apple system Python -"
    echo " the python.org installer includes Tkinter, which this app needs.)"
    read -p "Press Enter to close..." _
    exit 1
fi

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    echo "Tkinter (the GUI library) is missing from your Python."
    echo "Reinstall Python from https://www.python.org/downloads/ - the"
    echo "python.org installer bundles Tkinter (the Homebrew one does not)."
    read -p "Press Enter to close..." _
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating local environment (one-time, ~30 seconds)..."
    if ! python3 -m venv .venv 2>/tmp/venv.err; then
        echo "Could not create virtual environment. Details:"
        cat /tmp/venv.err
        read -p "Press Enter to close..." _
        exit 1
    fi
fi

source .venv/bin/activate
echo "Checking dependencies (first run takes a few minutes)..."
pip install --quiet --upgrade pip
# Don't quiet this one - mediapipe is ~100 MB; without progress output the
# install looks like a hang and people force-close the launcher.
pip install -r requirements.txt || {
    echo "Dependency installation failed. See the messages above."
    read -p "Press Enter to close..." _
    exit 1
}

echo "Launching..."
python puzzle_app.py
