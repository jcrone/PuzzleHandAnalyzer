#!/bin/bash
# Linux launcher for Puzzle Hand Analyzer.
# Uses a local virtualenv (./.venv) to avoid PEP 668 issues on modern distros.

cd "$(dirname "$0")"

echo "============================================"
echo "  Puzzle Hand Analyzer"
echo "============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Install it via your distro's package"
    echo "manager, then run this launcher again."
    read -p "Press Enter to close..." _
    exit 1
fi

# Distro-specific install hint for missing tkinter / venv
HINT="install your distro's python3-tk and python3-venv packages"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
        ubuntu|debian|linuxmint|pop|elementary)
            HINT="sudo apt install -y python3-tk python3-venv";;
        fedora|rhel|centos|rocky|alma)
            HINT="sudo dnf install -y python3-tkinter python3-virtualenv";;
        arch|manjaro|endeavouros|garuda)
            HINT="sudo pacman -S --needed tk python-virtualenv";;
        opensuse*|sles|suse)
            HINT="sudo zypper install -y python3-tk python3-virtualenv";;
    esac
fi

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    echo "Tkinter (the GUI library) is not installed."
    echo "On your system, run:"
    echo
    echo "    $HINT"
    echo
    echo "...then run this launcher again."
    read -p "Press Enter to close..." _
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating local environment (one-time, ~30 seconds)..."
    if ! python3 -m venv .venv 2>/tmp/venv.err; then
        echo "Could not create virtual environment. On your system, run:"
        echo "    $HINT"
        echo "Details:"
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
    echo "Dependency installation failed. See messages above."
    read -p "Press Enter to close..." _
    exit 1
}

echo "Launching..."
python puzzle_app.py
