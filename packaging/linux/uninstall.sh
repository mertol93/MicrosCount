#!/usr/bin/env bash
# Remove a per-user MicrosCount installation made by install.sh.
set -euo pipefail
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -rf "$DATA/microscount"
rm -f "$HOME/.local/bin/microscount" "$HOME/.local/bin/microscount-cli"
rm -f "$DATA/applications/microscount.desktop" "$DATA/icons/hicolor/256x256/apps/microscount.png"
echo "MicrosCount removed."
