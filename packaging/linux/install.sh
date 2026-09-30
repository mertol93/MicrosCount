#!/usr/bin/env bash
# Install MicrosCount for the current user (no root needed).
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/microscount"
BIN="$HOME/.local/bin"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
if [ "$SRC" != "$DEST" ]; then
  rm -rf "$DEST"
  mkdir -p "$(dirname "$DEST")"
  cp -a "$SRC" "$DEST"
fi
mkdir -p "$BIN" "$APPS" "$ICONS"
ln -sf "$DEST/MicrosCount" "$BIN/microscount"
ln -sf "$DEST/microscount-cli" "$BIN/microscount-cli"
cp "$DEST/microscount.png" "$ICONS/microscount.png"
cat > "$APPS/microscount.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MicrosCount
GenericName=Microscopy image analysis
Comment=Nuclear translocation in fluorescence images and porosity in SEM images
Exec=$DEST/MicrosCount %F
Icon=$ICONS/microscount.png
Terminal=false
Categories=Science;Biology;Education;Graphics;
StartupWMClass=MicrosCount
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" >/dev/null 2>&1 || true
echo "MicrosCount installed in $DEST"
echo "Start it from your applications menu, or run: $BIN/microscount"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "(add $BIN to your PATH to use 'microscount' in a terminal)";; esac
