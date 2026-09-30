#!/usr/bin/env bash
# Build MicrosCount.app and a drag-to-Applications DMG on macOS.
# Usage (from the repository root, inside a Python environment with MicrosCount's dependencies):
#   bash packaging/build_macos.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from microscount import __version__; print(__version__)")
ARCH=$(uname -m)   # arm64 (Apple silicon) or x86_64 (Intel)
OUT="$ROOT/dist/release"
echo "== MicrosCount $VERSION (macOS $ARCH)"

rm -rf build dist/MicrosCount dist/MicrosCount.app
python -m PyInstaller --noconfirm --clean packaging/microscount.spec --distpath dist --workpath build
APP="$ROOT/dist/MicrosCount.app"

echo "== smoke tests of the frozen application"
"$APP/Contents/MacOS/microscount-cli" selftest --report "$ROOT/dist/selftest-macos-$ARCH.txt"
MICROSCOUNT_QUIT_AFTER_MS=3000 QT_QPA_PLATFORM=offscreen "$APP/Contents/MacOS/MicrosCount"

echo "== ad-hoc signature (the app is not notarised)"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" || echo "warning: strict signature verification reported issues (ad-hoc signed build)"

echo "== DMG"
mkdir -p "$OUT"
STAGE="$ROOT/build/dmg"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp README.md "$STAGE/README.md"
cat > "$STAGE/First launch - read me.txt" <<'EOF'
MicrosCount is not notarised by Apple. After dragging it to Applications:
  right-click (or Control-click) MicrosCount and choose Open, then Open again.
If macOS still refuses: System Settings > Privacy & Security > "Open Anyway".
EOF
DMG="$OUT/MicrosCount-$VERSION-macOS-$ARCH.dmg"
rm -f "$DMG"
hdiutil create -volname "MicrosCount $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
ls -lh "$OUT"
