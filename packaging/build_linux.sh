#!/usr/bin/env bash
# Build MicrosCount for Linux: PyInstaller folder -> .tar.gz (with install.sh), .deb and AppImage.
# Usage (from the repository root, inside a Python environment with MicrosCount's dependencies):
#   bash packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from microscount import __version__; print(__version__)")
ARCH=$(uname -m)
DEB_ARCH=$([ "$ARCH" = "x86_64" ] && echo amd64 || echo "$ARCH")
OUT="$ROOT/dist/release"
echo "== MicrosCount $VERSION ($ARCH)"

rm -rf build dist/MicrosCount
python -m PyInstaller --noconfirm --clean packaging/microscount.spec --distpath dist --workpath build
APP="$ROOT/dist/MicrosCount"

echo "== smoke tests of the frozen application"
"$APP/microscount-cli" selftest --report "$ROOT/dist/selftest-linux.txt"
MICROSCOUNT_QUIT_AFTER_MS=3000 QT_QPA_PLATFORM=offscreen "$APP/MicrosCount"

mkdir -p "$OUT"
cp packaging/icons/microscount.png "$APP/microscount.png"
cp packaging/linux/install.sh packaging/linux/uninstall.sh "$APP/"
chmod +x "$APP/install.sh" "$APP/uninstall.sh"
cp README.md LICENSE THIRD_PARTY_NOTICES.md CITATION.cff "$APP/"

echo "== tar.gz"
tar -C dist -czf "$OUT/MicrosCount-$VERSION-Linux-$ARCH.tar.gz" MicrosCount

echo "== .deb"
DEB="$ROOT/build/deb"
rm -rf "$DEB"
mkdir -p "$DEB/DEBIAN" "$DEB/opt/microscount" "$DEB/usr/bin" "$DEB/usr/share/applications" \
         "$DEB/usr/share/icons/hicolor/256x256/apps" "$DEB/usr/share/doc/microscount"
cp -a "$APP/." "$DEB/opt/microscount/"
rm -f "$DEB/opt/microscount/install.sh" "$DEB/opt/microscount/uninstall.sh"
ln -s /opt/microscount/MicrosCount "$DEB/usr/bin/microscount"
ln -s /opt/microscount/microscount-cli "$DEB/usr/bin/microscount-cli"
sed "s|@EXEC@|/opt/microscount/MicrosCount|; s|@ICON@|microscount|" packaging/linux/microscount.desktop \
    > "$DEB/usr/share/applications/microscount.desktop"
python - <<'PY'
from PIL import Image
Image.open("packaging/icons/microscount.png").resize((256, 256), Image.LANCZOS).save(
    "build/deb/usr/share/icons/hicolor/256x256/apps/microscount.png")
PY
cp LICENSE "$DEB/usr/share/doc/microscount/copyright"
SIZE=$(du -sk "$DEB/opt" | cut -f1)
cat > "$DEB/DEBIAN/control" <<EOF
Package: microscount
Version: $VERSION
Section: science
Priority: optional
Architecture: $DEB_ARCH
Installed-Size: $SIZE
Maintainer: Mertol Tüfekci <https://github.com/mertol93/microscount>
Homepage: https://github.com/mertol93/microscount
Depends: libc6 (>= 2.35), libgl1, libegl1, libxkbcommon0, libfontconfig1, libdbus-1-3
Recommends: libxcb-cursor0, libxkbcommon-x11-0
Description: Microscopy image analysis with a graphical interface
 MicrosCount measures nuclear translocation (nuclear/cytoplasmic intensity
 ratio) in fluorescence images and porosity and pore-size distribution in SEM
 images. Reads TIFF, PNG and JPEG.
EOF
chmod -R u+rwX,go+rX "$DEB"
dpkg-deb --build --root-owner-group "$DEB" "$OUT/microscount_${VERSION}_${DEB_ARCH}.deb"

echo "== AppImage"
if command -v appimagetool >/dev/null 2>&1 || [ -n "${APPIMAGETOOL:-}" ]; then
  TOOL="${APPIMAGETOOL:-appimagetool}"
  APPDIR="$ROOT/build/AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/lib"
  cp -a "$APP" "$APPDIR/usr/lib/microscount"
  rm -f "$APPDIR/usr/lib/microscount/install.sh" "$APPDIR/usr/lib/microscount/uninstall.sh"
  cp packaging/linux/AppRun "$APPDIR/AppRun"
  chmod +x "$APPDIR/AppRun"
  sed "s|@EXEC@|MicrosCount|; s|@ICON@|microscount|" packaging/linux/microscount.desktop > "$APPDIR/microscount.desktop"
  cp packaging/icons/microscount.png "$APPDIR/microscount.png"
  ARCH="$ARCH" "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUT/MicrosCount-$VERSION-$ARCH.AppImage" \
    || ARCH="$ARCH" "$TOOL" "$APPDIR" "$OUT/MicrosCount-$VERSION-$ARCH.AppImage"
else
  echo "appimagetool not found: skipping the AppImage (set APPIMAGETOOL=/path/to/appimagetool)"
fi
ls -lh "$OUT"
