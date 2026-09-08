#!/usr/bin/env bash
# Builds the "privileged" flavor release APK and assembles it with the rest
# of this directory into an installable Magisk module zip.
#
# The built APK is never committed to git (binaries don't belong in the
# repo, and this module must never be built from anything but a fresh local
# build — see README.md's "why the privileged flavor can't be pre-built and
# shipped" note). This script is the only supported way to produce the zip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APK_DEST="$SCRIPT_DIR/system/priv-app/VoiceGuard/VoiceGuard.apk"
OUT_ZIP="$SCRIPT_DIR/voiceguard-privileged-capture.zip"

echo "==> Building privileged flavor release APK"
(cd "$REPO_ROOT" && flutter build apk --flavor privileged --release)

BUILT_APK="$REPO_ROOT/build/app/outputs/flutter-apk/app-privileged-release.apk"
if [ ! -f "$BUILT_APK" ]; then
  echo "!! Expected build output not found at $BUILT_APK" >&2
  echo "!! Check the actual output path printed above and adjust this script." >&2
  exit 1
fi

echo "==> Staging APK into module payload"
cp "$BUILT_APK" "$APK_DEST"

echo "==> Packaging module zip"
rm -f "$OUT_ZIP"
(cd "$SCRIPT_DIR" && zip -r "$OUT_ZIP" \
  module.prop customize.sh system \
  -x "*.gitkeep")

echo "==> Done: $OUT_ZIP"
echo "    Install via Magisk Manager > Modules > Install from storage, then reboot."
