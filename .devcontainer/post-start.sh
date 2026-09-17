#!/usr/bin/env bash
# Lightweight, runs on every codespace start/resume (postStartCommand).
set -uo pipefail

echo "== git identity check =="
git config --get user.email >/dev/null 2>&1 || echo "note: run 'git config --global user.email/user.name' if not already set via your GitHub account defaults."

echo "== quick status =="
git -C /workspaces/* status --short 2>/dev/null | head -20 || true

echo "Codespace ready. Flutter: /opt/flutter/bin, Android SDK: /opt/android-sdk."
