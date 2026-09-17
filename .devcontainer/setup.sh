#!/usr/bin/env bash
# One-time Codespace provisioning: Flutter SDK, Android SDK, Python deps.
# Runs once when the codespace/container is first created (onCreateCommand).
set -euo pipefail

echo "== System packages =="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  curl unzip xz-utils zip git clang cmake ninja-build pkg-config \
  libgtk-3-dev liblzma-dev libglu1-mesa file

echo "== Flutter SDK (stable, pinned to match repo's 3.47.2) =="
if [ ! -d /opt/flutter ]; then
  sudo git clone --branch 3.47.2 --depth 1 https://github.com/flutter/flutter.git /opt/flutter
  sudo chown -R "$(whoami)" /opt/flutter
fi
export PATH="$PATH:/opt/flutter/bin"
flutter config --no-analytics
flutter precache --android
yes | flutter doctor --android-licenses || true

echo "== Android SDK (cmdline-tools + platform for this repo's compileSdk) =="
ANDROID_SDK_ROOT=/opt/android-sdk
sudo mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
sudo chown -R "$(whoami)" "$ANDROID_SDK_ROOT"
if [ ! -d "$ANDROID_SDK_ROOT/cmdline-tools/latest" ]; then
  curl -sSL -o /tmp/cmdline-tools.zip \
    https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
  unzip -q /tmp/cmdline-tools.zip -d /tmp/cmdline-tools
  mv /tmp/cmdline-tools/cmdline-tools "$ANDROID_SDK_ROOT/cmdline-tools/latest"
  rm /tmp/cmdline-tools.zip
fi
export ANDROID_SDK_ROOT
export PATH="$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools"
yes | sdkmanager --licenses >/dev/null || true
sdkmanager --install "platform-tools" "platforms;android-35" "build-tools;35.0.0" >/dev/null

echo "== Repo Python deps (backend + laptop_monitor) =="
python3 -m pip install --user -r voice_guard/backend/requirements.txt
[ -f vaani/requirements.txt ] && python3 -m pip install --user -r vaani/requirements.txt || true

echo "== Flutter deps =="
(cd voice_guard && flutter pub get)

echo "== flutter doctor summary =="
flutter doctor -v || true

echo "Setup complete. Open a new terminal to pick up PATH, or: source ~/.bashrc"
