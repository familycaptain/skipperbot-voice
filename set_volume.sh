#!/usr/bin/env bash
# set_volume.sh — pin + test the speaker output volume for the voice satellite.
#
# Why this exists (skipperbot-voice issue #1, "Volume resets to lowest level"):
# the EMEET conference speakerphone (and some USB speakers) reset their hardware
# mixer to the LOWEST level on USB re-enumeration — e.g. each time the Docker
# container restarts. Nothing in the app sets device volume, so the fix is to
# pin the ALSA mixer to a known level. This script lets you test that by hand on
# the Pi: if pinning the volume here keeps the speaker audible, we wire the same
# call into service startup.
#
# Config (read from .env in this dir, or the environment):
#   VOICE_OUTPUT_VOLUME   target percent 0-100 (default 80)
#   VOICE_OUTPUT_CARD     ALSA card index or name (optional; default: auto-detect)
#   VOICE_OUTPUT_MIXER    ALSA mixer control name (optional; default: auto-detect)
#
# Usage:
#   ./set_volume.sh          # set volume to VOICE_OUTPUT_VOLUME, then play a test tone
#   ./set_volume.sh 90       # override the target to 90%
#   ./set_volume.sh --get    # print the current volume, change nothing
#   ./set_volume.sh --list   # list cards + mixer controls, change nothing
#
# Runs on the Pi host, or inside the container:
#   docker compose exec <service> ./set_volume.sh

set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }

command -v amixer >/dev/null 2>&1 || { echo "ERROR: amixer not found — install alsa-utils."; exit 1; }

# --- resolve the card: prefer an EMEET/USB playback card, else the first one ---
detect_card() {
  local hint
  hint=$(aplay -l 2>/dev/null | grep -iE "emeet|conference|usb" \
           | sed -n 's/^card \([0-9]*\):.*/\1/p' | head -1 || true)
  [ -n "$hint" ] && { echo "$hint"; return; }
  aplay -l 2>/dev/null | sed -n 's/^card \([0-9]*\):.*/\1/p' | head -1
}
CARD="${VOICE_OUTPUT_CARD:-$(detect_card)}"
[ -n "$CARD" ] || { echo "ERROR: no playback card found. Run --list and set VOICE_OUTPUT_CARD."; exit 1; }

# --- resolve the mixer control: first simple control on the card ---
detect_mixer() {
  amixer -c "$CARD" scontrols 2>/dev/null \
    | sed -n "s/^Simple mixer control '\([^']*\)'.*/\1/p" | head -1
}
MIXER="${VOICE_OUTPUT_MIXER:-$(detect_mixer)}"

case "${1:-}" in
  --list)
    echo "=== Playback cards (aplay -l) ==="; aplay -l 2>/dev/null || true
    echo; echo "=== Mixer controls on card $CARD ==="; amixer -c "$CARD" scontrols 2>/dev/null || true
    exit 0 ;;
  --get)
    [ -n "$MIXER" ] || { echo "No mixer control found on card $CARD; run --list."; exit 1; }
    amixer -c "$CARD" sget "$MIXER"; exit 0 ;;
esac

VOL="${1:-${VOICE_OUTPUT_VOLUME:-80}}"
VOL="${VOL%\%}"  # tolerate a trailing %
case "$VOL" in (*[!0-9]*|'') echo "ERROR: volume must be 0-100, got '$VOL'"; exit 1 ;; esac
[ "$VOL" -le 100 ] || { echo "ERROR: volume must be 0-100, got '$VOL'"; exit 1; }
[ -n "$MIXER" ] || { echo "ERROR: no mixer control on card $CARD. Run --list, set VOICE_OUTPUT_MIXER."; exit 1; }

echo "Setting card '$CARD' control '$MIXER' -> ${VOL}%"
amixer -c "$CARD" sset "$MIXER" "${VOL}%" unmute >/dev/null 2>&1 \
  || amixer -c "$CARD" sset "$MIXER" "${VOL}%" >/dev/null
echo "--- now ---"; amixer -c "$CARD" sget "$MIXER" | grep -iE "%|\[on\]|\[off\]" || true

# --- play a test tone so you can hear the result ---
if command -v speaker-test >/dev/null 2>&1; then
  echo "Playing a short 440Hz test tone (Ctrl-C to stop)…"
  speaker-test -D "plughw:${CARD}" -c 2 -t sine -f 440 -l 1 >/dev/null 2>&1 \
    || echo "(test tone failed — device may be busy; the volume was still set)"
else
  echo "(install alsa-utils' speaker-test to auto-play a tone; volume was set)"
fi
