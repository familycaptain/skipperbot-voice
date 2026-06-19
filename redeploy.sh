#!/usr/bin/env bash
#
# redeploy.sh — pull latest, rebuild the voice container, and tail its logs.
#
# The satellite's code is BAKED INTO the image (docker-compose `build: .`, no bind-mount),
# so a code change needs a rebuild — unlike the bind-mounted platform (`skipper update`).
# This is the one-shot for that: git pull -> rebuild + recreate -> follow logs.
#
# Ctrl-C stops the log tail only; the container keeps running.
#
set -euo pipefail

# Always run from the repo root (this script's own directory), regardless of cwd.
cd "$(dirname "$0")"

echo "==> git pull"
git pull

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> following logs (Ctrl-C to stop tailing; container stays up)"
exec docker compose logs -f
