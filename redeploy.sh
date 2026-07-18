#!/usr/bin/env bash
#
# redeploy.sh — pull latest, rebuild the voice container from scratch, and tail its log.
#
# The satellite's code is BAKED INTO the image (docker-compose `build: .`, no bind-mount),
# so a code change needs a full rebuild — unlike the bind-mounted platform (`skipper update`).
# This runs the exact sequence to pick up new code:
#     git pull  ->  docker compose down  ->  docker compose up -d --build  ->  logs -f voice
#
# Ctrl-C stops the log tail only; the container keeps running.
#
set -euo pipefail

# Always run from the repo root (this script's own directory), regardless of cwd.
cd "$(dirname "$0")"

echo "==> git pull"
git pull

echo "==> docker compose down"
docker compose down

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> following the voice log (Ctrl-C to stop tailing; container stays up)"
exec docker compose logs -f voice
