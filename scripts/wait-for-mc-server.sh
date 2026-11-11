#!/usr/bin/env bash
# Waits, best-effort and bounded, for the training MC server container to
# report healthy before a boot-time trainer launch proceeds.
#
# This is a COURTESY wait, not a hard dependency: scripts/supervise-rsi-train.sh
# already retries rsi-train with its own crash-loop backoff if the server
# isn't reachable yet, so a timeout here just means skipping straight to
# that existing retry loop instead of also doing a duplicate one. That's
# why this always exits 0, even on timeout - see docs/boot-time-autostart.md.
#
# Usage: wait-for-mc-server.sh [container-name] [max-wait-seconds]
set -u

container=${1:-mc-rsi-trainer-mc-1-21-5-rsi-recu}
max_wait=${2:-90}
poll_interval=3

waited=0
while [ "$waited" -lt "$max_wait" ]; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)
  if [ "$status" = "healthy" ]; then
    echo "wait-for-mc-server: $container is healthy after ${waited}s"
    exit 0
  fi
  sleep "$poll_interval"
  waited=$((waited + poll_interval))
done

echo "wait-for-mc-server: $container not healthy after ${max_wait}s - proceeding anyway; rsi-train's own supervisor will retry" >&2
exit 0
