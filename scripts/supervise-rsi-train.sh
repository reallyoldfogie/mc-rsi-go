#!/usr/bin/env bash
# Keeps a long training run alive: restarts the command whenever it exits on
# its own, and stops for good only when THIS script is signalled.
#
#   scripts/supervise-rsi-train.sh LOGFILE COMMAND [ARGS...]
#
# Why: rsi-train stops the whole run when any one bot session ends (a dropped
# connection, a packet-handler error), and it exits normally when it does, so
# the exit code can't tell that from a finished run. Each such stop used to
# cost however long it took a person to notice. The trainer resumes from its
# checkpoint on restart, so restarting costs seconds.
#
# To stop the run (gracefully; the trainer saves its checkpoint first):
#     kill -TERM <supervisor pid>
# Killing only the trainer just makes the supervisor start it again. A run
# that is meant to finish (-max-rounds) should not be supervised.
#
# Start it in its own session so nothing else's signals reach it. setsid may
# fork, so ask for a pidfile rather than trusting $!:
#     SUPERVISE_PIDFILE=run.pid setsid nohup scripts/supervise-rsi-train.sh \
#         run.log ./bin/rsi-train ... &
#     kill -TERM "$(cat run.pid)"      # later, to stop the run for good
#
# Crash-loop guard: a run shorter than SUPERVISE_HEALTHY_SECONDS (default 300)
# doubles the wait before the next start (SUPERVISE_MIN_BACKOFF, default 10s,
# up to SUPERVISE_MAX_BACKOFF, default 300s), so an unreachable server isn't
# hammered; a longer run resets it.

set -u

if [ "$#" -lt 2 ]; then
  echo "usage: $0 LOGFILE COMMAND [ARGS...]" >&2
  exit 2
fi
log=$1
shift

min_backoff=${SUPERVISE_MIN_BACKOFF:-10}
max_backoff=${SUPERVISE_MAX_BACKOFF:-300}
healthy_seconds=${SUPERVISE_HEALTHY_SECONDS:-300}

child=0
sleeper=0
stopping=0

if [ -n "${SUPERVISE_PIDFILE:-}" ]; then
  echo $$ >"$SUPERVISE_PIDFILE"
  trap 'rm -f "$SUPERVISE_PIDFILE"' EXIT
fi

say() { printf '%s supervisor: %s\n' "$(date '+%Y/%m/%d %H:%M:%S')" "$*" >>"$log"; }

on_signal() {
  stopping=1
  say "signalled; stopping the trainer (pid $child) and not restarting"
  if [ "$child" -gt 0 ]; then kill -TERM "$child" 2>/dev/null; fi
  if [ "$sleeper" -gt 0 ]; then kill "$sleeper" 2>/dev/null; fi
}
trap on_signal TERM INT HUP

backoff=$min_backoff
while [ "$stopping" -eq 0 ]; do
  say "starting: $*"
  start=$(date +%s)
  "$@" >>"$log" 2>&1 </dev/null &
  child=$!
  rc=0
  # wait returns early (>128) when a trapped signal arrives; keep waiting
  # until the trainer has really exited so it can finish saving.
  while kill -0 "$child" 2>/dev/null; do
    wait "$child"
    rc=$?
  done
  child=0
  [ "$stopping" -eq 1 ] && break

  ran=$(( $(date +%s) - start ))
  say "trainer exited on its own (code $rc) after ${ran}s"
  # Code 3 is the trainer reporting a condition a restart cannot fix (a dead
  # policy network); restarting would only resume into it.
  if [ "$rc" -eq 3 ]; then
    say "trainer reported a fatal condition (code 3); not restarting"
    break
  fi
  if [ "$ran" -ge "$healthy_seconds" ]; then
    backoff=$min_backoff
  else
    backoff=$(( backoff * 2 > max_backoff ? max_backoff : backoff * 2 ))
  fi
  say "restarting in ${backoff}s"
  sleep "$backoff" &
  sleeper=$!
  wait "$sleeper" 2>/dev/null
  sleeper=0
done
say "exiting"
