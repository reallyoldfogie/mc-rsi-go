# Surviving a host reboot

This machine runs Docker and the trainer directly (not inside Docker itself), on a WSL2 VM where
systemd runs as PID 1 (confirmed via `systemctl is-system-running`), so ordinary system-level
systemd units work exactly as they would on bare Linux. "Boot" here means whenever this WSL2
instance (re)starts — e.g. after `wsl --shutdown` or a Windows restart — not a login event.

Two independent pieces now survive that automatically. Neither existed before 2026-11-11; before
that, a reboot required manually restarting the MC server container, re-running `rcon-cli "tick
rate 300"`, and manually re-running `start_trainer.sh`.

## 1. The MC server container

The training MC server container (`mc-rsi-trainer-mc-1-21-5-rsi-recu`, an `itzg/minecraft-server`
image) now has:

- **`--restart unless-stopped`** — Docker itself brings the container back up whenever `dockerd`
  starts, with no systemd involvement. This is the same mechanism the `monitoring/docker-compose.yml`
  stack (Grafana/Prometheus) already used.
- **`-e RCON_CMDS_STARTUP='tick rate 300'`** — an `itzg/minecraft-server` feature: it runs this RCON
  command once per server boot, automatically. Before this, every fresh server process (a plain
  `docker start` after a crash, or a full container recreate) silently reset the tick rate back to
  vanilla's 20/s, and that was easy to miss. If you ever recreate this container by hand (e.g. to
  add another mod), carry both flags over — `docker inspect mc-rsi-trainer-mc-1-21-5-rsi-recu` shows
  the full current config to copy from. The `docker run` invocation itself isn't tracked anywhere
  as a script, since recreating this container is normally a rare, one-off operation.

**Caveat found while testing this**: right after the container boots and before the first bot
connects, the server logs `Server empty for 60 seconds, pausing` and a single `rcon-cli "tick
query"` run in that window can misreport the active rate. This is cosmetic — confirmed live that the
real rate holds at 300 once bots actually connect. Don't let a stale-looking read right after boot
cause you to re-run the tick command; check again after the trainer has actually connected.

## 2. The trainer process

`start_trainer.sh` launches `scripts/supervise-rsi-train.sh`, which runs `rsi-train` in a loop with
its own crash-loop backoff, detached from the launching shell via `setsid nohup ... &`. That
supervisor loop is — and remains — the *only* thing that decides whether to restart the trainer
after it exits. Nothing described below changes that.

What's new is a systemd unit, `scripts/systemd/mc-rsi-trainer.service` (tracked in this repo; the
installed copy under `/etc/systemd/system/` is not), that runs `start_trainer.sh` once at boot:

```ini
[Unit]
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=scripts/wait-for-mc-server.sh   # best-effort, bounded wait for the MC container's healthcheck
ExecStart=/usr/bin/bash start_trainer.sh

[Install]
WantedBy=multi-user.target
```

### Install / enable

```bash
sudo cp scripts/systemd/mc-rsi-trainer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mc-rsi-trainer.service
```

### Why `Type=oneshot` + `RemainAfterExit=yes`, specifically

`start_trainer.sh` itself returns almost immediately — its only real action is backgrounding the
supervisor via `setsid nohup ... &`. `Type=oneshot` matches that: the unit's job is just to *fire*
the launch, not to stay in the foreground supervising it (that would duplicate
`supervise-rsi-train.sh`'s own backoff logic, and the two would fight over who gets to restart a
crashed trainer). `RemainAfterExit=yes` only affects how `systemctl status` displays things —
without it, the unit would flip back to `inactive` the instant `start_trainer.sh` returns, which
reads as "it didn't work" even on a perfectly successful launch.

### `wait-for-mc-server.sh` is a courtesy, not a dependency

`Requires=docker.service` only guarantees the Docker *daemon* is up, not that the MC server
*container* has finished booting and passed its healthcheck (we measured that taking roughly
10-15 seconds after `docker start`). `scripts/wait-for-mc-server.sh` polls for up to 90 seconds
(configurable via its two positional args) for the container to report `healthy`, then proceeds
regardless — it always exits `0`, timeout or not. If the MC server genuinely isn't up by the time
`rsi-train` tries to connect, that's not new or fatal: `supervise-rsi-train.sh`'s own backoff loop
already handles "can't reach the server yet" by retrying with increasing delay. This script just
avoids *also* burning through a few of those retries unnecessarily on every boot.

### Operational notes — read before using `systemctl` on this unit

These are the genuinely non-obvious parts, found while setting this up:

1. **`systemctl status mc-rsi-trainer.service` does not tell you whether the trainer is running.**
   Because this is `Type=oneshot` with `RemainAfterExit=yes`, the status reflects *"did the boot-time
   trigger fire successfully"*, permanently, until the unit is explicitly stopped or the host
   reboots again — not *"is `rsi-train` currently alive"*. To check that, use the existing method:
   `cat run.pid` and `ps -p "$(cat run.pid)"`, or just `ps aux | grep rsi-train`.

2. **`systemctl stop mc-rsi-trainer.service` *does* actually stop the real, detached processes** —
   this is easy to assume won't work (it's a oneshot unit; its own `ExecStart` process already
   exited long ago) but it does, because of a cgroups detail: `setsid`/`nohup`/`&` detach a process
   from the launching *shell* (so it survives that shell exiting, and signals to the shell's own
   process group don't reach it), but they do **not** remove it from the systemd unit's *cgroup*.
   systemd's default `KillMode=control-group` means `systemctl stop` sends `SIGTERM` to every
   process still in that cgroup — which includes the detached supervisor and the trainer it spawned.
   `supervise-rsi-train.sh` already traps `TERM` and forwards it to the trainer (saving its
   checkpoint first), so the net effect is identical to the pre-existing manual procedure
   (`kill -TERM "$(cat run.pid)"`) — just invoked through systemd instead.

3. **Prefer `systemctl start|stop|restart mc-rsi-trainer.service` over the manual
   `kill -TERM`/`bash start_trainer.sh` dance once this unit is installed.** Either approach
   actually works (see #2), but mixing them can leave systemd's recorded state out of sync with
   reality — e.g. manually re-running `start_trainer.sh` while the systemd-launched run is still
   alive would start a second, competing supervisor+trainer pair, since nothing stopped the first
   one. If you do stop it manually, running `systemctl stop mc-rsi-trainer.service` afterward
   (even though nothing is left to kill) will at least bring the unit's own status back in sync.

4. **The trainer's own logs still go to `train.log`, not the systemd journal.** `journalctl -u
   mc-rsi-trainer.service` only captures this unit's own stdout/stderr — in practice just
   `wait-for-mc-server.sh`'s one or two lines, since `start_trainer.sh` redirects everything else
   itself. Keep using `tail -f runs/.../train.log` (or Grafana) for actual training activity.

5. **To stop this from auto-starting on a future boot** (without touching anything currently
   running): `sudo systemctl disable mc-rsi-trainer.service`. To also stop the current run right
   now: add `--now` to that command, or run `systemctl stop` separately per #2 above.

### Quick reference

| Want to...                          | Run |
|---|---|
| Check if the trainer is actually running | `ps -p "$(cat run.pid)"` |
| Stop the run (checkpoint-safe)      | `systemctl stop mc-rsi-trainer.service` |
| Start it (if currently stopped)     | `systemctl start mc-rsi-trainer.service` |
| See the boot-trigger's own log line | `journalctl -u mc-rsi-trainer.service` |
| See actual training activity        | `tail -f runs/<run>/train.log` |
| Stop auto-starting on future boots  | `sudo systemctl disable mc-rsi-trainer.service` |
| Re-enable auto-start                | `sudo systemctl enable mc-rsi-trainer.service` |
