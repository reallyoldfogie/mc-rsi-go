# Combat training/competition sample configs

Two-phase flow: **train** (self-play, produces frozen checkpoints) then
**compete** (a tournament bracket plays those checkpoints against each
other, pure inference, no further learning).

## 1. Train (produces checkpoints)

```bash
go run ./cmd/rsi-pvp-train -config docs/examples/pvp-train-sample.json -metrics-addr :9400
```

Requires a running Minecraft server reachable at the address in
`mc-agent-config.json` (see `testing/README.md` for spinning one up, or
point it at any server you already run). Runs until `training.duration`
elapses (10 minutes in the sample - long enough to produce a non-trivial
checkpoint, nowhere near full convergence). Trains one match, two 2-member
teams (red vs blue). Checkpoints land at
`<checkpoint_root>/<match id>/<participant id>.json` - with this sample,
`./checkpoints/arena-1/red1.json`, `./checkpoints/arena-1/red2.json`, etc.

`-metrics-addr` is optional; when set, it also serves a live 3D view of
both teams fighting at `/dashboard` (plus `/metrics`).

## 2. Compete (plays the checkpoints in a bracket)

```bash
go run ./cmd/rsi-pvp-compete -config docs/examples/pvp-compete-sample.json -dashboard-addr :9500
```

`pvp-compete-sample.json`'s two competitors (red and blue, both 2-member
teams - a single pairing, straight to the final) each reference a
checkpoint path under `./checkpoints/arena-1/` - the exact files step 1
produces. If you used a different `checkpoint_root`/match ID, or already
have checkpoints from elsewhere, just point `checkpoint_path` at those
files instead; nothing else needs to change. Add more competitors
(copy the `red`/`blue` block, give it a new `id`) for a larger bracket -
each needs its own checkpoints, e.g. from a second training run with
different team names.

`-dashboard-addr` is optional; when set it serves the live scoreboard at
`/`, bracket status JSON at `/status`, the live 3D agent view at
`/dashboard`, and Prometheus metrics at `/metrics`.

## Notes

- Both configs' `server.world.mc_agent_config` point at the same
  `mc-agent-config.json` - adjust its `connection`/`rcon` fields for your
  own server (address, RCON password, offline-mode name/UUID if not
  using Microsoft auth).
- `arena.containment: "soft_area"` actively teleports anyone who wanders
  past `arena.radius` back to center; `"world_border"` uses vanilla's
  own world border command instead (only sensible for a single match per
  server, since it's not per-match-scoped).
- A flat, decoration-free world is strongly recommended for combat -
  `ConnectManagedParticipant`'s arena-origin resolution has no
  ground-safety check, and a spawn point on top of tree canopy means
  every reset teleports bots right back onto it (found live: this can
  make every game resolve almost instantly via fall damage, not combat).
  If launching your own test server via `itzg/minecraft-server`, set
  `LEVEL_TYPE=FLAT`.
- `pvp-train-sample.json`'s `trainer_config` point at a deliberately tiny
  `crlconfig.Settings` - enough to produce a *valid, loadable* checkpoint
  quickly for trying this out, not a realistic training recipe. A real
  run should use production-scale rollout/batch sizes (see
  `docs/plans/19-combat-pvp-training.md`).
