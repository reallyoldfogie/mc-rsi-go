# Getting Started

This page walks through setting up this project on your own machine. It covers what's actually
usable today; sections describing not-yet-built functionality are clearly marked **TODO** and will
be filled in as the corresponding implementation plan (linked in each section) lands.

If you're not sure what any of this means conceptually, read `architecture.md` and `glossary.md`
first.

## Prerequisites

- **Go 1.27** or newer (`go version` to check).
- **Git**, and access to clone this project's two sibling repositories:
  [`cRL-go`](https://github.com/reallyoldfogie/cRL-go) and
  [`mc-agent`](https://github.com/reallyoldfogie/mc-agent).
- (Only once training is runnable — see the TODO sections below) a Minecraft server you control,
  with RCON enabled, that mc-agent can connect to.

## 1. Clone the three repos side by side

This project assumes a specific folder layout: this repo and its two sibling repos live as
directories next to each other, e.g.:

```
src/github.com/reallyoldfogie/
├── cRL-go/
├── mc-agent/
└── mc-rsi-trainer/     ← this repo
```

In practice this layout no longer matters for building this repo itself: both sibling repos are now
tagged, real Go module dependencies (`go.mod` pins `cRL-go` and `mc-agent` by version, not by local
path), so a plain `go build`/`go test` works regardless of where you clone things — see
`plans/02-dependency-resolution-and-cmd-scaffold.md` for why an earlier version of this plan expected
a `go.work` file here and why that turned out to be unnecessary. The side-by-side layout still
matters if you want to develop against local, unpushed changes to a sibling repo (see that same
document's "local development convenience" note on `go.work`).

```
mkdir -p src/github.com/reallyoldfogie
cd src/github.com/reallyoldfogie
git clone git@github.com:reallyoldfogie/cRL-go.git
git clone git@github.com:reallyoldfogie/mc-agent.git
git clone git@github.com:reallyoldfogie/mc-rsi-trainer.git
```

## 2. Build and test this repo

From `mc-rsi-trainer/`:

```
go build ./...
go test ./...
```

This builds and tests everything currently implemented — as of this writing, that's `pkg/lineage`
(the Teacher/Student generation bookkeeping) and `pkg/leapfrog` (the actual Teacher-vs-Student
evaluation loop — see `architecture.md`'s "What exists today" section). You should see all tests
pass with no live Minecraft server or mc-agent session needed; both packages' default test suites
run against local files and a toy environment, not a real bot session.

## 3. Explore `pkg/lineage` and `pkg/leapfrog`

If you want to see the generation-bookkeeping system in action without any Minecraft involved at
all, read `pkg/lineage/lineage_test.go` — it exercises saving/loading generation records against a
temporary directory, which is a good way to see what a "generation" actually looks like on disk
(a JSON record plus a cRL-go checkpoint file).

`pkg/leapfrog/leapfrog_test.go` shows the Teacher-vs-Student comparison itself, run against a small
toy grid-world environment instead of a real Minecraft session — a fast way to see what "the Student
trains, then both sides get evaluated, then the better one wins" actually looks like in code, before
reading `testing/leapfrog_live_test.go` (this repo's live-server integration tests live in their own
`testing/` package, mirroring mc-agent's own convention — see that file's doc comment for how to run
it for real; it's skipped by default via the `MC_RSI_TRAINER_LIVE_CONFIG` environment variable, not
a Go build tag, so it stays visible to normal editing/type-checking even when you're not running it).

You don't need to have a Minecraft server already running to try this: `testing/mcserver.go`'s
`EnsureServer` checks whether one is already reachable at your config's `connection.address` and, if
not, launches one via Docker automatically (offline mode, no real Mojang login needed) — see
`docs/plans/10-live-server-management.md`. It's torn down again when the test finishes unless you
set `MC_RSI_TRAINER_KEEP_SERVER=1`, which is handy if you want to leave it running to poke at with a
real Minecraft client or a second terminal in between test runs. Needs a working Docker daemon.

## 4. Running a real training session

Once you have a Minecraft server you control (with RCON enabled) and mc-agent can connect to it,
`cmd/rsi-train` runs the actual leapfrog training loop against it.

First, build it:

```
go build -o rsi-train ./cmd/rsi-train
```

`rsi-train` needs two pieces of configuration:

- **`-mc-agent-config`** — a path to a JSON file in mc-agent's own `config.Settings` format (the
  same format mc-agent's own `cmd/rl-train` uses): server address, RCON credentials, and the
  `env` section describing the task (`target_offset`, `mine_target_block`, and so on — see
  `../mc-agent/rlenv`'s `Config` for what each field means, or mc-agent's own
  `configs/config.json` for a worked example). A minimal one looks like:

  ```json
  {
    "connection": { "address": "localhost:25565", "offline": true, "name": "RSITrainerBot" },
    "rcon": { "address": "localhost:25575", "password": "changeme" },
    "env": { "target_offset": [5, 0, 0], "arrival_threshold": 1.5, "step_timeout_seconds": 10 }
  }
  ```
- **`-checkpoint-dir`** — a directory `rsi-train` reads its latest saved generation from (if any)
  and writes new ones into as training progresses. Point it at an empty directory the first time;
  it fills in generation 0 automatically.

Then run it:

```
./rsi-train -checkpoint-dir ./checkpoints -mc-agent-config ./my-config.json
```

By default it runs indefinitely (stop it with `Ctrl-C`, which shuts down cleanly and finishes
whatever it's doing without corrupting the checkpoint directory); pass `-max-rounds N` to stop after
a fixed number of leapfrog rounds instead — useful for a first, short test run. See `rsi-train -h`
for every flag (how many PPO epochs the Student trains per round, how many episodes each side is
evaluated over, how often interim checkpoints are saved, and so on).

It also serves live Prometheus metrics by default (`-metrics-addr`, default `:9400`) — see step 5,
below, for a Grafana dashboard that graphs them instead of reading the log.

### Reading the log output

Each Student-training epoch logs a progress line (`epoch N: average return ..., samples ...`), and
each completed round logs a summary: which side won, both sides' mean evaluation reward, how long
the round took, and the resulting generation number. A generation number that goes up means the
Student just beat the Teacher and became the new Teacher; a round with no generation-number change
means the Teacher held.

### Where things end up

- `<checkpoint-dir>/generation-NNNNNNNNN-params.json` and `generation-NNNNNNNNN.json` — one pair per
  generation (the trained policy, and its lineage record — parent generation, when it was created,
  and so on). See `pkg/lineage`.
- `<checkpoint-dir>/interim/` — periodic mid-round checkpoints (not full generations), a recovery
  aid if the process is killed partway through a long Student-training phase. Interim files retain
  epoch, best-return, and total-gradient-update metadata; `final.json` records the latest
  completed parameters and metadata when the run exits cleanly.
- Replay files (`.mcpr`), if your `-mc-agent-config` sets `"replay": {"enable": true}` — written
  under your mc-agent cache directory's `replays/<version>/` folder by default, or wherever
  `"output"` points if you set it explicitly. Play these back with
  [ReplayMod](https://www.replaymod.com/) in a matching Minecraft client — this repo doesn't do
  anything with them itself beyond recording them.

## 5. Running inference against a trained policy

`cmd/rsi-infer` runs a checkpoint's policy live against a real bot session, acting greedily (or
optionally sampling) rather than training — for watching what a generation has actually learned to
do, or for evaluating it outside of `rsi-train`'s own paired Teacher/Student comparison.

```
go build -o rsi-infer ./cmd/rsi-infer
./rsi-infer -mc-agent-config ./my-config.json -checkpoint-dir ./checkpoints -episodes 5
```

It needs the same two pieces of configuration as `rsi-train` (`-mc-agent-config`, `-checkpoint-dir`),
plus:

- **`-generation`** — which generation to load; `-1` (the default) loads the highest one found.
- **`-episodes`** — how many episodes to run before exiting; `0` (the default) runs until you
  press `Ctrl-C`.
- **`-greedy`** — act on the highest-probability action every step rather than sampling from the
  policy's distribution; `true` by default. Turning this off is worth trying if greedy play looks
  stuck doing nothing: at a state where two actions are close to equally likely, the greedy choice
  can be the worse one even though the policy would pick correctly most of the time it actually
  matters — `rsi-train`'s own round evaluation defaults the other way (`Config.EvalGreedy`, off by
  default — see its doc comment in `pkg/leapfrog/leapfrog.go`) for exactly this reason, after a
  greedy-only evaluation made a policy that had genuinely improved look like it had collapsed,
  round after round.
- **`-normalize-observation`** — must match whatever the training run that produced this checkpoint
  used. It's folded into the checkpoint's environment ID, so a mismatch is rejected outright rather
  than silently feeding the policy the wrong kind of input.

See `rsi-infer -h` for the rest (episode length, an explicit `-generation` for evaluating an older
one specifically).

**Connecting it to the same server a training run is already using:** this works, and shares the
same world, but needs a bot name (`connection.name` in your `-mc-agent-config`) distinct from
whatever `rsi-train` is already using, or it collides with an in-use player slot outright. Also give
it its own `env.reset_origin` well clear of the training run's own working area (`env.use_reset_origin:
true` plus a `reset_origin` far from wherever the training bots operate) — a fresh bot dropped into
an active bot's own square meter of world mostly just gets in its way. A `reset_origin` you're
using for the first time can also take a few seconds for its chunk data to reach this new
connection even once the server itself has it loaded (`FindVisibleBlock`'s own doc comment in
mc-agent covers this — a search there can legitimately find nothing for a moment after teleporting
somewhere new, which is a different thing from the target genuinely not being there).

## 6. Widening a trained network's hidden layer

`cmd/widen-checkpoint` applies cRL-go's `actorcritic.Net2WiderNet` to a saved generation, growing its
hidden layers' width without discarding what training already learned — an alternative to starting a
bigger network from scratch and re-climbing the whole learning curve. See that function's own doc
comment in cRL-go for the technique itself (Chen, Goodfellow, Shlens 2016, "Net2Net"); this section
covers using the tool.

```
go build -o widen-checkpoint ./cmd/widen-checkpoint
./widen-checkpoint -checkpoint-dir ./checkpoints -out-dir ./checkpoints-widened -new-hidden-size 256
```

This reads `-checkpoint-dir`'s latest generation and writes the result as a **fresh generation 0 in
`-out-dir`** — a separate directory, not appended to the source lineage. Nothing about the source
checkpoint directory changes, and `rsi-train`/`rsi-infer` only ever pick up the widened network if
you point their own `-checkpoint-dir` at `-out-dir` yourself. `-out-dir` also gets a `source.txt`
recording which generation it came from and what parameters were used.

Other flags worth knowing about:

- **`-noise-std`** (default `0.05`) — each newly-created unit is an exact copy of an existing one
  until this much random noise is added to it, which is what actually lets it diverge and use its
  added capacity during further training; `0` gives an exactly function-preserving widen with no such
  head start. Verified against a real, currently-trained checkpoint: at the default, the policy's
  output shifted by only a few percent, but the value head (whose output is unbounded, unlike the
  policy's probabilities) moved by close to 2 — pick something smaller than `0.05` if you want the
  value head's starting point preserved more tightly.
- **`-seed`** — reproduces the exact same widened network from the same source checkpoint, so you
  can compare `-noise-std` values on an otherwise identical draw of which units got cloned.
- **`-verify-probes`** (default `8`) — runs that many random observations through both the source
  and widened network afterward and reports the largest difference in their outputs, as a quick
  sanity check.

To actually train against the result, point `rsi-train` at it directly:

```
./rsi-train -checkpoint-dir ./checkpoints-widened -mc-agent-config ./my-config.json \
  -normalize-observation  # if the source run used it — the widened checkpoint keeps whatever
                           # environment ID the source had
```

## 7. Monitoring a training run

`rsi-train` exposes live Prometheus metrics on `-metrics-addr` (default `:9400`) for the whole
duration of a run — generation, round outcomes, training throughput, task mix, and more (see
[`docs/glossary.md`](glossary.md)'s "Metrics" section for exactly what each one means). Bring up a
local Prometheus + Grafana stack once:

```
cd monitoring
docker compose up -d
```

then open `http://localhost:3000` — a "RSI Curriculum Training" dashboard is already there, no
setup needed. It keeps working for every future run against the same `-metrics-addr`; you don't
need to redo this per run. See [`monitoring/README.md`](../monitoring/README.md) for the full
picture (multiple concurrent runs, stopping/restarting the stack, editing the dashboard).

## TODO: Setting up a Minecraft server for training

Not yet written in detail — for now, any server mc-agent's own setup docs already describe working
with (RCON enabled, a version mc-agent supports) is enough; nothing about training specifically
needs anything beyond what step 4 above's `-mc-agent-config` already covers.

## Troubleshooting

Not yet populated — see `plans/09-beginner-documentation.md`'s tracking table. This section will
fill in with real issues as they're hit during `plans/02`-`05`'s implementation, rather than
speculated ones.
