# mc-rsi-trainer

The orchestration layer for training a Minecraft-playing policy via Recursively/Iteratively
Self-Improving (RSI) self-play — a Teacher/Student "leapfrog" loop, as sketched in
[`../cRL-go/docs/thoughts/rsi-with-rl.md`](../cRL-go/docs/thoughts/rsi-with-rl.md) — on top of two
sibling repos:

- [`github.com/reallyoldfogie/cRL-go`](../cRL-go) — the environment-agnostic RL algorithm library
  (autograd, PPO, REINFORCE, checkpoints).
- [`github.com/reallyoldfogie/mc-agent`](../mc-agent) — the live Minecraft protocol client/bot
  providing movement, pathfinding, combat, mining, mounting, etc.

This repo exists so that neither of those two has to depend on the other, or on
Minecraft-specific/RSI-specific training-orchestration code: cRL-go stays a small,
environment-agnostic library, and mc-agent stays a Minecraft bot capability library. See
[`docs/plans/00-rsi-trainer-roadmap.md`](docs/plans/00-rsi-trainer-roadmap.md) for the full
rationale, current blockers, and planned scope.

## Status

As of 2026-09-13, all three of this repo's original blockers are cleared (dependency pinning,
mc-agent concurrency bugs, per-episode task selection/goal-conditioning — see
[`docs/plans/00`](docs/plans/00-rsi-trainer-roadmap.md)), and there is now a runnable training
entrypoint: **[`cmd/rsi-train`](cmd/rsi-train)** connects one live mc-agent bot session, resumes (or
freshly initializes) a Teacher from `-checkpoint-dir`, runs
**[`pkg/leapfrog`](pkg/leapfrog)**'s Teacher-vs-Student evaluation loop repeatedly, and saves a new
generation via **[`pkg/lineage`](pkg/lineage)** on every Student win — see
[`docs/plans/04`](docs/plans/04-training-entrypoint-and-observability.md). It builds, is unit-tested
(flag validation, resume-vs-fresh-init checkpoint logic), and connects to mc-agent through the same
config/connection bootstrap `mc-agent`'s own `cmd/rl-train` uses. **A real live smoke test
(`cmd/rsi-train/main_live_test.go`) has now completed a full leapfrog round end to end, twice** — the
first time this whole implementation effort has proven the single-environment loop against a real
server (46.6s and 27.8s for a tiny round). Getting there took two rounds of live-connection failures
first — traced to bot usernames exceeding Minecraft's 16-character limit in the test setup, not a
bug in this repo's own training code, and now permanently fixed at the source: mc-agent's
`agent.ResolveAuth` rejects an over-length username locally, before any network round-trip; see
`docs/plans/08`'s own "Status" and
`../mc-agent/docs/bugs/offline-login-hello-packet-decode-disconnect.md` (Status: CLOSED, fully
confirmed) for the full story.

`pkg/leapfrog.Round` runs Teacher and Student **sequentially** by design (not concurrently — see
`docs/plans/03`), even though mc-agent's concurrency bugs blocking a concurrent version are now fixed
(`docs/plans/05`). `docs/plans/06`'s per-episode task selection (`rlenv.Config.TaskSelector`) and
goal-conditioning observation block are now used by **[`pkg/curriculum`](pkg/curriculum)** +
**[`pkg/curriculum/rlenvadapter`](pkg/curriculum/rlenvadapter)** (`docs/plans/07`) — a
`Generator` (with a uniform-random implementation) that varies which task an episode poses, the
anti-capability-lock mechanism this whole project is ultimately for. Fully unit-tested;
**`pkg/leapfrog.Round` itself needed no changes at all** to support it, since task variation happens
entirely at `rlenv.Environment` construction time. A real live test for this exists too
(`testing/curriculum_live_test.go`) — the first attempt at a live test in this project, and the one
that first hit the login issue above (its own scratch test config used an over-length username too).
Rerun with a corrected (13-character) username: 3 real episodes completed with correct task
exclusivity and goal-conditioning observations, before hitting an unrelated, known
`SeedNearbyBlock` visibility-timeout issue on non-flat terrain — not a login or curriculum bug.

Also fixed: the goto task's fixed `TargetOffset` could silently land somewhere a real (non-flat)
world's pathfinder could never reach, producing a zero-signal episode with no error at all — found,
fixed, and live-confirmed absent across five rounds of iteration in mc-agent's `rlenv` package
(standability + reachability checks, a retry-with-jitter loop) and `cmd/rsi-train` (`-auto-reset-origin`
plus a spawn-quality check) — see `docs/plans/08`'s own "Status" for the full account, including a
real bug it found along the way in a shared test fixture (`mc-agent/testing/mock_world.go` wasn't
flooring fractional coordinates the way the real world implementation does).

Also new: **[`testing/mcserver.go`](testing/mcserver.go)**'s `EnsureServer` (`docs/plans/10`) checks
for a live Minecraft server at a config's address and launches one via Docker if none is found —
every live test above uses it, so none of them need a manually pre-started server anymore.

Also new: `cmd/rsi-train` exposes live Prometheus metrics (generation, round outcomes, training
throughput, curriculum task mix — `-metrics-addr`, default `:9400`), and **[`monitoring/`](monitoring)**
holds a Prometheus + Grafana stack (`docker compose up -d`) with a dashboard already provisioned
against them — see [`monitoring/README.md`](monitoring/README.md) and
[`docs/glossary.md`](docs/glossary.md)'s "Metrics" section for what each one means. A training run
no longer needs a shell watching its log to know how it's doing.

See [`docs/plans/00-rsi-trainer-roadmap.md`](docs/plans/00-rsi-trainer-roadmap.md) for the full
rationale and current status, and `docs/plans/02` through `10` for the numbered sequence that takes
this repo from here to a fully working leapfrog trainer.

New to the project? Start at [`docs/README.md`](docs/README.md) for a beginner-level orientation,
or [`docs/getting-started.md`](docs/getting-started.md) to build and test what exists today.
