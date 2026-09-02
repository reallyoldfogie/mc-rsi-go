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

Mostly planning. Two pieces that only need `cRL-go`'s stable, tagged API exist:
[`pkg/lineage`](pkg/lineage) (Teacher/Student generation/checkpoint bookkeeping) and dependency
wiring (`go.mod` pins `cRL-go@v0.6.0` — a snapshot, not final, since cRL-go is under active
development). mc-agent's live `rl.Environment` adapter (`rlenv`) is now real and merged on
mc-agent's side, but mc-agent has no git remote or tags yet, so this repo has no clean way to
depend on it, and `rlenv` itself still only knows one task. There is no leapfrog loop, task
generator, or mc-agent integration here yet — see the roadmap document for the full picture,
including two mc-agent concurrency bugs found along the way.
