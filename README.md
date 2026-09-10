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
wiring (`go.mod` pins `cRL-go@v0.6.0` — a stale snapshot as of 2026-09-10; cRL-go is tagged through
`v0.10.2`, re-bump before relying on anything past `v0.6.0`). mc-agent's live `rl.Environment`
adapter (`rlenv`) is real and merged on mc-agent's side, and now knows three task types
(goto/mine/craft, not one), but mc-agent still has no version tags (it does now have a git remote),
so this repo still has no clean pinned way to depend on it. Each `rlenv.Environment` also still
fixes its task at construction — no per-episode task switching yet, so there's nothing for a
curriculum to vary within one run (see `docs/plans/01-curriculum-generator.md`). There is no
leapfrog loop or mc-agent integration here yet — see the roadmap document for the full picture,
including two mc-agent concurrency bugs found along the way.
