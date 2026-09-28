# mc-rsi-trainer

The orchestration layer for training a Minecraft-playing policy via Recursively/Iteratively
Self-Improving (RSI) self-play — a Teacher/Student "leapfrog" loop, as sketched in
[`../cRL-go/docs/thoughts/rsi-with-rl.md`](../cRL-go/docs/thoughts/rsi-with-rl.md) — built on top of
two sibling repos:

- [`github.com/reallyoldfogie/cRL-go`](../cRL-go) — the environment-agnostic RL algorithm library
  (autograd, PPO, REINFORCE, checkpoints).
- [`github.com/reallyoldfogie/mc-agent`](../mc-agent) — the live Minecraft protocol client/bot
  providing movement, pathfinding, combat, mining, crafting, and more.

Neither sibling depends on the other, or on Minecraft-specific/RSI-specific training-orchestration
code: cRL-go stays a small, environment-agnostic library; mc-agent stays a Minecraft bot capability
library. This repo sits above both, using cRL-go's algorithms to train a policy and mc-agent's bot to
try it out in a real Minecraft world. See
[`docs/plans/00-rsi-trainer-roadmap.md`](docs/plans/00-rsi-trainer-roadmap.md) for the full
rationale behind that split.

## How leapfrog self-play works, briefly

A **Teacher** (the current best policy) and a **Student** (a fresh copy of it) both train against the
same curriculum of tasks. Periodically, both are evaluated head-to-head over the same episodes; if
the Student's mean reward beats the Teacher's by more than noise would explain, it becomes the new
Teacher, and a new Student spins up from it to keep improving. Repeated over many rounds, this
produces a policy that keeps getting better without a human hand-tuning it after each one. See
[`docs/glossary.md`](docs/glossary.md) for every term this document uses without redefining it, and
[`docs/README.md`](docs/README.md) for a from-scratch explanation aimed at someone new to RL.

## What's built

- **[`pkg/lineage`](pkg/lineage)** — Teacher/Student generation bookkeeping: which checkpoint is
  which generation, its parent, and how it came to exist (leapfrog self-play, seeded from elsewhere,
  or grown from an earlier one — see `cmd/widen-checkpoint` below).
- **[`pkg/leapfrog`](pkg/leapfrog)** — the Teacher-vs-Student round loop: trains a Student for a
  fixed epoch budget, evaluates both sides by sampling from their policies rather than acting
  greedily (a greedy read of a stochastic policy can look collapsed at a close decision boundary even
  when the policy itself is fine), and only promotes a win that clears a configurable statistical
  margin over the paired per-episode difference — a bare strict-inequality rule alone promotes noise
  about as often as real improvement.
- **[`pkg/curriculum`](pkg/curriculum) + [`pkg/curriculum/rlenvadapter`](pkg/curriculum/rlenvadapter)**
  — varies which task an episode poses (go to a point, mine a block, craft an item, "go there, then
  mine/craft" composites, and multi-step chains: gather logs, craft planks and a crafting table,
  place the table, then craft a goal item at it), the mechanism this project uses against
  capability lock — a policy that only ever practices one fixed skill.
- **[`pkg/parallelenv`](pkg/parallelenv)** — derives settings for many concurrent live bot sessions
  (against one shared server or several) for training throughput.
- **[`cmd/rsi-train`](cmd/rsi-train)** — the training entrypoint: resumes or freshly initializes a
  Teacher, runs leapfrog rounds against a live curriculum continuously, checkpoints as it goes, and
  serves live Prometheus metrics. Includes a policy-health check that refuses to start from, and
  aborts training on, a network whose hidden units have stopped responding to their input — a real
  failure mode this project hit and that went unnoticed for a week before the check existed.
- **[`cmd/rsi-infer`](cmd/rsi-infer)** — runs a saved generation live against a real bot session
  without training, for watching what it's actually learned to do.
- **[`cmd/widen-checkpoint`](cmd/widen-checkpoint)** — grows a trained network's hidden layer width
  without discarding what it already learned (cRL-go's `Net2WiderNet`), entirely offline — no live
  server needed to try a bigger network.
- **[`monitoring/`](monitoring)** — a Prometheus + Grafana stack with a dashboard already
  provisioned against `rsi-train`'s metrics.
- **[`scripts/supervise-rsi-train.sh`](scripts/supervise-rsi-train.sh)** — restarts a training run
  that stops on its own (today, a single dropped bot connection ends the whole run), without
  restarting past a condition the run itself flags as unrecoverable.
- **[`testing/`](testing)** — live integration tests against a real Minecraft server, launched
  automatically via Docker if one isn't already running.

All of the above builds, is unit-tested, and has been run for real: multi-day training runs with
several concurrent bots against a shared server, producing multiple generations of genuine
Teacher-beats-Student promotion across the current task mix.

## Where this goes next

With the leapfrog loop itself proven — training runs, evaluates fairly, and promotes real
improvement — the open question has shifted from "does self-play work here" to "what should it be
practicing." Comparing a completed run's actual return against the reward function's own theoretical
ceiling found the current task mix close to saturated: the policy is already near-optimal for
goto/mine/craft and their far-composite variants, so more training on the same mix won't move that
number much further. The likely next steps are a harder curriculum (new task types, tighter timing,
richer sub-goals) and/or reward-shaping changes, with `cmd/widen-checkpoint` built ahead of time for
when a richer curriculum needs more raw network capacity than today's architecture provides.

See [`docs/plans/00-rsi-trainer-roadmap.md`](docs/plans/00-rsi-trainer-roadmap.md) for the detailed,
point-in-time implementation history behind all of the above, and `docs/plans/02` onward for the
numbered sequence of design documents it links to.

## New to the project?

Start at [`docs/README.md`](docs/README.md) for a beginner-level orientation (what RSI/leapfrog
means, how the three repos fit together, defined terms), or jump straight to
[`docs/getting-started.md`](docs/getting-started.md) to clone, build, and run what's here today.
