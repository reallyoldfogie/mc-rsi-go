# Glossary

Plain-language definitions of terms used throughout this project's documentation, in
reinforcement learning (RL), and specific to how this project applies RL to Minecraft. Terms are
grouped by theme; within a group, roughly in the order you'd need them.

## Reinforcement learning basics

- **Agent** — the decision-maker being trained. In this project, "agent" is a bit overloaded: it
  can mean the trained policy (the RL sense) or mc-agent's own Go package named `agent` (the
  Minecraft-bot sense). Context usually makes it clear which; when it matters, these docs say
  "policy" for the RL sense.
- **Environment** — whatever the agent interacts with and gets feedback from. In this project, the
  environment is a live Minecraft bot session (via mc-agent's `rlenv` package). In cRL-go's simpler
  example environments, it might be a toy grid world or a game of Snake instead.
- **Episode** — one complete attempt at a task, from start to finish (or until it times out/fails).
  Training happens over many episodes.
- **Observation** — the information the policy sees at each step: a fixed-length list of numbers
  describing the current situation (e.g. distance to target, what's visible nearby).
- **Action** — a choice the policy can make at each step (e.g. "move toward target," "mine," "wait").
- **Reward** — a number the environment gives back after each action, describing how good that
  action was. Training tries to make future actions earn more reward.
- **Policy** — the trained decision-making function: given an observation, it picks an action.
  Represented in this project as a set of numeric weights (see **Params**, below).
- **Step** — one observation → action → reward cycle within an episode.
- **Checkpoint** — a saved snapshot of a policy's weights (and some metadata about training
  progress) on disk, so training can be paused/resumed, or a specific version can be reloaded later.

## This project's specific vocabulary

- **RSI (Recursively/Iteratively Self-Improving)** — the overall idea this project implements:
  a policy improves itself over repeated rounds without a human redesigning it each time.
- **Leapfrog (self-play)** — this project's specific RSI technique: keep a Teacher and a Student
  (see below), compare them periodically, and promote the winner. Named because the Student
  "leapfrogs" ahead of the Teacher when it wins.
- **Teacher** — the current best-known policy. Used with low exploration (it mostly does what it
  already knows works) as the benchmark a Student must beat.
- **Student** — a copy of the Teacher that keeps training, with more exploration (it tries new
  things more often, since it needs to discover improvements). If a Student beats its Teacher over
  many evaluation episodes, it becomes the new Teacher.
- **Generation** — one numbered "best policy so far" in the Teacher lineage. Generation 0 is the
  first Teacher (often untrained or minimally trained); each time a Student wins, a new,
  higher-numbered generation is recorded.
- **Lineage** — the record of which generation came from which, when, and why (see `pkg/lineage` in
  this repo's code) — like a family tree for trained policies.
- **Provenance** — how a specific generation's policy came to exist: either trained through this
  project's own leapfrog rounds, or seeded from a policy that kept learning live while deployed in
  the real game (see `pkg/lineage`'s `Provenance` type in the code for the precise distinction).
- **Curriculum / task generator** — the part of this project (still being built — see
  `plans/07-curriculum-generator.md`) responsible for varying *what task* the policy practices from
  episode to episode (walk somewhere vs. mine a block vs. craft an item), so training doesn't
  overfit to one narrow skill.
- **Capability lock** — the failure mode a curriculum is meant to prevent: a policy that gets very
  good at one task and then, because it's never asked to do anything else, effectively "forgets" or
  never develops other useful skills. Named and explained in cRL-go's `rsi-with-rl.md` design note,
  §4.
- **Goal-conditioning** — a way of telling the policy, as part of its observation, *which* task it's
  currently supposed to be doing this episode — necessary once a curriculum varies tasks per
  episode, so the policy knows what it's being asked to do right now.
- **Action masking** — a mechanism that tells the policy which actions are actually legal/possible
  right now (e.g. "mine" only makes sense if there's something visible to mine), so it doesn't waste
  effort considering impossible actions.

## Minecraft / mc-agent specific

- **RCON** — "Remote CONsole," a Minecraft server protocol for running admin commands remotely
  (e.g. teleporting the bot, placing a block for it to mine). mc-agent uses this to set up
  repeatable training episodes.
- **Episode seeding** — using RCON to set up the world before an episode starts (e.g. placing a
  block nearby for a "mine" task), so the episode has something achievable to do.
- **rlenv** — the mc-agent Go package that wraps a live bot session so it looks like a standard RL
  "Environment" to cRL-go's training code. This is the bridge between the RL side (cRL-go) and the
  Minecraft side (mc-agent).
- **ReplayMod / `.mcpr`** — a Minecraft mod/file format for recording and replaying exactly what
  happened during a play session, reused by this project (rather than building its own replay
  tooling) so a training episode can be watched back later.

## Metrics (Prometheus / Grafana)

`cmd/rsi-train` exposes these as Prometheus metrics on `-metrics-addr` (default `:9400`) — see
[`../monitoring/README.md`](../monitoring/README.md) for the Grafana dashboard that graphs them.
Every name below is exactly the Prometheus metric name, so it doubles as a reference when writing
your own queries.

- **`rsi_generation`** — the current Teacher's generation number. Same thing as **Generation**,
  above, just exposed as a live gauge instead of only appearing in `pkg/lineage`'s saved records.
- **`rsi_current_round`** / **`rsi_current_epoch`** — the round and Student-training epoch
  currently in progress. `rsi_current_epoch` reads `-1` between rounds, before that round's first
  epoch has completed.
- **`rsi_rounds_total{outcome="won"|"lost"}`** — a running count of completed rounds, split by
  whether the Student won (was promoted) or the Teacher held. `won / (won + lost)` is the win rate.
- **`rsi_round_duration_seconds`** — a histogram of how long each completed round took,
  wall-clock. `rsi_round_duration_seconds_sum / rsi_round_duration_seconds_count` gives the mean.
- **`rsi_epoch_duration_seconds`** — a histogram of how long each completed Student-training epoch
  took, wall-clock (measured as the time between consecutive `OnEpoch` firings, or since the
  round's own start for that round's first epoch). Not simply
  `rsi_round_duration_seconds / rsi_epochs_per_generation`: task-mix variance across an epoch's
  trajectories (see `rsi_episode_length_steps`) means epochs within the same round aren't actually
  uniform in length, so this is measured directly rather than inferred — useful for telling apart
  "this round was slow because of fewer parallel envs or harder tasks" from "this round was slow
  because something stalled" (see the "Average epoch duration" dashboard panel).
- **`rsi_teacher_reward`** / **`rsi_student_reward`** — each side's mean evaluation reward from the
  most recently completed round (see **Reward**, above — these aren't normalized across task
  types, so compare Teacher vs. Student within one round rather than across rounds with a
  different task mix).
- **`rsi_epoch_avg_return`** — the average rollout return of the most recently completed
  Student-training epoch. This is the core "is it learning" signal — see **Epoch**, above.
- **`rsi_epochs_total`** — a running count of every Student-training epoch completed, across every
  round.
- **`rsi_epoch_samples_total`** — a running count of rollout steps consumed by Student training.
  `rate(rsi_epoch_samples_total[5m])` is the actual training throughput (samples/sec) — the
  "training rate" a Grafana panel graphs directly, rather than inferring it from log timestamps.
- **`rsi_gradient_updates_total`** — a running count of gradient-update steps applied during
  Student training. This complements samples/sec by showing how much optimizer work has actually
  completed.
- **`rsi_epoch_policy_loss`** / **`rsi_epoch_value_loss`** / **`rsi_epoch_entropy`** — the latest
  epoch's PPO replay diagnostics. Losses are averaged over real rollout steps and across PPO
  replay passes; entropy is reported in nats. These are objective signals, not episode rewards.
- **`rsi_epoch_advantage_mean`** / **`rsi_epoch_advantage_std`** / **`rsi_epoch_advantage_min`** /
  **`rsi_epoch_advantage_max`** — raw GAE advantage statistics before the batch normalization
  used by PPO. A large spread or drifting extrema can explain noisy policy updates even when
  average return is unchanged.
- **`rsi_epoch_gradient_norm`** / **`rsi_epoch_applied_gradient_norm`** — mean L2 norm of the
  sample-count-averaged gradient before and after configured Adam gradient clipping. A large gap
  means clipping is active; recurrent and plain paths can be compared directly using these gauges.
- **`rsi_episodes_started_total{task}`** / **`rsi_episode_steps_total{task}`** — counts of
  successfully reset episodes and environment steps, respectively. In curriculum runs, the task
  label is the task selected for that episode (`goto`, `mine`, `craft`, their `_far` variants, or the
  multi-step `chain_gather` / `chain_table` / `chain_place` / `chain_use`).
- **`rsi_episodes_total{task,outcome}`** — completed episodes split into `success`, `failure`, or
  `truncated` (a new reset began before the prior episode reported `Done`), with the same
  per-episode task label. Success means the final step's reward reached the current
  episode-success threshold of 5; it is a reward-based completion proxy, not a separate
  task-specific success predicate.
- **`rsi_reset_failures_total{task}`** / **`rsi_step_errors_total{task}`** — environment reset and
  step errors surfaced by the live environment wrapper. These are useful for spotting server or
  network health problems that may otherwise look like poor training.
- **`rsi_episode_length_steps{task,outcome,phase}`** — histogram of how many environment steps an
  episode actually ran before ending, by task, outcome, and `phase` (`"training"` | `"eval"`, via
  `evaluationInProgress` - the same flag `rsi_epoch_trajectories_started` uses). `truncated` lands
  at (or just under) whichever step cap was active almost by definition (100 for a training
  episode, 200 for evaluation's `-eval-episode-len`). The useful read is `success`/`failure`'s own
  distribution: short with a thin tail near the cap means episodes that don't finish are getting
  stuck early, not merely running a little long; a tail that hugs the cap means some near-misses
  really would finish with a bit more room. Added 2026-10-03 specifically to settle that question
  for `chain_place`/`chain_use` before changing any step-budget lever - see docs/plans/13's
  step-budget investigation. **The `phase` label was added later the same day**, after the metric's
  original 2-label version (no phase) made a success/truncated episode past 100 steps look
  impossible on the dashboard - it was actually an evaluation episode, legitimately using its own
  200-step budget, not a training one. The main dashboard's "Episode length by outcome" panel
  filters to `phase="training"` for exactly this reason.
- **`rsi_epoch_rollout_size`** / **`rsi_epoch_trajectories_started`** — the current epoch's target
  trajectory count (`config.Settings.RolloutSize`, set once at startup) and how many trajectories
  have started toward it so far (reset to 0 whenever `rsi_current_epoch` advances). cRL-go has no
  mid-epoch callback, so the latter is inferred from `instrumentedEnvironment.Reset` firing once per
  trajectory in real time, regardless of which cRL-go collection function is driving it.
  `100 * rsi_epoch_trajectories_started / rsi_epoch_rollout_size` is a true in-progress-epoch
  completion percentage, not just "which epoch index is running" (that's all `rsi_current_epoch`
  tells you). Added 2026-10-03. Pairs with `rsi_task_episodes_total`'s rate (already live
  throughout an epoch, not just at boundaries) for which task is actively running right now.
  **Bug found and fixed the same day:** `instrumentedEnvironment.Reset` fires for every episode on
  the shared connected environments, training rollout collection *and* each round's Teacher+Student
  evaluation alike - without a gate, a round's evaluation episodes (`EvalEpisodes * len(envs)` per
  side, entirely unrelated to `RolloutSize`) also counted toward `rsi_epoch_trajectories_started`
  after the last training epoch's own reset, pushing the progress bar past 100% during evaluation
  (caught live at 312%: 200 eval-episode resets / 64 rollout size, exactly 2 sides × 10
  eval-episodes × ~10 envs). Fixed with an `evaluationInProgress` flag
  (`cmd/rsi-train/metrics.go`), set by wrapping `leapfrog.Config.OnBeforeEvaluation` via
  `markEvaluationInProgress` and cleared once `leapfrog.Round` returns - `Reset` only increments the
  gauge while the flag is clear.
- **`rsi_epochs_per_generation`** — how many epochs make up one round (`-epochs-per-generation`,
  default `50`), set once at startup. `100 * (rsi_current_epoch + 1) / rsi_epochs_per_generation`
  is round progress - how far through its own epoch budget the current round is, distinct from
  `rsi_epoch_rollout_size`'s within-epoch progress (one is "how far through this epoch", the other
  is "how far through this round's epochs"). Added 2026-10-03 alongside the epoch-progress gauges.
- **`rsi_task_episodes_total{task="goto"|"mine"|"craft"}`** — how many episodes the curriculum has
  posed of each task type so far. Only present when `rsi-train` was started with
  `-curriculum-config` (see **Curriculum / task generator**, above) — without one, there's no
  per-episode task-selection hook for this command to count from at all.
- **`rsi_network_active_units_fraction{layer="1"|"2"}`** — fraction of that hidden layer's ReLU
  units that fire for at least one of a fixed set of probe observations. A dead unit never
  recovers; `0` in layer 2 means the network is dead (see `cmd/rsi-train/health.go`'s
  `deadLayer2Active`/`deadLogitStd` thresholds — the run aborts rather than keep training into a
  dead network).
- **`rsi_policy_logit_std`** — mean standard deviation of the policy's raw output logits (before
  softmax) across the same probes. `0` means the policy is identical in every state (dead); a
  rising value over time means the policy is becoming more confident/peaked, for better or worse.
- **`rsi_policy_entropy`** — mean Shannon entropy (nats) of `softmax(logits)` across the probes —
  how uncertain the actual action distribution is, as opposed to `rsi_policy_logit_std`'s raw
  pre-softmax spread. Falls as the policy sharpens; near `0` means it has become effectively
  deterministic for every probe (a possible sign of **softmax saturation** — see
  `plans/12-multi-step-crafting-curriculum.md`'s "Diagnosing a training plateau" section).
- **`rsi_network_unit_correlation{layer="1"|"2"}`** — mean absolute Pearson correlation, over every
  pair of that layer's units, of their activations across the probes. Not a dead-unit check (that's
  `rsi_network_active_units_fraction`) — a *capacity* signal: high means many units behave
  near-identically (redundant, unused headroom); low means units are already doing mutually
  distinct work (closer to actually capacity-constrained). See `plans/12`'s "Diagnosing a training
  plateau" section for how to read this alongside the others.
- **`rsi_best_return_alltime`** — the best Student-training epoch average return ever seen in this
  run, carried forward across restarts, rounds, *and* promotions (never reset). A value that stays
  flat across many rounds despite continued training is one signal — not proof on its own — that
  the architecture, not optimization, may be the ceiling.
- **`up{job="rsi-train"}`** — not one of this project's own metrics; Prometheus sets this
  automatically for every scrape target. `1` means the last scrape of `rsi-train`'s `/metrics`
  endpoint succeeded (the process is alive and reachable); `0` means it didn't (crashed, still
  starting up, or `-metrics-addr` is unreachable).

### Metrics and trainer restarts

Restarting `rsi-train` (for any reason — a code change, a config edit, a crash) resets its
in-process Prometheus state, but what that looks like on the dashboard differs by metric *type*,
which can look inconsistent until you know which kind you're looking at:

- **Gauges restored from an external source of truth** (`rsi_best_return_alltime`,
  `rsi_epoch_rollout_size`, `rsi_epochs_per_generation`, `rsi_teacher_reward`/`rsi_student_reward`
  and the rest of the per-round evaluation gauges) — these are explicitly re-`Set()` at startup from
  the checkpoint or a small persisted sidecar file (`<checkpoint-dir>/interim/round-eval.json`, see
  `cmd/rsi-train/roundeval.go`), so they read correctly immediately on resume, not just once the
  next epoch/round completes.
- **Counters and histograms** (`rsi_episodes_total`, `rsi_task_episodes_total`,
  `rsi_rounds_total`, `rsi_round_duration_seconds`, `rsi_episode_length_steps`, etc.) — these
  genuinely do reset to 0 in the exporter's memory on every restart; the Prometheus client library
  has no built-in way to persist them, and trying to hand-roll that would just duplicate what
  Prometheus's own server already does correctly. **The fix is in how you query them, not the
  metric itself**: Prometheus's own scrape history hasn't lost anything, and `increase()`/`rate()`
  already detect and correctly bridge a counter reset. Read the raw `/metrics` value directly (e.g.
  via `curl`) and you'll only see the total since the last restart; wrap the same metric in
  `increase(metric[30d])` (or `rate(metric[5m])` for a live/recent view) and you get the true
  cumulative total across every restart. Every dashboard panel showing a cumulative total ("Rounds
  won/lost", "Win rate", "Task mix", "Average round duration") uses `increase()` for exactly this
  reason — a raw instant-value query on any of these would misleadingly show 0 right after a
  restart. A short-window `rate()` panel (e.g. "Episode outcomes/sec", "Episode length by outcome")
  will still show a brief, expected gap immediately after a restart — not a bug, just that window
  not having refilled with fresh data yet.

## Where these terms come from

Most of the RL-specific terms above are standard across the field, not unique to this project. The
RSI/leapfrog/curriculum terms are this project's own vocabulary, defined precisely (with citations
to the exact design documents and code) in `plans/00-rsi-trainer-roadmap.md` and
`plans/01-curriculum-generator.md` — this glossary gives the short, beginner-friendly version; those
documents give the full, technical one.
