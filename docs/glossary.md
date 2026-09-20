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
- **`rsi_task_episodes_total{task="goto"|"mine"|"craft"}`** — how many episodes the curriculum has
  posed of each task type so far. Only present when `rsi-train` was started with
  `-curriculum-config` (see **Curriculum / task generator**, above) — without one, there's no
  per-episode task-selection hook for this command to count from at all.
- **`up{job="rsi-train"}`** — not one of this project's own metrics; Prometheus sets this
  automatically for every scrape target. `1` means the last scrape of `rsi-train`'s `/metrics`
  endpoint succeeded (the process is alive and reachable); `0` means it didn't (crashed, still
  starting up, or `-metrics-addr` is unreachable).

**A known gap:** `mc-agent`'s Mine/Craft actions log occasional confirmation-timeout errors (a
fixed 2-second window that a busy shared server can miss under load) straight to the training log,
not to any of the metrics above — `rlenv.Environment.Step` doesn't currently surface that signal in
a way `cmd/rsi-train` could turn into a counter. Watch the log (`grep -c "Mine error\|Craft error"`)
for that specific rate today; exposing it as a real metric would need a small `mc-agent` change
first.

## Where these terms come from

Most of the RL-specific terms above are standard across the field, not unique to this project. The
RSI/leapfrog/curriculum terms are this project's own vocabulary, defined precisely (with citations
to the exact design documents and code) in `plans/00-rsi-trainer-roadmap.md` and
`plans/01-curriculum-generator.md` — this glossary gives the short, beginner-friendly version; those
documents give the full, technical one.
