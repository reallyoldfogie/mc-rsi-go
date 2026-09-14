# Architecture

This page explains how this project's three sibling repositories fit together, and what lives
where. If a term below is unfamiliar, check `glossary.md`.

## The three repositories

```
   ┌─────────────────────────────────────────────────────────────┐
   │                       mc-rsi-trainer  (this repo)            │
   │   "the coach" — runs Teacher vs. Student training rounds,    │
   │   decides which tasks to practice, saves progress            │
   └───────────────┬───────────────────────────────┬─────────────┘
                    │ uses                          │ uses
                    ▼                                ▼
   ┌────────────────────────────┐    ┌───────────────────────────────┐
   │          cRL-go             │    │           mc-agent              │
   │  "the learning engine" —    │    │  "the Minecraft bot" —          │
   │  generic RL algorithms      │    │  connects to a real server,     │
   │  (how a policy learns from  │    │  can move/mine/craft/fight,     │
   │  reward), no Minecraft      │    │  and exposes itself as an       │
   │  knowledge at all           │    │  RL "Environment" via `rlenv`   │
   └────────────────────────────┘    └───────────────────────────────┘
```

- **cRL-go** never imports mc-agent, and doesn't know Minecraft exists. It provides the
  environment-agnostic pieces: how to represent an observation/action/reward, how to train a policy
  from experience (the PPO and REINFORCE algorithms), how to save/load a trained policy
  (checkpoints), and increasingly, richer things like hierarchical policies and action masking. It's
  deliberately kept small and dependency-light — see `plans/00-rsi-trainer-roadmap.md`'s "Why a
  third repo" section for the full reasoning.
- **mc-agent** never imports cRL-go's training code as a dependency it drives itself — it exposes a
  live Minecraft bot session as something *conforming to* cRL-go's `Environment` interface (in its
  `rlenv` package), so cRL-go's generic training code can drive it without mc-agent needing to know
  anything about how training works. mc-agent is a large, fast-moving project with lots of
  Minecraft-protocol code that has nothing to do with RL at all — pathfinding, combat, crafting,
  chat commands, etc.
- **mc-rsi-trainer** (this repo) is the only one of the three that imports both of the others. It
  doesn't reimplement anything either already does — it orchestrates: decide when to compare
  Teacher and Student, decide what task to pose this episode (once the curriculum generator is
  built — see `plans/07`), and keep the record of which trained policy is the current best one
  (`pkg/lineage`).

## Why not put this in cRL-go or mc-agent?

Four reasons, elaborated fully in `plans/00-rsi-trainer-roadmap.md`:

1. Pulling Minecraft-specific dependencies into cRL-go would force every user of cRL-go's generic
   RL code (even someone just using it for a toy grid-world example) to pull in a much bigger
   dependency tree.
2. mc-agent's own design already assumes it's the one depending on cRL-go, not the reverse — so a
   trainer that needs both has to sit above both.
3. The two repos release at different paces — cRL-go is a deliberately small, careful library;
   mc-agent is fast-moving. A training/curriculum layer would force unwanted churn on whichever one
   it lived inside.
4. It matches a design both repos already describe on their own: an "RL Trainer" sitting above an
   Environment boundary, separate from the environment/client code itself.

## What exists today vs. what's planned

As of this writing, two concrete pieces of this repo are built: `pkg/lineage` — the bookkeeping for
"which trained policy is generation N, and how did it get there" — and `pkg/leapfrog` — the actual
Teacher-vs-Student comparison loop (train a Student, evaluate both sides, report the winner).
Everything else (the curriculum/task generator, the runnable training entrypoint that ties
`pkg/leapfrog` and `pkg/lineage` together) is planned but not yet built — see
`plans/00-rsi-trainer-roadmap.md`'s "Status" section for the current, honest picture, and
`plans/04` onward for the step-by-step plan to build the rest.

## How a training run will work, once built

At a high level (see `how-leapfrog-works.md`, once written, for the full walkthrough):

1. This repo loads the current Teacher (or starts a fresh one, if there isn't one yet) and connects
   to a live Minecraft session via mc-agent's `rlenv`.
2. It clones the Teacher into a Student and lets the Student keep training, trying more new things
   than the Teacher would.
3. After a while, it runs both Teacher and Student through the same evaluation tasks and compares
   their average performance.
4. If the Student did better, it becomes the new Teacher, its progress gets saved
   (`pkg/lineage.Save`), and the cycle repeats with a fresh Student.
5. Over many rounds, this produces a policy that keeps improving — and, once the curriculum
   generator (`plans/07`) is built, one that's exercised against a variety of tasks each round
   rather than just one, so it doesn't get stuck being good at only one thing.
