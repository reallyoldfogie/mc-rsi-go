# mc-rsi-trainer docs

Welcome — this page is the front door to this project's documentation. It's written for someone
who is new to the project, and possibly new to reinforcement learning (RL) too. If you already
know this codebase well, you probably want `docs/plans/` instead (see "Where the implementation
plans live," below).

## What is this project, in plain language?

Imagine you want to teach a computer program to play Minecraft well — walk somewhere, mine a
block, craft an item — by having it practice and get better over time, rather than by
hand-programming every move. That's reinforcement learning: the program (the "policy") tries
things in the game, gets a numeric score ("reward") for how well it did, and gradually adjusts
itself to get better scores.

This project trains such a policy using a specific technique this project's design docs call
**leapfrog self-play**: keep a "Teacher" (the current best policy) and a "Student" (a copy of the
Teacher that keeps practicing). Periodically, compare them. If the Student has gotten better, it
becomes the new Teacher, and a new Student is spun up from it to keep improving. Over many rounds,
this produces a policy that keeps getting better without a human manually redesigning it after each
round — hence "Recursively/Iteratively Self-Improving" (RSI).

See `glossary.md` for definitions of every term used above and elsewhere in these docs, and
`how-leapfrog-works.md` (once written — see `plans/09-beginner-documentation.md`'s tracking table)
for a longer, example-driven walkthrough.

## Why three repositories?

This project (`mc-rsi-trainer`) is one of three sibling repositories that work together:

- **cRL-go** — the reinforcement-learning "engine": the math and algorithms (how a policy learns
  from reward), independent of Minecraft.
- **mc-agent** — the actual Minecraft bot: it connects to a real Minecraft server and can move,
  mine, fight, craft, etc.
- **mc-rsi-trainer** (this repo) — sits on top of both: it's the "coach" that runs the
  Teacher-vs-Student training loop, using cRL-go's learning algorithms to train a policy, and
  mc-agent's bot to actually try things out in a real Minecraft world.

See `architecture.md` for a fuller explanation of how these fit together, with a diagram.

## Where to go next

- **New to the project?** Read `architecture.md`, then `glossary.md` as you hit unfamiliar terms.
- **Want to set this up and run it yourself?** Read `getting-started.md`.
- **Want to know what's actually built vs. still planned?** Read `plans/00-rsi-trainer-roadmap.md`
  — it's more technical, but its "Status" section at the top always has the current, honest picture
  (this project's documentation culture insists on re-verifying claims against real code rather
  than trusting stale status headers, so that section is kept current).
- **Want to see the full implementation plan?** Read through `plans/00` onward
  (`01`, `02`, `03`, ... in order) — each is a self-contained, numbered step toward a fully working
  trainer.

## A note on how this documentation is organized

This `docs/` folder (beginner-facing explanations) is deliberately separate from `docs/plans/`
(implementation plans for whoever builds the next piece, written more densely and technically).
If something in `docs/plans/` seems to disagree with something here, `docs/plans/` is more likely to
be current — this project's plans get updated the moment new facts are discovered; the beginner
docs get updated in step, but there can be a short lag. See `plans/09-beginner-documentation.md` for
how the two are meant to stay in sync.
