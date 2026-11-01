# Buried Mining Success: From Zero Wins to Transferable Skill

## Status
IN PROGRESS 2026-11-01 (implementation on branches `21-buried-mining-success` in mc-rsi-trainer and mc-agent worktrees). Originally PROPOSED 2026-10-31. Builds on `14-buried-block-mining-and-memory-map.md` (feature existence)
and live runs that still report **zero `mine_buried` / `chain_mine` successes** after the
infrastructure in `14` landed. This document does **not** re-scope the original feature; it
diagnoses why the tasks remain unlearnable in practice and lays out a phased path to non-zero
success, then to a distribution closer to vanilla.

**Repos already in tree (do not rebuild):**
- Curriculum kinds `mine_buried` and `chain_mine` (`pkg/curriculum`, `rlenvadapter`).
- `Config.MineTargetBuried` / `ChainMine` + memory-map observation block
  (`unknownDx/Dy/Dz/unknownValid`, `mineKnown*`) in mc-agent `rlenv`.
- `ActionGoToTarget` re-enabled while `buriedTargetActive()` (without this, mask was Wait-only —
  confirmed live 2026-10-26 via `rsi_action_mask_total`).
- Per-quadrant arrival bonus + `digProgressBonus` (0.5) during buried search.
- Blind-destruction completion via `BuriedTargetAgent.BlockExistsInRadius` (Efficiency-V
  pathfinder can destroy ore before LOS ever opens).
- Startup template ore seeding (`cmd/rsi-train/template_seed.go`, defaults: count 60, depth 2–4).
- Tool seeding + enchantments for `mine_buried` (e.g. netherite + efficiency 5 in
  `configs/curriculum.json`).
- `task_weights` on `curriculum.Pool` (weighted `Select`) to counter multi-kind dilution.

**Live context this plan answers:**
- Run shape like `runs/multi-step-2026-10-24-recurrent-cold-256x3`: mixed curriculum including
  buried kinds, `-clone-area-from 0,-1,5000 -clone-area-radius 16 -clone-area-depth 8`,
  `seedTemplateOres` logging 60× `coal_ore`, trainer `episode_len` 200 / eval 400, still no
  buried completions worth calling success.

## Purpose
Get `mine_buried` (and then `chain_mine`) to a **reliably non-zero success rate under training**,
then harden the curriculum so a policy that wins on the teacher distribution still has a path
toward vanilla-like mining—without requiring full vanilla ore gen on day one.

## Non-goals
- Replacing pathfinding with a custom dig controller (may appear later; not required for first
  non-zero success).
- Full vanilla worldgen parity as a gate for phase 1–2.
- X-ray / chunk-query shortcuts (still forbidden; keep LOS + existence-for-reward-only split from
  `14`).
- Solving architecture competition / PvP interaction with buried tasks.

---

## Current failure mode (verified in code + live config)

### How a buried episode is supposed to work
1. Ore is **not** placed per-episode beside the bot (`MineTargetBuried`); it must already exist
   in the clone template / working-area copy.
2. `ActionMine` is legal **only when ore is currently visible**.
3. While unseen, the only search action is `ActionGoToTarget`, steered by
   `currentGoToTarget()` → nearest unexplored quadrant point from
   `observationUnknownDirection`.
4. Digging is mostly **pathfinder overburden clearing** on the way to that point, not a
   deliberate mine action.
5. Success = LOS opens and Mine completes, **or** blind pathfinder destruction detected via
   `BlockExistsInRadius`.

### Why that yields ~zero wins today

1. **Exploration Y does not match ore Y (geometry bug).**  
   Quadrant targets use `targetY = mineSearchAnchorY - MineSearchRadius/2`.  
   With `mine_buried_search_radius: 16`, that is **8 blocks below** the anchor.  
   Template ore defaults are only **2–4 blocks** below template Y.  
   The policy is told to dig past the ore band to a deep waypoint; after four such arrivals,
   `unknownValid` goes false and the search signal dies even though the ore layer was never
   systematically swept.

2. **Search is extremely coarse.**  
   Exactly **four** horizontal waypoints in a ~(2r+1)² footprint. Comments in
   `markExploredQuadrant` already note that finishing even those four is rare at real dig cost
   (multi-second breaks). No subdivision until budgets allow.

3. **Sparse synthetic ore still sparse relative to path luck.**  
   60 ores / 1089 cells ≈ 5.5% at radius 16. Random staircases to four deep points are a lottery,
   not a curriculum.

4. **Horizon vs dig cost.**  
   Episode length ~200 (eval 400) with each GoTo potentially a long dig/pathfind. Easy
   goto/mine/craft episodes still dominate the mixed pool (~1/11 kinds ≈ 9% buried share without
   weights).

5. **Weak intermediate reward vs time penalty.**  
   `digProgressBonus = 0.5` per confirmed break; live runs often use `-time-penalty 0.2`.  
   Digging that misses ore is slow negative return. Policy can learn “skip buried” while looking
   fine on the rest of the mix.

6. **`chain_mine` is strictly harder.**  
   Full gather→table→place→craft pickaxe, **then** the same buried search. Wooden pickaxe is
   enough for coal tier-wise, but the horizon stacks two hard skills. Training it before
   `mine_buried` works is wasted gradient.

7. **Teacher vs vanilla distribution (transfer, not day-one blocker).**  
   Dense shallow coal on flat clone ≠ vanilla veins/depth bands/terrain. Needed later for
   deployment; harmful as the *first* learning distribution if so sparse that reward never
   appears.

### What is *not* the current blocker (already fixed once)
- Empty template (ore never baked) — fixed by `seedTemplateOres`; confirm log line every startup.
- GoTo masked off for pure `mine_buried` — fixed via `goToTargetActive()` / `buriedTargetActive()`.
- Missing tool — fixed via `MineRequiredTool` (+ enchantments when configured).
- Clone depth shallower than ore max depth — use `-clone-area-depth` ≥ max ore depth (live runs
  used 8; flag default depth 3 is unsafe if ore max depth is 4 and depth flag is left default).

---

## Design principles
1. **Reachable success first, realism second.** Synthetic density/depth is a teacher distribution.
2. **One new hardness axis at a time.** Fix geometry → prove wins → densify search → mix weights →
   anneal toward vanilla-like sparsity.
3. **`mine_buried` before `chain_mine`.** Chain-buried is composition, not a substitute for the
   dig skill.
4. **Invariants vs variants.** Train dig/search/mine-when-visible as invariants; treat density,
   depth band, ore identity, terrain as scheduled variants.
5. **Eval must include a harder distribution than train** once acquisition works, or transfer
   gap stays invisible.

---

## Phased implementation

### Phase 0 — Diagnostics and guardrails (trainer + ops, short)
**Goal:** Know which failure mode a run is in before changing learning dynamics.
**Implemented 2026-11-01** (code/metrics/guardrails; live RCON checklist remains operator-side).

- [x] Startup always logs `seedTemplateOres: placed N instance(s) of <block>` for every buried
      target block; fail fast (or hard warn) if curriculum has buried kinds, `-clone-area-from`
      is set, and N==0 / seeding skipped.
- [x] Validate `-clone-area-depth >= seed-template-ore-max-depth` when both apply; error in
      `parseFlags` / `templateSeedConfig` rather than silently seeding outside the clone volume
      (default clone depth 3 vs default ore max depth 4 is a footgun).
- [x] Metrics (or log counters) per task kind:
  - episodes started / succeeded / hit step-cap / stuck-timeout (pre-existing + terminations)
  - for buried: `rsi_buried_search_episodes_total{ever_visible,outcome}` via `BuriedSearchStats`
  - quadrants: `rsi_buried_quadrants_explored`
- [x] One-shot RCON/operator checklist: after Reset into a working area, `coal_ore` exists under
      feet band (depth 2–4) inside clone radius.

**Done when:** A zero-success run can be classified as “no ore”, “never visible”, “visible but
never mined”, or “never finishes dig budget” from metrics alone.

### Phase 1 — Make `mine_buried` geometrically learnable (mc-agent `rlenv` + curriculum plumbing)
**Implemented 2026-11-01** (`MineBuriedExploreDepth` Config/TaskOverride + curriculum field; legacy `radius/2` when unset).
**Goal:** Exploration targets point at whatever ore band **this run configured**, not an
arbitrary `MineSearchRadius/2` depth baked into agent logic.

**Hard requirement — do not hardcode the distribution into the agent.**
Changing seed depth / vanilla-ish bands / eval sparsity must be a **config/curriculum change
only** (JSON + flags), not an mc-agent recompile. The agent only gains a *knob*; the trainer
owns the *value*. Same pattern as `MineSearchRadius`, `MineRequiredTool`, ore count flags:
rlenv does not know “coal is at Y−3 in our template,” it knows “this episode’s explore depth
is N.”

**Problem code (conceptual):**
`observationUnknownDirection` / `markExploredQuadrant` use
`targetY = mineSearchAnchorY - MineSearchRadius/2`.
Horizontal search radius is wrongly reused as vertical explore depth.

**Change (config-driven end-to-end):**

1. **mc-agent `rlenv.Config` + `TaskOverride`:** add something like
   `MineBuriedExploreDepth int` — blocks **below** the mine-search anchor used as the Y of
   quadrant waypoints / `unknownDy` when `buriedTargetActive()`.
   - `<= 0`: keep today’s behavior (`MineSearchRadius/2`) for backward compatibility **or**
     treat as “unset, caller must set” once curriculum always supplies it — pick one and
     document; prefer explicit unset→legacy only so old checkpoints/configs don’t silently
     change mid-flight without a flag.
   - Applied from `TaskOverride` in Reset like every other per-episode mine field.
   - **No** compile-time constant for “the” ore band (2–4, vanilla diamond band, etc.).

2. **mc-rsi-trainer curriculum:** add a pool field, e.g. `mine_buried_explore_depth`, on
   `curriculum.Pool` + JSON loader + `mineBuriedTask` / `chainMine` `ToOverride()` so every
   buried episode carries the depth the operator chose for that curriculum file.

3. **Keep ore *placement* depth on existing trainer flags**
   (`-seed-template-ore-min-depth` / `-seed-template-ore-max-depth`). Explore depth should be
   chosen to **agree** with that band for a given run (typically midpoint or max of the seed
   band). Optional later nicety: derive explore depth automatically as
   `(min+max)/2` from those flags when the curriculum field is omitted — still runtime, still
   no agent recompile. Do **not** make rlenv import trainer seed flags.

4. Horizontal half-radius can remain `MineSearchRadius/2` initially, or become its own config
   later; either way it stays a knob, not a distribution constant.

5. LOS / no-X-ray rules unchanged.

**Unit tests:**
- `MineSearchRadius=16`, `MineBuriedExploreDepth=3` → `unknownDy` aims ~3 down, not 8.
- `MineBuriedExploreDepth=10` (deeper eval band) → aims ~10 down with **no code change**.
- `markExploredQuadrant` uses the same configured depth.
- Non-buried mine tasks unchanged.

**Operator workflow when changing distribution (Phase 6 annealing included):**
```text
# acquisition: shallow dense
-seed-template-ore-min-depth 2 -seed-template-ore-max-depth 3
curriculum: "mine_buried_explore_depth": 3

# later: deeper / sparser teacher
-seed-template-ore-min-depth 8 -seed-template-ore-max-depth 16
curriculum: "mine_buried_explore_depth": 12
```
Restart trainer with new config only — **no mc-agent rebuild**.

**Done when:** Fake-agent tests pass for at least two distinct explore depths; a live smoke of
N `mine_buried`-only episodes shows bots pathing through the **configured** depth band
(RCON/trace or video), not systematically to `radius/2`; changing depth is demonstrated via
config alone.

### Phase 2 — Acquisition curriculum (mc-rsi-trainer config + light code)
**Implemented 2026-11-01** (config sample + wiring; live acquisition run still operator-owned).
**Goal:** Non-zero success rate on `mine_buried` in isolation.

**New config sample:** `configs/curriculum-mine-buried-sample.json` (mirror
`curriculum-sleep-sample.json` spirit):

- Only `mine_buried_*` populated (no chain, no far, optional tiny goto weight 0).
- Suggested starting knobs (tune from metrics, not dogma):
  - `mine_buried_search_radius`: **8–12** (was 16)
  - `mine_buried_explore_depth`: **3** (or midpoint of seed min/max — must match seed band)
  - `mine_buried_required_tool` + efficiency enchant kept for break-time
  - `-seed-template-ore-count`: **100–200**
  - `-seed-template-ore-min-depth` / `max-depth`: **2–3**, matching explore depth
  - `-clone-area-depth`: **≥ max depth** (explicit, not default 3)
  - trainer `episode_len`: **≥ 300** for this run (eval ≥ 400)
  - `-time-penalty`: start milder than 0.2 if dig still starves (e.g. 0.05–0.1) *or* raise
    dig shaping in Phase 3

**Wiring (required with Phase 1, not optional):** curriculum `mine_buried_explore_depth` →
`TaskOverride.MineBuriedExploreDepth` on every buried/`chain_mine` draw (and chain_mine after
craftedGoal uses the same field). Seed flag midpoint auto-fill is optional sugar only.

**Training procedure:**
1. Cold or warm-start from a competent multi-task checkpoint (movement/mine basics help).
2. Train until `mine_buried` success rate is clearly above noise (target: **≥ 5–10%** of buried
   episodes end in mine success, not merely “ran”).
3. Do **not** enable `chain_mine` yet.

**Done when:** Documented live run with non-zero buried success rate and metrics export; success
is not a single fluke episode.

### Phase 3 — Shaping and action-mask polish (mc-agent `rlenv`)
**Implemented 2026-11-01** (`digProgressBonus` 0.5→1.5; `depthProgressBonus` capped at explore depth).
**Goal:** Credit assignment matches “tunneling is good; missing ore is recoverable.”

- Revisit `digProgressBonus` (0.5 today): raise so multi-block dig outweighs several steps of
  time penalty under the acquisition time-penalty setting (candidate band 1.0–2.0; tune live).
- Consider a small **depth-progress** term while `buriedTargetUnresolved()` (first time below
  anchor − 1, −2, …) so pure horizontal pacing is worse than going underground—without paying
  for infinite bedrock diving past the ore band (cap depth bonus at explore depth).
- Keep per-quadrant arrival bonus, now that quadrant Y matches ore band.
- Confirm blind-destroy still ends episode + pays `mineRewardBonus` with real agent
  (`BuriedTargetAgent`).
- Optional: while buried and unresolved, ensure GoTo stays legal even after “arrived” at a
  waypoint until a *new* waypoint is selected (avoid one-step stall if mask/arrival interact
  poorly—verify with ActionMask metrics).

**Done when:** During acquisition runs, dig steps show higher average return than Wait; success
rate rises vs Phase 2 baseline under the same ore density.

### Phase 4 — Reintroduce mixture without dilution
**Implemented 2026-11-01** (`configs/curriculum.json` task_weights; `chain_mine: 0`).
**Goal:** Buried skill survives multi-task training.

- Use `task_weights` (already implemented):
  - High weight on `mine_buried` (e.g. 5–10).
  - Core goto/mine/craft at 1.
  - `chain_mine` weight **0** until Phase 5.
- Optionally keep a fraction of episodes on the dense acquisition ore settings while the mixed
  pool runs (two clone templates or two ore-count schedules is ideal but not required for v1;
  single denser template + weights may suffice).
- Regression: success rate on buried must not collapse to ~0 after N rounds of mixed training.

**Done when:** Mixed curriculum run maintains buried success above a documented floor (e.g. ≥ half
of isolated acquisition rate).

### Phase 5 — `chain_mine` composition
**Implemented 2026-11-01** (config `curriculum-with-chain-mine.json`; live non-zero bar still gated on Phase 2 acquisition).
**Goal:** Craft-then-dig works once dig alone works.

- Enable `chain_mine` with moderate weight only after Phase 4 floor holds.
- Prefer `chain_goal_items` that include `wooden_pickaxe` (or better) when target is coal; do not
  jump to ores that need iron/diamond tiers until tool progression exists.
- Expect longer `episode_len` for chain_mine episodes (same global len must fit the long tail, or
  accept lower success and weight accordingly).
- Metrics split: chain milestones reached vs final buried mine success (failure after craft vs
  failure before craft).

**Done when:** Non-zero `chain_mine` completions live; craft-then-mine path observed end-to-end.

### Phase 6 — Anneal teacher distribution toward vanilla-like (transfer)
**Implemented 2026-11-01** (eval protocol doc + sparse sample curriculum; live anneal runs operator-owned).
**Goal:** Reduce domain gap without extinguishing the skill.

Schedule (example; drive by eval, not calendar):

| Step | Density | Depth band | Notes |
|------|---------|------------|--------|
| 6a | high (acquisition) | shallow 2–3 | already done |
| 6b | medium | 2–6 | fewer ores, slightly deeper |
| 6c | low + multi-block veins | band per ore type | vein blobs, not salt |
| 6d | eval-only vanilla-ish worldgen | real bands | **train** may stay mixture; **eval** must include this |

- Keep a mixture: some dense episodes + some sparse (like experience replay).
- Add held-out eval: seeds/world never used in template seeding; report time-to-first-ore,
  success@N steps, depth histogram.
- Multiple ore types / tool tiers only after coal-scale success is stable.
- Document explicitly: matching full vanilla gen is an **eval/finetune** target, not a
  prerequisite for Phases 1–5.

**Done when:** Policy retains non-zero success on a sparser/deeper eval curriculum; gap to
vanilla is measured, not assumed.

### Phase 7 — Optional interface upgrades (only if 1–5 plateau)
Defer unless metrics say geometry+density+weights are insufficient:

- Denser waypoint set or one subdivision level inside each quadrant (quadtree)—only with higher
  step budget.
- Legal “dig forward/down” action during `buriedTargetUnresolved()` so search is not only
  pathfinder-luck.
- Per-task episode length overrides (buried/chain get longer horizons than goto).
- Memory beyond four bits (coarse occupancy) — larger observation change; needs checkpoint
  growth story (`cmd/widen-checkpoint` / resume-into-grown-shape already exists in-repo).

---

## Cross-repo touch points

| Area | Repo | Notes |
|------|------|--------|
| Explore depth / quadrant Y | mc-agent `rlenv` | Phase 1 core |
| digProgressBonus / depth shaping | mc-agent `rlenv/reward.go` | Phase 3 |
| Curriculum samples + weights | mc-rsi-trainer `configs/`, `pkg/curriculum` | Phases 2, 4, 5 |
| Ore seed count/depth + clone depth validation | mc-rsi-trainer `template_seed.go`, `main.go` flags | Phases 0, 2 |
| Metrics | mc-rsi-trainer `cmd/rsi-train` metrics | Phase 0 |
| Worldgen anneal | ops + possibly mc-agent test server presets | Phase 6 |

`go.mod` replace / version bump discipline stays as in `14` when mc-agent changes ship.

---

## Suggested implementation order (session-sized)

1. Phase 0 validations + metrics (half session).
2. Phase 1 explore-depth fix + unit tests (one session, mc-agent).
3. Phase 2 buried-only sample config + live acquisition run (ops + train time).
4. Phase 3 shaping tweak if success still weak (short).
5. Phase 4 weights into full curriculum.
6. Phase 5 chain_mine.
7. Phase 6 annealing + vanilla-ish eval when deployment cares.

Do not start Phase 5–7 until Phase 2’s non-zero bar is met.

---

## Done when (document-level)
- [x] Phase 1 merged: buried exploration Y matches configurable/shallow ore band; tests green.
- [ ] Live `mine_buried`-only run with sustained success rate ≥ 5% (or better agreed floor).
- [ ] Mixed curriculum with weights keeps buried success above an agreed floor.
- [ ] `chain_mine` shows non-zero full completions after `mine_buried` works.
- [x] Written eval protocol for denser-vs-sparser (and later vanilla-ish) distributions exists
      under `docs/buried-mining-eval-protocol.md`.
- [x] Clone-depth vs ore-depth footgun cannot silently recur (`parseFlags` validates).

---

## References
- `docs/plans/14-buried-block-mining-and-memory-map.md` — original feature design, template bake-in,
  two task types, no-X-ray constraint.
- `docs/plans/12-multi-step-crafting-curriculum.md` — chain stages `chain_mine` extends.
- `docs/plans/07-curriculum-generator.md` — uniform selection; difficulty progression deferred
  (partially superseded by `task_weights` on `Pool`).
- mc-agent `rlenv/environment.go` — `buriedTargetActive`, `currentGoToTarget`,
  `observationUnknownDirection`, `markExploredQuadrant`, `BuriedTargetAgent`.
- mc-agent `rlenv/reward.go` — `arrivalBonus` per-quadrant, `digProgressBonus`, `mineRewardBonus`.
- mc-agent `rlenv/action.go` — `ActionMine` requires `mineVisible`; GoTo via `goToTargetActive()`.
- `cmd/rsi-train/template_seed.go` — `defaultSeedTemplateOreCount/MinDepth/MaxDepth`, deterministic
  positions.
- `configs/curriculum.json` — production mix including buried fields.
- `configs/curriculum-full.json` — full action pool + example `task_weights`.
- Live: `runs/multi-step-2026-10-24-recurrent-cold-256x3/` (seeding confirmed, successes still
  missing at plan time).
