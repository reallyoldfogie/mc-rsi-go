# Buried mining evaluation protocol

Companion to `docs/plans/21-buried-mining-success.md` Phase 6.

## Metrics to watch
- `rsi_episodes_total{task="mine_buried",outcome="success"}` rate vs started
- `rsi_buried_search_episodes_total{ever_visible,outcome}` — if `ever_visible=false` dominates, geometry/density still broken
- `rsi_buried_quadrants_explored` — mass at 0–1 means bots never finish search waypoints
- `rsi_episode_terminations_total{task="mine_buried",reason}` — `mine` vs step-cap / stuck

## Acquisition bar (Phase 2)
Train on `configs/curriculum-mine-buried-sample.json` until success rate of completed `mine_buried` episodes is **≥ 5–10%** sustained (not a single fluke).

## Mixed bar (Phase 4)
`configs/curriculum.json` with `mine_buried` weight high and `chain_mine` 0. Buried success floor ≥ half of acquisition rate.

## Anneal schedule
| Step | Curriculum | Seed flags (examples) |
|------|------------|------------------------|
| 6a acquisition | `curriculum-mine-buried-sample.json` | count 150, depth 2–3, explore_depth 3 |
| 6b medium | `curriculum-mine-buried-sparse.json` | count 40–60, depth 4–10, explore_depth 8 |
| 6c low | custom / lower count, multi-block veins if implemented | match explore_depth to band |
| 6d vanilla-ish eval | held-out worldgen; **do not** use the training template seed | measure time-to-first-ore, success@N |

Changing bands is **config-only** (`mine_buried_explore_depth` + seed depth flags). No mc-agent rebuild.

## Operator checklist after Reset
RCON: confirm `coal_ore` exists under the working area within the configured depth band inside clone radius.
