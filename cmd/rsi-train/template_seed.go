package main

import (
	"context"
	"fmt"
	"hash/fnv"
	"log"
	"sort"

	"math/rand/v2"

	"github.com/reallyoldfogie/mc-agent/agent"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// This file generalizes runs/buried-block-mining-2026-10-14/gen_template.py
// into the trainer's own startup path: that script produced setblock
// command *text* (ore_cmds.txt/surface_cmds.txt) for a human to copy into
// RCON by hand, for one hardcoded block (coal_ore) at one hardcoded
// location. Found live, 2026-10-28: for the run actually in flight, that
// manual step was never done — every coordinate the script generated
// (re-checked via `execute if block ... minecraft:coal_ore`) came back
// empty, and a broader RCON probe around the template found zero ore
// blocks of any kind. mine_buried's and chain_mine's own task config
// deliberately skips seeding a fresh, visible ore instance per episode
// (Config.MineTargetBuried's whole point is "the target already exists,
// baked into the shared clone template" - see
// pkg/curriculum/rlenvadapter/adapter.go's mineBuriedTask.ToOverride),
// so with nothing actually baked into the template, those tasks had been
// searching an empty volume since this run began - no exploration,
// timeout-budget, or reward-shaping fix could ever have produced a
// success against that, independent of anything else changed this week.

// defaultSeedTemplateOreCount/MinDepth/MaxDepth are -seed-template-ore-*'s
// own defaults - deliberately NOT gen_template.py's original values (10
// instances on a single fixed plane 7 blocks down): with the clone
// template's horizontal footprint capped at (2*radius+1)^2 (1089 cells at
// this run's own -clone-area-radius 16), 10 instances on one Y plane gives
// a bot digging anywhere nearby an astronomically small chance of ever
// crossing one (found live, 2026-10-28 - zero successes in ~18 hours and
// 300+ mine_buried episodes once the *actual* root cause, zero ore
// anywhere near the template, was confirmed). A higher count and a
// shallow, multi-layer band instead of one fixed plane both raise the
// odds a bot's ordinary downward digging encounters one quickly, without
// making it instantly visible (minDepth > 0 still means real digging, not
// a surface-exposed block) or trivial to memorize a single exact Y.
const (
	defaultSeedTemplateOreCount    = 60
	defaultSeedTemplateOreMinDepth = 2
	defaultSeedTemplateOreMaxDepth = 4
)

// uniqueOreTargets collects every block name mine_buried/chain_mine's own
// curriculum fields can pose this run (Pool.MineBuriedTargetBlocks and
// Pool.ChainMineTargetBlocks - the two fields DefaultEpisodeSeeder's own
// MineTargetBuried branch promises are "already present" rather than
// freshly placed per episode, see mineBuriedTask/chainMineTask's own
// ToOverride), deduplicated and sorted so seedTemplateOres's own
// deterministic-per-block-name seeding (see its doc comment) iterates in a
// stable order run to run. nil/empty if curriculumPool is nil or neither
// field is set - seedTemplateOres then has nothing to do, matching every
// other curriculum-gated feature in this file's "nil pool, no-op" norm.
func uniqueOreTargets(pool *curriculum.Pool) []string {
	if pool == nil {
		return nil
	}
	seen := make(map[string]bool)
	var blocks []string
	for _, block := range pool.MineBuriedTargetBlocks {
		if block != "" && !seen[block] {
			seen[block] = true
			blocks = append(blocks, block)
		}
	}
	for _, block := range pool.ChainMineTargetBlocks {
		if block != "" && !seen[block] {
			seen[block] = true
			blocks = append(blocks, block)
		}
	}
	sort.Strings(blocks)
	return blocks
}

// templateSeedConfig bundles everything connectEnvironments needs to call
// seedTemplateOres once, resolved from flags/curriculumPool up front so
// that function itself only has plain values to work with - mirrors
// episodeLimits' own role for the clone-area-from fields it already
// carries, kept separate rather than folded into episodeLimits since those
// fields describe every environment's own per-Reset behavior, while this
// describes a one-time startup action unrelated to any one environment.
type templateSeedConfig struct {
	cloneAreaFrom   *[3]float64 // nil when -clone-area-from is unset - seedTemplateOres has nothing to do
	cloneAreaRadius int
	oreBlocks       []string
	oreCount        int
	oreMinDepth     int
	oreMaxDepth     int
}

// templateSeedConfig resolves f's -clone-area-from/-clone-area-radius and
// -seed-template-ore-* flags, plus curriculumPool's own buried-ore target
// block lists, into one templateSeedConfig - called once in main(), mirrors
// -clone-area-radius's own "0 falls back to -clear-area-radius" resolution
// (f's validateFlags doesn't persist that resolved value anywhere else, so
// this re-derives it the same way rather than assuming f.cloneAreaRadius
// alone is already final).
func (f flags) templateSeedConfig(curriculumPool *curriculum.Pool) templateSeedConfig {
	cfg := templateSeedConfig{
		oreCount:    f.seedTemplateOreCount,
		oreMinDepth: f.seedTemplateOreMinDepth,
		oreMaxDepth: f.seedTemplateOreMaxDepth,
	}
	if f.cloneAreaFrom == "" {
		return cfg
	}
	from := f.cloneAreaFromParsed
	cfg.cloneAreaFrom = &from
	radius := f.cloneAreaRadius
	if radius <= 0 {
		radius = f.clearAreaRadius
	}
	cfg.cloneAreaRadius = radius
	cfg.oreBlocks = uniqueOreTargets(curriculumPool)
	return cfg
}

// blockPositionSeed derives a stable 64-bit seed from blockName alone
// (fnv-1a, not time- or process-based) so seedTemplateOres picks the exact
// same set of positions for a given block name every time it runs - repeat
// trainer startups (the common case: every restart of a long-lived
// supervised run re-applies this, matching applySharedServerGamerules's own
// "just always re-apply, no state to track" precedent) must not accumulate
// a growing, drifting set of ore positions by drawing a fresh random
// sample each time.
func blockPositionSeed(blockName string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(blockName))
	return h.Sum64()
}

// orePosition is one block instance seedTemplateOres resolves before
// issuing its /setblock - absolute world coordinates, already offset from
// center.
type orePosition struct{ x, y, z int }

// oreTemplatePositions deterministically chooses up to count distinct
// horizontal offsets within radius of (0,0) for block (a full
// shuffle-and-take-first-N over every offset in range, mirroring
// gen_template.py's own approach exactly, just seeded per block name via
// blockPositionSeed instead of a single hardcoded constant - see that
// function's own doc comment for why determinism matters here), each
// resolved to an absolute position against center with a Y uniformly drawn
// between minDepth and maxDepth blocks below center's own Y. Pulled out of
// seedTemplateOres as its own pure function so the choice of *which*
// positions get seeded is unit-testable without a live RCON connection.
// Returns fewer than count positions only if radius is small enough that
// the whole (2*radius+1)^2 footprint has fewer cells than count.
func oreTemplatePositions(block string, center [3]float64, radius, count, minDepth, maxDepth int) []orePosition {
	if minDepth > maxDepth {
		minDepth, maxDepth = maxDepth, minDepth
	}
	type offset struct{ dx, dz int }
	all := make([]offset, 0, (2*radius+1)*(2*radius+1))
	for dx := -radius; dx <= radius; dx++ {
		for dz := -radius; dz <= radius; dz++ {
			all = append(all, offset{dx, dz})
		}
	}

	rng := rand.New(rand.NewPCG(blockPositionSeed(block), 0))
	rng.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })

	if count > len(all) {
		count = len(all)
	}
	centerX, centerY, centerZ := int(center[0]), int(center[1]), int(center[2])
	depthSpan := maxDepth - minDepth + 1
	positions := make([]orePosition, count)
	for i, o := range all[:count] {
		y := centerY - minDepth
		if depthSpan > 1 {
			y = centerY - minDepth - rng.IntN(depthSpan)
		}
		positions[i] = orePosition{x: centerX + o.dx, y: y, z: centerZ + o.dz}
	}
	return positions
}

// seedTemplateOres generalizes gen_template.py's ore-placement half (its
// surface-variety half is cosmetic and out of scope here - see this file's
// own doc comment) into a repeatable, multi-block-type trainer startup
// step: for each of blocks, places oreTemplatePositions' own chosen
// instances via one `/setblock ... replace` each. No-op (returns nil
// immediately) if countPerBlock <= 0, radius <= 0, or blocks is empty -
// the -clone-area-from-only, no-ore-blocks-configured case this must
// leave exactly as it already behaved.
//
// Always safe to call on every trainer startup, not just a fresh run's
// first one: setblock is idempotent (an already-correct block is a no-op),
// and oreTemplatePositions' own determinism means the positions themselves
// never drift between calls - this is what makes it safe to wire in
// unconditionally rather than behind some "is this template already
// seeded" check this function would otherwise need and get wrong.
func seedTemplateOres(ctx context.Context, rconAddress, rconPassword string, center [3]float64, radius int, blocks []string, countPerBlock, minDepth, maxDepth int) error {
	if countPerBlock <= 0 || radius <= 0 || len(blocks) == 0 {
		return nil
	}

	rcon, err := agent.DialRCON(ctx, rconAddress, rconPassword)
	if err != nil {
		return fmt.Errorf("dialing RCON for template ore seeding: %w", err)
	}
	defer func() { _ = rcon.Close() }()

	for _, block := range blocks {
		positions := oreTemplatePositions(block, center, radius, countPerBlock, minDepth, maxDepth)
		for _, p := range positions {
			if _, err := rcon.SetBlock(ctx, int64(p.x), int64(p.y), int64(p.z), block, "replace").Exec(ctx); err != nil {
				return fmt.Errorf("seeding template ore %s at (%d,%d,%d): %w", block, p.x, p.y, p.z, err)
			}
		}
		log.Printf("seedTemplateOres: placed %d instance(s) of %s in the clone-area-from template", len(positions), block)
	}
	return nil
}
