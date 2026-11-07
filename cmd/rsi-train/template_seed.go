package main

import (
	"context"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"sort"
	"strings"

	"math/rand/v2"

	"github.com/reallyoldfogie/mc-agent/agent"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
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
	cloneAreaFrom          *[3]float64 // nil when -clone-area-from is unset - seedTemplateOres has nothing to do
	cloneAreaRadius        int
	oreBlocks              []string
	oreCount               int
	oreMinDepth            int
	oreMaxDepth            int
	veinSizeMin            int
	veinSizeMax            int
	veinDiscardOnAirChance float64
}

// templateSeedConfig resolves f's -clone-area-from/-clone-area-radius and
// -seed-template-ore-* flags, plus curriculumPool's own buried-ore target
// block lists, into one templateSeedConfig - called once in main(), mirrors
// -clone-area-radius's own "0 falls back to -clear-area-radius" resolution
// (f's validateFlags doesn't persist that resolved value anywhere else, so
// this re-derives it the same way rather than assuming f.cloneAreaRadius
// alone is already final).
func (f flags) templateSeedConfig(curriculumPool *curriculum.Pool) templateSeedConfig {
	veinSizeMin, veinSizeMax := f.seedTemplateOreVeinSizeMin, f.seedTemplateOreVeinSizeMax
	if veinSizeMax > 0 && veinSizeMin > veinSizeMax {
		// Same swap-if-reversed convention as oreTemplatePositions' own
		// minDepth/maxDepth handling - an operator transposing the two
		// flags shouldn't silently produce an empty/degenerate range.
		veinSizeMin, veinSizeMax = veinSizeMax, veinSizeMin
	}
	cfg := templateSeedConfig{
		oreCount:               f.seedTemplateOreCount,
		oreMinDepth:            f.seedTemplateOreMinDepth,
		oreMaxDepth:            f.seedTemplateOreMaxDepth,
		veinSizeMin:            veinSizeMin,
		veinSizeMax:            veinSizeMax,
		veinDiscardOnAirChance: f.seedTemplateOreVeinDiscardOnAirChance,
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
	// Widened by -clone-area-jitter: seedTemplateOres needs ore spread
	// across every window -clone-area-jitter could ever pick, not just the
	// one centered on -clone-area-from itself - see -clone-area-jitter's
	// own doc comment for why a fixed single-window seeding would defeat
	// the whole point of jittering which window gets cloned.
	// episodeLimits.cloneAreaRadius (the actual per-Reset /clone box size)
	// is unaffected - only this seeding radius grows.
	cfg.cloneAreaRadius = radius + f.cloneAreaJitter
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

// veinShapeSeed derives a stable per-vein RNG seed from the block name and
// its chosen origin position - distinct from blockPositionSeed (which only
// governs the deterministic choice of *which* origins exist for a block,
// shared across every instance of that block) so two different origins for
// the same block don't stamp identical-looking blobs, while staying fully
// reproducible across restarts: no wall-clock/process-based randomness
// anywhere in this file, matching blockPositionSeed's own requirement.
func veinShapeSeed(block string, origin orePosition) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(fmt.Sprintf("%s@%d,%d,%d", block, origin.x, origin.y, origin.z)))
	return h.Sum64()
}

// veinPoint is one surviving point of a generated vein blob - a sphere
// center plus radius, ready to resolve into candidate cells.
type veinPoint struct {
	x, y, z float64
	radius  float64
}

// generateVeinBlob ports OreFeature.generate/generateVeinPart's shape
// algorithm - confirmed directly from decompiled 1.21.5 source,
// mc-data-gen/extractedSrc/1.21.5/net/minecraft/world/gen/feature/
// OreFeature.java, see docs/plans/21-buried-mining-success.md Phase 6c -
// scaled down for this template's much smaller working volume: lay `size`
// points along a randomly rotated horizontal line through origin, each
// with its own radius drawn from a sine-curve envelope (small at the
// line's two ends, biggest in the middle - this is what makes a vein a
// lumpy ellipsoid blob rather than a straight pipe or a random walk), then
// cull points whose spheres overlap too much with a neighboring point's
// (keeping the larger one), exactly mirroring OreFeature's own pairwise
// test.
//
// Deliberately does not port OreFeature's own chunk/height-limit
// bookkeeping (ChunkSectionCache, isOutOfHeightLimit, the
// OCEAN_FLOOR_WG top-of-world search) - this template is a small,
// already-loaded, flat volume with none of those concerns, and vanilla's
// own per-chunk attempt count / height distribution is already handled by
// this file's existing depth-band flags choosing where each vein's origin
// goes, not by this function. Also deliberately does not clamp a vein to
// stay inside -clone-area-radius/-clone-area-depth: a vein near the
// configured band's edge may stamp cells outside the nominal working
// area, same as vanilla terrain itself never special-cases a vein
// crossing a chunk boundary - the per-episode /clone that later copies a
// bounded box out of this template may simply truncate whatever falls
// outside it (resolved design decision, 2026-11-03; see Phase 6c).
func generateVeinBlob(rng *rand.Rand, origin orePosition, size int) []veinPoint {
	if size <= 0 {
		return nil
	}
	angle := rng.Float64() * math.Pi
	g := float64(size) / 8.0
	startX := float64(origin.x) + math.Sin(angle)*g
	endX := float64(origin.x) - math.Sin(angle)*g
	startZ := float64(origin.z) + math.Cos(angle)*g
	endZ := float64(origin.z) - math.Cos(angle)*g
	startY := float64(origin.y) + float64(rng.IntN(3)-2)
	endY := float64(origin.y) + float64(rng.IntN(3)-2)

	points := make([]veinPoint, size)
	for k := 0; k < size; k++ {
		f := float64(k) / float64(size)
		x := startX + (endX-startX)*f
		y := startY + (endY-startY)*f
		z := startZ + (endZ-startZ)*f
		h := rng.Float64() * float64(size) / 16.0
		radius := ((math.Sin(math.Pi*f)+1.0)*h + 1.0) / 2.0
		points[k] = veinPoint{x: x, y: y, z: z, radius: radius}
	}

	for k := 0; k < size-1; k++ {
		if points[k].radius <= 0 {
			continue
		}
		for m := k + 1; m < size; m++ {
			if points[m].radius <= 0 {
				continue
			}
			dx := points[k].x - points[m].x
			dy := points[k].y - points[m].y
			dz := points[k].z - points[m].z
			dr := points[k].radius - points[m].radius
			if dr*dr > dx*dx+dy*dy+dz*dz {
				if dr > 0 {
					points[m].radius = -1
				} else {
					points[k].radius = -1
				}
			}
		}
	}

	survivors := points[:0]
	for _, p := range points {
		if p.radius > 0 {
			survivors = append(survivors, p)
		}
	}
	return survivors
}

// veinCells deterministically resolves the full set of absolute,
// deduplicated candidate block positions one vein of the given size
// rooted at origin would stamp (see generateVeinBlob for the shape
// algorithm) - pulled out as its own pure function, the same way
// oreTemplatePositions is, so a vein's geometry is unit-testable without a
// live RCON connection. The caller is responsible for any air-exposure
// discard and for actually issuing /setblock per cell - this only
// resolves *where* a vein's blob touches, matching
// OreFeature.generateVeinPart's shape before its own placement-time
// discard check runs.
func veinCells(rng *rand.Rand, origin orePosition, size int) []orePosition {
	points := generateVeinBlob(rng, origin, size)
	seen := make(map[orePosition]bool)
	var cells []orePosition
	for _, p := range points {
		minX := int(math.Floor(p.x - p.radius))
		maxX := int(math.Floor(p.x + p.radius))
		minY := int(math.Floor(p.y - p.radius))
		maxY := int(math.Floor(p.y + p.radius))
		minZ := int(math.Floor(p.z - p.radius))
		maxZ := int(math.Floor(p.z + p.radius))
		for x := minX; x <= maxX; x++ {
			u := (float64(x) + 0.5 - p.x) / p.radius
			if u*u >= 1.0 {
				continue
			}
			for y := minY; y <= maxY; y++ {
				v := (float64(y) + 0.5 - p.y) / p.radius
				if u*u+v*v >= 1.0 {
					continue
				}
				for z := minZ; z <= maxZ; z++ {
					w := (float64(z) + 0.5 - p.z) / p.radius
					if u*u+v*v+w*w >= 1.0 {
						continue
					}
					cell := orePosition{x: x, y: y, z: z}
					if seen[cell] {
						continue
					}
					seen[cell] = true
					cells = append(cells, cell)
				}
			}
		}
	}
	return cells
}

// blockIsAir reports whether the block at (x,y,z) is air, cave air, or
// void air - the three states vanilla's own #minecraft:air tag covers.
// RCON here has no direct block-query command (see testenv.RCONHelper),
// so this probes via a conditional /say: the marker text only appears in
// the response when the /execute if condition actually holds.
func blockIsAir(ctx context.Context, rcon testenv.RCONHelper, x, y, z int) (bool, error) {
	marker := fmt.Sprintf("vein-exposure-probe-%d-%d-%d", x, y, z)
	resp, err := rcon.Exec(ctx, fmt.Sprintf("execute if block %d %d %d #minecraft:air run say %s", x, y, z, marker))
	if err != nil {
		return false, err
	}
	return strings.Contains(resp, marker), nil
}

// anyNeighborIsAir checks (x,y,z)'s 6 face-adjacent neighbors,
// short-circuiting on the first one found air - mirrors OreFeature's own
// isExposedToAir, which likewise only needs one exposed face to discard a
// block.
func anyNeighborIsAir(ctx context.Context, rcon testenv.RCONHelper, x, y, z int) (bool, error) {
	offsets := [6][3]int{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}
	for _, o := range offsets {
		air, err := blockIsAir(ctx, rcon, x+o[0], y+o[1], z+o[2])
		if err != nil {
			return false, err
		}
		if air {
			return true, nil
		}
	}
	return false, nil
}

// seedOreVein stamps one vein (see generateVeinBlob/veinCells) of block
// into the world, rooted at origin, with size drawn uniformly from
// [sizeMin, sizeMax]. discardOnAirChance mirrors
// OreFeatureConfig.discardOnAirChance / OreFeature.shouldPlace exactly:
// <= 0 disables the check entirely (every candidate cell is placed,
// matching this file's existing single-block seeding's own unconditional
// -replace behavior); above 0, a candidate cell is placed unconditionally
// unless it both fails the discard roll AND has at least one air
// neighbor - the roll is checked first so the (more expensive) neighbor
// query only runs when it can actually change the outcome.
//
// Does not check whether the existing block at a candidate cell is
// stone/dirt/anything else before overwriting it (vanilla's OreFeature
// does, via a RuleTest) - this template's dig band is already assumed
// solid ground by convention, the same assumption oreTemplatePositions'
// own single-block callers already make; adding that check would cost a
// second RCON round-trip per candidate cell for no live benefit on this
// template's flat, controlled geometry.
func seedOreVein(ctx context.Context, rcon testenv.RCONHelper, block string, origin orePosition, sizeMin, sizeMax int, discardOnAirChance float64) (int, error) {
	rng := rand.New(rand.NewPCG(veinShapeSeed(block, origin), 0))
	size := sizeMin
	if sizeMax > sizeMin {
		size = sizeMin + rng.IntN(sizeMax-sizeMin+1)
	}
	cells := veinCells(rng, origin, size)

	placed := 0
	for _, cell := range cells {
		if discardOnAirChance > 0 && rng.Float64() < discardOnAirChance {
			exposed, err := anyNeighborIsAir(ctx, rcon, cell.x, cell.y, cell.z)
			if err != nil {
				return placed, fmt.Errorf("checking air exposure for vein %s at (%d,%d,%d): %w", block, cell.x, cell.y, cell.z, err)
			}
			if exposed {
				continue
			}
		}
		if _, err := rcon.SetBlock(ctx, int64(cell.x), int64(cell.y), int64(cell.z), block, "replace").Exec(ctx); err != nil {
			return placed, fmt.Errorf("seeding vein block %s at (%d,%d,%d): %w", block, cell.x, cell.y, cell.z, err)
		}
		placed++
	}
	return placed, nil
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
//
// veinSizeMax > 0 switches every instance from a single `/setblock` to a
// vein (seedOreVein, size drawn from [veinSizeMin, veinSizeMax]) rooted at
// the exact same position oreTemplatePositions would otherwise place a
// single block at - instance selection itself is unchanged either way, so
// 6a/6b's existing density numbers (docs/plans/21-buried-mining-success.md
// Phase 6) stay meaningful; only the per-instance shape changes.
// veinSizeMax <= 0 preserves the original single-block behavior exactly,
// including skipping veinDiscardOnAirChance entirely.
func seedTemplateOres(ctx context.Context, rconAddress, rconPassword string, center [3]float64, radius int, blocks []string, countPerBlock, minDepth, maxDepth, veinSizeMin, veinSizeMax int, veinDiscardOnAirChance float64) error {
	if countPerBlock <= 0 || radius <= 0 || len(blocks) == 0 {
		return nil
	}

	rcon, err := agent.DialRCON(ctx, rconAddress, rconPassword)
	if err != nil {
		return fmt.Errorf("dialing RCON for template ore seeding: %w", err)
	}
	defer func() { _ = rcon.Close() }()

	veins := veinSizeMax > 0
	for _, block := range blocks {
		positions := oreTemplatePositions(block, center, radius, countPerBlock, minDepth, maxDepth)
		placedTotal := 0
		for _, p := range positions {
			if veins {
				n, err := seedOreVein(ctx, rcon, block, p, veinSizeMin, veinSizeMax, veinDiscardOnAirChance)
				if err != nil {
					return err
				}
				placedTotal += n
				continue
			}
			if _, err := rcon.SetBlock(ctx, int64(p.x), int64(p.y), int64(p.z), block, "replace").Exec(ctx); err != nil {
				return fmt.Errorf("seeding template ore %s at (%d,%d,%d): %w", block, p.x, p.y, p.z, err)
			}
			placedTotal++
		}
		if veins {
			log.Printf("seedTemplateOres: placed %d vein instance(s) of %s (%d total blocks) in the clone-area-from template", len(positions), block, placedTotal)
		} else {
			log.Printf("seedTemplateOres: placed %d instance(s) of %s in the clone-area-from template", placedTotal, block)
		}
	}
	return nil
}

// validateBuriedOreSeeding fails fast when the curriculum poses buried ore
// tasks against a clone template but template ore seeding is disabled or
// has nothing to place — those episodes would search empty stone forever
// (docs/plans/21-buried-mining-success.md Phase 0; found live before
// seedTemplateOres existed).
func validateBuriedOreSeeding(f flags, pool *curriculum.Pool) error {
	if pool == nil || f.cloneAreaFrom == "" {
		return nil
	}
	blocks := uniqueOreTargets(pool)
	if len(blocks) == 0 {
		return nil
	}
	if f.seedTemplateOreCount <= 0 {
		return fmt.Errorf("rsi-train: curriculum poses buried ore (%v) with -clone-area-from set, but -seed-template-ore-count is %d (disabled); refuse to train against an empty template", blocks, f.seedTemplateOreCount)
	}
	return nil
}
