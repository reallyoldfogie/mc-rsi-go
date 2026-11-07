package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

func TestUniqueOreTargetsDedupesAndSortsAcrossBothFields(t *testing.T) {
	pool := &curriculum.Pool{
		MineBuriedTargetBlocks: []string{"minecraft:coal_ore", "minecraft:iron_ore", ""},
		ChainMineTargetBlocks:  []string{"minecraft:coal_ore", "minecraft:diamond_ore"},
	}
	got := uniqueOreTargets(pool)
	want := []string{"minecraft:coal_ore", "minecraft:diamond_ore", "minecraft:iron_ore"}
	if len(got) != len(want) {
		t.Fatalf("uniqueOreTargets() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("uniqueOreTargets() = %v, want %v", got, want)
		}
	}
}

func TestUniqueOreTargetsNilPool(t *testing.T) {
	if got := uniqueOreTargets(nil); got != nil {
		t.Fatalf("uniqueOreTargets(nil) = %v, want nil", got)
	}
}

// TestOreTemplatePositionsDeterministic confirms two independent calls for
// the same block name produce the exact same set of positions - the
// property seedTemplateOres' own doc comment relies on to stay idempotent
// across repeated trainer startups (a supervised run restarts routinely)
// instead of accumulating a growing, drifting set of ore across restarts
// that each drew a fresh random sample.
func TestOreTemplatePositionsDeterministic(t *testing.T) {
	center := [3]float64{0, -1, 5000}
	first := oreTemplatePositions("minecraft:coal_ore", center, 16, 10, 7, 7)
	second := oreTemplatePositions("minecraft:coal_ore", center, 16, 10, 7, 7)
	if len(first) != len(second) {
		t.Fatalf("got %d and %d positions on two calls, want equal", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("position %d differs between calls: %v vs %v", i, first[i], second[i])
		}
	}
}

// TestOreTemplatePositionsDifferentBlocksDiffer confirms distinct block
// names don't collide on the exact same position set - otherwise two
// different ore types configured together would just overwrite each
// other's placements.
func TestOreTemplatePositionsDifferentBlocksDiffer(t *testing.T) {
	center := [3]float64{0, -1, 5000}
	coal := oreTemplatePositions("minecraft:coal_ore", center, 16, 10, 7, 7)
	iron := oreTemplatePositions("minecraft:iron_ore", center, 16, 10, 7, 7)
	identical := len(coal) == len(iron)
	if identical {
		for i := range coal {
			if coal[i] != iron[i] {
				identical = false
				break
			}
		}
	}
	if identical {
		t.Fatalf("minecraft:coal_ore and minecraft:iron_ore produced identical position sets: %v", coal)
	}
}

// TestOreTemplatePositionsWithinRadiusAndDepthBand confirms every returned
// position's horizontal offset from center stays within radius and its Y
// falls within [center.Y-maxDepth, center.Y-minDepth] - the band
// seedTemplateOres promises, and the one a search radius smaller than this
// band (mine_buried_search_radius vs -clone-area-radius, see this
// session's own live findings) would otherwise silently violate.
func TestOreTemplatePositionsWithinRadiusAndDepthBand(t *testing.T) {
	center := [3]float64{0, -1, 5000}
	radius, minDepth, maxDepth := 16, 5, 9
	positions := oreTemplatePositions("minecraft:coal_ore", center, radius, 10, minDepth, maxDepth)
	if len(positions) != 10 {
		t.Fatalf("len(positions) = %d, want 10", len(positions))
	}
	seen := make(map[orePosition]bool)
	for _, p := range positions {
		if seen[p] {
			t.Fatalf("duplicate position %v", p)
		}
		seen[p] = true
		dx, dz := p.x-int(center[0]), p.z-int(center[2])
		if dx < -radius || dx > radius || dz < -radius || dz > radius {
			t.Fatalf("position %v outside radius %d of center %v", p, radius, center)
		}
		minY, maxY := int(center[1])-maxDepth, int(center[1])-minDepth
		if p.y < minY || p.y > maxY {
			t.Fatalf("position %v's Y outside depth band [%d, %d]", p, minY, maxY)
		}
	}
}

// TestOreTemplatePositionsCountCappedByFootprint confirms asking for more
// positions than a small radius's footprint actually has returns only
// what exists, rather than panicking or looping forever.
func TestOreTemplatePositionsCountCappedByFootprint(t *testing.T) {
	center := [3]float64{0, -1, 5000}
	positions := oreTemplatePositions("minecraft:coal_ore", center, 1, 100, 7, 7)
	want := (2*1 + 1) * (2*1 + 1)
	if len(positions) != want {
		t.Fatalf("len(positions) = %d, want %d (the whole radius-1 footprint, since 100 was asked for)", len(positions), want)
	}
}

// TestVeinShapeSeedDeterministicAndDistinct confirms the seed depends on
// both block name and origin - two origins for the same block, or the
// same origin for two blocks, must not collide, mirroring
// TestOreTemplatePositionsDifferentBlocksDiffer's reasoning for
// blockPositionSeed.
func TestVeinShapeSeedDeterministicAndDistinct(t *testing.T) {
	a := orePosition{x: 10, y: -5, z: 20}
	b := orePosition{x: 11, y: -5, z: 20}

	if veinShapeSeed("minecraft:coal_ore", a) != veinShapeSeed("minecraft:coal_ore", a) {
		t.Fatalf("veinShapeSeed not deterministic for the same (block, origin)")
	}
	if veinShapeSeed("minecraft:coal_ore", a) == veinShapeSeed("minecraft:coal_ore", b) {
		t.Fatalf("different origins produced the same seed")
	}
	if veinShapeSeed("minecraft:coal_ore", a) == veinShapeSeed("minecraft:iron_ore", a) {
		t.Fatalf("different blocks at the same origin produced the same seed")
	}
}

// TestGenerateVeinBlobSizeZeroOrLess confirms a non-positive size is a
// clean no-op (nil, no panic) rather than something seedOreVein would need
// to guard against itself.
func TestGenerateVeinBlobSizeZeroOrLess(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 0))
	if got := generateVeinBlob(rng, orePosition{}, 0); got != nil {
		t.Fatalf("generateVeinBlob(size=0) = %v, want nil", got)
	}
}

// TestGenerateVeinBlobDeterministic confirms two independently-constructed
// RNGs seeded identically produce the exact same blob - the property
// seedOreVein relies on for restart-stability, the same role
// TestOreTemplatePositionsDeterministic plays for instance positions.
func TestGenerateVeinBlobDeterministic(t *testing.T) {
	origin := orePosition{x: 100, y: -10, z: -50}
	first := generateVeinBlob(rand.New(rand.NewPCG(42, 0)), origin, 9)
	second := generateVeinBlob(rand.New(rand.NewPCG(42, 0)), origin, 9)
	if len(first) != len(second) {
		t.Fatalf("got %d and %d surviving points, want equal", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("point %d differs between calls: %v vs %v", i, first[i], second[i])
		}
	}
}

// TestGenerateVeinBlobStaysNearOrigin is a basic sanity bound: a blob's
// surviving points shouldn't wander arbitrarily far from where it was
// rooted - OreFeature's own line length is size/8 plus each point's own
// radius (at most roughly size/16), so every coordinate should stay well
// within size blocks of origin on each axis.
func TestGenerateVeinBlobStaysNearOrigin(t *testing.T) {
	origin := orePosition{x: 0, y: 0, z: 0}
	size := 17
	points := generateVeinBlob(rand.New(rand.NewPCG(7, 0)), origin, size)
	if len(points) == 0 {
		t.Fatalf("expected at least one surviving point for size %d", size)
	}
	bound := float64(size)
	for _, p := range points {
		if math.Abs(p.x) > bound || math.Abs(p.y) > bound || math.Abs(p.z) > bound {
			t.Fatalf("point %v strayed further than %v blocks from origin", p, bound)
		}
		if p.radius <= 0 {
			t.Fatalf("surviving point %v has non-positive radius, should have been culled", p)
		}
	}
}

// TestVeinCellsDeterministic mirrors TestGenerateVeinBlobDeterministic one
// level up - the actual absolute cells seedOreVein would issue /setblock
// for must also be stable across independently-seeded calls.
func TestVeinCellsDeterministic(t *testing.T) {
	origin := orePosition{x: 100, y: -10, z: -50}
	first := veinCells(rand.New(rand.NewPCG(42, 0)), origin, 9)
	second := veinCells(rand.New(rand.NewPCG(42, 0)), origin, 9)
	if len(first) != len(second) {
		t.Fatalf("got %d and %d cells, want equal", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("cell %d differs between calls: %v vs %v", i, first[i], second[i])
		}
	}
}

// TestVeinCellsNoDuplicates confirms overlapping spheres from different
// surviving points never produce the same cell twice - seedOreVein would
// otherwise double-count it in its returned placed-block total.
func TestVeinCellsNoDuplicates(t *testing.T) {
	origin := orePosition{x: 0, y: 0, z: 0}
	cells := veinCells(rand.New(rand.NewPCG(7, 0)), origin, 17)
	seen := make(map[orePosition]bool)
	for _, c := range cells {
		if seen[c] {
			t.Fatalf("duplicate cell %v", c)
		}
		seen[c] = true
	}
}

func TestValidateBuriedOreSeedingRequiresCountWhenBuriedAndClone(t *testing.T) {
	pool := &curriculum.Pool{MineBuriedTargetBlocks: []string{"minecraft:coal_ore"}}
	f := flags{cloneAreaFrom: "0,-1,5000", seedTemplateOreCount: 0}
	err := validateBuriedOreSeeding(f, pool)
	require.Error(t, err)

	f.seedTemplateOreCount = 60
	require.NoError(t, validateBuriedOreSeeding(f, pool))

	require.NoError(t, validateBuriedOreSeeding(flags{}, pool), "no clone-area-from: nothing to validate")
	require.NoError(t, validateBuriedOreSeeding(f, &curriculum.Pool{}), "no buried blocks")
}

// TestTemplateSeedConfigWidensRadiusByCloneAreaJitter confirms
// (flags).templateSeedConfig widens its own seeding radius by
// -clone-area-jitter - see that flag's own doc comment for why: ore must
// be seeded across every window -clone-area-jitter could ever pick for
// the actual per-Reset clone, not just the one centered on
// -clone-area-from itself.
func TestTemplateSeedConfigWidensRadiusByCloneAreaJitter(t *testing.T) {
	pool := &curriculum.Pool{MineBuriedTargetBlocks: []string{"minecraft:coal_ore"}}
	f := flags{
		cloneAreaFrom:        "0,-1,5000",
		cloneAreaFromParsed:  [3]float64{0, -1, 5000},
		cloneAreaRadius:      32,
		seedTemplateOreCount: 8,
	}
	cfg := f.templateSeedConfig(pool)
	require.Equal(t, 32, cfg.cloneAreaRadius, "no jitter: seeding radius must match -clone-area-radius exactly")
	require.Equal(t, 8, cfg.oreCount, "no jitter: ore count must stay exactly -seed-template-ore-count (derived density must round-trip exactly)")

	f.cloneAreaJitter = 64
	cfg = f.templateSeedConfig(pool)
	require.Equal(t, 96, cfg.cloneAreaRadius, "with jitter: seeding radius must cover -clone-area-radius+-clone-area-jitter")
	require.Equal(t, 71, cfg.oreCount, "with jitter: ore count must scale up to hold the same density (~0.19%%) across the widened area")

	// episodeLimits' own clone radius (the actual per-Reset /clone box
	// size) must stay unaffected by jitter - only the seeding radius grows.
	require.Equal(t, 32, f.episodeLimits().cloneAreaRadius)
}

// TestTemplateSeedConfigDensityPctOverride confirms
// -seed-template-ore-density-pct, when explicitly set, replaces the
// density -seed-template-ore-count/-clone-area-radius would otherwise
// imply, rather than only ever being derived from them.
func TestTemplateSeedConfigDensityPctOverride(t *testing.T) {
	pool := &curriculum.Pool{MineBuriedTargetBlocks: []string{"minecraft:coal_ore"}}
	f := flags{
		cloneAreaFrom:             "0,-1,5000",
		cloneAreaFromParsed:       [3]float64{0, -1, 5000},
		cloneAreaRadius:           32,
		cloneAreaJitter:           64,
		seedTemplateOreCount:      8, // must be ignored once densityPct overrides it
		seedTemplateOreDensityPct: 1.0,
	}
	cfg := f.templateSeedConfig(pool)
	// seedRadius = 96, footprint = 193^2 = 37249, 1% of that = 372.49 -> 372
	require.Equal(t, 372, cfg.oreCount, "an explicit density-pct override must be used as-is, not derived from -seed-template-ore-count")
}
