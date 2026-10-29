package main

import (
	"testing"

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
