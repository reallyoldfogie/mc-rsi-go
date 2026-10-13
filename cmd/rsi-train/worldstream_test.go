package main

import "testing"

func TestIsAirBlockName(t *testing.T) {
	air := []string{"air", "minecraft:air", "cave_air", "minecraft:cave_air", "void_air", "minecraft:void_air"}
	for _, name := range air {
		if !isAirBlockName(name) {
			t.Errorf("isAirBlockName(%q) = false, want true", name)
		}
	}

	notAir := []string{"", "stone", "minecraft:dirt", "air_portal", "void_airship"}
	for _, name := range notAir {
		if isAirBlockName(name) {
			t.Errorf("isAirBlockName(%q) = true, want false", name)
		}
	}
}
