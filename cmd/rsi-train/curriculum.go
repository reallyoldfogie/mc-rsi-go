package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// curriculumPoolJSON is -curriculum-config's on-disk shape: a plain,
// rsi-train-owned JSON encoding of curriculum.Pool (pkg/curriculum
// itself deliberately has no JSON dependency — see its own doc comment
// on staying free of anything beyond plain data/pure functions), mirroring
// how -mc-agent-config/-trainer-config are each some other package's own
// format loaded independently by this command.
type curriculumPoolJSON struct {
	TargetOffsets    [][3]float64 `json:"target_offsets"`
	MineTargetBlocks []string     `json:"mine_target_blocks"`
	MineSearchRadius int          `json:"mine_search_radius"`
	CraftTargetItems []string     `json:"craft_target_items"`
}

// loadCurriculumPool reads path as curriculumPoolJSON and converts it
// into a curriculum.Pool.
func loadCurriculumPool(path string) (curriculum.Pool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return curriculum.Pool{}, fmt.Errorf("reading curriculum config %s: %w", path, err)
	}
	var cfg curriculumPoolJSON
	if err := json.Unmarshal(data, &cfg); err != nil {
		return curriculum.Pool{}, fmt.Errorf("parsing curriculum config %s: %w", path, err)
	}
	return curriculum.Pool{
		TargetOffsets:    cfg.TargetOffsets,
		MineTargetBlocks: cfg.MineTargetBlocks,
		MineSearchRadius: cfg.MineSearchRadius,
		CraftTargetItems: cfg.CraftTargetItems,
	}, nil
}
