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

	MineFarTargetOffsets  [][3]float64 `json:"mine_far_target_offsets"`
	MineFarBlocks         []string     `json:"mine_far_blocks"`
	MineFarSearchRadius   int          `json:"mine_far_search_radius"`
	CraftFarTargetOffsets [][3]float64 `json:"craft_far_target_offsets"`
	CraftFarItems         []string     `json:"craft_far_items"`
	CraftFarSearchRadius  int          `json:"craft_far_search_radius"`

	ChainTargetOffsets [][3]float64 `json:"chain_target_offsets"`
	ChainGoalItems     []string     `json:"chain_goal_items"`
	ChainSearchRadius  int          `json:"chain_search_radius"`

	// MineBuried*/ChainMineTargetBlocks pose docs/plans/14's two buried-
	// block task kinds ("mine_buried" and "chain_buried_mining").
	MineBuriedTargetBlocks             []string       `json:"mine_buried_target_blocks"`
	MineBuriedSearchRadius             int            `json:"mine_buried_search_radius"`
	MineBuriedRequiredTool             string         `json:"mine_buried_required_tool"`
	MineBuriedRequiredToolEnchantments map[string]int `json:"mine_buried_required_tool_enchantments"`
	MineBuriedExploreDepth             int            `json:"mine_buried_explore_depth"`
	ChainMineTargetBlocks              []string       `json:"chain_mine_target_blocks"`

	// SleepSearchRadius (docs/plans/SLEEP_ACTION_PLAN.md, local/untracked)
	// - see curriculum.Pool.SleepSearchRadius's own doc comment. Left
	// unset (0) in configs/curriculum.json's active pool.
	SleepSearchRadius int `json:"sleep_search_radius"`

	// PortalSearchRadius/LightPortalSearchRadius — see curriculum.Pool's
	// own doc comments. Cross-cutting like sleep.
	PortalSearchRadius      int `json:"portal_search_radius"`
	LightPortalSearchRadius int `json:"light_portal_search_radius"`

	ArriveSneaking bool `json:"arrive_sneaking"`

	EquipTargetItems []string `json:"equip_target_items"`

	PickupTargetItems  []string `json:"pickup_target_items"`
	PickupSearchRadius int      `json:"pickup_search_radius"`
	PickupAnyItem      bool     `json:"pickup_any_item"`

	PlaceTargetItems []string `json:"place_target_items"`

	FlyTargetOffsets [][3]float64 `json:"fly_target_offsets"`
	FlyArrivalRadius float64      `json:"fly_arrival_radius"`

	RideVehicleTypes []string `json:"ride_vehicle_types"`
	RideSearchRadius int      `json:"ride_search_radius"`

	CombatTargetMobTypes []string `json:"combat_target_mob_types"`
	CombatRadius         float64  `json:"combat_radius"`
	CombatIncludePlayers bool     `json:"combat_include_players"`
	CombatIncludeNeutral bool     `json:"combat_include_neutral"`

	CreeperGunpowderActive bool    `json:"creeper_gunpowder_active"`
	CreeperGunpowderRadius float64 `json:"creeper_gunpowder_radius"`

	SpearChargeTargetMobTypes []string `json:"spear_charge_target_mob_types"`
	SpearChargeRadius         float64  `json:"spear_charge_radius"`

	MaceSmashTargetMobTypes []string `json:"mace_smash_target_mob_types"`
	MaceSmashRadius         float64  `json:"mace_smash_radius"`

	// FollowTargetPlayerNames requires a second real connected player/bot
	// with the sampled name already in the session — see curriculum.Pool.
	FollowTargetPlayerNames []string `json:"follow_target_player_names"`

	// TaskWeights reweights Select among available kinds — see
	// curriculum.Pool.TaskWeights. Keys are task kind names ("goto",
	// "combat", "chain_mine", …). Missing keys default to 1.
	TaskWeights map[string]float64 `json:"task_weights"`
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

		MineFarTargetOffsets:  cfg.MineFarTargetOffsets,
		MineFarBlocks:         cfg.MineFarBlocks,
		MineFarSearchRadius:   cfg.MineFarSearchRadius,
		CraftFarTargetOffsets: cfg.CraftFarTargetOffsets,
		CraftFarItems:         cfg.CraftFarItems,
		CraftFarSearchRadius:  cfg.CraftFarSearchRadius,

		ChainTargetOffsets: cfg.ChainTargetOffsets,
		ChainGoalItems:     cfg.ChainGoalItems,
		ChainSearchRadius:  cfg.ChainSearchRadius,

		MineBuriedTargetBlocks:             cfg.MineBuriedTargetBlocks,
		MineBuriedSearchRadius:             cfg.MineBuriedSearchRadius,
		MineBuriedRequiredTool:             cfg.MineBuriedRequiredTool,
		MineBuriedRequiredToolEnchantments: cfg.MineBuriedRequiredToolEnchantments,
		MineBuriedExploreDepth:             cfg.MineBuriedExploreDepth,
		ChainMineTargetBlocks:              cfg.ChainMineTargetBlocks,

		SleepSearchRadius:       cfg.SleepSearchRadius,
		PortalSearchRadius:      cfg.PortalSearchRadius,
		LightPortalSearchRadius: cfg.LightPortalSearchRadius,

		ArriveSneaking: cfg.ArriveSneaking,

		EquipTargetItems: cfg.EquipTargetItems,

		PickupTargetItems:  cfg.PickupTargetItems,
		PickupSearchRadius: cfg.PickupSearchRadius,
		PickupAnyItem:      cfg.PickupAnyItem,

		PlaceTargetItems: cfg.PlaceTargetItems,

		FlyTargetOffsets: cfg.FlyTargetOffsets,
		FlyArrivalRadius: cfg.FlyArrivalRadius,

		RideVehicleTypes: cfg.RideVehicleTypes,
		RideSearchRadius: cfg.RideSearchRadius,

		CombatTargetMobTypes: cfg.CombatTargetMobTypes,
		CombatRadius:         cfg.CombatRadius,
		CombatIncludePlayers: cfg.CombatIncludePlayers,
		CombatIncludeNeutral: cfg.CombatIncludeNeutral,

		CreeperGunpowderActive: cfg.CreeperGunpowderActive,
		CreeperGunpowderRadius: cfg.CreeperGunpowderRadius,

		SpearChargeTargetMobTypes: cfg.SpearChargeTargetMobTypes,
		SpearChargeRadius:         cfg.SpearChargeRadius,

		MaceSmashTargetMobTypes: cfg.MaceSmashTargetMobTypes,
		MaceSmashRadius:         cfg.MaceSmashRadius,

		FollowTargetPlayerNames: cfg.FollowTargetPlayerNames,

		TaskWeights: cfg.TaskWeights,
	}, nil
}
