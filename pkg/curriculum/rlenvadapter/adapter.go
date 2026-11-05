// Package rlenvadapter defines every task curriculum.Pool can pose and
// turns one of them into a real mc-agent rlenv.Config.TaskSelector — the
// "thin adapter" docs/plans/07-curriculum-generator.md's own "Package
// layout" section calls for, kept in its own package specifically so
// pkg/curriculum itself never needs to import mc-agent/rlenv (see that
// package's own doc comment). Anything that wants to actually drive a
// live rlenv.Environment against a curriculum.Pool imports this package;
// anything that only wants to load/validate a curriculum config imports
// pkg/curriculum alone and never pulls in mc-agent at all.
//
// Task kinds (goto, mine, craft, the "far" composites, each chain stage)
// are each their own small struct implementing Task, registered in
// allFactories — see Task's own doc comment for why: the previous design
// here (and in cmd/rsi-train, before that) classified an episode's task
// from which combination of rlenv.TaskOverride fields happened to be set,
// in a switch that a new task kind had to be added to correctly, in sync
// with every other switch doing the same classification elsewhere. A new
// task kind is now a new struct (+ a factory for it, appended to
// allFactories) — self-contained, nothing elsewhere to keep in sync.
package rlenvadapter

import (
	"fmt"
	oldrand "math/rand" // rlenv.TaskSelector's own rng parameter type — see TaskSelector's doc comment for why it's unused here.
	"math/rand/v2"
	"strings"

	"github.com/reallyoldfogie/mc-agent/rlenv"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// Task is one episode's chosen task: a concrete value drawn from a
// TaskFactory, carrying whatever parameters that draw settled on (a
// goto's target offset, a mine's block, ...). Name labels the task kind
// (metric labels, the dashboard's Task stat); GoalDescription states its
// actual success condition in plain language (the dashboard's Goal
// stat); ToOverride poses it to rlenv. Every task kind is its own small
// struct satisfying this interface — see allFactories.
type Task interface {
	Name() string
	GoalDescription() string
	ToOverride() rlenv.TaskOverride
}

// TaskFactory draws a random Task of its own kind from pool, or reports
// that pool has nothing to draw (Available) so Select leaves it out of
// that episode's draw. Kind is the stable name Select uses for
// curriculum.Pool.TaskWeights lookup and must match the Name() of every
// Task Draw returns (e.g. "goto", "chain_mine") — weights are keyed by
// that string, not by factory type.
type TaskFactory interface {
	Kind() string
	Available(pool curriculum.Pool) bool
	Draw(pool curriculum.Pool, rng *rand.Rand) Task
}

// allFactories is every task kind Select can draw, in a fixed order (so
// that, for a given rng stream, which kinds are available never itself
// depends on map iteration or similar non-determinism) — the one place a
// new task kind's factory needs to be registered to make it selectable.
// Every other property of a task kind (its name, its goal, its override)
// lives entirely on the kind's own Task/TaskFactory implementation below,
// not here.
var allFactories = []TaskFactory{
	gotoFactory{},
	mineFactory{},
	craftFactory{},
	mineFarFactory{},
	craftFarFactory{},
	mineBuriedFactory{},
	chainGatherFactory{},
	chainTableFactory{},
	chainPlaceFactory{},
	chainUseFactory{},
	chainMineFactory{},
	equipFactory{},
	pickupFactory{},
	pickupAnyFactory{},
	placeFactory{},
	flyFactory{},
	rideFactory{},
	combatFactory{},
	creeperGunpowderFactory{},
	spearChargeFactory{},
	maceSmashFactory{},
	followFactory{},
}

func shortName(id string) string { return strings.TrimPrefix(id, "minecraft:") }

func gotoGoalDescription(offset [3]float64) string {
	return fmt.Sprintf("goto (%+.0f, %+.0f, %+.0f) from spawn", offset[0], offset[1], offset[2])
}

// gotoTask poses "reach a point offset from wherever Reset finds the bot."
type gotoTask struct {
	offset         [3]float64
	arriveSneaking bool
}

func (t gotoTask) Name() string            { return "goto" }
func (t gotoTask) GoalDescription() string { return gotoGoalDescription(t.offset) }
func (t gotoTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: false,
		TargetOffset:       t.offset,
		ArriveSneaking:     t.arriveSneaking,
	}
}

type gotoFactory struct{}

func (gotoFactory) Kind() string { return "goto" }

func (gotoFactory) Available(pool curriculum.Pool) bool { return len(pool.TargetOffsets) > 0 }
func (gotoFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return gotoTask{
		offset:         pool.TargetOffsets[rng.IntN(len(pool.TargetOffsets))],
		arriveSneaking: pool.ArriveSneaking,
	}
}

// mineTask poses mining a block already within reach.
type mineTask struct {
	block        string
	searchRadius int
}

func (t mineTask) Name() string            { return "mine" }
func (t mineTask) GoalDescription() string { return "mine " + shortName(t.block) }
func (t mineTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, MineTargetBlock: t.block, MineSearchRadius: t.searchRadius}
}

type mineFactory struct{}

func (mineFactory) Kind() string { return "mine" }

func (mineFactory) Available(pool curriculum.Pool) bool { return len(pool.MineTargetBlocks) > 0 }
func (mineFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	block := pool.MineTargetBlocks[rng.IntN(len(pool.MineTargetBlocks))]
	return mineTask{block: block, searchRadius: pool.MineSearchRadius}
}

// mineBuriedTask poses mining a block buried in the shared template's
// underground stone band (docs/plans/14) — structurally identical to
// mineTask (no goto/far-seed component: the target is already within
// reach, just not yet visible), plus a tool seeded into inventory first
// since the target can't realistically be reached bare-handed.
type mineBuriedTask struct {
	block                string
	searchRadius         int
	requiredTool         string
	requiredToolEnchants map[string]int
	exploreDepth         int
}

func (t mineBuriedTask) Name() string            { return "mine_buried" }
func (t mineBuriedTask) GoalDescription() string { return "mine buried " + shortName(t.block) }
func (t mineBuriedTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:           true,
		MineTargetBlock:              t.block,
		MineSearchRadius:             t.searchRadius,
		MineRequiredTool:             t.requiredTool,
		MineRequiredToolEnchantments: t.requiredToolEnchants,
		MineBuriedExploreDepth:       t.exploreDepth,
		// The target already exists, baked into the shared clone
		// template — DefaultEpisodeSeeder must not also place a fresh,
		// immediately-visible instance beside the bot (caught live,
		// against a real server: it otherwise does exactly that, since
		// the real buried instance is never visible by design).
		MineTargetBuried: true,
		// Without this, mc-agent's mine/minedBuried used to credit
		// success the instant the target block broke, regardless of
		// whether the bot ever kept the drop — found live, 2026-11-04:
		// roughly half of mine_buried's agents never held the ore they'd
		// just "successfully" mined. chainTask.ToOverride below already
		// sets this (ChainMine needed the gathered log collected to craft
		// with); mine_buried never did.
		CollectDrops: true,
	}
}

type mineBuriedFactory struct{}

func (mineBuriedFactory) Kind() string { return "mine_buried" }

func (mineBuriedFactory) Available(pool curriculum.Pool) bool {
	return len(pool.MineBuriedTargetBlocks) > 0
}
func (mineBuriedFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	block := pool.MineBuriedTargetBlocks[rng.IntN(len(pool.MineBuriedTargetBlocks))]
	return mineBuriedTask{
		block:                block,
		searchRadius:         pool.MineBuriedSearchRadius,
		requiredTool:         pool.MineBuriedRequiredTool,
		requiredToolEnchants: pool.MineBuriedRequiredToolEnchantments,
		exploreDepth:         pool.MineBuriedExploreDepth,
	}
}

// craftTask poses crafting an item already craftable from held ingredients.
type craftTask struct{ item string }

func (t craftTask) Name() string            { return "craft" }
func (t craftTask) GoalDescription() string { return "craft " + shortName(t.item) }
func (t craftTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, CraftTargetItem: t.item}
}

type craftFactory struct{}

func (craftFactory) Kind() string { return "craft" }

func (craftFactory) Available(pool curriculum.Pool) bool { return len(pool.CraftTargetItems) > 0 }
func (craftFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return craftTask{item: pool.CraftTargetItems[rng.IntN(len(pool.CraftTargetItems))]}
}

// mineFarTask poses "go there, then mine": the block is seeded beside the
// goto target, out of sight until the bot travels there.
type mineFarTask struct {
	offset       [3]float64
	block        string
	searchRadius int
}

func (t mineFarTask) Name() string            { return "mine_far" }
func (t mineFarTask) GoalDescription() string { return "mine " + shortName(t.block) }
func (t mineFarTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: false,
		TargetOffset:       t.offset,
		MineTargetBlock:    t.block,
		MineSearchRadius:   t.searchRadius,
		SeedAtGoal:         true,
	}
}

type mineFarFactory struct{}

func (mineFarFactory) Kind() string { return "mine_far" }

func (mineFarFactory) Available(pool curriculum.Pool) bool {
	return len(pool.MineFarTargetOffsets) > 0 && len(pool.MineFarBlocks) > 0
}
func (mineFarFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.MineFarTargetOffsets[rng.IntN(len(pool.MineFarTargetOffsets))]
	block := pool.MineFarBlocks[rng.IntN(len(pool.MineFarBlocks))]
	return mineFarTask{offset: offset, block: block, searchRadius: pool.MineFarSearchRadius}
}

// craftFarTask poses "go there, then craft": a crafting table is seeded
// beside the goto target, out of the craft-ready search radius until the
// bot travels there.
type craftFarTask struct {
	offset       [3]float64
	item         string
	searchRadius int
}

func (t craftFarTask) Name() string            { return "craft_far" }
func (t craftFarTask) GoalDescription() string { return "craft " + shortName(t.item) }
func (t craftFarTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: false,
		TargetOffset:       t.offset,
		CraftTargetItem:    t.item,
		CraftSearchRadius:  t.searchRadius,
		SeedAtGoal:         true,
	}
}

type craftFarFactory struct{}

func (craftFarFactory) Kind() string { return "craft_far" }

func (craftFarFactory) Available(pool curriculum.Pool) bool {
	return len(pool.CraftFarTargetOffsets) > 0 && len(pool.CraftFarItems) > 0 && pool.CraftFarSearchRadius > 0
}
func (craftFarFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.CraftFarTargetOffsets[rng.IntN(len(pool.CraftFarTargetOffsets))]
	item := pool.CraftFarItems[rng.IntN(len(pool.CraftFarItems))]
	return craftFarTask{offset: offset, item: item, searchRadius: pool.CraftFarSearchRadius}
}

// chainStageNames names each stage of the multi-step crafting chain
// (docs/plans/12 and, for stage 5, docs/plans/14) — the one list that
// needs a new entry if rlenv ever adds another stage; everything else
// about a chain episode (its goal, its override) is shared across all
// five stages below, since they differ only in stage number and which
// craft item/mine target (if any) they carry.
var chainStageNames = map[int]string{1: "chain_gather", 2: "chain_table", 3: "chain_place", 4: "chain_use", 5: "chain_mine"}

// chainTask poses one stage of the multi-step crafting chain: a tree of
// curriculum.ChainTreeBlock is seeded beside the goto target, out of
// sight from the start, and the episode asks for stage 1 (gather a log),
// 2 (craft planks and a table), 3 (place the table), 4 (craft the goal
// item at it), or 5/ChainMine (then find and mine mineTarget, a block
// buried in the shared template's underground stone band —
// docs/plans/14-buried-block-mining-and-memory-map.md's
// "chain_buried_mining": unlike mine_buried, the pickaxe is crafted from
// scratch here, never given). All five stages are modeled as one struct,
// not five near-identical ones, since they share everything but
// stage/craftItem/mineTarget.
type chainTask struct {
	stage        int
	offset       [3]float64
	searchRadius int
	craftItem    string // "" for stage 1; curriculum.ChainTableItem for 2/3; the goal item for 4/5.
	mineTarget   string // only meaningful at stage 5 (ChainMine): the buried block to mine once craftItem is crafted.
	exploreDepth int    // MineBuriedExploreDepth for stage 5; ignored earlier.
}

func (t chainTask) Name() string {
	if name, ok := chainStageNames[t.stage]; ok {
		return name
	}
	return "chain_unknown"
}

// GoalDescription states the chain's actual, final success condition —
// crafting the goal item for every stage except ChainMine, where the
// chain continues past that craft and only mining the buried block
// afterward actually ends the episode.
func (t chainTask) GoalDescription() string {
	if t.stage == rlenv.ChainMine {
		return "mine buried " + shortName(t.mineTarget)
	}
	return "craft " + shortName(t.craftItem)
}
func (t chainTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:     false,
		TargetOffset:           t.offset,
		MineTargetBlock:        curriculum.ChainTreeBlock,
		MineSearchRadius:       t.searchRadius,
		CraftTargetItem:        t.craftItem,
		ChainStage:             t.stage,
		ChainMineTargetBlock:   t.mineTarget,
		MineBuriedExploreDepth: t.exploreDepth,
		CollectDrops:           true,
		SeedAtGoal:             true,
	}
}

// chainAvailable is shared by every chain stage's factory: all four need
// the same offsets/radius; only stage 4 (chainUseFactory) additionally
// needs a goal item.
func chainAvailable(pool curriculum.Pool) bool {
	return len(pool.ChainTargetOffsets) > 0 && pool.ChainSearchRadius > 0
}

type chainGatherFactory struct{}

func (chainGatherFactory) Kind() string { return "chain_gather" }

func (chainGatherFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainGatherFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 1, offset: offset, searchRadius: pool.ChainSearchRadius}
}

type chainTableFactory struct{}

func (chainTableFactory) Kind() string { return "chain_table" }

func (chainTableFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainTableFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 2, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: curriculum.ChainTableItem}
}

type chainPlaceFactory struct{}

func (chainPlaceFactory) Kind() string { return "chain_place" }

func (chainPlaceFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainPlaceFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 3, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: curriculum.ChainTableItem}
}

type chainUseFactory struct{}

func (chainUseFactory) Kind() string { return "chain_use" }

func (chainUseFactory) Available(pool curriculum.Pool) bool {
	return chainAvailable(pool) && len(pool.ChainGoalItems) > 0
}
func (chainUseFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	item := pool.ChainGoalItems[rng.IntN(len(pool.ChainGoalItems))]
	return chainTask{stage: 4, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: item}
}

// chainMineFactory draws "chain_buried_mining" (docs/plans/14): needs a
// goal item to craft (the same pool ChainUse draws from) *and* at least
// one buried block to mine afterward — missing either leaves the type
// unavailable, the same way craftFarFactory needs both its offsets and
// its items.
type chainMineFactory struct{}

func (chainMineFactory) Kind() string { return "chain_mine" }

func (chainMineFactory) Available(pool curriculum.Pool) bool {
	return chainAvailable(pool) && len(pool.ChainGoalItems) > 0 && len(pool.ChainMineTargetBlocks) > 0
}
func (chainMineFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	item := pool.ChainGoalItems[rng.IntN(len(pool.ChainGoalItems))]
	mineTarget := pool.ChainMineTargetBlocks[rng.IntN(len(pool.ChainMineTargetBlocks))]
	return chainTask{stage: rlenv.ChainMine, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: item, mineTarget: mineTarget, exploreDepth: pool.MineBuriedExploreDepth}
}

// equipTask poses equipping (and optionally using) a held/worn item.
type equipTask struct{ item string }

func (t equipTask) Name() string            { return "equip" }
func (t equipTask) GoalDescription() string { return "equip " + shortName(t.item) }
func (t equipTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, EquipTargetItem: t.item}
}

type equipFactory struct{}

func (equipFactory) Kind() string { return "equip" }

func (equipFactory) Available(pool curriculum.Pool) bool { return len(pool.EquipTargetItems) > 0 }
func (equipFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return equipTask{item: pool.EquipTargetItems[rng.IntN(len(pool.EquipTargetItems))]}
}

// pickupTask poses collecting a named ground item.
type pickupTask struct {
	item         string
	searchRadius int
}

func (t pickupTask) Name() string            { return "pickup" }
func (t pickupTask) GoalDescription() string { return "pickup " + shortName(t.item) }
func (t pickupTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: true,
		PickupTargetItem:   t.item,
		PickupSearchRadius: t.searchRadius,
	}
}

type pickupFactory struct{}

func (pickupFactory) Kind() string { return "pickup" }

func (pickupFactory) Available(pool curriculum.Pool) bool { return len(pool.PickupTargetItems) > 0 }
func (pickupFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return pickupTask{
		item:         pool.PickupTargetItems[rng.IntN(len(pool.PickupTargetItems))],
		searchRadius: pool.PickupSearchRadius,
	}
}

// pickupAnyTask poses collecting whatever item is nearest, regardless of content.
type pickupAnyTask struct{ searchRadius int }

func (t pickupAnyTask) Name() string            { return "pickup_any" }
func (t pickupAnyTask) GoalDescription() string { return "pickup any nearby item" }
func (t pickupAnyTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: true,
		PickupAnyItem:      true,
		PickupSearchRadius: t.searchRadius,
	}
}

type pickupAnyFactory struct{}

func (pickupAnyFactory) Kind() string { return "pickup_any" }

func (pickupAnyFactory) Available(pool curriculum.Pool) bool { return pool.PickupAnyItem }
func (pickupAnyFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return pickupAnyTask{searchRadius: pool.PickupSearchRadius}
}

// placeTask poses standalone block/item placement outside a chain episode.
type placeTask struct{ item string }

func (t placeTask) Name() string            { return "place" }
func (t placeTask) GoalDescription() string { return "place " + shortName(t.item) }
func (t placeTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, PlaceTargetItem: t.item}
}

type placeFactory struct{}

func (placeFactory) Kind() string { return "place" }

func (placeFactory) Available(pool curriculum.Pool) bool { return len(pool.PlaceTargetItems) > 0 }
func (placeFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return placeTask{item: pool.PlaceTargetItems[rng.IntN(len(pool.PlaceTargetItems))]}
}

// flyTask poses elytra flight to an offset target.
type flyTask struct {
	offset        [3]float64
	arrivalRadius float64
}

func (t flyTask) Name() string { return "fly" }
func (t flyTask) GoalDescription() string {
	return fmt.Sprintf("fly to (%+.0f, %+.0f, %+.0f) from spawn", t.offset[0], t.offset[1], t.offset[2])
}
func (t flyTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: true,
		FlyTargetOffset:    t.offset,
		FlyTargetActive:    true,
		FlyArrivalRadius:   t.arrivalRadius,
	}
}

type flyFactory struct{}

func (flyFactory) Kind() string { return "fly" }

func (flyFactory) Available(pool curriculum.Pool) bool { return len(pool.FlyTargetOffsets) > 0 }
func (flyFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return flyTask{
		offset:        pool.FlyTargetOffsets[rng.IntN(len(pool.FlyTargetOffsets))],
		arrivalRadius: pool.FlyArrivalRadius,
	}
}

// rideTask poses mounting/dismounting/jumping a vehicle.
type rideTask struct {
	vehicleType  string
	searchRadius int
}

func (t rideTask) Name() string            { return "ride" }
func (t rideTask) GoalDescription() string { return "ride " + shortName(t.vehicleType) }
func (t rideTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: true,
		RideVehicleType:    t.vehicleType,
		RideSearchRadius:   t.searchRadius,
	}
}

type rideFactory struct{}

func (rideFactory) Kind() string { return "ride" }

func (rideFactory) Available(pool curriculum.Pool) bool { return len(pool.RideVehicleTypes) > 0 }
func (rideFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return rideTask{
		vehicleType:  pool.RideVehicleTypes[rng.IntN(len(pool.RideVehicleTypes))],
		searchRadius: pool.RideSearchRadius,
	}
}

// combatTask poses fighting a mob type via ActionCombat.
type combatTask struct {
	mobType        string
	radius         float64
	includePlayers bool
	includeNeutral bool
}

func (t combatTask) Name() string            { return "combat" }
func (t combatTask) GoalDescription() string { return "combat " + shortName(t.mobType) }
func (t combatTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:   true,
		CombatTargetMobType:  t.mobType,
		CombatRadius:         t.radius,
		CombatIncludePlayers: t.includePlayers,
		CombatIncludeNeutral: t.includeNeutral,
	}
}

type combatFactory struct{}

func (combatFactory) Kind() string { return "combat" }

func (combatFactory) Available(pool curriculum.Pool) bool { return len(pool.CombatTargetMobTypes) > 0 }
func (combatFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return combatTask{
		mobType:        pool.CombatTargetMobTypes[rng.IntN(len(pool.CombatTargetMobTypes))],
		radius:         pool.CombatRadius,
		includePlayers: pool.CombatIncludePlayers,
		includeNeutral: pool.CombatIncludeNeutral,
	}
}

// creeperGunpowderTask poses the hardcoded creeper-for-gunpowder encounter.
type creeperGunpowderTask struct{ radius float64 }

func (t creeperGunpowderTask) Name() string            { return "creeper_gunpowder" }
func (t creeperGunpowderTask) GoalDescription() string { return "kill creeper for gunpowder" }
func (t creeperGunpowderTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:     true,
		CreeperGunpowderActive: true,
		CreeperGunpowderRadius: t.radius,
	}
}

type creeperGunpowderFactory struct{}

func (creeperGunpowderFactory) Kind() string { return "creeper_gunpowder" }

func (creeperGunpowderFactory) Available(pool curriculum.Pool) bool {
	return pool.CreeperGunpowderActive
}
func (creeperGunpowderFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return creeperGunpowderTask{radius: pool.CreeperGunpowderRadius}
}

// spearChargeTask poses a spear-charge approach against a mob type.
type spearChargeTask struct {
	mobType string
	radius  float64
}

func (t spearChargeTask) Name() string            { return "spear_charge" }
func (t spearChargeTask) GoalDescription() string { return "spear charge " + shortName(t.mobType) }
func (t spearChargeTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:       true,
		SpearChargeTargetMobType: t.mobType,
		SpearChargeRadius:        t.radius,
	}
}

type spearChargeFactory struct{}

func (spearChargeFactory) Kind() string { return "spear_charge" }

func (spearChargeFactory) Available(pool curriculum.Pool) bool {
	return len(pool.SpearChargeTargetMobTypes) > 0
}
func (spearChargeFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return spearChargeTask{
		mobType: pool.SpearChargeTargetMobTypes[rng.IntN(len(pool.SpearChargeTargetMobTypes))],
		radius:  pool.SpearChargeRadius,
	}
}

// maceSmashTask poses a mace-smash approach against a mob type.
type maceSmashTask struct {
	mobType string
	radius  float64
}

func (t maceSmashTask) Name() string            { return "mace_smash" }
func (t maceSmashTask) GoalDescription() string { return "mace smash " + shortName(t.mobType) }
func (t maceSmashTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:     true,
		MaceSmashTargetMobType: t.mobType,
		MaceSmashRadius:        t.radius,
	}
}

type maceSmashFactory struct{}

func (maceSmashFactory) Kind() string { return "mace_smash" }

func (maceSmashFactory) Available(pool curriculum.Pool) bool {
	return len(pool.MaceSmashTargetMobTypes) > 0
}
func (maceSmashFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return maceSmashTask{
		mobType: pool.MaceSmashTargetMobTypes[rng.IntN(len(pool.MaceSmashTargetMobTypes))],
		radius:  pool.MaceSmashRadius,
	}
}

// followTask poses following (and stopping following) a named player.
type followTask struct{ playerName string }

func (t followTask) Name() string            { return "follow" }
func (t followTask) GoalDescription() string { return "follow " + t.playerName }
func (t followTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:     true,
		FollowTargetPlayerName: t.playerName,
	}
}

type followFactory struct{}

func (followFactory) Kind() string { return "follow" }

func (followFactory) Available(pool curriculum.Pool) bool {
	return len(pool.FollowTargetPlayerNames) > 0
}
func (followFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return followTask{playerName: pool.FollowTargetPlayerNames[rng.IntN(len(pool.FollowTargetPlayerNames))]}
}

// errEmptyPool is ValidatePool's error when pool has nothing to draw.
var errEmptyPool = fmt.Errorf("rlenvadapter: pool has no candidate values for any task kind")

// errNoPositiveWeight is ValidatePool's error when every available kind
// is excluded by TaskWeights (weight <= 0) — the pool has candidates,
// but Select would still have nothing to draw.
var errNoPositiveWeight = fmt.Errorf("rlenvadapter: every available task kind has task_weights <= 0")

// defaultTaskWeight is the relative draw weight Select uses for a kind
// whose key is missing from curriculum.Pool.TaskWeights (or when that
// map is nil). 1 preserves the historical uniform draw among available
// kinds when no weights are configured.
const defaultTaskWeight = 1.0

// kindWeight returns pool.TaskWeights[kind], or defaultTaskWeight when
// the key is absent. Does not clamp negatives — ValidatePool rejects
// those; Select treats weight <= 0 as "skip this kind."
func kindWeight(pool curriculum.Pool, kind string) float64 {
	if pool.TaskWeights == nil {
		return defaultTaskWeight
	}
	weight, ok := pool.TaskWeights[kind]
	if !ok {
		return defaultTaskWeight
	}
	return weight
}

// availableFactories returns every factory whose parameter pool is
// non-empty and whose TaskWeights entry is > 0, in allFactories order.
// A kind with weight <= 0 is skipped even when Available — soft disable
// without clearing its parameter pool.
func availableFactories(pool curriculum.Pool) []TaskFactory {
	var available []TaskFactory
	for _, factory := range allFactories {
		if !factory.Available(pool) {
			continue
		}
		if kindWeight(pool, factory.Kind()) <= 0 {
			continue
		}
		available = append(available, factory)
	}
	return available
}

// ValidatePool reports an error if Select would have nothing to draw:
// either no task kind has candidates, every available kind is weight-
// excluded, or some TaskWeights entry is negative (config mistake —
// fail fast rather than silently treating it like zero). Call once at
// startup against an operator-supplied curriculum config.
func ValidatePool(pool curriculum.Pool) error {
	anyCandidates := false
	for _, factory := range allFactories {
		if !factory.Available(pool) {
			continue
		}
		anyCandidates = true
		weight := kindWeight(pool, factory.Kind())
		if weight < 0 {
			return fmt.Errorf("rlenvadapter: task_weights[%q] is negative (%v)", factory.Kind(), weight)
		}
	}
	if !anyCandidates {
		return errEmptyPool
	}
	if len(availableFactories(pool)) == 0 {
		return errNoPositiveWeight
	}
	return nil
}

// Select picks a task kind from whichever of allFactories pool actually
// has candidates for, weighted by curriculum.Pool.TaskWeights (missing
// keys default to 1 — uniform among available kinds when the map is
// empty/nil). Then asks that kind's own factory to draw a uniformly
// random parameter set from pool. No difficulty progression — see
// docs/plans/07's own "deliberate v2 deferral" note on why that's not
// attempted here; TaskWeights is the operator-side counter to pool
// dilution when many kinds are available at once. Panics if pool is
// empty — callers validate with ValidatePool first.
func Select(pool curriculum.Pool, rng *rand.Rand) Task {
	available := availableFactories(pool)
	if len(available) == 0 {
		panic("rlenvadapter: Select called with an empty pool; call ValidatePool first")
	}
	if len(available) == 1 {
		return available[0].Draw(pool, rng)
	}
	totalWeight := 0.0
	for _, factory := range available {
		totalWeight += kindWeight(pool, factory.Kind())
	}
	// Weighted pick via cumulative thresholds over [0, totalWeight).
	// Float64 edge noise can land exactly on totalWeight; the final
	// bucket absorbs that rather than panicking.
	pick := rng.Float64() * totalWeight
	cumulative := 0.0
	for _, factory := range available {
		cumulative += kindWeight(pool, factory.Kind())
		if pick < cumulative {
			return factory.Draw(pool, rng)
		}
	}
	return available[len(available)-1].Draw(pool, rng)
}

// TaskSelector wraps pool as an rlenv.TaskSelector for a caller that only
// needs to pose the task rlenv.Config.TaskSelector expects, not the Task
// itself (its Name/GoalDescription — see cmd/rsi-train's own
// TaskSelector closure, which calls Select directly instead, for exactly
// that). Each call draws a fresh Task via Select and converts it.
func TaskSelector(pool curriculum.Pool, rng *rand.Rand) rlenv.TaskSelector {
	return func(int, *oldrand.Rand) rlenv.TaskOverride {
		override := Select(pool, rng).ToOverride()
		// SleepSearchRadius / PortalSearchRadius / LightPortalSearchRadius
		// are deliberately injected here, after whatever task-specific
		// ToOverride() built, rather than added to each task type's own
		// override literal individually: unlike MineTargetBlock/
		// CraftTargetItem (each task's own thing to pose), these are
		// cross-cutting capabilities available DURING any other active
		// task (ActionSleep as a general-purpose defensive option;
		// ActionUsePortal/ActionLightPortal once a portal/frame is in
		// range). Zero (the default) leaves each capability off.
		override.SleepSearchRadius = pool.SleepSearchRadius
		override.PortalSearchRadius = pool.PortalSearchRadius
		override.LightPortalSearchRadius = pool.LightPortalSearchRadius
		return override
	}
}
