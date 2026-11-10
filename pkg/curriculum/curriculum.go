// Package curriculum implements docs/plans/07-curriculum-generator.md:
// docs/plans/01-curriculum-generator.md's Option B (per-episode task
// switching + goal-conditioning), the anti-capability-lock mechanism
// ../cRL-go/docs/thoughts/rsi-with-rl.md section 4 describes — vary which
// task an episode poses (goto/mine/craft) rather than train a policy
// against just one, so it doesn't lock onto a single skill.
//
// This package holds only Pool, the plain, JSON-loadable config schema
// describing which tasks are available and what parameters each can
// draw from — deliberately free of any mc-agent/rlenv import, so loading
// and validating a curriculum config needs no live bot session and never
// pulls in mc-agent's own, much heavier dependency graph (Docker, etc. —
// see docs/architecture.md's "Why not put this in cRL-go or mc-agent"
// reasoning, the same concern applied one level down here). Everything
// that actually defines a task — what it's called, what its goal is, how
// it becomes an rlenv.TaskOverride — lives in the separate rlenvadapter
// subpackage instead, as a Task implementation per task kind (see that
// package's own doc comment): the one place allowed to know about rlenv,
// so a new task kind is a new, self-contained struct there, not a case
// added to a switch in this package.
package curriculum

// ChainTreeBlock is the block a chain episode's tree is made of.
const ChainTreeBlock = "minecraft:oak_log"

// ChainTableItem is the item a chain's table stages craft.
const ChainTableItem = "minecraft:crafting_table"

// Pool holds the candidate parameter values rlenvadapter's task selection
// samples from for each task kind. A task kind with an empty pool is
// never selected (there is nothing to sample) — see rlenvadapter.
// ValidatePool.
type Pool struct {
	// TargetOffsets is the set of candidate rlenv.Config.TargetOffset
	// values a goto episode samples uniformly from.
	TargetOffsets [][3]float64

	// MineTargetBlocks is the set of candidate block names a mine
	// episode samples uniformly from (e.g. "minecraft:stone").
	MineTargetBlocks []string
	// MineSearchRadius is shared across every mine episode, not itself
	// sampled — mirrors rlenv.Config.MineSearchRadius's own single-value
	// shape; 0 defers to rlenv's own built-in default (see
	// rlenv.Config.MineSearchRadius's doc comment).
	MineSearchRadius int

	// CraftTargetItems is the set of candidate item names a craft
	// episode samples uniformly from (e.g. "minecraft:stick").
	CraftTargetItems []string

	// MineBuriedTargetBlocks/MineBuriedSearchRadius/MineBuriedRequiredTool
	// pose "mine_buried" (docs/plans/14): like the plain mine task above —
	// no goto/far-seed component, the target is already within reach — but
	// targeting ore baked into the shared template's underground stone
	// band instead of the on-the-surface variety MineTargetBlocks draws
	// from, with a tool (e.g. "minecraft:wooden_pickaxe") seeded into
	// inventory first, since mc-agent has no tool-tier gating of its own
	// but the live server does, and digging through stone bare-handed is
	// both unrealistic and slow. MineBuriedRequiredTool is shared across
	// every mine_buried episode, not itself sampled, the same way
	// MineSearchRadius is for the plain mine task. Empty blocks disable
	// the type.
	MineBuriedTargetBlocks []string
	MineBuriedSearchRadius int
	MineBuriedRequiredTool string

	// MineBuriedRequiredToolEnchantments, if non-empty, enchants
	// MineBuriedRequiredTool (full enchantment ID -> level, e.g.
	// {"minecraft:efficiency": 5} — see rlenv.Config.
	// MineRequiredToolEnchantments) instead of giving it plain. Found
	// live: once mine_buried's exploration signal actually leads a bot to
	// the ore, break time (not movement) becomes the next-cheapest thing
	// to fix, and there's no reason to leave a seeded (not crafted) tool
	// at wooden-pickaxe speed. Only meaningful alongside
	// MineBuriedRequiredTool; never applies to chain_mine, whose
	// pickaxe is crafted by the agent, not seeded.
	MineBuriedRequiredToolEnchantments map[string]int

	// MineBuriedExploreDepth is how many blocks below the episode's
	// mine-search anchor buried-search waypoints aim (rlenv.Config.
	// MineBuriedExploreDepth). Shared across mine_buried and chain_mine.
	// 0 leaves rlenv's legacy MineSearchRadius/2 behavior. Should match
	// the run's -seed-template-ore-min/max-depth band (typically the
	// midpoint or max), not a hardcoded distribution in the agent.
	MineBuriedExploreDepth int

	// MineBuriedExploreRadius is how far from the episode's mine-search
	// anchor buried-search waypoints' quadrant corners sit horizontally
	// (rlenv.Config.MineBuriedExploreRadius) — the horizontal counterpart
	// to MineBuriedExploreDepth above, decoupling the blind-exploration
	// patrol's footprint from MineBuriedSearchRadius (which also bounds
	// FindVisibleBlock's literal detection distance). 0 leaves rlenv's
	// legacy MineSearchRadius/2 behavior — found live, 2026-11-09: at this
	// run's own MineBuriedSearchRadius 12, that legacy behavior gave a
	// 12x12 total patrol box, far smaller than the area ore is actually
	// seeded across (65x65 at -clone-area-radius 32), so the bot's blind
	// search routinely never covered any of the seeded ore. Should be set
	// close to twice the run's own -clone-area-radius so the patrol
	// actually spans the working area, independent of
	// MineBuriedSearchRadius's own (smaller, intentional) detection range.
	MineBuriedExploreRadius int

	// MineFarTargetOffsets/MineFarBlocks/MineFarSearchRadius pose "go there,
	// then mine": a mine block is placed beside a goto target sampled from
	// MineFarTargetOffsets. The search radius must be smaller than the
	// offsets' length so the block starts out of sight, and the blocks must
	// not occur in the terrain itself (in a flat world dirt and grass are
	// the floor and would always be "visible"). Empty offsets or blocks
	// disable the type.
	MineFarTargetOffsets [][3]float64
	MineFarBlocks        []string
	MineFarSearchRadius  int

	// CraftFarTargetOffsets/CraftFarItems/CraftFarSearchRadius pose "go
	// there, then craft": a crafting table is placed beside the goto target.
	// Crafting only becomes ready once a table is within the search radius,
	// which must be smaller than the offsets' length so the table starts out
	// of range (the agent's own table search is 32 blocks and it walks there
	// itself, so without the radius gate a nearby table would need no
	// travelling). The items must be recipes that need a table (chest,
	// bowl). Empty offsets or items, or a radius <= 0, disable the type.
	CraftFarTargetOffsets [][3]float64
	CraftFarItems         []string
	CraftFarSearchRadius  int

	// ChainTargetOffsets/ChainSearchRadius pose the multi-step crafting chain
	// (docs/plans/12): a tree of ChainTreeBlock is seeded beside a goto
	// target sampled from ChainTargetOffsets, out of sight from the start, and
	// the episode asks for stage 1-4 of the chain. Stages 1-3 (gather, table,
	// place) need only the offsets and a radius > 0; stage 4 (craft a goal
	// item at the placed table) additionally needs ChainGoalItems, each a
	// rlenv chain goal (bowl, chest, wooden_pickaxe). The search radius must
	// be smaller than the offsets' length so the tree starts out of sight.
	ChainTargetOffsets [][3]float64
	ChainGoalItems     []string
	ChainSearchRadius  int

	// ChainMineTargetBlocks poses stage 5 (ChainMine, "chain_buried_mining,"
	// docs/plans/14): same gather/table/place/craft sequence as stage 4
	// (ChainGoalItems still picks the tool to craft — e.g.
	// "minecraft:wooden_pickaxe"), but once crafted the chain switches to
	// finding and mining one of these block names, baked into the shared
	// working-area template's underground stone band rather than seeded
	// per episode (unlike the tree, which still is). Shares
	// ChainTargetOffsets/ChainSearchRadius with every other chain stage.
	// Empty disables the type, same as every other *TargetBlocks field.
	ChainMineTargetBlocks []string

	// SleepSearchRadius, if > 0, turns on ActionSleep
	// (docs/plans/SLEEP_ACTION_PLAN.md, local/untracked) for every
	// episode this Pool poses, regardless of which task type Select
	// picks that episode - unlike every field above, this is not itself
	// a task type Select chooses among; it's a cross-cutting capability
	// TaskSelector injects onto whatever override Select's own task
	// produced (see rlenvadapter.TaskSelector). Mirrors
	// rlenv.Config.SleepSearchRadius's own shape directly. 0 (the
	// default, and what every entry in configs/curriculum.json's own
	// active pool leaves this at today) means no episode this Pool
	// poses has the sleep capability turned on - built and available,
	// deliberately not yet part of what the live trainer actually
	// draws episodes from.
	SleepSearchRadius int

	// PortalSearchRadius/LightPortalSearchRadius, if > 0, turn on
	// ActionUsePortal / ActionLightPortal the same cross-cutting way
	// SleepSearchRadius turns on ActionSleep (rlenv.Config.
	// PortalSearchRadius / LightPortalSearchRadius). Unlike sleep,
	// DefaultEpisodeSeeder does not place a portal/frame today — the
	// working-area template (or some other world setup) must already
	// provide one within the radius, or the actions stay masked off as
	// "nothing visible." 0 leaves both capabilities off.
	PortalSearchRadius      int
	LightPortalSearchRadius int

	// ArriveSneaking, if true, makes every goto episode this Pool poses
	// dispatch movetoandsneakquiet instead of movetoquiet (rlenv.Config.
	// ArriveSneaking). Purely a dispatch-time choice on ActionGoToTarget;
	// false (the default) preserves standing arrival.
	ArriveSneaking bool

	// EquipTargetItems is the set of candidate item names an equip
	// episode samples from (e.g. "minecraft:elytra"). Enables ActionEquip
	// and ActionUseItem for that episode (rlenv.Config.EquipTargetItem).
	// Empty disables the type.
	EquipTargetItems []string

	// PickupTargetItems/PickupSearchRadius pose a named item-pickup
	// task (ActionPickUpItem). Empty items disable the named type.
	// PickupAnyItem, if true, additionally enables a separate "collect
	// whatever's nearest" pickup kind (rlenv.Config.PickupAnyItem) that
	// does not need a name pool.
	PickupTargetItems  []string
	PickupSearchRadius int
	PickupAnyItem      bool

	// PlaceTargetItems pose standalone ActionPlace outside a chain
	// episode (rlenv.Config.PlaceTargetItem). Chain stage 3 still uses
	// its own hardcoded crafting-table placement; this pool is the
	// non-chain path. Empty disables the type.
	PlaceTargetItems []string

	// FlyTargetOffsets/FlyArrivalRadius pose ActionFlyTo (rlenv.Config.
	// FlyTargetOffset + FlyTargetActive). Empty offsets disable the type.
	// FlyArrivalRadius <= 0 defers to rlenv's own DefaultConfig value.
	FlyTargetOffsets [][3]float64
	FlyArrivalRadius float64

	// RideVehicleTypes/RideSearchRadius pose ActionMount/ActionDismount/
	// ActionVehicleJump (rlenv.Config.RideVehicleType). Empty types
	// disable the type. RideSearchRadius <= 0 defers to rlenv's default.
	RideVehicleTypes []string
	RideSearchRadius int

	// CombatTargetMobTypes/CombatRadius/CombatIncludePlayers/
	// CombatIncludeNeutral pose ActionCombat (rlenv.Config.
	// CombatTargetMobType). Empty types disable the type. Radius <= 0
	// defers to rlenv's default. Include flags are shared across every
	// combat episode this Pool poses, not themselves sampled.
	CombatTargetMobTypes []string
	CombatRadius         float64
	CombatIncludePlayers bool
	CombatIncludeNeutral bool

	// CreeperGunpowderActive/CreeperGunpowderRadius pose
	// ActionKillCreeperForGunpowder (rlenv.Config.CreeperGunpowderActive).
	// False (the default) disables the type. Radius <= 0 defers to
	// rlenv's default.
	CreeperGunpowderActive bool
	CreeperGunpowderRadius float64

	// SpearChargeTargetMobTypes/SpearChargeRadius pose
	// ActionSpearChargeWithApproach. Empty types disable the type.
	SpearChargeTargetMobTypes []string
	SpearChargeRadius         float64

	// MaceSmashTargetMobTypes/MaceSmashRadius pose
	// ActionMaceSmashWithApproach. Empty types disable the type.
	MaceSmashTargetMobTypes []string
	MaceSmashRadius         float64

	// FollowTargetPlayerNames pose ActionFollow/ActionStopFollow
	// (rlenv.Config.FollowTargetPlayerName). Empty disables the type.
	// Real precondition not solvable by this package: a second connected
	// player (or bot) with the sampled name must already be in the
	// session — see rlenv.Config.FollowTargetPlayerName's own doc comment.
	FollowTargetPlayerNames []string

	// TaskWeights optionally reweights how often Select draws each task
	// kind (keys are the kind names Task.Name returns: "goto", "mine",
	// "combat", "chain_gather", …). A missing key defaults to 1 — so a
	// nil/empty map preserves the historical uniform draw among available
	// kinds. Weight <= 0 excludes a kind from the draw even when its
	// parameter pool is non-empty (soft disable without clearing the
	// pool). Negative weights are rejected by rlenvadapter.ValidatePool.
	// Relative magnitudes matter, not absolute ones: {"combat": 4,
	// "goto": 1} makes combat four times as likely as goto among those
	// two when both are available.
	TaskWeights map[string]float64
}
