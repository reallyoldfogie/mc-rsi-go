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
}
