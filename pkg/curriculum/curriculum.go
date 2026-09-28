// Package curriculum implements docs/plans/07-curriculum-generator.md:
// docs/plans/01-curriculum-generator.md's Option B (per-episode task
// switching + goal-conditioning), the anti-capability-lock mechanism
// ../cRL-go/docs/thoughts/rsi-with-rl.md section 4 describes — vary which
// task an episode poses (goto/mine/craft) rather than train a policy
// against just one, so it doesn't lock onto a single skill.
//
// This package is deliberately kept free of any mc-agent/rlenv import:
// Generator/TaskSpec/UniformRandom are all plain data and pure functions,
// independently unit-testable with no live bot session — see
// docs/plans/07's own "Package layout" note on why. The piece that
// actually needs rlenv (converting a Generator into an
// rlenv.Config.TaskSelector) lives in the separate rlenvadapter
// subpackage instead, so importing this package alone never pulls in
// mc-agent's own, much heavier dependency graph (Docker, etc. — see
// docs/architecture.md's "Why not put this in cRL-go or mc-agent"
// reasoning, the same concern applied one level down here).
package curriculum

import (
	"errors"
	"math/rand/v2"
)

// TaskType is which of rlenv's three tasks a TaskSpec poses.
type TaskType int

const (
	TaskGoto TaskType = iota
	TaskMine
	TaskCraft
	// TaskMineFar and TaskCraftFar are composite "go there, then mine/craft"
	// tasks: the block or crafting table is placed beside a goto target far
	// enough away that it is out of sight (and, for crafting, out of the
	// agent's table search radius) from the start, so the policy has to
	// GoToTarget first and only then Mine or Craft.
	TaskMineFar
	TaskCraftFar
	// TaskChainGather..TaskChainUse are the stages of the multi-step crafting
	// chain (docs/plans/12, rlenv.Config.ChainStage): a tree out of sight,
	// then gather a log (stage 1), craft planks and a crafting table (2),
	// place the table (3), or craft a goal item at it (4). Each is its own
	// type so the mix can weight them and the easier stages keep
	// regularizing the policy while the harder ones are learned.
	TaskChainGather
	TaskChainTable
	TaskChainPlace
	TaskChainUse
)

// ChainTreeBlock is the block a chain episode's tree is made of.
const ChainTreeBlock = "minecraft:oak_log"

// ChainTableItem is the item a chain's table stages craft.
const ChainTableItem = "minecraft:crafting_table"

// ChainStage returns which rlenv chain stage t is (1-4), or 0 for a task
// that is not part of the chain.
func (t TaskType) ChainStage() int {
	switch t {
	case TaskChainGather:
		return 1
	case TaskChainTable:
		return 2
	case TaskChainPlace:
		return 3
	case TaskChainUse:
		return 4
	default:
		return 0
	}
}

func (t TaskType) String() string {
	switch t {
	case TaskGoto:
		return "goto"
	case TaskMine:
		return "mine"
	case TaskCraft:
		return "craft"
	case TaskMineFar:
		return "mine_far"
	case TaskCraftFar:
		return "craft_far"
	case TaskChainGather:
		return "chain_gather"
	case TaskChainTable:
		return "chain_table"
	case TaskChainPlace:
		return "chain_place"
	case TaskChainUse:
		return "chain_use"
	default:
		return "unknown"
	}
}

// TaskSpec is this repo's own representation of one episode's chosen
// task — the same information rlenv.TaskOverride needs (see
// rlenvadapter), but as a plain, concretely-typed struct rather than
// docs/plans/07's original map[string]any sketch: every field
// rlenv.TaskOverride can vary is already known and fixed, so a typed
// struct is strictly safer (no runtime type assertions, no risk of a
// missing/mistyped key) with no loss of expressiveness. Only the fields
// relevant to Type are meaningful; the rest are simply left at their
// zero value — mirrors rlenv.TaskOverride's own "irrelevant fields at
// zero value" convention exactly, since a TaskSpec's whole job is to
// become one.
type TaskSpec struct {
	Type TaskType

	// TargetOffset is meaningful when Type is TaskGoto, TaskMineFar or
	// TaskCraftFar.
	TargetOffset [3]float64

	// MineTargetBlock/MineSearchRadius are meaningful only when
	// Type is TaskMine or TaskMineFar.
	MineTargetBlock  string
	MineSearchRadius int

	// CraftTargetItem is meaningful only when Type is TaskCraft or
	// TaskCraftFar.
	CraftTargetItem string

	// CraftSearchRadius is meaningful only when Type is TaskCraftFar: how
	// close a crafting table must be for the craft to be ready.
	CraftSearchRadius int
}

// Generator produces one TaskSpec per episode. episode is 0-indexed,
// matching rlenv.TaskSelector's own episode argument (rlenvadapter
// passes it straight through) — a Generator that wants a fixed schedule
// rather than pure randomness can key off it directly. rng is supplied
// by the caller (not held internally by a Generator) so a whole curriculum
// + training run's randomness stays traceable to one seed, the same
// determinism discipline rlenv.Config.Jitter/JitterSeed and this repo's
// own leapfrog.Round already follow.
type Generator interface {
	Next(episode int, rng *rand.Rand) TaskSpec
}

// Pool holds the candidate parameter values UniformRandom samples from
// for each task type. A task type with an empty pool is never selected
// (there is nothing to sample) — see NewUniformRandom.
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

// availableTypes returns which of TaskGoto/TaskMine/TaskCraft have at
// least one candidate value in p, in a fixed, deterministic order
// (Goto, Mine, Craft) so that, for a given rng stream, which types are
// available never itself depends on map iteration order or similar
// non-determinism.
func (p Pool) availableTypes() []TaskType {
	var types []TaskType
	if len(p.TargetOffsets) > 0 {
		types = append(types, TaskGoto)
	}
	if len(p.MineTargetBlocks) > 0 {
		types = append(types, TaskMine)
	}
	if len(p.CraftTargetItems) > 0 {
		types = append(types, TaskCraft)
	}
	// Appended after the original three, so a Pool without far tasks
	// selects exactly as it always did for any given rng stream.
	if len(p.MineFarTargetOffsets) > 0 && len(p.MineFarBlocks) > 0 {
		types = append(types, TaskMineFar)
	}
	if len(p.CraftFarTargetOffsets) > 0 && len(p.CraftFarItems) > 0 && p.CraftFarSearchRadius > 0 {
		types = append(types, TaskCraftFar)
	}
	// The chain types come last too, for the same reason.
	if len(p.ChainTargetOffsets) > 0 && p.ChainSearchRadius > 0 {
		types = append(types, TaskChainGather, TaskChainTable, TaskChainPlace)
		if len(p.ChainGoalItems) > 0 {
			types = append(types, TaskChainUse)
		}
	}
	return types
}

// UniformRandom is a Generator implementing docs/plans/07's "Selection
// strategy": each episode, pick a task type uniformly at random from
// whichever types Pool actually has candidates for (equal probability
// among Goto/Mine/Craft when all three are populated — task-type mixing
// from the start, per that document's own resolution of
// docs/plans/01-curriculum-generator.md's open question), then pick a
// uniformly random candidate value of that type's own parameters. No
// difficulty progression — see docs/plans/07's own "deliberate v2
// deferral" note on why that's not attempted here.
type UniformRandom struct {
	pool Pool
}

// NewUniformRandom validates pool has at least one task type with at
// least one candidate value (otherwise there is nothing Next could ever
// return) and wraps it in a Generator.
func NewUniformRandom(pool Pool) (*UniformRandom, error) {
	if len(pool.availableTypes()) == 0 {
		return nil, errEmptyPool
	}
	return &UniformRandom{pool: pool}, nil
}

// Next implements Generator.
func (g *UniformRandom) Next(_ int, rng *rand.Rand) TaskSpec {
	types := g.pool.availableTypes()
	typ := types[rng.IntN(len(types))]
	switch typ {
	case TaskGoto:
		return TaskSpec{Type: TaskGoto, TargetOffset: g.pool.TargetOffsets[rng.IntN(len(g.pool.TargetOffsets))]}
	case TaskMine:
		return TaskSpec{
			Type:             TaskMine,
			MineTargetBlock:  g.pool.MineTargetBlocks[rng.IntN(len(g.pool.MineTargetBlocks))],
			MineSearchRadius: g.pool.MineSearchRadius,
		}
	case TaskCraft:
		return TaskSpec{Type: TaskCraft, CraftTargetItem: g.pool.CraftTargetItems[rng.IntN(len(g.pool.CraftTargetItems))]}
	case TaskMineFar:
		return TaskSpec{
			Type:             TaskMineFar,
			TargetOffset:     g.pool.MineFarTargetOffsets[rng.IntN(len(g.pool.MineFarTargetOffsets))],
			MineTargetBlock:  g.pool.MineFarBlocks[rng.IntN(len(g.pool.MineFarBlocks))],
			MineSearchRadius: g.pool.MineFarSearchRadius,
		}
	case TaskCraftFar:
		return TaskSpec{
			Type:              TaskCraftFar,
			TargetOffset:      g.pool.CraftFarTargetOffsets[rng.IntN(len(g.pool.CraftFarTargetOffsets))],
			CraftTargetItem:   g.pool.CraftFarItems[rng.IntN(len(g.pool.CraftFarItems))],
			CraftSearchRadius: g.pool.CraftFarSearchRadius,
		}
	case TaskChainGather, TaskChainTable, TaskChainPlace, TaskChainUse:
		spec := TaskSpec{
			Type:             typ,
			TargetOffset:     g.pool.ChainTargetOffsets[rng.IntN(len(g.pool.ChainTargetOffsets))],
			MineTargetBlock:  ChainTreeBlock,
			MineSearchRadius: g.pool.ChainSearchRadius,
		}
		switch typ {
		case TaskChainTable, TaskChainPlace:
			spec.CraftTargetItem = ChainTableItem
		case TaskChainUse:
			spec.CraftTargetItem = g.pool.ChainGoalItems[rng.IntN(len(g.pool.ChainGoalItems))]
		}
		return spec
	default:
		// Unreachable: availableTypes only ever returns the cases
		// above, and NewUniformRandom already rejected an empty result.
		panic("curriculum: UniformRandom.Next: unreachable task type")
	}
}

var errEmptyPool = errors.New("curriculum: pool has no candidate values for any task type")
