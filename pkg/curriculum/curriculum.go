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
)

func (t TaskType) String() string {
	switch t {
	case TaskGoto:
		return "goto"
	case TaskMine:
		return "mine"
	case TaskCraft:
		return "craft"
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

	// TargetOffset is meaningful only when Type == TaskGoto.
	TargetOffset [3]float64

	// MineTargetBlock/MineSearchRadius are meaningful only when
	// Type == TaskMine.
	MineTargetBlock  string
	MineSearchRadius int

	// CraftTargetItem is meaningful only when Type == TaskCraft.
	CraftTargetItem string
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
	switch types[rng.IntN(len(types))] {
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
	default:
		// Unreachable: availableTypes only ever returns the three cases
		// above, and NewUniformRandom already rejected an empty result.
		panic("curriculum: UniformRandom.Next: unreachable task type")
	}
}

var errEmptyPool = errors.New("curriculum: pool has no candidate values for any task type")
