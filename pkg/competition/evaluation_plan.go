package competition

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum/rlenvadapter"
)

// EvaluationScenario is a fully resolved curriculum episode. It contains no
// RNG-dependent choices, so every competitor can apply the same scenario even
// when its environment resets at a different speed.
type EvaluationScenario struct {
	Index    int                `json:"index"`
	Task     string             `json:"task"`
	Goal     string             `json:"goal"`
	Override rlenv.TaskOverride `json:"override"`
}

// EvaluationPlan is the immutable, persisted curriculum sequence for one
// competition evaluation phase.
type EvaluationPlan struct {
	Seed      uint64               `json:"seed"`
	Scenarios []EvaluationScenario `json:"scenarios"`
}

// NewEvaluationPlan draws and freezes count scenarios from pool. The plan's
// hash is computed from the serialized plan after construction and can be
// persisted alongside every competitor's results.
func NewEvaluationPlan(pool curriculum.Pool, seed uint64, count int) (EvaluationPlan, error) {
	if err := rlenvadapter.ValidatePool(pool); err != nil {
		return EvaluationPlan{}, err
	}
	if count <= 0 {
		return EvaluationPlan{}, fmt.Errorf("evaluation scenario count must be positive, got %d", count)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	plan := EvaluationPlan{Seed: seed, Scenarios: make([]EvaluationScenario, count)}
	for i := range plan.Scenarios {
		task := rlenvadapter.Select(pool, rng)
		plan.Scenarios[i] = EvaluationScenario{
			Index: i, Task: task.Name(), Goal: task.GoalDescription(), Override: task.ToOverride(),
		}
	}
	return plan, nil
}

// Scenario returns a copy of scenario index i. A caller can safely use the
// result while another competitor is consuming its own copy of the plan.
func (p EvaluationPlan) Scenario(index int) (EvaluationScenario, bool) {
	if index < 0 || index >= len(p.Scenarios) {
		return EvaluationScenario{}, false
	}
	return p.Scenarios[index], true
}

// Validate checks that the plan is internally ordered and contains complete
// task metadata before it is handed to a live environment.
func (p EvaluationPlan) Validate() error {
	if len(p.Scenarios) == 0 {
		return fmt.Errorf("evaluation plan must contain at least one scenario")
	}
	for i, scenario := range p.Scenarios {
		if scenario.Index != i {
			return fmt.Errorf("scenario %d has index %d", i, scenario.Index)
		}
		if scenario.Task == "" || scenario.Goal == "" {
			return fmt.Errorf("scenario %d is missing task or goal", i)
		}
	}
	return nil
}

// Hash returns the SHA-256 of the compact JSON plan. Compact encoding avoids
// formatting differences changing the identity while preserving scenario
// order, which is part of the fairness contract.
func (p EvaluationPlan) Hash() (string, error) {
	if err := p.Validate(); err != nil {
		return "", err
	}
	data, err := json.Marshal(p)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
