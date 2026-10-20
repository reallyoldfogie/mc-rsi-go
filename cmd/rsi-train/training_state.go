package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// persistedTrainingState is deliberately an additive sidecar. The model
// checkpoint remains readable by cRL-go tools, while this file carries the
// continuation data that older checkpoint formats did not know about.
type persistedTrainingState struct {
	SchemaVersion    int                    `json:"schema_version"`
	ParentGeneration int                    `json:"parent_generation"`
	Epoch            int                    `json:"epoch"`
	Optimizer        *actorcritic.AdamState `json:"optimizer,omitempty"`
	RNG              []byte                 `json:"rng,omitempty"`
	Environments     []environmentState     `json:"environments,omitempty"`
}

// environmentState is an extensibility point for environments that can
// expose a serializable live state. The live Minecraft adapter currently
// cannot atomically snapshot a world, so recovery remains safe by resetting
// the affected episode; state-capable test/custom environments get exact
// restoration through this sidecar.
type environmentState struct {
	Available bool            `json:"available"`
	Data      json.RawMessage `json:"data,omitempty"`
}

type statefulEnvironment interface {
	SaveState() (json.RawMessage, error)
	LoadState(json.RawMessage) error
}

func savePersistedTrainingState(dir string, state persistedTrainingState) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	path := filepath.Join(dir, "training-state.json")
	tmp, err := os.CreateTemp(dir, "training-state.json.tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func loadPersistedTrainingState(dir string) (persistedTrainingState, bool) {
	data, err := os.ReadFile(filepath.Join(dir, "training-state.json"))
	if err != nil {
		return persistedTrainingState{}, false
	}
	var state persistedTrainingState
	if json.Unmarshal(data, &state) != nil || state.SchemaVersion != 1 {
		return persistedTrainingState{}, false
	}
	return state, true
}

func captureEnvironmentStates(envs []rl.Environment) []environmentState {
	states := make([]environmentState, len(envs))
	for i, value := range envs {
		if env, ok := value.(statefulEnvironment); ok {
			data, err := env.SaveState()
			if err == nil {
				states[i] = environmentState{Available: true, Data: data}
			}
		}
	}
	return states
}

func restoreEnvironmentStates(envs []rl.Environment, states []environmentState) error {
	for i := range envs {
		if i >= len(states) || !states[i].Available {
			continue
		}
		env, ok := envs[i].(statefulEnvironment)
		if !ok {
			continue
		}
		if err := env.LoadState(states[i].Data); err != nil {
			return fmt.Errorf("environment %d: %w", i, err)
		}
	}
	return nil
}
