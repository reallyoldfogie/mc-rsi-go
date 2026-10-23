package competition

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Manifest is the reproducibility record written before any live competitor
// starts training.
type Manifest struct {
	RunID            string         `json:"run_id"`
	ConfigHash       string         `json:"config_hash"`
	EvaluationHash   string         `json:"evaluation_plan_hash"`
	EvaluationHashes []string       `json:"evaluation_plan_hashes,omitempty"`
	Window           TrainingWindow `json:"training_window"`
	CreatedAt        time.Time      `json:"created_at"`
	Assignments      []Identity     `json:"assignments"`
}

// NewManifest hashes the resolved config and evaluation plan and captures the
// stable competitor identities. It does not touch the filesystem.
func NewManifest(cfg Config, plans []EvaluationPlan, window TrainingWindow, createdAt time.Time) (Manifest, error) {
	if err := cfg.Validate(); err != nil {
		return Manifest{}, err
	}
	if len(plans) == 0 {
		return Manifest{}, fmt.Errorf("at least one evaluation plan is required")
	}
	if window.StartedAt.IsZero() || window.Deadline.IsZero() || !window.Deadline.After(window.StartedAt) {
		return Manifest{}, fmt.Errorf("invalid training window")
	}
	configHash, err := cfg.Hash()
	if err != nil {
		return Manifest{}, fmt.Errorf("hashing competition config: %w", err)
	}
	evaluationHashes := make([]string, len(plans))
	for i, plan := range plans {
		if err := plan.Validate(); err != nil {
			return Manifest{}, err
		}
		evaluationHashes[i], err = plan.Hash()
		if err != nil {
			return Manifest{}, fmt.Errorf("hashing evaluation plan %d: %w", i, err)
		}
	}
	assignments := cfg.Assignments()
	identities := make([]Identity, 0, len(assignments))
	for _, assignment := range assignments {
		identities = append(identities, assignment.Identity)
	}
	return Manifest{
		RunID: cfg.RunID, ConfigHash: configHash, EvaluationHash: evaluationHashes[0], EvaluationHashes: evaluationHashes,
		Window: window, CreatedAt: createdAt, Assignments: identities,
	}, nil
}

// WriteJSON writes the manifest atomically, creating its parent directory.
func (m Manifest) WriteJSON(path string) error {
	if m.RunID == "" || m.ConfigHash == "" || m.EvaluationHash == "" || len(m.EvaluationHashes) == 0 {
		return fmt.Errorf("manifest is incomplete")
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-")
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
