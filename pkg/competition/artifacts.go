package competition

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// PrepareRunRoot creates a unique directory for one competition invocation
// beneath the configured checkpoint root. Keeping each invocation in its own
// directory prevents a rerun from silently resuming or overwriting another
// competition's artifacts.
func PrepareRunRoot(checkpointRoot, runID string, startedAt time.Time) (string, error) {
	if checkpointRoot == "" {
		return "", fmt.Errorf("competition: checkpoint root must not be empty")
	}
	if runID == "" {
		return "", fmt.Errorf("competition: run ID must not be empty")
	}
	if startedAt.IsZero() {
		return "", fmt.Errorf("competition: run start time must not be zero")
	}
	if err := os.MkdirAll(checkpointRoot, 0o755); err != nil {
		return "", fmt.Errorf("creating checkpoint root: %w", err)
	}

	// The nanosecond component makes back-to-back invocations distinct while
	// the UTC timestamp keeps the directory names sortable and unambiguous.
	runRoot := filepath.Join(checkpointRoot, startedAt.UTC().Format("20060102T150405.000000000Z"))
	if err := os.Mkdir(runRoot, 0o755); err != nil {
		return "", fmt.Errorf("creating run artifact directory %s: %w", runRoot, err)
	}
	return runRoot, nil
}

// PrepareArtifactDirs creates the per-competitor directories used for model
// lineage, checkpoints, and other competitor-specific artifacts.
func PrepareArtifactDirs(runRoot string, competitors []Competitor) error {
	if runRoot == "" {
		return fmt.Errorf("competition: run artifact root must not be empty")
	}
	for _, competitor := range competitors {
		if competitor.ID == "" {
			return fmt.Errorf("competition: competitor ID must not be empty")
		}
		if err := os.MkdirAll(filepath.Join(runRoot, competitor.ID), 0o755); err != nil {
			return fmt.Errorf("creating artifact directory for competitor %s: %w", competitor.ID, err)
		}
	}
	return nil
}
