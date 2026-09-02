// Package lineage tracks the Teacher/Student generation history behind
// this repo's leapfrog self-play loop: which cRL-go checkpoint became
// which generation, which generation it beat to earn that spot, and
// whether it was produced by leapfrog training or seeded from mc-agent's
// live continuous-learning checkpoint. This bookkeeping is RSI-specific
// and deliberately layered on top of cRL-go's generic
// actorcritic.Params checkpoint format rather than folded into it — see
// docs/plans/00-rsi-trainer-roadmap.md's "Teacher/Student generations
// backed by real checkpoints" section.
//
// This package only persists and retrieves generation records; it does
// not decide when a new generation is created (the leapfrog evaluation
// loop, not yet buildable — see the roadmap's "Current blockers") or
// evaluate one generation against another.
package lineage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
)

// Provenance records how a generation's params came to exist.
type Provenance string

const (
	// ProvenanceLeapfrog is a generation produced by this repo's own
	// leapfrog curriculum training: a Student that beat its Teacher
	// over N evaluation episodes and was cloned into the new Teacher
	// slot.
	ProvenanceLeapfrog Provenance = "leapfrog"

	// ProvenanceSeededLive is a generation seeded from a checkpoint
	// mc-agent persisted after opt-in continuous/online learning during
	// live deployment. Per the roadmap, this is not a tracked
	// generation in its own right until it has been evaluated against
	// the current Teacher through the normal leapfrog evaluation loop
	// like any other candidate — Record.SeededFromGeneration exists so
	// that provenance survives that evaluation.
	ProvenanceSeededLive Provenance = "seeded-live"
)

// Record is the RSI-specific metadata for one generation, stored
// alongside (not inside) the actorcritic.Params checkpoint it describes.
type Record struct {
	// Generation is this record's own generation number. Generation
	// numbers start at 0 for the first Teacher and increase by one per
	// leapfrog round that produces a new Teacher.
	Generation int `json:"generation"`

	// ParentGeneration is the generation this one was cloned or derived
	// from. It is -1 for generation 0, which has no parent.
	ParentGeneration int `json:"parent_generation"`

	// Provenance records how this generation's params came to exist.
	Provenance Provenance `json:"provenance"`

	// SeededFromGeneration is set only when Provenance is
	// ProvenanceSeededLive: the generation whose live deployment
	// produced the continuous-learning checkpoint this generation was
	// seeded from. It is -1 otherwise.
	SeededFromGeneration int `json:"seeded_from_generation"`

	// CreatedAt is when this record was saved.
	CreatedAt time.Time `json:"created_at"`

	// EnvironmentID is the environment/action-space identifier this
	// generation's params were trained against, matching the value
	// passed to the underlying actorcritic checkpoint.
	EnvironmentID string `json:"environment_id"`

	// Metadata is the run-progress metadata saved alongside this
	// generation's underlying actorcritic checkpoint.
	Metadata checkpoint.Metadata `json:"metadata"`
}

// Validate reports whether rec is internally consistent: Generation and
// ParentGeneration are non-negative (except ParentGeneration == -1 for a
// root generation), Provenance is one of the two known values,
// SeededFromGeneration is only set (>= 0) when Provenance is
// ProvenanceSeededLive, and EnvironmentID is non-empty.
func (rec Record) Validate() error {
	if rec.Generation < 0 {
		return fmt.Errorf("lineage: generation %d must be non-negative", rec.Generation)
	}
	if rec.ParentGeneration < -1 {
		return fmt.Errorf("lineage: parent generation %d must be -1 or non-negative", rec.ParentGeneration)
	}
	if rec.ParentGeneration == -1 && rec.Generation != 0 {
		return fmt.Errorf("lineage: generation %d has no parent but is not the root generation", rec.Generation)
	}
	switch rec.Provenance {
	case ProvenanceLeapfrog:
		if rec.SeededFromGeneration != -1 {
			return fmt.Errorf(
				"lineage: generation %d has provenance %q but a seeded-from generation %d set",
				rec.Generation, rec.Provenance, rec.SeededFromGeneration,
			)
		}
	case ProvenanceSeededLive:
		if rec.SeededFromGeneration < 0 {
			return fmt.Errorf(
				"lineage: generation %d has provenance %q but no seeded-from generation set",
				rec.Generation, rec.Provenance,
			)
		}
	default:
		return fmt.Errorf("lineage: generation %d has unknown provenance %q", rec.Generation, rec.Provenance)
	}
	if rec.EnvironmentID == "" {
		return fmt.Errorf("lineage: generation %d has no environment ID", rec.Generation)
	}
	return nil
}

// recordFilePrefix is the file-name prefix Save/Load/Latest use for
// generation record files, distinct from the underlying actorcritic
// checkpoint file so the two can be told apart by name alone.
const recordFilePrefix = "generation"

func recordPath(dir string, generation int) string {
	return filepath.Join(dir, fmt.Sprintf("%s-%09d.json", recordFilePrefix, generation))
}

func paramsPath(dir string, generation int) string {
	return filepath.Join(dir, fmt.Sprintf("%s-%09d-params.json", recordFilePrefix, generation))
}

// Save persists params as an actorcritic checkpoint and rec as its
// accompanying lineage record, both under dir, named after rec's
// generation number. It fails if rec doesn't validate.
func Save(dir string, params *actorcritic.Params, rec Record) error {
	if err := rec.Validate(); err != nil {
		return err
	}

	if err := actorcritic.SaveFile(paramsPath(dir, rec.Generation), params, rec.EnvironmentID, rec.Metadata); err != nil {
		return fmt.Errorf("lineage: saving generation %d: %w", rec.Generation, err)
	}

	file, err := os.Create(recordPath(dir, rec.Generation))
	if err != nil {
		return fmt.Errorf("lineage: saving generation %d record: %w", rec.Generation, err)
	}
	if err := json.NewEncoder(file).Encode(rec); err != nil {
		_ = file.Close()
		return fmt.Errorf("lineage: saving generation %d record: %w", rec.Generation, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("lineage: saving generation %d record: %w", rec.Generation, err)
	}
	return nil
}

// Load reads back the actorcritic checkpoint and lineage record
// previously written by Save for the given generation under dir,
// rejecting the pair if the record's own Generation or EnvironmentID
// don't match what was asked for.
func Load(dir string, generation int, environmentID string) (*actorcritic.Params, Record, error) {
	recordFile, err := os.Open(recordPath(dir, generation))
	if err != nil {
		return nil, Record{}, fmt.Errorf("lineage: loading generation %d record: %w", generation, err)
	}
	defer recordFile.Close()

	var rec Record
	if err := json.NewDecoder(recordFile).Decode(&rec); err != nil {
		return nil, Record{}, fmt.Errorf("lineage: loading generation %d record: %w", generation, err)
	}
	if rec.Generation != generation {
		return nil, Record{}, fmt.Errorf(
			"lineage: generation %d record file contains generation %d", generation, rec.Generation,
		)
	}
	if rec.EnvironmentID != environmentID {
		return nil, Record{}, fmt.Errorf(
			"lineage: generation %d saved for environment %q, want %q",
			generation, rec.EnvironmentID, environmentID,
		)
	}

	params, _, err := actorcritic.LoadFile(paramsPath(dir, generation), environmentID)
	if err != nil {
		return nil, Record{}, fmt.Errorf("lineage: loading generation %d: %w", generation, err)
	}

	return params, rec, nil
}

// Latest returns the record with the highest generation number under
// dir, determined by parsing each matching file name's generation
// number rather than directory-listing or lexical order. It returns an
// error if dir doesn't exist or contains no generation record, so
// callers can fall back to creating generation 0 fresh on either case.
func Latest(dir string) (Record, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return Record{}, fmt.Errorf("lineage: listing %s: %w", dir, err)
	}

	latestGeneration := -1
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, recordFilePrefix+"-") || !strings.HasSuffix(name, ".json") || strings.HasSuffix(name, "-params.json") {
			continue
		}
		numeral := strings.TrimSuffix(strings.TrimPrefix(name, recordFilePrefix+"-"), ".json")
		generation, err := strconv.Atoi(numeral)
		if err != nil {
			continue
		}
		if generation > latestGeneration {
			latestGeneration = generation
		}
	}

	if latestGeneration == -1 {
		return Record{}, fmt.Errorf("lineage: no generation record found in %s", dir)
	}

	recordFile, err := os.Open(recordPath(dir, latestGeneration))
	if err != nil {
		return Record{}, fmt.Errorf("lineage: loading generation %d record: %w", latestGeneration, err)
	}
	defer recordFile.Close()

	var rec Record
	if err := json.NewDecoder(recordFile).Decode(&rec); err != nil {
		return Record{}, fmt.Errorf("lineage: loading generation %d record: %w", latestGeneration, err)
	}
	return rec, nil
}
