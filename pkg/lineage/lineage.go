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
//
// Crash safety (docs/plans/04-training-entrypoint-and-observability.md's
// "Done when": a process killed mid-write must not corrupt dir): the
// params file is written first, unconditionally before the record file —
// so a crash between the two can only ever leave a harmless orphaned
// params file with no record ever pointing at it, never a record whose
// params file is missing or truncated. The record file itself is written
// atomically (writeRecordAtomically: a temp file plus os.Rename, not an
// in-place truncate) — Latest/Load key off this exact file's existence
// and contents to decide whether a generation exists at all, so a
// process killed while writing it must only ever be observable as either
// fully present (rename completed) or fully absent (killed before
// rename), never partially written. The underlying
// actorcritic.SaveFile's own params-file write is not itself atomic
// (writes in place, upstream in cRL-go) — a crash mid-params-write can
// still leave a truncated params file on disk, but since that leaves no
// corresponding record file, Latest/Load never observes or tries to load
// it; it's inert, not corruption a caller can trip over.
func Save(dir string, params *actorcritic.Params, rec Record) error {
	if err := rec.Validate(); err != nil {
		return err
	}

	if err := actorcritic.SaveFile(paramsPath(dir, rec.Generation), params, rec.EnvironmentID, rec.Metadata); err != nil {
		return fmt.Errorf("lineage: saving generation %d: %w", rec.Generation, err)
	}

	if err := writeRecordAtomically(recordPath(dir, rec.Generation), rec); err != nil {
		return fmt.Errorf("lineage: saving generation %d record: %w", rec.Generation, err)
	}
	return nil
}

// writeRecordAtomically writes rec as JSON to path via a temp file in the
// same directory plus os.Rename, so path itself only ever changes in one
// atomic filesystem step — see Save's own doc comment for why this
// matters.
func writeRecordAtomically(path string, rec Record) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath) // no-op once the rename below succeeds; cleans up the temp file on any earlier error return.

	if err := json.NewEncoder(tmp).Encode(rec); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
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
