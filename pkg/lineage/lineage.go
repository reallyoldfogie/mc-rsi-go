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

	// ProvenanceWidened is a generation produced by cmd/widen-checkpoint
	// applying cRL-go's actorcritic.Net2WiderNet to a generation from
	// another (or, in principle, this same) lineage directory - same
	// "not evaluated against the current Teacher yet" status as
	// ProvenanceSeededLive, but not itself seeded from a live deployment,
	// so it gets its own value rather than overloading that one.
	// cmd/widen-checkpoint always writes it as a fresh generation 0
	// (ParentGeneration -1) in its own output directory, not appended to
	// the source lineage, so SeededFromGeneration plays no role here
	// either - see that command's own doc comment for why, and for where
	// the actual source generation gets recorded (a plain-text note
	// alongside the new generation's files, not this struct).
	ProvenanceWidened Provenance = "widened"

	// ProvenanceDeepened is a generation produced by cmd/deepen-checkpoint
	// applying cRL-go's actorcritic.Net2DeeperNet to a generation from
	// another (or, in principle, this same) lineage directory - the same
	// "fresh generation 0 in its own output directory, source lineage
	// untouched" shape as ProvenanceWidened, and for the same reasons
	// (see that command's own doc comment); kept as its own value rather
	// than reusing ProvenanceWidened so a generation's history records
	// which capacity-growing transform actually produced it.
	ProvenanceDeepened Provenance = "deepened"

	// ProvenanceWarmStarted is a generation produced by warm-starting a
	// non-recurrent generation into a recurrent one
	// (actorcritic.WarmStartRecurrentParams) - triggered automatically by
	// cmd/rsi-train when a recurrent-configured run (RecurrentChunkLen >
	// 0) finds an existing non-recurrent latest generation. Like
	// ProvenanceWidened/ProvenanceDeepened, always written as a fresh
	// generation in a way that never overwrites the source generation in
	// place - see loadOrInitTeacherRecurrent's own doc comment. Record.Recurrent
	// is true for this (and every later, further-trained) generation;
	// Provenance and Recurrent are deliberately orthogonal fields - a
	// warm-started generation that later wins a leapfrog round gets
	// ProvenanceLeapfrog for that later generation, while staying
	// Recurrent, since warm-starting only ever happens once per lineage.
	ProvenanceWarmStarted Provenance = "warm-started"
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

	// Recurrent is true if this generation's underlying checkpoint is an
	// actorcritic.RecurrentParams (saved/loaded via RecurrentCodec)
	// rather than a plain actorcritic.Params (PlainCodec) - the
	// generation's own file carries the actual weights either way;
	// this just tells a caller which Codec to Load it with. false (the
	// zero value) for every generation saved before this field existed,
	// which is exactly correct - every one of them is a plain checkpoint.
	// Kept as its own explicit field rather than inferred from Provenance
	// == ProvenanceWarmStarted, since "is this recurrent" and "how did
	// this generation come to exist" are orthogonal - see
	// ProvenanceWarmStarted's own doc comment.
	Recurrent bool `json:"recurrent,omitempty"`
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
	case ProvenanceWidened, ProvenanceDeepened, ProvenanceWarmStarted:
		if rec.SeededFromGeneration != -1 {
			return fmt.Errorf(
				"lineage: generation %d has provenance %q but a seeded-from generation %d set",
				rec.Generation, rec.Provenance, rec.SeededFromGeneration,
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

// paramsPath is suffixed by codec (Codec.Suffix), so a plain and a
// recurrent generation can coexist in the same directory (e.g. across
// the warm-start transition) without their params files colliding -
// Codec.Suffix is "" for PlainCodec, so this is byte-for-byte the same
// path a pre-generics caller would have written.
func paramsPath(dir string, generation int, suffix string) string {
	return filepath.Join(dir, fmt.Sprintf("%s-%09d%s-params.json", recordFilePrefix, generation, suffix))
}

// Codec bundles the two cRL-go checkpoint functions that differ between
// a plain actorcritic.Params generation and a recurrent
// actorcritic.RecurrentParams one — the only thing Save/Load need to
// know how to do differently per params type P. Suffix distinguishes
// the two kinds' params files on disk (see paramsPath).
type Codec[P any] struct {
	Suffix string
	Save   func(path string, params P, environmentID string, metadata checkpoint.Metadata) error
	Load   func(path string, environmentID string) (P, checkpoint.Metadata, error)
}

// PlainCodec is the Codec for ordinary actorcritic.Params generations —
// every generation saved before recurrence existed, and the default for
// any caller that doesn't train recurrently.
var PlainCodec = Codec[*actorcritic.Params]{
	Suffix: "",
	Save:   actorcritic.SaveFile,
	Load:   actorcritic.LoadFile,
}

// RecurrentCodec is the Codec for actorcritic.RecurrentParams
// generations. RecurrentParams.SaveFile is a method rather than a free
// function (the one API asymmetry Part B's checkpoint design left
// between the two params types), so Save needs a one-line wrapper
// rather than assigning the method directly the way PlainCodec does.
var RecurrentCodec = Codec[*actorcritic.RecurrentParams]{
	Suffix: "-recurrent",
	Save: func(path string, p *actorcritic.RecurrentParams, environmentID string, metadata checkpoint.Metadata) error {
		return p.SaveFile(path, environmentID, metadata)
	},
	Load: actorcritic.RecurrentLoadFile,
}

// Save persists params as a cRL-go checkpoint (via codec) and rec as its
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
// rename), never partially written. The params file uses the same
// temp-file-plus-rename pattern, so a crash can only leave an orphaned
// params file; no record points at it and Latest/Load ignores it.
func Save[P any](dir string, params P, rec Record, codec Codec[P]) error {
	if err := rec.Validate(); err != nil {
		return err
	}

	if err := saveParamsAtomically(paramsPath(dir, rec.Generation, codec.Suffix), params, rec.EnvironmentID, rec.Metadata, codec); err != nil {
		return fmt.Errorf("lineage: saving generation %d: %w", rec.Generation, err)
	}

	if err := writeRecordAtomically(recordPath(dir, rec.Generation), rec); err != nil {
		return fmt.Errorf("lineage: saving generation %d record: %w", rec.Generation, err)
	}
	return nil
}

func saveParamsAtomically[P any](path string, params P, environmentID string, metadata checkpoint.Metadata, codec Codec[P]) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	defer os.Remove(tmpPath)
	if err := codec.Save(tmpPath, params, environmentID, metadata); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
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

// Load reads back the cRL-go checkpoint (via codec) and lineage record
// previously written by Save for the given generation under dir,
// rejecting the pair if the record's own Generation or EnvironmentID
// don't match what was asked for. Pass PlainCodec/RecurrentCodec
// matching whichever codec Save was given - a mismatch (e.g. loading a
// recurrent generation with PlainCodec) looks for the wrong params
// filename and fails with an ordinary "file not found" error, since
// Record.Recurrent (see Latest) is what tells a caller which one to pass.
func Load[P any](dir string, generation int, environmentID string, codec Codec[P]) (P, Record, error) {
	var zero P
	recordFile, err := os.Open(recordPath(dir, generation))
	if err != nil {
		return zero, Record{}, fmt.Errorf("lineage: loading generation %d record: %w", generation, err)
	}
	defer recordFile.Close()

	var rec Record
	if err := json.NewDecoder(recordFile).Decode(&rec); err != nil {
		return zero, Record{}, fmt.Errorf("lineage: loading generation %d record: %w", generation, err)
	}
	if rec.Generation != generation {
		return zero, Record{}, fmt.Errorf(
			"lineage: generation %d record file contains generation %d", generation, rec.Generation,
		)
	}
	if rec.EnvironmentID != environmentID {
		return zero, Record{}, fmt.Errorf(
			"lineage: generation %d saved for environment %q, want %q",
			generation, rec.EnvironmentID, environmentID,
		)
	}

	params, _, err := codec.Load(paramsPath(dir, generation, codec.Suffix), environmentID)
	if err != nil {
		return zero, Record{}, fmt.Errorf("lineage: loading generation %d: %w", generation, err)
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
