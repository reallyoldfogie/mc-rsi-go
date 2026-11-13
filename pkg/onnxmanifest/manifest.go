// Package onnxmanifest defines the small JSON file that travels
// alongside an exported model.onnx (docs/plans/23-ONNX-support.md §4.3):
// everything the ONNX graph itself can't express — observation/action
// shape, whether the model is recurrent, and which live environment it
// is and isn't compatible with. cmd/onnx-infer is the only thing that
// reads one; cmd/onnx-export is the only thing that writes one.
package onnxmanifest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// SchemaVersion is this package's own on-disk format version — distinct
// from the ONNX IR/opset version recorded inside model.onnx itself (see
// cRL-go's pkg/onnx.IRVersion/OpsetVersion) and from EnvironmentID
// (which versions the observation/action shape, not this file's shape).
// Load rejects a manifest whose SchemaVersion it doesn't recognize
// rather than guessing at an unfamiliar layout.
const SchemaVersion = 1

const fileName = "manifest.json"

// Manifest is the on-disk JSON shape. Field names are stable once
// released; add fields rather than rename/remove them across a
// SchemaVersion bump.
type Manifest struct {
	SchemaVersion int `json:"schema_version"`

	// EnvironmentID is the exact string this model's checkpoint was
	// trained against (mirrors pkg/lineage.Record.EnvironmentID and
	// cmd/rsi-infer's own environmentID construction:
	// "mc-agent-rlenv:actions=N:obs=M[:norm=1]"). CheckEnvironment
	// requires an exact match — unlike pkg/lineage.Load's "Expanding"
	// loaders, there is no resizing story for an already-exported ONNX
	// graph's fixed tensor shapes, so a mismatch here is always a hard
	// reject, never a widen.
	EnvironmentID string `json:"environment_id"`

	// Recurrent selects which cRL-go exporter/graph shape this model
	// is: false means pkg/onnxexport.ExportFeedforward's single-branch
	// graph (Gemm+Relu trunk, Softmax policy head, scalar value head);
	// true means ExportRecurrent's two-LSTM-branch graph, and Policy*/
	// Value* graph tensor names (pkg/onnxexport's exported constants)
	// are only meaningful when this is true.
	Recurrent bool `json:"recurrent"`

	InputSize       int `json:"input_size"`
	HiddenSize      int `json:"hidden_size"`
	NumHiddenLayers int `json:"num_hidden_layers"`
	OutputSize      int `json:"output_size"`

	// NormalizeObservation must match the live rlenv.Config this model
	// is run against, exactly as EnvironmentID's own ":norm=1" suffix
	// already implies — kept as its own explicit field so a caller
	// doesn't need to parse EnvironmentID's string convention just to
	// configure its environment correctly.
	NormalizeObservation bool `json:"normalize_observation"`

	// Generation and SourceCheckpointDir are provenance only, never
	// checked for compatibility: which pkg/lineage checkpoint this
	// model was exported from, for a human reading the directory later.
	Generation          int    `json:"generation"`
	SourceCheckpointDir string `json:"source_checkpoint_dir"`

	// ONNXFile is the model file's name, relative to this manifest's
	// own directory (always "model.onnx" today; kept as a field, not a
	// hardcoded constant, in case a future manifest ever needs to name
	// more than one file).
	ONNXFile string `json:"onnx_file"`

	// Sha256 is the hex-encoded SHA-256 of ONNXFile's bytes at export
	// time. Load recomputes and compares it, rejecting a manifest
	// whose paired model.onnx has since changed (a half-copied
	// directory, an accidental overwrite) rather than silently running
	// whatever bytes happen to be there.
	Sha256 string `json:"sha256"`

	ExporterVersion string    `json:"exporter_version"`
	CreatedAt       time.Time `json:"created_at"`
}

// Write computes onnxData's checksum, fills Sha256, and writes both m
// (as manifest.json) and onnxData (as m.ONNXFile) into dir, which must
// already exist.
func Write(dir string, m Manifest, onnxData []byte) error {
	if m.ONNXFile == "" {
		return fmt.Errorf("onnxmanifest: ONNXFile must not be empty")
	}
	m.SchemaVersion = SchemaVersion
	m.Sha256 = sha256Hex(onnxData)

	if err := os.WriteFile(filepath.Join(dir, m.ONNXFile), onnxData, 0o644); err != nil {
		return fmt.Errorf("onnxmanifest: writing %s: %w", m.ONNXFile, err)
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("onnxmanifest: encoding manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, fileName), data, 0o644); err != nil {
		return fmt.Errorf("onnxmanifest: writing %s: %w", fileName, err)
	}
	return nil
}

// Load reads manifest.json and its paired ONNXFile from dir, verifying
// the model bytes' checksum still matches Sha256, and returns both the
// decoded Manifest and the model's raw bytes (ready for
// cRL-go/pkg/onnxinfer.Load).
func Load(dir string) (Manifest, []byte, error) {
	var m Manifest
	manifestData, err := os.ReadFile(filepath.Join(dir, fileName))
	if err != nil {
		return Manifest{}, nil, fmt.Errorf("onnxmanifest: reading %s: %w", fileName, err)
	}
	if err := json.Unmarshal(manifestData, &m); err != nil {
		return Manifest{}, nil, fmt.Errorf("onnxmanifest: decoding %s: %w", fileName, err)
	}
	if m.SchemaVersion != SchemaVersion {
		return Manifest{}, nil, fmt.Errorf("onnxmanifest: %s has schema_version %d, this binary only understands %d", fileName, m.SchemaVersion, SchemaVersion)
	}

	onnxData, err := os.ReadFile(filepath.Join(dir, m.ONNXFile))
	if err != nil {
		return Manifest{}, nil, fmt.Errorf("onnxmanifest: reading %s: %w", m.ONNXFile, err)
	}
	if got := sha256Hex(onnxData); got != m.Sha256 {
		return Manifest{}, nil, fmt.Errorf("onnxmanifest: %s checksum %s does not match manifest's recorded %s — model and manifest are out of sync", m.ONNXFile, got, m.Sha256)
	}

	return m, onnxData, nil
}

// CheckEnvironment rejects m outright if it wasn't trained against
// exactly liveEnvironmentID — see EnvironmentID's own doc comment for
// why this is always an exact match, never a resize.
func (m Manifest) CheckEnvironment(liveEnvironmentID string) error {
	if m.EnvironmentID != liveEnvironmentID {
		return fmt.Errorf("onnxmanifest: model trained for environment %q, live environment is %q", m.EnvironmentID, liveEnvironmentID)
	}
	return nil
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
