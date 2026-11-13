// Command onnx-export converts a pkg/lineage checkpoint into a
// standalone model directory — model.onnx plus manifest.json
// (pkg/onnxmanifest) — that cmd/onnx-infer (or Netron, for model.onnx
// alone) can use without any dependency on cRL-go's training packages
// or this checkpoint directory. See docs/plans/23-ONNX-support.md
// Phase 4.
//
// This is an offline operation: unlike cmd/rsi-infer, it never connects
// to Minecraft. The checkpoint's own pkg/lineage.Record already carries
// the exact EnvironmentID it was trained against, so there is nothing
// to independently reconstruct or validate against a live agent here —
// that validation happens later, in cmd/onnx-infer, against whatever
// live environment the exported model is actually run against.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/protobuf/proto"

	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxexport"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/onnxmanifest"
)

// exporterVersion identifies this command in the manifest's
// ExporterVersion field — not a real semver, since neither this command
// nor cRL-go's exporter packages are tagged independently yet.
const exporterVersion = "mc-rsi-trainer/cmd/onnx-export (docs/plans/23-ONNX-support.md Phase 4)"

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

type flags struct {
	checkpointDir string
	generation    int
	outDir        string
}

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("onnx-export", flag.ContinueOnError)
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "directory of pkg/lineage checkpoints to export from (rsi-train's -checkpoint-dir)")
	fs.IntVar(&f.generation, "generation", -1, "which generation to export; -1 exports the highest generation found (pkg/lineage.Latest)")
	fs.StringVar(&f.outDir, "out-dir", "", "directory to write model.onnx and manifest.json into; created if it doesn't exist")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("onnx-export: -checkpoint-dir is required")
	}
	if f.outDir == "" {
		return flags{}, fmt.Errorf("onnx-export: -out-dir is required")
	}
	return f, nil
}

func run(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}

	// Every generation under one checkpoint-dir shares the same
	// environment and plain-vs-recurrent architecture, exactly as
	// rsi-train/rsi-infer already assume for a single -checkpoint-dir —
	// so Latest's own record tells us both the EnvironmentID to pass to
	// Load (satisfying its exact-match check trivially, since it's the
	// same string Save originally wrote) and which Codec this
	// directory's checkpoints use, without needing a live mc-agent
	// connection to independently reconstruct either one.
	latest, err := lineage.Latest(f.checkpointDir)
	if err != nil {
		return fmt.Errorf("finding latest generation in %s: %w", f.checkpointDir, err)
	}

	generation := f.generation
	if generation < 0 {
		generation = latest.Generation
	}

	if err := os.MkdirAll(f.outDir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", f.outDir, err)
	}

	var model *onnxpb.ModelProto
	var inputSize, hiddenSize, numHiddenLayers, outputSize int
	var rec lineage.Record

	if latest.Recurrent {
		params, r, err := lineage.Load(f.checkpointDir, generation, latest.EnvironmentID, lineage.RecurrentCodec)
		if err != nil {
			return fmt.Errorf("loading generation %d: %w", generation, err)
		}
		rec = r
		model, err = onnxexport.ExportRecurrent(params, rec.EnvironmentID)
		if err != nil {
			return fmt.Errorf("exporting generation %d: %w", generation, err)
		}
		inputSize, hiddenSize, numHiddenLayers, outputSize = params.InputSize(), params.HiddenSize(), params.NumHiddenLayers(), params.OutputSize()
	} else {
		params, r, err := lineage.Load(f.checkpointDir, generation, latest.EnvironmentID, lineage.PlainCodec)
		if err != nil {
			return fmt.Errorf("loading generation %d: %w", generation, err)
		}
		rec = r
		model, err = onnxexport.ExportFeedforward(params, rec.EnvironmentID)
		if err != nil {
			return fmt.Errorf("exporting generation %d: %w", generation, err)
		}
		inputSize, hiddenSize, numHiddenLayers, outputSize = params.InputSize(), params.HiddenSize(), params.NumHiddenLayers(), params.OutputSize()
	}

	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("serializing ONNX model: %w", err)
	}

	absCheckpointDir, err := filepath.Abs(f.checkpointDir)
	if err != nil {
		absCheckpointDir = f.checkpointDir
	}

	manifest := onnxmanifest.Manifest{
		EnvironmentID:        rec.EnvironmentID,
		Recurrent:            latest.Recurrent,
		InputSize:            inputSize,
		HiddenSize:           hiddenSize,
		NumHiddenLayers:      numHiddenLayers,
		OutputSize:           outputSize,
		NormalizeObservation: normalizeFromEnvironmentID(rec.EnvironmentID),
		Generation:           rec.Generation,
		SourceCheckpointDir:  absCheckpointDir,
		ONNXFile:             "model.onnx",
		ExporterVersion:      exporterVersion,
	}
	if err := onnxmanifest.Write(f.outDir, manifest, data); err != nil {
		return fmt.Errorf("writing manifest/model to %s: %w", f.outDir, err)
	}

	log.Printf("exported generation %d (environment=%q, recurrent=%v) to %s", rec.Generation, rec.EnvironmentID, latest.Recurrent, f.outDir)
	return nil
}

// normalizeFromEnvironmentID reports whether environmentID carries
// rsi-infer's own ":norm=1" suffix (see cmd/rsi-infer's environmentID
// construction), so the manifest's NormalizeObservation field doesn't
// require a second, independently-passed flag that could disagree with
// the EnvironmentID the checkpoint actually recorded.
func normalizeFromEnvironmentID(environmentID string) bool {
	return strings.HasSuffix(environmentID, ":norm=1")
}
