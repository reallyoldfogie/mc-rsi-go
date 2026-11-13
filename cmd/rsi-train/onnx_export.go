package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/protobuf/proto"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxexport"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/onnxmanifest"
)

// onnxExporterVersion identifies this automatic per-generation exporter
// in a manifest's ExporterVersion field — distinct from
// cmd/onnx-export's own, since this is a different (automatic, inline)
// code path producing the same file format.
const onnxExporterVersion = "mc-rsi-trainer/cmd/rsi-train (automatic per-generation export, docs/plans/23-ONNX-support.md)"

// savePlainGeneration saves params exactly as lineage.Save(dir, params,
// rec, lineage.PlainCodec) would, additionally exporting an ONNX copy
// (see exportGenerationONNX) when exportONNX is true. An ONNX export
// failure is logged, never returned: the native checkpoint above is the
// authoritative artifact a training run depends on to resume, so a bug
// in the optional ONNX path must never stall or crash a run — see
// docs/plans/23-ONNX-support.md.
func savePlainGeneration(dir string, params *actorcritic.Params, rec lineage.Record, exportONNX bool) error {
	if err := lineage.Save(dir, params, rec, lineage.PlainCodec); err != nil {
		return err
	}
	if !exportONNX {
		return nil
	}
	model, err := onnxexport.ExportFeedforward(params, rec.EnvironmentID)
	if err != nil {
		log.Printf("generation %d: ONNX export failed (training continues): %v", rec.Generation, err)
		return nil
	}
	if err := exportGenerationONNX(dir, rec, model, params.InputSize(), params.HiddenSize(), params.NumHiddenLayers(), params.OutputSize()); err != nil {
		log.Printf("generation %d: ONNX export failed (training continues): %v", rec.Generation, err)
	}
	return nil
}

// saveRecurrentGeneration mirrors savePlainGeneration for a
// *actorcritic.RecurrentParams generation (lineage.RecurrentCodec +
// onnxexport.ExportRecurrent).
func saveRecurrentGeneration(dir string, params *actorcritic.RecurrentParams, rec lineage.Record, exportONNX bool) error {
	if err := lineage.Save(dir, params, rec, lineage.RecurrentCodec); err != nil {
		return err
	}
	if !exportONNX {
		return nil
	}
	model, err := onnxexport.ExportRecurrent(params, rec.EnvironmentID)
	if err != nil {
		log.Printf("generation %d: ONNX export failed (training continues): %v", rec.Generation, err)
		return nil
	}
	if err := exportGenerationONNX(dir, rec, model, params.InputSize(), params.HiddenSize(), params.NumHiddenLayers(), params.OutputSize()); err != nil {
		log.Printf("generation %d: ONNX export failed (training continues): %v", rec.Generation, err)
	}
	return nil
}

// exportGenerationONNX writes model (already built by
// onnxexport.ExportFeedforward/ExportRecurrent for rec's generation) as
// model.onnx + manifest.json (pkg/onnxmanifest's format — the same one
// cmd/onnx-export produces) under dir's onnx/generation-<N> subdirectory,
// alongside the native lineage.Save checkpoint for the same generation.
// A subdirectory per generation, rather than generation-<N>.onnx flat
// alongside lineage's own generation-<N>.json, because onnxmanifest's
// format is a model.onnx+manifest.json pair that must stay named exactly
// that within its own directory — reusing it unmodified (as
// cmd/onnx-export already does) means every consumer of a model
// directory (cmd/onnx-infer, Netron) works identically whether that
// directory came from this automatic path or a manual cmd/onnx-export
// run.
//
// inputSize/hiddenSize/numHiddenLayers/outputSize describe the params
// model was exported from — passed explicitly rather than re-derived
// from model's graph, since ModelProto doesn't expose them back out
// cheaply and the caller already has the real Params/RecurrentParams in
// hand.
func exportGenerationONNX(dir string, rec lineage.Record, model *onnxpb.ModelProto, inputSize, hiddenSize, numHiddenLayers, outputSize int) error {
	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("serializing ONNX model: %w", err)
	}

	outDir := filepath.Join(dir, "onnx", fmt.Sprintf("generation-%09d", rec.Generation))
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", outDir, err)
	}

	manifest := onnxmanifest.Manifest{
		EnvironmentID:        rec.EnvironmentID,
		Recurrent:            rec.Recurrent,
		InputSize:            inputSize,
		HiddenSize:           hiddenSize,
		NumHiddenLayers:      numHiddenLayers,
		OutputSize:           outputSize,
		NormalizeObservation: strings.HasSuffix(rec.EnvironmentID, ":norm=1"), // same convention cmd/onnx-export's normalizeFromEnvironmentID uses
		Generation:           rec.Generation,
		SourceCheckpointDir:  dir,
		ONNXFile:             "model.onnx",
		ExporterVersion:      onnxExporterVersion,
	}
	if err := onnxmanifest.Write(outDir, manifest, data); err != nil {
		return fmt.Errorf("writing %s: %w", outDir, err)
	}
	return nil
}
