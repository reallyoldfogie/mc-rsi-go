package main

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxinfer"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/onnxmanifest"
)

const testEnvironmentID = "mc-agent-rlenv:actions=4:obs=6"

func TestRunExportsFeedforwardCheckpoint(t *testing.T) {
	checkpointDir := t.TempDir()
	outDir := t.TempDir()

	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 6, 5, 4, 2)
	if err := lineage.Save(checkpointDir, params, lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        testEnvironmentID,
	}, lineage.PlainCodec); err != nil {
		t.Fatalf("lineage.Save: %v", err)
	}

	if err := run([]string{"-checkpoint-dir=" + checkpointDir, "-out-dir=" + outDir}); err != nil {
		t.Fatalf("run: %v", err)
	}

	manifest, onnxData, err := onnxmanifest.Load(outDir)
	if err != nil {
		t.Fatalf("onnxmanifest.Load: %v", err)
	}
	if manifest.Recurrent {
		t.Error("manifest.Recurrent = true, want false (feedforward checkpoint)")
	}
	if manifest.EnvironmentID != testEnvironmentID {
		t.Errorf("manifest.EnvironmentID = %q, want %q", manifest.EnvironmentID, testEnvironmentID)
	}
	if manifest.InputSize != 6 || manifest.HiddenSize != 5 || manifest.NumHiddenLayers != 2 || manifest.OutputSize != 4 {
		t.Errorf("manifest sizes = %+v, want input=6 hidden=5 layers=2 output=4", manifest)
	}

	model, err := onnxinfer.Load(onnxData)
	if err != nil {
		t.Fatalf("onnxinfer.Load: %v", err)
	}
	outs, err := model.Run(map[string]*onnxinfer.Tensor{
		"observation": {Dims: []int64{1, 6}, Data: []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6}},
	})
	if err != nil {
		t.Fatalf("model.Run: %v", err)
	}
	policy := outs["policy"].Data
	if len(policy) != 4 {
		t.Fatalf("policy output has %d values, want 4", len(policy))
	}
	var sum float32
	for _, p := range policy {
		sum += p
	}
	if diff := abs32(sum - 1); diff > 1e-5 {
		t.Errorf("policy output sums to %v, want ~1 (softmax)", sum)
	}
}

func TestRunExportsRecurrentCheckpoint(t *testing.T) {
	checkpointDir := t.TempDir()
	outDir := t.TempDir()

	rng := rand.New(rand.NewPCG(3, 4))
	params := actorcritic.NewRecurrentParams(rng, 6, 5, 4, 2)
	if err := lineage.Save(checkpointDir, params, lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        testEnvironmentID,
		Recurrent:            true,
	}, lineage.RecurrentCodec); err != nil {
		t.Fatalf("lineage.Save: %v", err)
	}

	if err := run([]string{"-checkpoint-dir=" + checkpointDir, "-out-dir=" + outDir}); err != nil {
		t.Fatalf("run: %v", err)
	}

	manifest, onnxData, err := onnxmanifest.Load(outDir)
	if err != nil {
		t.Fatalf("onnxmanifest.Load: %v", err)
	}
	if !manifest.Recurrent {
		t.Error("manifest.Recurrent = false, want true (recurrent checkpoint)")
	}

	model, err := onnxinfer.Load(onnxData)
	if err != nil {
		t.Fatalf("onnxinfer.Load: %v", err)
	}
	zero := func() *onnxinfer.Tensor { return &onnxinfer.Tensor{Dims: []int64{1, 5}, Data: make([]float32, 5)} }
	outs, err := model.Run(map[string]*onnxinfer.Tensor{
		"observation": {Dims: []int64{1, 6}, Data: []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6}},
		"policy_h_in": zero(),
		"policy_c_in": zero(),
		"value_h_in":  zero(),
		"value_c_in":  zero(),
	})
	if err != nil {
		t.Fatalf("model.Run: %v", err)
	}
	if _, ok := outs["policy_h_out"]; !ok {
		t.Error("model.Run did not produce policy_h_out")
	}
}

func TestRunRejectsMissingFlags(t *testing.T) {
	if err := run(nil); err == nil {
		t.Error("run(nil) = nil error, want one (-checkpoint-dir/-out-dir missing)")
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
