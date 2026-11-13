package onnxmanifest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	onnxData := []byte("not a real onnx file, just bytes for the checksum")

	want := Manifest{
		EnvironmentID:        "mc-agent-rlenv:actions=5:obs=12:norm=1",
		Recurrent:            true,
		InputSize:            12,
		HiddenSize:           64,
		NumHiddenLayers:      2,
		OutputSize:           5,
		NormalizeObservation: true,
		Generation:           7,
		SourceCheckpointDir:  "/runs/example/checkpoints",
		ONNXFile:             "model.onnx",
		ExporterVersion:      "0.0.0-phase4",
	}

	if err := Write(dir, want, onnxData); err != nil {
		t.Fatalf("Write: %v", err)
	}

	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err != nil {
		t.Fatalf("model.onnx not written: %v", err)
	}

	got, gotData, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(gotData) != string(onnxData) {
		t.Errorf("loaded onnx data = %q, want %q", gotData, onnxData)
	}
	if got.EnvironmentID != want.EnvironmentID || got.Recurrent != want.Recurrent ||
		got.InputSize != want.InputSize || got.HiddenSize != want.HiddenSize ||
		got.NumHiddenLayers != want.NumHiddenLayers || got.OutputSize != want.OutputSize ||
		got.NormalizeObservation != want.NormalizeObservation || got.Generation != want.Generation {
		t.Errorf("loaded manifest = %+v, want fields matching %+v", got, want)
	}
	if got.Sha256 == "" {
		t.Error("loaded manifest has no Sha256")
	}

	if err := got.CheckEnvironment(want.EnvironmentID); err != nil {
		t.Errorf("CheckEnvironment(matching) = %v, want nil", err)
	}
	if err := got.CheckEnvironment("mc-agent-rlenv:actions=99:obs=12:norm=1"); err == nil {
		t.Error("CheckEnvironment(mismatched) = nil, want an error")
	}
}

func TestLoadRejectsCorruptedModel(t *testing.T) {
	dir := t.TempDir()
	if err := Write(dir, Manifest{ONNXFile: "model.onnx"}, []byte("original bytes")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("corrupted"), 0o644); err != nil {
		t.Fatalf("corrupting model.onnx: %v", err)
	}

	if _, _, err := Load(dir); err == nil {
		t.Error("Load with a corrupted model.onnx = nil error, want a checksum mismatch error")
	}
}

func TestLoadRejectsUnknownSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	if err := Write(dir, Manifest{ONNXFile: "model.onnx"}, []byte("bytes")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("reading manifest.json: %v", err)
	}
	corrupted := strings.Replace(string(data), `"schema_version": 1`, `"schema_version": 2`, 1)
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(corrupted), 0o644); err != nil {
		t.Fatalf("writing manifest.json: %v", err)
	}

	if _, _, err := Load(dir); err == nil {
		t.Error("Load with an unknown schema_version = nil error, want one")
	}
}
