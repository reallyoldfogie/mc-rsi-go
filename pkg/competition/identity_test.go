package competition

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAssignmentsUseConfiguredOrderAndSeparateArtifactDirs(t *testing.T) {
	cfg := validConfig()
	cfg.CheckpointRoot = filepath.Join("runs", "demo")
	cfg.Server.SeparationChunks = 16
	cfg.Competitors = []Competitor{
		{ID: "wide", DisplayName: "Wide", HiddenLayers: 2, HiddenSize: 256},
		{ID: "deep", DisplayName: "Deep", HiddenLayers: 3, HiddenSize: 128},
	}
	assignments := cfg.Assignments()

	assert.Equal(t, 0, assignments[0].Identity.AreaIndex)
	assert.Equal(t, 1, assignments[1].Identity.AreaIndex)
	assert.Equal(t, [3]float64{0, 0, 0}, assignments[0].AreaOffset)
	assert.Equal(t, [3]float64{256, 0, 0}, assignments[1].AreaOffset)
	assert.Equal(t, filepath.Join("runs", "demo", "wide"), assignments[0].ArtifactDir)
	assert.Equal(t, filepath.Join("runs", "demo", "deep"), assignments[1].ArtifactDir)
}

func TestAssignmentUsernameIsStableAndBounded(t *testing.T) {
	cfg := validConfig()
	cfg.Competitors[0].ID = "recurrent-deep-wide"
	assignment := cfg.Assignments()[0]
	first := assignment.Username("very-long-training-bot", 8)
	second := assignment.Username("very-long-training-bot", 8)

	assert.Equal(t, first, second)
	assert.LessOrEqual(t, len(first), 16)
	assert.Contains(t, first, "-0")
}
