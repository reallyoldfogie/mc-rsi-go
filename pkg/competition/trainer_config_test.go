package competition

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveTrainerSettingsAppliesArchitectureFields(t *testing.T) {
	cfg := validConfig()
	competitor := cfg.Competitors[0]
	competitor.HiddenSize = 256
	competitor.Recurrent = true
	competitor.RecurrentChunkLen = 12

	settings, err := ResolveTrainerSettings(cfg, competitor)
	require.NoError(t, err)
	assert.Equal(t, 256, settings.HiddenSize)
	assert.Equal(t, 12, settings.RecurrentChunkLen)
}

func TestResolveTrainerSettingsOverridesSeedWhenTrainingSeedIsSet(t *testing.T) {
	cfg := validConfig()
	competitor := cfg.Competitors[0]
	competitor.TrainingSeed = 12345

	settings, err := ResolveTrainerSettings(cfg, competitor)
	require.NoError(t, err)
	assert.Equal(t, uint64(12345), settings.Seed)
}

func TestResolveTrainerSettingsLeavesSeedAloneWhenTrainingSeedIsUnset(t *testing.T) {
	dir := t.TempDir()
	trainerPath := filepath.Join(dir, "trainer.json")
	require.NoError(t, os.WriteFile(trainerPath, []byte(`{"seed": 777}`), 0o644))
	cfg := validConfig()
	competitor := cfg.Competitors[0]
	competitor.TrainerConfig = trainerPath

	settings, err := ResolveTrainerSettings(cfg, competitor)
	require.NoError(t, err)
	assert.Equal(t, uint64(777), settings.Seed, "TrainingSeed==0 must not clobber the trainer config's own seed")
}

func TestResolveTrainerSettingsLoadsCompetitorOverride(t *testing.T) {
	dir := t.TempDir()
	trainerPath := filepath.Join(dir, "trainer.json")
	require.NoError(t, os.WriteFile(trainerPath, []byte(`{
		"learning_rate": 0.012,
		"rollout_size": 32
	}`), 0o644))
	cfg := validConfig()
	competitor := cfg.Competitors[0]
	competitor.TrainerConfig = trainerPath

	settings, err := ResolveTrainerSettings(cfg, competitor)
	require.NoError(t, err)
	assert.Equal(t, float32(0.012), settings.LearningRate)
	assert.Equal(t, 32, settings.RolloutSize)
	assert.Equal(t, competitor.HiddenSize, settings.HiddenSize)
}

func TestResolveTrainerSettingsDisablesRecurrentModeForPlainCompetitor(t *testing.T) {
	dir := t.TempDir()
	trainerPath := filepath.Join(dir, "trainer.json")
	require.NoError(t, os.WriteFile(trainerPath, []byte(`{"recurrent_chunk_len":8}`), 0o644))
	cfg := validConfig()
	cfg.TrainerConfig = trainerPath

	settings, err := ResolveTrainerSettings(cfg, cfg.Competitors[0])
	require.NoError(t, err)
	assert.Zero(t, settings.RecurrentChunkLen)
}
