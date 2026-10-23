package competition

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadFileResolvesPathsAndDefaultsRecurrentChunkLength(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "competition.json")
	require.NoError(t, os.WriteFile(path, []byte(`{
		"run_id":"test-run",
		"mc_agent_config":"mc-agent.json",
		"curriculum_config":"curriculum.json",
		"checkpoint_root":"runs/test",
		"training":{"duration":"2h","evaluation_start_delay":"5s","epochs_per_generation":2,"evaluation_seed":[7],"evaluation_episodes":3,"evaluation_episode_len":20},
		"server":{"shared":true,"separation_chunks":16},
		"competitors":[{"id":"recurrent","display_name":"Recurrent","hidden_layers":2,"hidden_size":128,"recurrent":true}]
	}`), 0o644))

	cfg, err := LoadFile(path)
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(dir, "mc-agent.json"), cfg.MCAgentConfig)
	assert.Equal(t, filepath.Join(dir, "curriculum.json"), cfg.CurriculumConfig)
	assert.Equal(t, filepath.Join(dir, "runs", "test"), cfg.CheckpointRoot)
	assert.Equal(t, 16, cfg.Competitors[0].RecurrentChunkLen)
	assert.Equal(t, 2*time.Hour, time.Duration(cfg.Training.Duration))
}

func TestConfigValidationRejectsDuplicateAndInvalidCompetitors(t *testing.T) {
	cfg := validConfig()
	cfg.Competitors = append(cfg.Competitors, cfg.Competitors[0])
	err := cfg.Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "duplicate competitor id")

	cfg = validConfig()
	cfg.Competitors[0].ID = "not valid"
	err = cfg.Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "unsupported characters")
}

func TestEvaluationTimeLimitDefaultsTo50PercentFlooredAt15Minutes(t *testing.T) {
	long := TrainingConfig{Duration: Duration(4 * time.Hour)}
	assert.Equal(t, 2*time.Hour, long.EvaluationTimeLimit(), "50% of a 4h training window")

	short := TrainingConfig{Duration: Duration(10 * time.Minute)}
	assert.Equal(t, 15*time.Minute, short.EvaluationTimeLimit(), "50% of 10m is 5m, floored at the 15m minimum")
}

func TestEvaluationTimeLimitHonorsConfiguredPercentAndMinimum(t *testing.T) {
	cfg := TrainingConfig{Duration: Duration(4 * time.Hour), EvaluationTimeLimitPercent: 0.25}
	assert.Equal(t, time.Hour, cfg.EvaluationTimeLimit())

	cfg = TrainingConfig{Duration: Duration(4 * time.Hour), EvaluationTimeLimitPercent: 0.1, EvaluationTimeLimitMin: Duration(90 * time.Minute)}
	assert.Equal(t, 90*time.Minute, cfg.EvaluationTimeLimit(), "10% of 4h (24m) is still below the configured 90m minimum")
}

func TestConfigValidationRejectsNegativeEvaluationTimeLimitFields(t *testing.T) {
	cfg := validConfig()
	cfg.Training.EvaluationTimeLimitPercent = -0.1
	err := cfg.Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "evaluation_time_limit_percent")

	cfg = validConfig()
	cfg.Training.EvaluationTimeLimitMin = Duration(-time.Minute)
	err = cfg.Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "evaluation_time_limit_min")
}

func TestEnvironmentIDIncludesArchitectureShape(t *testing.T) {
	id := EnvironmentID(Competitor{ID: "deep-wide", HiddenLayers: 3, HiddenSize: 256, Recurrent: true, RecurrentChunkLen: 8})
	assert.Equal(t, "architecture=deep-wide:layers=3:width=256:mode=recurrent:chunk=8", id)
}

func TestConfigHashChangesWhenCompetitorChanges(t *testing.T) {
	a := validConfig()
	b := validConfig()
	b.Competitors[0].HiddenSize = 256
	ha, err := a.Hash()
	require.NoError(t, err)
	hb, err := b.Hash()
	require.NoError(t, err)
	assert.NotEqual(t, ha, hb)
}

func validConfig() Config {
	return Config{
		RunID: "test-run", MCAgentConfig: "mc-agent.json", CurriculumConfig: "curriculum.json", CheckpointRoot: "runs/test",
		Training:    TrainingConfig{Duration: Duration(time.Hour), EpochsPerGeneration: 1, EvaluationSeeds: []uint64{7}, EvaluationEpisodes: 1, EvaluationEpisodeLen: 1},
		Server:      ServerConfig{Shared: true, SeparationChunks: 16},
		Competitors: []Competitor{{ID: "basic", DisplayName: "Basic", HiddenLayers: 2, HiddenSize: 128}},
	}
}
