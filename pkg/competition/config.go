// Package competition contains the configuration and orchestration building
// blocks for fixed-budget architecture competitions.
package competition

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/sshtunnel"
)

const defaultRecurrentChunkLen = 16

// defaultEvaluationTimeLimitPercent and defaultEvaluationTimeLimitMin are
// TrainingConfig.EvaluationTimeLimit's defaults when the config leaves
// EvaluationTimeLimitPercent/EvaluationTimeLimitMin unset (zero). The
// evaluation phase otherwise has no time limit at all: it runs every
// competitor's full evaluation plan sequentially regardless of training
// duration, and one competitor whose trained policy rarely lets episodes
// terminate early can make evaluation run far longer than training itself
// (seen live: a 4h training window followed by a 10.5h run because one
// competitor's episodes routinely ran to the full per-episode step budget
// instead of ending early). 50% of training time, floored at 15 minutes so
// a short smoke run still gets a usable evaluation window, is a starting
// heuristic, not a measured optimum.
const (
	defaultEvaluationTimeLimitPercent = 0.5
	defaultEvaluationTimeLimitMin     = 15 * time.Minute
)

var competitorIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Duration is a JSON duration written in the same form accepted by
// time.ParseDuration, for example "24h" or "30s".
type Duration time.Duration

func (d *Duration) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("duration must be a string: %w", err)
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", value, err)
	}
	*d = Duration(parsed)
	return nil
}

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

// Config is the JSON-owned competition definition. Paths are made absolute
// relative to the config file by LoadFile, so a resolved Config is portable
// between callers without relying on their working directory.
type Config struct {
	RunID            string            `json:"run_id"`
	MCAgentConfig    string            `json:"mc_agent_config"`
	TrainerConfig    string            `json:"trainer_config,omitempty"`
	CurriculumConfig string            `json:"curriculum_config"`
	CheckpointRoot   string            `json:"checkpoint_root"`
	Training         TrainingConfig    `json:"training"`
	Server           ServerConfig      `json:"server"`
	SSH              *sshtunnel.Config `json:"ssh,omitempty"`
	Competitors      []Competitor      `json:"competitors"`
}

type TrainingConfig struct {
	Duration             Duration `json:"duration"`
	EvaluationStartDelay Duration `json:"evaluation_start_delay"`
	EpochsPerGeneration  int      `json:"epochs_per_generation"`
	EvaluationSeeds      []uint64 `json:"evaluation_seed"`
	EvaluationEpisodes   int      `json:"evaluation_episodes"`
	EvaluationEpisodeLen int      `json:"evaluation_episode_len"`
	EvalGreedy           bool     `json:"eval_greedy"`
	// EvaluationTimeLimitPercent and EvaluationTimeLimitMin tune
	// EvaluationTimeLimit's heuristic; 0 means "use its default" for
	// either field (see defaultEvaluationTimeLimitPercent/Min).
	EvaluationTimeLimitPercent float64  `json:"evaluation_time_limit_percent,omitempty"`
	EvaluationTimeLimitMin     Duration `json:"evaluation_time_limit_min,omitempty"`
}

// EvaluationTimeLimit bounds the competition's evaluation phase: a
// percentage of the training duration (EvaluationTimeLimitPercent, default
// 50%), floored at EvaluationTimeLimitMin (default 15m) so a short training
// window still leaves evaluation a usable amount of time. This is a
// heuristic, not a guarantee the evaluation plan completes - a competitor
// whose policy rarely lets episodes end early can still exhaust it before
// finishing every plan (see pkg/competition.Run, which reports that
// outcome as "evaluation_time_limit_exceeded" rather than as a failure, on
// whatever episodes it did complete).
func (t TrainingConfig) EvaluationTimeLimit() time.Duration {
	percent := t.EvaluationTimeLimitPercent
	if percent <= 0 {
		percent = defaultEvaluationTimeLimitPercent
	}
	minimum := time.Duration(t.EvaluationTimeLimitMin)
	if minimum <= 0 {
		minimum = defaultEvaluationTimeLimitMin
	}
	limit := time.Duration(float64(time.Duration(t.Duration)) * percent)
	if limit < minimum {
		return minimum
	}
	return limit
}

type ServerConfig struct {
	Shared           bool `json:"shared"`
	SeparationChunks int  `json:"separation_chunks"`
	AutoResetOrigin  bool `json:"auto_reset_origin"`
}

// Competitor defines one independently trained policy. TrainerConfig is an
// optional path to a cRL-go settings file that overrides the common trainer
// config for this competitor.
type Competitor struct {
	ID                string `json:"id"`
	DisplayName       string `json:"display_name"`
	HiddenLayers      int    `json:"hidden_layers"`
	HiddenSize        int    `json:"hidden_size"`
	Recurrent         bool   `json:"recurrent"`
	RecurrentChunkLen int    `json:"recurrent_chunk_len"`
	TrainerConfig     string `json:"trainer_config,omitempty"`
	// TrainingSeed overrides the common or competitor trainer config's own
	// seed when non-zero (0 means "use whatever the loaded trainer config
	// specifies" - see crlconfig.Default's own non-zero default, so an
	// unset TrainingSeed is never silently mistaken for a deliberate
	// seed-0 run). This is what lets a competition config run the same
	// architecture against itself under different initializations - e.g.
	// three competitor entries with identical hidden_layers/hidden_size/
	// recurrent, distinct ids, and distinct TrainingSeed values - to tell
	// a structurally broken architecture from one unlucky init, without
	// maintaining a near-duplicate trainer-config file per seed.
	TrainingSeed uint64 `json:"training_seed,omitempty"`
}

// LoadFile loads and validates a competition definition. Relative paths are
// resolved against the directory containing path before validation.
func LoadFile(path string) (Config, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return Config{}, fmt.Errorf("competition: resolving %s: %w", path, err)
	}
	data, err := os.ReadFile(absolutePath)
	if err != nil {
		return Config{}, fmt.Errorf("competition: reading %s: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("competition: parsing %s: %w", path, err)
	}
	base := filepath.Dir(absolutePath)
	for field, value := range map[*string]string{
		&cfg.MCAgentConfig:    cfg.MCAgentConfig,
		&cfg.TrainerConfig:    cfg.TrainerConfig,
		&cfg.CurriculumConfig: cfg.CurriculumConfig,
		&cfg.CheckpointRoot:   cfg.CheckpointRoot,
	} {
		if value != "" && !filepath.IsAbs(value) {
			*field = filepath.Clean(filepath.Join(base, value))
		}
	}
	for i := range cfg.Competitors {
		if value := cfg.Competitors[i].TrainerConfig; value != "" && !filepath.IsAbs(value) {
			cfg.Competitors[i].TrainerConfig = filepath.Clean(filepath.Join(base, value))
		}
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("competition: %s: %w", absolutePath, err)
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.RunID) == "" {
		return fmt.Errorf("run_id must not be empty")
	}
	if !competitorIDPattern.MatchString(c.RunID) {
		return fmt.Errorf("run_id %q contains unsupported characters", c.RunID)
	}
	for name, value := range map[string]string{
		"mc_agent_config": c.MCAgentConfig, "curriculum_config": c.CurriculumConfig,
		"checkpoint_root": c.CheckpointRoot,
	} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("%s must not be empty", name)
		}
	}
	if c.SSH != nil {
		if err := c.SSH.Validate(); err != nil {
			return fmt.Errorf("ssh: %w", err)
		}
	}
	if time.Duration(c.Training.Duration) <= 0 {
		return fmt.Errorf("training.duration must be positive")
	}
	if time.Duration(c.Training.EvaluationStartDelay) < 0 {
		return fmt.Errorf("training.evaluation_start_delay must not be negative")
	}
	if c.Training.EpochsPerGeneration <= 0 {
		return fmt.Errorf("training.epochs_per_generation must be positive")
	}
	if len(c.Training.EvaluationSeeds) == 0 {
		return fmt.Errorf("training.evaluation_seed must contain at least one seed")
	}
	if c.Training.EvaluationEpisodes <= 0 || c.Training.EvaluationEpisodeLen <= 0 {
		return fmt.Errorf("training evaluation episode counts and length must be positive")
	}
	if c.Training.EvaluationTimeLimitPercent < 0 {
		return fmt.Errorf("training.evaluation_time_limit_percent must not be negative")
	}
	if time.Duration(c.Training.EvaluationTimeLimitMin) < 0 {
		return fmt.Errorf("training.evaluation_time_limit_min must not be negative")
	}
	if !c.Server.Shared {
		return fmt.Errorf("server.shared=true is required for a competition")
	}
	if c.Server.SeparationChunks <= 0 {
		return fmt.Errorf("server.separation_chunks must be positive for a shared server")
	}
	if len(c.Competitors) == 0 {
		return fmt.Errorf("competitors must contain at least one entry")
	}
	seen := make(map[string]struct{}, len(c.Competitors))
	for i := range c.Competitors {
		competitor := &c.Competitors[i]
		if !competitorIDPattern.MatchString(competitor.ID) {
			return fmt.Errorf("competitors[%d].id %q contains unsupported characters", i, competitor.ID)
		}
		if _, ok := seen[competitor.ID]; ok {
			return fmt.Errorf("duplicate competitor id %q", competitor.ID)
		}
		seen[competitor.ID] = struct{}{}
		if strings.TrimSpace(competitor.DisplayName) == "" {
			return fmt.Errorf("competitors[%d].display_name must not be empty", i)
		}
		if competitor.HiddenLayers <= 0 || competitor.HiddenSize <= 0 {
			return fmt.Errorf("competitor %q must have positive hidden_layers and hidden_size", competitor.ID)
		}
		if competitor.Recurrent {
			if competitor.RecurrentChunkLen < 0 {
				return fmt.Errorf("competitor %q recurrent_chunk_len must not be negative", competitor.ID)
			}
			if competitor.RecurrentChunkLen == 0 {
				competitor.RecurrentChunkLen = defaultRecurrentChunkLen
			}
		} else if competitor.RecurrentChunkLen != 0 {
			return fmt.Errorf("competitor %q sets recurrent_chunk_len but recurrent is false", competitor.ID)
		}
	}
	return nil
}

// EnvironmentID is the architecture portion of the checkpoint identity. The
// caller appends the live environment's observation/action shape and any
// reward/normalization settings to this value.
func EnvironmentID(c Competitor) string {
	mode := "plain"
	chunk := 0
	if c.Recurrent {
		mode = "recurrent"
		chunk = c.RecurrentChunkLen
	}
	return fmt.Sprintf("architecture=%s:layers=%d:width=%d:mode=%s:chunk=%d", c.ID, c.HiddenLayers, c.HiddenSize, mode, chunk)
}

// Hash returns a stable hash of the canonical JSON representation of the
// resolved configuration. Competitor order is meaningful for area assignment,
// so it is intentionally preserved.
func (c Config) Hash() (string, error) {
	data, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	var canonical bytes.Buffer
	if err := json.Indent(&canonical, data, "", ""); err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

// SortedCompetitorIDs is useful to reporting code that needs deterministic
// output without changing the configured area/training order.
func (c Config) SortedCompetitorIDs() []string {
	ids := make([]string, 0, len(c.Competitors))
	for _, competitor := range c.Competitors {
		ids = append(ids, competitor.ID)
	}
	sort.Strings(ids)
	return ids
}
