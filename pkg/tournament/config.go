package tournament

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Competitor is one bracket entry - a team of one or more participants
// (reusing pvp.Participant's managed/unmanaged shape directly, including
// its Phase-2-only CheckpointPath field). Every participant's own Team
// field is set to this Competitor's own ID automatically by LoadFile
// (not configured per-participant - within one Competitor, every member
// is on the same side by definition).
type Competitor struct {
	ID           string            `json:"id"`
	DisplayName  string            `json:"display_name"`
	Participants []pvp.Participant `json:"participants"`
}

// ManagedParticipants returns every Managed participant in order.
func (c Competitor) ManagedParticipants() []pvp.Participant {
	managed := make([]pvp.Participant, 0, len(c.Participants))
	for _, p := range c.Participants {
		if p.Managed {
			managed = append(managed, p)
		}
	}
	return managed
}

// Config is the JSON-owned competition-mode (bracket) run definition.
type Config struct {
	RunID       string `json:"run_id"`
	ResultsRoot string `json:"results_root"`
	// Greedy selects deterministic (argmax) vs. sampled action selection
	// for every managed participant's inference session.
	Greedy bool `json:"greedy"`
	// BestOf is how many games decide one pairing - must be odd and >=
	// 1; 3 per COMBAT_COMPETITION_PLAN.md decision 3.
	BestOf int `json:"best_of"`
	// EpisodeLen bounds one game's step budget before it resolves by
	// damage differential (decision 6) - mirrors pkg/pvp's own
	// training-side episode length, but pure inference has no
	// cRL-go/pkg/config.Settings of its own to carry it.
	EpisodeLen  int              `json:"episode_len"`
	Weapon      string           `json:"weapon"`
	Arena       pvp.ArenaConfig  `json:"arena"`
	Server      pvp.ServerConfig `json:"server"`
	Competitors []Competitor     `json:"competitors"`
}

// LoadFile loads and validates a competition-mode run definition.
// Relative paths are resolved against the directory containing path
// before validation. Sets every participant's Team to its owning
// Competitor's ID (see Competitor's own doc comment).
func LoadFile(path string) (Config, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return Config{}, fmt.Errorf("tournament: resolving %s: %w", path, err)
	}
	data, err := os.ReadFile(absolutePath)
	if err != nil {
		return Config{}, fmt.Errorf("tournament: reading %s: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("tournament: parsing %s: %w", path, err)
	}
	base := filepath.Dir(absolutePath)
	resolve := func(value string) string {
		if value == "" || filepath.IsAbs(value) {
			return value
		}
		return filepath.Clean(filepath.Join(base, value))
	}
	cfg.ResultsRoot = resolve(cfg.ResultsRoot)
	cfg.Server.World.MCAgentConfig = resolve(cfg.Server.World.MCAgentConfig)
	for i := range cfg.Competitors {
		for j := range cfg.Competitors[i].Participants {
			cfg.Competitors[i].Participants[j].Team = cfg.Competitors[i].ID
			if cfg.Competitors[i].Participants[j].CheckpointPath != "" {
				cfg.Competitors[i].Participants[j].CheckpointPath = resolve(cfg.Competitors[i].Participants[j].CheckpointPath)
			}
		}
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("tournament: %s: %w", absolutePath, err)
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.RunID) == "" {
		return fmt.Errorf("run_id must not be empty")
	}
	if !idPattern.MatchString(c.RunID) {
		return fmt.Errorf("run_id %q contains unsupported characters", c.RunID)
	}
	if strings.TrimSpace(c.ResultsRoot) == "" {
		return fmt.Errorf("results_root must not be empty")
	}
	if c.BestOf < 1 || c.BestOf%2 == 0 {
		return fmt.Errorf("best_of must be a positive odd number, got %d", c.BestOf)
	}
	if c.EpisodeLen <= 0 {
		return fmt.Errorf("episode_len must be positive")
	}
	if c.Weapon == "" {
		return fmt.Errorf("weapon must not be empty")
	}
	if err := c.Arena.Validate(1); err != nil {
		return fmt.Errorf("arena: %w", err)
	}
	if err := c.Server.World.Validate(); err != nil {
		return fmt.Errorf("server.world: %w", err)
	}
	if len(c.Competitors) < 2 {
		return fmt.Errorf("competitors must contain at least 2 entries")
	}
	seenCompetitorIDs := make(map[string]struct{}, len(c.Competitors))
	seenPlayerNames := make(map[string]struct{})
	for i := range c.Competitors {
		competitor := &c.Competitors[i]
		if !idPattern.MatchString(competitor.ID) {
			return fmt.Errorf("competitors[%d].id %q contains unsupported characters", i, competitor.ID)
		}
		if _, ok := seenCompetitorIDs[competitor.ID]; ok {
			return fmt.Errorf("duplicate competitor id %q", competitor.ID)
		}
		seenCompetitorIDs[competitor.ID] = struct{}{}
		if strings.TrimSpace(competitor.DisplayName) == "" {
			return fmt.Errorf("competitors[%d].display_name must not be empty", i)
		}
		if err := validateCompetitorParticipants(*competitor, seenPlayerNames); err != nil {
			return fmt.Errorf("competitor %q: %w", competitor.ID, err)
		}
	}
	return nil
}

func validateCompetitorParticipants(c Competitor, seenPlayerNames map[string]struct{}) error {
	if len(c.Participants) == 0 {
		return fmt.Errorf("must have at least 1 participant")
	}
	hasManaged := false
	seenIDs := make(map[string]struct{}, len(c.Participants))
	for i := range c.Participants {
		p := &c.Participants[i]
		if !idPattern.MatchString(p.ID) {
			return fmt.Errorf("participants[%d].id %q contains unsupported characters", i, p.ID)
		}
		if _, ok := seenIDs[p.ID]; ok {
			return fmt.Errorf("duplicate participant id %q", p.ID)
		}
		seenIDs[p.ID] = struct{}{}
		if p.Managed {
			hasManaged = true
			if p.HiddenLayers <= 0 || p.HiddenSize <= 0 {
				return fmt.Errorf("managed participant %q must have positive hidden_layers and hidden_size", p.ID)
			}
			if p.Recurrent && p.RecurrentChunkLen <= 0 {
				return fmt.Errorf("managed participant %q recurrent_chunk_len must be positive", p.ID)
			} else if !p.Recurrent && p.RecurrentChunkLen != 0 {
				return fmt.Errorf("managed participant %q sets recurrent_chunk_len but recurrent is false", p.ID)
			}
			if strings.TrimSpace(p.CheckpointPath) == "" {
				return fmt.Errorf("managed participant %q must set checkpoint_path (competition mode never trains from scratch)", p.ID)
			}
			if p.PlayerName != "" {
				return fmt.Errorf("managed participant %q must not set player_name", p.ID)
			}
		} else {
			if strings.TrimSpace(p.PlayerName) == "" {
				return fmt.Errorf("unmanaged participant %q must set player_name", p.ID)
			}
			if _, ok := seenPlayerNames[p.PlayerName]; ok {
				return fmt.Errorf("duplicate player_name %q across competitors", p.PlayerName)
			}
			seenPlayerNames[p.PlayerName] = struct{}{}
		}
	}
	if !hasManaged {
		return fmt.Errorf("at least one participant must be managed")
	}
	return nil
}
