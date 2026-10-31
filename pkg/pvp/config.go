package pvp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/sshtunnel"
)

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Duration is a JSON duration written in the same form accepted by
// time.ParseDuration (e.g. "24h", "30s") - mirrors
// pkg/competition.Duration's own shape, kept as a separate type rather
// than imported since pkg/pvp's config schema is otherwise independent
// (see docs/plans/19-combat-pvp-training.md).
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

// Containment selects how a match's arena is bounded - see
// docs/plans/19-combat-pvp-training.md and
// COMBAT_COMPETITION_PLAN.md's "Many matches on one shared server,
// soft-bounded" decision.
type Containment string

const (
	// ContainmentSoftArea actively enforces a working area via periodic
	// RCON position checks, teleporting a combatant back inside on exit -
	// the default, safe to use when the server hosts more than one match
	// concurrently (a vanilla world border would be shared by every match
	// on the server, which is wrong once there's more than one arena).
	ContainmentSoftArea Containment = "soft_area"
	// ContainmentWorldBorder uses vanilla's own world border command.
	// Only valid when the server hosts exactly one match - see
	// Config.Validate.
	ContainmentWorldBorder Containment = "world_border"
)

// WorldSourceKind selects how a match's server is obtained - see
// COMBAT_COMPETITION_PLAN.md's config-driven world-gen requirement
// ("never a hardcoded flat-world assumption").
type WorldSourceKind string

const (
	// WorldSourceExisting connects to a server the operator already runs,
	// terrain/world-gen taken exactly as that server already has it (flat
	// or not - pkg/pvp's own containment/arena logic never assumes flat
	// terrain, since it only ever reasons about horizontal distance from
	// an arena center and RCON teleports, never a fixed absolute Y or a
	// flat-world-only movement command like mc-agent's lineTo).
	WorldSourceExisting WorldSourceKind = "existing"
	// WorldSourceFresh would launch a new server with the given
	// generation settings. Not yet implemented - Config.Validate accepts
	// it (so a config can declare the intent) but LoadFile's caller must
	// reject it before connecting; see Config.Kind's own doc comment.
	WorldSourceFresh WorldSourceKind = "fresh"
)

// WorldSource is the match server's origin.
type WorldSource struct {
	Kind WorldSourceKind `json:"kind"`
	// MCAgentConfig is required when Kind is WorldSourceExisting: a path
	// to an mc-agent connection config (see mcconfig.Settings) for the
	// already-running server to connect to.
	MCAgentConfig string `json:"mc_agent_config,omitempty"`
	// Generation describes a fresh server's world-gen settings, only
	// meaningful (and only required) when Kind is WorldSourceFresh.
	Generation *WorldGeneration `json:"generation,omitempty"`
}

// WorldGeneration is a fresh server's requested world-gen settings.
// Placeholder for the not-yet-implemented WorldSourceFresh path - fields
// are intentionally string-typed/minimal until that path is built.
type WorldGeneration struct {
	Type       string `json:"type,omitempty"` // e.g. "flat", "default", "custom"
	Difficulty string `json:"difficulty,omitempty"`
}

// ArenaConfig is one match's containment settings.
type ArenaConfig struct {
	Containment Containment `json:"containment"`
	// Radius is, depending on Containment: the soft-area enforcement
	// radius in blocks from the match's working-area origin
	// (ContainmentSoftArea), or the world border's own radius
	// (ContainmentWorldBorder).
	Radius float64 `json:"radius"`
	// CheckInterval is the soft-area poll cadence. Ignored (and may be
	// left zero) for ContainmentWorldBorder. Defaults to 2s when zero and
	// Containment is ContainmentSoftArea - see DefaultCheckInterval.
	CheckInterval Duration `json:"check_interval,omitempty"`
}

// DefaultCheckInterval is ArenaConfig.CheckInterval's default when unset.
const DefaultCheckInterval = Duration(2 * time.Second)

// Participant is one combatant in a match.
//
// A managed participant is a trained bot: mc-agent connects and trains it
// via its own cRL-go ppo.Trainer/actorcritic.Params (decision 1,
// "independent learners"). An unmanaged participant is a named real (or
// otherwise externally controlled) player: mc-agent never connects on
// their behalf, and every RCON effect on them (team join, heal/re-equip,
// containment teleport) is gated by VerifyPlayerNearPosition - see
// COMBAT_COMPETITION_PLAN.md's "Unmanaged participants" section.
type Participant struct {
	ID         string `json:"id"`
	PlayerName string `json:"player_name,omitempty"`
	Managed    bool   `json:"managed"`
	// Team is the vanilla scoreboard team name this participant joins, or
	// "" for free-for-all (COMBAT_COMPETITION_PLAN.md decision 2).
	Team string `json:"team,omitempty"`

	// The following are required when Managed is true and ignored
	// otherwise - mirrors pkg/competition.Competitor's architecture
	// fields.
	HiddenLayers      int  `json:"hidden_layers,omitempty"`
	HiddenSize        int  `json:"hidden_size,omitempty"`
	Recurrent         bool `json:"recurrent,omitempty"`
	RecurrentChunkLen int  `json:"recurrent_chunk_len,omitempty"`

	// CheckpointPath is a Phase 2 (pkg/tournament) concern only - an
	// already-trained checkpoint a competition entry loads for pure
	// inference play. Phase 1's own training path (this package's own
	// Config/MatchConfig.Validate) neither sets nor checks it; left here
	// rather than duplicated onto a parallel participant type so
	// pkg/tournament can reuse this type as-is. See
	// docs/plans/20-combat-pvp-competition.md.
	CheckpointPath string `json:"checkpoint_path,omitempty"`
}

// MatchConfig is one arena's roster. Every participant - managed and
// unmanaged alike - shares exactly one working-area origin: unlike
// pkg/parallelenv's own per-competitor offset (which spreads isolated
// learners apart), combatants in the same match must be co-located. See
// docs/plans/19-combat-pvp-training.md.
type MatchConfig struct {
	ID    string      `json:"id"`
	Arena ArenaConfig `json:"arena"`
	// Weapon is the fixed, identical loadout every participant is given
	// at match reset (COMBAT_COMPETITION_PLAN.md decision 4).
	Weapon       string        `json:"weapon"`
	Participants []Participant `json:"participants"`
}

// ManagedParticipants returns every Managed participant in order.
func (m MatchConfig) ManagedParticipants() []Participant {
	managed := make([]Participant, 0, len(m.Participants))
	for _, p := range m.Participants {
		if p.Managed {
			managed = append(managed, p)
		}
	}
	return managed
}

// TrainingConfig controls the overall run's duration and checkpoint
// cadence. Duration of 0 means train indefinitely (until the process is
// canceled) - combat training is continuous (COMBAT_COMPETITION_PLAN.md
// Phase 1, "training mode"), unlike pkg/competition's fixed-budget
// architecture competitions, so an explicit deadline is optional, not
// required.
type TrainingConfig struct {
	Duration           Duration `json:"duration,omitempty"`
	CheckpointInterval Duration `json:"checkpoint_interval"`
}

// ServerConfig describes the shared server every match in this run
// connects to.
type ServerConfig struct {
	World WorldSource `json:"world"`
}

// Config is the JSON-owned pvp training run definition. Paths are made
// absolute relative to the config file by LoadFile.
type Config struct {
	RunID          string            `json:"run_id"`
	CheckpointRoot string            `json:"checkpoint_root"`
	TrainerConfig  string            `json:"trainer_config,omitempty"`
	Training       TrainingConfig    `json:"training"`
	Server         ServerConfig      `json:"server"`
	SSH            *sshtunnel.Config `json:"ssh,omitempty"`
	Matches        []MatchConfig     `json:"matches"`
}

// LoadFile loads and validates a pvp training run definition. Relative
// paths are resolved against the directory containing path before
// validation.
func LoadFile(path string) (Config, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return Config{}, fmt.Errorf("pvp: resolving %s: %w", path, err)
	}
	data, err := os.ReadFile(absolutePath)
	if err != nil {
		return Config{}, fmt.Errorf("pvp: reading %s: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("pvp: parsing %s: %w", path, err)
	}
	base := filepath.Dir(absolutePath)
	resolve := func(value string) string {
		if value == "" || filepath.IsAbs(value) {
			return value
		}
		return filepath.Clean(filepath.Join(base, value))
	}
	cfg.CheckpointRoot = resolve(cfg.CheckpointRoot)
	cfg.TrainerConfig = resolve(cfg.TrainerConfig)
	cfg.Server.World.MCAgentConfig = resolve(cfg.Server.World.MCAgentConfig)

	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("pvp: %s: %w", absolutePath, err)
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
	if strings.TrimSpace(c.CheckpointRoot) == "" {
		return fmt.Errorf("checkpoint_root must not be empty")
	}
	if c.SSH != nil {
		if err := c.SSH.Validate(); err != nil {
			return fmt.Errorf("ssh: %w", err)
		}
	}
	if time.Duration(c.Training.Duration) < 0 {
		return fmt.Errorf("training.duration must not be negative")
	}
	if time.Duration(c.Training.CheckpointInterval) <= 0 {
		return fmt.Errorf("training.checkpoint_interval must be positive")
	}
	if err := c.Server.World.Validate(); err != nil {
		return fmt.Errorf("server.world: %w", err)
	}
	if len(c.Matches) == 0 {
		return fmt.Errorf("matches must contain at least one entry")
	}
	seenMatchIDs := make(map[string]struct{}, len(c.Matches))
	for i := range c.Matches {
		match := &c.Matches[i]
		if !idPattern.MatchString(match.ID) {
			return fmt.Errorf("matches[%d].id %q contains unsupported characters", i, match.ID)
		}
		if _, ok := seenMatchIDs[match.ID]; ok {
			return fmt.Errorf("duplicate match id %q", match.ID)
		}
		seenMatchIDs[match.ID] = struct{}{}
		if err := match.validate(len(c.Matches)); err != nil {
			return fmt.Errorf("match %q: %w", match.ID, err)
		}
	}
	return nil
}

func (w WorldSource) Validate() error {
	switch w.Kind {
	case WorldSourceExisting:
		if strings.TrimSpace(w.MCAgentConfig) == "" {
			return fmt.Errorf("mc_agent_config must not be empty for kind %q", w.Kind)
		}
	case WorldSourceFresh:
		if w.Generation == nil {
			return fmt.Errorf("generation must be set for kind %q", w.Kind)
		}
	default:
		return fmt.Errorf("kind %q is not one of %q, %q", w.Kind, WorldSourceExisting, WorldSourceFresh)
	}
	return nil
}

func (m MatchConfig) validate(totalMatches int) error {
	if m.Weapon == "" {
		return fmt.Errorf("weapon must not be empty")
	}
	if err := m.Arena.Validate(totalMatches); err != nil {
		return fmt.Errorf("arena: %w", err)
	}
	if len(m.Participants) < 2 {
		return fmt.Errorf("participants must contain at least 2 entries")
	}
	seenIDs := make(map[string]struct{}, len(m.Participants))
	seenPlayerNames := make(map[string]struct{}, len(m.Participants))
	hasManaged := false
	for i := range m.Participants {
		p := &m.Participants[i]
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
			if p.Recurrent {
				if p.RecurrentChunkLen <= 0 {
					return fmt.Errorf("managed participant %q recurrent_chunk_len must be positive", p.ID)
				}
			} else if p.RecurrentChunkLen != 0 {
				return fmt.Errorf("managed participant %q sets recurrent_chunk_len but recurrent is false", p.ID)
			}
			if p.PlayerName != "" {
				return fmt.Errorf("managed participant %q must not set player_name (derived from its connected bot username)", p.ID)
			}
		} else {
			if strings.TrimSpace(p.PlayerName) == "" {
				return fmt.Errorf("unmanaged participant %q must set player_name", p.ID)
			}
			if _, ok := seenPlayerNames[p.PlayerName]; ok {
				return fmt.Errorf("duplicate player_name %q", p.PlayerName)
			}
			seenPlayerNames[p.PlayerName] = struct{}{}
		}
	}
	if !hasManaged {
		return fmt.Errorf("at least one participant must be managed (mc-agent trains nothing in an all-unmanaged match)")
	}
	return nil
}

func (a ArenaConfig) Validate(totalMatches int) error {
	if a.Radius <= 0 {
		return fmt.Errorf("radius must be positive")
	}
	switch a.Containment {
	case ContainmentSoftArea:
		if time.Duration(a.CheckInterval) < 0 {
			return fmt.Errorf("check_interval must not be negative")
		}
	case ContainmentWorldBorder:
		if totalMatches > 1 {
			return fmt.Errorf("world_border containment requires exactly one match per server (this run configures %d)", totalMatches)
		}
	default:
		return fmt.Errorf("containment %q is not one of %q, %q", a.Containment, ContainmentSoftArea, ContainmentWorldBorder)
	}
	return nil
}

// EffectiveCheckInterval returns a.CheckInterval, or DefaultCheckInterval
// if a.CheckInterval is zero.
func (a ArenaConfig) EffectiveCheckInterval() time.Duration {
	if a.CheckInterval <= 0 {
		return time.Duration(DefaultCheckInterval)
	}
	return time.Duration(a.CheckInterval)
}
