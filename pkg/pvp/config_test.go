package pvp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func validConfig() Config {
	return Config{
		RunID:          "combat-smoke",
		CheckpointRoot: "./runs/combat-smoke",
		Training:       TrainingConfig{CheckpointInterval: Duration(1)},
		Server: ServerConfig{World: WorldSource{
			Kind:          WorldSourceExisting,
			MCAgentConfig: "./mc-agent-config.json",
		}},
		Matches: []MatchConfig{
			{
				ID:     "arena-1",
				Weapon: "minecraft:iron_sword",
				Arena:  ArenaConfig{Containment: ContainmentSoftArea, Radius: 20},
				Participants: []Participant{
					{ID: "red", Managed: true, HiddenLayers: 2, HiddenSize: 128, Team: "Red"},
					{ID: "blue", Managed: true, HiddenLayers: 2, HiddenSize: 128, Team: "Blue"},
				},
			},
		},
	}
}

func TestConfig_ValidatesCleanly(t *testing.T) {
	if err := validConfig().Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestConfig_RejectsEmptyRunID(t *testing.T) {
	cfg := validConfig()
	cfg.RunID = ""
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for an empty run_id")
	}
}

func TestConfig_RejectsZeroMatches(t *testing.T) {
	cfg := validConfig()
	cfg.Matches = nil
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when no matches are configured")
	}
}

func TestConfig_RejectsDuplicateMatchIDs(t *testing.T) {
	cfg := validConfig()
	cfg.Matches = append(cfg.Matches, cfg.Matches[0])
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for duplicate match IDs")
	}
}

func TestConfig_RejectsFewerThanTwoParticipants(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants = cfg.Matches[0].Participants[:1]
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a match with fewer than 2 participants")
	}
}

func TestConfig_RejectsAllUnmanagedMatch(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants = []Participant{
		{ID: "p1", PlayerName: "Alice"},
		{ID: "p2", PlayerName: "Bob"},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when no participant in a match is managed")
	}
}

func TestConfig_RejectsManagedParticipantWithPlayerName(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants[0].PlayerName = "ShouldNotBeSet"
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when a managed participant sets player_name")
	}
}

func TestConfig_RejectsUnmanagedParticipantWithoutPlayerName(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants = append(cfg.Matches[0].Participants, Participant{ID: "ghost"})
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when an unmanaged participant has no player_name")
	}
}

func TestConfig_RejectsDuplicateUnmanagedPlayerNames(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants = []Participant{
		{ID: "managed", Managed: true, HiddenLayers: 2, HiddenSize: 128},
		{ID: "p1", PlayerName: "Alice"},
		{ID: "p2", PlayerName: "Alice"},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for duplicate unmanaged player_name values")
	}
}

func TestConfig_RejectsZeroArenaRadius(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Arena.Radius = 0
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a zero arena radius")
	}
}

func TestConfig_RejectsWorldBorderWithMultipleMatches(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Arena.Containment = ContainmentWorldBorder
	second := cfg.Matches[0]
	second.ID = "arena-2"
	cfg.Matches = append(cfg.Matches, second)
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when world_border containment is used with more than one match")
	}
}

func TestConfig_AllowsWorldBorderWithExactlyOneMatch(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Arena.Containment = ContainmentWorldBorder
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v, want world_border to be valid with exactly one match", err)
	}
}

func TestConfig_RejectsFreshWorldSourceWithoutGeneration(t *testing.T) {
	cfg := validConfig()
	cfg.Server.World = WorldSource{Kind: WorldSourceFresh}
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when kind=fresh has no generation settings")
	}
}

func TestConfig_RejectsUnknownWorldSourceKind(t *testing.T) {
	cfg := validConfig()
	cfg.Server.World.Kind = "bogus"
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for an unrecognized world source kind")
	}
}

func TestConfig_RejectsRecurrentWithoutChunkLen(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants[0].Recurrent = true
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when recurrent=true but recurrent_chunk_len is unset")
	}
}

func TestLoadFile_ResolvesRelativePaths(t *testing.T) {
	dir := t.TempDir()
	cfg := validConfig()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	configPath := filepath.Join(dir, "pvp-config.json")
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	loaded, err := LoadFile(configPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if !filepath.IsAbs(loaded.CheckpointRoot) {
		t.Errorf("CheckpointRoot = %q, want an absolute path", loaded.CheckpointRoot)
	}
	if !filepath.IsAbs(loaded.Server.World.MCAgentConfig) {
		t.Errorf("Server.World.MCAgentConfig = %q, want an absolute path", loaded.Server.World.MCAgentConfig)
	}
}

func TestMatchConfig_ManagedParticipants(t *testing.T) {
	cfg := validConfig()
	cfg.Matches[0].Participants = append(cfg.Matches[0].Participants, Participant{ID: "ghost", PlayerName: "Carol"})
	managed := cfg.Matches[0].ManagedParticipants()
	if len(managed) != 2 {
		t.Fatalf("ManagedParticipants() = %+v, want 2 entries", managed)
	}
	for _, p := range managed {
		if !p.Managed {
			t.Errorf("ManagedParticipants() included a non-managed entry: %+v", p)
		}
	}
}

func TestArenaConfig_EffectiveCheckInterval(t *testing.T) {
	zero := ArenaConfig{}
	if got := zero.EffectiveCheckInterval(); got != time.Duration(DefaultCheckInterval) {
		t.Errorf("EffectiveCheckInterval() = %v, want default %v", got, DefaultCheckInterval)
	}
	custom := ArenaConfig{CheckInterval: Duration(5 * time.Second)}
	if got := custom.EffectiveCheckInterval(); got != time.Duration(custom.CheckInterval) {
		t.Errorf("EffectiveCheckInterval() = %v, want %v", got, custom.CheckInterval)
	}
}
