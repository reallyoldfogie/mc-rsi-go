package tournament

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func validTournamentConfig() Config {
	return Config{
		RunID:       "combat-tourney",
		ResultsRoot: "./results/combat-tourney",
		BestOf:      3,
		EpisodeLen:  200,
		Weapon:      "minecraft:iron_sword",
		Arena:       pvp.ArenaConfig{Containment: pvp.ContainmentSoftArea, Radius: 20},
		Server: pvp.ServerConfig{World: pvp.WorldSource{
			Kind: pvp.WorldSourceExisting, MCAgentConfig: "./mc-agent-config.json",
		}},
		Competitors: []Competitor{
			{ID: "red", DisplayName: "Red Team", Participants: []pvp.Participant{
				{ID: "red1", Managed: true, HiddenLayers: 2, HiddenSize: 128, CheckpointPath: "./checkpoints/red1.json"},
			}},
			{ID: "blue", DisplayName: "Blue Team", Participants: []pvp.Participant{
				{ID: "blue1", Managed: true, HiddenLayers: 2, HiddenSize: 128, CheckpointPath: "./checkpoints/blue1.json"},
			}},
		},
	}
}

func TestConfig_ValidatesCleanly(t *testing.T) {
	if err := validTournamentConfig().Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestConfig_RejectsEvenBestOf(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.BestOf = 2
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for an even best_of")
	}
}

func TestConfig_RejectsZeroEpisodeLen(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.EpisodeLen = 0
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a zero episode_len")
	}
}

func TestConfig_RejectsFewerThanTwoCompetitors(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors = cfg.Competitors[:1]
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for fewer than 2 competitors")
	}
}

func TestConfig_RejectsDuplicateCompetitorIDs(t *testing.T) {
	cfg := validTournamentConfig()
	dup := cfg.Competitors[0]
	cfg.Competitors = append(cfg.Competitors, dup)
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a duplicate competitor id")
	}
}

func TestConfig_RejectsEmptyCompetitorParticipants(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants = nil
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a competitor with no participants")
	}
}

func TestConfig_RejectsManagedParticipantWithoutCheckpoint(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants[0].CheckpointPath = ""
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when a managed participant has no checkpoint_path")
	}
}

func TestConfig_RejectsManagedParticipantWithPlayerName(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants[0].PlayerName = "ShouldNotBeSet"
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when a managed participant sets player_name")
	}
}

func TestConfig_RejectsUnmanagedParticipantWithoutPlayerName(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants = append(cfg.Competitors[0].Participants, pvp.Participant{ID: "ghost"})
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when an unmanaged participant has no player_name")
	}
}

func TestConfig_RejectsDuplicatePlayerNamesAcrossCompetitors(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants = append(cfg.Competitors[0].Participants, pvp.Participant{ID: "p1", PlayerName: "Alice"})
	cfg.Competitors[1].Participants = append(cfg.Competitors[1].Participants, pvp.Participant{ID: "p2", PlayerName: "Alice"})
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error for a duplicate player_name across different competitors")
	}
}

func TestConfig_AllowsUnmanagedEntryAlongsideManaged(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants = append(cfg.Competitors[0].Participants, pvp.Participant{ID: "human1", PlayerName: "Alice"})
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v, want an unmanaged participant alongside a managed one to be valid", err)
	}
}

func TestConfig_RejectsAllUnmanagedCompetitor(t *testing.T) {
	cfg := validTournamentConfig()
	cfg.Competitors[0].Participants = []pvp.Participant{{ID: "human1", PlayerName: "Alice"}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("want an error when a competitor has no managed participant at all")
	}
}

func TestLoadFile_SetsParticipantTeamFromCompetitorID(t *testing.T) {
	dir := t.TempDir()
	cfg := validTournamentConfig()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	configPath := filepath.Join(dir, "tournament.json")
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	loaded, err := LoadFile(configPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if loaded.Competitors[0].Participants[0].Team != "red" {
		t.Errorf("Team = %q, want %q", loaded.Competitors[0].Participants[0].Team, "red")
	}
	if loaded.Competitors[1].Participants[0].Team != "blue" {
		t.Errorf("Team = %q, want %q", loaded.Competitors[1].Participants[0].Team, "blue")
	}
	if !filepath.IsAbs(loaded.Competitors[0].Participants[0].CheckpointPath) {
		t.Errorf("CheckpointPath = %q, want an absolute path", loaded.Competitors[0].Participants[0].CheckpointPath)
	}
	if !filepath.IsAbs(loaded.ResultsRoot) {
		t.Errorf("ResultsRoot = %q, want an absolute path", loaded.ResultsRoot)
	}
}

func TestCompetitor_ManagedParticipants(t *testing.T) {
	c := Competitor{Participants: []pvp.Participant{
		{ID: "a", Managed: true},
		{ID: "b", PlayerName: "Alice"},
		{ID: "c", Managed: true},
	}}
	managed := c.ManagedParticipants()
	if len(managed) != 2 {
		t.Fatalf("ManagedParticipants() = %+v, want 2 entries", managed)
	}
}
