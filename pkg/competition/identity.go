package competition

import (
	"fmt"
	"path/filepath"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
)

// Identity is the stable identity carried by logs, metrics, dashboard
// snapshots, and checkpoint metadata for one competitor.
type Identity struct {
	RunID        string `json:"run_id"`
	CompetitorID string `json:"competitor_id"`
	DisplayName  string `json:"display_name"`
	AreaIndex    int    `json:"area_index"`
}

// Assignment combines a competitor with its fixed shared-server slot and
// artifact namespace. AreaIndex is deliberately derived from configured
// order, not from the current number of live connections.
type Assignment struct {
	Identity    Identity
	Competitor  Competitor
	AreaOffset  [3]float64
	ArtifactDir string
}

// Assignments returns one stable assignment per configured competitor. The
// offsets are expressed in blocks and are intentionally left as plain data so
// the caller can pass them to the mc-agent/rlenv reset-origin adapter.
func (c Config) Assignments() []Assignment {
	assignments := make([]Assignment, 0, len(c.Competitors))
	for i, competitor := range c.Competitors {
		assignments = append(assignments, Assignment{
			Identity: Identity{
				RunID: c.RunID, CompetitorID: competitor.ID,
				DisplayName: competitor.DisplayName, AreaIndex: i,
			},
			Competitor:  competitor,
			AreaOffset:  parallelenv.WorkingAreaOffset(i, c.Server.SeparationChunks),
			ArtifactDir: filepath.Join(c.CheckpointRoot, competitor.ID),
		})
	}
	return assignments
}

// SharedServerConnection is the connection-side portion of an assignment.
// Settings intentionally retain the base server and RCON addresses; only the
// bot username differs between competitors.
type SharedServerConnection struct {
	Assignment
	Settings mcconfig.Settings
}

// SharedServerConnections derives all competitor settings from one base
// mc-agent configuration. It refuses non-shared configurations so callers
// cannot accidentally use this helper while believing the bots share a
// server.
func (c Config) SharedServerConnections(base mcconfig.Settings) ([]SharedServerConnection, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if !c.Server.Shared {
		return nil, fmt.Errorf("competition: shared-server connections require server.shared=true")
	}
	assignments := c.Assignments()
	connections := make([]SharedServerConnection, 0, len(assignments))
	for _, assignment := range assignments {
		settings := parallelenv.DeriveSharedServerSettings(base, assignment.Identity.AreaIndex, len(assignments))
		connections = append(connections, SharedServerConnection{Assignment: assignment, Settings: settings})
	}
	return connections, nil
}

// Username returns a Minecraft-safe username derived from the configured
// competitor ID. It is only a connection label; metrics and artifacts must
// use CompetitorID as their canonical identity.
func (a Assignment) Username(base string, total int) string {
	if total <= 1 {
		return base
	}
	return boundedUsername(fmt.Sprintf("%s-%s", base, a.Competitor.ID), a.Identity.AreaIndex)
}

func boundedUsername(base string, areaIndex int) string {
	// Keep room for a deterministic numeric suffix even when a caller gives a
	// long base username. Minecraft usernames are limited to 16 characters.
	suffix := fmt.Sprintf("-%d", areaIndex)
	maxBase := 16 - len(suffix)
	if len(base) > maxBase {
		base = base[:maxBase]
	}
	return base + suffix
}
