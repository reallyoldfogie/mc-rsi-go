package pvp

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
)

// RCONMatchController is the narrow slice of a connected mc-agent agent
// pkg/pvp needs to perform match-wide RCON effects - satisfied by
// mc-agent's own agent.New-returned models.Agent (see mc-agent's
// agent/rl_seed.go and agent/combat_teams.go: VerifyPlayerNearPosition,
// SeedMatchReset, SeedMatchTeam - COMBAT_COMPETITION_PLAN.md Phase 0).
// Any one connected participant's agent can act as the controller for
// every participant in the match, managed or unmanaged alike - these are
// generic RCON helpers, not self-referential.
type RCONMatchController interface {
	VerifyPlayerNearPosition(ctx context.Context, playerName string, center models.V3, radius float64) (bool, error)
	SeedMatchReset(ctx context.Context, playerName, weapon string, center models.V3, radius float64) error
	SeedMatchTeam(ctx context.Context, teamName string, memberNames []string, center models.V3, radius float64) error
}

// MatchArena is one match's resolved world/arena state: the fixed, shared
// origin every participant (managed or unmanaged) is healed/re-equipped/
// repositioned to on each round's reset, and the RCON-target names of
// every participant. It implements MatchReset (see barrier.go).
type MatchArena struct {
	Controller RCONMatchController
	// RCON is used directly for repositioning (Teleport), which
	// RCONMatchController does not cover - SeedMatchReset only heals/
	// re-equips (see its own doc comment in mc-agent's agent/rl_seed.go).
	RCON             testenv.RCONHelper
	Weapon           string
	Center           models.V3
	Radius           float64
	ParticipantNames []string
}

// ResetMatch implements MatchReset (barrier.go): repositions, then
// heals and re-equips, every named participant in the match, managed
// and unmanaged alike - an unmanaged participant has no Reset() call of
// its own to synchronize on, so it is reset purely by this RCON side
// effect (COMBAT_COMPETITION_PLAN.md's "Unmanaged participants"
// section). Teleport runs first deliberately: SeedMatchReset's own
// VerifyPlayerNearPosition gate (mc-agent's agent/rl_seed.go) refuses to
// touch anyone currently outside Radius of Center, which a participant
// who hasn't been moved to the arena yet - true for every participant
// except whichever one the arena's own origin was derived from, on the
// very first round - would otherwise always fail. Found live: the
// second of two freshly connected combatants spawned well outside the
// arena and SeedMatchReset refused it before this fix reordered the two
// calls.
func (a MatchArena) ResetMatch(ctx context.Context) error {
	for _, name := range a.ParticipantNames {
		if _, err := a.RCON.Teleport(ctx, name, a.Center.X, a.Center.Y, a.Center.Z).Exec(ctx); err != nil {
			return fmt.Errorf("teleport %q: %w", name, err)
		}
		if err := a.Controller.SeedMatchReset(ctx, name, a.Weapon, a.Center, a.Radius); err != nil {
			return fmt.Errorf("reset %q: %w", name, err)
		}
	}
	return nil
}

// FormTeams groups participants by their configured Team (empty = free-
// for-all, left untouched) and calls SeedMatchTeam once per non-empty
// team. namesByID resolves each participant's configured ID to its
// RCON-target name (a managed participant's connected bot username, or
// an unmanaged participant's player_name). Call once before training
// starts - team membership does not change round to round, unlike
// ResetMatch.
func FormTeams(ctx context.Context, controller RCONMatchController, participants []Participant, namesByID map[string]string, center models.V3, radius float64) error {
	teams := make(map[string][]string)
	for _, p := range participants {
		if p.Team == "" {
			continue
		}
		teams[p.Team] = append(teams[p.Team], namesByID[p.ID])
	}
	for team, members := range teams {
		if err := controller.SeedMatchTeam(ctx, team, members, center, radius); err != nil {
			return fmt.Errorf("form team %q: %w", team, err)
		}
	}
	return nil
}

// MatchOrigin returns the arena origin for the matchIndex-th of several
// concurrent matches sharing one server, offset from base by
// separationChunks chunks along X (see pkg/parallelenv.WorkingAreaOffset).
// Every participant within that one match must then use this exact same
// origin unmodified - unlike pkg/parallelenv's own per-competitor spread,
// which is for isolated, non-interacting learners and would incorrectly
// scatter one match's own combatants apart from each other. See
// docs/plans/19-combat-pvp-training.md.
func MatchOrigin(base models.V3, matchIndex, separationChunks int) models.V3 {
	offset := parallelenv.WorkingAreaOffset(matchIndex, separationChunks)
	return models.V3{X: base.X + offset[0], Y: base.Y + offset[1], Z: base.Z + offset[2]}
}

// EnforceContainment polls every named participant's position every
// interval and teleports anyone found farther than radius from center
// back to center - the ContainmentSoftArea enforcement mechanism. Blocks
// until ctx is done. A participant not currently locatable (offline, or
// between a death and respawn) is skipped for that tick rather than
// treated as an error, matching VerifyPlayerNearPosition's own "not
// found is not an error" convention.
//
// Checks ctx.Err() before each per-name RCON call, not just once per
// tick: found live that a dead/unreachable RCON connection (the
// underlying library's read doesn't itself respect ctx cancellation,
// so a single call can block for several real seconds before timing
// out) otherwise keeps this loop making one doomed call after another,
// for every name, every tick, for as long as ctx takes to actually be
// checked again - visibly still retrying minutes after the caller's own
// context had already been canceled.
func EnforceContainment(ctx context.Context, rcon testenv.RCONHelper, names []string, center models.V3, radius float64, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, name := range names {
				if ctx.Err() != nil {
					return
				}
				x, y, z, err := rcon.GetEntityPos(ctx, name)
				if err != nil {
					continue
				}
				dx, dy, dz := x-center.X, y-center.Y, z-center.Z
				if math.Sqrt(dx*dx+dy*dy+dz*dz) > radius {
					_, _ = rcon.Teleport(ctx, name, center.X, center.Y, center.Z).Exec(ctx)
				}
			}
		}
	}
}

// SetWorldBorder sets a vanilla world border centered on center with
// diameter 2*radius - the ContainmentWorldBorder enforcement mechanism,
// valid only for a server hosting exactly one match (Config.Validate
// rejects it otherwise, since a world border is shared by the whole
// server, not scoped to one arena).
func SetWorldBorder(ctx context.Context, rcon testenv.RCONHelper, center models.V3, radius float64) error {
	if _, err := rcon.Exec(ctx, fmt.Sprintf("worldborder center %f %f", center.X, center.Z)); err != nil {
		return fmt.Errorf("set world border center: %w", err)
	}
	if _, err := rcon.Exec(ctx, fmt.Sprintf("worldborder set %f", radius*2)); err != nil {
		return fmt.Errorf("set world border size: %w", err)
	}
	return nil
}
