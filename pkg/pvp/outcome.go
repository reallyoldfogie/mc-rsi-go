package pvp

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
)

// combatHealthObservationIndex mirrors mc-agent/rlenv/observation.go's own
// documented layout ("index 5: health"). Duplicated here, not imported,
// since rlenv has no exported constant for it and
// COMBAT_COMPETITION_PLAN.md finding 3 explicitly chose this over adding
// one - the orchestrator reading the observation directly is the intended
// design, not a workaround.
const combatHealthObservationIndex = 5

// FullHealth is every participant's health immediately after
// SeedMatchReset: it unconditionally overheals (instant_health amplifier
// 10, confirmed in Phase 0) past any real player's max health, so "health
// at round start" is always this same known constant rather than
// something that needs to be read back. See
// docs/plans/20-combat-pvp-competition.md.
const FullHealth float64 = 20.0

// Team is an opaque team identifier for outcome bookkeeping - typically a
// Participant's own Team field, or the participant's own ID for a
// teamless (free-for-all) entry so every entrant still has a usable key.
type Team = string

// GameOutcome is one game's (episode's) result.
type GameOutcome struct {
	Winner Team
	Draw   bool
	// DamageDealt is keyed by Team: "damage I dealt" = the opposing
	// side's aggregate health lost this round.
	DamageDealt map[Team]float64
}

// ParticipantHealth is one participant's health reading at round end -
// the only input ComputeGameOutcome needs, from either a managed
// participant's own matchEnvironment.LastObservation() or an unmanaged
// one's RCON-read health (ReadHealthRCON).
type ParticipantHealth struct {
	Team      Team
	EndHealth float64
}

// HealthFromObservation extracts a managed participant's own health from
// its matchEnvironment's most recent Observation - see
// combatHealthObservationIndex's own doc comment.
func HealthFromObservation(obs rl.Observation) float64 {
	if combatHealthObservationIndex >= len(obs.Values) {
		return 0
	}
	return float64(obs.Values[combatHealthObservationIndex])
}

// ReadHealthRCON reads a named participant's current health via RCON -
// the unmanaged-participant counterpart to a managed participant's own
// HealthFromObservation reading. Returns 0 (treated the same as death)
// rather than an error if the entity isn't currently locatable, matching
// VerifyPlayerNearPosition's own "not found is not an error" convention -
// an unmanaged participant who has left the server entirely having 0
// recorded health is also the right outcome for a competition bracket:
// this framework can't distinguish "left" from "died," and neither should
// let a pairing hang waiting for a health reading that will never come.
func ReadHealthRCON(ctx context.Context, rcon testenv.RCONHelper, name string) float64 {
	_, data, err := rcon.GetEntityData(ctx, name, "Health")
	if err != nil {
		return 0
	}
	data = strings.TrimSuffix(strings.TrimSpace(data), "f")
	health, err := strconv.ParseFloat(data, 64)
	if err != nil {
		return 0
	}
	if health < 0 {
		return 0
	}
	return health
}

// ComputeGameOutcome determines a game's winner from every participant's
// health at round end (see docs/plans/20-combat-pvp-competition.md's "Per-
// game outcome" section for the full reasoning): death-first (anyone at
// or below zero health loses their whole team the game outright, unless
// both sides have a death - a mutual kill, possible since the match
// barrier's "ends together" semantics mean one side's death doesn't
// instantly freeze the other's last action - in which case this falls
// back to the same damage-differential rule a timeout uses), then damage
// differential (whichever team's opposing side lost more aggregate
// health wins; an exact tie is a true draw, per
// COMBAT_COMPETITION_PLAN.md decision 6).
func ComputeGameOutcome(readings []ParticipantHealth) (GameOutcome, error) {
	if len(readings) == 0 {
		return GameOutcome{}, fmt.Errorf("pvp: ComputeGameOutcome requires at least one participant reading")
	}

	teams := make(map[Team]bool)
	deadTeams := make(map[Team]bool)
	for _, r := range readings {
		teams[r.Team] = true
		if r.EndHealth <= 0 {
			deadTeams[r.Team] = true
		}
	}
	if len(teams) < 2 {
		return GameOutcome{}, fmt.Errorf("pvp: ComputeGameOutcome requires at least two teams, got %d", len(teams))
	}

	damageDealt := make(map[Team]float64, len(teams))
	for team := range teams {
		var dealt float64
		for _, r := range readings {
			if r.Team != team {
				dealt += FullHealth - r.EndHealth
			}
		}
		damageDealt[team] = dealt
	}

	if len(deadTeams) == 1 {
		var deadTeam Team
		for t := range deadTeams {
			deadTeam = t
		}
		winner, draw := highestDamageAmong(damageDealt, func(t Team) bool { return t != deadTeam })
		return GameOutcome{Winner: winner, Draw: draw, DamageDealt: damageDealt}, nil
	}

	// Either nobody died (timeout) or more than one team had a death
	// (mutual kill) - both resolve by damage differential.
	winner, draw := highestDamageAmong(damageDealt, func(Team) bool { return true })
	return GameOutcome{Winner: winner, Draw: draw, DamageDealt: damageDealt}, nil
}

func highestDamageAmong(damageDealt map[Team]float64, include func(Team) bool) (winner Team, draw bool) {
	best := -1.0
	tied := false
	for team, dealt := range damageDealt {
		if !include(team) {
			continue
		}
		switch {
		case dealt > best:
			best = dealt
			winner = team
			tied = false
		case dealt == best:
			tied = true
		}
	}
	return winner, tied
}
