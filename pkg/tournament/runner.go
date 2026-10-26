package tournament

import (
	"context"
	"fmt"
	"sync"
	"time"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

// PairingOutcome is one pairing's full result (up to Config.BestOf games).
type PairingOutcome struct {
	WinnerCompetitorID   string
	GamesWonA, GamesWonB int
	Games                []pvp.GameOutcome
}

// Run connects every ready pairing's two teams in bracket order, plays up
// to Config.BestOf games through pkg/pvp's own arena/barrier primitives
// (reused unchanged from Phase 1), persists results incrementally, and
// returns the completed bracket. Pairings run sequentially, one at a
// time - later rounds depend on earlier ones anyway, and this keeps a
// single shared server's arena/containment state unambiguous without
// needing concurrent-match bookkeeping (pkg/pvp's own MatchOrigin exists
// for a future multi-match-at-once extension, not used here).
func Run(ctx context.Context, cfg Config) (*Bracket, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.Server.World.Kind != pvp.WorldSourceExisting {
		return nil, fmt.Errorf("tournament: world source kind %q is not yet implemented (only %q)", cfg.Server.World.Kind, pvp.WorldSourceExisting)
	}
	base, err := mcconfig.Load(cfg.Server.World.MCAgentConfig)
	if err != nil {
		return nil, fmt.Errorf("tournament: loading mc-agent config: %w", err)
	}

	competitorIDs := make([]string, len(cfg.Competitors))
	competitorsByID := make(map[string]Competitor, len(cfg.Competitors))
	for i, c := range cfg.Competitors {
		competitorIDs[i] = c.ID
		competitorsByID[c.ID] = c
	}
	bracket, err := NewBracket(competitorIDs)
	if err != nil {
		return nil, err
	}

	results, err := NewResultsWriter(cfg)
	if err != nil {
		return nil, err
	}

	botIndex := 0
	for {
		pairing, ok := bracket.NextReadyPairing()
		if !ok {
			break
		}
		a := competitorsByID[pairing.CompetitorA]
		b := competitorsByID[pairing.CompetitorB]
		outcome, err := runPairing(ctx, cfg, base, &botIndex, a, b)
		if err != nil {
			return bracket, fmt.Errorf("tournament: pairing %q vs %q: %w", a.ID, b.ID, err)
		}
		if err := bracket.RecordResult(pairing.Round, pairing.Slot, outcome.WinnerCompetitorID); err != nil {
			return bracket, err
		}
		if err := results.RecordPairing(pairing, outcome); err != nil {
			return bracket, err
		}
	}

	champion, _ := bracket.Champion()
	if err := results.Finalize(champion); err != nil {
		return bracket, err
	}
	return bracket, nil
}

// runPairing connects every managed participant on both competitors,
// forms the two teams, and plays games until the pairing is decided.
func runPairing(ctx context.Context, cfg Config, base mcconfig.Settings, botIndex *int, a, b Competitor) (PairingOutcome, error) {
	all := append(append([]pvp.Participant{}, a.Participants...), b.Participants...)
	totalBots := 0
	for _, p := range all {
		if p.Managed {
			totalBots++
		}
	}

	connected := make(map[string]*pvp.ConnectedParticipant)
	namesByID := make(map[string]string, len(all))
	var origin *models.V3
	var controller pvp.RCONMatchController
	var rcon testenv.RCONHelper

	closeAll := func() {
		for _, c := range connected {
			_ = c.Close()
		}
	}

	radius := cfg.Arena.Radius
	for _, p := range all {
		if !p.Managed {
			namesByID[p.ID] = p.PlayerName
			continue
		}
		settings := parallelenv.DeriveSharedServerSettings(base, *botIndex, totalBots)
		*botIndex++
		connectedParticipant, resolvedOrigin, err := pvp.ConnectManagedParticipant(ctx, p, settings, radius, origin)
		if err != nil {
			closeAll()
			return PairingOutcome{}, fmt.Errorf("connecting %q: %w", p.ID, err)
		}
		connected[p.ID] = connectedParticipant
		namesByID[p.ID] = connectedParticipant.Username
		if origin == nil {
			originCopy := resolvedOrigin
			origin = &originCopy
		}
		if controller == nil {
			if ctrl, ok := connectedParticipant.Agent.(pvp.RCONMatchController); ok {
				controller = ctrl
				rcon = connectedParticipant.Agent.Config().RCON
			}
		}
	}
	defer closeAll()

	if controller == nil {
		return PairingOutcome{}, fmt.Errorf("no managed participant's agent satisfies RCONMatchController")
	}

	participantNames := make([]string, 0, len(all))
	for _, p := range all {
		participantNames = append(participantNames, namesByID[p.ID])
	}

	// Initial teleport, managed participants only - same safety
	// reasoning as pkg/pvp's own Run(): an unmanaged participant must
	// already be verified near the arena center before this framework
	// does anything to them at all (COMBAT_COMPETITION_PLAN.md's
	// "Unmanaged participants" section); we own every managed bot's
	// connection, so teleporting it unconditionally here is not the
	// same safety concern.
	for _, p := range all {
		if !p.Managed {
			continue
		}
		name := namesByID[p.ID]
		if _, err := rcon.Teleport(ctx, name, origin.X, origin.Y, origin.Z).Exec(ctx); err != nil {
			return PairingOutcome{}, fmt.Errorf("initial teleport of %q: %w", name, err)
		}
	}

	if err := pvp.FormTeams(ctx, controller, all, namesByID, *origin, radius); err != nil {
		return PairingOutcome{}, fmt.Errorf("forming teams: %w", err)
	}

	containmentCtx, cancelContainment := context.WithCancel(ctx)
	defer cancelContainment()
	if cfg.Arena.Containment == pvp.ContainmentWorldBorder {
		if err := pvp.SetWorldBorder(ctx, rcon, *origin, radius); err != nil {
			return PairingOutcome{}, fmt.Errorf("setting world border: %w", err)
		}
	} else {
		go pvp.EnforceContainment(containmentCtx, rcon, participantNames, *origin, radius, cfg.Arena.EffectiveCheckInterval())
	}

	arena := pvp.MatchArena{
		Controller: controller, RCON: rcon, Weapon: cfg.Weapon,
		Center: *origin, Radius: radius, ParticipantNames: participantNames,
	}
	managed := make([]pvp.Participant, 0, len(all))
	for _, p := range all {
		if p.Managed {
			managed = append(managed, p)
		}
	}
	barrier := pvp.NewMatchBarrier(len(managed), arena)

	sessions := make(map[string]*pvp.InferenceSession, len(managed))
	rngSeed := uint64(time.Now().UnixNano())
	for _, p := range managed {
		wrapped := pvp.NewMatchEnvironment(connected[p.ID].Env, barrier)
		session, err := pvp.NewInferenceSession(p.ID, p, p.CheckpointPath, cfg.Greedy, wrapped, rngSeed)
		if err != nil {
			return PairingOutcome{}, fmt.Errorf("building inference session for %q: %w", p.ID, err)
		}
		sessions[p.ID] = session
		rngSeed++
	}

	outcome := PairingOutcome{}
	majority := cfg.BestOf/2 + 1
	for gameIndex := 0; gameIndex < cfg.BestOf; gameIndex++ {
		if outcome.GamesWonA >= majority || outcome.GamesWonB >= majority {
			break
		}
		game, err := playOneGame(ctx, cfg, all, namesByID, sessions, rcon)
		if err != nil {
			return PairingOutcome{}, fmt.Errorf("game %d: %w", gameIndex, err)
		}
		outcome.Games = append(outcome.Games, game)
		switch game.Winner {
		case a.ID:
			outcome.GamesWonA++
		case b.ID:
			outcome.GamesWonB++
		}
	}

	switch {
	case outcome.GamesWonA > outcome.GamesWonB:
		outcome.WinnerCompetitorID = a.ID
	case outcome.GamesWonB > outcome.GamesWonA:
		outcome.WinnerCompetitorID = b.ID
	default:
		// Tied game-win count after Config.BestOf games (possible with
		// one or more drawn games) - break the tie by cumulative damage
		// dealt across every game in this pairing. Not explicitly
		// specified by COMBAT_COMPETITION_PLAN.md decision 6 (which
		// only covers a single game's own timeout), but the most
		// consistent extension of the same principle to a pairing that
		// itself ends level.
		var cumulativeA, cumulativeB float64
		for _, g := range outcome.Games {
			cumulativeA += g.DamageDealt[a.ID]
			cumulativeB += g.DamageDealt[b.ID]
		}
		if cumulativeA >= cumulativeB {
			outcome.WinnerCompetitorID = a.ID
		} else {
			outcome.WinnerCompetitorID = b.ID
		}
	}
	return outcome, nil
}

// playOneGame runs every managed participant's PlayGame concurrently (so
// the shared MatchBarrier actually synchronizes them, the same way
// pkg/pvp's own training sessions run concurrently in Run()), then reads
// every unmanaged participant's health via RCON, and computes the
// game's outcome from all of it together.
func playOneGame(ctx context.Context, cfg Config, all []pvp.Participant, namesByID map[string]string, sessions map[string]*pvp.InferenceSession, rcon testenv.RCONHelper) (pvp.GameOutcome, error) {
	var managed []pvp.Participant
	for _, p := range all {
		if p.Managed {
			managed = append(managed, p)
		}
	}

	var wg sync.WaitGroup
	errs := make([]error, len(managed))
	healths := make([]float64, len(managed))
	for i, p := range managed {
		wg.Add(1)
		go func(i int, p pvp.Participant) {
			defer wg.Done()
			health, err := sessions[p.ID].PlayGame(ctx, cfg.EpisodeLen)
			errs[i], healths[i] = err, health
		}(i, p)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return pvp.GameOutcome{}, err
		}
	}

	readings := make([]pvp.ParticipantHealth, 0, len(all))
	for i, p := range managed {
		readings = append(readings, pvp.ParticipantHealth{Team: p.Team, EndHealth: healths[i]})
	}
	for _, p := range all {
		if p.Managed {
			continue
		}
		health := pvp.ReadHealthRCON(ctx, rcon, namesByID[p.ID])
		readings = append(readings, pvp.ParticipantHealth{Team: p.Team, EndHealth: health})
	}
	return pvp.ComputeGameOutcome(readings)
}
