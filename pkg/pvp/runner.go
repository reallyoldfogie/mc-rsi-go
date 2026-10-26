package pvp

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
)

// defaultMatchSeparationChunks is how far apart (in chunks, along X)
// concurrent matches' arenas are placed on a shared server - see
// MatchOrigin. Not currently exposed in Config; revisit if a real
// deployment needs several matches sharing one server at once (today's
// live verification runs a single match per server).
const defaultMatchSeparationChunks = 32

// matchRuntime is one match's live connections and derived state, built
// by connectMatch and consumed by runMatch.
type matchRuntime struct {
	config           MatchConfig
	connected        map[string]*ConnectedParticipant // keyed by participant ID
	controller       RCONMatchController              // one connected agent, acting for the whole match
	rcon             testenv.RCONHelper               // that same agent's own RCON handle, for Teleport/GetEntityPos/world border
	origin           models.V3
	participantNames []string // every participant's RCON-target name, managed + unmanaged
	namesByID        map[string]string
}

func (rt *matchRuntime) close() {
	for _, c := range rt.connected {
		_ = c.Close()
	}
}

// Run connects every configured match's participants to the shared
// server, forms teams, starts containment enforcement, and trains every
// managed participant continuously until ctx is done (or
// cfg.Training.Duration elapses, if set, whichever comes first). It
// returns once every match's sessions have stopped and every connection
// has been closed.
//
// Only Config.Server.World.Kind == WorldSourceExisting is implemented;
// WorldSourceFresh is accepted by the config schema (see config.go) but
// rejected here with a clear error, matching this codebase's convention
// of failing loudly on an accepted-but-unimplemented path rather than
// silently doing something unintended.
func Run(ctx context.Context, cfg Config) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if cfg.Server.World.Kind != WorldSourceExisting {
		return fmt.Errorf("pvp: world source kind %q is not yet implemented (only %q)", cfg.Server.World.Kind, WorldSourceExisting)
	}
	base, err := mcconfig.Load(cfg.Server.World.MCAgentConfig)
	if err != nil {
		return fmt.Errorf("pvp: loading mc-agent config: %w", err)
	}
	trainerSettings := crlconfig.Default()
	if cfg.TrainerConfig != "" {
		trainerSettings, err = crlconfig.Load(cfg.TrainerConfig)
		if err != nil {
			return fmt.Errorf("pvp: loading trainer config: %w", err)
		}
	}

	totalBots := 0
	for _, m := range cfg.Matches {
		totalBots += len(m.ManagedParticipants())
	}

	runtimes := make([]*matchRuntime, 0, len(cfg.Matches))
	closeAll := func() {
		for _, rt := range runtimes {
			rt.close()
		}
	}

	botIndex := 0
	for matchIndex, matchCfg := range cfg.Matches {
		rt, err := connectMatch(ctx, matchCfg, base, matchIndex, &botIndex, totalBots)
		if err != nil {
			closeAll()
			return fmt.Errorf("pvp: connecting match %q: %w", matchCfg.ID, err)
		}
		runtimes = append(runtimes, rt)
	}
	defer closeAll()

	for _, rt := range runtimes {
		// Every MANAGED participant must physically be at the arena
		// center before FormTeams/ResetMatch can touch them at all:
		// SeedMatchTeam and SeedMatchReset both refuse (via
		// VerifyPlayerNearPosition) to act on anyone outside the arena
		// radius, and a freshly connected bot - other than whichever one
		// the origin itself was derived from - has no reason to already
		// be there. Found live: FormTeams failed on the second of two
		// freshly connected combatants for exactly this reason before
		// this initial teleport was added.
		//
		// Deliberately managed-only: an unmanaged (real player)
		// participant gets no unconditional teleport here or anywhere
		// else - COMBAT_COMPETITION_PLAN.md's "Unmanaged participants"
		// safety rule requires them to already be verified near the
		// arena center before the framework does anything to them at
		// all, including forming a team. We own every managed bot's
		// connection (we just spawned it for this match), so teleporting
		// it unconditionally here is not the same safety concern.
		for _, p := range rt.config.ManagedParticipants() {
			name := rt.namesByID[p.ID]
			if _, err := rt.rcon.Teleport(ctx, name, rt.origin.X, rt.origin.Y, rt.origin.Z).Exec(ctx); err != nil {
				return fmt.Errorf("pvp: initial teleport of %q for match %q: %w", name, rt.config.ID, err)
			}
		}
		if err := FormTeams(ctx, rt.controller, rt.config.Participants, rt.namesByID, rt.origin, rt.config.Arena.Radius); err != nil {
			return fmt.Errorf("pvp: forming teams for match %q: %w", rt.config.ID, err)
		}
		if err := startContainment(ctx, rt); err != nil {
			return fmt.Errorf("pvp: starting containment for match %q: %w", rt.config.ID, err)
		}
	}

	// The training deadline starts here, after every match has finished
	// connecting, forming teams, and starting containment - not from
	// Run's own entry, which would otherwise erode real training time by
	// however long live bot connections happen to take (found live: a
	// short smoke-test deadline expired mid-Reset because two sequential
	// live bot connections alone had already consumed most of it).
	trainCtx := ctx
	if time.Duration(cfg.Training.Duration) > 0 {
		var cancel context.CancelFunc
		trainCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.Training.Duration))
		defer cancel()
	}

	var wg sync.WaitGroup
	errs := make([]error, len(runtimes))
	for i, rt := range runtimes {
		wg.Add(1)
		go func(i int, rt *matchRuntime) {
			defer wg.Done()
			errs[i] = runMatch(trainCtx, rt, cfg, trainerSettings)
		}(i, rt)
	}
	wg.Wait()

	for i, err := range errs {
		if isUnexpectedStopError(err) {
			return fmt.Errorf("pvp: match %q: %w", runtimes[i].config.ID, err)
		}
	}
	return nil
}

// isUnexpectedStopError reports whether err is something other than a
// normal, expected stop (the caller's own context deadline/cancellation).
// Uses errors.Is, not direct equality: a deadline-exceeded/canceled error
// reaching this point has typically been wrapped by several layers
// (ppo.Trainer's own "resetting environment: %w", the RCON library's own
// "rcon exec timeout: %w", ...) by the time it gets here, so comparing
// against the bare sentinel with != would (and, found live, did) never
// match, misreporting a clean deadline-based stop as a hard failure.
func isUnexpectedStopError(err error) bool {
	return err != nil && !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled)
}

// connectMatch connects every managed participant in matchCfg (via
// ConnectManagedParticipant; unmanaged participants have nothing to
// connect) and resolves the match's shared arena origin. botIndex is a
// shared counter across every match in the run, used to derive each
// managed bot's unique username (parallelenv.DeriveSharedServerSettings)
// - every bot on the server needs a globally unique name, not just a
// unique name within its own match.
func connectMatch(ctx context.Context, matchCfg MatchConfig, base mcconfig.Settings, matchIndex int, botIndex *int, totalBots int) (*matchRuntime, error) {
	rt := &matchRuntime{
		config:    matchCfg,
		connected: make(map[string]*ConnectedParticipant),
		namesByID: make(map[string]string),
	}

	var origin *models.V3
	for _, p := range matchCfg.ManagedParticipants() {
		settings := parallelenv.DeriveSharedServerSettings(base, *botIndex, totalBots)
		*botIndex++

		connected, resolvedOrigin, err := ConnectManagedParticipant(ctx, p, settings, matchCfg.Arena.Radius, origin)
		if err != nil {
			rt.close()
			return nil, fmt.Errorf("participant %q: %w", p.ID, err)
		}
		rt.connected[p.ID] = connected
		rt.namesByID[p.ID] = connected.Username
		rt.participantNames = append(rt.participantNames, connected.Username)

		if origin == nil {
			matchOrigin := MatchOrigin(resolvedOrigin, matchIndex, defaultMatchSeparationChunks)
			origin = &matchOrigin
			rt.origin = matchOrigin
		}

		if rt.controller == nil {
			if controller, ok := connected.Agent.(RCONMatchController); ok {
				rt.controller = controller
				rt.rcon = connected.Agent.Config().RCON
			}
		}
	}
	if rt.controller == nil {
		rt.close()
		return nil, fmt.Errorf("no managed participant's agent satisfies RCONMatchController")
	}

	for _, p := range matchCfg.Participants {
		if !p.Managed {
			rt.namesByID[p.ID] = p.PlayerName
			rt.participantNames = append(rt.participantNames, p.PlayerName)
		}
	}

	return rt, nil
}

// startContainment sets up the match's configured containment mode:
// a background poll loop for ContainmentSoftArea (stopped by ctx, not
// tracked further - it has no error to report once started), or a
// one-time vanilla world border command for ContainmentWorldBorder.
func startContainment(ctx context.Context, rt *matchRuntime) error {
	switch rt.config.Arena.Containment {
	case ContainmentWorldBorder:
		return SetWorldBorder(ctx, rt.rcon, rt.origin, rt.config.Arena.Radius)
	default:
		go EnforceContainment(ctx, rt.rcon, rt.participantNames, rt.origin, rt.config.Arena.Radius, rt.config.Arena.EffectiveCheckInterval())
		return nil
	}
}

// runMatch builds the match's synchronized barrier and trains every
// managed participant's session concurrently until ctx is done.
func runMatch(ctx context.Context, rt *matchRuntime, cfg Config, trainerSettings crlconfig.Settings) error {
	arena := MatchArena{
		Controller:       rt.controller,
		RCON:             rt.rcon,
		Weapon:           rt.config.Weapon,
		Center:           rt.origin,
		Radius:           rt.config.Arena.Radius,
		ParticipantNames: rt.participantNames,
	}
	managed := rt.config.ManagedParticipants()
	barrier := NewMatchBarrier(len(managed), arena)

	checkpointDir := filepath.Join(cfg.CheckpointRoot, rt.config.ID)
	if err := os.MkdirAll(checkpointDir, 0o755); err != nil {
		return fmt.Errorf("creating checkpoint directory %s: %w", checkpointDir, err)
	}

	sessions := make([]*ParticipantSession, 0, len(managed))
	for _, p := range managed {
		connected := rt.connected[p.ID]
		wrapped := NewMatchEnvironment(connected.Env, barrier)
		checkpointPath := filepath.Join(checkpointDir, p.ID+".json")
		session, err := NewParticipantSession(p.ID, p, checkpointPath, time.Duration(cfg.Training.CheckpointInterval), trainerSettings, wrapped)
		if err != nil {
			return fmt.Errorf("building session for %q: %w", p.ID, err)
		}
		sessions = append(sessions, session)
	}

	var wg sync.WaitGroup
	errs := make([]error, len(sessions))
	for i, session := range sessions {
		wg.Add(1)
		go func(i int, session *ParticipantSession) {
			defer wg.Done()
			errs[i] = session.Train(ctx)
		}(i, session)
	}
	wg.Wait()

	for _, err := range errs {
		if isUnexpectedStopError(err) {
			return err
		}
	}
	return nil
}
