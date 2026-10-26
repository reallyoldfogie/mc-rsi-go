package pvp

import (
	"context"
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

	if time.Duration(cfg.Training.Duration) > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(cfg.Training.Duration))
		defer cancel()
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
		if err := FormTeams(ctx, rt.controller, rt.config.Participants, rt.namesByID, rt.origin, rt.config.Arena.Radius); err != nil {
			return fmt.Errorf("pvp: forming teams for match %q: %w", rt.config.ID, err)
		}
		if err := startContainment(ctx, rt); err != nil {
			return fmt.Errorf("pvp: starting containment for match %q: %w", rt.config.ID, err)
		}
	}

	var wg sync.WaitGroup
	errs := make([]error, len(runtimes))
	for i, rt := range runtimes {
		wg.Add(1)
		go func(i int, rt *matchRuntime) {
			defer wg.Done()
			errs[i] = runMatch(ctx, rt, cfg, trainerSettings)
		}(i, rt)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil && err != context.DeadlineExceeded && err != context.Canceled {
			return fmt.Errorf("pvp: match %q: %w", runtimes[i].config.ID, err)
		}
	}
	return nil
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
	barrier := newMatchBarrier(len(managed), arena)

	checkpointDir := filepath.Join(cfg.CheckpointRoot, rt.config.ID)
	if err := os.MkdirAll(checkpointDir, 0o755); err != nil {
		return fmt.Errorf("creating checkpoint directory %s: %w", checkpointDir, err)
	}

	sessions := make([]*ParticipantSession, 0, len(managed))
	for _, p := range managed {
		connected := rt.connected[p.ID]
		wrapped := newMatchEnvironment(connected.Env, barrier)
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
		if err != nil && err != context.DeadlineExceeded && err != context.Canceled {
			return err
		}
	}
	return nil
}
