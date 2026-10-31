package pvp

import (
	"context"
	"fmt"

	"github.com/reallyoldfogie/mc-agent/actions"
	"github.com/reallyoldfogie/mc-agent/agent"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"
)

// ConnectedParticipant is one managed participant's live connection: the
// underlying mc-agent models.Agent (also an RCONMatchController - see
// arena.go) and its rlenv.Environment, not yet wrapped in a
// MatchEnvironment (the caller does that once the match's shared
// MatchBarrier exists).
type ConnectedParticipant struct {
	Participant Participant
	Username    string
	Agent       models.Agent
	Env         *rlenv.Environment
}

// Close shuts down the underlying agent connection.
func (c ConnectedParticipant) Close() error {
	return c.Agent.Close(context.Background())
}

// ConnectManagedParticipant connects one managed participant's bot and
// builds its rlenv.Environment configured for PvP combat
// (COMBAT_COMPETITION_PLAN.md: CombatTargetMobType = "minecraft:player",
// CombatIncludePlayers = true). If origin is nil, this bot's own live
// spawn position becomes the arena origin and is returned as resolved -
// this is how a match's first connected participant establishes the
// shared origin every other participant in the same match must then
// reuse unmodified (pass that value as origin for every later call - see
// MatchOrigin's own doc comment for why one match's own combatants must
// never be offset apart from each other).
//
// This is a deliberately lean connector (no packet logging, no replay
// recording, no curriculum/task-selector wiring) compared to
// cmd/rsi-train's own connectAgent/connectEnvironmentRange - combat
// training poses exactly one task (PvP combat) and has no curriculum to
// select from. A future iteration could add replay recording for
// reviewing matches, but that's deferred, not assumed needed here.
func ConnectManagedParticipant(ctx context.Context, p Participant, settings mcconfig.Settings, radius float64, origin *models.V3) (*ConnectedParticipant, models.V3, error) {
	conn := settings.Connection

	auth, err := agent.ResolveAuth(conn.Offline, conn.Name, conn.UUID, conn.Token, settings.Auth)
	if err != nil {
		return nil, models.V3{}, fmt.Errorf("pvp: resolving auth: %w", err)
	}

	version := conn.Version
	if version == "" {
		detectedVersion, _, err := rofutils.CheckServerVersion(conn.Address, 0)
		if err != nil {
			return nil, models.V3{}, fmt.Errorf("pvp: auto-detecting version from %s: %w", conn.Address, err)
		}
		version = detectedVersion
	}

	rcon, err := agent.DialRCON(ctx, settings.RCON.Address, settings.RCON.Password)
	if err != nil {
		return nil, models.V3{}, fmt.Errorf("pvp: dialing RCON: %w", err)
	}

	cfg := models.AgentConfig{
		Name:             auth.Name,
		Address:          conn.Address,
		Version:          version,
		Auth:             auth,
		MCDataGenPath:    conn.MCDataGenPath,
		MCProtocolGoPath: conn.MCProtocolGoPath,
		StopFilePath:     ".agentStop-pvp-" + auth.Name,
		RCON:             rcon,
	}

	a, err := agent.New(cfg)
	if err != nil {
		return nil, models.V3{}, fmt.Errorf("pvp: creating agent: %w", err)
	}
	if err := a.Init(ctx); err != nil {
		_ = a.Close(context.Background())
		return nil, models.V3{}, fmt.Errorf("pvp: init %s: %w", auth.Name, err)
	}
	if err := a.Start(ctx); err != nil {
		_ = a.Close(context.Background())
		return nil, models.V3{}, fmt.Errorf("pvp: start %s: %w", auth.Name, err)
	}

	liveAgent, ok := a.(rlenv.LiveAgent)
	if !ok {
		_ = a.Close(context.Background())
		return nil, models.V3{}, fmt.Errorf("pvp: agent %s does not satisfy rlenv.LiveAgent", auth.Name)
	}

	if _, err := rcon.Exec(ctx, fmt.Sprintf("op %s", auth.Name)); err != nil {
		_ = a.Close(context.Background())
		return nil, models.V3{}, fmt.Errorf("pvp: granting operator status to %s: %w", auth.Name, err)
	}

	resolvedOrigin := models.V3{}
	if origin != nil {
		resolvedOrigin = *origin
	} else {
		pos, ok := liveAgent.GetPositionSimple()
		if !ok {
			_ = a.Close(context.Background())
			return nil, models.V3{}, fmt.Errorf("pvp: bot %s position not known yet; cannot establish arena origin", auth.Name)
		}
		resolvedOrigin = pos
	}

	// Starts from rlenv.DefaultConfig() rather than a zero-value Config:
	// ArrivalThreshold/StepTimeout are validated as mandatory (> 0) by
	// rlenv.New regardless of whether this Environment poses the goto
	// task they belong to - combat training never uses them itself, but
	// still needs some positive value to pass validation.
	originCopy := [3]float64{resolvedOrigin.X, resolvedOrigin.Y, resolvedOrigin.Z}
	envCfg := rlenv.DefaultConfig()
	envCfg.CombatTargetMobType = "minecraft:player"
	envCfg.CombatIncludePlayers = true
	envCfg.CombatRadius = radius
	envCfg.ResetOrigin = &originCopy
	env, err := rlenv.New(liveAgent, actions.NewRegistry(), envCfg)
	if err != nil {
		_ = a.Close(context.Background())
		return nil, models.V3{}, fmt.Errorf("pvp: building rlenv.Environment for %s: %w", auth.Name, err)
	}

	return &ConnectedParticipant{Participant: p, Agent: a, Env: env, Username: auth.Name}, resolvedOrigin, nil
}
