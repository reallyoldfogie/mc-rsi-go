// Command rsi-infer connects one live mc-agent bot session and drives it
// with a checkpoint already produced by cmd/rsi-train's leapfrog loop
// (pkg/lineage), instead of training anything further. It exists to
// actually watch/use a trained generation: rsi-train only ever writes
// checkpoints, it never runs one against a live bot for its own sake.
//
// The inference path (cRL-go's actorcritic.NewActor + Actor.Act) has no
// dependency on training-only state (optimizer, gradients) — a
// checkpoint's *actorcritic.Params is enough on its own, so this command
// is a thin loop: load the checkpoint, connect the bot, then repeatedly
// Reset/Act/Step exactly like leapfrog.evaluateEpisode does internally.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	"github.com/reallyoldfogie/mc-agent/actions"
	"github.com/reallyoldfogie/mc-agent/agent"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-agent/utils"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

type flags struct {
	mcAgentConfigPath string
	checkpointDir     string
	generation        int
	episodes          int
	episodeLen        int
	greedy            bool
}

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("rsi-infer", flag.ContinueOnError)
	fs.StringVar(&f.mcAgentConfigPath, "mc-agent-config", "", "path to mc-agent's config.Settings JSON (connection, RCON, env task) — same file rsi-train used for the run this checkpoint came from")
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "directory of pkg/lineage checkpoints to load from (rsi-train's -checkpoint-dir)")
	fs.IntVar(&f.generation, "generation", -1, "which generation to load; -1 loads the highest generation found (pkg/lineage.Latest)")
	fs.IntVar(&f.episodes, "episodes", 0, "number of episodes to run, then exit; 0 runs until interrupted (Ctrl+C)")
	fs.IntVar(&f.episodeLen, "episode-len", 200, "max steps per episode, matching rsi-train's -eval-episode-len default")
	fs.BoolVar(&f.greedy, "greedy", true, "act on the policy's highest-probability action every step (matches rsi-train's own evaluation behavior); false samples from the policy's distribution instead")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.mcAgentConfigPath == "" {
		return flags{}, fmt.Errorf("rsi-infer: -mc-agent-config is required")
	}
	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("rsi-infer: -checkpoint-dir is required")
	}
	if f.episodes < 0 {
		return flags{}, fmt.Errorf("rsi-infer: -episodes must not be negative, got %d", f.episodes)
	}
	if f.episodeLen <= 0 {
		return flags{}, fmt.Errorf("rsi-infer: -episode-len must be positive, got %d", f.episodeLen)
	}
	return f, nil
}

func run(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}

	mcSettings, err := mcconfig.Load(f.mcAgentConfigPath)
	if err != nil {
		return fmt.Errorf("loading mc-agent config: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	a, err := connectAgent(ctx, mcSettings)
	if err != nil {
		return err
	}
	defer func() {
		if err := a.Close(context.Background()); err != nil {
			log.Printf("close error: %v", err)
		}
	}()
	if done := a.Done(); done != nil {
		go func() {
			select {
			case <-ctx.Done():
			case <-done:
				stop()
			}
		}()
	}

	liveAgent, ok := a.(rlenv.LiveAgent)
	if !ok {
		return fmt.Errorf("rsi-infer: agent does not satisfy rlenv.LiveAgent (missing InventoryCount/Craftable/BlockNameAt/HealthProvider?)")
	}

	if err := waitForPosition(ctx, a); err != nil {
		return err
	}

	env, err := rlenv.New(liveAgent, actions.NewRegistry(), mcSettings.Env.ToRlenvConfig())
	if err != nil {
		return fmt.Errorf("constructing environment: %w", err)
	}

	environmentID := fmt.Sprintf("mc-agent-rlenv:actions=%d:obs=%d", env.ActionSpace(), env.ObservationSize())

	generation := f.generation
	if generation < 0 {
		rec, err := lineage.Latest(f.checkpointDir)
		if err != nil {
			return fmt.Errorf("finding latest generation in %s: %w", f.checkpointDir, err)
		}
		generation = rec.Generation
	}
	params, rec, err := lineage.Load(f.checkpointDir, generation, environmentID)
	if err != nil {
		return fmt.Errorf("loading generation %d: %w", generation, err)
	}
	log.Printf("loaded generation %d (parent=%d, provenance=%s, created=%s)", rec.Generation, rec.ParentGeneration, rec.Provenance, rec.CreatedAt.Format(time.RFC3339))

	actor, err := actorcritic.NewActor(params)
	if err != nil {
		return fmt.Errorf("building actor: %w", err)
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))

	var totalReward float32
	episode := 0
	for f.episodes == 0 || episode < f.episodes {
		if err := ctx.Err(); err != nil {
			log.Printf("stopping before episode %d: %v", episode+1, err)
			break
		}
		episode++

		reward, err := runEpisode(ctx, env, actor, f.episodeLen, f.greedy, rng)
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("episode %d interrupted: %v", episode, err)
				break
			}
			return fmt.Errorf("episode %d: %w", episode, err)
		}
		totalReward += reward
		log.Printf("episode %d: reward=%.3f", episode, reward)
	}

	if episode > 0 {
		log.Printf("ran %d episode(s), mean reward=%.3f", episode, totalReward/float32(episode))
	}
	return nil
}

// runEpisode runs one episode with actor and returns its total reward,
// mirroring pkg/leapfrog's own evaluateEpisode: greedy picks the
// highest-probability action every step (deterministic best judgment,
// matching how rsi-train itself scores a Teacher/Student), while
// sampling (greedy=false) draws from the policy's actual distribution
// via Actor.Act instead.
func runEpisode(ctx context.Context, env rl.Environment, actor *actorcritic.Actor, episodeLen int, greedy bool, rng *rand.Rand) (float32, error) {
	observation, err := env.Reset(ctx)
	if err != nil {
		return 0, fmt.Errorf("resetting environment: %w", err)
	}

	var totalReward float32
	for step := 0; step < episodeLen; step++ {
		if err := ctx.Err(); err != nil {
			return totalReward, err
		}

		var mask []bool
		if masker, ok := env.(rl.ActionMasker); ok {
			mask = masker.ActionMask()
		}

		var action rl.Action
		if greedy {
			decision, err := actor.ActWithInfo(observation, mask, rng)
			if err != nil {
				return totalReward, fmt.Errorf("deciding action: %w", err)
			}
			action = greedyAction(decision.Probabilities)
		} else {
			action, err = actor.Act(observation, mask, rng)
			if err != nil {
				return totalReward, fmt.Errorf("deciding action: %w", err)
			}
		}

		result, err := env.Step(ctx, action)
		if err != nil {
			return totalReward, fmt.Errorf("stepping environment: %w", err)
		}
		totalReward += result.Reward
		log.Printf("  step %d: action=%s reward=%.3f", step+1, actionName(action), result.Reward)
		observation = result.Observation
		if result.Done {
			break
		}
	}
	return totalReward, nil
}

// greedyAction returns the index of the highest probability in probs,
// breaking ties toward the lowest index — identical to
// pkg/leapfrog.greedyAction, duplicated here since that one is
// unexported and this command has no other reason to depend on
// pkg/leapfrog.
func greedyAction(probs []float32) rl.Action {
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return rl.Action(best)
}

// actionName renders action using rlenv's own exported action constants,
// falling back to its bare numeral if it's out of range (defensive only —
// resolveDispatch would already have rejected such a value in env.Step).
func actionName(action rl.Action) string {
	switch action {
	case rlenv.ActionWait:
		return "Wait"
	case rlenv.ActionGoToTarget:
		return "GoToTarget"
	case rlenv.ActionMine:
		return "Mine"
	case rlenv.ActionCraft:
		return "Craft"
	default:
		return fmt.Sprintf("Action(%d)", action)
	}
}

// positionWaitTimeout bounds waitForPosition: a.Init returning doesn't
// guarantee the server's first position-bearing packet has arrived yet
// (unlike cmd/rsi-train, which has -parallel-envs other bots' own
// connect/setup work filling that gap for free before its first
// env.Reset — see applyAutoResetOrigin's doc comment for the same race),
// so env.Reset's own errPositionUnknown can fire immediately after
// connecting a single bot here. 10s comfortably covers a normal
// login/spawn sequence without hanging indefinitely on a truly stuck
// connection.
const positionWaitTimeout = 30 * time.Second

// waitForPosition polls a's position until known or positionWaitTimeout
// elapses, so the caller's first env.Reset doesn't race a's own
// in-flight join/spawn sequence. See positionWaitTimeout's doc comment.
func waitForPosition(ctx context.Context, a positionProvider) error {
	deadline := time.Now().Add(positionWaitTimeout)
	for {
		if _, ok := a.GetPositionSimple(); ok {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("rsi-infer: bot position not known after %s", positionWaitTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// positionProvider is the minimal capability waitForPosition needs,
// matching cmd/rsi-train's own positionProvider exactly.
type positionProvider interface {
	GetPositionSimple() (pos models.V3, initialized bool)
}

// connectAgent connects one live bot session from settings, mirroring
// cmd/rsi-train's own connectAgent minus the -parallel-envs-specific
// instanceSuffix/replay-path plumbing this single-bot command has no use
// for.
func connectAgent(ctx context.Context, settings mcconfig.Settings) (models.Agent, error) {
	conn := settings.Connection

	auth, err := agent.ResolveAuth(conn.Offline, conn.Name, conn.UUID, conn.Token, settings.Auth)
	if err != nil {
		return nil, err
	}
	if conn.Offline {
		log.Printf("offline mode: name=%s uuid=%s", auth.Name, auth.UUID)
	} else {
		log.Printf("authenticated as %s (%s)", auth.Name, auth.UUID)
	}

	version := conn.Version
	if version == "" {
		detectedVersion, _, err := rofutils.CheckServerVersion(conn.Address, 0)
		if err != nil {
			return nil, fmt.Errorf("auto-detect version from %s: %w", conn.Address, err)
		}
		version = detectedVersion
	}

	rcon, err := agent.DialRCON(ctx, settings.RCON.Address, settings.RCON.Password)
	if err != nil {
		return nil, err
	}
	if rcon != nil {
		log.Printf("connected to RCON at %s", settings.RCON.Address)
	}

	logLevel, err := utils.ParseLevel(settings.Logging.Level)
	if err != nil {
		return nil, err
	}

	cfg := models.AgentConfig{
		Name:             auth.Name,
		Address:          conn.Address,
		Version:          version,
		Auth:             auth,
		MCDataGenPath:    conn.MCDataGenPath,
		MCProtocolGoPath: conn.MCProtocolGoPath,
		StopFilePath:     ".agentStop-infer",
		LogLevel:         logLevel,
		RCON:             rcon,
	}

	a, err := agent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating agent: %w", err)
	}
	if err := a.Init(ctx); err != nil {
		return nil, fmt.Errorf("init: %w", err)
	}
	if err := a.Start(ctx); err != nil {
		return nil, fmt.Errorf("start: %w", err)
	}
	return a, nil
}
