// Command onnx-infer connects one live mc-agent bot session and drives
// it with a model directory produced by cmd/onnx-export (model.onnx +
// manifest.json), instead of a native cRL-go checkpoint. It is
// cmd/rsi-infer's ONNX-backed counterpart: same connect/episode-loop
// shape, but it imports no cRL-go training package (no pkg/actorcritic,
// pkg/ppo, pkg/reinforce, pkg/lineage) — only pkg/onnx's generated IR
// types and pkg/onnxinfer's interpreter, via cRL-go/pkg/onnxexport for
// the graph's tensor-name constants. See
// docs/plans/23-ONNX-support.md Phase 4: this is the "inference binary"
// half of "a binary, a model file, and a config file usable
// independently of the source or trainer."
//
// Action masking and action sampling (reinforce.SampleMaskedAction in
// the native path) are reimplemented locally in decision.go rather than
// imported, for the same reason.
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

	"github.com/reallyoldfogie/cRL-go/pkg/onnxexport"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxinfer"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	"github.com/reallyoldfogie/mc-agent/actions"
	"github.com/reallyoldfogie/mc-agent/agent"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-agent/utils"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/onnxmanifest"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

type flags struct {
	mcAgentConfigPath string
	modelDir          string
	episodes          int
	episodeLen        int
	greedy            bool
}

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("onnx-infer", flag.ContinueOnError)
	fs.StringVar(&f.mcAgentConfigPath, "mc-agent-config", "", "path to mc-agent's config.Settings JSON (connection, RCON, env task) — same file the run that produced this model used")
	fs.StringVar(&f.modelDir, "model-dir", "", "directory containing model.onnx and manifest.json, as written by cmd/onnx-export")
	fs.IntVar(&f.episodes, "episodes", 0, "number of episodes to run, then exit; 0 runs until interrupted (Ctrl+C)")
	fs.IntVar(&f.episodeLen, "episode-len", 200, "max steps per episode, matching rsi-train's -eval-episode-len default")
	fs.BoolVar(&f.greedy, "greedy", true, "act on the policy's highest-probability action every step (matches rsi-train's own evaluation behavior); false samples from the policy's distribution instead")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.mcAgentConfigPath == "" {
		return flags{}, fmt.Errorf("onnx-infer: -mc-agent-config is required")
	}
	if f.modelDir == "" {
		return flags{}, fmt.Errorf("onnx-infer: -model-dir is required")
	}
	if f.episodes < 0 {
		return flags{}, fmt.Errorf("onnx-infer: -episodes must not be negative, got %d", f.episodes)
	}
	if f.episodeLen <= 0 {
		return flags{}, fmt.Errorf("onnx-infer: -episode-len must be positive, got %d", f.episodeLen)
	}
	return f, nil
}

func run(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}

	manifest, onnxData, err := onnxmanifest.Load(f.modelDir)
	if err != nil {
		return fmt.Errorf("loading model directory %s: %w", f.modelDir, err)
	}
	model, err := onnxinfer.Load(onnxData)
	if err != nil {
		return fmt.Errorf("loading %s: %w", manifest.ONNXFile, err)
	}
	log.Printf("loaded model from %s (generation=%d, environment=%q, recurrent=%v)", f.modelDir, manifest.Generation, manifest.EnvironmentID, manifest.Recurrent)

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
		return fmt.Errorf("onnx-infer: agent does not satisfy rlenv.LiveAgent (missing InventoryCount/Craftable/BlockNameAt/HealthProvider?)")
	}

	if err := waitForPosition(ctx, a); err != nil {
		return err
	}

	envCfg := mcSettings.Env.ToRlenvConfig()
	envCfg.NormalizeObservation = manifest.NormalizeObservation
	env, err := rlenv.New(liveAgent, actions.NewRegistry(), envCfg)
	if err != nil {
		return fmt.Errorf("constructing environment: %w", err)
	}

	liveEnvironmentID := fmt.Sprintf("mc-agent-rlenv:actions=%d:obs=%d", env.ActionSpace(), env.ObservationSize())
	if manifest.NormalizeObservation {
		liveEnvironmentID += ":norm=1" // same suffix cmd/rsi-infer's own environmentID construction adds
	}
	if err := manifest.CheckEnvironment(liveEnvironmentID); err != nil {
		return fmt.Errorf("onnx-infer: %w", err)
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

		reward, err := runEpisode(ctx, env, model, manifest, f.episodeLen, f.greedy, rng)
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

// zeroState allocates a [1, width] all-zero recurrent-state tensor —
// manifest.Recurrent's episode-start state, mirroring
// actorcritic.RecurrentActor.Reset's zeroing.
func zeroState(width int) *onnxinfer.Tensor {
	return &onnxinfer.Tensor{Dims: []int64{1, int64(width)}, Data: make([]float32, width)}
}

// runEpisode runs one episode against model and returns its total
// reward, mirroring cmd/rsi-infer's own runEpisode: greedy picks the
// highest-probability allowed action every step; sampling draws from
// the (mask-renormalized) distribution instead — see decision.go for
// both, reimplemented locally rather than imported from
// cRL-go/pkg/reinforce.
func runEpisode(ctx context.Context, env rl.Environment, model *onnxinfer.Model, manifest onnxmanifest.Manifest, episodeLen int, greedy bool, rng *rand.Rand) (float32, error) {
	observation, err := env.Reset(ctx)
	if err != nil {
		return 0, fmt.Errorf("resetting environment: %w", err)
	}

	var policyH, policyC, valueH, valueC *onnxinfer.Tensor
	if manifest.Recurrent {
		policyH, policyC = zeroState(manifest.HiddenSize), zeroState(manifest.HiddenSize)
		valueH, valueC = zeroState(manifest.HiddenSize), zeroState(manifest.HiddenSize)
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

		inputs := map[string]*onnxinfer.Tensor{
			onnxexport.ObservationInputName: {Dims: []int64{1, int64(len(observation.Values))}, Data: observation.Values},
		}
		if manifest.Recurrent {
			inputs[onnxexport.PolicyHiddenStateInputName] = policyH
			inputs[onnxexport.PolicyCellStateInputName] = policyC
			inputs[onnxexport.ValueHiddenStateInputName] = valueH
			inputs[onnxexport.ValueCellStateInputName] = valueC
		}

		outs, err := model.Run(inputs)
		if err != nil {
			return totalReward, fmt.Errorf("running model: %w", err)
		}
		if manifest.Recurrent {
			policyH, policyC = outs[onnxexport.PolicyHiddenStateOutputName], outs[onnxexport.PolicyCellStateOutputName]
			valueH, valueC = outs[onnxexport.ValueHiddenStateOutputName], outs[onnxexport.ValueCellStateOutputName]
		}

		probs, err := maskedProbabilities(outs[onnxexport.PolicyOutputName].Data, mask)
		if err != nil {
			return totalReward, fmt.Errorf("deciding action: %w", err)
		}
		var action rl.Action
		if greedy {
			action = greedyAction(probs)
		} else {
			action = sampleAction(probs, rng)
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

// actionName renders action using rlenv's own exported action constants,
// falling back to its bare numeral if it's out of range — identical to
// cmd/rsi-infer's own actionName, duplicated here per this repo's
// existing per-command convention (see cmd/rsi-infer's connectAgent doc
// comment for the same pattern).
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
	case rlenv.ActionPlace:
		return "Place"
	default:
		return fmt.Sprintf("Action(%d)", action)
	}
}

// positionWaitTimeout and waitForPosition are identical to
// cmd/rsi-infer's own — see that command's doc comments for why.
const positionWaitTimeout = 30 * time.Second

func waitForPosition(ctx context.Context, a positionProvider) error {
	deadline := time.Now().Add(positionWaitTimeout)
	for {
		if _, ok := a.GetPositionSimple(); ok {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("onnx-infer: bot position not known after %s", positionWaitTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

type positionProvider interface {
	GetPositionSimple() (pos models.V3, initialized bool)
}

// connectAgent is identical to cmd/rsi-infer's own connectAgent (down to
// the StopFilePath, aside from its own name) — see that command's doc
// comment for why each cmd/ binary keeps its own copy rather than
// sharing one.
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
		Authenticated:    !conn.Offline,
		MCDataGenPath:    conn.MCDataGenPath,
		MCProtocolGoPath: conn.MCProtocolGoPath,
		StopFilePath:     ".agentStop-onnx-infer",
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
