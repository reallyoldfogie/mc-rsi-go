// Command rsi-train runs mc-rsi-trainer's leapfrog training loop
// (docs/plans/03-single-task-leapfrog-evaluation-loop.md,
// docs/plans/04-training-entrypoint-and-observability.md): connect one
// live mc-agent bot session, repeatedly run leapfrog rounds against it,
// and persist winning generations via pkg/lineage.
//
// Configuration is split across two independently-owned formats, not
// unified into one of this command's own: -mc-agent-config points at
// mc-agent's own config.Settings JSON (connection, RCON, the rlenv task
// itself, replay recording, ...) — reused as-is, matching how
// mc-agent's own cmd/rl-train already does this, rather than this repo
// inventing a parallel connection-config format. -trainer-config, if
// given, points at cRL-go's own config.Settings JSON (PPO
// hyperparameters) — also reused as-is; config.Default() applies if
// omitted.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	oldrand "math/rand"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	"github.com/reallyoldfogie/mc-agent/actions"
	"github.com/reallyoldfogie/mc-agent/agent"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-agent/utils"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum/rlenvadapter"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Print(err)
		if errors.Is(err, errNetworkDead) {
			os.Exit(exitNetworkDead)
		}
		os.Exit(1)
	}
}

// flags holds this command's own parsed CLI flags. Kept as an unexported
// struct (not passed around as loose *string/*int flag.Value pointers)
// so parseFlags's validation and run's actual use of the values stay
// clearly separated — matches mc-agent/cmd/rl-train's overall shape,
// minus that command's config.RegisterXFlags layering, which is
// mc-agent's own package and not reusable from here.
type flags struct {
	checkpointDir                string
	packetLogs                   bool
	mcAgentConfigPath            string
	trainerConfigPath            string
	epochsPerGeneration          int
	evalEpisodes                 int
	evalEpisodeLen               int
	checkpointInterval           int
	maxRounds                    int
	maxStepTimeouts              int
	timePenalty                  float64
	evalGreedy                   bool
	normalizeObservation         bool
	winZScore                    float64
	minWinMargin                 float64
	clearAreaRadius              int
	cloneAreaFrom                string
	cloneAreaFromParsed          [3]float64
	cloneAreaRadius              int
	cloneAreaHeight              int
	cloneAreaDepth               int
	autoResetOrigin              bool
	parallelEnvs                 int
	sharedServer                 bool
	sharedServerSeparationChunks int
	curriculumConfigPath         string
	metricsAddr                  string
	hiddenLayers                 int
}

func parseFlags(args []string) (flags, error) {
	fs := flag.NewFlagSet("rsi-train", flag.ContinueOnError)
	f := flags{}
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "directory to load the latest generation from (if any) and save new ones into (required)")
	fs.BoolVar(&f.packetLogs, "packet-logs", false, "write each bot's packet/agent log to <checkpoint-dir>/packet-logs (roughly 15 GB a day with 7 bots; off by default)")
	fs.StringVar(&f.mcAgentConfigPath, "mc-agent-config", "", "path to mc-agent's own config.Settings JSON file — connection, RCON, the rlenv task, replay recording (required)")
	fs.StringVar(&f.trainerConfigPath, "trainer-config", "", "optional path to a cRL-go config.Settings JSON file (PPO hyperparameters); config.Default() applies if unset")
	fs.IntVar(&f.epochsPerGeneration, "epochs-per-generation", 50, "PPO epochs the Student trains for before facing evaluation against the Teacher")
	fs.IntVar(&f.evalEpisodes, "eval-episodes", 5, "episodes per side (Teacher, then Student) in each round's evaluation, on EACH environment: with -parallel-envs N the environments evaluate concurrently, so the sample per side is N times this at the wall-clock cost of one")
	fs.IntVar(&f.evalEpisodeLen, "eval-episode-len", 200, "max steps per evaluation episode")
	fs.IntVar(&f.checkpointInterval, "checkpoint-interval", 10, "save an interim (non-generation) checkpoint every N Student-training epochs; 0 disables")
	fs.IntVar(&f.maxRounds, "max-rounds", 0, "stop after this many leapfrog rounds; 0 means run until interrupted")
	fs.Float64Var(&f.winZScore, "win-z-score", 1.645, "a Student is promoted only if its mean evaluation advantage over the Teacher exceeds this many standard errors of the paired per-episode difference (1.645 = one-sided 95%). Guards against promoting a coin flip once tasks are saturated (11 of 20 rounds were 'won' with return flat); 0 restores the plain strictly-greater rule")
	fs.Float64Var(&f.minWinMargin, "min-win-margin", 0, "additionally require the Student's mean evaluation reward to beat the Teacher's by at least this much; 0 disables")
	fs.IntVar(&f.maxStepTimeouts, "max-step-timeouts", 5, "end an episode once this many dispatched steps in a row hit the step timeout without their action resolving - bounds a wedged bot that would otherwise run its whole step budget while every other environment waits on it at the epoch barrier (found live: ~17 minutes); 0 disables. Ignored for an environment whose -mc-agent-config already sets its own")
	fs.BoolVar(&f.normalizeObservation, "normalize-observation", false, "scale every observation feature to about [-1, 1] (rlenv.Config.NormalizeObservation). Raw blocks and degrees saturate the policy's first layer and, with a learning rate that isn't tiny, kill its ReLU units for good (found live). Changes what the policy sees, so it is part of the checkpoint's environment ID: a checkpoint trained one way is rejected by the other. Ignored for an environment whose -mc-agent-config already sets it")
	fs.BoolVar(&f.evalGreedy, "eval-greedy", false, "evaluate each side by its highest-probability action instead of sampling from its policy. Off by default: the trained policy is stochastic, and a greedy read of it can be a much worse policy (found live: goto 0.8 vs mine 0.19 at the target made greedy evaluation stand still for 200 steps at -40 while sampling mined within a few steps, so a fine Student looked collapsed round after round)")
	fs.Float64Var(&f.timePenalty, "time-penalty", 0, "reward subtracted every step (rlenv.Config.TimePenalty); 0 keeps rlenv's built-in 0.01, which is negligible next to the +10 success bonus - a policy that wastes ten steps loses 0.1, so nothing makes it efficient (found live: the Wait action took 38-70% of all steps and mine_far episodes averaged ~10.7 steps where ~3 suffice). Around 0.2 makes a wasted step cost about 2% of the bonus. Changes the reward scale, so returns before and after aren't comparable. Ignored for an environment whose -mc-agent-config already sets its own")
	fs.IntVar(&f.clearAreaRadius, "clear-area-radius", 0, "each Reset, fill with air the box this many blocks either side of the reset origin in X and Z, from the origin's own Y up 4 blocks - wipes seeded blocks and crafting tables that pile up into obstacles over a run. FLAT WORLDS ONLY (it would delete trees and hills); needs -auto-reset-origin and RCON; 0 (the default) disables. Keep (2r+1)^2*5 under 32768. Ignored if -clone-area-from is also set.")
	fs.StringVar(&f.cloneAreaFrom, "clone-area-from", "", "\"x,y,z\" of a pristine template region's own ground-top block (rlenv.Config.CloneAreaFrom) - each Reset clones this template (ground layers and the air above them) onto the reset origin's working area via RCON /clone, instead of -clear-area-radius's fill-air-above-a-computed-boundary: this also heals ground damage, and never derives the ground/air boundary from a live, possibly-imprecise bot position the way -clear-area-radius does (docs/plans/13). FLAT WORLDS ONLY, same as -clear-area-radius; needs -auto-reset-origin and RCON; empty (the default) disables. The template location must be one no working area (now or after a future -parallel-envs increase) will ever reach - e.g. for -shared-server, pkg/parallelenv.WorkingAreaOffset only ever moves along X, so anywhere on the Z axis is safe.")
	fs.IntVar(&f.cloneAreaRadius, "clone-area-radius", 0, "half-width/depth in blocks of both the template and every destination working area; 0 (the default) falls back to -clear-area-radius if that's set too, otherwise disables cloning even if -clone-area-from is set")
	fs.IntVar(&f.cloneAreaHeight, "clone-area-height", 0, "blocks above -clone-area-from's own Y to include (the air-clearing portion); 0 (the default) uses 4, same as -clear-area-radius's own default")
	fs.IntVar(&f.cloneAreaDepth, "clone-area-depth", 0, "blocks below -clone-area-from's own Y to include (the ground-healing portion -clear-area-radius never had); 0 (the default) uses 3")
	fs.BoolVar(&f.autoResetOrigin, "auto-reset-origin", false, "if RCON is configured and -mc-agent-config didn't already set env.use_reset_origin, teleport back to the bot's actual spawn position every episode (via a captured Config.ResetOrigin) plus a small default Config.Jitter, instead of letting the goto task's target drift from wherever the previous episode ended — see run's own doc comment for why this exists. Opt-in: false preserves every existing config's behavior unchanged.")
	fs.IntVar(&f.parallelEnvs, "parallel-envs", 1, "number of concurrent live Minecraft environments to train against (see docs/plans/08-parallel-environments-and-scaling.md); each needs its own already-running server — addresses/RCON/bot usernames beyond the first are derived from -mc-agent-config via pkg/parallelenv.DeriveSettings. 1 (the default) preserves single-environment behavior exactly.")
	fs.BoolVar(&f.sharedServer, "shared-server", false, "run all -parallel-envs bots against ONE already-running server instead of one server each, each confined to its own working area (see -shared-server-separation-chunks). Requires RCON to be configured. The operator is responsible for launching that server with a reduced VIEW_DISTANCE/SIMULATION_DISTANCE (e.g. 4-6) small enough not to overlap adjacent bots' working areas.")
	fs.IntVar(&f.sharedServerSeparationChunks, "shared-server-separation-chunks", 16, "chunks between adjacent bots' working areas when -shared-server is set; ignored otherwise")
	fs.StringVar(&f.curriculumConfigPath, "curriculum-config", "", "optional path to a pkg/curriculum pool JSON file (target_offsets/mine_target_blocks/mine_search_radius/craft_target_items) — if set, each environment's Config.TaskSelector picks a fresh task+goal every episode (pkg/curriculum.UniformRandom via pkg/curriculum/rlenvadapter) instead of -mc-agent-config's single static env.target_offset/mine/craft settings for the whole run. Unset preserves that static-Config behavior exactly.")
	fs.StringVar(&f.metricsAddr, "metrics-addr", ":9400", "address to serve Prometheus metrics on (see monitoring/README.md); empty disables the metrics server entirely")
	fs.IntVar(&f.hiddenLayers, "hidden-layers", 2, "number of hidden layers in a freshly-created generation-0 policy network (docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md Part A); only used when -checkpoint-dir has no existing generation to resume — an existing checkpoint's own depth always wins, whatever this flag says, including one produced by cmd/deepen-checkpoint with a depth other than 2. 2 is the default this repo has always used.")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}
	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("rsi-train: -checkpoint-dir is required")
	}
	if f.mcAgentConfigPath == "" {
		return flags{}, fmt.Errorf("rsi-train: -mc-agent-config is required")
	}
	if f.epochsPerGeneration <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -epochs-per-generation must be positive, got %d", f.epochsPerGeneration)
	}
	if f.evalEpisodes <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -eval-episodes must be positive, got %d", f.evalEpisodes)
	}
	if f.evalEpisodeLen <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -eval-episode-len must be positive, got %d", f.evalEpisodeLen)
	}
	if f.checkpointInterval < 0 {
		return flags{}, fmt.Errorf("rsi-train: -checkpoint-interval must not be negative, got %d", f.checkpointInterval)
	}
	if f.maxRounds < 0 {
		return flags{}, fmt.Errorf("rsi-train: -max-rounds must not be negative, got %d", f.maxRounds)
	}
	if f.winZScore < 0 {
		return flags{}, fmt.Errorf("rsi-train: -win-z-score must not be negative, got %v", f.winZScore)
	}
	if f.timePenalty < 0 {
		return flags{}, fmt.Errorf("rsi-train: -time-penalty must not be negative, got %v", f.timePenalty)
	}
	if f.minWinMargin < 0 {
		return flags{}, fmt.Errorf("rsi-train: -min-win-margin must not be negative, got %v", f.minWinMargin)
	}
	if f.maxStepTimeouts < 0 {
		return flags{}, fmt.Errorf("rsi-train: -max-step-timeouts must not be negative, got %d", f.maxStepTimeouts)
	}
	if f.clearAreaRadius < 0 || (2*f.clearAreaRadius+1)*(2*f.clearAreaRadius+1)*(clearAreaHeightBlocks+1) > 32768 {
		return flags{}, fmt.Errorf("rsi-train: -clear-area-radius must be between 0 and %d (vanilla's fill command caps at 32768 blocks), got %d", maxClearAreaRadius, f.clearAreaRadius)
	}
	if f.cloneAreaFrom != "" {
		parsed, err := parseXYZ(f.cloneAreaFrom)
		if err != nil {
			return flags{}, fmt.Errorf("rsi-train: -clone-area-from: %w", err)
		}
		f.cloneAreaFromParsed = parsed
		r := f.cloneAreaRadius
		if r <= 0 {
			r = f.clearAreaRadius
		}
		height := f.cloneAreaHeight
		if height <= 0 {
			height = clearAreaHeightBlocks
		}
		depth := f.cloneAreaDepth
		if depth <= 0 {
			depth = defaultCloneAreaDepthBlocks
		}
		if r < 0 || (2*r+1)*(2*r+1)*(height+depth+1) > 32768 {
			return flags{}, fmt.Errorf("rsi-train: -clone-area-radius/-clone-area-height/-clone-area-depth combination is too large (vanilla's clone command caps at 32768 blocks): radius %d, height %d, depth %d", r, height, depth)
		}
	}
	if f.parallelEnvs <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -parallel-envs must be positive, got %d", f.parallelEnvs)
	}
	if f.sharedServer && f.sharedServerSeparationChunks <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -shared-server-separation-chunks must be positive, got %d", f.sharedServerSeparationChunks)
	}
	if f.hiddenLayers <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -hidden-layers must be positive, got %d", f.hiddenLayers)
	}
	return f, nil
}

func run(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}

	trainerSettings := crlconfig.Default()
	if f.trainerConfigPath != "" {
		trainerSettings, err = crlconfig.Load(f.trainerConfigPath)
		if err != nil {
			return fmt.Errorf("loading trainer config: %w", err)
		}
	}
	log.Printf("trainer settings: learning_rate=%v entropy_coef=%v max_grad_norm=%v hidden_size=%d", trainerSettings.LearningRate, trainerSettings.EntropyCoef, trainerSettings.MaxGradNorm, trainerSettings.HiddenSize)

	mcSettings, err := mcconfig.Load(f.mcAgentConfigPath)
	if err != nil {
		return fmt.Errorf("loading mc-agent config: %w", err)
	}

	var curriculumGen *curriculum.UniformRandom
	if f.curriculumConfigPath != "" {
		pool, err := loadCurriculumPool(f.curriculumConfigPath)
		if err != nil {
			return err
		}
		curriculumGen, err = curriculum.NewUniformRandom(pool)
		if err != nil {
			return fmt.Errorf("building curriculum generator from %s: %w", f.curriculumConfigPath, err)
		}
	}

	exporter := newCheckpointExporter(f.checkpointDir)
	envCountReq := newEnvCountRequest()
	restartReq := newRestartRequest()
	envConfirmer := newParallelEnvsConfirmer(f.checkpointDir)
	// Always constructed, even with -metrics-addr="" (nothing to serve it),
	// so instrumentedEnvironment never needs a nil check when recording
	// into it — see that type's own doc comment.
	agents := newAgentRegistry()
	if f.metricsAddr != "" {
		startMetricsServer(f.metricsAddr, exporter, envCountReq, restartReq, agents)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	// Safe to start before any bot is registered — pollOnce on zero bots
	// is a no-op (see agentRegistry.runPoller's own doc comment) — and
	// stops with the same ctx everything else in run() stops with.
	go agents.runPoller(ctx)

	// A live scale (POST /config/parallel-envs) only ever overrides this
	// startup count once it's survived parallelEnvsStabilityWindow without
	// the process exiting — see parallelEnvsConfirmer's own doc comment.
	// Anything else (never scaled, or the last scale crashed before
	// proving itself) falls straight through to -parallel-envs's own flag
	// value, unchanged from before this existed.
	initialParallelEnvs := f.parallelEnvs
	if persisted, ok := loadPersistedParallelEnvs(f.checkpointDir); ok {
		log.Printf("using persisted parallel-envs count %d (overriding -parallel-envs=%d)", persisted, f.parallelEnvs)
		initialParallelEnvs = persisted
	}

	connected, err := connectEnvironments(ctx, mcSettings, initialParallelEnvs, f.autoResetOrigin, f.sharedServer, f.sharedServerSeparationChunks, curriculumGen, f.packetLogDir(), f.episodeLimits())
	if err != nil {
		return err
	}
	defer func() {
		for _, c := range connected {
			closeConnectedAgent(c.agent)
		}
	}()
	// A live bot session can end on its own (e.g. the .agentStop file)
	// independent of this process's own signal handling — treat that the
	// same as SIGINT/SIGTERM, mirroring mc-agent's own cmd/rl-train. Any
	// one of the N sessions ending stops the whole run, matching how a
	// single-environment run already stops on its one session ending.
	for _, c := range connected {
		watchForUnexpectedExit(ctx, stop, c)
	}

	envs := make([]rl.Environment, len(connected))
	episodeTask := "curriculum"
	if curriculumGen == nil {
		episodeTask = taskName(mcSettings.Env)
	}
	for i, c := range connected {
		agents.add(c.username, c.agent)
		envs[i] = &instrumentedEnvironment{env: c.env, task: episodeTask, taskForEpisode: c.taskForEpisode, registry: agents, index: i}
	}
	metricParallelEnvs.Set(float64(len(connected)))

	// Self-versioning by construction, matching mc-agent's own
	// cmd/rl-train convention exactly: any observation-shape change (e.g.
	// docs/plans/06's goal-conditioning block, 14 -> 17) changes this
	// string automatically, so actorcritic.Load's existing EnvironmentID
	// check rejects a stale checkpoint with no separate version field or
	// flag needed here. Every environment shares the same rlenv.Config
	// (parallelenv.DeriveSettings only changes connection/RCON/username,
	// never Env), so envs[0]'s shape speaks for all of them.
	environmentID := fmt.Sprintf("mc-agent-rlenv:actions=%d:obs=%d", envs[0].ActionSpace(), envs[0].ObservationSize())
	if f.normalizeObservation {
		// Same shape, different meaning: a policy trained on raw features
		// would run on scaled ones without complaint, so the ID has to differ.
		environmentID += ":norm=1"
	}

	// RecurrentChunkLen > 0 is the sole trigger for recurrent (LSTM
	// step-memory) training - see config.Settings.RecurrentChunkLen's own
	// doc comment and runRecurrentRounds's. Every environment-connection
	// and metrics/exporter/envCountReq setup above this point is shared
	// unchanged by both paths; everything below it (teacher
	// loading/warm-starting, the round loop itself, checkpointing) is
	// *actorcritic.Params-specific, so it forks here rather than being
	// made generic in place - runRecurrentRounds mirrors it for
	// *actorcritic.RecurrentParams instead.
	if trainerSettings.RecurrentChunkLen > 0 {
		return runRecurrentRounds(ctx, stop, f, trainerSettings, connected, envs, environmentID, restartReq)
	}

	teacherParams, rec, err := loadOrInitTeacher(f.checkpointDir, environmentID, envs[0].ObservationSize(), trainerSettings.HiddenSize, envs[0].ActionSpace(), f.hiddenLayers)
	if err != nil {
		return err
	}
	exporter.setTeacher(teacherParams, rec)
	log.Printf("starting from generation %d (provenance=%s)", rec.Generation, rec.Provenance)
	startHealth := measureNetworkHealth(teacherParams)
	recordNetworkHealth(startHealth)
	log.Printf("policy network health: %s active, logit std %.3f", formatLayerActive(startHealth.LayerActive), startHealth.LogitStd)
	if startHealth.dead() {
		return fmt.Errorf("generation %d: %w; start from a fresh network instead (move the checkpoint directory aside)", rec.Generation, errNetworkDead)
	}
	metricGeneration.Set(float64(rec.Generation))
	metricEvalWinZThreshold.Set(f.winZScore)
	metricRewardTimePenalty.Set(effectiveTimePenalty(f.timePenalty))
	metricCurrentRound.Set(0)
	metricCurrentEpoch.Set(-1)
	metricEpochRolloutSize.Set(float64(trainerSettings.RolloutSize))
	metricEpochsPerGeneration.Set(float64(f.epochsPerGeneration))
	metricEpochTrajectoriesStarted.Set(0)
	progress := rec.Metadata
	metricBestReturnAllTime.Set(float64(progress.BestReturn))
	latestParams := teacherParams
	// madeEpochProgress becomes true the first time OnEpoch actually
	// fires. Guards the final-checkpoint save below against overwriting
	// a *better* existing checkpoint with a no-op: if this process
	// resumed from an interim checkpoint (loadResumableStudent below)
	// and then got interrupted again before completing even one more
	// epoch, latestParams never advances past its initial value
	// (teacherParams — the pristine generation, not even the resumed
	// checkpoint's own weights, since ResumeStudent/ResumeStudentStartEpoch
	// only take effect inside leapfrog.Round's own trainStudent, not
	// here), so saving it as "final" would destroy the genuinely
	// further-along checkpoint this process just resumed from. Found
	// live: exactly this sequence (resume, interrupt within the first
	// minute, save) overwrote a real epoch-17 checkpoint with the
	// original teacher's own weights mislabeled with the resumed
	// checkpoint's progress counters.
	madeEpochProgress := false

	// networkDead is set by OnEpoch once the student has looked dead for
	// deadEpochsBeforeAbort epochs in a row; the run then stops and returns
	// errNetworkDead. deadEpochs counts the current streak.
	networkDead := false
	deadEpochs := 0

	// lastEpochEnd feeds metricEpochDurationSeconds: reset to the round's
	// own start time at the top of the round loop below, then to time.Now()
	// after every OnEpoch firing, so each epoch's observed duration is the
	// wall-clock time since the previous epoch ended (or since the round
	// started, for that round's first epoch).
	var lastEpochEnd time.Time

	interimDir := filepath.Join(f.checkpointDir, "interim")
	cfg := leapfrog.Config[*actorcritic.Params]{
		Trainer:             trainerSettings,
		EpochsPerGeneration: f.epochsPerGeneration,
		EvalEpisodes:        f.evalEpisodes,
		EvalEpisodeLen:      f.evalEpisodeLen,
		WinZScore:           float32(f.winZScore),
		MinWinMargin:        float32(f.minWinMargin),
		EvalGreedy:          f.evalGreedy,
		OnBeforeEvaluation:  markEvaluationInProgress(pairedEvalTaskReseeder(evalTaskReseeders(connected))),
		OnEpoch: func(stats ppo.EpochStats, params *actorcritic.Params) {
			madeEpochProgress = true
			latestParams = params
			exporter.setStudent(params, environmentID, progress, rec.Generation)
			progress.Epoch = stats.Epoch
			progress.TotalUpdates += stats.UpdateCount
			if stats.AverageReturn > progress.BestReturn {
				progress.BestReturn = stats.AverageReturn
			}
			metricBestReturnAllTime.Set(float64(progress.BestReturn))
			log.Printf("  epoch %d: average return %.3f, samples %d", stats.Epoch, stats.AverageReturn, stats.SampleCount)
			health := measureNetworkHealth(params)
			recordNetworkHealth(health)
			if health.dead() {
				deadEpochs++
				log.Printf("  WARNING: policy network looks dead (%s active, logit std %.4f), %d of %d epochs before aborting", formatLayerActive(health.LayerActive), health.LogitStd, deadEpochs, deadEpochsBeforeAbort)
				if deadEpochs >= deadEpochsBeforeAbort && !networkDead {
					networkDead = true
					stop()
				}
			} else {
				deadEpochs = 0
			}
			metricCurrentEpoch.Set(float64(stats.Epoch))
			metricEpochTrajectoriesStarted.Set(0)
			metricEpochAvgReturn.Set(float64(stats.AverageReturn))
			metricEpochsTotal.Inc()
			metricEpochSamplesTotal.Add(float64(stats.SampleCount))
			metricGradientUpdatesTotal.Add(float64(stats.UpdateCount))
			metricEpochDurationSeconds.Observe(time.Since(lastEpochEnd).Seconds())
			lastEpochEnd = time.Now()
			if restartReq.take() {
				log.Printf("  restart requested; stopping after epoch %d", stats.Epoch)
				stop()
			}
			if f.checkpointInterval > 0 && (stats.Epoch+1)%f.checkpointInterval == 0 {
				if err := saveInterimCheckpoint(interimDir, params, environmentID, stats.Epoch, progress, rec.Generation); err != nil {
					log.Printf("  saving interim checkpoint: %v", err)
				}
				// Defensive: lineage.Save otherwise only ever runs at startup
				// (if nothing exists yet) and on a round win, so a checkpoint
				// directory wiped by mistake mid-round has no generation file
				// to resume from until the round happens to be won - any
				// restart before then (this trainer exits, and gets restarted,
				// whenever any one bot's connection drops) silently falls back
				// to a fresh random generation 0. Re-persisting the generation
				// this round is actually training against - unchanged, just
				// teacherParams again - on the same cadence as the interim
				// checkpoint recreates it well within one round, not only at
				// the next win. Found live: an accidental `rm` of the whole
				// checkpoint dir, mid-round, with the next restart's fate
				// resting entirely on the round finishing first.
				if err := lineage.Save(f.checkpointDir, teacherParams, rec, lineage.PlainCodec); err != nil {
					log.Printf("  re-saving generation %d checkpoint: %v", rec.Generation, err)
				}
			}
		},
	}

	if resumedParams, resumedEpoch, resumedMetadata := loadResumableStudent(interimDir, environmentID, rec.Generation); resumedParams != nil {
		cfg.ResumeStudent = resumedParams
		cfg.ResumeStudentStartEpoch = resumedEpoch + 1
		progress = resumedMetadata
		// progress was already set from rec.Metadata (the last *promoted*
		// generation's record) above, and metricBestReturnAllTime set from
		// that stale value - a round that's lost one or more times since its
		// own promotion keeps improving via the interim checkpoint without
		// ever updating rec.Metadata on disk (only a win does that), so
		// without this the gauge visibly regresses to the older, lower
		// number on every restart until training organically exceeds it
		// again. Found live 2026-10-03: restarted for a config-only change,
		// unrelated metric read 38.218 right after, though this same log
		// line's own best return (41.992) was already correct.
		metricBestReturnAllTime.Set(float64(progress.BestReturn))
		log.Printf("resuming round 1's student training from interim checkpoint: epoch %d, generation %d, best return %.3f, total updates %d",
			resumedEpoch, rec.Generation, resumedMetadata.BestReturn, resumedMetadata.TotalUpdates)
	}

	// Restores the "Teacher vs. Student reward per round" / "Student
	// advantage vs. promotion threshold" dashboard panels' gauges, which
	// otherwise sit at a misleading 0 from process start until whenever the
	// in-progress round next completes - possibly hours, since training
	// resumes mid-round rather than restarting it. See roundeval.go's own
	// doc comment.
	if state, ok := loadRoundEvalState(interimDir); ok {
		recordRoundEvaluation(leapfrog.Result[*actorcritic.Params]{
			TeacherReward: state.TeacherReward,
			StudentReward: state.StudentReward,
			EvalEpisodes:  state.EvalEpisodes,
			TeacherStd:    state.TeacherStd,
			StudentStd:    state.StudentStd,
			MeanDiff:      state.MeanDiff,
			DiffStdErr:    state.DiffStdErr,
		})
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))

	for round := 1; f.maxRounds == 0 || round <= f.maxRounds; round++ {
		if err := ctx.Err(); err != nil {
			log.Printf("stopping before round %d: %v", round, err)
			break
		}

		if n, ok := envCountReq.take(); ok {
			// Only ever applied here, between rounds: nothing about
			// live-adjusting connected/envs is safe to do while
			// leapfrog.Round is actively driving rollout collection
			// against them.
			newConnected, newEnvs, rcErr := reconcileEnvCount(ctx, stop, connected, envs, n,
				mcSettings, f.autoResetOrigin, f.sharedServer, f.sharedServerSeparationChunks,
				curriculumGen, f.packetLogDir(),
				f.episodeLimits(),
				episodeTask, agents)
			if rcErr != nil {
				log.Printf("round %d: scaling to %d parallel envs failed, continuing with %d: %v", round, n, len(connected), rcErr)
			} else {
				connected, envs = newConnected, newEnvs
				// Every environment's reseedEvalTasks feeds
				// cfg.OnBeforeEvaluation, built once from whatever
				// connected looked like at the time - a newly added
				// environment's reseeder isn't in that closure until this
				// rebuilds it, so without this, new environments' Teacher
				// and Student evaluation episodes wouldn't be paired to
				// the same task sequence (see pairedEvalTaskReseeder's own
				// doc comment for why that pairing matters).
				cfg.OnBeforeEvaluation = markEvaluationInProgress(pairedEvalTaskReseeder(evalTaskReseeders(connected)))
				metricParallelEnvs.Set(float64(len(connected)))
				envConfirmer.scheduleConfirm(len(connected))
				log.Printf("round %d: scaled to %d parallel envs", round, len(connected))
			}
		}

		metricCurrentRound.Set(float64(round))
		metricCurrentEpoch.Set(-1)
		metricEpochTrajectoriesStarted.Set(0)

		start := time.Now()
		lastEpochEnd = start
		result, err := leapfrog.Round(ctx, envs, teacherParams, cfg, leapfrog.PlainSeams, rng)
		// leapfrog.Round has no "evaluation ended" hook to clear this from
		// inside (only OnBeforeEvaluation, which sets it) - Round returning,
		// successfully or not, is the one point every path converges on.
		evaluationInProgress.Store(false)
		// cfg.ResumeStudent only ever applies to the very first round a
		// resumed process runs — every later round's Student is always a
		// fresh clone of *that* round's own teacherParams (which only
		// this loop, not leapfrog.Round, updates), so it must not carry
		// over. cfg is passed by value into Round, so clearing it here
		// (the caller's own copy) is required — Round has no way to do
		// this itself.
		cfg.ResumeStudent = nil
		cfg.ResumeStudentStartEpoch = 0
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("round %d interrupted: %v", round, err)
				break
			}
			return fmt.Errorf("round %d: %w", round, err)
		}
		elapsed := time.Since(start).Round(time.Second)
		metricRoundDurationSeconds.Observe(elapsed.Seconds())
		recordRoundEvaluation(result)
		if err := saveRoundEvalState(interimDir, result); err != nil {
			log.Printf("round %d: persisting round-eval state for dashboard restart-survival: %v", round, err)
		}

		if result.StudentWon {
			rec = lineage.Record{
				Generation:           rec.Generation + 1,
				ParentGeneration:     rec.Generation,
				Provenance:           lineage.ProvenanceLeapfrog,
				SeededFromGeneration: -1,
				CreatedAt:            time.Now(),
				EnvironmentID:        environmentID,
				Metadata:             progress,
			}
			if err := lineage.Save(f.checkpointDir, result.StudentParams, rec, lineage.PlainCodec); err != nil {
				return fmt.Errorf("round %d: saving generation %d: %w", round, rec.Generation, err)
			}
			teacherParams = result.StudentParams
			latestParams = teacherParams
			exporter.setTeacher(teacherParams, rec)
			metricRoundsTotal.WithLabelValues("won").Inc()
			metricGeneration.Set(float64(rec.Generation))
		} else {
			metricRoundsTotal.WithLabelValues("lost").Inc()
		}

		log.Printf("round %d (%s): teacher=%.3f student=%.3f studentWon=%v generation=%d (n=%d/side, std teacher=%.2f student=%.2f, paired diff=%+.3f ± %.3f)",
			round, elapsed, result.TeacherReward, result.StudentReward, result.StudentWon, rec.Generation,
			result.EvalEpisodes, result.TeacherStd, result.StudentStd, result.MeanDiff, result.DiffStdErr)

		// The round is over, so its recovery checkpoints describe a student
		// that is now either the new teacher (saved above) or discarded. Left
		// in place, a restart early in the next round would resume - or
		// re-save as "final" - that finished student under the same parent
		// generation (a lost round keeps it), wrongly starting the next round
		// from it, already past its epoch budget.
		clearInterimCheckpoints(interimDir)
		madeEpochProgress = false
	}
	if networkDead {
		// A dead student must not be resumed or saved as the latest progress.
		clearInterimCheckpoints(interimDir)
		return errNetworkDead
	}
	if !madeEpochProgress {
		log.Printf("no epoch completed this process's lifetime; leaving any existing recovery checkpoint in %s untouched", interimDir)
	} else if err := saveFinalCheckpoint(interimDir, latestParams, environmentID, progress, rec.Generation); err != nil {
		return fmt.Errorf("saving final recovery checkpoint: %w", err)
	} else {
		log.Printf("saved final recovery checkpoint: %s", filepath.Join(interimDir, "final.json"))
	}

	return nil
}

// defaultAutoJitter is applied by applyAutoResetOrigin when the operator
// didn't already configure a Config.Jitter of their own — small on
// purpose (a modest horizontal nudge, no vertical component so terrain
// height is left entirely to rlenv's own ground-snap/reachability check
// rather than double-perturbing it), just enough that consecutive
// episodes don't replay the exact same (origin, target) pair once
// ResetOrigin pins the origin itself — see rlenv.Config.Jitter's own doc
// comment for why that matters (a live-confirmed REINFORCE
// gradient-collapse bug, not a hypothetical concern).
var defaultAutoJitter = [3]float64{2, 0, 2}

// positionProvider is the minimal capability applyAutoResetOrigin needs —
// see its own doc comment for why this is narrower than models.Agent.
type positionProvider interface {
	GetPositionSimple() (pos models.V3, initialized bool)
}

// spawnQualityChecker is the optional capability applyAutoResetOrigin
// uses, if the connected agent has it, to reject an obviously bad
// captured spawn rather than blindly pinning every episode's
// Config.ResetOrigin to it for the rest of the run — see
// isGoodSpawnPosition's own doc comment for what "bad" means here.
// Optional and type-asserted (not folded into positionProvider) so a
// caller that only implements positionProvider — this file's own earlier
// unit tests included — keeps working unchanged; skipping the quality
// check when unavailable degrades gracefully, the same way
// applyAutoResetOrigin already degrades when position itself isn't known
// yet. The real connected agent (models.Agent, via WorldOperations)
// already satisfies this structurally.
type spawnQualityChecker interface {
	GetWorld() models.World
	BlockShapeManager() models.BlockShapeManager
}

// isGoodSpawnPosition reports whether pos is a reasonable place to reset
// to repeatedly: standable (models.IsWalkablePosition — solid ground,
// passable feet/head, the same check rlenv's own walkability gate uses),
// not submerged in water, and not boxed in (hasWalkableWayOut — see its
// own doc comment). Standability alone doesn't catch the submerged case:
// water counts as "passable" (mc-agent's
// models.BlockShapeManager treats it that way — a bot can occupy a water
// cell), so a bot standing on solid ground under a couple of blocks of
// water reads as perfectly walkable by that check alone, while actually
// being slow, current-affected, and generally a poor place to spend every
// single episode of an entire training run. Confirmed live: an
// auto-captured spawn once landed in what its own logs strongly
// suggested was a water/fish-heavy area (docs/plans/08-parallel-environments-and-scaling.md's
// own "Status"), and -auto-reset-origin pinning every episode's origin
// there for the rest of the run meant the whole run never got a chance to
// find a better spot — exactly what this check exists to catch before
// committing to it, not after.
func isGoodSpawnPosition(world models.World, shapeMgr models.BlockShapeManager, pos models.V3) bool {
	if !models.IsWalkablePosition(world, shapeMgr, pos) {
		return false
	}
	feetID, _ := world.GetBlockAt(pos.X, pos.Y, pos.Z)
	headID, _ := world.GetBlockAt(pos.X, pos.Y+1, pos.Z)
	if shapeMgr.IsWater(feetID) || shapeMgr.IsWater(headID) {
		return false
	}
	return hasWalkableWayOut(world, shapeMgr, pos)
}

// clearAreaHeightBlocks is how far above the reset origin's Y -clear-area-radius
// clears (rlenv.Config.ClearAreaHeight's default), and maxClearAreaRadius the
// largest radius keeping the fill box under vanilla's 32,768-block limit.
const (
	clearAreaHeightBlocks = 4
	maxClearAreaRadius    = 39 // (2*39+1)^2 * 5 = 31,205; radius 40 would be 32,805, over the limit
	// defaultCloneAreaDepthBlocks is -clone-area-depth's default (this
	// run's own soil depth - stop short of bedrock, which never needs
	// healing since it never changes), mirroring rlenv's own
	// defaultCloneAreaDepth so -clone-area-radius's size validation
	// matches what rlenv.Config actually ends up using.
	defaultCloneAreaDepthBlocks = 3
)

// parseXYZ parses "x,y,z" (as -clone-area-from takes it) into a [3]float64,
// rejecting anything that isn't exactly three comma-separated numbers.
func parseXYZ(s string) ([3]float64, error) {
	parts := strings.Split(s, ",")
	if len(parts) != 3 {
		return [3]float64{}, fmt.Errorf("want \"x,y,z\", got %q", s)
	}
	var v [3]float64
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return [3]float64{}, fmt.Errorf("want \"x,y,z\", got %q: %w", s, err)
		}
		v[i] = f
	}
	return v, nil
}

// episodeLimits are the per-episode safety nets applied to every
// environment; see the -max-step-timeouts and -clear-area-radius flags.
type episodeLimits struct {
	maxStepTimeouts int
	clearAreaRadius int
	timePenalty     float32

	normalizeObservation bool

	cloneAreaFrom   *[3]float64
	cloneAreaRadius int
	cloneAreaHeight int
	cloneAreaDepth  int
}

// episodeLimits builds the episodeLimits every connected environment shares
// from f, the one place -clone-area-from's already-validated
// cloneAreaFromParsed becomes the *[3]float64 episodeLimits/rlenv.Config
// actually want (nil when -clone-area-from was never set, rather than a
// pointer to a meaningless zero value).
func (f flags) episodeLimits() episodeLimits {
	limits := episodeLimits{
		maxStepTimeouts:      f.maxStepTimeouts,
		clearAreaRadius:      f.clearAreaRadius,
		timePenalty:          float32(f.timePenalty),
		normalizeObservation: f.normalizeObservation,
		cloneAreaRadius:      f.cloneAreaRadius,
		cloneAreaHeight:      f.cloneAreaHeight,
		cloneAreaDepth:       f.cloneAreaDepth,
	}
	if f.cloneAreaFrom != "" {
		from := f.cloneAreaFromParsed
		limits.cloneAreaFrom = &from
	}
	return limits
}

// applyEpisodeLimits sets cfg's episode safety nets from limits, leaving any
// value the operator already set in -mc-agent-config alone.
func applyEpisodeLimits(cfg *rlenv.Config, limits episodeLimits) {
	if limits.normalizeObservation {
		cfg.NormalizeObservation = true
	}
	if cfg.TimePenalty == 0 {
		cfg.TimePenalty = limits.timePenalty
	}
	if cfg.MaxConsecutiveStepTimeouts == 0 {
		cfg.MaxConsecutiveStepTimeouts = limits.maxStepTimeouts
	}
	if cfg.ClearAreaRadius == 0 && limits.clearAreaRadius > 0 {
		cfg.ClearAreaRadius = limits.clearAreaRadius
		cfg.ClearAreaHeight = clearAreaHeightBlocks
	}
	if cfg.CloneAreaFrom == nil && limits.cloneAreaFrom != nil {
		cfg.CloneAreaFrom = limits.cloneAreaFrom
		cfg.CloneAreaRadius = limits.cloneAreaRadius
		cfg.CloneAreaHeight = limits.cloneAreaHeight
		cfg.CloneAreaDepth = limits.cloneAreaDepth
	}
}

// maxSpawnLift bounds liftOutOfGround's upward search.
const maxSpawnLift = 8

// liftOutOfGround finds the first good spawn position directly above pos
// (same column, up to maxSpawnLift blocks) for a bot whose captured
// position is inside solid blocks - e.g. y=-63 in a superflat world whose
// surface is at y=-60. Found live: a bot's last-saved position from the
// previous run was buried like this, applyAutoResetOrigin correctly
// refused it, but applySharedServerWorkingArea then fell back to that same
// buried position, so every reset teleported the bot into the ground where
// it suffocated and died; each death respawned it thousands of blocks from
// its target, scoring a huge negative distance reward that wrecked the
// first epoch after every restart. Vertical only, on purpose: a spawn
// rejected for being submerged or boxed in must stay rejected, not be
// quietly moved next door.
func liftOutOfGround(world models.World, shapeMgr models.BlockShapeManager, pos models.V3) (models.V3, bool) {
	base := math.Floor(pos.Y)
	for dy := 1; dy <= maxSpawnLift; dy++ {
		candidate := models.V3{X: pos.X, Y: base + float64(dy), Z: pos.Z}
		if isGoodSpawnPosition(world, shapeMgr, candidate) {
			return candidate, true
		}
	}
	return models.V3{}, false
}

// spawnPathOutCheckRadius/spawnPathOutYTolerance/spawnPathOutMinOpenDirections
// bound hasWalkableWayOut's cheap, real-pathfinder-free heuristic against a
// spawn that's technically standable (and dry — isGoodSpawnPosition's own
// two earlier checks) but effectively boxed in: a narrow ledge, a
// one-block-wide pillar, the inside corner of an overhang. Found live: an
// -auto-reset-origin-captured spawn passed both of those checks yet still
// left its bot generating "Stuck recovery: no path found from current
// position" (mc-agent's agent.go, its physics executor's own recovery
// callback) roughly 1,700x the established baseline rate across one run,
// because -auto-reset-origin then pinned every single episode's origin to
// that one spot for the rest of the run — isGoodSpawnPosition existed
// specifically to screen out a bad one-time capture like this before
// committing an entire run to reusing it, but only checked the spawn point
// itself, never whether there was actually anywhere to go from it.
//
// spawnPathOutCheckRadius (3 blocks) and spawnPathOutMinOpenDirections (at
// least 2 of the 4 cardinal directions) are deliberately modest: this is a
// screen against a spawn point that's obviously enclosed, not a real
// reachability proof — rlenv's own per-episode walkability/reachability
// gate (Config.TargetOffset's own doc comment) already does the real work
// once an actual task target is known; this only runs once, at connect
// time, before any target exists. spawnPathOutYTolerance (1 block) lets
// each direction's search treat a single step up or down as still "open",
// since real terrain isn't flat and IsWalkablePosition itself has no such
// tolerance (interact_position.go checks pos.Y exactly).
const (
	spawnPathOutCheckRadius       = 3
	spawnPathOutYTolerance        = 1
	spawnPathOutMinOpenDirections = 2
)

// hasWalkableWayOut reports whether at least spawnPathOutMinOpenDirections
// of the 4 cardinal directions from pos have a walkable cell within
// spawnPathOutCheckRadius blocks — see that constant's own doc comment for
// why and how modest this check is.
func hasWalkableWayOut(world models.World, shapeMgr models.BlockShapeManager, pos models.V3) bool {
	directions := [4][2]float64{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	openDirections := 0
	for _, d := range directions {
		for step := 1.0; step <= spawnPathOutCheckRadius; step++ {
			open := false
			for dy := -spawnPathOutYTolerance; dy <= spawnPathOutYTolerance; dy++ {
				candidate := models.V3{X: pos.X + d[0]*step, Y: pos.Y + float64(dy), Z: pos.Z + d[1]*step}
				if models.IsWalkablePosition(world, shapeMgr, candidate) {
					open = true
					break
				}
			}
			if open {
				openDirections++
				break
			}
		}
	}
	return openDirections >= spawnPathOutMinOpenDirections
}

// applyAutoResetOrigin implements -auto-reset-origin: if cfg doesn't
// already have a ResetOrigin configured (an operator's own
// env.use_reset_origin/env.reset_origin in -mc-agent-config always wins,
// unchanged) and RCON is actually available (rconAddress != "" — without
// it, rlenv.Config.ResetOrigin can never work: Environment.Reset requires
// the agent to satisfy rlenv.ResetAgent, whose TeleportTo dials out over
// RCON), captures a's actual current position (a real, live-verified
// spawn point — not a coordinate guessed blind to whatever the current
// random world seed generated) and uses it as ResetOrigin, plus
// defaultAutoJitter if cfg.Jitter wasn't already set either.
//
// Exists because of a compounding failure mode confirmed live
// (docs/plans/08-parallel-environments-and-scaling.md's own "Status"):
// without ResetOrigin, Config.TargetOffset's own doc comment's "wherever
// Reset found the bot" workaround means the origin drifts to wherever the
// previous episode ended — so once an episode gets stuck near a
// standable-but-poorly-connected spot, every subsequent Reset recomputes
// almost the same target from almost the same stuck position, with
// nothing to break the streak. A real teleport back to a known-good spawn
// every episode removes that compounding effect entirely, independent of
// how good rlenv's own walkability/reachability checks are.
//
// A silent no-op (cfg left unchanged) if a's position isn't known yet
// (GetPositionSimple's ok is false) — startup degrades gracefully rather
// than blocking on it.
//
// Takes the narrow positionProvider rather than the full models.Agent
// (which the real connected agent already satisfies structurally, so the
// call site passes it unchanged) purely so this stays testable with a
// one-method fake, matching loadOrInitTeacher's own "keep it testable
// without a live bot session" precedent below.
func applyAutoResetOrigin(cfg *rlenv.Config, a positionProvider, rconAddress string) {
	if cfg.ResetOrigin != nil || rconAddress == "" {
		return
	}
	pos, ok := a.GetPositionSimple()
	if !ok {
		log.Printf("-auto-reset-origin: bot position not known yet; leaving Config.ResetOrigin unset")
		return
	}
	if checker, ok := a.(spawnQualityChecker); ok {
		world, shapeMgr := checker.GetWorld(), checker.BlockShapeManager()
		if world != nil && shapeMgr != nil && !isGoodSpawnPosition(world, shapeMgr, pos) {
			lifted, ok := liftOutOfGround(world, shapeMgr, pos)
			if !ok {
				log.Printf("-auto-reset-origin: spawn position (%.1f,%.1f,%.1f) looks unsuitable (unwalkable or submerged); leaving Config.ResetOrigin unset", pos.X, pos.Y, pos.Z)
				return
			}
			log.Printf("-auto-reset-origin: spawn position (%.1f,%.1f,%.1f) is buried in the ground; using (%.1f,%.1f,%.1f) instead", pos.X, pos.Y, pos.Z, lifted.X, lifted.Y, lifted.Z)
			pos = lifted
		}
	}
	origin := [3]float64{pos.X, pos.Y, pos.Z}
	cfg.ResetOrigin = &origin
	log.Printf("-auto-reset-origin: using spawn position (%.1f,%.1f,%.1f) as Config.ResetOrigin", pos.X, pos.Y, pos.Z)
	if cfg.Jitter == ([3]float64{}) {
		cfg.Jitter = defaultAutoJitter
		cfg.JitterSeed = time.Now().UnixNano()
	}
}

// applySharedServerWorkingArea implements -shared-server's per-bot
// working-area separation (docs/plans/08-parallel-environments-and-scaling.md):
// gives bot index its own working area, separationChunks chunks from a
// reference point, along X (see pkg/parallelenv.WorkingAreaOffset). The
// reference point is cfg.ResetOrigin if already set (by the operator's
// own config, or by a preceding applyAutoResetOrigin call for this same
// bot — both compose for free), otherwise a's own live position,
// captured the same way applyAutoResetOrigin does.
//
// Unlike applyAutoResetOrigin, this always sets cfg.ResetOrigin
// (overwriting one that's already there) and returns an error rather
// than silently leaving it unset: every bot MUST get its own distinct
// working area for -shared-server mode to mean anything — reusing a
// single shared reference point unmodified for every bot would be a
// spawn collision, not a graceful degradation.
func applySharedServerWorkingArea(cfg *rlenv.Config, a positionProvider, index, separationChunks int) error {
	base := [3]float64{}
	if cfg.ResetOrigin != nil {
		base = *cfg.ResetOrigin
	} else {
		pos, ok := a.GetPositionSimple()
		if !ok {
			return fmt.Errorf("shared-server: bot %d position not known yet", index)
		}
		base = [3]float64{pos.X, pos.Y, pos.Z}
	}

	offset := parallelenv.WorkingAreaOffset(index, separationChunks)
	origin := [3]float64{base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]}
	cfg.ResetOrigin = &origin
	log.Printf("shared-server: bot %d working area at (%.1f,%.1f,%.1f)", index, origin[0], origin[1], origin[2])
	return nil
}

// sharedServerGamerules is applied once, automatically, whenever
// -shared-server is set — reduces ambient chunk/entity churn from
// mechanics this training setup doesn't care about (wandering traders,
// fire spread, weather, random ticks), confirmed live via this
// project's own shared-server feasibility spike
// (testing/spike_shared_server_test.go). Fixed for v1, not
// configurable — a human isn't present to tune these on every real
// deployment, so a sensible baked-in default beats requiring a manual
// RCON step every run.
//
// doMobSpawning added after a live multi-hour run: passive-mob entity
// counts climbed into the dozens per bot's working area within minutes
// (nothing here previously stopped animal spawning specifically —
// doPatrolSpawning/doTraderSpawning/doInsomnia don't cover it), enough
// extra simulated entities to make bounded reachability searches
// (rlenv's own reachable(), 3s budget) occasionally time out under load
// rather than converge — a real contributor to a live-observed
// no-walkable-reachable-cell failure, not just a hypothetical concern.
var sharedServerGamerules = map[string]string{
	"doFireTick":       "false",
	"doTraderSpawning": "false",
	"doPatrolSpawning": "false",
	"doInsomnia":       "false",
	"doWeatherCycle":   "false",
	"doMobSpawning":    "false",
	"randomTickSpeed":  "0",
}

// applySharedServerGamerules dials its own RCON connection (separate
// from any of the N bots' own — this runs once, before any bot
// connects) and applies every rule in sharedServerGamerules.
func applySharedServerGamerules(ctx context.Context, rconAddress, rconPassword string) error {
	rcon, err := agent.DialRCON(ctx, rconAddress, rconPassword)
	if err != nil {
		return fmt.Errorf("dialing RCON for shared-server gamerules: %w", err)
	}
	defer func() { _ = rcon.Close() }()

	for rule, value := range sharedServerGamerules {
		if _, err := rcon.SetGamerule(ctx, rule, value).Exec(ctx); err != nil {
			return fmt.Errorf("setting gamerule %s=%s: %w", rule, value, err)
		}
	}
	return nil
}

// evalTaskReseeders collects every connected environment's reseedEvalTasks
// (nil entries - curriculum task selection off - are skipped).
func evalTaskReseeders(connected []connectedEnvironment) []func(seed1, seed2 uint64) {
	var reseeds []func(seed1, seed2 uint64)
	for _, c := range connected {
		if c.reseedEvalTasks != nil {
			reseeds = append(reseeds, c.reseedEvalTasks)
		}
	}
	return reseeds
}

// pairedEvalTaskReseeder returns a leapfrog.Config.OnBeforeEvaluation
// callback that pins Teacher's and Student's evaluation episodes to the
// identical sequence of curriculum-sampled tasks each round, on every
// environment: on EvalTeacher it draws a fresh seed pair per environment
// from its own internal rng and reseeds each environment with its pair; on
// EvalStudent it reseeds again with those same pairs, so each environment's
// env.Reset calls draw from an identical rng stream starting from the same
// state for both sides (environments get different pairs from each other,
// so together they cover different tasks). Without this, Round's win
// condition conflates "which random tasks each side happened to draw" with
// actual skill difference. No reseeders (curriculum task selection off)
// makes this return nil, so cfg.OnBeforeEvaluation stays nil and Round
// behaves exactly as it did before this existed.
func pairedEvalTaskReseeder(reseeds []func(seed1, seed2 uint64)) func(leapfrog.EvalSide) {
	if len(reseeds) == 0 {
		return nil
	}
	seedRNG := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0xE5A1))
	seeds := make([][2]uint64, len(reseeds))
	return func(side leapfrog.EvalSide) {
		if side == leapfrog.EvalTeacher {
			for i := range seeds {
				seeds[i] = [2]uint64{seedRNG.Uint64(), seedRNG.Uint64()}
			}
		}
		for i, reseed := range reseeds {
			reseed(seeds[i][0], seeds[i][1])
		}
	}
}

// loadOrInitTeacher resumes the latest saved generation from
// checkpointDir (lineage.Latest/Load), or, if none exists yet,
// initializes a fresh, randomly-weighted generation 0 and immediately
// persists it — so a second process start before any leapfrog round has
// ever won reloads that same fixed generation 0 rather than fabricating
// a new random one every time (which would silently discard whatever a
// previous run's not-yet-winning Student training had been working
// against). observationSize/actionSpace shape the fresh network when
// initializing; deliberately plain ints rather than an *rlenv.Environment
// parameter, since that's all this function actually needs from one —
// keeps it testable without a live bot session. numHiddenLayers (see
// -hidden-layers) likewise only applies when initializing fresh: an
// existing checkpoint's own depth always wins, whatever this says.
func loadOrInitTeacher(checkpointDir, environmentID string, observationSize, hiddenSize, actionSpace, numHiddenLayers int) (*actorcritic.Params, lineage.Record, error) {
	if latest, err := lineage.Latest(checkpointDir); err == nil {
		params, rec, err := lineage.Load(checkpointDir, latest.Generation, environmentID, lineage.PlainCodec)
		if err != nil {
			return nil, lineage.Record{}, fmt.Errorf("loading latest generation %d: %w", latest.Generation, err)
		}
		return params, rec, nil
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
	params := actorcritic.NewParams(rng, observationSize, hiddenSize, actionSpace, numHiddenLayers)
	rec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        environmentID,
	}
	if err := lineage.Save(checkpointDir, params, rec, lineage.PlainCodec); err != nil {
		return nil, lineage.Record{}, fmt.Errorf("saving fresh generation 0: %w", err)
	}
	return params, rec, nil
}

// loadOrInitTeacherRecurrent mirrors loadOrInitTeacher's own contract
// (see its doc comment) for a recurrent-configured run
// (config.Settings.RecurrentChunkLen > 0 - see runRecurrentRounds), with
// one added branch: an existing latest generation that is *not* already
// recurrent (lineage.Record.Recurrent false) is warm-started
// (actorcritic.WarmStartRecurrentParams) into a fresh recurrent
// generation instead of being resumed or re-initialized from scratch -
// see lineage.ProvenanceWarmStarted's own doc comment for why this is a
// warm start, not an exact conversion. The warm-started generation is
// saved as the source generation's number plus one, never overwriting
// the source generation in place - the same "never touch the source"
// convention cmd/widen-checkpoint/cmd/deepen-checkpoint use, just
// triggered automatically here (the sole trigger is
// config.Settings.RecurrentChunkLen > 0 - there is no separate opt-in
// flag for this) rather than via a separate manual tool. The warm-started
// generation starts with fresh progress counters (checkpoint.Metadata{}),
// matching that same widened/deepened convention: a fresh LSTM changes
// the network's behavior enough that counting epochs/updates/best-return
// from before the warm start would be misleading.
func loadOrInitTeacherRecurrent(checkpointDir, environmentID string, observationSize, hiddenSize, actionSpace, numHiddenLayers int) (*actorcritic.RecurrentParams, lineage.Record, error) {
	if latest, err := lineage.Latest(checkpointDir); err == nil {
		if latest.Recurrent {
			params, rec, err := lineage.Load(checkpointDir, latest.Generation, environmentID, lineage.RecurrentCodec)
			if err != nil {
				return nil, lineage.Record{}, fmt.Errorf("loading latest generation %d: %w", latest.Generation, err)
			}
			return params, rec, nil
		}

		plain, plainRec, err := lineage.Load(checkpointDir, latest.Generation, environmentID, lineage.PlainCodec)
		if err != nil {
			return nil, lineage.Record{}, fmt.Errorf("loading latest (non-recurrent) generation %d to warm-start: %w", latest.Generation, err)
		}
		rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
		warmStarted := actorcritic.WarmStartRecurrentParams(plain, rng)
		rec := lineage.Record{
			Generation:           plainRec.Generation + 1,
			ParentGeneration:     plainRec.Generation,
			Provenance:           lineage.ProvenanceWarmStarted,
			SeededFromGeneration: -1,
			CreatedAt:            time.Now(),
			EnvironmentID:        environmentID,
			Metadata:             checkpoint.Metadata{},
			Recurrent:            true,
		}
		if err := lineage.Save(checkpointDir, warmStarted, rec, lineage.RecurrentCodec); err != nil {
			return nil, lineage.Record{}, fmt.Errorf("saving warm-started generation %d: %w", rec.Generation, err)
		}
		log.Printf("warm-started generation %d (recurrent) from non-recurrent generation %d", rec.Generation, plainRec.Generation)
		return warmStarted, rec, nil
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
	params := actorcritic.NewRecurrentParams(rng, observationSize, hiddenSize, actionSpace, numHiddenLayers)
	rec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        environmentID,
		Recurrent:            true,
	}
	if err := lineage.Save(checkpointDir, params, rec, lineage.RecurrentCodec); err != nil {
		return nil, lineage.Record{}, fmt.Errorf("saving fresh generation 0: %w", err)
	}
	return params, rec, nil
}

// runRecurrentRounds is run()'s own round loop (see that function's doc
// comment, and leapfrog.Round), mirrored for a RecurrentChunkLen > 0
// trainer config: trains and evaluates *actorcritic.RecurrentParams via
// leapfrog.RecurrentSeams instead of *actorcritic.Params via
// leapfrog.PlainSeams. Entered automatically from run() once connected,
// envs and environmentID are already built — see run()'s own branch
// point for why everything up to there is shared unchanged by both
// paths.
//
// Deliberately narrower than run()'s own loop for now (see the
// warm-start plan this was built from, docs/plans - "Explicitly
// deferred"): no interim checkpoint save/resume (-checkpoint-interval is
// logged as ignored here, not silently broken), no manual checkpoint
// export endpoint (cmd/rsi-train/export.go's checkpointExporter is
// *actorcritic.Params-only), and no live -parallel-envs rescaling
// mid-run. Each is real, separate future work for the recurrent path,
// not an oversight - a restart of a recurrent run always resumes from
// the latest generation lineage.Save actually wrote (same as a plain run
// that hasn't completed even one epoch yet), never from a mid-round
// student.
func runRecurrentRounds(ctx context.Context, stop context.CancelFunc, f flags, trainerSettings crlconfig.Settings, connected []connectedEnvironment, envs []rl.Environment, environmentID string, restartReq *restartRequest) error {
	if f.checkpointInterval > 0 {
		log.Printf("recurrent mode: -checkpoint-interval is not yet supported; interim checkpoints will not be saved")
	}
	if f.metricsAddr != "" {
		log.Printf("recurrent mode: the /checkpoint/export endpoint is not yet supported and will always report not-ready")
	}
	log.Printf("recurrent mode: live -parallel-envs rescaling is not yet supported; running with %d environments for the whole run", len(envs))

	teacherParams, rec, err := loadOrInitTeacherRecurrent(f.checkpointDir, environmentID, envs[0].ObservationSize(), trainerSettings.HiddenSize, envs[0].ActionSpace(), f.hiddenLayers)
	if err != nil {
		return err
	}
	log.Printf("starting from generation %d (provenance=%s, recurrent)", rec.Generation, rec.Provenance)
	startHealth := measureNetworkHealth(teacherParams.Trunk)
	recordNetworkHealth(startHealth)
	log.Printf("policy network health: %s active, logit std %.3f", formatLayerActive(startHealth.LayerActive), startHealth.LogitStd)
	if startHealth.dead() {
		return fmt.Errorf("generation %d: %w; start from a fresh network instead (move the checkpoint directory aside)", rec.Generation, errNetworkDead)
	}
	metricGeneration.Set(float64(rec.Generation))
	metricEvalWinZThreshold.Set(f.winZScore)
	metricRewardTimePenalty.Set(effectiveTimePenalty(f.timePenalty))
	metricCurrentRound.Set(0)
	metricCurrentEpoch.Set(-1)
	metricEpochRolloutSize.Set(float64(trainerSettings.RolloutSize))
	metricEpochsPerGeneration.Set(float64(f.epochsPerGeneration))
	metricEpochTrajectoriesStarted.Set(0)
	progress := rec.Metadata
	metricBestReturnAllTime.Set(float64(progress.BestReturn))

	networkDead := false
	deadEpochs := 0

	// See run()'s own lastEpochEnd for what this feeds and why.
	var lastEpochEnd time.Time

	cfg := leapfrog.Config[*actorcritic.RecurrentParams]{
		Trainer:             trainerSettings,
		EpochsPerGeneration: f.epochsPerGeneration,
		EvalEpisodes:        f.evalEpisodes,
		EvalEpisodeLen:      f.evalEpisodeLen,
		WinZScore:           float32(f.winZScore),
		MinWinMargin:        float32(f.minWinMargin),
		EvalGreedy:          f.evalGreedy,
		OnBeforeEvaluation:  markEvaluationInProgress(pairedEvalTaskReseeder(evalTaskReseeders(connected))),
		OnEpoch: func(stats ppo.EpochStats, params *actorcritic.RecurrentParams) {
			progress.Epoch = stats.Epoch
			progress.TotalUpdates += stats.UpdateCount
			if stats.AverageReturn > progress.BestReturn {
				progress.BestReturn = stats.AverageReturn
			}
			metricBestReturnAllTime.Set(float64(progress.BestReturn))
			log.Printf("  epoch %d: average return %.3f, samples %d", stats.Epoch, stats.AverageReturn, stats.SampleCount)
			health := measureNetworkHealth(params.Trunk)
			recordNetworkHealth(health)
			if health.dead() {
				deadEpochs++
				log.Printf("  WARNING: policy network looks dead (%s active, logit std %.4f), %d of %d epochs before aborting", formatLayerActive(health.LayerActive), health.LogitStd, deadEpochs, deadEpochsBeforeAbort)
				if deadEpochs >= deadEpochsBeforeAbort && !networkDead {
					networkDead = true
					stop()
				}
			} else {
				deadEpochs = 0
			}
			metricCurrentEpoch.Set(float64(stats.Epoch))
			metricEpochTrajectoriesStarted.Set(0)
			metricEpochAvgReturn.Set(float64(stats.AverageReturn))
			metricEpochsTotal.Inc()
			metricEpochSamplesTotal.Add(float64(stats.SampleCount))
			metricGradientUpdatesTotal.Add(float64(stats.UpdateCount))
			metricEpochDurationSeconds.Observe(time.Since(lastEpochEnd).Seconds())
			lastEpochEnd = time.Now()
			if restartReq.take() {
				log.Printf("  restart requested; stopping after epoch %d", stats.Epoch)
				stop()
			}
		},
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))

	for round := 1; f.maxRounds == 0 || round <= f.maxRounds; round++ {
		if err := ctx.Err(); err != nil {
			log.Printf("stopping before round %d: %v", round, err)
			break
		}

		metricCurrentRound.Set(float64(round))
		metricCurrentEpoch.Set(-1)
		metricEpochTrajectoriesStarted.Set(0)

		start := time.Now()
		lastEpochEnd = start
		result, err := leapfrog.Round(ctx, envs, teacherParams, cfg, leapfrog.RecurrentSeams, rng)
		evaluationInProgress.Store(false)
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("round %d interrupted: %v", round, err)
				break
			}
			return fmt.Errorf("round %d: %w", round, err)
		}
		elapsed := time.Since(start).Round(time.Second)
		metricRoundDurationSeconds.Observe(elapsed.Seconds())
		recordRoundEvaluation(result)

		if result.StudentWon {
			rec = lineage.Record{
				Generation:           rec.Generation + 1,
				ParentGeneration:     rec.Generation,
				Provenance:           lineage.ProvenanceLeapfrog,
				SeededFromGeneration: -1,
				CreatedAt:            time.Now(),
				EnvironmentID:        environmentID,
				Metadata:             progress,
				Recurrent:            true,
			}
			if err := lineage.Save(f.checkpointDir, result.StudentParams, rec, lineage.RecurrentCodec); err != nil {
				return fmt.Errorf("round %d: saving generation %d: %w", round, rec.Generation, err)
			}
			teacherParams = result.StudentParams
			metricRoundsTotal.WithLabelValues("won").Inc()
			metricGeneration.Set(float64(rec.Generation))
		} else {
			metricRoundsTotal.WithLabelValues("lost").Inc()
		}

		log.Printf("round %d (%s): teacher=%.3f student=%.3f studentWon=%v generation=%d (n=%d/side, std teacher=%.2f student=%.2f, paired diff=%+.3f ± %.3f)",
			round, elapsed, result.TeacherReward, result.StudentReward, result.StudentWon, rec.Generation,
			result.EvalEpisodes, result.TeacherStd, result.StudentStd, result.MeanDiff, result.DiffStdErr)
	}
	if networkDead {
		return errNetworkDead
	}
	return nil
}

// saveInterimCheckpoint saves params to dir as a plain cRL-go checkpoint
// (pkg/checkpoint.Save/actorcritic.SaveFile), not a lineage.Save call —
// per docs/plans/04's own "Checkpoint interval" section, an interim
// checkpoint mid-Student-training isn't a lineage generation (it never
// beat the Teacher; it's not Student's final result for this round
// either), so it doesn't belong in pkg/lineage's generation sequence.
// This is a recovery aid, not yet wired into an automatic resume path:
// if this process is killed mid-round, restarting resumes from the last
// saved *generation* (loadOrInitTeacher), not from this interim epoch —
// see docs/plans/04's own note that a full mid-round resume capability is
// a further refinement, not required here.
func saveInterimCheckpoint(dir string, params *actorcritic.Params, environmentID string, epoch int, metadata checkpoint.Metadata, parentGeneration int) error {
	if err := saveResumeState(dir, parentGeneration); err != nil {
		return fmt.Errorf("saving resume state: %w", err)
	}
	return checkpoint.Save(dir, "interim", epoch, func(path string) error {
		return saveParamsAtomically(path, params, environmentID, metadata)
	})
}

func saveFinalCheckpoint(dir string, params *actorcritic.Params, environmentID string, metadata checkpoint.Metadata, parentGeneration int) error {
	if params == nil {
		return fmt.Errorf("params are nil")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := saveResumeState(dir, parentGeneration); err != nil {
		return fmt.Errorf("saving resume state: %w", err)
	}
	return saveParamsAtomically(filepath.Join(dir, "final.json"), params, environmentID, metadata)
}

// clearInterimCheckpoints removes a finished round's recovery files
// (interim-epoch-*.json, final.json and the resume-state sidecar) from dir.
// Best effort: a failure only means a stale checkpoint survives, which the
// caller logs but can't do more about.
func clearInterimCheckpoints(dir string) {
	var paths []string
	if matches, err := filepath.Glob(filepath.Join(dir, "interim-epoch-*.json")); err == nil {
		paths = append(paths, matches...)
	}
	paths = append(paths, filepath.Join(dir, "final.json"), filepath.Join(dir, resumeStateFileName))
	for _, path := range paths {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			log.Printf("clearing finished round's recovery checkpoint %s: %v", path, err)
		}
	}
}

// resumeStateFileName holds a small sidecar JSON — not one of cRL-go's
// own checkpoint.Metadata-tagged files — recording which teacher
// generation the interim-epoch-*.json/final.json checkpoints
// alongside it in the same directory were trained against. Owned
// entirely by this command (no cRL-go/mc-agent change needed): a
// resumable interim checkpoint is only meaningful together with the
// exact teacher it was cloned from, and loadResumableStudent uses this
// to refuse resuming a checkpoint left over from a since-superseded
// round (e.g. a stale interim save from a round whose Student went on
// to win and become a new generation in a later process — resuming
// that Student's mid-round weights as if they were still training
// against the old teacher would be training toward a already-obsolete
// comparison).
const resumeStateFileName = "resume-state.json"

type resumeState struct {
	ParentGeneration int `json:"parent_generation"`
}

func saveResumeState(dir string, parentGeneration int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(resumeState{ParentGeneration: parentGeneration})
	if err != nil {
		return err
	}
	path := filepath.Join(dir, resumeStateFileName)
	tmp, err := os.CreateTemp(dir, resumeStateFileName+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func loadResumeState(dir string) (resumeState, error) {
	data, err := os.ReadFile(filepath.Join(dir, resumeStateFileName))
	if err != nil {
		return resumeState{}, err
	}
	var state resumeState
	if err := json.Unmarshal(data, &state); err != nil {
		return resumeState{}, err
	}
	return state, nil
}

// loadResumableStudent looks for a resumable interim/final checkpoint
// in interimDir left over from an earlier, interrupted attempt at the
// same round this process is about to start — i.e. one whose recorded
// resumeState.ParentGeneration matches parentGeneration (see that
// type's own doc comment for why this check is required, not
// optional). Returns (nil, -1, checkpoint.Metadata{}) — a silent
// "nothing usable to resume, start this round fresh from a clone of
// the teacher" — for every case that isn't a confirmed match: no
// resume-state sidecar yet (a brand new checkpoint directory, or one
// from before this feature existed), a parent-generation mismatch, no
// interim/final checkpoint files at all, or an environmentID mismatch
// (a changed observation/action shape).
//
// Picks whichever of saveInterimCheckpoint's periodic
// interim-epoch-*.json files or saveFinalCheckpoint's fixed-name
// final.json has the newest file modification time — deliberately NOT
// the highest recorded Metadata.Epoch (checkpoint.Resume/Latest's own
// convention, and this function's own first version): epoch numbers
// reset to 0 on every fresh (non-resumed) round, so across many
// restarts of a process that's spent a long time on one still-unwon
// generation (this run's own history: generation 7 for over a day
// across ten+ separate launches), a *much older* attempt's
// higher-epoch leftover file can still be sitting in the same
// directory, satisfy the exact same parent-generation check, and look
// numerically "more advanced" than the file that actually matters -
// the most recently saved one. Found live: this resumed a real
// training run from a 24-hour-old, pre-every-tonight's-fix
// interim-epoch-000000049.json (parent generation 7, same as every
// other launch that day) instead of the previous night's much more
// relevant, much lower-numbered final.json. File mtime has no such
// ambiguity - it's always "when was this actually written," regardless
// of what epoch number happens to be embedded in the filename or
// Metadata.
func loadResumableStudent(interimDir, environmentID string, parentGeneration int) (*actorcritic.Params, int, checkpoint.Metadata) {
	state, err := loadResumeState(interimDir)
	if err != nil || state.ParentGeneration != parentGeneration {
		return nil, -1, checkpoint.Metadata{}
	}

	entries, err := os.ReadDir(interimDir)
	if err != nil {
		return nil, -1, checkpoint.Metadata{}
	}

	var newestName string
	var newestModTime time.Time
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || (name != "final.json" && !strings.HasPrefix(name, "interim-epoch-")) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if newestName == "" || info.ModTime().After(newestModTime) {
			newestName, newestModTime = name, info.ModTime()
		}
	}
	if newestName == "" {
		return nil, -1, checkpoint.Metadata{}
	}

	params, metadata, err := actorcritic.LoadFile(filepath.Join(interimDir, newestName), environmentID)
	if err != nil {
		return nil, -1, checkpoint.Metadata{}
	}
	return params, metadata.Epoch, metadata
}

func saveParamsAtomically(path string, params *actorcritic.Params, environmentID string, metadata checkpoint.Metadata) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	defer os.Remove(tmpPath)
	if err := actorcritic.SaveFile(tmpPath, params, environmentID, metadata); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func taskName(settings mcconfig.EnvSettings) string {
	switch {
	case settings.MineTargetBlock != "":
		return "mine"
	case settings.CraftTargetItem != "":
		return "craft"
	default:
		return "goto"
	}
}

func taskNameForOverride(override rlenv.TaskOverride) string {
	switch {
	case override.ChainStage == rlenv.ChainGather:
		return "chain_gather"
	case override.ChainStage == rlenv.ChainTable:
		return "chain_table"
	case override.ChainStage == rlenv.ChainPlace:
		return "chain_place"
	case override.ChainStage == rlenv.ChainUse:
		return "chain_use"
	case !override.GoToTargetDisabled && override.MineTargetBlock != "":
		return "mine_far"
	case !override.GoToTargetDisabled && override.CraftTargetItem != "":
		return "craft_far"
	case !override.GoToTargetDisabled:
		return "goto"
	case override.MineTargetBlock != "":
		return "mine"
	case override.CraftTargetItem != "":
		return "craft"
	default:
		return "unknown"
	}
}

// craftEconomyItems are the items behind every craft_target_item/
// chain_goal_item this run's curriculum can pose (see curriculum.json):
// minecraft:oak_log -> oak_planks -> stick/crafting_table/chest/bowl, with
// coal feeding torch. Logged as a group (not just the episode's own single
// target) because a bot's craft economy is shared across every
// craft-involving episode it draws, in curriculum-selection order, not
// scoped per task - a "stick" episode's leftover surplus is exactly what a
// later unrelated "chain_table" episode on the same bot depends on finding
// already in inventory, and a shortfall shows up as 0 across the board,
// not just on whichever item happened to be asked for.
var craftEconomyItems = []string{
	"minecraft:oak_log",
	"minecraft:oak_planks",
	"minecraft:stick",
	"minecraft:coal",
	"minecraft:crafting_table",
}

// logCraftEconomyInventory makes the "does this bot actually have what it
// needs" question answerable from train.log directly, instead of having to
// infer it after the fact from actions_total/episode_length_steps patterns
// (see the 2026-10-06 generation-1 round-1 collapse: craft-dependent tasks
// truncated at ~100% for ~20 epochs with zero step errors, and the only way
// to confirm "ran out of held ingredients, not an infra failure" was to
// reconstruct it indirectly from rsi_actions_total's wait/craft split and
// rsi_step_errors_total staying at zero - there was no direct record of
// what the bot was actually holding at the time).
//
// Called once per episode, only when the drawn task actually needs craft
// materials (plain craft/craft_far, or any chain stage - chain_gather's own
// "mine a tree" step feeds every later stage's inventory, so it counts
// too), right after TaskSelector draws it and before Reset acts on it.
// liveAgent.InventoryCount/Craftable read already-tracked client-side
// state synchronously - no RCON round-trip, safe to call from here.
func logCraftEconomyInventory(envName string, liveAgent rlenv.LiveAgent, selectedTask string, override rlenv.TaskOverride) {
	if override.CraftTargetItem == "" && override.ChainStage == 0 {
		return
	}
	counts := make([]string, len(craftEconomyItems))
	for i, item := range craftEconomyItems {
		counts[i] = fmt.Sprintf("%s=%d", item, liveAgent.InventoryCount(item))
	}
	readyNote := ""
	if override.CraftTargetItem != "" {
		readyNote = fmt.Sprintf(" craftReady(%s)=%v", override.CraftTargetItem, liveAgent.Craftable(override.CraftTargetItem))
	}
	log.Printf("[%s] episode task=%s craft economy: %s%s", envName, selectedTask, strings.Join(counts, " "), readyNote)
}

// craftIngredientSeeder is rlenv.Config.Seeder's implementation for
// curriculum-driven runs — without it (the state this run was in through
// 2026-10-06, see logCraftEconomyInventory's own doc comment), nothing
// ever gives a bot the ingredients a plain craft/craft_far episode's
// target item needs, or a chain episode's tree/table: every one of those
// task types is left depending entirely on whatever the bot happens to
// already be holding from earlier, unrelated episodes, with no
// guaranteed source. That's not merely "sometimes fails" — it's
// structurally unlearnable the moment the bot's held materials run out,
// however briefly, until it happens to pick more up.
//
// Setting Config.Seeder to this also activates environment.go's own
// seedFarTargets/seedChain path for every "go there, then X" and chain
// episode (gated on Config.Seeder != nil, not on this function doing
// anything itself for those - see Reset's own seeding block): a
// craft_far/mine_far/chain_* episode arrives here with
// cfg.CraftTargetItem/MineTargetBlock already blanked out by Reset
// (seedAtGoal's own convention, so this function doesn't double-seed
// beside the bot AND at the goal), so this only ever actually acts for a
// plain (non-far, non-chain) craft task.
//
// ClearInventory first, rather than giving on top of whatever's already
// held: agent.SeedCraftIngredients' own doc comment documents a real
// incident (testing/rl_train_test.go) where an uncleared give-every-
// episode policy filled all 36 inventory slots with accumulated surplus
// after ~200-230 successful craft episodes, after which every further
// craft failed its own inventory-increase confirmation indefinitely -
// clearing first keeps every plain craft episode self-contained instead
// of accumulating state across hundreds of them.
func craftIngredientSeeder(ctx context.Context, seedAgent rlenv.SeedAgent, cfg rlenv.Config) error {
	if cfg.CraftTargetItem == "" {
		return nil
	}
	if clearer, ok := seedAgent.(interface {
		ClearInventory(ctx context.Context) error
	}); ok {
		if err := clearer.ClearInventory(ctx); err != nil {
			return fmt.Errorf("clearing inventory before seeding craft ingredients: %w", err)
		}
	}
	return seedAgent.SeedCraftIngredients(ctx, cfg.CraftTargetItem)
}

// connectAgent establishes one live bot session from settings — auth
// resolution, version auto-detect, RCON dial, replay-recording wiring,
// agent.New/Init/Start — mirroring mc-agent's own cmd/rl-train/main.go
// connectAgent (not reusable directly: unexported, in mc-agent's own
// `main` package). Trimmed of packet-log-file rotation and skin
// fetching, neither of which this training entrypoint needs to function,
// to avoid pulling their extra dependencies (lumberjack, an HTTP skin
// cache) into this repo for a debugging convenience — add them back here
// if a real training run turns out to need packet-level debugging. If
// mc-agent's own connectAgent shape changes meaningfully, re-sync this
// copy against it.
// connectedEnvironment pairs one live rl.Environment with the
// models.Agent backing it, so run's shutdown path can Close every agent
// it opened regardless of how many -parallel-envs were requested.
type connectedEnvironment struct {
	agent models.Agent
	// username is this bot's connection name (settings.Connection.Name),
	// kept alongside agent purely for agentRegistry's dashboard identity —
	// nothing about training itself needs it off of settings.
	username       string
	env            rl.Environment
	taskForEpisode func() string
	// reseedEvalTasks, when curriculum task selection is active, reseeds
	// this environment's curriculum task rng in place (via
	// *rand.PCG.Seed, not swapping in a new source — the TaskSelector
	// closure captured a fixed *rand.Rand pointer, so reseeding the PCG
	// it wraps is what actually changes what the *next* Reset draws).
	// nil when curriculum task selection is off. Only connected[0]'s is
	// ever used, matching leapfrog.Round's own "evaluation only ever
	// touches envs[0]" — see run's leapfrog.Config.OnBeforeEvaluation
	// wiring for why this exists: pinning Teacher's and Student's eval
	// episodes to the identical task sequence each round.
	reseedEvalTasks func(seed1, seed2 uint64)
	// stopWatch is closed right before this environment's agent is closed
	// intentionally (reconcileEnvCount's scale-down path), telling its
	// watchForUnexpectedExit goroutine this wasn't an unexpected exit, so
	// it must return without stopping the whole run. Always non-nil.
	stopWatch chan struct{}
}

// packetLogDir is the directory the per-bot packet logs go in, or "" when
// -packet-logs is off. They were once always written and reached tens of GB
// per run, so they are opt-in.
func (f flags) packetLogDir() string {
	if !f.packetLogs {
		return ""
	}
	return f.checkpointDir
}

// connectEnvironments builds n live bot sessions and their
// rlenv.Environments from one base mc-agent config (see
// docs/plans/08-parallel-environments-and-scaling.md). n == 1 with
// sharedServer == false behaves exactly like the single-environment
// path this command had before -parallel-envs existed:
// parallelenv.DeriveSettings and connectAgent's own instanceSuffix are
// both no-ops in that case.
//
// n > 1 without sharedServer derives n distinct connection
// addresses/RCON/bot usernames via parallelenv.DeriveSettings (one
// server per bot). n > 1 with sharedServer instead derives only
// distinct bot usernames via parallelenv.DeriveSharedServerSettings —
// every bot connects to the exact same already-running server — and
// gives each bot its own working area via applySharedServerWorkingArea,
// after first applying sharedServerGamerules once (RCON is mandatory in
// that mode). Either mode also gets a distinct StopFilePath/ReplayOutput
// suffix per instance via connectAgent, avoiding mc-agent's documented
// same-process default-path collision without any mc-agent change (see
// connectAgent's own doc comment).
//
// On a mid-loop failure, every already-connected agent is closed before
// the error is returned, so a partial connect never leaks live bot
// sessions.
func connectEnvironments(ctx context.Context, base mcconfig.Settings, n int, autoResetOrigin, sharedServer bool, separationChunks int, curriculumGen *curriculum.UniformRandom, trainingDataDir string, limits episodeLimits) ([]connectedEnvironment, error) {
	if sharedServer {
		if base.RCON.Address == "" {
			return nil, fmt.Errorf("rsi-train: -shared-server requires RCON to be configured in -mc-agent-config")
		}
		if err := applySharedServerGamerules(ctx, base.RCON.Address, base.RCON.Password); err != nil {
			return nil, fmt.Errorf("applying shared-server gamerules: %w", err)
		}
	}
	return connectEnvironmentRange(ctx, base, 0, n, autoResetOrigin, sharedServer, separationChunks, curriculumGen, trainingDataDir, limits)
}

// connectEnvironmentRange connects environments at indices [start, n) out
// of n total, reusing connectEnvironments' own per-bot setup (minus the
// shared-server gamerules call, which only ever needs applying once and
// is this function's caller's job). Every one of a bot's derived
// settings — working-area offset (parallelenv.WorkingAreaOffset, a
// function of its own index only), username suffix, RCON port offset —
// depends only on its own index, never on n, so this is safe to call
// again later with a larger n to add more bots alongside ones an earlier,
// smaller-n call already connected and is actively using. See
// reconcileEnvCount, the live POST /config/parallel-envs path this
// exists for.
func connectEnvironmentRange(ctx context.Context, base mcconfig.Settings, start, n int, autoResetOrigin, sharedServer bool, separationChunks int, curriculumGen *curriculum.UniformRandom, trainingDataDir string, limits episodeLimits) ([]connectedEnvironment, error) {
	connected := make([]connectedEnvironment, 0, n-start)
	closeAll := func() {
		for _, c := range connected {
			closeConnectedAgent(c.agent)
		}
	}

	for i := start; i < n; i++ {
		var settings mcconfig.Settings
		if sharedServer {
			settings = parallelenv.DeriveSharedServerSettings(base, i, n)
		} else {
			settings = parallelenv.DeriveSettings(base, i, n)
		}
		suffix := ""
		if n > 1 {
			suffix = fmt.Sprintf("-env%d", i)
		}

		a, err := connectAgent(ctx, settings, suffix, trainingDataDir)
		if err != nil {
			closeAll()
			return nil, fmt.Errorf("connecting environment %d/%d: %w", i, n, err)
		}
		if err := opAgent(ctx, a, settings.Connection.Name); err != nil {
			closeConnectedAgent(a)
			closeAll()
			return nil, fmt.Errorf("granting operator status to environment %d/%d: %w", i, n, err)
		}

		liveAgent, ok := a.(rlenv.LiveAgent)
		if !ok {
			closeConnectedAgent(a)
			closeAll()
			return nil, fmt.Errorf("environment %d/%d: agent does not satisfy rlenv.LiveAgent (missing InventoryCount/Craftable/BlockNameAt/HealthProvider?)", i, n)
		}

		envCfg := settings.Env.ToRlenvConfig()
		applyEpisodeLimits(&envCfg, limits)
		if autoResetOrigin {
			applyAutoResetOrigin(&envCfg, a, settings.RCON.Address)
		}
		if sharedServer {
			if err := applySharedServerWorkingArea(&envCfg, a, i, separationChunks); err != nil {
				closeConnectedAgent(a)
				closeAll()
				return nil, fmt.Errorf("environment %d/%d: %w", i, n, err)
			}
		}
		if curriculumGen != nil {
			// See craftIngredientSeeder's own doc comment: this is also what
			// turns on mc-agent's own seedFarTargets/seedChain path for
			// every craft_far/mine_far/chain_* episode (gated on
			// Config.Seeder != nil there, not on this function doing
			// anything itself for those).
			envCfg.Seeder = craftIngredientSeeder
			var taskMu sync.RWMutex
			selectedTask := taskName(settings.Env)
			// Each environment gets its own rng, not a shared one: N
			// environments' Reset calls run concurrently (see
			// leapfrog.Round's rollout collection), and math/rand/v2's
			// Rand is not safe for concurrent use. curriculumGen itself
			// (a *curriculum.UniformRandom) holds no mutable state beyond
			// its read-only Pool, so sharing it across every env's own
			// rlenvadapter.TaskSelector closure is safe — only the rng
			// each closure supplies needs to be distinct.
			// time.Now().UnixNano()+i mirrors applyAutoResetOrigin's own
			// defaultAutoJitter seeding for the same reason: nanosecond
			// resolution across this loop's own real, if brief, elapsed
			// time already all-but-guarantees distinct seeds, and +i
			// removes any remaining doubt.
			seed := uint64(time.Now().UnixNano()) + uint64(i)
			taskPCG := rand.NewPCG(seed, uint64(i))
			taskRNG := rand.New(taskPCG)
			base := rlenvadapter.TaskSelector(curriculumGen, taskRNG)
			envCfg.TaskSelector = func(episode int, rng *oldrand.Rand) rlenv.TaskOverride {
				override := base(episode, rng)
				taskMu.Lock()
				selectedTask = taskNameForOverride(override)
				taskMu.Unlock()
				recordTaskSelection(override)
				logCraftEconomyInventory(settings.Connection.Name, liveAgent, selectedTask, override)
				return override
			}
			// rlenv calls TaskSelector during the wrapped environment's Reset;
			// the metrics wrapper reads this after Reset returns. The mutex also
			// keeps this safe if a caller ever overlaps operations on one env.
			taskForEpisode := func() string {
				taskMu.RLock()
				defer taskMu.RUnlock()
				return selectedTask
			}
			env, err := rlenv.New(liveAgent, actions.NewRegistry(), envCfg)
			if err != nil {
				closeConnectedAgent(a)
				closeAll()
				return nil, fmt.Errorf("constructing environment %d/%d: %w", i, n, err)
			}
			connected = append(connected, connectedEnvironment{
				agent:           a,
				username:        settings.Connection.Name,
				env:             env,
				taskForEpisode:  taskForEpisode,
				reseedEvalTasks: taskPCG.Seed,
				stopWatch:       make(chan struct{}),
			})
			continue
		}
		env, err := rlenv.New(liveAgent, actions.NewRegistry(), envCfg)
		if err != nil {
			closeConnectedAgent(a)
			closeAll()
			return nil, fmt.Errorf("constructing environment %d/%d: %w", i, n, err)
		}

		connected = append(connected, connectedEnvironment{agent: a, username: settings.Connection.Name, env: env, taskForEpisode: func() string { return taskName(settings.Env) }, stopWatch: make(chan struct{})})
	}
	return connected, nil
}

// reconcileEnvCount adjusts connected/envs to have exactly n entries,
// connecting n-len(connected) additional bot sessions (via
// connectEnvironmentRange) or closing the len(connected)-n highest-
// indexed ones, and returns the updated slices. Called only between
// rounds, in run's own round loop — never while leapfrog.Round is
// actively driving rollout collection against the current env set, since
// nothing about this is safe to do concurrently with that. A no-op,
// returning connected/envs unchanged, when n already equals
// len(connected).
//
// Closing an excess environment closes its stopWatch first (see that
// field's own doc comment) so watchForUnexpectedExit's watcher for it
// exits quietly instead of treating the close as an unexpected session
// end and stopping the whole run.
func reconcileEnvCount(ctx context.Context, stop context.CancelFunc, connected []connectedEnvironment, envs []rl.Environment, n int, base mcconfig.Settings, autoResetOrigin, sharedServer bool, separationChunks int, curriculumGen *curriculum.UniformRandom, trainingDataDir string, limits episodeLimits, episodeTask string, agents *agentRegistry) ([]connectedEnvironment, []rl.Environment, error) {
	current := len(connected)
	if n == current {
		return connected, envs, nil
	}
	if n < 1 {
		return connected, envs, fmt.Errorf("parallel envs must be at least 1, got %d", n)
	}

	if n < current {
		for i := n; i < current; i++ {
			close(connected[i].stopWatch)
			closeConnectedAgent(connected[i].agent)
		}
		agents.truncate(n)
		return connected[:n], envs[:n], nil
	}

	added, err := connectEnvironmentRange(ctx, base, current, n, autoResetOrigin, sharedServer, separationChunks, curriculumGen, trainingDataDir, limits)
	if err != nil {
		return connected, envs, fmt.Errorf("connecting %d additional environment(s): %w", n-current, err)
	}
	for _, c := range added {
		watchForUnexpectedExit(ctx, stop, c)
		index := len(connected)
		agents.add(c.username, c.agent)
		connected = append(connected, c)
		envs = append(envs, &instrumentedEnvironment{env: c.env, task: episodeTask, taskForEpisode: c.taskForEpisode, registry: agents, index: index})
	}
	return connected, envs, nil
}

// watchForUnexpectedExit starts the background watcher that treats c's
// live bot session ending on its own (e.g. the .agentStop file, a
// dropped connection) the same as SIGINT/SIGTERM — stopping the whole
// run — unless c.stopWatch is closed first, which marks the exit as
// intentional (see reconcileEnvCount's scale-down path, the only place
// that happens). Safe to call for every connectedEnvironment, including
// ones added well after startup.
func watchForUnexpectedExit(ctx context.Context, stop context.CancelFunc, c connectedEnvironment) {
	done := c.agent.Done()
	if done == nil {
		return
	}
	go func() {
		select {
		case <-ctx.Done():
		case <-c.stopWatch:
		case <-done:
			stop()
		}
	}()
}

// closeConnectedAgent closes both the live agent and the optional raw packet
// log writer installed by connectAgent. The latter is deliberately owned by
// the trainer so packet logs are flushed before the run exits.
func closeConnectedAgent(a models.Agent) {
	if a == nil {
		return
	}
	if err := a.Close(context.Background()); err != nil {
		log.Printf("close error: %v", err)
	}
	if closer, ok := a.GetPacketLogWriter().(io.Closer); ok {
		if err := closer.Close(); err != nil {
			log.Printf("packet log close error: %v", err)
		}
	}
}

// opAgent grants operator status before the environment can perform episode
// seeding through the player's own command connection. RCON Exec returns the
// server's response separately from transport errors; log that response for
// diagnostics while treating only transport failures as fatal.
func opAgent(ctx context.Context, a models.Agent, name string) error {
	rcon := a.Config().RCON
	if rcon == nil {
		return fmt.Errorf("RCON is not configured for %s", name)
	}
	response, err := rcon.Exec(ctx, fmt.Sprintf("op %s", name))
	if err != nil {
		return fmt.Errorf("op %s via RCON: %w", name, err)
	}
	log.Printf("RCON operator grant response for %s: %s", name, response)
	return nil
}

// connectAgent connects one live bot session from settings. instanceSuffix
// is appended to this session's StopFilePath (default ".agentStop") and,
// if the operator explicitly pinned a fixed ReplayOutput path in
// settings (reused as config across every -parallel-envs instance —
// see connectEnvironments), to that path too, before its extension —
// both otherwise-shared defaults that would collide across multiple
// agent instances in this same process (see
// docs/plans/08-parallel-environments-and-scaling.md's own research
// into mc-agent's concurrency surface). instanceSuffix == "" (used by
// every existing single-environment caller, i.e. -parallel-envs=1)
// leaves both paths byte-for-byte unchanged from before this parameter
// existed.
func connectAgent(ctx context.Context, settings mcconfig.Settings, instanceSuffix, trainingDataDir string) (models.Agent, error) {
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

	var packetLog io.WriteCloser
	keepPacketLog := false
	defer func() {
		if !keepPacketLog && packetLog != nil {
			_ = packetLog.Close()
		}
	}()
	if trainingDataDir != "" {
		packetDir := filepath.Join(trainingDataDir, "packet-logs")
		if err := os.MkdirAll(packetDir, 0o755); err != nil {
			return nil, fmt.Errorf("create packet log directory: %w", err)
		}
		packetPath := filepath.Join(packetDir, "agent_"+settings.Connection.Name+"_"+time.Now().Format("20060102_150405.000")+".jsonl")
		file, err := os.OpenFile(packetPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
		if err != nil {
			return nil, fmt.Errorf("create packet log %s: %w", packetPath, err)
		}
		packetLog = file
		log.Printf("packet logging enabled for %s: %s", settings.Connection.Name, packetPath)
	}

	logLevel, err := utils.ParseLevel(settings.Logging.Level)
	if err != nil {
		return nil, err
	}

	replay := settings.Replay
	if replay.Enable && replay.Output == "" {
		cacheDir, err := utils.FindOrCreateCacheDir()
		if err != nil {
			return nil, fmt.Errorf("find cache directory for replay output: %w", err)
		}
		replay.Output = filepath.Join(cacheDir, "replays", version, auth.Name+"_"+time.Now().Format("20060102_150405")+".mcpr")
	} else if replay.Enable && instanceSuffix != "" {
		// The operator explicitly pinned a fixed Output path, reused as
		// config across every -parallel-envs instance (see
		// connectEnvironments) — insert instanceSuffix before the
		// extension so N instances never write the same file. The
		// auto-derived branch above already includes auth.Name (which
		// DeriveUsername already makes distinct per instance), so it
		// needs no extra suffixing here.
		ext := filepath.Ext(replay.Output)
		replay.Output = strings.TrimSuffix(replay.Output, ext) + instanceSuffix + ext
	}
	if replay.Enable {
		log.Printf("replay recording enabled: %s", replay.Output)
	}

	cfg := models.AgentConfig{
		Name:             auth.Name,
		Address:          conn.Address,
		Version:          version,
		Auth:             auth,
		MCDataGenPath:    conn.MCDataGenPath,
		MCProtocolGoPath: conn.MCProtocolGoPath,
		StopFilePath:     ".agentStop" + instanceSuffix,
		LogLevel:         logLevel,
		LogWriter:        packetLog,
		RCON:             rcon,
		EnableReplay:     replay.Enable,
		ReplayOutput:     replay.Output,
		ReplayGenerator:  replay.Generator,
	}

	a, err := agent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating agent: %w", err)
	}
	if err := a.Init(ctx); err != nil {
		_ = a.Close(context.Background())
		return nil, fmt.Errorf("init: %w", err)
	}
	if err := a.Start(ctx); err != nil {
		_ = a.Close(context.Background())
		return nil, fmt.Errorf("start: %w", err)
	}
	keepPacketLog = true
	return a, nil
}
