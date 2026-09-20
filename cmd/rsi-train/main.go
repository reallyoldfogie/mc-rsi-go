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
	"flag"
	"fmt"
	"log"
	oldrand "math/rand"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
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
		log.Fatal(err)
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
	mcAgentConfigPath            string
	trainerConfigPath            string
	epochsPerGeneration          int
	evalEpisodes                 int
	evalEpisodeLen               int
	checkpointInterval           int
	maxRounds                    int
	autoResetOrigin              bool
	parallelEnvs                 int
	sharedServer                 bool
	sharedServerSeparationChunks int
	curriculumConfigPath         string
	metricsAddr                  string
}

func parseFlags(args []string) (flags, error) {
	fs := flag.NewFlagSet("rsi-train", flag.ContinueOnError)
	f := flags{}
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "directory to load the latest generation from (if any) and save new ones into (required)")
	fs.StringVar(&f.mcAgentConfigPath, "mc-agent-config", "", "path to mc-agent's own config.Settings JSON file — connection, RCON, the rlenv task, replay recording (required)")
	fs.StringVar(&f.trainerConfigPath, "trainer-config", "", "optional path to a cRL-go config.Settings JSON file (PPO hyperparameters); config.Default() applies if unset")
	fs.IntVar(&f.epochsPerGeneration, "epochs-per-generation", 50, "PPO epochs the Student trains for before facing evaluation against the Teacher")
	fs.IntVar(&f.evalEpisodes, "eval-episodes", 5, "episodes per side (Teacher, then Student) in each round's evaluation")
	fs.IntVar(&f.evalEpisodeLen, "eval-episode-len", 200, "max steps per evaluation episode")
	fs.IntVar(&f.checkpointInterval, "checkpoint-interval", 10, "save an interim (non-generation) checkpoint every N Student-training epochs; 0 disables")
	fs.IntVar(&f.maxRounds, "max-rounds", 0, "stop after this many leapfrog rounds; 0 means run until interrupted")
	fs.BoolVar(&f.autoResetOrigin, "auto-reset-origin", false, "if RCON is configured and -mc-agent-config didn't already set env.use_reset_origin, teleport back to the bot's actual spawn position every episode (via a captured Config.ResetOrigin) plus a small default Config.Jitter, instead of letting the goto task's target drift from wherever the previous episode ended — see run's own doc comment for why this exists. Opt-in: false preserves every existing config's behavior unchanged.")
	fs.IntVar(&f.parallelEnvs, "parallel-envs", 1, "number of concurrent live Minecraft environments to train against (see docs/plans/08-parallel-environments-and-scaling.md); each needs its own already-running server — addresses/RCON/bot usernames beyond the first are derived from -mc-agent-config via pkg/parallelenv.DeriveSettings. 1 (the default) preserves single-environment behavior exactly.")
	fs.BoolVar(&f.sharedServer, "shared-server", false, "run all -parallel-envs bots against ONE already-running server instead of one server each, each confined to its own working area (see -shared-server-separation-chunks). Requires RCON to be configured. The operator is responsible for launching that server with a reduced VIEW_DISTANCE/SIMULATION_DISTANCE (e.g. 4-6) small enough not to overlap adjacent bots' working areas.")
	fs.IntVar(&f.sharedServerSeparationChunks, "shared-server-separation-chunks", 16, "chunks between adjacent bots' working areas when -shared-server is set; ignored otherwise")
	fs.StringVar(&f.curriculumConfigPath, "curriculum-config", "", "optional path to a pkg/curriculum pool JSON file (target_offsets/mine_target_blocks/mine_search_radius/craft_target_items) — if set, each environment's Config.TaskSelector picks a fresh task+goal every episode (pkg/curriculum.UniformRandom via pkg/curriculum/rlenvadapter) instead of -mc-agent-config's single static env.target_offset/mine/craft settings for the whole run. Unset preserves that static-Config behavior exactly.")
	fs.StringVar(&f.metricsAddr, "metrics-addr", ":9400", "address to serve Prometheus metrics on (see monitoring/README.md); empty disables the metrics server entirely")
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
	if f.parallelEnvs <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -parallel-envs must be positive, got %d", f.parallelEnvs)
	}
	if f.sharedServer && f.sharedServerSeparationChunks <= 0 {
		return flags{}, fmt.Errorf("rsi-train: -shared-server-separation-chunks must be positive, got %d", f.sharedServerSeparationChunks)
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

	if f.metricsAddr != "" {
		startMetricsServer(f.metricsAddr)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	connected, err := connectEnvironments(ctx, mcSettings, f.parallelEnvs, f.autoResetOrigin, f.sharedServer, f.sharedServerSeparationChunks, curriculumGen)
	if err != nil {
		return err
	}
	defer func() {
		for _, c := range connected {
			if err := c.agent.Close(context.Background()); err != nil {
				log.Printf("close error: %v", err)
			}
		}
	}()
	// A live bot session can end on its own (e.g. the .agentStop file)
	// independent of this process's own signal handling — treat that the
	// same as SIGINT/SIGTERM, mirroring mc-agent's own cmd/rl-train. Any
	// one of the N sessions ending stops the whole run, matching how a
	// single-environment run already stops on its one session ending.
	for _, c := range connected {
		if done := c.agent.Done(); done != nil {
			go func(done <-chan struct{}) {
				select {
				case <-ctx.Done():
				case <-done:
					stop()
				}
			}(done)
		}
	}

	envs := make([]rl.Environment, len(connected))
	for i, c := range connected {
		envs[i] = c.env
	}

	// Self-versioning by construction, matching mc-agent's own
	// cmd/rl-train convention exactly: any observation-shape change (e.g.
	// docs/plans/06's goal-conditioning block, 14 -> 17) changes this
	// string automatically, so actorcritic.Load's existing EnvironmentID
	// check rejects a stale checkpoint with no separate version field or
	// flag needed here. Every environment shares the same rlenv.Config
	// (parallelenv.DeriveSettings only changes connection/RCON/username,
	// never Env), so envs[0]'s shape speaks for all of them.
	environmentID := fmt.Sprintf("mc-agent-rlenv:actions=%d:obs=%d", envs[0].ActionSpace(), envs[0].ObservationSize())

	teacherParams, rec, err := loadOrInitTeacher(f.checkpointDir, environmentID, envs[0].ObservationSize(), trainerSettings.HiddenSize, envs[0].ActionSpace())
	if err != nil {
		return err
	}
	log.Printf("starting from generation %d (provenance=%s)", rec.Generation, rec.Provenance)
	metricGeneration.Set(float64(rec.Generation))

	interimDir := filepath.Join(f.checkpointDir, "interim")
	cfg := leapfrog.Config{
		Trainer:             trainerSettings,
		EpochsPerGeneration: f.epochsPerGeneration,
		EvalEpisodes:        f.evalEpisodes,
		EvalEpisodeLen:      f.evalEpisodeLen,
		OnEpoch: func(stats ppo.EpochStats, params *actorcritic.Params) {
			log.Printf("  epoch %d: average return %.3f, samples %d", stats.Epoch, stats.AverageReturn, stats.SampleCount)
			metricCurrentEpoch.Set(float64(stats.Epoch))
			metricEpochAvgReturn.Set(float64(stats.AverageReturn))
			metricEpochsTotal.Inc()
			metricEpochSamplesTotal.Add(float64(stats.SampleCount))
			if f.checkpointInterval > 0 && (stats.Epoch+1)%f.checkpointInterval == 0 {
				if err := saveInterimCheckpoint(interimDir, params, environmentID, stats.Epoch); err != nil {
					log.Printf("  saving interim checkpoint: %v", err)
				}
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

		start := time.Now()
		result, err := leapfrog.Round(ctx, envs, teacherParams, cfg, rng)
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("round %d interrupted: %v", round, err)
				break
			}
			return fmt.Errorf("round %d: %w", round, err)
		}
		elapsed := time.Since(start).Round(time.Second)
		metricRoundDurationSeconds.Observe(elapsed.Seconds())
		metricTeacherReward.Set(float64(result.TeacherReward))
		metricStudentReward.Set(float64(result.StudentReward))

		if result.StudentWon {
			rec = lineage.Record{
				Generation:           rec.Generation + 1,
				ParentGeneration:     rec.Generation,
				Provenance:           lineage.ProvenanceLeapfrog,
				SeededFromGeneration: -1,
				CreatedAt:            time.Now(),
				EnvironmentID:        environmentID,
			}
			if err := lineage.Save(f.checkpointDir, result.StudentParams, rec); err != nil {
				return fmt.Errorf("round %d: saving generation %d: %w", round, rec.Generation, err)
			}
			teacherParams = result.StudentParams
			metricRoundsTotal.WithLabelValues("won").Inc()
			metricGeneration.Set(float64(rec.Generation))
		} else {
			metricRoundsTotal.WithLabelValues("lost").Inc()
		}

		log.Printf("round %d (%s): teacher=%.3f student=%.3f studentWon=%v generation=%d",
			round, elapsed, result.TeacherReward, result.StudentReward, result.StudentWon, rec.Generation)
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
// passable feet/head, the same check rlenv's own walkability gate uses)
// and not submerged in water. Standability alone doesn't catch the
// submerged case: water counts as "passable" (mc-agent's
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
	return !shapeMgr.IsWater(feetID) && !shapeMgr.IsWater(headID)
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
			log.Printf("-auto-reset-origin: spawn position (%.1f,%.1f,%.1f) looks unsuitable (unwalkable or submerged); leaving Config.ResetOrigin unset", pos.X, pos.Y, pos.Z)
			return
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
// keeps it testable without a live bot session.
func loadOrInitTeacher(checkpointDir, environmentID string, observationSize, hiddenSize, actionSpace int) (*actorcritic.Params, lineage.Record, error) {
	if latest, err := lineage.Latest(checkpointDir); err == nil {
		params, rec, err := lineage.Load(checkpointDir, latest.Generation, environmentID)
		if err != nil {
			return nil, lineage.Record{}, fmt.Errorf("loading latest generation %d: %w", latest.Generation, err)
		}
		return params, rec, nil
	}

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
	params := actorcritic.NewParams(rng, observationSize, hiddenSize, actionSpace)
	rec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        environmentID,
	}
	if err := lineage.Save(checkpointDir, params, rec); err != nil {
		return nil, lineage.Record{}, fmt.Errorf("saving fresh generation 0: %w", err)
	}
	return params, rec, nil
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
func saveInterimCheckpoint(dir string, params *actorcritic.Params, environmentID string, epoch int) error {
	return checkpoint.Save(dir, "interim", epoch, func(path string) error {
		return actorcritic.SaveFile(path, params, environmentID, checkpoint.Metadata{Epoch: epoch})
	})
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
	env   rl.Environment
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
func connectEnvironments(ctx context.Context, base mcconfig.Settings, n int, autoResetOrigin, sharedServer bool, separationChunks int, curriculumGen *curriculum.UniformRandom) ([]connectedEnvironment, error) {
	if sharedServer {
		if base.RCON.Address == "" {
			return nil, fmt.Errorf("rsi-train: -shared-server requires RCON to be configured in -mc-agent-config")
		}
		if err := applySharedServerGamerules(ctx, base.RCON.Address, base.RCON.Password); err != nil {
			return nil, fmt.Errorf("applying shared-server gamerules: %w", err)
		}
	}

	connected := make([]connectedEnvironment, 0, n)
	closeAll := func() {
		for _, c := range connected {
			if err := c.agent.Close(context.Background()); err != nil {
				log.Printf("close error: %v", err)
			}
		}
	}

	for i := 0; i < n; i++ {
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

		a, err := connectAgent(ctx, settings, suffix)
		if err != nil {
			closeAll()
			return nil, fmt.Errorf("connecting environment %d/%d: %w", i, n, err)
		}

		liveAgent, ok := a.(rlenv.LiveAgent)
		if !ok {
			_ = a.Close(context.Background())
			closeAll()
			return nil, fmt.Errorf("environment %d/%d: agent does not satisfy rlenv.LiveAgent (missing InventoryCount/Craftable/BlockNameAt/HealthProvider?)", i, n)
		}

		envCfg := settings.Env.ToRlenvConfig()
		if autoResetOrigin {
			applyAutoResetOrigin(&envCfg, a, settings.RCON.Address)
		}
		if sharedServer {
			if err := applySharedServerWorkingArea(&envCfg, a, i, separationChunks); err != nil {
				_ = a.Close(context.Background())
				closeAll()
				return nil, fmt.Errorf("environment %d/%d: %w", i, n, err)
			}
		}
		if curriculumGen != nil {
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
			taskRNG := rand.New(rand.NewPCG(seed, uint64(i)))
			base := rlenvadapter.TaskSelector(curriculumGen, taskRNG)
			envCfg.TaskSelector = func(episode int, rng *oldrand.Rand) rlenv.TaskOverride {
				override := base(episode, rng)
				recordTaskSelection(override)
				return override
			}
		}
		env, err := rlenv.New(liveAgent, actions.NewRegistry(), envCfg)
		if err != nil {
			_ = a.Close(context.Background())
			closeAll()
			return nil, fmt.Errorf("constructing environment %d/%d: %w", i, n, err)
		}

		connected = append(connected, connectedEnvironment{agent: a, env: env})
	}
	return connected, nil
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
func connectAgent(ctx context.Context, settings mcconfig.Settings, instanceSuffix string) (models.Agent, error) {
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
		return nil, fmt.Errorf("init: %w", err)
	}
	if err := a.Start(ctx); err != nil {
		return nil, fmt.Errorf("start: %w", err)
	}
	return a, nil
}
