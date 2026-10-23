package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/reallyoldfogie/mc-agent/models"
	competitioncfg "github.com/reallyoldfogie/mc-rsi-trainer/pkg/competition"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum/rlenvadapter"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/sshtunnel"
)

// runCompetition is the first live entrypoint for the competition framework.
// It intentionally lives beside rsi-train for now so it can reuse the
// existing mc-agent connection, environment, and dashboard helpers without
// changing the normal single-run command. A dedicated cmd/rsi-competition
// wrapper can be split out once the shared session code is moved below cmd.
func runCompetition(f flags) error {
	cfg, err := competitioncfg.LoadFile(f.competitionConfigPath)
	if err != nil {
		return err
	}
	competitionMetrics, err := competitioncfg.NewMetrics(cfg.RunID)
	if err != nil {
		return fmt.Errorf("creating competition metrics: %w", err)
	}
	mcSettings, err := mcconfig.Load(cfg.MCAgentConfig)
	if err != nil {
		return fmt.Errorf("loading competition mc-agent config: %w", err)
	}
	pool, err := loadCurriculumPool(cfg.CurriculumConfig)
	if err != nil {
		return err
	}
	if err := validateCompetitionPool(pool); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		log.Printf("competition context canceled: err=%v", ctx.Err())
		competitionMetrics.RecordStopReason("context_canceled")
	}()
	var tunnel *sshtunnel.Tunnel
	if cfg.SSH != nil {
		tunnel, err = sshtunnel.Start(ctx, *cfg.SSH)
		if err != nil {
			return fmt.Errorf("starting competition SSH tunnel: %w", err)
		}
		defer tunnel.Close()
		mcSettings.Connection.Address = tunnel.MinecraftAddress()
		mcSettings.RCON.Address = tunnel.RCONAddress()
		log.Printf("competition SSH tunnel active: Minecraft=%s RCON=%s", mcSettings.Connection.Address, mcSettings.RCON.Address)
	}
	connected, _, err := connectEnvironments(ctx, mcSettings, len(cfg.Competitors), cfg.Server.AutoResetOrigin, true, cfg.Server.SeparationChunks, &pool, cfg.CheckpointRoot, episodeLimits{})
	if err != nil {
		return err
	}

	agents := newAgentRegistry()
	for i, c := range connected {
		competitor := cfg.Competitors[i]
		competitorID := competitor.ID
		username := c.username
		agents.addCompetition(c.username, c.agent, cfg.RunID, competitor.ID, competitor.DisplayName, competitioncfg.EnvironmentID(competitor))
		watchForUnexpectedExitWithPolicy(ctx, stop, c, false, func() {
			competitionMetrics.AgentExited(competitorID, username, "connection_lost")
		})
		connected[i].env = &instrumentedEnvironment{
			env: c.env, health: healthProviderForAgent(c.agent), task: "curriculum", taskForEpisode: c.taskForEpisode,
			goalForEpisode: c.goalForEpisode, targetOffsetForEpisode: c.targetOffsetForEpisode,
			registry: agents, index: i,
		}
		instrumented := connected[i].env.(*instrumentedEnvironment)
		connected[i].updateDashboardAgent = func(agent models.Agent) {
			agents.replaceAgent(i, agent)
		}
		connected[i].env = instrumented
	}

	plans := make([]competitioncfg.EvaluationPlan, len(cfg.Training.EvaluationSeeds))
	for i, seed := range cfg.Training.EvaluationSeeds {
		plans[i], err = competitioncfg.NewEvaluationPlan(pool, seed, cfg.Training.EvaluationEpisodes)
		if err != nil {
			return fmt.Errorf("building evaluation plan %d: %w", i, err)
		}
	}
	start := time.Now()
	runRoot, err := competitioncfg.PrepareRunRoot(cfg.CheckpointRoot, cfg.RunID, start)
	if err != nil {
		return err
	}
	if err := competitioncfg.PrepareArtifactDirs(runRoot, cfg.Competitors); err != nil {
		return err
	}
	// Keep the resolved config and manifest tied to the configured competition,
	// while placing this invocation's mutable artifacts in its own run folder.
	artifactCfg := cfg
	artifactCfg.CheckpointRoot = runRoot
	window, err := competitioncfg.NewTrainingWindow(start, cfg.Training.Duration)
	if err != nil {
		return err
	}
	manifest, err := competitioncfg.NewManifest(cfg, plans, window, start)
	if err != nil {
		return fmt.Errorf("building competition manifest: %w", err)
	}
	if err := manifest.WriteJSON(filepath.Join(runRoot, "manifest.json")); err != nil {
		return fmt.Errorf("writing competition manifest: %w", err)
	}
	if err := writeCompetitionJSON(filepath.Join(runRoot, "evaluation-plans.json"), plans); err != nil {
		return fmt.Errorf("writing evaluation plan: %w", err)
	}

	assignments := artifactCfg.Assignments()
	if f.metricsAddr != "" {
		gatherers := prometheus.Gatherers{prometheus.DefaultGatherer, competitionMetrics.Registry}
		startMetricsServerWithGatherer(f.metricsAddr, nil, nil, nil, agents, gatherers)
	}
	go agents.runPoller(ctx)
	sessions := make([]competitioncfg.Session, 0, len(connected))
	for i, competitor := range cfg.Competitors {
		settings, err := competitioncfg.ResolveTrainerSettings(cfg, competitor)
		if err != nil {
			return err
		}
		environmentID := fmt.Sprintf("%s:actions=%d:obs=%d", competitioncfg.EnvironmentID(competitor), connected[i].env.ActionSpace(), connected[i].env.ObservationSize())
		competitorMetrics := competitionMetrics.Competitor(competitor.ID)
		competitorMetrics.SetTrainingComplete(false)
		competitorMetrics.SetTrainingFailed(false)
		sessions = append(sessions, &liveCompetitionSession{
			identity: assignments[i].Identity, competitor: competitor, connected: connected[i],
			env: connected[i].env, settings: settings, checkpointDir: assignments[i].ArtifactDir,
			environmentID: environmentID, epochsPerGeneration: cfg.Training.EpochsPerGeneration,
			evalEpisodeLen: cfg.Training.EvaluationEpisodeLen, evalGreedy: cfg.Training.EvalGreedy,
			metrics:      competitorMetrics,
			instrumented: connected[i].env.(*instrumentedEnvironment),
		})
	}
	result, runErr := competitioncfg.Run(ctx, sessions, window, plans, cfg.Training.EvaluationTimeLimit())
	if err := writeCompetitionJSON(filepath.Join(runRoot, "results.json"), result); err != nil {
		if runErr == nil {
			runErr = fmt.Errorf("writing competition results: %w", err)
		} else {
			log.Printf("writing competition results after run failure: %v", err)
		}
	}
	log.Printf("competition results artifact: %s", filepath.Join(runRoot, "results.json"))
	fmt.Print(competitioncfg.FormatResultsTable(result))
	if runErr != nil && ctx.Err() == nil {
		return runErr
	}
	return nil
}

func validateCompetitionPool(pool curriculum.Pool) error {
	if err := rlenvadapter.ValidatePool(pool); err != nil {
		return fmt.Errorf("competition curriculum: %w", err)
	}
	return nil
}

func writeCompetitionJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
