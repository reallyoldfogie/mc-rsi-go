// Command rsi-pvp-train is the entrypoint for mc-agent's
// COMBAT_COMPETITION_PLAN.md Phase 1 ("training mode") - see
// mc-rsi-trainer's docs/plans/19-combat-pvp-training.md for the full
// design. It parses a pkg/pvp config and starts the orchestrator; kept
// as its own binary (rather than a third branch on cmd/rsi-train's own
// already-large main.go), matching how cmd/rsi-competition's eventual
// split for architecture competition is planned
// (docs/plans/18-training-competition-framework.md's "Next
// implementation steps").
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func main() {
	configPath := flag.String("config", "", "path to a pvp training config JSON file (see pkg/pvp.Config)")
	metricsAddr := flag.String("metrics-addr", "", "if set, serve run-scoped Prometheus metrics (COMBAT_COMPETITION_PLAN.md Phase 3 item 1) at /metrics on this address")
	flag.Parse()

	if err := run(*configPath, *metricsAddr); err != nil {
		log.Fatalf("rsi-pvp-train: %v", err)
	}
}

func run(configPath, metricsAddr string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required")
	}

	cfg, err := pvp.LoadFile(configPath)
	if err != nil {
		return err
	}

	var metrics *pvp.Metrics
	if metricsAddr != "" {
		metrics, err = pvp.NewMetrics(cfg.RunID)
		if err != nil {
			return err
		}
		startMetricsServer(metricsAddr, metrics)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log.Printf("rsi-pvp-train: starting run %q with %d match(es)", cfg.RunID, len(cfg.Matches))
	if err := pvp.RunWithMetrics(ctx, cfg, metrics); err != nil {
		return err
	}
	log.Printf("rsi-pvp-train: run %q finished", cfg.RunID)
	return nil
}

// startMetricsServer serves metrics' own run-scoped registry at /metrics,
// in the background. Errors (e.g. the port is already taken) are logged,
// not fatal - this server going down is an observability problem, not a
// reason to stop an otherwise-healthy training run (same convention as
// cmd/rsi-train's own startMetricsServerWithGatherer).
func startMetricsServer(addr string, metrics *pvp.Metrics) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(metrics.Registry, promhttp.HandlerOpts{}))
	go func() {
		log.Printf("rsi-pvp-train: serving /metrics on %s", addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Printf("rsi-pvp-train: metrics server on %s stopped: %v", addr, err)
		}
	}()
}
