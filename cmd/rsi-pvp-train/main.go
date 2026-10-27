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
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/liveview"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func main() {
	configPath := flag.String("config", "", "path to a pvp training config JSON file (see pkg/pvp.Config)")
	metricsAddr := flag.String("metrics-addr", "", "if set, serve run-scoped Prometheus metrics at /metrics, plus a live 3D agent view at /dashboard (COMBAT_COMPETITION_PLAN.md Phase 3), on this address - matches cmd/rsi-train's own -metrics-addr convention")
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
	var view *liveview.Registry
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if metricsAddr != "" {
		metrics, err = pvp.NewMetrics(cfg.RunID)
		if err != nil {
			return err
		}
		view = liveview.NewRegistry()
		go view.RunPoller(ctx)
		startMetricsServer(metricsAddr, metrics, view)
	}

	log.Printf("rsi-pvp-train: starting run %q with %d match(es)", cfg.RunID, len(cfg.Matches))
	if err := pvp.RunWithObservability(ctx, cfg, metrics, view); err != nil {
		return err
	}
	log.Printf("rsi-pvp-train: run %q finished", cfg.RunID)
	return nil
}

// startMetricsServer serves metrics' own run-scoped registry at /metrics,
// and - since view is always non-nil when this is called - a live 3D
// agent view at /dashboard (see pkg/liveview's own doc comment: training
// mode is two bots fighting live via self-play, arguably even more worth
// watching live than a tournament game since it's continuous, not
// game-bounded). All in the background. Errors (e.g. the port is already
// taken) are logged, not fatal - this server going down is an
// observability problem, not a reason to stop an otherwise-healthy
// training run (same convention as cmd/rsi-train's own
// startMetricsServerWithGatherer).
func startMetricsServer(addr string, metrics *pvp.Metrics, view *liveview.Registry) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(metrics.Registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("/dashboard", liveview.ServeDashboardPage)
	mux.Handle("/agents", view)
	mux.HandleFunc("/agents/world", view.WorldStreamHandler)
	mux.Handle("/icons/", liveview.ServeItemIcons())
	mux.Handle("/vendor/", liveview.ServeVendor())
	go func() {
		log.Printf("rsi-pvp-train: serving /metrics, /dashboard (live 3D agent view), /agents, /agents/world on %s", addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Printf("rsi-pvp-train: metrics server on %s stopped: %v", addr, err)
		}
	}()
}
