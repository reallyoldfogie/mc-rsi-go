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
	"os"
	"os/signal"
	"syscall"

	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func main() {
	configPath := flag.String("config", "", "path to a pvp training config JSON file (see pkg/pvp.Config)")
	flag.Parse()

	if err := run(*configPath); err != nil {
		log.Fatalf("rsi-pvp-train: %v", err)
	}
}

func run(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required")
	}

	cfg, err := pvp.LoadFile(configPath)
	if err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log.Printf("rsi-pvp-train: starting run %q with %d match(es)", cfg.RunID, len(cfg.Matches))
	if err := pvp.Run(ctx, cfg); err != nil {
		return err
	}
	log.Printf("rsi-pvp-train: run %q finished", cfg.RunID)
	return nil
}
