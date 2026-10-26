// Command rsi-pvp-compete is the entrypoint for mc-agent's
// COMBAT_COMPETITION_PLAN.md Phase 2 ("competition mode") - see
// mc-rsi-trainer's docs/plans/20-combat-pvp-competition.md for the full
// design. It parses a pkg/tournament config and runs the bracket to
// completion; kept as its own binary for the same reason
// cmd/rsi-pvp-train is (docs/plans/19-combat-pvp-training.md).
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
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/tournament"
)

func main() {
	configPath := flag.String("config", "", "path to a tournament config JSON file (see pkg/tournament.Config)")
	flag.Parse()

	if err := run(*configPath); err != nil {
		log.Fatalf("rsi-pvp-compete: %v", err)
	}
}

func run(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required")
	}

	cfg, err := tournament.LoadFile(configPath)
	if err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log.Printf("rsi-pvp-compete: starting run %q with %d competitors", cfg.RunID, len(cfg.Competitors))
	bracket, err := tournament.Run(ctx, cfg)
	if err != nil {
		return err
	}
	if champion, ok := bracket.Champion(); ok {
		log.Printf("rsi-pvp-compete: run %q finished - champion: %s", cfg.RunID, champion)
	} else {
		log.Printf("rsi-pvp-compete: run %q stopped before a champion was decided", cfg.RunID)
	}
	return nil
}
