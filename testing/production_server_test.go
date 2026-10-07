package testing

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
)

// TestEnsureProductionServerLive is this repo's codified answer to "how was
// the production training server actually created" - found live
// 2026-10-05 to have no answer anywhere: the server crashed (a 60-second
// single-tick watchdog stall, twice), its container and Docker-managed
// data volume were both removed in one operation while investigating, and
// there was no docker-compose file, no shell history, and no doc describing
// how to stand it back up. Reverse-engineered from this package's own
// EnsureServer/testenv.Manager (mustShortID(rconPassword)[:8] == "rsi-over"
// exactly matches the removed container's own name suffix,
// mc-rsi-trainer-mc-1-21-5-rsi-over - this is provably the same mechanism
// that created it, not a guess).
//
// Unlike TestRoundAgainstLiveRlenv (leapfrog_live_test.go), this test's
// whole point is the side effect: ensuring the real production server
// named in configPath's own connection/rcon settings is up, with the real
// production options applied (view/simulation distance, world type,
// memory, a persistent bind-mounted data directory instead of an
// anonymous volume - see WithDataDir's own doc comment for why that
// matters now). It asserts nothing about training itself.
//
// Run it to (re)provision the production server - MC_RSI_TRAINER_KEEP_SERVER
// is not optional here the way it is for an ordinary live test: leaving it
// unset tears the server down the instant this function returns, via
// EnsureServer's own documented behavior.
//
//	MC_RSI_TRAINER_LIVE_CONFIG=runs/multi-step-2026-09-28/mc-agent-config.json \
//	  MC_RSI_TRAINER_KEEP_SERVER=1 \
//	  go test ./testing/... -run TestEnsureProductionServerLive -v
func TestEnsureProductionServerLive(t *testing.T) {
	configPath := os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG")
	if configPath == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping production server provisioning (see this test's own doc comment)")
	}
	if os.Getenv(KeepServerEnvVar) == "" {
		t.Fatalf("%s must be set when running this test - otherwise the server it (re)provisions is torn down the instant this function returns", KeepServerEnvVar)
	}

	settings, err := mcconfig.Load(configPath)
	require.NoError(t, err)
	require.NotEmpty(t, settings.Connection.Address, "config must set connection.address")
	require.NotEmpty(t, settings.RCON.Password, "config must set rcon.password - an already-running server found by address alone can't be reused across runs otherwise (see EnsureServer's own doc comment)")

	dataDir, err := filepath.Abs(filepath.Join(filepath.Dir(configPath), "mc-server-data"))
	require.NoError(t, err)
	t.Logf("production server data directory: %s", dataDir)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := EnsureServer(ctx, &settings,
		WithViewDistance(4), // -shared-server's working-area separation depends on this staying small - see WithViewDistance's own doc comment.
		WithDataDir(dataDir),
		WithExtraEnv(map[string]string{
			"LEVEL_TYPE": "FLAT", // vanilla's default flat recipe (bedrock, 2x dirt, grass_block) - matches what every working area's ground was verified to be, live, 2026-10-04. (TYPE picks the server software, e.g. VANILLA/PAPER - not the world generator; that's LEVEL_TYPE.)
			"MEMORY":     "1G",   // matches the crash reports' own "-Xmx1G -Xms1G" from before this test existed.
		}),
	)
	require.NoError(t, err)
	t.Logf("production server ready at %s:%d (version %s)", server.Host, server.Port, server.Version)
}
