package testing

import (
	"context"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEnsureServerLaunchesThenReusesTheSameServer is EnsureServer's own
// live regression test: launches a real Minecraft server via Docker
// (mcserver.go), confirms a second EnsureServer call against the same
// address finds and reuses it instead of trying to launch a second one
// on the now-occupied port, then tears it down. Needs Docker; slow (a
// real Minecraft server takes real time to start) — reuses this
// package's existing MC_RSI_TRAINER_LIVE_CONFIG opt-in gate as a shared
// "run slow Docker/network live tests" switch, even though this
// particular test doesn't read that file's contents (it builds its own
// throwaway Settings against a free local port, rather than needing a
// real mc-agent bot config) — see this package's own doc comment.
func TestEnsureServerLaunchesThenReusesTheSameServer(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping (see this test's own doc comment for why it uses that gate without reading the file)")
	}

	port := freeTCPPort(t)
	settings := mcconfig.Settings{
		Connection: mcconfig.ConnectionSettings{
			Address: fmt.Sprintf("127.0.0.1:%d", port),
			Offline: true,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := EnsureServer(ctx, &settings)
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = server.Close(context.Background())
	})

	assert.Equal(t, "127.0.0.1", server.Host)
	assert.Equal(t, port, server.Port)
	assert.NotEmpty(t, server.Version, "EnsureServer should report the version it launched")
	assert.NotEmpty(t, settings.RCON.Address, "EnsureServer should have filled in RCON connection info for the server it launched")
	assert.NotEmpty(t, settings.RCON.Password)
	t.Logf("launched %s (version %s), RCON at %s", settings.Connection.Address, server.Version, settings.RCON.Address)

	// A second call against the same, now-occupied address must find the
	// server just launched rather than attempting (and failing) to bind
	// a second container to the same host port.
	second, err := EnsureServer(ctx, &settings)
	require.NoError(t, err)
	assert.Equal(t, server.Host, second.Host)
	assert.Equal(t, server.Port, second.Port)
	// second never launched anything, so its own Close must be a
	// structural no-op — the real assertion this proves is that
	// EnsureServer's probe path, not its launch path, was taken the
	// second time (a launch-path Close would actually stop the
	// container out from under the first ManagedServer's own eventual
	// Close).
	assert.NoError(t, second.Close(context.Background()))
}

// freeTCPPort asks the OS for a currently-unused TCP port by binding to
// :0 and reading back what it picked, then immediately releasing it —
// the standard "ask the OS, don't guess" pattern for tests that need a
// real, currently-free port number rather than a hardcoded one that
// might collide with something already running on the test machine.
// Inherently has a small time-of-check-to-time-of-use race (something
// else could grab the port between this returning and EnsureServer
// binding to it) — acceptable for a test, not attempted to be fully
// closed here.
func freeTCPPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
