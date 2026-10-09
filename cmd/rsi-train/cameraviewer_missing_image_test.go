package main

import (
	"context"
	"strings"
	"testing"
	"time"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/require"
)

// TestCameraViewerMissingImageFailsGracefully checks the "external
// dependency isn't there" path specifically: unlike
// TestCameraViewerManagerLive (which needs a whole live Minecraft
// server and the real image built), this only needs a reachable Docker
// daemon - checkImage's ImageList call never touches MC/RCON at all, so
// there's nothing live-server-shaped to gate this behind. Skips if
// Docker itself isn't reachable (a real "can't test this here"), not if
// the image is merely missing (that's the exact case under test).
func TestCameraViewerMissingImageFailsGracefully(t *testing.T) {
	settings := mcconfig.Default()
	settings.Connection.Address = "127.0.0.1:34700"
	settings.RCON.Address = "127.0.0.1:25575"
	settings.RCON.Password = "unused-in-this-test"

	viewer, err := newCameraViewerManager(settings, "mc-rsi-camera-viewer-this-tag-should-never-exist:latest")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = viewer.checkImage(ctx)
	if err != nil && strings.Contains(err.Error(), "listing images") && isDockerUnreachable(err) {
		t.Skipf("Docker daemon not reachable in this environment: %v", err)
	}
	require.Error(t, err, "checkImage must fail for an image that was never built")
	require.Contains(t, err.Error(), "not found locally", "error should name the real problem")
	require.Contains(t, err.Error(), "docker build -t", "error should tell the operator how to fix it")

	// The same failure, reached the way a real "Switch to 3D" click
	// reaches it: ensureContainer must fail with the identical message,
	// never attempting ContainerCreate for an image that was never
	// confirmed to exist.
	err = viewer.ensureContainer(ctx)
	require.Error(t, err)
	require.Contains(t, err.Error(), "not found locally")

	err = viewer.follow(ctx, "irrelevant-bot")
	require.Error(t, err, "follow must surface ensureContainer's error, not hang or panic")
}

// isDockerUnreachable is a best-effort heuristic distinguishing "no
// Docker daemon here at all" from checkImage's own normal not-found
// error - both start with "camera-viewer: listing images:", so this
// looks at the wrapped error text itself for the connection-refused/
// no-such-file shape a missing daemon actually produces.
func isDockerUnreachable(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "connect: connection refused") ||
		strings.Contains(msg, "no such file or directory") ||
		strings.Contains(msg, "Cannot connect to the Docker daemon")
}
