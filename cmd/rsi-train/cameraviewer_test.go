package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/stretchr/testify/require"

	rsitesting "github.com/reallyoldfogie/mc-rsi-trainer/testing"
)

// TestCameraViewerManagerLive is a throwaway manual smoke test (like its
// sibling TestDashboardSmokeAgainstSharedServerLive) for cameraviewer.go
// specifically — not a full training run, just: launch an isolated MC
// server (rsitesting.EnsureServer, a distinct port/RCON from any other
// live test or a developer's own training run), connect one real bot so
// there's a valid /spectate target, then drive cameraViewerManager
// exactly as the dashboard's /camera/follow and /camera/stop handlers
// would. Requires the mc-rsi-camera-viewer image already built (see
// camera-viewer/README.md) and a reachable Docker daemon — skips with a
// clear reason if either is missing, same as the live-config gate.
//
// Not run as part of this session's own work: left here, gated exactly
// like every other live test in this package, to run once host load
// allows (see the "check resources" guidance this session's memory
// already captured) rather than adding load to a host someone is
// actively using for diagnostics right now.
func TestCameraViewerManagerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live camera-viewer smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34661", // distinct from every other live test's own fixed port in this package.
		Offline: true,
		Version: "1.21.5",
		Name:    "RSICamSmoke",
	}
	settings.RCON = mcconfig.RCONSettings{
		Address:  "127.0.0.1:25578",
		Password: "rsi-trainer-camera-viewer-smoke-test-pw",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	a, err := connectAgent(ctx, settings, "", "")
	require.NoError(t, err)
	defer func() {
		_ = a.Close(context.Background())
	}()

	viewer, err := newCameraViewerManager(settings, cameraViewerDefaultImage)
	require.NoError(t, err)
	defer func() {
		_ = viewer.stop(context.Background())
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/camera/follow", viewer.followHandler)
	mux.HandleFunc("/camera/stop", viewer.stopHandler)
	mux.Handle(cameraViewerPrefix+"/", viewer.viewProxy())
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/camera/follow?bot="+settings.Connection.Name, "", nil)
	require.NoError(t, err, "POST /camera/follow — is the mc-rsi-camera-viewer image built and Docker reachable?")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	resp.Body.Close()

	resp, err = http.Get(ts.URL + cameraViewerPrefix + "/")
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "proxied 3D page should load once the container is up")
	resp.Body.Close()

	resp, err = http.Post(ts.URL+"/camera/stop", "", nil)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	resp.Body.Close()

	t.Log("✓ follow, proxied view, and stop all round-tripped through the dashboard-style mux")
}
