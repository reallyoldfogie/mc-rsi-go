package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/moby/moby/api/types/container"
	"github.com/moby/moby/client"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
)

// cameraViewerDefaultImage is the Docker image rsi-train looks for —
// already built, not built by this process (see
// camera-viewer/README.md's "docker build -t mc-rsi-camera-viewer
// camera-viewer/" step; this mirrors how EnsureServer pulls
// itzg/minecraft-server but never builds a custom image from source).
const cameraViewerDefaultImage = "mc-rsi-camera-viewer:latest"

// cameraViewerContainerName is fixed, not per-run: at most one
// camera-viewer container is ever managed by a given rsi-train process —
// one dashboard, one spectator camera, re-targeted via followHandler as
// the selected agent changes (dashboard.html's selectAgent). A stale
// container left over from a previous rsi-train run that exited without
// calling stop (a crash, -9, ...) is removed before creating a fresh one
// (ensureContainer), so that doesn't block the next run.
const cameraViewerContainerName = "rsi-camera-viewer"

// cameraViewerViewerPort/cameraViewerControlPort are camera-viewer's own
// VIEWER_PORT/CONTROL_PORT defaults (camera-viewer/index.js) — fixed
// here rather than flags, since this integration only ever runs at most
// one camera-viewer container at a time. Running the standalone tool
// directly (camera-viewer/README.md) on the same host while rsi-train's
// own -camera-viewer-enabled is also on would collide on these same
// ports; don't do both at once.
const (
	cameraViewerViewerPort  = 3001
	cameraViewerControlPort = 3002
)

// cameraViewerPrefix is the path both this process's own reverse proxy
// (viewProxy) and the container's own prismarine-viewer agree the 3D
// page lives under. prismarine-viewer's client bundle derives its own
// socket.io path from window.location.pathname at load time (see
// camera-viewer/node_modules/prismarine-viewer/lib/index.js, and
// camera-viewer/index.js's own VIEWER_PREFIX doc comment), so the prefix
// the container's VIEWER_PREFIX is launched with must exactly match the
// path the browser actually loads the page at — verified live (both the
// HTML/JS under this prefix and the matching /socket.io handshake)
// while building this integration.
const cameraViewerPrefix = "/camera/view"

// cameraViewerManager owns the lifecycle of at most one camera-viewer
// container (camera-viewer/README.md) on behalf of the dashboard's 3D
// mode toggle (dashboard.html's toggleViewMode). It starts the container
// lazily on the first followHandler request, re-targets it in place on
// every later one (cheap — just an HTTP call into the already-running
// container's own control API), and stops+removes it the moment nobody
// is watching (stopHandler, or this process's own shutdown — see
// Close).
//
// The browser never talks to the container directly: viewProxy reverse-
// proxies the 3D page itself (HTML/JS/WebSocket) through this process's
// own dashboard port under cameraViewerPrefix, and followHandler/
// stopHandler translate the dashboard's own query-param HTTP convention
// (matching envCountRequest's ?n=, restartRequest's bare POST — see
// those types' own ServeHTTP) into the container's JSON-body control API
// (camera-viewer/index.js's POST /follow, documented there) — this is
// the "tunnel traffic and commands... so the browser only has to
// interact directly with the trainer" piece.
type cameraViewerManager struct {
	cli *client.Client

	image        string
	mcHost       string
	mcPort       string
	mcVersion    string
	rconHost     string
	rconPort     string
	rconPassword string

	proxy *httputil.ReverseProxy

	mu          sync.Mutex
	containerID string // "" when no container is running
}

// newCameraViewerManager connects to the local Docker daemon (the same
// DOCKER_HOST/etc. convention testing/mcserver.go's own Manager uses)
// and prepares a manager for mcSettings' own MC server/RCON. It does not
// start a container yet — see ensureContainer, called lazily by
// followHandler on the dashboard's first "switch to 3D" request — so a
// Docker-less or misconfigured-RCON environment only fails once that's
// actually tried, not merely because -camera-viewer-enabled was passed.
func newCameraViewerManager(mcSettings mcconfig.Settings, image string) (*cameraViewerManager, error) {
	mcHost, mcPort, err := net.SplitHostPort(mcSettings.Connection.Address)
	if err != nil {
		return nil, fmt.Errorf("camera-viewer: parsing connection.address %q: %w", mcSettings.Connection.Address, err)
	}
	if mcSettings.RCON.Address == "" {
		return nil, fmt.Errorf("camera-viewer: -mc-agent-config has no rcon.address set — the viewer needs RCON to put its spectator client into spectator mode and run /spectate")
	}
	rconHost, rconPort, err := net.SplitHostPort(mcSettings.RCON.Address)
	if err != nil {
		return nil, fmt.Errorf("camera-viewer: parsing rcon.address %q: %w", mcSettings.RCON.Address, err)
	}

	cli, err := client.New(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("camera-viewer: connecting to Docker: %w", err)
	}

	return &cameraViewerManager{
		cli:          cli,
		image:        image,
		mcHost:       mcHost,
		mcPort:       mcPort,
		mcVersion:    mcSettings.Connection.Version,
		rconHost:     rconHost,
		rconPort:     rconPort,
		rconPassword: mcSettings.RCON.Password,
		proxy:        httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: fmt.Sprintf("127.0.0.1:%d", cameraViewerViewerPort)}),
	}, nil
}

// checkImage reports whether m.image exists in the local Docker image
// store — the one check both ensureContainer (as a hard precondition,
// every time it would otherwise create a container) and
// checkImageAtStartup (as a one-time, non-fatal heads-up — see that
// function's own doc comment) need, kept in one place so they can never
// disagree about what "available" means.
func (m *cameraViewerManager) checkImage(ctx context.Context) error {
	images, err := m.cli.ImageList(ctx, client.ImageListOptions{
		Filters: make(client.Filters).Add("reference", m.image),
	})
	if err != nil {
		return fmt.Errorf("camera-viewer: listing images: %w", err)
	}
	if len(images.Items) == 0 {
		return fmt.Errorf("camera-viewer: image %q not found locally — build it once with: docker build -t %s camera-viewer/", m.image, m.image)
	}
	return nil
}

// checkImageAtStartup logs a warning (never fails the caller) if this
// manager's image isn't available yet, or Docker itself isn't reachable
// — called once, right after -camera-viewer-enabled constructs a
// manager (run, main.go), so a misconfigured or not-yet-built external
// dependency is visible in train.log immediately, rather than only
// surfacing hours later when someone first clicks "Switch to 3D" (that
// click still gets its own clear error either way — see followHandler —
// this is purely an earlier heads-up). Deliberately non-fatal: the 3D
// view is a dashboard nicety, not something worth refusing to train
// over.
//
// The build command is always present in what gets logged, even when
// checkImage's own error is the "can't even reach Docker to check"
// variant rather than "confirmed missing" (whose error text already
// names it — see the strings.Contains guard below, which avoids saying
// it twice in that case): that failure is usually the same root cause
// (image never built, daemon never started for it), and an operator
// reading train.log wants the fix either way, not a report that stops
// short of it.
func (m *cameraViewerManager) checkImageAtStartup(ctx context.Context) {
	err := m.checkImage(ctx)
	if err == nil {
		return
	}
	msg := fmt.Sprintf("%v (the dashboard's 3D mode will fail until this is fixed; training is unaffected)", err)
	if !strings.Contains(err.Error(), "docker build") {
		msg += fmt.Sprintf(" — build the image with: docker build -t %s camera-viewer/", m.image)
	}
	log.Print(msg)
}

// ensureContainer starts the camera-viewer container if one isn't
// already running under this manager, otherwise it's a no-op — the
// common case once the dashboard is already in 3D mode and the selected
// agent changes (follow calls this on every request, cheaply).
func (m *cameraViewerManager) ensureContainer(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.containerID != "" {
		return nil
	}

	if err := m.checkImage(ctx); err != nil {
		return err
	}

	_, _ = m.cli.ContainerRemove(ctx, cameraViewerContainerName, client.ContainerRemoveOptions{Force: true, RemoveVolumes: true})

	env := []string{
		"MC_HOST=" + m.mcHost,
		"MC_PORT=" + m.mcPort,
		"MC_VERSION=" + m.mcVersion,
		"RCON_HOST=" + m.rconHost,
		"RCON_PORT=" + m.rconPort,
		"RCON_PASSWORD=" + m.rconPassword,
		fmt.Sprintf("VIEWER_PORT=%d", cameraViewerViewerPort),
		fmt.Sprintf("CONTROL_PORT=%d", cameraViewerControlPort),
		"VIEWER_PREFIX=" + cameraViewerPrefix,
	}

	createResp, err := m.cli.ContainerCreate(ctx, client.ContainerCreateOptions{
		Name: cameraViewerContainerName,
		Config: &container.Config{
			Image: m.image,
			Env:   env,
		},
		HostConfig: &container.HostConfig{
			// --network host: MC_HOST/RCON_HOST (mcSettings' own
			// connection.address/rcon.address, typically 127.0.0.1 for
			// this project's -shared-server setups) are reached as
			// literal host loopback from inside the container, which
			// only works if it shares the host's own network namespace —
			// see camera-viewer/README.md's "Docker or local" section on
			// why ordinary bridge networking can't reach a loopback-only
			// RCON port.
			NetworkMode: "host",
		},
	})
	if err != nil {
		return fmt.Errorf("camera-viewer: creating container: %w", err)
	}

	if _, err := m.cli.ContainerStart(ctx, createResp.ID, client.ContainerStartOptions{}); err != nil {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = m.cli.ContainerRemove(cleanupCtx, createResp.ID, client.ContainerRemoveOptions{Force: true, RemoveVolumes: true})
		return fmt.Errorf("camera-viewer: starting container: %w", err)
	}

	m.containerID = createResp.ID
	log.Printf("camera-viewer: started container %s (image %s)", createResp.ID[:12], m.image)
	return nil
}

// cameraViewerFollowRequest mirrors camera-viewer/index.js's own POST
// /follow request body exactly — this process is just another HTTP
// client of it, same as a developer's own curl in that tool's README.
type cameraViewerFollowRequest struct {
	Bot string `json:"bot"`
}

// follow ensures the container is running, then tells it (over its own
// control API, not RCON directly — camera-viewer/index.js owns the
// RCON connection) which bot to spectate. Retries for a few seconds
// since a just-started container's control API isn't listening
// instantly (Docker + Node cold start); a container that's already up
// (the common re-target case) answers on the first try.
func (m *cameraViewerManager) follow(ctx context.Context, bot string) error {
	if err := m.ensureContainer(ctx); err != nil {
		return err
	}

	body, err := json.Marshal(cameraViewerFollowRequest{Bot: bot})
	if err != nil {
		return err
	}

	deadline := time.Now().Add(15 * time.Second)
	var lastErr error
	for {
		if lastErr = m.postFollow(ctx, body); lastErr == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("camera-viewer: control API never came up: %w", lastErr)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(300 * time.Millisecond):
		}
	}
}

func (m *cameraViewerManager) postFollow(ctx context.Context, body []byte) error {
	target := fmt.Sprintf("http://127.0.0.1:%d/follow", cameraViewerControlPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("control API returned %s", resp.Status)
	}
	return nil
}

// stop tears down the running container, if any — a no-op otherwise.
// Best-effort: a failed ContainerStop still proceeds to ContainerRemove
// (Force: true kills it directly), matching reconcileEnvCount's own
// "always attempt the next cleanup step" philosophy elsewhere in this
// file's package.
func (m *cameraViewerManager) stop(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.containerID == "" {
		return nil
	}
	id := m.containerID
	m.containerID = ""

	timeoutSeconds := 5
	if _, err := m.cli.ContainerStop(ctx, id, client.ContainerStopOptions{Timeout: &timeoutSeconds}); err != nil {
		log.Printf("camera-viewer: stopping container %s: %v", id[:12], err)
	}
	if _, err := m.cli.ContainerRemove(ctx, id, client.ContainerRemoveOptions{Force: true, RemoveVolumes: true}); err != nil {
		log.Printf("camera-viewer: removing container %s: %v", id[:12], err)
		return err
	}
	log.Printf("camera-viewer: stopped container %s", id[:12])
	return nil
}

// Close tears down any running container on this process's own shutdown
// (run's defer chain) — the dashboard's own beforeunload handler
// (dashboard.html) already covers the ordinary "switched back to 2D, or
// closed the tab" cases; this is the backstop for the process exiting
// without either of those having fired.
func (m *cameraViewerManager) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := m.stop(ctx); err != nil {
		log.Printf("camera-viewer: cleanup on shutdown: %v", err)
	}
}

// followHandler implements POST /camera/follow?bot=<username>.
func (m *cameraViewerManager) followHandler(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	bot := req.URL.Query().Get("bot")
	if bot == "" {
		http.Error(w, "bot query parameter is required", http.StatusBadRequest)
		return
	}
	if err := m.follow(req.Context(), bot); err != nil {
		log.Printf("follow %s: %v", bot, err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	respondJSON(w, http.StatusOK, map[string]string{"following": bot})
}

// stopHandler implements POST /camera/stop.
func (m *cameraViewerManager) stopHandler(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := m.stop(req.Context()); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	respondJSON(w, http.StatusOK, map[string]string{"status": "stopped"})
}

// viewProxy reverse-proxies the 3D page itself (HTML/JS/WebSocket) from
// the running container's prismarine-viewer server through this
// process's own dashboard port, under cameraViewerPrefix — see this
// type's own doc comment. A request here before any followHandler call
// (no container running yet) simply fails with whatever error the
// container's own absence produces (connection refused): the dashboard
// only ever loads this after a successful followHandler response, so
// that ordering is dashboard.html's responsibility, not this proxy's.
func (m *cameraViewerManager) viewProxy() http.Handler {
	return m.proxy
}
