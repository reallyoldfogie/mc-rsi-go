package testing

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/parallelenv"
)

// This file is this repo's own, much smaller counterpart to mc-agent's
// testing/framework.go: not a general integration-test harness (agent
// spawning, position tracking, world presets — none of that lives here),
// just "is there already a Minecraft server at the address this run's
// config points at; if not, start one there" — what live tests in this
// repo (testing/leapfrog_live_test.go) and, eventually, a developer
// running training locally actually need before anything else can
// connect. See EnsureServer's own doc comment for the full design.

// KeepServerEnvVar, when set to any non-empty value, tells
// ManagedServer.Close to leave a server EnsureServer launched running
// instead of tearing it down — e.g. so a developer can point a second
// terminal's mc-agent session at it, or re-run a live test against the
// same still-warm server without waiting for Minecraft to start again.
// Has no effect on a server EnsureServer merely found already running:
// that one was never this framework's to stop in the first place,
// override or no.
const KeepServerEnvVar = "MC_RSI_TRAINER_KEEP_SERVER"

// defaultServerVersion is used to launch a fresh server when
// settings.Connection.Version is empty — mirrors mc-agent's own
// testing.DefaultServerConfig's version default, for consistency between
// the two repos' test infrastructure rather than picking an unrelated
// value.
const defaultServerVersion = "1.21.5"

// ManagedServer is EnsureServer's result: a Minecraft server reachable at
// Host:Port, plus a Close method that tears it down again if (and only
// if) EnsureServer is the one that launched it.
type ManagedServer struct {
	Host    string
	Port    int
	Version string

	launched bool
	mgr      testenv.Manager
	inst     *testenv.Instance
}

// Close tears down the server via Docker if EnsureServer launched it,
// unless KeepServerEnvVar is set — in which case it logs that it's
// leaving the server running and returns nil. A server EnsureServer
// merely found already running is never touched here at all: Close is
// always a no-op for one of those, regardless of KeepServerEnvVar, since
// stopping a server this framework didn't start would surprise whatever
// else is using it.
func (s *ManagedServer) Close(ctx context.Context) error {
	if !s.launched {
		return nil
	}
	if os.Getenv(KeepServerEnvVar) != "" {
		log.Printf("mcserver: %s is set; leaving %s:%d running", KeepServerEnvVar, s.Host, s.Port)
		return nil
	}
	return s.mgr.Stop(ctx, s.inst, true)
}

// EnsureServer checks for a live Minecraft server at settings.Connection's
// address (a real protocol-level check — rofutils.CheckServerVersion,
// already used elsewhere in this repo to auto-detect a server's version —
// not just a TCP dial, so something else merely listening on that port
// isn't mistaken for a real server) and, if none answers, launches one
// there via Docker (github.com/reallyoldfogie/mc-client-test-go/testenv,
// the same library mc-agent's own testing.Framework builds on), offline
// (no real Mojang account needed) by default and matching
// settings.Connection.Version if set.
//
// settings is taken by pointer and mutated when EnsureServer has to
// launch a server: Connection.Version (resolved to defaultServerVersion
// if it was empty) and RCON.Address/RCON.Password (defaulted if either
// was empty) are filled in with what was actually launched, so the
// caller's very next connectAgent(ctx, *settings) call already has
// accurate connection info without needing to re-derive any of it. A
// server EnsureServer merely finds already running is never mutated into
// settings beyond what the caller already provided — there is no way to
// discover a pre-existing server's RCON password if one wasn't already in
// settings.RCON.Password, which is a real, documented limitation: **set
// an explicit RCON password in your config if you want a later run to be
// able to reuse a server this framework launched** — otherwise every
// launch gets its own random password only that one process ever learns.
//
// opts, if given, customize a server this call actually launches (a
// pre-existing server found already running is used as-is, opts or no —
// there is no way to retroactively change one that's already up). See
// WithExtraEnv for the one option that exists today.
func EnsureServer(ctx context.Context, settings *mcconfig.Settings, opts ...ServerOption) (*ManagedServer, error) {
	host, portStr, err := net.SplitHostPort(settings.Connection.Address)
	if err != nil {
		return nil, fmt.Errorf("mcserver: parsing connection.address %q: %w", settings.Connection.Address, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, fmt.Errorf("mcserver: connection.address %q has a non-numeric port: %w", settings.Connection.Address, err)
	}

	if version, _, err := rofutils.CheckServerVersion(settings.Connection.Address, 0); err == nil {
		log.Printf("mcserver: found an existing server at %s (version %s)", settings.Connection.Address, version)
		return &ManagedServer{Host: host, Port: port, Version: version}, nil
	}

	return launchServer(ctx, settings, host, port, opts...)
}

// ServerOption customizes a server EnsureServer launches. Applied after
// this file's own default ExtraEnv, so an option can override a default
// (e.g. LEVEL_TYPE) as well as add new keys.
type ServerOption func(*testenv.ServerConfig)

// WithExtraEnv merges env into a launched server's container environment
// (itzg/minecraft-server's own env-var config surface — see
// https://github.com/itzg/docker-minecraft-server), on top of this
// file's own defaults (DIFFICULTY, MODE, VIEW_DISTANCE, ...). Existing
// keys are overwritten by env, not merged value-by-value. Added for
// docs/plans/08's production-scale timing run, which needs LEVEL_TYPE=FLAT
// to get a clean step-cost measurement isolated from a real generated
// world's terrain occasionally putting a task's target somewhere
// unreachable (see that document's own "Status" for the live-verified
// failure mode this sidesteps) — kept general rather than a
// single-purpose "flat world" flag since any other env var an itzg image
// supports might turn out useful to a future test the same way.
func WithExtraEnv(env map[string]string) ServerOption {
	return func(cfg *testenv.ServerConfig) {
		if cfg.ExtraEnv == nil {
			cfg.ExtraEnv = map[string]string{}
		}
		for k, v := range env {
			cfg.ExtraEnv[k] = v
		}
	}
}

// WithNamePrefix overrides a launched server's container NamePrefix
// (default "mc-rsi-trainer-"), applied after this file's own default —
// so it always wins over whatever opts a caller passed before it. Added
// for EnsureServers, which uses this to give N concurrently launched
// containers distinct, human-readable names for easy `docker ps`
// identification during a multi-server live test, independent of
// whether their RCON passwords (which already factor into the container
// name via a short id — see testenv's own naming) happen to differ.
func WithNamePrefix(prefix string) ServerOption {
	return func(cfg *testenv.ServerConfig) {
		cfg.NamePrefix = prefix
	}
}

// EnsureServers is EnsureServer's N-way counterpart for
// docs/plans/08-parallel-environments-and-scaling.md's parallel-environment
// work: derives n independent settings copies from base via
// parallelenv.DeriveSettings and calls EnsureServer once per copy
// (sequentially — simpler partial-failure cleanup than concurrent
// launch/teardown, and a few tens of seconds of extra startup latency
// for a handful of servers is an acceptable tradeoff; a future
// optimization if that latency becomes annoying, not attempted here),
// each with a distinct WithNamePrefix applied last so it always wins
// even if a caller's own opts don't differentiate them.
//
// Unlike EnsureServer, base itself is never mutated — there is no
// single "the" settings to mutate for N independent servers. Instead
// every derived, EnsureServer-mutated copy is returned alongside its
// ManagedServer, in the same order. If any of the n launches fails,
// every server successfully launched so far is closed before the error
// is returned, so a partial launch never leaks containers.
//
// Test-support only, like EnsureServer itself — production code
// (cmd/rsi-train/main.go) never imports this package; a real multi-
// environment run instead assumes n already-running external servers
// reachable at addresses derived the same way, via
// pkg/parallelenv.DeriveSettings directly (see connectEnvironments in
// cmd/rsi-train/main.go).
func EnsureServers(ctx context.Context, base mcconfig.Settings, n int, opts ...ServerOption) ([]*ManagedServer, []mcconfig.Settings, error) {
	if n <= 0 {
		return nil, nil, fmt.Errorf("mcserver: n must be positive, got %d", n)
	}

	servers := make([]*ManagedServer, 0, n)
	settingsOut := make([]mcconfig.Settings, 0, n)
	for i := 0; i < n; i++ {
		derived := parallelenv.DeriveSettings(base, i, n)
		indexOpts := append(append([]ServerOption{}, opts...), WithNamePrefix(fmt.Sprintf("mc-rsi-trainer-env%d-", i)))

		srv, err := EnsureServer(ctx, &derived, indexOpts...)
		if err != nil {
			for _, s := range servers {
				_ = s.Close(context.Background())
			}
			return nil, nil, fmt.Errorf("mcserver: launching environment %d/%d: %w", i, n, err)
		}
		servers = append(servers, srv)
		settingsOut = append(settingsOut, derived)
	}
	return servers, settingsOut, nil
}

// launchServer is EnsureServer's "nothing answered, start one" path,
// split out so EnsureServer's own body stays a straightforward
// probe-then-launch shape.
func launchServer(ctx context.Context, settings *mcconfig.Settings, host string, port int, opts ...ServerOption) (*ManagedServer, error) {
	version := settings.Connection.Version
	if version == "" {
		version = defaultServerVersion
	}

	rconHost, rconPort, err := resolveRCONAddress(settings.RCON.Address, host)
	if err != nil {
		return nil, err
	}
	rconPassword := settings.RCON.Password
	if rconPassword == "" {
		rconPassword, err = randomRCONPassword()
		if err != nil {
			return nil, fmt.Errorf("mcserver: generating RCON password: %w", err)
		}
	}

	mgr, err := testenv.NewManager()
	if err != nil {
		return nil, fmt.Errorf("mcserver: creating container manager: %w", err)
	}

	cfg := testenv.ServerConfig{
		Version:        version,
		OnlineMode:     false, // offline by default — see this file's own doc comment.
		HostServerPort: port,
		HostRCONPort:   rconPort,
		RCONPassword:   rconPassword,
		NamePrefix:     "mc-rsi-trainer-",
		ExtraEnv: map[string]string{
			"DIFFICULTY":          "peaceful",
			"MODE":                "survival",
			"VIEW_DISTANCE":       "6",
			"SIMULATION_DISTANCE": "4",
			"GENERATE_STRUCTURES": "false", // fewer unrelated same-type entities (villages, mineshafts, ...) to confuse an RL episode's own seeded/tracked ones — mirrors mc-agent's own testing.Framework default and its reasoning.
		},
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	log.Printf("mcserver: no server found at %s:%d; launching one (version %s, offline mode)", host, port, version)
	inst, err := mgr.Start(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("mcserver: starting server: %w", err)
	}

	if err := mgr.WaitReady(ctx, inst); err != nil {
		_ = mgr.Stop(context.Background(), inst, true)
		return nil, fmt.Errorf("mcserver: waiting for server to become ready: %w", err)
	}

	settings.Connection.Version = version
	settings.RCON.Address = net.JoinHostPort(rconHost, strconv.Itoa(rconPort))
	settings.RCON.Password = rconPassword

	log.Printf("mcserver: server ready at %s:%d (RCON at %s)", host, port, settings.RCON.Address)
	return &ManagedServer{Host: host, Port: port, Version: version, launched: true, mgr: mgr, inst: inst}, nil
}

// resolveRCONAddress splits an already-configured RCON address, or — if
// none was configured — derives one on defaultHost at Minecraft's
// conventional default RCON port, mirroring how a fresh
// mcconfig.Settings has no RCON section populated by default but this
// framework still needs some port to bind the launched container's RCON
// to.
func resolveRCONAddress(configured, defaultHost string) (host string, port int, err error) {
	if configured == "" {
		return defaultHost, 25575, nil
	}
	host, portStr, err := net.SplitHostPort(configured)
	if err != nil {
		return "", 0, fmt.Errorf("mcserver: parsing rcon.address %q: %w", configured, err)
	}
	port, err = strconv.Atoi(portStr)
	if err != nil {
		return "", 0, fmt.Errorf("mcserver: rcon.address %q has a non-numeric port: %w", configured, err)
	}
	return host, port, nil
}

// randomRCONPassword generates a random hex password for a launched
// server's RCON port, used only when the caller's config didn't already
// specify one — see EnsureServer's own doc comment on why setting one
// explicitly is what makes a launched server reusable by a later run.
func randomRCONPassword() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}
