// Package parallelenv derives N distinct mc-agent connection settings
// from one base config, for running N concurrent live Minecraft
// environments (see docs/plans/08-parallel-environments-and-scaling.md).
// It is pure Go with no Docker dependency, so it's safe to import from
// production code (cmd/rsi-train/main.go), not just test helpers.
package parallelenv

import (
	"fmt"
	"net"
	"strconv"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
)

// defaultRCONPort is the conventional Minecraft RCON port, used when
// base leaves RCON.Address unset — mirrors
// mc-rsi-trainer/testing/mcserver.go's own resolveRCONAddress default,
// so a derived config's implied RCON port matches what EnsureServer
// would itself pick for the same host.
const defaultRCONPort = 25575

// DeriveSettings returns a copy of base suitable for the (index+1)-th
// of n concurrent Minecraft environments. When n <= 1, base is returned
// completely unchanged — this is what keeps -parallel-envs=1 (the
// default) byte-for-byte identical to today's single-environment
// behavior.
//
// For n > 1: Connection.Address's port and RCON.Address's port
// (RCON.Address defaulted to host:defaultRCONPort first, using
// Connection.Address's own host, if base left RCON.Address unset) are
// each offset by index. Connection.Name is replaced via DeriveUsername.
// RCON.Password, if base already set one, gets an "-envN" suffix (so N
// auto-launched containers get distinct names — see
// testing.EnsureServers — without discarding an operator's own explicit
// password, which docs/plans/10-live-server-management.md documents as
// the way to make a launched server reusable by a later run); left
// empty if base left it empty (testing.EnsureServer/launchServer's own
// randomRCONPassword already generates an independent password per
// launch call in that case).
func DeriveSettings(base mcconfig.Settings, index, n int) mcconfig.Settings {
	if n <= 1 {
		return base
	}

	derived := base

	if addr, err := offsetPort(base.Connection.Address, index); err == nil {
		derived.Connection.Address = addr
	}
	derived.Connection.Name = DeriveUsername(base.Connection.Name, index, n)

	rconBase := base.RCON.Address
	if rconBase == "" {
		if host, _, err := net.SplitHostPort(base.Connection.Address); err == nil {
			rconBase = net.JoinHostPort(host, strconv.Itoa(defaultRCONPort))
		}
	}
	if rconBase != "" {
		if addr, err := offsetPort(rconBase, index); err == nil {
			derived.RCON.Address = addr
		}
	}

	if base.RCON.Password != "" {
		derived.RCON.Password = fmt.Sprintf("%s-env%d", base.RCON.Password, index)
	}

	return derived
}

// offsetPort parses a "host:port" address and returns "host:port+index".
func offsetPort(address string, index int) (string, error) {
	host, portStr, err := net.SplitHostPort(address)
	if err != nil {
		return "", fmt.Errorf("parallelenv: parsing address %q: %w", address, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return "", fmt.Errorf("parallelenv: parsing port %q: %w", portStr, err)
	}
	return net.JoinHostPort(host, strconv.Itoa(port+index)), nil
}

// DeriveSharedServerSettings returns a copy of base for the (index+1)-th
// of n bots sharing ONE server: only Connection.Name differs (via
// DeriveUsername) — Connection.Address/RCON.Address/RCON.Password are
// left completely unchanged from base for every index, since every bot
// connects to the exact same server (unlike DeriveSettings' N-separate-
// servers derivation, which offsets ports per index). n <= 1 returns
// base unchanged, matching DeriveSettings' own convention.
func DeriveSharedServerSettings(base mcconfig.Settings, index, n int) mcconfig.Settings {
	if n <= 1 {
		return base
	}
	derived := base
	derived.Connection.Name = DeriveUsername(base.Connection.Name, index, n)
	return derived
}

// blocksPerChunk is Minecraft's own chunk size — the unit a shared-
// server working-area separation is naturally expressed in, converted
// to blocks here so callers can reason in blocks
// (rlenv.Config.ResetOrigin's own unit) without duplicating the
// conversion themselves.
const blocksPerChunk = 16

// WorkingAreaOffset returns the (index+1)-th of n bots' working-area
// offset from a shared reference point, separationChunks*blocksPerChunk
// blocks along X — a simple 1D line, not a grid: sufficient separation
// at this project's target scale (a handful of bots) without the added
// complexity a grid layout would need (see
// docs/plans/08-parallel-environments-and-scaling.md's own shared-
// server design for why a grid isn't attempted). index 0 always returns
// a zero offset, so the first bot's working area is exactly the
// reference point itself.
func WorkingAreaOffset(index, separationChunks int) [3]float64 {
	return [3]float64{float64(index * separationChunks * blocksPerChunk), 0, 0}
}

// maxUsernameLength is Minecraft's own bot-username limit, enforced
// server-side and, more usefully, by ../mc-agent/agent.ResolveAuth
// before ever reaching the server (see
// docs/plans/08-parallel-environments-and-scaling.md's "Status" section
// for the bug this closed). DeriveUsername guarantees every name it
// returns fits within this so a derived name is never the reason
// ResolveAuth rejects a config.
const maxUsernameLength = 16

// DeriveUsername returns a Minecraft-legal (<=16 character) bot
// username for the (index+1)-th of n environments. n <= 1 returns base
// unchanged. Otherwise base is truncated as needed and given a "-N"
// suffix, guaranteed to fit within maxUsernameLength since n's target
// scale (single-digit indices) never needs more than a 2-character
// suffix.
func DeriveUsername(base string, index, n int) string {
	if n <= 1 {
		return base
	}

	suffix := fmt.Sprintf("-%d", index)
	maxBaseLen := maxUsernameLength - len(suffix)
	if len(base) > maxBaseLen {
		base = base[:maxBaseLen]
	}
	return base + suffix
}
