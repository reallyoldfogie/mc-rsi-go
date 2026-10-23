// Package sshtunnel manages the local forwards needed to reach a remote
// Minecraft server whose only exposed service is SSH.
package sshtunnel

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	DefaultMinecraftPort = 25565
	DefaultRCONPort      = 25575
	readyTimeout         = 15 * time.Second
	readyPoll            = 100 * time.Millisecond
)

// Config describes one SSH connection forwarding Minecraft and RCON from the
// remote host's loopback into local TCP ports. Zero local ports are allocated
// automatically; remote ports default to Minecraft's conventional values.
type Config struct {
	Target           string `json:"target"`
	RemoteServerPort int    `json:"remote_server_port,omitempty"`
	RemoteRCONPort   int    `json:"remote_rcon_port,omitempty"`
	LocalServerPort  int    `json:"local_server_port,omitempty"`
	LocalRCONPort    int    `json:"local_rcon_port,omitempty"`
}

func (c Config) Validate() error {
	if c.Target == "" {
		return fmt.Errorf("ssh tunnel target must not be empty")
	}
	for name, port := range map[string]int{
		"remote_server_port": c.RemoteServerPort, "remote_rcon_port": c.RemoteRCONPort,
		"local_server_port": c.LocalServerPort, "local_rcon_port": c.LocalRCONPort,
	} {
		if port < 0 || port > 65535 {
			return fmt.Errorf("ssh tunnel %s must be between 0 and 65535", name)
		}
	}
	if c.LocalServerPort != 0 && c.LocalServerPort == c.LocalRCONPort {
		return fmt.Errorf("ssh tunnel local server and RCON ports must differ")
	}
	return nil
}

// Tunnel is an owned SSH forwarding process. Close is idempotent.
type Tunnel struct {
	cmd     *exec.Cmd
	stderr  bytes.Buffer
	server  string
	rcon    string
	wait    chan struct{}
	waitErr error
	mu      sync.Mutex
	closed  bool
	target  string
}

// Start opens the SSH tunnel and waits until both local forwarding sockets
// exist. Authentication is deliberately non-interactive so missing keys or
// host configuration fail at startup instead of hanging training.
func Start(ctx context.Context, cfg Config) (*Tunnel, error) {
	if cfg.RemoteServerPort == 0 {
		cfg.RemoteServerPort = DefaultMinecraftPort
	}
	if cfg.RemoteRCONPort == 0 {
		cfg.RemoteRCONPort = DefaultRCONPort
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if _, err := exec.LookPath("ssh"); err != nil {
		return nil, fmt.Errorf("ssh tunnel: %w", err)
	}
	serverPort := cfg.LocalServerPort
	if serverPort == 0 {
		serverPort, _ = freePort()
	}
	rconPort := cfg.LocalRCONPort
	if rconPort == 0 {
		for {
			var err error
			rconPort, err = freePort()
			if err != nil {
				return nil, err
			}
			if rconPort != serverPort {
				break
			}
		}
	}
	args := []string{
		"-N", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
		"-o", "ExitOnForwardFailure=yes",
		"-L", fmt.Sprintf("127.0.0.1:%d:127.0.0.1:%d", serverPort, cfg.RemoteServerPort),
		"-L", fmt.Sprintf("127.0.0.1:%d:127.0.0.1:%d", rconPort, cfg.RemoteRCONPort),
		cfg.Target,
	}
	cmd := exec.CommandContext(ctx, "ssh", args...)
	t := &Tunnel{cmd: cmd, server: net.JoinHostPort("127.0.0.1", fmt.Sprint(serverPort)), rcon: net.JoinHostPort("127.0.0.1", fmt.Sprint(rconPort)), wait: make(chan struct{}), target: cfg.Target}
	cmd.Stderr = &t.stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ssh tunnel: start: %w", err)
	}
	go func() {
		err := cmd.Wait()
		t.mu.Lock()
		t.waitErr = err
		intentional := t.closed
		stderr := strings.TrimSpace(t.stderr.String())
		t.mu.Unlock()
		log.Printf("ssh tunnel process exited: target=%s intentional=%t err=%v stderr=%q", t.target, intentional, err, stderr)
		close(t.wait)
	}()
	if err := t.waitReady(ctx); err != nil {
		_ = t.Close()
		return nil, err
	}
	return t, nil
}

func (t *Tunnel) MinecraftAddress() string { return t.server }
func (t *Tunnel) RCONAddress() string      { return t.rcon }

func (t *Tunnel) waitReady(ctx context.Context) error {
	deadline := time.Now().Add(readyTimeout)
	for {
		select {
		case <-t.wait:
			return fmt.Errorf("ssh tunnel exited before ready: %v\nstderr:\n%s", t.waitErr, t.stderr.String())
		default:
		}
		serverOK := canDial(t.server)
		rconOK := canDial(t.rcon)
		if serverOK && rconOK {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("ssh tunnel: local forwards not ready after %s\nstderr:\n%s", readyTimeout, t.stderr.String())
		}
		time.Sleep(readyPoll)
	}
}

func canDial(address string) bool {
	c, err := net.DialTimeout("tcp", address, 250*time.Millisecond)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

func (t *Tunnel) Close() error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil
	}
	t.closed = true
	t.mu.Unlock()
	if t.cmd == nil || t.cmd.Process == nil {
		return nil
	}
	if err := t.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return err
	}
	select {
	case <-t.wait:
	case <-time.After(5 * time.Second):
	}
	return nil
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("ssh tunnel: find local port: %w", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
