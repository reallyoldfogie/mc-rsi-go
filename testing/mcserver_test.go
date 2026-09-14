package testing

import (
	"context"
	"os"
	"testing"

	"github.com/moby/moby/client"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveRCONAddressDefaultsWhenUnconfigured(t *testing.T) {
	host, port, err := resolveRCONAddress("", "example.test")
	require.NoError(t, err)
	assert.Equal(t, "example.test", host)
	assert.Equal(t, 25575, port, "should default to Minecraft's conventional RCON port")
}

func TestResolveRCONAddressParsesConfigured(t *testing.T) {
	host, port, err := resolveRCONAddress("some-host:12345", "unused")
	require.NoError(t, err)
	assert.Equal(t, "some-host", host)
	assert.Equal(t, 12345, port)
}

func TestResolveRCONAddressRejectsMalformed(t *testing.T) {
	_, _, err := resolveRCONAddress("not-a-valid-host-port", "unused")
	assert.Error(t, err)
}

func TestRandomRCONPasswordIsNonEmptyAndVaries(t *testing.T) {
	a, err := randomRCONPassword()
	require.NoError(t, err)
	b, err := randomRCONPassword()
	require.NoError(t, err)

	assert.NotEmpty(t, a)
	assert.NotEmpty(t, b)
	assert.NotEqual(t, a, b, "two independently generated passwords should not collide")
}

// TestManagedServerCloseIsNoOpWhenNotLaunched verifies Close never
// touches mgr/inst (both left nil here — would panic if dereferenced) for
// a server EnsureServer merely found already running: this framework
// must never stop a server it didn't start, KeepServerEnvVar or not.
func TestManagedServerCloseIsNoOpWhenNotLaunched(t *testing.T) {
	s := &ManagedServer{Host: "example.test", Port: 25565, launched: false}
	assert.NoError(t, s.Close(context.Background()))
}

// stubManager is a minimal testenv.Manager whose Stop calls are
// observable, for testing ManagedServer.Close's KeepServerEnvVar branch
// without touching Docker.
type stubManager struct {
	stopCalls int
}

func (m *stubManager) Start(context.Context, testenv.ServerConfig) (*testenv.Instance, error) {
	panic("not used by these tests")
}
func (m *stubManager) WaitReady(context.Context, *testenv.Instance) error {
	panic("not used by these tests")
}
func (m *stubManager) Stop(context.Context, *testenv.Instance, bool) error {
	m.stopCalls++
	return nil
}
func (m *stubManager) Logs(context.Context, string, client.ContainerLogsOptions) (client.ContainerLogsResult, error) {
	panic("not used by these tests")
}
func (m *stubManager) RCONClient(context.Context, *testenv.Instance) (testenv.RCON, error) {
	panic("not used by these tests")
}

func TestManagedServerCloseSkipsStopWhenKeepEnvVarSet(t *testing.T) {
	t.Setenv(KeepServerEnvVar, "1")
	mgr := &stubManager{}
	s := &ManagedServer{Host: "example.test", Port: 25565, launched: true, mgr: mgr, inst: &testenv.Instance{}}

	require.NoError(t, s.Close(context.Background()))
	assert.Equal(t, 0, mgr.stopCalls, "KeepServerEnvVar must prevent Stop from being called")
}

func TestManagedServerCloseCallsStopWhenLaunchedAndKeepEnvVarUnset(t *testing.T) {
	require.NoError(t, os.Unsetenv(KeepServerEnvVar)) // ensure a clean slate regardless of test order
	mgr := &stubManager{}
	s := &ManagedServer{Host: "example.test", Port: 25565, launched: true, mgr: mgr, inst: &testenv.Instance{}}

	require.NoError(t, s.Close(context.Background()))
	assert.Equal(t, 1, mgr.stopCalls, "a server this framework launched must be stopped by default")
}
