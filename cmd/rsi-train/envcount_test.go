package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnvCountRequestTakeReturnsAndClearsPending(t *testing.T) {
	r := newEnvCountRequest()

	_, ok := r.take()
	assert.False(t, ok, "nothing requested yet")

	r.pending = ptr(5)
	n, ok := r.take()
	require.True(t, ok)
	assert.Equal(t, 5, n)

	_, ok = r.take()
	assert.False(t, ok, "take must clear the pending request, not hand out the same one twice")
}

func TestEnvCountRequestServeHTTPRejectsNonPost(t *testing.T) {
	r := newEnvCountRequest()
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/config/parallel-envs?n=5", nil))

	assert.Equal(t, http.StatusMethodNotAllowed, rec.Code)
	_, ok := r.take()
	assert.False(t, ok, "a rejected request must not still record a pending change")
}

func TestEnvCountRequestServeHTTPRejectsNonPositiveN(t *testing.T) {
	for _, n := range []string{"0", "-1", "", "not-a-number"} {
		r := newEnvCountRequest()
		rec := httptest.NewRecorder()

		r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/config/parallel-envs?n="+n, nil))

		assert.Equal(t, http.StatusBadRequest, rec.Code, "n=%q", n)
		_, ok := r.take()
		assert.False(t, ok, "n=%q must not record a pending change", n)
	}
}

func TestEnvCountRequestServeHTTPAcceptsAndRecords(t *testing.T) {
	r := newEnvCountRequest()
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/config/parallel-envs?n=10", nil))

	assert.Equal(t, http.StatusAccepted, rec.Code)
	n, ok := r.take()
	require.True(t, ok)
	assert.Equal(t, 10, n)
}

func TestEnvCountRequestLatestRequestWins(t *testing.T) {
	r := newEnvCountRequest()
	r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/config/parallel-envs?n=3", nil))
	r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/config/parallel-envs?n=9", nil))

	n, ok := r.take()
	require.True(t, ok)
	assert.Equal(t, 9, n, "a second request before the first is applied must replace it, not queue both")
}

// TestReconcileEnvCountNoopWhenAlreadyAtTarget and
// TestReconcileEnvCountRejectsLessThanOne exercise reconcileEnvCount's two
// checks that run before it ever touches a real agent connection (no-op,
// and the n<1 guard) — the rest of reconcileEnvCount (actually connecting
// or closing bot sessions) isn't unit-tested here, consistent with
// connectEnvironments' own: this repo's live bot-connection code is
// exercised by its integration test suites, not hand-rolled models.Agent
// fakes for the whole (large) interface.
func TestReconcileEnvCountNoopWhenAlreadyAtTarget(t *testing.T) {
	connected := []connectedEnvironment{{}, {}}
	envs := []rl.Environment{nil, nil}

	gotConnected, gotEnvs, err := reconcileEnvCount(nil, nil, connected, envs, 2,
		mcconfig.Settings{}, false, false, 0, nil, "", episodeLimits{}, "task", nil)

	require.NoError(t, err)
	assert.Len(t, gotConnected, 2)
	assert.Len(t, gotEnvs, 2)
}

func TestReconcileEnvCountRejectsLessThanOne(t *testing.T) {
	connected := []connectedEnvironment{{}, {}}
	envs := []rl.Environment{nil, nil}

	_, _, err := reconcileEnvCount(nil, nil, connected, envs, 0,
		mcconfig.Settings{}, false, false, 0, nil, "", episodeLimits{}, "task", nil)

	assert.ErrorContains(t, err, "at least 1")
}

func ptr[T any](v T) *T { return &v }

func TestLoadPersistedParallelEnvsAbsentFileFallsBack(t *testing.T) {
	_, ok := loadPersistedParallelEnvs(t.TempDir())
	assert.False(t, ok, "no file ever written must behave exactly like no scale was ever requested")
}

func TestLoadPersistedParallelEnvsRejectsInvalidContent(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(parallelEnvsStateFile(dir), []byte("not json"), 0o644))
	_, ok := loadPersistedParallelEnvs(dir)
	assert.False(t, ok)

	require.NoError(t, os.WriteFile(parallelEnvsStateFile(dir), []byte(`{"count":0}`), 0o644))
	_, ok = loadPersistedParallelEnvs(dir)
	assert.False(t, ok, "a non-positive count must not override -parallel-envs")
}

func TestSaveAndLoadPersistedParallelEnvsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, savePersistedParallelEnvs(dir, 10))

	n, ok := loadPersistedParallelEnvs(dir)
	require.True(t, ok)
	assert.Equal(t, 10, n)
}

// TestParallelEnvsConfirmerPersistsAfterTheDelay and
// TestParallelEnvsConfirmerNeverCrashedNeverWritesAnything are this
// feature's actual point, from a live incident (2026-10-01): a scale-up
// to 10 envs crashed the whole process about 5 minutes in (one of the
// new bots timed out), and the next restart must fall back to
// -parallel-envs's own flag value, not repeat (and potentially re-crash
// on) the same untested count.
func TestParallelEnvsConfirmerPersistsAfterTheDelay(t *testing.T) {
	dir := t.TempDir()
	c := newParallelEnvsConfirmer(dir)
	c.delay = 10 * time.Millisecond

	c.scheduleConfirm(10)

	require.Eventually(t, func() bool {
		n, ok := loadPersistedParallelEnvs(dir)
		return ok && n == 10
	}, time.Second, 5*time.Millisecond, "scheduleConfirm must persist once its delay elapses")
}

func TestParallelEnvsConfirmerNeverCrashedNeverWritesAnything(t *testing.T) {
	dir := t.TempDir()
	c := newParallelEnvsConfirmer(dir)
	c.delay = time.Hour // simulates "the process exits before this fires" without actually waiting

	c.scheduleConfirm(10)

	_, ok := loadPersistedParallelEnvs(dir)
	assert.False(t, ok, "nothing must be written before the stability delay elapses - this is the whole protection a crash shortly after a scale relies on")
}

func TestParallelEnvsConfirmerSupersedesAnEarlierPendingScale(t *testing.T) {
	dir := t.TempDir()
	c := newParallelEnvsConfirmer(dir)
	c.delay = 10 * time.Millisecond

	c.scheduleConfirm(8)
	c.scheduleConfirm(10) // supersedes 8 before it had a chance to persist

	require.Eventually(t, func() bool {
		n, ok := loadPersistedParallelEnvs(dir)
		return ok && n == 10
	}, time.Second, 5*time.Millisecond)

	// Give the (cancelled) first timer every chance to have fired too, so
	// this isn't just checking which one happened to run last.
	time.Sleep(50 * time.Millisecond)
	n, ok := loadPersistedParallelEnvs(dir)
	require.True(t, ok)
	assert.Equal(t, 10, n, "the superseded scale (8) must never overwrite the one that replaced it")
}

func TestParallelEnvsStateFileIsInsideCheckpointDir(t *testing.T) {
	assert.Equal(t, filepath.Join("/ckpt", "parallel-envs.json"), parallelEnvsStateFile("/ckpt"))
}
