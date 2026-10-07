package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// pollOnce (and anything else that calls a models.Agent method) is
// deliberately not unit-tested here, consistent with reconcileEnvCount's
// own tests: this repo's live bot-connection code is exercised by its
// integration test suites, not hand-rolled models.Agent fakes for the
// whole (large) interface. A nil agent is fine for everything below since
// none of it calls into agent.

func TestAgentRegistryAddAssignsSequentialIndices(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.add("bot-1", nil)

	snap := r.snapshot()
	require.Len(t, snap, 2)
	assert.Equal(t, 0, snap[0].Index)
	assert.Equal(t, "bot-0", snap[0].Username)
	assert.Equal(t, 1, snap[1].Index)
	assert.Equal(t, "bot-1", snap[1].Username)
}

func TestAgentRegistryTruncateDropsHighestIndexed(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.add("bot-1", nil)
	r.add("bot-2", nil)

	r.truncate(1)

	snap := r.snapshot()
	require.Len(t, snap, 1)
	assert.Equal(t, "bot-0", snap[0].Username)
}

func TestAgentRegistryTruncateAboveCountIsNoop(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	r.truncate(5)

	assert.Len(t, r.snapshot(), 1)
}

func TestAgentRegistryUpdateEpisodeAppliesFnAndStampsUpdatedAt(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	r.updateEpisode(0, func(s *agentSnapshot) {
		s.Task = "goto"
		s.EpisodeStep = 3
	})

	snap := r.snapshot()
	require.Len(t, snap, 1)
	assert.Equal(t, "goto", snap[0].Task)
	assert.Equal(t, 3, snap[0].EpisodeStep)
	assert.False(t, snap[0].UpdatedAt.IsZero())
}

func TestAgentRegistryUpdateEpisodeOutOfRangeIsNoop(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	assert.NotPanics(t, func() {
		r.updateEpisode(5, func(s *agentSnapshot) { s.Task = "should-not-apply" })
	})
}

func TestAgentRegistryUpdateEpisodeNilReceiverIsNoop(t *testing.T) {
	var r *agentRegistry
	assert.NotPanics(t, func() {
		r.updateEpisode(0, func(s *agentSnapshot) { s.Task = "unreachable" })
	})
}

func TestAgentRegistryServeHTTPReturnsSnapshotAsJSON(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.updateEpisode(0, func(s *agentSnapshot) { s.Task = "mine" })

	req := httptest.NewRequest(http.MethodGet, "/agents", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var got []agentSnapshot
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	require.Len(t, got, 1)
	assert.Equal(t, "bot-0", got[0].Username)
	assert.Equal(t, "mine", got[0].Task)
}

func TestAgentRegistryServeHTTPRejectsNonGet(t *testing.T) {
	r := newAgentRegistry()

	req := httptest.NewRequest(http.MethodPost, "/agents", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func TestServeDashboardPageServesEmbeddedHTML(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/dashboard", nil)
	w := httptest.NewRecorder()
	serveDashboardPage(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Header().Get("Content-Type"), "text/html")
	assert.Contains(t, w.Body.String(), "<!doctype html>")
}
