package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRestartRequestTakeReturnsAndClearsPending(t *testing.T) {
	r := newRestartRequest()

	assert.False(t, r.take(), "nothing requested yet")

	r.pending.Store(true)
	assert.True(t, r.take())
	assert.False(t, r.take(), "take must clear the pending request, not hand it out twice")
}

func TestRestartRequestServeHTTPRejectsNonPost(t *testing.T) {
	r := newRestartRequest()
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/control/restart-at-epoch-boundary", nil))

	assert.Equal(t, http.StatusMethodNotAllowed, rec.Code)
	assert.False(t, r.take(), "a rejected request must not still record a pending restart")
}

func TestRestartRequestServeHTTPAcceptsAndRecords(t *testing.T) {
	r := newRestartRequest()
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/control/restart-at-epoch-boundary", nil))

	assert.Equal(t, http.StatusAccepted, rec.Code)
	assert.True(t, r.take())
}
