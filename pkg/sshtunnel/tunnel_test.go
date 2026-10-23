package sshtunnel

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConfigDefaultsAndValidation(t *testing.T) {
	cfg := Config{Target: "user@example.com"}
	require.NoError(t, cfg.Validate())
}

func TestConfigRejectsInvalidPorts(t *testing.T) {
	err := (Config{Target: "user@example.com", RemoteServerPort: 70000}).Validate()
	require.ErrorContains(t, err, "remote_server_port")
}

func TestConfigRejectsLocalPortCollision(t *testing.T) {
	err := (Config{Target: "user@example.com", LocalServerPort: 30000, LocalRCONPort: 30000}).Validate()
	require.ErrorContains(t, err, "must differ")
}
