package competition

import (
	"testing"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSharedServerConnectionsKeepServerIdentityAndVaryBotNames(t *testing.T) {
	cfg := validConfig()
	cfg.Server.Shared = true
	cfg.Server.SeparationChunks = 16
	cfg.Competitors = []Competitor{
		{ID: "basic", DisplayName: "Basic", HiddenLayers: 2, HiddenSize: 128},
		{ID: "wide", DisplayName: "Wide", HiddenLayers: 2, HiddenSize: 256},
	}
	base := mcconfig.Default()
	base.Connection.Address = "127.0.0.1:25565"
	base.Connection.Name = "competition-bot"
	base.RCON.Address = "127.0.0.1:25575"

	connections, err := cfg.SharedServerConnections(base)
	require.NoError(t, err)
	require.Len(t, connections, 2)
	assert.Equal(t, base.Connection.Address, connections[0].Settings.Connection.Address)
	assert.Equal(t, base.Connection.Address, connections[1].Settings.Connection.Address)
	assert.Equal(t, base.RCON.Address, connections[0].Settings.RCON.Address)
	assert.Equal(t, base.RCON.Address, connections[1].Settings.RCON.Address)
	assert.NotEqual(t, connections[0].Settings.Connection.Name, connections[1].Settings.Connection.Name)
	assert.Equal(t, [3]float64{0, 0, 0}, connections[0].AreaOffset)
	assert.Equal(t, [3]float64{256, 0, 0}, connections[1].AreaOffset)
}

func TestSharedServerConnectionsRejectNonSharedConfig(t *testing.T) {
	cfg := validConfig()
	cfg.Server.Shared = false
	_, err := cfg.SharedServerConnections(mcconfig.Default())
	require.Error(t, err)
	assert.ErrorContains(t, err, "server.shared=true")
}
