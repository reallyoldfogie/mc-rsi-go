package parallelenv

import (
	"strconv"
	"testing"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/assert"
)

func baseSettings() mcconfig.Settings {
	var s mcconfig.Settings
	s.Connection.Address = "localhost:25565"
	s.Connection.Name = "botname"
	return s
}

func TestDeriveSettingsNoopWhenNIsOne(t *testing.T) {
	base := baseSettings()

	derived := DeriveSettings(base, 0, 1)

	assert.Equal(t, base, derived)
}

func TestDeriveSettingsNoopWhenNIsZero(t *testing.T) {
	base := baseSettings()

	derived := DeriveSettings(base, 0, 0)

	assert.Equal(t, base, derived)
}

func TestDeriveSettingsOffsetsConnectionPort(t *testing.T) {
	base := baseSettings()

	for i := range 4 {
		derived := DeriveSettings(base, i, 4)
		assert.Equal(t, "localhost:"+strconv.Itoa(25565+i), derived.Connection.Address)
	}
}

func TestDeriveSettingsDefaultsAndOffsetsRCONPortWhenBaseLeavesItUnset(t *testing.T) {
	base := baseSettings()
	base.RCON.Address = ""

	derived := DeriveSettings(base, 2, 4)

	assert.Equal(t, "localhost:25577", derived.RCON.Address)
}

func TestDeriveSettingsOffsetsExplicitRCONAddress(t *testing.T) {
	base := baseSettings()
	base.RCON.Address = "localhost:25575"

	derived := DeriveSettings(base, 3, 4)

	assert.Equal(t, "localhost:25578", derived.RCON.Address)
}

func TestDeriveSettingsSuffixesPasswordOnlyWhenBaseSetOne(t *testing.T) {
	base := baseSettings()

	withPassword := base
	withPassword.RCON.Password = "secret"
	derived := DeriveSettings(withPassword, 1, 4)
	assert.Equal(t, "secret-env1", derived.RCON.Password)

	derivedEmpty := DeriveSettings(base, 1, 4)
	assert.Equal(t, "", derivedEmpty.RCON.Password)
}

func TestDeriveSettingsDerivesDistinctUsernames(t *testing.T) {
	base := baseSettings()

	seen := map[string]bool{}
	for i := range 4 {
		derived := DeriveSettings(base, i, 4)
		assert.False(t, seen[derived.Connection.Name], "username %q reused across indices", derived.Connection.Name)
		seen[derived.Connection.Name] = true
		assert.LessOrEqual(t, len(derived.Connection.Name), maxUsernameLength)
	}
}

func TestDeriveSharedServerSettingsNoopWhenNIsOne(t *testing.T) {
	base := baseSettings()

	derived := DeriveSharedServerSettings(base, 0, 1)

	assert.Equal(t, base, derived)
}

func TestDeriveSharedServerSettingsNoopWhenNIsZero(t *testing.T) {
	base := baseSettings()

	derived := DeriveSharedServerSettings(base, 0, 0)

	assert.Equal(t, base, derived)
}

func TestDeriveSharedServerSettingsKeepsConnectionAndRCONIdentical(t *testing.T) {
	base := baseSettings()
	base.RCON.Address = "localhost:25575"
	base.RCON.Password = "secret"

	for i := range 4 {
		derived := DeriveSharedServerSettings(base, i, 4)
		assert.Equal(t, base.Connection.Address, derived.Connection.Address, "index %d", i)
		assert.Equal(t, base.RCON.Address, derived.RCON.Address, "index %d", i)
		assert.Equal(t, base.RCON.Password, derived.RCON.Password, "index %d", i)
	}
}

func TestDeriveSharedServerSettingsDerivesDistinctUsernames(t *testing.T) {
	base := baseSettings()

	seen := map[string]bool{}
	for i := range 4 {
		derived := DeriveSharedServerSettings(base, i, 4)
		assert.False(t, seen[derived.Connection.Name], "username %q reused across indices", derived.Connection.Name)
		seen[derived.Connection.Name] = true
		assert.Equal(t, DeriveUsername(base.Connection.Name, i, 4), derived.Connection.Name)
	}
}

func TestWorkingAreaOffsetIsZeroAtIndexZero(t *testing.T) {
	assert.Equal(t, [3]float64{0, 0, 0}, WorkingAreaOffset(0, 16))
}

func TestWorkingAreaOffsetScalesWithIndexAndSeparation(t *testing.T) {
	assert.Equal(t, [3]float64{256, 0, 0}, WorkingAreaOffset(1, 16))
	assert.Equal(t, [3]float64{512, 0, 0}, WorkingAreaOffset(2, 16))
	assert.Equal(t, [3]float64{128, 0, 0}, WorkingAreaOffset(2, 4))
}

func TestDeriveUsernameNoop(t *testing.T) {
	assert.Equal(t, "botname", DeriveUsername("botname", 0, 1))
	assert.Equal(t, "botname", DeriveUsername("botname", 0, 0))
}

func TestDeriveUsernameFitsWithinLimitWhenBaseAlreadyShort(t *testing.T) {
	// 7-char base + "-3" (2 chars) = 9, well within 16.
	assert.Equal(t, "shortname-3", DeriveUsername("shortname", 3, 4))
}

func TestDeriveUsernameTruncatesBaseAtMaxLength(t *testing.T) {
	base := "1234567890123456" // exactly 16 chars
	derived := DeriveUsername(base, 3, 4)
	assert.LessOrEqual(t, len(derived), maxUsernameLength)
	assert.Equal(t, "12345678901234-3", derived)
}

func TestDeriveUsernameBoundaryFitsExactly(t *testing.T) {
	// 14-char base + "-3" (2 chars) = exactly 16.
	base := "12345678901234"
	derived := DeriveUsername(base, 3, 4)
	assert.Equal(t, maxUsernameLength, len(derived))
	assert.Equal(t, base+"-3", derived)
}
