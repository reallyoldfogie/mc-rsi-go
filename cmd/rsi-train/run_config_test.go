package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestScanRunConfigFlag(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantPath string
		wantErr  bool
	}{
		{"absent", []string{"-eval-episodes", "10"}, "", false},
		{"space-separated", []string{"-run-config", "foo.json"}, "foo.json", false},
		{"equals-separated", []string{"-run-config=foo.json"}, "foo.json", false},
		{"double-dash space-separated", []string{"--run-config", "foo.json"}, "foo.json", false},
		{"double-dash equals-separated", []string{"--run-config=foo.json"}, "foo.json", false},
		{"preceded by other, unrelated flags", []string{"-eval-episodes", "10", "-parallel-envs", "2", "-run-config", "foo.json"}, "foo.json", false},
		{"followed by other flags (override pattern)", []string{"-run-config", "foo.json", "-parallel-envs", "2"}, "foo.json", false},
		{"missing value (last arg)", []string{"-run-config"}, "", true},
		{"a similarly-named but different flag never matches", []string{"-eval-episodes", "-run-config-ish-but-not-quite"}, "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := scanRunConfigFlag(tc.args)
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.wantPath, got)
		})
	}
}

// newTestFlagSet mirrors a tiny slice of parseFlags' own registrations -
// just enough variety (int, float64, bool, string) to exercise
// applyRunConfigFile's type conversion without needing the real,
// ~40-flag FlagSet from main.go.
func newTestFlagSet() (*flag.FlagSet, *struct {
	count    int
	penalty  float64
	enabled  bool
	label    string
}) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	vals := &struct {
		count   int
		penalty float64
		enabled bool
		label   string
	}{}
	fs.IntVar(&vals.count, "count", 0, "")
	fs.Float64Var(&vals.penalty, "penalty", 0, "")
	fs.BoolVar(&vals.enabled, "enabled", false, "")
	fs.StringVar(&vals.label, "label", "", "")
	return fs, vals
}

func writeRunConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "run-flags.json")
	require.NoError(t, os.WriteFile(path, []byte(contents), 0o644))
	return path
}

func TestApplyRunConfigFileSetsEachFlagTypeCorrectly(t *testing.T) {
	fs, vals := newTestFlagSet()
	path := writeRunConfig(t, `{"count": 8, "penalty": 0.2, "enabled": true, "label": "hello"}`)

	require.NoError(t, applyRunConfigFile(fs, path))

	assert.Equal(t, 8, vals.count, "a whole-number JSON float (decoded from the literal 8) must apply cleanly to an IntVar flag")
	assert.Equal(t, 0.2, vals.penalty)
	assert.True(t, vals.enabled)
	assert.Equal(t, "hello", vals.label)
}

func TestApplyRunConfigFileIgnoresUnderscorePrefixedKeys(t *testing.T) {
	fs, vals := newTestFlagSet()
	path := writeRunConfig(t, `{"_comment": "why these values were chosen", "count": 5}`)

	require.NoError(t, applyRunConfigFile(fs, path))

	assert.Equal(t, 5, vals.count)
}

func TestApplyRunConfigFileRejectsUnknownKey(t *testing.T) {
	fs, _ := newTestFlagSet()
	path := writeRunConfig(t, `{"count": 5, "cuont": 5}`)

	err := applyRunConfigFile(fs, path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cuont")
}

func TestApplyRunConfigFileRejectsRunConfigKeyInsideFile(t *testing.T) {
	fs, _ := newTestFlagSet()
	path := writeRunConfig(t, `{"count": 5, "run-config": "other.json"}`)

	err := applyRunConfigFile(fs, path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run-config")
}

func TestApplyRunConfigFileRejectsMalformedJSON(t *testing.T) {
	fs, _ := newTestFlagSet()
	path := writeRunConfig(t, `{not valid json`)

	err := applyRunConfigFile(fs, path)
	require.Error(t, err)
}

func TestApplyRunConfigFileRejectsMissingFile(t *testing.T) {
	fs, _ := newTestFlagSet()

	err := applyRunConfigFile(fs, filepath.Join(t.TempDir(), "does-not-exist.json"))
	require.Error(t, err)
}

func TestApplyRunConfigFileRejectsAnUnsupportedValueType(t *testing.T) {
	fs, _ := newTestFlagSet()
	path := writeRunConfig(t, `{"count": [1, 2, 3]}`)

	err := applyRunConfigFile(fs, path)
	require.Error(t, err)
}

func TestParseFlagsAppliesRunConfigFileAsDefaults(t *testing.T) {
	runConfigPath := writeRunConfig(t, `{
		"mc-agent-config": "/tmp/from-file-config.json",
		"eval-episodes": 10,
		"parallel-envs": 4,
		"time-penalty": 0.2,
		"normalize-observation": true
	}`)

	// -checkpoint-dir is given on the command line (required, never put
	// in the file for this test) alongside -run-config; -parallel-envs
	// is ALSO given on the command line, after -run-config, to confirm
	// it overrides the file's own value for that same key per the
	// "command line always wins" rule - everything else should come
	// from the file.
	f, err := parseFlags([]string{
		"-run-config", runConfigPath,
		"-checkpoint-dir", "/tmp/ckpt",
		"-parallel-envs", "2",
	})
	require.NoError(t, err)

	assert.Equal(t, "/tmp/from-file-config.json", f.mcAgentConfigPath, "unset-on-the-command-line flag should take the file's value")
	assert.Equal(t, 10, f.evalEpisodes)
	assert.Equal(t, 0.2, f.timePenalty)
	assert.True(t, f.normalizeObservation)
	assert.Equal(t, 2, f.parallelEnvs, "a flag given on the command line must win over the same key in the file")
	assert.Equal(t, "/tmp/ckpt", f.checkpointDir, "a flag that's only ever on the command line, never in the file, must still apply normally")
}

func TestParseFlagsRunConfigFileAloneIsEnoughWhenCommandLineOmitsEverythingElse(t *testing.T) {
	runConfigPath := writeRunConfig(t, `{
		"mc-agent-config": "/tmp/from-file-config.json",
		"checkpoint-dir": "/tmp/ckpt-from-file"
	}`)

	f, err := parseFlags([]string{"-run-config", runConfigPath})
	require.NoError(t, err)

	assert.Equal(t, "/tmp/from-file-config.json", f.mcAgentConfigPath)
	assert.Equal(t, "/tmp/ckpt-from-file", f.checkpointDir)
}

func TestParseFlagsRejectsAnUnknownKeyInRunConfigFile(t *testing.T) {
	runConfigPath := writeRunConfig(t, `{"checkpoint-dir": "/tmp/ckpt", "mc-agent-config": "/tmp/c.json", "eval-epsiodes": 10}`)

	_, err := parseFlags([]string{"-run-config", runConfigPath})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "eval-epsiodes")
}
