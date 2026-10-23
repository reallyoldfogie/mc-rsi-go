package competition

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrepareRunRootCreatesUniqueRunDirectory(t *testing.T) {
	root := t.TempDir()
	startedAt := time.Date(2026, 10, 19, 17, 5, 27, 173573560, time.UTC)

	runRoot, err := PrepareRunRoot(root, "architecture-competition", startedAt)
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(root, "20261019T170527.173573560Z"), runRoot)
	info, err := os.Stat(runRoot)
	require.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestPrepareArtifactDirsCreatesCompetitorDirectories(t *testing.T) {
	runRoot := filepath.Join(t.TempDir(), "run")
	competitors := []Competitor{{ID: "basic"}, {ID: "recurrent-wide"}}

	require.NoError(t, PrepareArtifactDirs(runRoot, competitors))
	for _, competitor := range competitors {
		info, err := os.Stat(filepath.Join(runRoot, competitor.ID))
		require.NoError(t, err)
		assert.True(t, info.IsDir())
	}
}
