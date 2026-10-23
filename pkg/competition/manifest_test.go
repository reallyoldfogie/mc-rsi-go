package competition

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

func TestNewManifestCapturesHashesAndAssignments(t *testing.T) {
	cfg := validConfig()
	plan, err := NewEvaluationPlan(curriculum.Pool{TargetOffsets: [][3]float64{{5, 0, 0}}}, 7, 2)
	require.NoError(t, err)
	window, err := NewTrainingWindow(time.Now(), Duration(time.Hour))
	require.NoError(t, err)
	manifest, err := NewManifest(cfg, []EvaluationPlan{plan}, window, time.Unix(10, 0).UTC())
	require.NoError(t, err)

	assert.Equal(t, cfg.RunID, manifest.RunID)
	assert.Len(t, manifest.ConfigHash, 64)
	assert.Len(t, manifest.EvaluationHash, 64)
	assert.Equal(t, "basic", manifest.Assignments[0].CompetitorID)
}

func TestManifestWriteJSONIsReadable(t *testing.T) {
	cfg := validConfig()
	plan, err := NewEvaluationPlan(curriculum.Pool{TargetOffsets: [][3]float64{{5, 0, 0}}}, 7, 1)
	require.NoError(t, err)
	window, err := NewTrainingWindow(time.Now(), Duration(time.Hour))
	require.NoError(t, err)
	manifest, err := NewManifest(cfg, []EvaluationPlan{plan}, window, time.Now())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "nested", "manifest.json")
	require.NoError(t, manifest.WriteJSON(path))
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"run_id": "test-run"`)
}
