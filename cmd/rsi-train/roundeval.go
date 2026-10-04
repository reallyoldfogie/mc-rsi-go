package main

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
)

// roundEvalState is the subset of leapfrog.Result that recordRoundEvaluation
// turns into metrics - everything except StudentWon (already covered by the
// restart-surviving rsi_rounds_total counter, via increase()) and
// StudentParams (irrelevant to the gauges this restores). Persisted so the
// "Teacher vs. Student reward per round" and "Student advantage vs.
// promotion threshold" dashboard panels don't sit at a misleading 0 between
// a restart and whenever the in-progress round next completes, possibly
// hours later - the same restart-visibility gap rsi_best_return_alltime had
// before loadResumableStudent's own fix (see cmd/rsi-train/main.go).
type roundEvalState struct {
	TeacherReward float32 `json:"teacher_reward"`
	StudentReward float32 `json:"student_reward"`
	EvalEpisodes  int     `json:"eval_episodes"`
	TeacherStd    float32 `json:"teacher_std"`
	StudentStd    float32 `json:"student_std"`
	MeanDiff      float32 `json:"mean_diff"`
	DiffStdErr    float32 `json:"diff_std_err"`
}

// roundEvalStateFile is where saveRoundEvalState/loadRoundEvalState keep the
// latest completed round's evaluation stats, alongside the interim student
// checkpoint that already lives in interimDir.
func roundEvalStateFile(interimDir string) string {
	return filepath.Join(interimDir, "round-eval.json")
}

// loadRoundEvalState reads back the last round's evaluation stats saved by
// saveRoundEvalState, or returns ok=false if none exists yet (a fresh run,
// or one from before this existed) - never an error a caller needs to
// handle, mirroring loadPersistedParallelEnvs's own "missing means nothing
// to restore" convention.
func loadRoundEvalState(interimDir string) (roundEvalState, bool) {
	data, err := os.ReadFile(roundEvalStateFile(interimDir))
	if err != nil {
		return roundEvalState{}, false
	}
	var state roundEvalState
	if err := json.Unmarshal(data, &state); err != nil {
		return roundEvalState{}, false
	}
	return state, true
}

// saveRoundEvalState atomically (temp file plus rename, so a crash mid-write
// can never leave a truncated/corrupt file behind) records result's
// evaluation stats, overwriting whatever the previous round left behind.
func saveRoundEvalState(interimDir string, result leapfrog.Result[*actorcritic.Params]) error {
	state := roundEvalState{
		TeacherReward: result.TeacherReward,
		StudentReward: result.StudentReward,
		EvalEpisodes:  result.EvalEpisodes,
		TeacherStd:    result.TeacherStd,
		StudentStd:    result.StudentStd,
		MeanDiff:      result.MeanDiff,
		DiffStdErr:    result.DiffStdErr,
	}
	path := roundEvalStateFile(interimDir)
	tmp, err := os.CreateTemp(interimDir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := json.NewEncoder(tmp).Encode(state); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
