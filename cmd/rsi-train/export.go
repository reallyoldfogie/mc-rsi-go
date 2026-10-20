package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

// checkpointExporter serves POST /checkpoint/export: an on-demand way to
// force the trainer's currently in-memory models to disk, independent of
// lineage.Save's own two triggers (startup-if-nothing-exists, and a round
// win). Found the gap live: an accidentally deleted checkpoint directory
// left the only surviving copy of a promoted generation sitting in one
// process's memory, with nothing able to re-persist it before either a
// round win or a crash - and a crash first would have silently reset the
// whole leapfrog lineage to a fresh random generation 0. This endpoint
// lets an operator force that save at the moment they notice, rather than
// hope one of those two triggers comes first.
type checkpointExporter struct {
	dir string

	mu                        sync.Mutex
	teacherParams             *actorcritic.Params
	teacherRec                lineage.Record
	studentParams             *actorcritic.Params
	studentEnvID              string
	studentMetadata           checkpoint.Metadata
	studentParentGen          int
	recurrentTeacherParams    *actorcritic.RecurrentParams
	recurrentTeacherRec       lineage.Record
	recurrentStudentParams    *actorcritic.RecurrentParams
	recurrentStudentEnvID     string
	recurrentStudentMetadata  checkpoint.Metadata
	recurrentStudentParentGen int
}

func newCheckpointExporter(dir string) *checkpointExporter {
	return &checkpointExporter{dir: dir}
}

// setTeacher records the generation currently being trained against -
// called once after loadOrInitTeacher and again whenever a round is won.
func (e *checkpointExporter) setTeacher(params *actorcritic.Params, rec lineage.Record) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.teacherParams, e.teacherRec = params, rec
}

// setStudent records the current round's in-progress Student params -
// called from OnEpoch every epoch, so an export mid-round reflects
// whatever training has produced so far, not just the last checkpoint
// interval.
func (e *checkpointExporter) setStudent(params *actorcritic.Params, environmentID string, metadata checkpoint.Metadata, parentGeneration int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.studentParams, e.studentEnvID, e.studentMetadata, e.studentParentGen = params, environmentID, metadata, parentGeneration
}

func (e *checkpointExporter) setRecurrentTeacher(params *actorcritic.RecurrentParams, rec lineage.Record) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.recurrentTeacherParams, e.recurrentTeacherRec = params, rec
}

func (e *checkpointExporter) setRecurrentStudent(params *actorcritic.RecurrentParams, environmentID string, metadata checkpoint.Metadata, parentGeneration int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.recurrentStudentParams, e.recurrentStudentEnvID = params, environmentID
	e.recurrentStudentMetadata, e.recurrentStudentParentGen = metadata, parentGeneration
}

type checkpointExportResult struct {
	Teacher string `json:"teacher,omitempty"`
	Student string `json:"student,omitempty"`
	Error   string `json:"error,omitempty"`
}

// ServeHTTP re-saves the current generation via lineage.Save (the same,
// idempotent write it would make on a win - safe to repeat) and, if a
// round is mid-training, writes the Student's current params to a
// timestamped file under <dir>/manual/. That directory and name are
// deliberately distinct from interim/'s interim-epoch-*.json and
// final.json, so a manual export can never be mistaken by
// loadResumableStudent for that round's own recovery checkpoint.
func (e *checkpointExporter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	e.mu.Lock()
	teacherParams, teacherRec := e.teacherParams, e.teacherRec
	studentParams := e.studentParams
	studentEnvID, studentMetadata, studentParentGen := e.studentEnvID, e.studentMetadata, e.studentParentGen
	recurrentTeacherParams, recurrentTeacherRec := e.recurrentTeacherParams, e.recurrentTeacherRec
	recurrentStudentParams := e.recurrentStudentParams
	recurrentStudentEnvID, recurrentStudentMetadata, recurrentStudentParentGen := e.recurrentStudentEnvID, e.recurrentStudentMetadata, e.recurrentStudentParentGen
	e.mu.Unlock()

	var result checkpointExportResult
	if teacherParams == nil && recurrentTeacherParams == nil {
		result.Error = "not ready: no generation loaded yet"
		respondJSON(w, http.StatusServiceUnavailable, result)
		return
	}
	if teacherParams != nil {
		if err := lineage.Save(e.dir, teacherParams, teacherRec, lineage.PlainCodec); err != nil {
			result.Error = fmt.Sprintf("saving generation %d: %v", teacherRec.Generation, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		result.Teacher = fmt.Sprintf("generation %d re-saved under %s", teacherRec.Generation, e.dir)
	}
	if recurrentTeacherParams != nil {
		if err := lineage.Save(e.dir, recurrentTeacherParams, recurrentTeacherRec, lineage.RecurrentCodec); err != nil {
			result.Error = fmt.Sprintf("saving recurrent generation %d: %v", recurrentTeacherRec.Generation, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		result.Teacher = fmt.Sprintf("generation %d re-saved under %s", recurrentTeacherRec.Generation, e.dir)
	}

	if studentParams != nil {
		manualDir := filepath.Join(e.dir, "manual")
		if err := os.MkdirAll(manualDir, 0o755); err != nil {
			result.Error = fmt.Sprintf("creating %s: %v", manualDir, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		path := filepath.Join(manualDir, fmt.Sprintf("student-%s.json", time.Now().UTC().Format("20060102T150405.000Z")))
		if err := saveParamsAtomically(path, studentParams, studentEnvID, studentMetadata); err != nil {
			result.Error = fmt.Sprintf("saving student snapshot (parent generation %d): %v", studentParentGen, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		result.Student = path
	}
	if recurrentStudentParams != nil {
		manualDir := filepath.Join(e.dir, "manual")
		if err := os.MkdirAll(manualDir, 0o755); err != nil {
			result.Error = fmt.Sprintf("creating %s: %v", manualDir, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		path := filepath.Join(manualDir, fmt.Sprintf("student-recurrent-%s.json", time.Now().UTC().Format("20060102T150405.000Z")))
		if err := saveParamsAtomicallyRecurrent(path, recurrentStudentParams, recurrentStudentEnvID, recurrentStudentMetadata); err != nil {
			result.Error = fmt.Sprintf("saving recurrent student snapshot (parent generation %d): %v", recurrentStudentParentGen, err)
			respondJSON(w, http.StatusInternalServerError, result)
			return
		}
		result.Student = path
	}
	respondJSON(w, http.StatusOK, result)
}

func respondJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
