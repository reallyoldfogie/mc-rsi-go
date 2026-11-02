package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
)

// epochRolloutDir returns the directory holding parentGeneration/epoch's
// partial-rollout cache - see saveEpochRollout's own doc comment for why
// this exists at all.
func epochRolloutDir(baseDir string, parentGeneration, epoch int) string {
	return filepath.Join(baseDir, fmt.Sprintf("generation-%09d-epoch-%09d", parentGeneration, epoch))
}

// epochRolloutSlotPath names one trajectory's file within
// epochRolloutDir - %03d comfortably covers any realistic RolloutSize
// (hundreds at most) while keeping filenames fixed-width and sorted.
func epochRolloutSlotPath(dir string, index int) string {
	return filepath.Join(dir, fmt.Sprintf("slot-%03d.json", index))
}

// saveEpochRollout persists one freshly collected trajectory from the
// current, not-yet-finished epoch - wired as Config.OnRolloutCollected -
// so a crash before the epoch completes doesn't lose every trajectory
// collected before it, only whichever one was in flight; see
// loadEpochRollouts, the other half of this, and clearEpochRollouts for
// cleanup once the epoch actually finishes. One file per slot (not one
// shared file for the whole epoch): collectRollouts never has two
// workers touch the same slot index, so this never needs its own
// locking, and a torn or missing write for one slot can't corrupt
// another's. Errors are returned for the caller to log, never fatal:
// losing this cache only costs the recovery it enables, never
// correctness - the epoch just re-collects that trajectory on restart.
func saveEpochRollout(baseDir string, parentGeneration, epoch, index int, rollout *ppo.RecurrentRollout) error {
	dir := epochRolloutDir(baseDir, parentGeneration, epoch)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := epochRolloutSlotPath(dir, index)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := json.NewEncoder(tmp).Encode(rollout); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// loadEpochRollouts loads whatever slot-*.json files
// epochRolloutDir(baseDir, parentGeneration, epoch) holds from an
// earlier, interrupted attempt at this exact epoch, returning a
// rolloutSize-length slice with nil for every slot not found - the exact
// shape ppo.RecurrentTrainer.ResumeRollouts requires. Returns an all-nil
// slice (never an error) if the directory doesn't exist, can't be read,
// or a slot file is corrupt: this is a best-effort recovery aid, not
// something a missing or damaged cache should ever block training over
// - at worst, a bad cache just means every trajectory gets re-collected,
// exactly like today's behavior with no cache at all.
func loadEpochRollouts(baseDir string, parentGeneration, epoch, rolloutSize int) []*ppo.RecurrentRollout {
	dir := epochRolloutDir(baseDir, parentGeneration, epoch)
	rollouts := make([]*ppo.RecurrentRollout, rolloutSize)
	if _, err := os.Stat(dir); err != nil {
		return rollouts
	}
	recovered := 0
	for index := range rolloutSize {
		data, err := os.ReadFile(epochRolloutSlotPath(dir, index))
		if err != nil {
			continue
		}
		var rollout ppo.RecurrentRollout
		if err := json.Unmarshal(data, &rollout); err != nil {
			log.Printf("  epoch %d: ignoring unreadable cached rollout slot %d: %v", epoch, index, err)
			continue
		}
		rollouts[index] = &rollout
		recovered++
	}
	if recovered > 0 {
		log.Printf("  epoch %d: recovered %d/%d already-collected trajectories from an earlier, interrupted attempt", epoch, recovered, rolloutSize)
	}
	return rollouts
}

// clearEpochRollouts removes parentGeneration/epoch's partial-rollout
// cache - called once that epoch actually completes (its data is now
// fully subsumed by that epoch's own interim checkpoint), so a later
// crash on a later epoch never mistakes a stale, already-finished
// epoch's cache for its own. Best-effort: a failure only leaves a
// harmless stale directory behind (loadEpochRollouts is always called
// with the exact generation/epoch about to run, so a leftover
// mismatched directory is simply never read) - never worth failing
// training over.
func clearEpochRollouts(baseDir string, parentGeneration, epoch int) {
	_ = os.RemoveAll(epochRolloutDir(baseDir, parentGeneration, epoch))
}

// pruneStaleEpochRollouts removes every epoch-rollout cache directory
// under baseDir except the one for (keepGeneration, keepEpoch) - the
// generation/epoch about to resume into. Guards against a cache from a
// generation or epoch that's no longer the one being resumed (e.g. the
// checkpoint interval skipped saving an interim checkpoint for the exact
// epoch that crashed, so resume lands on an earlier epoch instead)
// accumulating forever across many restarts. Best-effort, logged but
// never fatal - same reasoning as clearEpochRollouts.
func pruneStaleEpochRollouts(baseDir string, keepGeneration, keepEpoch int) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return
	}
	keep := filepath.Base(epochRolloutDir(baseDir, keepGeneration, keepEpoch))
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == keep {
			continue
		}
		if err := os.RemoveAll(filepath.Join(baseDir, entry.Name())); err != nil {
			log.Printf("  pruning stale epoch-rollout cache %s: %v", entry.Name(), err)
		}
	}
}
