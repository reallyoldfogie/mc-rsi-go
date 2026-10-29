// Command expand-checkpoint applies cRL-go's actorcritic.ExpandParams/
// ExpandRecurrentParams to a checkpoint-dir's latest generation, writing
// the result as a fresh generation 0 in a separate output directory,
// same convention as cmd/widen-checkpoint/cmd/deepen-checkpoint - so
// trying it never risks a live run's own checkpoint history.
//
// Most callers never need this command at all: cmd/rsi-train's own
// resume path (via pkg/lineage.Load's "Expanding" codecs) already
// upgrades a smaller actions=/obs() shape transparently, with no
// explicit step and no new generation record. This command exists for
// the cases that aren't an ordinary resume - inspecting or handing off
// a reshaped checkpoint without touching a live run's own lineage
// directory, or converting one as a one-off outside cmd/rsi-train
// entirely.
//
// Entirely offline: reads and writes checkpoint files only, no live bot
// session or Minecraft server involved. Handles both plain and
// recurrent generations (lineage.Record.Recurrent decides which codec
// to use), unlike cmd/widen-checkpoint/cmd/deepen-checkpoint, which
// only ever handle plain ones - see cRL-go's ExpandRecurrentParams for
// what changes for the recurrent case (the LSTM layers are untouched;
// only the plain trunks underneath them resize).
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

type flags struct {
	checkpointDir string
	outDir        string
	newActions    int
	newObs        int
	verifyProbes  int
}

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("expand-checkpoint", flag.ContinueOnError)
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "pkg/lineage checkpoint directory to expand the latest generation of")
	fs.StringVar(&f.outDir, "out-dir", "", "directory to write the expanded network into, as a fresh generation 0 (must not already contain a generation 0 record)")
	fs.IntVar(&f.newActions, "new-actions", 0, "the expanded network's action-space size; must be at least the source checkpoint's own")
	fs.IntVar(&f.newObs, "new-obs", 0, "the expanded network's observation-vector size; must be at least the source checkpoint's own")
	fs.IntVar(&f.verifyProbes, "verify-probes", 8, "random observations to run through both the source and expanded network afterward, confirming every old action's probability ratio against the others held exactly (0 skips verification)")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("expand-checkpoint: -checkpoint-dir is required")
	}
	if f.outDir == "" {
		return flags{}, fmt.Errorf("expand-checkpoint: -out-dir is required")
	}
	if f.newActions <= 0 {
		return flags{}, fmt.Errorf("expand-checkpoint: -new-actions is required and must be positive")
	}
	if f.newObs <= 0 {
		return flags{}, fmt.Errorf("expand-checkpoint: -new-obs is required and must be positive")
	}
	if f.verifyProbes < 0 {
		return flags{}, fmt.Errorf("expand-checkpoint: -verify-probes must not be negative, got %d", f.verifyProbes)
	}
	return f, nil
}

func run(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}

	latest, err := lineage.Latest(f.checkpointDir)
	if err != nil {
		return fmt.Errorf("finding the latest generation in %s: %w", f.checkpointDir, err)
	}

	newEnvironmentID, ok := actorcritic.WithActionObsID(latest.EnvironmentID, f.newActions, f.newObs)
	if !ok {
		return fmt.Errorf("expand-checkpoint: generation %d's environment ID %q doesn't embed an \"actions=N:obs=M\" fragment to replace", latest.Generation, latest.EnvironmentID)
	}

	if _, err := lineage.Latest(f.outDir); err == nil {
		return fmt.Errorf("expand-checkpoint: %s already has a generation record; refusing to overwrite an existing lineage (pick an empty -out-dir)", f.outDir)
	}
	if err := os.MkdirAll(f.outDir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", f.outDir, err)
	}

	newRec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceExpanded,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        newEnvironmentID,
		Metadata:             checkpoint.Metadata{},
		Recurrent:            latest.Recurrent,
	}

	var oldActions, oldObs int
	rng := rand.New(rand.NewPCG(1, 1))
	if latest.Recurrent {
		params, rec, err := lineage.Load(f.checkpointDir, latest.Generation, latest.EnvironmentID, lineage.RecurrentCodec)
		if err != nil {
			return fmt.Errorf("loading generation %d: %w", latest.Generation, err)
		}
		oldActions, oldObs = params.PolicyTrunk.OutputSize(), params.PolicyTrunk.InputSize()
		expanded, err := actorcritic.ExpandRecurrentParams(params, f.newObs, f.newActions)
		if err != nil {
			return fmt.Errorf("expanding generation %d: %w", rec.Generation, err)
		}
		if err := lineage.Save(f.outDir, expanded, newRec, lineage.RecurrentCodec); err != nil {
			return fmt.Errorf("saving the expanded network to %s: %w", f.outDir, err)
		}
		if f.verifyProbes > 0 {
			maxDiff, err := verify(params.PolicyTrunk, expanded.PolicyTrunk, f.verifyProbes, rng)
			if err != nil {
				return fmt.Errorf("verifying the expanded network: %w", err)
			}
			fmt.Printf("verified against %d random observations: largest old-action log-probability-ratio difference %.6f (0 = exact preservation)\n", f.verifyProbes, maxDiff)
		}
	} else {
		params, rec, err := lineage.Load(f.checkpointDir, latest.Generation, latest.EnvironmentID, lineage.PlainCodec)
		if err != nil {
			return fmt.Errorf("loading generation %d: %w", latest.Generation, err)
		}
		oldActions, oldObs = params.OutputSize(), params.InputSize()
		expanded, err := actorcritic.ExpandParams(params, f.newObs, f.newActions)
		if err != nil {
			return fmt.Errorf("expanding generation %d: %w", rec.Generation, err)
		}
		if err := lineage.Save(f.outDir, expanded, newRec, lineage.PlainCodec); err != nil {
			return fmt.Errorf("saving the expanded network to %s: %w", f.outDir, err)
		}
		if f.verifyProbes > 0 {
			maxDiff, err := verify(params, expanded, f.verifyProbes, rng)
			if err != nil {
				return fmt.Errorf("verifying the expanded network: %w", err)
			}
			fmt.Printf("verified against %d random observations: largest old-action log-probability-ratio difference %.6f (0 = exact preservation)\n", f.verifyProbes, maxDiff)
		}
	}

	sourceAbs, err := filepath.Abs(f.checkpointDir)
	if err != nil {
		sourceAbs = f.checkpointDir
	}
	source := fmt.Sprintf(
		"expanded from %s generation %d (environment %q) at %s\nactions %d -> %d, obs %d -> %d\n",
		sourceAbs, latest.Generation, latest.EnvironmentID, newRec.CreatedAt.Format(time.RFC3339),
		oldActions, f.newActions, oldObs, f.newObs,
	)
	if err := os.WriteFile(filepath.Join(f.outDir, "source.txt"), []byte(source), 0o644); err != nil {
		return fmt.Errorf("writing source.txt in %s: %w", f.outDir, err)
	}

	fmt.Printf("wrote generation 0 in %s: actions %d -> %d, obs %d -> %d (from %s generation %d)\n", f.outDir, oldActions, f.newActions, oldObs, f.newObs, f.checkpointDir, latest.Generation)
	return nil
}

// verify runs n random old-shape observations through both orig and
// expanded (orig's own trunk Params either way - run's recurrent
// branch passes PolicyTrunk/expanded.PolicyTrunk, since the LSTM layers
// ExpandRecurrentParams leaves untouched have nothing to verify),
// reporting the largest absolute difference in log(p[k]/p[0]) between
// action 0 and every other old action - log-probability DIFFERENCE is
// exactly logit DIFFERENCE, the actual softmax-invariant property
// ExpandParams guarantees (see its own doc comment on why a new
// action's raw post-softmax probability is not itself preserved, only
// harmless once masked). Checked this way, not as a raw probability
// ratio, because a real, heavily-trained checkpoint's policy can be
// confident enough that p[0] itself rounds to something extremely
// close to zero for a given observation - found live, running this
// tool against a copy of this repo's own production checkpoint: a
// raw-ratio version of this check reported a swing of 268 on an
// otherwise byte-for-byte-correct expansion (confirmed directly against
// the raw policy vectors), purely from dividing by a near-zero float32.
// Subtracting logs never divides by anything that small. The new
// observation columns beyond orig's own InputSize() are filled with
// random garbage on purpose - they must not move this number at all.
func verify(orig, expanded *actorcritic.Params, n int, rng *rand.Rand) (maxDiff float64, err error) {
	oldSize, newSize := orig.InputSize(), expanded.InputSize()
	oldActions := orig.OutputSize()
	for i := 0; i < n; i++ {
		oldObs := make([]float32, oldSize)
		for j := range oldObs {
			oldObs[j] = float32(rng.NormFloat64())
		}
		newObs := make([]float32, newSize)
		copy(newObs, oldObs)
		for j := oldSize; j < newSize; j++ {
			newObs[j] = float32(rng.NormFloat64()) // garbage - must be ignored
		}

		origPolicy, err := policyOutput(orig, oldObs)
		if err != nil {
			return 0, err
		}
		expandedPolicy, err := policyOutput(expanded, newObs)
		if err != nil {
			return 0, err
		}
		for k := 1; k < oldActions; k++ {
			origLogRatio := math.Log(float64(origPolicy[k])) - math.Log(float64(origPolicy[0]))
			expandedLogRatio := math.Log(float64(expandedPolicy[k])) - math.Log(float64(expandedPolicy[0]))
			if d := absDiff(origLogRatio, expandedLogRatio); d > maxDiff {
				maxDiff = d
			}
		}
	}
	return maxDiff, nil
}

func policyOutput(p *actorcritic.Params, obs []float32) ([]float32, error) {
	net, err := actorcritic.NewInferenceNetwork(p)
	if err != nil {
		return nil, err
	}
	copy(net.Input.Val.Data, obs)
	net.Graph.Forward()
	policy := make([]float32, len(net.PolicyOutput.Val.Data))
	copy(policy, net.PolicyOutput.Val.Data)
	return policy, nil
}

func absDiff(a, b float64) float64 {
	d := a - b
	if d < 0 {
		return -d
	}
	return d
}

