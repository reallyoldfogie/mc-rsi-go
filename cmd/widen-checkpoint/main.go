// Command widen-checkpoint applies cRL-go's actorcritic.Net2WiderNet to a
// checkpoint-dir's latest generation, writing the result as a fresh
// generation 0 in a separate output directory rather than appending it to
// the source lineage — so trying a widened network out never risks a live
// run's own checkpoint history, and rsi-train only ever picks it up as the
// active teacher if pointed at -out-dir explicitly (via its own
// -checkpoint-dir flag) or has that directory's contents copied over.
//
// Entirely offline: reads and writes checkpoint files only, no live bot
// session or Minecraft server involved. See docs on cRL-go's
// Net2WiderNet for what the transform actually does and why it exists —
// this command's own job is just wiring it to mc-rsi-trainer's checkpoint
// format (pkg/lineage) and giving it a CLI.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

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
	newHiddenSize int
	noiseStd      float64
	seed          int64
	verifyProbes  int
}

// defaultNoiseStd is small relative to this network's typical Xavier-scale
// weight magnitudes (roughly 0.05-0.5 for the layer sizes this project
// uses) — a nudge for a new unit to start diverging from its clone-parent,
// not a re-initialization. See actorcritic.Net2WiderNet's own doc comment
// for why some noise is needed at all.
const defaultNoiseStd = 0.05

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("widen-checkpoint", flag.ContinueOnError)
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "pkg/lineage checkpoint directory to widen the latest generation of")
	fs.StringVar(&f.outDir, "out-dir", "", "directory to write the widened network into, as a fresh generation 0 (must not already contain a generation 0 record)")
	fs.IntVar(&f.newHiddenSize, "new-hidden-size", 0, "the widened network's hidden layer width; must be greater than the source checkpoint's own")
	fs.Float64Var(&f.noiseStd, "noise-std", defaultNoiseStd, "std of the noise added to each newly-created unit's weights, breaking the symmetry a pure duplication would otherwise leave forever (0 = exact function preservation, but the new units never diverge from their clone-parent on their own)")
	fs.Int64Var(&f.seed, "seed", 0, "RNG seed governing which existing units get cloned and the noise added to them; the same seed against the same source checkpoint reproduces the same widened network")
	fs.IntVar(&f.verifyProbes, "verify-probes", 8, "random observations to run through both the source and widened network afterward, reporting the largest output difference (0 skips verification)")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("widen-checkpoint: -checkpoint-dir is required")
	}
	if f.outDir == "" {
		return flags{}, fmt.Errorf("widen-checkpoint: -out-dir is required")
	}
	if f.newHiddenSize <= 0 {
		return flags{}, fmt.Errorf("widen-checkpoint: -new-hidden-size is required and must be positive")
	}
	if f.noiseStd < 0 {
		return flags{}, fmt.Errorf("widen-checkpoint: -noise-std must not be negative, got %v", f.noiseStd)
	}
	if f.verifyProbes < 0 {
		return flags{}, fmt.Errorf("widen-checkpoint: -verify-probes must not be negative, got %d", f.verifyProbes)
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
	params, rec, err := lineage.Load(f.checkpointDir, latest.Generation, latest.EnvironmentID)
	if err != nil {
		return fmt.Errorf("loading generation %d: %w", latest.Generation, err)
	}

	oldHiddenSize := params.HiddenSize()
	if f.newHiddenSize <= oldHiddenSize {
		return fmt.Errorf("widen-checkpoint: -new-hidden-size %d must be greater than generation %d's current hidden size %d", f.newHiddenSize, rec.Generation, oldHiddenSize)
	}

	rng := rand.New(rand.NewPCG(uint64(f.seed), uint64(f.seed)))
	wide, err := actorcritic.Net2WiderNet(params, f.newHiddenSize, float32(f.noiseStd), rng)
	if err != nil {
		return fmt.Errorf("widening generation %d: %w", rec.Generation, err)
	}

	if _, err := lineage.Latest(f.outDir); err == nil {
		return fmt.Errorf("widen-checkpoint: %s already has a generation record; refusing to overwrite an existing lineage (pick an empty -out-dir)", f.outDir)
	}
	if err := os.MkdirAll(f.outDir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", f.outDir, err)
	}

	newRec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceWidened,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        rec.EnvironmentID,
		Metadata:             checkpoint.Metadata{},
	}
	if err := lineage.Save(f.outDir, wide, newRec); err != nil {
		return fmt.Errorf("saving the widened network to %s: %w", f.outDir, err)
	}

	sourceAbs, err := filepath.Abs(f.checkpointDir)
	if err != nil {
		sourceAbs = f.checkpointDir
	}
	source := fmt.Sprintf(
		"widened from %s generation %d (environment %q) at %s\nhidden size %d -> %d, noise-std %v, seed %d\n",
		sourceAbs, rec.Generation, rec.EnvironmentID, newRec.CreatedAt.Format(time.RFC3339),
		oldHiddenSize, f.newHiddenSize, f.noiseStd, f.seed,
	)
	if err := os.WriteFile(filepath.Join(f.outDir, "source.txt"), []byte(source), 0o644); err != nil {
		return fmt.Errorf("writing source.txt in %s: %w", f.outDir, err)
	}

	fmt.Printf("wrote generation 0 in %s: hidden size %d -> %d (from %s generation %d)\n", f.outDir, oldHiddenSize, f.newHiddenSize, f.checkpointDir, rec.Generation)

	if f.verifyProbes > 0 {
		maxPolicyDiff, maxValueDiff, err := verify(params, wide, f.verifyProbes, rng)
		if err != nil {
			return fmt.Errorf("verifying the widened network: %w", err)
		}
		// Reported separately, not combined into one number: the policy head
		// is a bounded [0,1] probability, but the value head is not, and can
		// have a larger absolute scale that would otherwise make an
		// unremarkable difference look alarming next to it.
		fmt.Printf("verified against %d random observations: largest policy-probability difference %.5f, largest value difference %.5f (0/0 = exact preservation; noise-std %v means these should be small, not necessarily zero)\n", f.verifyProbes, maxPolicyDiff, maxValueDiff, f.noiseStd)
	}
	return nil
}

// verify runs n random observations through both orig and wide (which must
// share the same input size), returning the largest absolute difference
// seen across every policy probability, and separately across the value
// output — a quick sanity check that widening didn't do something wildly
// different from what Net2WiderNet promises, not a substitute for that
// package's own tests.
func verify(orig, wide *actorcritic.Params, n int, rng *rand.Rand) (maxPolicyDiff, maxValueDiff float64, err error) {
	for i := 0; i < n; i++ {
		values := make([]float32, orig.InputSize())
		for j := range values {
			values[j] = float32(rng.NormFloat64())
		}
		obs := rl.Observation{Values: values}

		origOut, err := forward(orig, obs)
		if err != nil {
			return 0, 0, err
		}
		wideOut, err := forward(wide, obs)
		if err != nil {
			return 0, 0, err
		}
		for k := range origOut.policy {
			if d := absDiff(origOut.policy[k], wideOut.policy[k]); d > maxPolicyDiff {
				maxPolicyDiff = d
			}
		}
		if d := absDiff(origOut.value, wideOut.value); d > maxValueDiff {
			maxValueDiff = d
		}
	}
	return maxPolicyDiff, maxValueDiff, nil
}

type forwardOutput struct {
	policy []float32
	value  float32
}

func forward(p *actorcritic.Params, obs rl.Observation) (forwardOutput, error) {
	net, err := actorcritic.NewInferenceNetwork(p)
	if err != nil {
		return forwardOutput{}, err
	}
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()
	policy := make([]float32, len(net.PolicyOutput.Val.Data))
	copy(policy, net.PolicyOutput.Val.Data)
	return forwardOutput{policy: policy, value: net.ValueOutput.Val.Data[0]}, nil
}

func absDiff(a, b float32) float64 {
	d := float64(a) - float64(b)
	if d < 0 {
		return -d
	}
	return d
}
