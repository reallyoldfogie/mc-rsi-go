// Command deepen-checkpoint applies cRL-go's actorcritic.Net2DeeperNet to a
// checkpoint-dir's latest generation, writing the result as a fresh
// generation 0 in a separate output directory rather than appending it to
// the source lineage — so trying a deeper network out never risks a live
// run's own checkpoint history, and rsi-train only ever picks it up as the
// active teacher if pointed at -out-dir explicitly (via its own
// -checkpoint-dir flag) or has that directory's contents copied over.
// Mirrors cmd/widen-checkpoint's own shape and doc comment; see that
// command for the sibling "grow width instead of depth" transform.
//
// Entirely offline: reads and writes checkpoint files only, no live bot
// session or Minecraft server involved. See docs on cRL-go's
// Net2DeeperNet for what the transform actually does and why it exists —
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
	atDepth       int
	seed          int64
	verifyProbes  int
}

func parseFlags(args []string) (flags, error) {
	var f flags
	fs := flag.NewFlagSet("deepen-checkpoint", flag.ContinueOnError)
	fs.StringVar(&f.checkpointDir, "checkpoint-dir", "", "pkg/lineage checkpoint directory to deepen the latest generation of")
	fs.StringVar(&f.outDir, "out-dir", "", "directory to write the deepened network into, as a fresh generation 0 (must not already contain a generation 0 record)")
	fs.IntVar(&f.atDepth, "at-depth", 0, "where to insert the new hidden layer: the number of existing hidden layers that come before it (1..the source checkpoint's own hidden-layer count; depth 0, before the first hidden layer, is never valid — see actorcritic.Net2DeeperNet's own doc comment for why)")
	fs.Int64Var(&f.seed, "seed", 0, "RNG seed for the random observations used to verify the deepened network (the insertion itself is deterministic, unlike widening — there is nothing else for a seed to govern)")
	fs.IntVar(&f.verifyProbes, "verify-probes", 8, "random observations to run through both the source and deepened network afterward, reporting the largest output difference (0 skips verification)")
	if err := fs.Parse(args); err != nil {
		return flags{}, err
	}

	if f.checkpointDir == "" {
		return flags{}, fmt.Errorf("deepen-checkpoint: -checkpoint-dir is required")
	}
	if f.outDir == "" {
		return flags{}, fmt.Errorf("deepen-checkpoint: -out-dir is required")
	}
	if f.atDepth <= 0 {
		return flags{}, fmt.Errorf("deepen-checkpoint: -at-depth is required and must be positive")
	}
	if f.verifyProbes < 0 {
		return flags{}, fmt.Errorf("deepen-checkpoint: -verify-probes must not be negative, got %d", f.verifyProbes)
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
	params, rec, err := lineage.Load(f.checkpointDir, latest.Generation, latest.EnvironmentID, lineage.PlainCodec)
	if err != nil {
		return fmt.Errorf("loading generation %d: %w", latest.Generation, err)
	}

	oldNumHiddenLayers := params.NumHiddenLayers()
	deeper, err := actorcritic.Net2DeeperNet(params, f.atDepth)
	if err != nil {
		return fmt.Errorf("deepening generation %d: %w", rec.Generation, err)
	}

	if _, err := lineage.Latest(f.outDir); err == nil {
		return fmt.Errorf("deepen-checkpoint: %s already has a generation record; refusing to overwrite an existing lineage (pick an empty -out-dir)", f.outDir)
	}
	if err := os.MkdirAll(f.outDir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", f.outDir, err)
	}

	newRec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceDeepened,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        rec.EnvironmentID,
		Metadata:             checkpoint.Metadata{},
	}
	if err := lineage.Save(f.outDir, deeper, newRec, lineage.PlainCodec); err != nil {
		return fmt.Errorf("saving the deepened network to %s: %w", f.outDir, err)
	}

	sourceAbs, err := filepath.Abs(f.checkpointDir)
	if err != nil {
		sourceAbs = f.checkpointDir
	}
	source := fmt.Sprintf(
		"deepened from %s generation %d (environment %q) at %s\nhidden layers %d -> %d, new layer inserted at depth %d\n",
		sourceAbs, rec.Generation, rec.EnvironmentID, newRec.CreatedAt.Format(time.RFC3339),
		oldNumHiddenLayers, deeper.NumHiddenLayers(), f.atDepth,
	)
	if err := os.WriteFile(filepath.Join(f.outDir, "source.txt"), []byte(source), 0o644); err != nil {
		return fmt.Errorf("writing source.txt in %s: %w", f.outDir, err)
	}

	fmt.Printf("wrote generation 0 in %s: hidden layers %d -> %d (from %s generation %d)\n", f.outDir, oldNumHiddenLayers, deeper.NumHiddenLayers(), f.checkpointDir, rec.Generation)

	if f.verifyProbes > 0 {
		rng := rand.New(rand.NewPCG(uint64(f.seed), uint64(f.seed)))
		maxPolicyDiff, maxValueDiff, err := verify(params, deeper, f.verifyProbes, rng)
		if err != nil {
			return fmt.Errorf("verifying the deepened network: %w", err)
		}
		// Reported separately, not combined into one number: the policy head
		// is a bounded [0,1] probability, but the value head is not, and can
		// have a larger absolute scale that would otherwise make an
		// unremarkable difference look alarming next to it.
		fmt.Printf("verified against %d random observations: largest policy-probability difference %.5f, largest value difference %.5f (0/0 = exact preservation, which Net2DeeperNet always gives — a nonzero reading here means something is wrong, unlike widen-checkpoint's noise-std case)\n", f.verifyProbes, maxPolicyDiff, maxValueDiff)
	}
	return nil
}

// verify runs n random observations through both orig and deeper (which
// must share the same input size), returning the largest absolute
// difference seen across every policy probability, and separately across
// the value output — a quick sanity check that deepening didn't do
// something wildly different from what Net2DeeperNet promises, not a
// substitute for that package's own tests. Unlike cmd/widen-checkpoint's
// identically-named helper, any nonzero difference here is a red flag:
// Net2DeeperNet has no noise parameter, so the source and deepened
// networks should compute the exact same function.
func verify(orig, deeper *actorcritic.Params, n int, rng *rand.Rand) (maxPolicyDiff, maxValueDiff float64, err error) {
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
		deeperOut, err := forward(deeper, obs)
		if err != nil {
			return 0, 0, err
		}
		for k := range origOut.policy {
			if d := absDiff(origOut.policy[k], deeperOut.policy[k]); d > maxPolicyDiff {
				maxPolicyDiff = d
			}
		}
		if d := absDiff(origOut.value, deeperOut.value); d > maxValueDiff {
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
