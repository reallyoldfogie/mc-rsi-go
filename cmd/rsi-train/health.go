package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// errNetworkDead is returned when the policy network has stopped responding
// to its input. It is fatal on purpose: a run in that state keeps consuming
// hours and looks alive (found live: every second-layer ReLU unit was dead
// from generation 2, the policy was a constant of four biases, and 20 rounds
// of "training" were noise before anyone looked). main exits with
// exitNetworkDead so the supervisor doesn't restart into it.
var errNetworkDead = errors.New("policy network is dead: no hidden unit responds to any input, so the policy is a constant")

// exitNetworkDead is the process exit status for errNetworkDead;
// scripts/supervise-rsi-train.sh treats it as "do not restart".
const exitNetworkDead = 3

// deadEpochsBeforeAbort is how many consecutive epochs must look dead before
// the run aborts, so one odd reading can't end a run.
const deadEpochsBeforeAbort = 3

// networkHealth summarises whether a policy network still depends on its
// input, measured on a fixed set of probe observations.
type networkHealth struct {
	// LayerActive is, for each hidden layer in order (index 0 is the layer
	// closest to the input), the fraction of its ReLU units that fire
	// (pre-activation > 0) for at least one probe. A dead unit passes no
	// gradient, so it can never recover.
	LayerActive []float64
	// LogitStd is the mean, over the policy's outputs, of the standard
	// deviation of that logit across the probes. 0 means the policy is the
	// same in every state.
	LogitStd float64
	// PolicyEntropy is the mean Shannon entropy (nats) of softmax(logits)
	// across the probes - how uncertain the action distribution actually is,
	// as opposed to LogitStd's raw spread of the pre-softmax values. A fresh
	// network sits near ln(actionSpace) (closest to uniform); it falls as
	// the policy sharpens, and a near-zero value means the policy is
	// effectively deterministic for every probe - the softmax-saturation
	// concern entropy_coef is meant to counteract.
	PolicyEntropy float64
	// LayerCorrelation is, for each hidden layer (index-aligned with
	// LayerActive), the mean absolute Pearson correlation, over every pair
	// of that layer's units, of their activations across the probes (dead
	// or constant units, whose std is ~0, are excluded - their correlation
	// with anything is undefined, not informative). This is a
	// capacity-direction proxy, not a dead-unit check (LayerActive already
	// covers that): HIGH correlation means many units are behaving
	// near-identically - redundant, so the layer has unused headroom
	// already and growing hidden_size would likely just add more redundant
	// capacity. LOW correlation means units are already doing mutually
	// distinct work - no redundancy to reclaim, which is the more
	// capacity-constrained-looking signature. See
	// docs/plans/12-multi-step-crafting-curriculum.md's "Diagnosing a
	// training plateau" section for how to read this alongside the other
	// signals.
	LayerCorrelation []float64
}

// Thresholds for dead. A freshly initialised network has ~99-100% of its
// last hidden layer active and a logit std around 0.2; the dead network that
// motivated this check measured 0.8% (one unit) and 0.002, in what was then
// (before docs/plans/20 Part A) always exactly the second and last hidden
// layer. The cutoffs sit an order of magnitude on the dead side of a healthy
// network.
const (
	deadLastLayerActive = 0.05
	deadLogitStd        = 0.02
)

// dead reports whether the network is beyond learning: almost nothing in its
// last hidden layer (the one feeding the policy/value heads directly) fires,
// or its output barely varies with its input. Checking the *last* layer
// specifically, regardless of how many hidden layers the network has,
// matches the original incident this check was built for: a dead last layer
// means the heads only ever see a constant input, so the policy is a
// constant no matter how healthy any earlier layer looks.
func (h networkHealth) dead() bool {
	lastLayerActive := h.LayerActive[len(h.LayerActive)-1]
	return lastLayerActive < deadLastLayerActive || h.LogitStd < deadLogitStd
}

// formatLayerActive renders LayerActive as a human-readable "layer1 83%,
// layer2 91%" summary for log messages (see main.go).
func formatLayerActive(layerActive []float64) string {
	parts := make([]string, len(layerActive))
	for i, active := range layerActive {
		parts[i] = fmt.Sprintf("layer%d %.0f%%", i+1, active*100)
	}
	return strings.Join(parts, ", ")
}

const healthProbeCount = 256

// healthProbes are deterministic observations spread over the normalized
// feature ranges (see rlenv.Config.NormalizeObservation): distances of a few
// units either way, angles and vitals in [-1, 1], and random task flags.
func healthProbes(inputSize int) [][]float32 {
	rng := rand.New(rand.NewPCG(20260926, 1))
	probes := make([][]float32, healthProbeCount)
	for i := range probes {
		x := make([]float32, inputSize)
		for j := range x {
			switch {
			case j < 3 || (j >= 9 && j < 12): // distances
				x[j] = float32(rng.NormFloat64())
			case j < 5: // yaw, pitch
				x[j] = float32(rng.Float64()*2 - 1)
			case j < 8: // health, food, saturation
				x[j] = float32(rng.Float64())
			default: // known/visible/ready and task flags
				x[j] = float32(rng.IntN(2))
			}
		}
		probes[i] = x
	}
	return probes
}

// measureNetworkHealth runs the probes through p by hand (the same ReLU
// trunk and linear policy head actorcritic builds, generalized to however
// many hidden layers p actually has - see docs/plans/20 Part A) and reports
// how much of the network responds.
func measureNetworkHealth(p *actorcritic.Params) networkHealth {
	probes := healthProbes(p.InputSize())
	numLayers := p.NumHiddenLayers()

	active := make([][]bool, numLayers)
	// acts is layer-major then unit-major (one []float64 per unit, one
	// entry per probe) so meanAbsCorrelation can compute each unit's own
	// mean/std once and every pairwise correlation without re-scanning the
	// probes.
	acts := make([][][]float64, numLayers)
	for l := range numLayers {
		width := p.Hidden[l].W.Rows
		active[l] = make([]bool, width)
		acts[l] = make([][]float64, width)
		for u := range acts[l] {
			acts[l][u] = make([]float64, 0, len(probes))
		}
	}

	logits := make([][]float64, p.Wpi.Rows)
	var entropySum float64
	for _, x := range probes {
		a := x
		for l := range numLayers {
			a = relu(affine(p.Hidden[l].W, p.Hidden[l].B, a))
			for u, v := range a {
				if v > 0 {
					active[l][u] = true
				}
				acts[l][u] = append(acts[l][u], float64(v))
			}
		}

		out := affine(p.Wpi, p.Bpi, a)
		for k, v := range out {
			logits[k] = append(logits[k], float64(v))
		}
		entropySum += softmaxEntropy(out)
	}

	var stdSum float64
	for _, l := range logits {
		_, std := meanStdOf(l)
		stdSum += std
	}

	layerActive := make([]float64, numLayers)
	layerCorrelation := make([]float64, numLayers)
	for l := range numLayers {
		layerActive[l] = fractionTrue(active[l])
		layerCorrelation[l] = meanAbsCorrelation(acts[l])
	}

	return networkHealth{
		LayerActive:      layerActive,
		LogitStd:         stdSum / float64(len(logits)),
		PolicyEntropy:    entropySum / float64(len(probes)),
		LayerCorrelation: layerCorrelation,
	}
}

// softmaxEntropy returns the Shannon entropy, in nats, of softmax(logits).
// The max-subtraction is the standard numerically-stable softmax (keeps
// exp's argument <= 0 regardless of how large the raw logits have grown).
func softmaxEntropy(logits []float32) float64 {
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sumExp float64
	exps := make([]float64, len(logits))
	for i, v := range logits {
		exps[i] = math.Exp(float64(v - maxLogit))
		sumExp += exps[i]
	}
	var entropy float64
	for _, e := range exps {
		p := e / sumExp
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy
}

// meanAbsCorrelation returns the mean absolute Pearson correlation over
// every pair of units in acts (one []float64 per unit, one entry per
// probe), excluding any unit whose activation is constant across every
// probe (std 0 - correlation with it is undefined, not zero, so including
// it would misleadingly pull the mean toward "uncorrelated"). Returns 0 if
// fewer than two units have nonzero variance.
func meanAbsCorrelation(acts [][]float64) float64 {
	means := make([]float64, len(acts))
	stds := make([]float64, len(acts))
	varying := make([]int, 0, len(acts))
	for i, a := range acts {
		means[i], stds[i] = meanStdOf(a)
		if stds[i] > 0 {
			varying = append(varying, i)
		}
	}
	if len(varying) < 2 {
		return 0
	}

	var sum float64
	var pairs int
	for a := 0; a < len(varying); a++ {
		i := varying[a]
		for b := a + 1; b < len(varying); b++ {
			j := varying[b]
			var cov float64
			for t := range acts[i] {
				cov += (acts[i][t] - means[i]) * (acts[j][t] - means[j])
			}
			cov /= float64(len(acts[i]))
			sum += math.Abs(cov / (stds[i] * stds[j]))
			pairs++
		}
	}
	return sum / float64(pairs)
}

func affine(w, b *mat.Matrix, x []float32) []float32 {
	out := make([]float32, w.Rows)
	for r := range out {
		s := b.Data[r]
		row := w.Data[r*w.Cols : (r+1)*w.Cols]
		for c, xv := range x {
			s += row[c] * xv
		}
		out[r] = s
	}
	return out
}

func relu(x []float32) []float32 {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
	return x
}

func fractionTrue(bs []bool) float64 {
	n := 0
	for _, b := range bs {
		if b {
			n++
		}
	}
	return float64(n) / float64(len(bs))
}

func meanStdOf(xs []float64) (mean, std float64) {
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	for _, x := range xs {
		std += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(std / float64(len(xs)))
}
