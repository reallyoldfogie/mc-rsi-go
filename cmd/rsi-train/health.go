package main

import (
	"errors"
	"math"
	"math/rand/v2"

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
	// Layer1Active and Layer2Active are the fractions of first- and
	// second-layer ReLU units that fire (pre-activation > 0) for at least one
	// probe. A dead unit passes no gradient, so it can never recover.
	Layer1Active, Layer2Active float64
	// LogitStd is the mean, over the policy's outputs, of the standard
	// deviation of that logit across the probes. 0 means the policy is the
	// same in every state.
	LogitStd float64
}

// Thresholds for dead. A freshly initialised network has ~99-100% of its
// second layer active and a logit std around 0.2; the dead network that
// motivated this check measured 0.8% (one unit) and 0.002. The cutoffs sit an
// order of magnitude on the dead side of a healthy network.
const (
	deadLayer2Active = 0.05
	deadLogitStd     = 0.02
)

// dead reports whether the network is beyond learning: almost nothing in its
// second layer fires, or its output barely varies with its input.
func (h networkHealth) dead() bool {
	return h.Layer2Active < deadLayer2Active || h.LogitStd < deadLogitStd
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

// measureNetworkHealth runs the probes through p by hand (the same two ReLU
// layers and linear policy head actorcritic builds) and reports how much of
// the network responds.
func measureNetworkHealth(p *actorcritic.Params) networkHealth {
	probes := healthProbes(p.InputSize())
	active1 := make([]bool, p.W0.Rows)
	active2 := make([]bool, p.W1.Rows)
	logits := make([][]float64, p.Wpi.Rows)
	for _, x := range probes {
		h1 := relu(affine(p.W0, p.B0, x))
		for j, v := range h1 {
			if v > 0 {
				active1[j] = true
			}
		}
		h2 := relu(affine(p.W1, p.B1, h1))
		for j, v := range h2 {
			if v > 0 {
				active2[j] = true
			}
		}
		for k, v := range affine(p.Wpi, p.Bpi, h2) {
			logits[k] = append(logits[k], float64(v))
		}
	}

	var stdSum float64
	for _, l := range logits {
		_, std := meanStdOf(l)
		stdSum += std
	}
	return networkHealth{
		Layer1Active: fractionTrue(active1),
		Layer2Active: fractionTrue(active2),
		LogitStd:     stdSum / float64(len(logits)),
	}
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
