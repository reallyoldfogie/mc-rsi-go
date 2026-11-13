package main

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// maskedProbabilities returns probs restricted to mask's allowed
// entries, renormalized to sum to 1 — reimplementing
// cRL-go/pkg/reinforce.SampleMaskedActionWithProbabilities's own
// renormalization locally (including its uniform-among-allowed
// fallback for the rare case where every allowed action's probability
// has underflowed to exactly 0 in float32 — live-confirmed in that
// package, not a theoretical concern), since this standalone binary
// deliberately imports no cRL-go training package — see
// docs/plans/23-ONNX-support.md §1. A nil mask, or one where every
// entry is true, returns a copy of probs unchanged.
func maskedProbabilities(probs []float32, mask []bool) ([]float32, error) {
	if mask == nil {
		return append([]float32(nil), probs...), nil
	}
	if len(mask) != len(probs) {
		return nil, fmt.Errorf("mask length %d does not match action space %d", len(mask), len(probs))
	}

	allAllowed := true
	anyAllowed := false
	var maskedSum float32
	for i, allowed := range mask {
		if allowed {
			anyAllowed = true
			maskedSum += probs[i]
		} else {
			allAllowed = false
		}
	}
	if allAllowed {
		return append([]float32(nil), probs...), nil
	}
	if !anyAllowed {
		return nil, fmt.Errorf("no legal action: mask excludes every action")
	}

	renormalized := make([]float32, len(probs))
	if maskedSum <= 0 {
		count := 0
		for _, allowed := range mask {
			if allowed {
				count++
			}
		}
		uniform := float32(1) / float32(count)
		for i, allowed := range mask {
			if allowed {
				renormalized[i] = uniform
			}
		}
		return renormalized, nil
	}

	for i, allowed := range mask {
		if allowed {
			renormalized[i] = probs[i] / maskedSum
		}
	}
	return renormalized, nil
}

// sampleAction draws an action from the categorical distribution probs
// (already mask-renormalized by maskedProbabilities, if masking
// applies) — the same cumulative-probability draw as
// cRL-go/pkg/reinforce.SampleAction, reimplemented locally for the same
// reason maskedProbabilities is.
func sampleAction(probs []float32, rng *rand.Rand) rl.Action {
	sample := rng.Float32()
	var cumulative float32
	for i, p := range probs {
		cumulative += p
		if sample <= cumulative {
			return rl.Action(i)
		}
	}
	return rl.Action(len(probs) - 1)
}

// greedyAction returns the index of the highest probability in probs,
// breaking ties toward the lowest index — identical to
// cmd/rsi-infer's own greedyAction, duplicated here per this repo's
// existing per-command convention.
func greedyAction(probs []float32) rl.Action {
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return rl.Action(best)
}
