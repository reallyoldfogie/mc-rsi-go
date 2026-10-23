package competition

import (
	"math"
	"testing"
)

// confidenceIntervalTolerance allows for last-bit differences between the
// Student-t critical value table and an independently computed reference
// interval; it is far tighter than anything that would matter for comparing
// competition results.
const confidenceIntervalTolerance = 1e-6

func TestSummarizeRewardsIncludesSamplesAndConfidenceInterval(t *testing.T) {
	summary := SummarizeRewards([]float64{1, 2, 3, 4, 5})
	if summary.Episodes != 5 || len(summary.EpisodeRewards) != 5 {
		t.Fatalf("unexpected reward samples: %+v", summary)
	}
	if summary.MeanReward != 3 || summary.RewardStdDev != 1.5811388300841898 {
		t.Fatalf("unexpected statistics: %+v", summary)
	}
	if summary.ConfidenceLevel != .95 || summary.ConfidenceLower >= summary.MeanReward || summary.ConfidenceUpper <= summary.MeanReward {
		t.Fatalf("expected a two-sided confidence interval: %+v", summary)
	}
	const wantLower, wantUpper = 1.036756838522439, 4.963243161477561
	if math.Abs(summary.ConfidenceLower-wantLower) > confidenceIntervalTolerance ||
		math.Abs(summary.ConfidenceUpper-wantUpper) > confidenceIntervalTolerance {
		t.Fatalf("unexpected 95%% confidence interval: %+v", summary)
	}
}

func TestSummarizeRewardsDoesNotAliasInput(t *testing.T) {
	rewards := []float64{2, 4}
	summary := SummarizeRewards(rewards)
	rewards[0] = 99
	if summary.EpisodeRewards[0] != 2 {
		t.Fatalf("episode rewards should be copied: %+v", summary.EpisodeRewards)
	}
}

func TestAggregateEvaluationSummariesConcatenatesEpisodeTasksAndReasonsInOrder(t *testing.T) {
	first := EvaluationSummary{
		EpisodeRewards:            []float64{1, 2},
		EpisodeTasks:              []string{"goto", "mine"},
		EpisodeTerminationReasons: []string{"arrived", "mined"},
	}
	second := EvaluationSummary{
		EpisodeRewards:            []float64{3},
		EpisodeTasks:              []string{"craft"},
		EpisodeTerminationReasons: []string{"timeout"},
	}
	aggregate := AggregateEvaluationSummaries([]EvaluationSummary{first, second})
	if got, want := aggregate.EpisodeTasks, []string{"goto", "mine", "craft"}; !equalStrings(got, want) {
		t.Fatalf("unexpected episode tasks: got %v want %v", got, want)
	}
	if got, want := aggregate.EpisodeTerminationReasons, []string{"arrived", "mined", "timeout"}; !equalStrings(got, want) {
		t.Fatalf("unexpected episode termination reasons: got %v want %v", got, want)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
