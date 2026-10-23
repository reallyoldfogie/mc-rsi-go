package competition

import (
	"strings"
	"testing"
	"time"
)

func TestFormatResultsTableIncludesCompletedAndFailedCompetitors(t *testing.T) {
	result := CompetitionResult{
		StartedAt: time.Time{},
		Competitors: []CompetitorResult{
			{Identity: Identity{CompetitorID: "basic", DisplayName: "Basic"}, Training: TrainingSummary{Epochs: 12, Samples: 100}, Evaluation: EvaluationSummary{Episodes: 4, MeanReward: 2.5, CompletionRate: .25}, Status: "complete"},
			{Identity: Identity{CompetitorID: "wide", DisplayName: "Wide"}, Training: TrainingSummary{Epochs: 2, Samples: 10}, Status: "training_failed", Error: "connection lost"},
		},
	}
	table := FormatResultsTable(result)
	for _, want := range []string{"Basic", "Wide", "complete", "training_failed", "2.500", "25.0%", "connection lost"} {
		if !strings.Contains(table, want) {
			t.Fatalf("results table missing %q:\n%s", want, table)
		}
	}
}

func TestFormatResultsTableShowsPartialStatsForATimeLimitedCompetitorButExcludesItFromRanking(t *testing.T) {
	result := CompetitionResult{Competitors: []CompetitorResult{
		{Identity: Identity{CompetitorID: "basic", DisplayName: "Basic"}, Status: "complete", Evaluation: EvaluationSummary{Episodes: 60, MeanReward: 30}},
		{Identity: Identity{CompetitorID: "slow", DisplayName: "Slow"}, Status: "evaluation_time_limit_exceeded", Evaluation: EvaluationSummary{Episodes: 20, MeanReward: 5.777}, Error: "context deadline exceeded"},
	}}
	table := FormatResultsTable(result)
	for _, want := range []string{"Slow", "evaluation_time_limit_exceeded", "5.777", "context deadline exceeded"} {
		if !strings.Contains(table, want) {
			t.Fatalf("results table missing %q (partial evaluation data should still show):\n%s", want, table)
		}
	}
	if strings.Contains(table, "1     Slow") {
		t.Fatalf("a time-limited competitor's partial episode count must not compete for rank:\n%s", table)
	}
}

func TestFormatResultsTableSortsDescendingByWinnerRank(t *testing.T) {
	result := CompetitionResult{Competitors: []CompetitorResult{
		{Identity: Identity{CompetitorID: "high", DisplayName: "High"}, Status: "complete", Evaluation: EvaluationSummary{Episodes: 1, MeanReward: 20}},
		{Identity: Identity{CompetitorID: "low", DisplayName: "Low"}, Status: "complete", Evaluation: EvaluationSummary{Episodes: 1, MeanReward: 2}},
	}}
	table := FormatResultsTable(result)
	low := strings.Index(table, "Low")
	high := strings.Index(table, "High")
	if low < 0 || high < 0 || high > low {
		t.Fatalf("expected descending reward order, got:\n%s", table)
	}
	if !strings.Contains(table, "1     High") {
		t.Fatalf("expected highest reward to receive rank 1, got:\n%s", table)
	}
}
