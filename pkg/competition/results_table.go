package competition

import (
	"fmt"
	"sort"
	"strings"
)

// FormatResultsTable renders a stable, human-readable scoreboard for the end
// of a competition. Failed or skipped competitors remain in the table so a
// partial run is immediately distinguishable from a complete tournament.
func FormatResultsTable(result CompetitionResult) string {
	type ranked struct {
		index int
		value float64
	}
	ranks := make([]ranked, 0, len(result.Competitors))
	for i, competitor := range result.Competitors {
		if competitor.Status == "complete" && competitor.Evaluation.Episodes > 0 {
			ranks = append(ranks, ranked{index: i, value: competitor.Evaluation.MeanReward})
		}
	}
	// Winner ranking puts the highest mean reward first. Stable sorting keeps
	// the original configuration order for exact ties.
	sort.SliceStable(ranks, func(i, j int) bool { return ranks[i].value > ranks[j].value })
	rankByIndex := make(map[int]int, len(ranks))
	rowOrder := make([]int, 0, len(result.Competitors))
	for rank, item := range ranks {
		rankByIndex[item.index] = rank + 1
		rowOrder = append(rowOrder, item.index)
	}
	for i := range result.Competitors {
		if _, ranked := rankByIndex[i]; !ranked {
			rowOrder = append(rowOrder, i)
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "\nOverall competition results (%s)\n", result.StartedAt.Format("2006-01-02 15:04:05 MST"))
	fmt.Fprintln(&b, "Rank  Competitor                 Status             Epochs  Samples  Episodes  Mean reward  Std dev  95% CI             Complete")
	fmt.Fprintln(&b, "----  -------------------------  -----------------  ------  -------  --------  -----------  -------  -----------------  --------")
	for _, i := range rowOrder {
		competitor := result.Competitors[i]
		rank := "-"
		if value, ok := rankByIndex[i]; ok {
			rank = fmt.Sprintf("%d", value)
		}
		mean, stddev, interval, completion, episodes := "-", "-", "-", "-", "-"
		// evaluation_time_limit_exceeded still carries whatever plans did
		// complete before the time limit (see pkg/competition.Run) - worth
		// showing, even though it's excluded from ranking above since its
		// episode count may be lower than a fully-evaluated competitor's.
		if competitor.Status == "complete" || competitor.Status == "evaluation_time_limit_exceeded" {
			mean = fmt.Sprintf("%.3f", competitor.Evaluation.MeanReward)
			stddev = fmt.Sprintf("%.3f", competitor.Evaluation.RewardStdDev)
			interval = fmt.Sprintf("[%.3f, %.3f]", competitor.Evaluation.ConfidenceLower, competitor.Evaluation.ConfidenceUpper)
			completion = fmt.Sprintf("%.1f%%", competitor.Evaluation.CompletionRate*100)
			episodes = fmt.Sprintf("%d", competitor.Evaluation.Episodes)
		}
		name := competitor.Identity.DisplayName
		if name == "" {
			name = competitor.Identity.CompetitorID
		}
		fmt.Fprintf(&b, "%-4s  %-25s  %-17s  %6d  %7d  %8s  %11s  %7s  %-17s  %8s\n",
			rank, name, competitor.Status, competitor.Training.Epochs,
			competitor.Training.Samples, episodes, mean, stddev, interval, completion)
		if competitor.Error != "" {
			fmt.Fprintf(&b, "      %s: %s\n", competitor.Identity.CompetitorID, competitor.Error)
		}
	}
	if len(ranks) == 0 {
		fmt.Fprintln(&b, "No competitor completed evaluation; no ranking is available.")
	}
	for index, run := range result.EvaluationRuns {
		fmt.Fprintf(&b, "\nEvaluation run %d (seed %d)\n", index+1, run.Seed)
		fmt.Fprintln(&b, "Rank  Competitor                 Status             Episodes  Mean reward  Std dev  95% CI")
		fmt.Fprintln(&b, "----  -------------------------  -----------------  --------  -----------  -------  -----------------")
		runRanks := make([]ranked, 0, len(run.Competitors))
		for i, competitor := range run.Competitors {
			if competitor.Status == "complete" && competitor.Evaluation.Episodes > 0 {
				runRanks = append(runRanks, ranked{index: i, value: competitor.Evaluation.MeanReward})
			}
		}
		sort.SliceStable(runRanks, func(i, j int) bool { return runRanks[i].value > runRanks[j].value })
		for rank, item := range runRanks {
			competitor := run.Competitors[item.index]
			name := competitor.Identity.DisplayName
			if name == "" {
				name = competitor.Identity.CompetitorID
			}
			fmt.Fprintf(&b, "%-4d  %-25s  %-17s  %8d  %11.3f  %7.3f  [%.3f, %.3f]\n",
				rank+1, name, competitor.Status, competitor.Evaluation.Episodes,
				competitor.Evaluation.MeanReward, competitor.Evaluation.RewardStdDev,
				competitor.Evaluation.ConfidenceLower, competitor.Evaluation.ConfidenceUpper)
		}
	}
	return b.String()
}
