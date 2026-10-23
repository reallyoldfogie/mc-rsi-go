package competition

import "math"

// SummarizeRewards computes descriptive statistics for one competitor's
// episode rewards. The interval is a two-sided 95% Student-t confidence
// interval for the mean, which is more appropriate than a normal interval for
// the relatively small evaluation samples used by smoke competitions.
func SummarizeRewards(rewards []float64) EvaluationSummary {
	if len(rewards) == 0 {
		return EvaluationSummary{}
	}
	values := append([]float64(nil), rewards...)
	var sum float64
	for _, reward := range values {
		sum += reward
	}
	mean := sum / float64(len(values))
	var squared float64
	for _, reward := range values {
		delta := reward - mean
		squared += delta * delta
	}
	stddev := 0.0
	if len(values) > 1 {
		stddev = math.Sqrt(squared / float64(len(values)-1))
	}
	margin := 0.0
	if len(values) > 1 {
		margin = studentT95(len(values)-1) * stddev / math.Sqrt(float64(len(values)))
	}
	return EvaluationSummary{
		Episodes:        len(values),
		EpisodeRewards:  values,
		MeanReward:      mean,
		RewardStdDev:    stddev,
		ConfidenceLevel: 0.95,
		ConfidenceLower: mean - margin,
		ConfidenceUpper: mean + margin,
	}
}

// AggregateEvaluationSummaries combines all episode samples from independent
// evaluation runs. Completion is weighted by the number of episodes in each
// run rather than averaged equally when run sizes differ.
func AggregateEvaluationSummaries(summaries []EvaluationSummary) EvaluationSummary {
	var rewards []float64
	var tasks, reasons []string
	var completed float64
	var episodes int
	for _, summary := range summaries {
		rewards = append(rewards, summary.EpisodeRewards...)
		tasks = append(tasks, summary.EpisodeTasks...)
		reasons = append(reasons, summary.EpisodeTerminationReasons...)
		if len(summary.EpisodeRewards) == 0 && summary.Episodes > 0 {
			episodes += summary.Episodes
			completed += summary.CompletionRate * float64(summary.Episodes)
		} else {
			episodes += len(summary.EpisodeRewards)
			completed += summary.CompletionRate * float64(len(summary.EpisodeRewards))
		}
	}
	aggregate := SummarizeRewards(rewards)
	aggregate.EpisodeTasks = tasks
	aggregate.EpisodeTerminationReasons = reasons
	if len(rewards) == 0 {
		aggregate.Episodes = episodes
	}
	if episodes > 0 {
		aggregate.CompletionRate = completed / float64(episodes)
	}
	return aggregate
}

// studentT95 returns a conservative two-sided 95% critical value. The
// tabulated values cover the small samples where the distinction from 1.96
// matters most; the larger-sample buckets remain conservative. Values are
// kept at full precision rather than the 3-decimal textbook rounding so the
// resulting confidence interval does not carry avoidable rounding error.
func studentT95(degreesOfFreedom int) float64 {
	if degreesOfFreedom <= 0 {
		return 0
	}
	critical := [...]float64{
		0,
		12.706204736174694, 4.302652729749462, 3.1824463052837078, 2.7764451051977934,
		2.5705818356363146, 2.4469118511449786, 2.364624251592784, 2.306004135204166,
		2.262157162798205, 2.228138851986274, 2.200985160091639, 2.1788128296672284,
		2.1603686564627913, 2.144786687917804, 2.131449545559776, 2.1199052992212546,
		2.1098155778333156, 2.1009220402410382, 2.0930240544083087, 2.085963447265864,
		2.0796138447276795, 2.0738730679040254, 2.0686576104190486, 2.0638985616280245,
		2.0595385527532972, 2.0555294386428735, 2.0518305164802846, 2.0484071417952454,
		2.045229642132703, 2.0422724563012378,
	}
	if degreesOfFreedom < len(critical) {
		return critical[degreesOfFreedom]
	}
	switch {
	case degreesOfFreedom <= 40:
		return 2.021075390306273
	case degreesOfFreedom <= 60:
		return 2.0002978220142604
	case degreesOfFreedom <= 120:
		return 1.9799304050824402
	default:
		return 1.959963984540054
	}
}
