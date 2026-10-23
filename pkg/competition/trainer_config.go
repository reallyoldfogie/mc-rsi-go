package competition

import (
	"fmt"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
)

// ResolveTrainerSettings loads the common or competitor-specific cRL-go
// settings, then applies the architecture fields owned by the competition
// definition. Architecture selection must not depend on whichever values a
// shared trainer config happened to contain.
func ResolveTrainerSettings(cfg Config, competitor Competitor) (crlconfig.Settings, error) {
	if err := cfg.Validate(); err != nil {
		return crlconfig.Settings{}, err
	}
	var path = cfg.TrainerConfig
	if competitor.TrainerConfig != "" {
		path = competitor.TrainerConfig
	}
	settings := crlconfig.Default()
	if path != "" {
		loaded, err := crlconfig.Load(path)
		if err != nil {
			return crlconfig.Settings{}, fmt.Errorf("loading trainer config for %q: %w", competitor.ID, err)
		}
		settings = loaded
	}
	settings.HiddenSize = competitor.HiddenSize
	if competitor.Recurrent {
		settings.RecurrentChunkLen = competitor.RecurrentChunkLen
	} else {
		settings.RecurrentChunkLen = 0
	}
	if competitor.TrainingSeed != 0 {
		settings.Seed = competitor.TrainingSeed
	}
	if err := settings.Validate(); err != nil {
		return crlconfig.Settings{}, fmt.Errorf("trainer settings for %q: %w", competitor.ID, err)
	}
	return settings, nil
}
