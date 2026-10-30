module github.com/reallyoldfogie/mc-rsi-trainer

go 1.27.3

exclude (
	github.com/buger/jsonparser v1.1.1
	golang.org/x/crypto v0.54.0
)

// Temporary local replace for the Sleep Action Plan's cross-repo wiring
// (docs/plans/SLEEP_ACTION_PLAN.md, local/untracked on both sides) -
// mc-agent's own sleep-action branch merged to its local master
// (2026-10-29) but isn't tagged/published yet, and that branch's own
// worktree was deleted once merged there, so this points at the main
// checkout directly rather than the now-gone worktree path. Remove once
// a tagged mc-agent release includes Sleep, same convention
// combat-competition's own go.mod used for an identical reason.
replace github.com/reallyoldfogie/mc-agent => /home/reallyoldfogie/src/github.com/reallyoldfogie/mc-agent

require (
	github.com/moby/moby/client v0.2.1
	github.com/prometheus/client_golang v1.24.0
	github.com/reallyoldfogie/cRL-go v0.15.0
	github.com/reallyoldfogie/mc-agent v0.0.52
	github.com/reallyoldfogie/mc-bot-go v0.2.7
	github.com/reallyoldfogie/mc-client-test-go v0.2.0
	github.com/stretchr/testify v1.12.1
)

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/Tnze/go-mc v1.20.3-0.20240907175330-9a1f5431370e // indirect
	github.com/aquasecurity/go-version v0.0.1 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/buger/jsonparser v1.6.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/containerd/errdefs v1.0.0 // indirect
	github.com/containerd/errdefs/pkg v0.3.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/distribution/reference v0.6.0 // indirect
	github.com/docker/go-connections v0.6.0 // indirect
	github.com/docker/go-units v0.5.0 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/gorcon/rcon v1.4.0 // indirect
	github.com/iancoleman/strcase v0.2.0 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/maxsupermanhd/go-mc-ms-auth v0.0.0-20230820124717-22f4d907eac4 // indirect
	github.com/moby/docker-image-spec v1.3.1 // indirect
	github.com/moby/moby/api v1.52.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/ollama/ollama v0.40.8 // indirect
	github.com/opencontainers/go-digest v1.0.0 // indirect
	github.com/opencontainers/image-spec v1.1.1 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.70.0 // indirect
	github.com/prometheus/procfs v0.21.1 // indirect
	github.com/reallyoldfogie/mc-data-gen/loader v0.0.4 // indirect
	github.com/reallyoldfogie/mc-protocol-go v0.1.0 // indirect
	github.com/reallyoldfogie/mc-replay-go v0.0.5 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.69.0 // indirect
	go.opentelemetry.io/otel v1.44.0 // indirect
	go.opentelemetry.io/otel/metric v1.44.0 // indirect
	go.opentelemetry.io/otel/trace v1.44.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/crypto v0.56.0 // indirect
	golang.org/x/sys v0.48.0 // indirect
	golang.org/x/xerrors v0.0.0-20231012003039-104605ab7028 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.2.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
