module github.com/reallyoldfogie/mc-rsi-trainer

go 1.27.0

// Temporary, local-only: testing/mcserver.go needs mc-client-test-go's
// ServerConfig.HostServerPort/HostRCONPort/RCONPassword fields (fixed-port
// + fixed-password server launch, for "find it again on a later run"),
// added but not yet tagged/pushed as of 2026-09-14. Remove this replace
// once a new mc-client-test-go tag exists and bump the require below to
// it — matching exactly how mc-agent's own go.mod handled this same
// situation for its cRL-go/mc-bot-go dependencies before they were tagged.
// replace github.com/reallyoldfogie/mc-client-test-go => ../mc-client-test-go

// Temporary, local-only: pkg/curriculum/rlenvadapter needs
// rlenv.Config.TaskSelector/TaskOverride (docs/plans/06-per-episode-task-selection-and-goal-conditioning.md),
// added to mc-agent's rlenv package but not yet tagged/pushed past v0.0.1
// as of 2026-09-14 — the first mc-rsi-trainer code to actually reference
// them (docs/plans/05's and 06's own work happened entirely inside
// mc-agent's own checkout, so this gap wasn't hit until now). Remove this
// replace once a new mc-agent tag exists and bump the require below to
// it.
// replace github.com/reallyoldfogie/mc-agent => ../mc-agent

require (
	github.com/moby/moby/client v0.2.1
	github.com/reallyoldfogie/cRL-go v0.11.1
	github.com/reallyoldfogie/mc-agent v0.0.2
	github.com/reallyoldfogie/mc-bot-go v0.2.4
	github.com/reallyoldfogie/mc-client-test-go v0.1.1
	github.com/stretchr/testify v1.12.1
)

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/Tnze/go-mc v1.20.3-0.20240907175330-9a1f5431370e // indirect
	github.com/aquasecurity/go-version v0.0.1 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.1 // indirect
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
	github.com/ollama/ollama v0.20.7 // indirect
	github.com/opencontainers/go-digest v1.0.0 // indirect
	github.com/opencontainers/image-spec v1.1.1 // indirect
	github.com/pkg/errors v0.9.1 // indirect
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
	golang.org/x/crypto v0.54.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/xerrors v0.0.0-20231012003039-104605ab7028 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.2.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
