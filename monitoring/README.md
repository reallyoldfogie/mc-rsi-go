# Monitoring

A Prometheus + Grafana stack for watching an `rsi-train` run without a shell attached to it. Start
it once; it keeps working across every future training run, since `rsi-train` itself exposes the
metrics — this stack just scrapes and graphs them.

## Start it

```
cd monitoring
docker compose up -d
```

This starts two containers (Prometheus on `localhost:9090`, Grafana on `localhost:3000`) with a
Grafana dashboard ("RSI Curriculum Training") and its Prometheus datasource already provisioned —
nothing to click through by hand. Grafana's anonymous-admin login is enabled for local,
single-operator use (see `docker-compose.yml`'s own comment if you ever expose port 3000 beyond
`localhost`).

Stop it with `docker compose down` (add `-v` to also drop the persisted Prometheus/Grafana data
volumes — normally you don't want that, since it throws away history from past runs).

## Point it at a training run

`rsi-train` serves Prometheus metrics on `-metrics-addr` (default `:9400`) automatically — nothing
extra to pass beyond what [`../docs/getting-started.md`](../docs/getting-started.md) already
describes for running a training session:

```
./rsi-train -checkpoint-dir ./checkpoints -mc-agent-config ./my-config.json
```

`monitoring/prometheus.yml` already points at `host.docker.internal:9400`, which reaches
`rsi-train` running as a normal process on the same machine as Docker (this works the same way on
Docker Desktop and on plain Docker Engine/WSL2 — see `docker-compose.yml`'s `extra_hosts`). If you
run several training processes at once, give each a distinct `-metrics-addr` port and add a
`static_configs` entry per port in `prometheus.yml` (restart Prometheus after editing it:
`docker compose restart prometheus`).

Pass `-metrics-addr ""` to disable metrics entirely for a given run.

## View it

Open `http://localhost:3000` — the dashboard loads directly, no login needed. See
[`../docs/glossary.md`](../docs/glossary.md)'s "Metrics" section for what every panel and metric
name actually means, and each panel's own description (hover the "i" in its top-left corner) for
the same text inline.

## Files here

- `docker-compose.yml` — the two services (Prometheus, Grafana) and their volumes.
- `prometheus.yml` — Prometheus's scrape config (what to scrape, how often).
- `grafana/provisioning/datasources/` — auto-registers the Prometheus datasource on Grafana startup.
- `grafana/provisioning/dashboards/` — tells Grafana to load whatever's in `grafana/dashboards/`.
- `grafana/dashboards/rsi-training.json` — the dashboard itself. Edit it in the Grafana UI and
  export the JSON back over this file to persist changes (UI edits alone don't survive
  `docker compose down -v`).
