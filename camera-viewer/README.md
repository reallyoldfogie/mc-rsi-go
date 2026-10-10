# Camera Viewer

A live 3D "ride along" camera for one `rsi-train` bot at a time, rendered in any browser. This is
the Phase 2 companion to the agent dashboard's 2D plot ([`../cmd/rsi-train/dashboard.html`](../cmd/rsi-train/dashboard.html)):
that view shows where every bot is from above; this shows what one specific bot is actually
seeing, in 3D, as it trains.

It runs two ways:

- **Integrated** — the dashboard's own "Switch to 3D" button (next to a selected agent's stats)
  manages this tool's container for you; the browser only ever talks to `rsi-train` itself. See
  "Running it via the dashboard (integrated mode)" below.
- **Standalone** — run it yourself, locally or in Docker, independent of `rsi-train` entirely. See
  "Running it locally" / "Docker or local" below. This is also how the integrated mode's own
  container gets built — both modes run the exact same tool.

## How it works

This is a small Node.js tool, not a Go change to `rsi-train` itself — it connects to the same
Minecraft server as a completely independent client, the same way any other Minecraft bot would:

1. It logs in as one extra player (default username `RSICameraViewer`) using
   [mineflayer](https://github.com/PrismarineJS/mineflayer), the same kind of library `mc-agent`'s
   Go bots are built on, just in JavaScript.
2. Over RCON, it puts that player into spectator mode (`gamemode spectator <name>`) and then runs
   Minecraft's own vanilla `/spectate <target> <name>` command. `/spectate` is a real server
   feature (added in 1.14, originally for players spectating each other), but **it only snaps the
   spectating player near the target once** — found live, the hard way: continuous camera-follow
   from `/spectate` is a *real-game-client* rendering trick (the client keeps reading the target's
   own synced movement every frame and draws from there), not a continuous server-side re-teleport
   of the spectator's own entity. mineflayer has no renderer, so it never receives whatever would
   make that client-side trick visible to it — `bot.entity.position` (what prismarine-viewer
   actually renders from) sat frozen for 30+ seconds in testing even while the target kept moving.
   `startMirroring` (`index.js`) is the actual fix: once `/spectate` has snapped us near the target
   (close enough for its entity to become "tracked," the same way any nearby entity would be),
   mineflayer *does* receive that entity's own ordinary movement packets continuously (confirmed:
   dozens of `entityMoved` events a second) — `startMirroring` listens for those, copies the
   target's live position/yaw/pitch onto our own `bot.entity`, and re-emits `move` so prismarine-
   viewer's own existing position-broadcast logic (unchanged, see step 3) picks it up as if it were
   our own movement. The periodic RCON re-`/spectate` (`setFollow`, every 10s by default) still
   matters alongside this: it's what keeps our spectator's *real*, server-tracked position (and
   therefore which chunks the server bothers streaming to it) in the target's neighborhood as it
   roams — the mirror only ever affects what we render from chunks already cached locally, it has
   no way to make the server send new ones on its own.
3. [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) is attached to that same
   mineflayer connection, completely unaware of any of the above — it renders from `bot.entity`
   exactly like it always does, and startMirroring's whole job is making sure that keeps reflecting
   the target rather than us. It meshes the visible terrain from the ordinary stream of chunk/
   entity packets the server sends any nearby client, and serves a three.js-based 3D page over a
   small built-in HTTP+WebSocket server. Open that page in a browser and you're looking through
   (approximately) the spectated bot's own eyes, live.
4. A tiny HTTP control API (`GET /status`, `POST /follow`) lets you change which bot is being
   followed without restarting the process — see "Switching who it's watching" below. This is the
   seam the dashboard's own integrated mode calls into (see "Running it via the dashboard" below).

No change to `mc-agent`, `rlenv`, or any training code was needed — the trainee bots have no idea
this extra spectator client exists. It's the same reason `mc-agent`'s own replay recorder (which
inspired this prototype) can capture a full `.mcpr` recording without the bot it's recording being
aware of it: everything a nearby client needs to render the world is already broadcast over the
ordinary Minecraft protocol to anyone connected and looking.

## Running it locally

```
cd camera-viewer
npm install
MC_HOST=127.0.0.1 MC_PORT=34700 MC_VERSION=1.21.5 \
RCON_HOST=127.0.0.1 RCON_PORT=25575 RCON_PASSWORD='<your rcon password>' \
FOLLOW=RSIFull-0 \
npm start
```

Then open `http://localhost:3001` in a browser. Point `MC_HOST`/`MC_PORT`/`MC_VERSION` at whatever
your `-mc-agent-config`'s `connection.address`/`connection.version` is, and `RCON_HOST`/`RCON_PORT`/
`RCON_PASSWORD` at that same config's `rcon.address`/`rcon.password`. `FOLLOW` is the in-game
username of the bot to start watching (e.g. `RSIFull-0` for a `-shared-server` run's first bot) —
leave it unset to connect without a target and pick one later via the control API.

All configuration is environment variables (see the top of `index.js` for the full list and
defaults): `MC_USERNAME` (this spectator's own player name, must be ≤16 characters — Minecraft
usernames longer than that fail to log in with a confusing protocol decode error, not a clear
"name too long"), `VIEWER_PORT` (default `3001`), `CONTROL_PORT` (default `3002`), `FIRST_PERSON`
(default `true`), `RESPECTATE_INTERVAL_MS` (default `10000`), `VIEWER_PREFIX` (default `""` — set
only when something reverse-proxies the 3D page under a sub-path; see `index.js`'s own doc comment
on why the value must exactly match the path the browser loads the page at. The integrated mode
below sets this for you).

### Switching who it's watching

```
curl -X POST http://localhost:3002/follow -H 'content-type: application/json' -d '{"bot":"RSIFull-3"}'
curl http://localhost:3002/status
```

### Stopping it

`Ctrl+C` (or `docker stop` / `SIGTERM` in a container) — the shutdown handler disconnects the
spectator client and closes both HTTP servers cleanly, so the extra player slot and RCON
connection don't linger. It leaves the server in exactly the state it found it.

## Running it via the dashboard (integrated mode)

Build the image once (`rsi-train` never builds it for you — same reasoning as
`EnsureServer` pulling `itzg/minecraft-server` but never building a custom image):

```
docker build -t mc-rsi-camera-viewer:latest camera-viewer/
```

Then run `rsi-train` with `-camera-viewer-enabled` (see `cmd/rsi-train/cameraviewer.go`):

```
./rsi-train -mc-agent-config ./my-config.json -checkpoint-dir ./checkpoints -camera-viewer-enabled
```

Open the dashboard (`/dashboard`), select an agent, and click "Switch to 3D" in the stats header.
`rsi-train` itself now manages this tool's container end to end:

- It starts the container (`--network host`, the same networking choice "Docker or local" below
  explains) the first time you switch to 3D, with `MC_HOST`/`MC_PORT`/`MC_VERSION`/`RCON_HOST`/
  `RCON_PORT`/`RCON_PASSWORD` filled in from the same `-mc-agent-config` the trainer itself is
  already using — nothing to configure separately.
- It reverse-proxies the 3D page itself (HTML/JS/WebSocket) through the dashboard's own HTTP
  server, under `/camera/view/`, and translates the dashboard's `POST /camera/follow?bot=<name>` /
  `POST /camera/stop` into this tool's own control API calls. **The browser only ever talks to
  `rsi-train`** — it has no idea a separate container, or this tool at all, is involved.
- Switching which agent is selected while still in 3D mode just re-targets the same running
  container (a cheap control-API call) — the container, and your browser's WebSocket connection to
  it, both stay up.
- Switching back to 2D (or closing the browser tab — see `dashboard.html`'s `beforeunload`
  handler) stops and removes the container. It only runs while something is actually watching.

`VIEWER_PORT`/`CONTROL_PORT` are fixed at `3001`/`3002` in this mode (`cmd/rsi-train/
cameraviewer.go`'s own constants) — don't also run the standalone tool on the same host at the
same time, they'll collide on those ports.

**If the image hasn't been built yet** (or Docker itself isn't reachable): `rsi-train` checks for
it once at startup and logs a warning (`camera-viewer: image "..." not found locally — build it
once with: docker build -t ...`) without refusing to start or affecting training at all — the 3D
view is a dashboard nicety, not something worth failing a training run over. The same check runs
again, for real, the first time anyone actually clicks "Switch to 3D": `followHandler` returns
that exact message as an HTTP 502, which the dashboard shows right under the toggle button (and
reverts to 2D) rather than failing silently.

## Docker or local — which do I need?

**Either works for the Minecraft/RCON connection itself — this tool is a pure network client, no
filesystem or process access to `rsi-train` required at all.** It doesn't need to run on the same
machine as the trainer, let alone in the same process; it just needs TCP reachability to the MC
server's game port and its RCON port, exactly like any other Minecraft client. A prebuilt Docker
image is genuinely the more portable way to ship this (see the Dockerfile's own comment on why
`canvas`, prismarine-viewer's one native dependency, needs a glibc-based image like `node:22-slim`
rather than `alpine`/musl).

**The one real wrinkle, found by actually testing both ways while prototyping this:** `rsi-train`'s
own `EnsureServer` (and the `launch.sh` scripts this repo's live runs use) deliberately bind RCON
to `127.0.0.1` only on the host, not `0.0.0.0` — reasonable, since RCON is full unauthenticated-by-
IP admin access once you have the password. That means a container on the **default Docker bridge
network** cannot reach it: `host.docker.internal`/`host-gateway` resolves to the bridge gateway IP
(e.g. `172.17.0.1`), and a `127.0.0.1`-bound listener on the host doesn't accept connections
arriving from that address — only from literal loopback. (The Minecraft game port itself isn't
affected, since `itzg/minecraft-server` publishes it on `0.0.0.0`; only RCON is loopback-only.)

Two ways to actually run this in Docker against a `127.0.0.1`-bound RCON, both verified against the
live training server while building this prototype:

- **`--network host`** (Linux only — this is how it was tested): the container shares the host's
  network namespace directly, so `127.0.0.1` inside the container *is* the host's loopback and
  reaches RCON exactly like a local process would. Simplest option on Linux/WSL2, which is this
  project's own deployment target.
  ```
  docker build -t mc-rsi-camera-viewer .
  docker run --network host \
    -e MC_HOST=127.0.0.1 -e MC_PORT=34700 -e MC_VERSION=1.21.5 \
    -e RCON_HOST=127.0.0.1 -e RCON_PORT=25575 -e RCON_PASSWORD='<your rcon password>' \
    -e FOLLOW=RSIFull-0 \
    mc-rsi-camera-viewer
  ```
- **Rebind RCON to `0.0.0.0`** on the Minecraft server side and use ordinary bridge networking
  (`-p 3001:3001 -p 3002:3002`, `MC_HOST`/`RCON_HOST=host.docker.internal` with
  `--add-host=host.docker.internal:host-gateway`). Works on Docker Desktop (Mac/Windows) too, where
  `--network host` isn't available, but widens RCON's own exposure — only do this if you also
  firewall the RCON port from anything but trusted hosts.

If you're running this on the exact same Linux/WSL2 host as the trainer (the common case for this
project), **running it as a plain local Node process (`npm start`, no Docker) is the least ceremony**
and was this prototype's first and simplest verified path — reach for Docker when you want to ship
this to a different machine, or isolate its dependencies from the host.

## Current limitations

- One spectator client follows one bot at a time. In integrated mode, the dashboard can only show
  one agent's 3D view at once (matching "select one agent" being the whole premise of that mode).
  Standalone, watching several bots at once means running several instances (distinct
  `MC_USERNAME`/`VIEWER_PORT`/`CONTROL_PORT`/`VIEWER_PREFIX` each).
- No auth on the control API or the viewer page itself — fine for `localhost`-only use (this
  project's whole dashboard story so far, including the integrated mode's own reverse proxy), not
  for exposing beyond that without adding some.
- `RESPECTATE_INTERVAL_MS`'s periodic re-issue is a blunt safety net, not a real "did spectate
  actually stick" check — it re-sends `/spectate` on a timer regardless of whether anything went
  wrong.
- Integrated mode's `cameraViewerManager` (`cmd/rsi-train/cameraviewer.go`) has an opt-in, env-var-
  gated live test (`cmd/rsi-train/cameraviewer_test.go`,
  `TestCameraViewerManagerLive`) rather than one that runs by default — same pattern as every other
  live test in that package (`MC_RSI_TRAINER_LIVE_CONFIG`), since it needs Docker and the image
  already built.
