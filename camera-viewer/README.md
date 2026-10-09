# Camera Viewer (prototype)

A live 3D "ride along" camera for one `rsi-train` bot at a time, rendered in any browser. This is
the Phase 2 prototype for the agent dashboard's 2D plot ([`../cmd/rsi-train/dashboard.html`](../cmd/rsi-train/dashboard.html)):
that view shows where every bot is from above; this shows what one specific bot is actually
seeing, in 3D, as it trains.

## How it works

This is a small Node.js tool, not a Go change to `rsi-train` itself — it connects to the same
Minecraft server as a completely independent client, the same way any other Minecraft bot would:

1. It logs in as one extra player (default username `RSICameraViewer`) using
   [mineflayer](https://github.com/PrismarineJS/mineflayer), the same kind of library `mc-agent`'s
   Go bots are built on, just in JavaScript.
2. Over RCON, it puts that player into spectator mode (`gamemode spectator <name>`) and then runs
   Minecraft's own vanilla `/spectate <target> <name>` command. `/spectate` is a real server
   feature (added in 1.14, originally for players spectating each other) that locks the camera
   that player's client receives onto the target entity's exact position and look direction — the
   *server* drives the camera from then on, frame by frame, exactly like riding in the target's
   head. Nothing in this tool computes or syncs position itself; it just tells the server once who
   to follow (see `setFollow` in `index.js`, re-issued every 10s as a safety net in case the target
   disconnects/reconnects and spectate silently drops).
3. [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) is attached to that same
   mineflayer connection. It receives the ordinary stream of chunk/entity packets the server sends
   any client near the spectated position, meshes the visible terrain, and serves a three.js-based
   3D page over a small built-in HTTP+WebSocket server. Open that page in a browser and you're
   looking through (approximately) the spectated bot's own eyes, live.
4. A tiny HTTP control API (`GET /status`, `POST /follow`) lets you change which bot is being
   followed without restarting the process — see "Switching who it's watching" below. This is the
   seam a future dashboard integration would use (e.g. a button next to each bot in the existing 2D
   plot's sidebar).

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
(default `true`), `RESPECTATE_INTERVAL_MS` (default `10000`).

### Switching who it's watching

```
curl -X POST http://localhost:3002/follow -H 'content-type: application/json' -d '{"bot":"RSIFull-3"}'
curl http://localhost:3002/status
```

### Stopping it

`Ctrl+C` (or `docker stop` / `SIGTERM` in a container) — the shutdown handler disconnects the
spectator client and closes both HTTP servers cleanly, so the extra player slot and RCON
connection don't linger. It leaves the server in exactly the state it found it.

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

## Current limitations (this is a prototype, not Phase 2 proper)

- One spectator client follows one bot at a time. Watching several bots at once means running
  several instances (distinct `MC_USERNAME`/`VIEWER_PORT`/`CONTROL_PORT` each).
- Not wired into the 2D dashboard's UI yet — no "open 3D view" button next to a bot in
  `dashboard.html`'s sidebar. The control API (`POST /follow`) is exactly the seam that integration
  would call; it just doesn't exist yet.
- No auth on the control API or the viewer page itself — fine for `localhost`-only use (this
  project's whole dashboard story so far), not for exposing beyond that without adding some.
- `RESPECTATE_INTERVAL_MS`'s periodic re-issue is a blunt safety net, not a real "did spectate
  actually stick" check — it re-sends `/spectate` on a timer regardless of whether anything went
  wrong.
