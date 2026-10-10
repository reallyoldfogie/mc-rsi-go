// camera-viewer connects one extra, lightweight Minecraft client to the
// same server rsi-train's bots are already training against, puts it in
// spectator mode, and uses the server's own vanilla `/spectate <target>`
// command (via RCON) to lock its camera onto a chosen training bot's
// exact position/look — no manual position-syncing code needed, the
// server does that. prismarine-viewer then renders that live view as a
// 3D page in any browser. See README.md for the full design and how to
// run this (locally or in Docker).
'use strict';

const http = require('http');
const mineflayer = require('mineflayer');
const { mineflayer: attachViewer } = require('prismarine-viewer');
const { Rcon } = require('rcon-client');

const config = {
  mcHost: process.env.MC_HOST || '127.0.0.1',
  mcPort: parseInt(process.env.MC_PORT || '25565', 10),
  mcVersion: process.env.MC_VERSION || '1.21.5',
  username: process.env.MC_USERNAME || 'RSICameraViewer',
  rconHost: process.env.RCON_HOST || process.env.MC_HOST || '127.0.0.1',
  rconPort: parseInt(process.env.RCON_PORT || '25575', 10),
  rconPassword: process.env.RCON_PASSWORD || '',
  initialFollow: process.env.FOLLOW || '',
  viewerPort: parseInt(process.env.VIEWER_PORT || '3001', 10),
  controlPort: parseInt(process.env.CONTROL_PORT || '3002', 10),
  firstPerson: (process.env.FIRST_PERSON || 'true') === 'true',
  // viewerPrefix is prismarine-viewer's own `prefix` option (default ''
  // for standalone use, where the page is served at the viewer port's
  // own root). Set this when something else reverse-proxies the 3D page
  // under a sub-path: prismarine-viewer's client bundle derives its own
  // socket.io path from window.location.pathname at load time (see
  // node_modules/prismarine-viewer/lib/index.js), so the server-side
  // prefix here must exactly match the path the browser actually loads
  // the page at, or the socket.io handshake 400s. rsi-train's own
  // reverse proxy (cmd/rsi-train/cameraviewer.go) sets this to
  // "/camera/view" and proxies that same path 1:1, with no rewriting -
  // see that file's own doc comment.
  viewerPrefix: process.env.VIEWER_PREFIX || '',
  // Minecraft's own /spectate occasionally drops (the target bot
  // disconnecting and reconnecting under the same name, a server
  // restart, ...). Re-issuing it on an interval is cheap and makes the
  // viewer self-heal instead of silently freezing on the last frame.
  respectateIntervalMs: parseInt(process.env.RESPECTATE_INTERVAL_MS || '10000', 10),
};

if (!config.rconPassword) {
  console.error('camera-viewer: RCON_PASSWORD is required (see README.md)');
  process.exit(1);
}

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

let following = config.initialFollow || null;
let rcon = null;
let respectateTimer = null;

// mirrorTargetEntity/mirrorHandler track the live entity-position mirror
// (see startMirroring's own doc comment) so a later setFollow call can
// unsubscribe the previous target's listener before attaching a new one
// - otherwise switching targets would leave a stale handler mirroring
// the WRONG entity's movement on top of the new one's.
let mirrorTargetEntity = null;
let mirrorHandler = null;

async function rconSend(cmd) {
  if (!rcon) throw new Error('rcon not connected');
  const res = await rcon.send(cmd);
  log(`rcon: ${cmd} -> ${res}`);
  return res;
}

// startMirroring is the fix for a real limitation of /spectate found
// live while building this: vanilla Minecraft's spectate camera-follow
// is a REAL-CLIENT-ONLY rendering trick (the client continuously reads
// the spectated entity's own synced movement and renders from there
// every frame) - it is not a continuous server-side re-teleport of the
// spectating player's own entity. mineflayer, being a headless protocol
// client with no renderer, never receives whatever would make that
// continuous following visible to it: bot.entity.position only moves
// when the SERVER explicitly repositions OUR OWN player (which
// /spectate only ever does once, as an initial snap-near-the-target).
// Confirmed directly: bot.entity.position sat frozen for 30+ seconds
// after a single /spectate call while the target kept moving, even
// though bot.players[target].entity's own position (received the
// ordinary way every tracked entity's movement always is) kept
// updating the whole time (81 'entityMoved' events over 15s in that
// same test). prismarine-viewer (lib/mineflayer.js) broadcasts to the
// browser from bot.entity.position/yaw/pitch on bot's own 'move' event
// - so this mirrors the target's own live, continuously-updating
// entity data into bot.entity and re-emits 'move' ourselves, giving
// prismarine-viewer the continuous feed it was already built to use,
// without needing to patch prismarine-viewer itself at all.
//
// The periodic RCON re-spectate (setFollow's own respectateTimer)
// still matters alongside this: it keeps this spectator's REAL,
// server-tracked position (and therefore which chunks the server
// streams it) in the target's neighborhood as it roams, which this
// purely-local mirror has no way to influence on its own - the two are
// complementary, not redundant (coarse position/chunk anchor vs. fine-
// grained smooth camera).
function startMirroring(targetName) {
  if (mirrorTargetEntity && mirrorHandler) {
    bot.removeListener('entityMoved', mirrorHandler);
  }
  mirrorTargetEntity = null;
  mirrorHandler = null;

  // The target's entity may not be tracked yet immediately after the
  // RCON spectate snap (it only becomes visible to us once we're close
  // enough) - retry briefly rather than giving up on the first miss.
  const deadline = Date.now() + 5000;
  const tryResolve = () => {
    const player = bot.players[targetName];
    const entity = player && player.entity;
    if (!entity) {
      if (Date.now() < deadline) {
        setTimeout(tryResolve, 200);
      } else {
        log(`startMirroring: never saw ${targetName}'s entity within 5s; camera will stay on the initial /spectate snap`);
      }
      return;
    }

    mirrorTargetEntity = entity;
    mirrorHandler = (moved) => {
      if (moved !== mirrorTargetEntity) return;
      bot.entity.position = moved.position.clone();
      bot.entity.yaw = moved.yaw;
      bot.entity.pitch = moved.pitch;
      bot.emit('move');
    };
    bot.on('entityMoved', mirrorHandler);
    log(`startMirroring: now mirroring ${targetName}'s own live movement`);
  };
  tryResolve();
}

async function setFollow(target) {
  following = target || null;
  if (respectateTimer) {
    clearInterval(respectateTimer);
    respectateTimer = null;
  }
  if (mirrorTargetEntity && mirrorHandler) {
    bot.removeListener('entityMoved', mirrorHandler);
    mirrorTargetEntity = null;
    mirrorHandler = null;
  }
  if (!following) return;
  await rconSend(`spectate ${following} ${config.username}`);
  startMirroring(following);
  respectateTimer = setInterval(() => {
    rconSend(`spectate ${following} ${config.username}`).catch((err) => {
      log(`respectate failed: ${err.message}`);
    });
  }, config.respectateIntervalMs);
}

const bot = mineflayer.createBot({
  host: config.mcHost,
  port: config.mcPort,
  username: config.username,
  version: config.mcVersion,
  auth: 'offline',
});

bot.once('spawn', async () => {
  log(`spawned at ${bot.entity.position}`);

  attachViewer(bot, { port: config.viewerPort, firstPerson: config.firstPerson, prefix: config.viewerPrefix });
  log(`prismarine-viewer listening on :${config.viewerPort}`);

  try {
    rcon = await Rcon.connect({ host: config.rconHost, port: config.rconPort, password: config.rconPassword });
    await rconSend(`gamemode spectator ${config.username}`);
    if (following) {
      await setFollow(following);
    } else {
      log('no FOLLOW target set; use the control API to pick one (see README.md)');
    }
  } catch (err) {
    log(`rcon setup failed: ${err.message}`);
  }
});

bot.on('error', (err) => log(`bot error: ${err.message}`));
bot.on('kicked', (reason) => log(`kicked: ${JSON.stringify(reason)}`));
bot.on('end', () => log('connection ended'));

// Small HTTP control surface so whoever picks the bot to watch (a person,
// or eventually the dashboard's own per-agent selector — see the 2D
// dashboard's resolveSelected) doesn't need shell/RCON access themselves.
const controlServer = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/status') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ following, viewerPort: config.viewerPort }));
    return;
  }
  if (req.method === 'POST' && req.url === '/follow') {
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => {
      let target;
      try {
        target = JSON.parse(body || '{}').bot;
      } catch {
        res.writeHead(400, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'invalid JSON body, expected {"bot": "<username>"}' }));
        return;
      }
      if (!target || typeof target !== 'string') {
        res.writeHead(400, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'missing "bot" field' }));
        return;
      }
      setFollow(target)
        .then(() => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ following }));
        })
        .catch((err) => {
          res.writeHead(502, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: err.message }));
        });
    });
    return;
  }
  res.writeHead(404);
  res.end();
});
controlServer.listen(config.controlPort, () => {
  log(`control API listening on :${config.controlPort} (GET /status, POST /follow)`);
});

async function shutdown() {
  log('shutting down');
  if (respectateTimer) clearInterval(respectateTimer);
  controlServer.close();
  try {
    if (rcon) await rcon.end();
  } catch {
    // already closed; nothing to do
  }
  bot.quit();
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
