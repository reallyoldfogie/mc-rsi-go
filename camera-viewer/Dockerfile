# node:*-slim (Debian/glibc), not node:*-alpine: the "canvas" dependency
# (prismarine-viewer's texture atlas) ships prebuilt glibc binaries with
# every native lib it needs already bundled alongside the addon (see
# README.md's "Does this need Docker?" section) - alpine's musl libc
# can't load those, and would force a from-source rebuild needing a full
# cairo/pango/jpeg/... toolchain this image deliberately avoids.
FROM node:22-slim

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY index.js ./

# VIEWER_PORT and CONTROL_PORT's own defaults (index.js) - keep these in
# sync if you change either env var's default there.
EXPOSE 3001 3002

CMD ["node", "index.js"]
