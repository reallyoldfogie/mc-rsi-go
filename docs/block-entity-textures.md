# Texturing block-entity blocks (chest, bed, banner, shulker_box, skull, sign, ...)

The dashboard's inventory icons (`cmd/rsi-train/itemicons/`, served by `dashboard.go`'s
`itemIconFallback`) and its 3D view (`dashboard.html`'s `resolveBlockMaterials`) both work by
filename convention: a block's textures live at `textures/block/<name>[_top|_side|_front].png` in
the client jar, and the dashboard just guesses which suffix exists.

That convention doesn't apply to a handful of blocks Minecraft renders through its block-entity
system instead of the ordinary block-model system. These have no `textures/block/*.png` or
`textures/item/*.png` at all — their appearance comes from a small 3D model (a Java class, built
from a handful of boxes) sampling one UV-atlas texture under `textures/entity/`. `chest`'s gap was
the first one found (dashboard.go's `itemIconFallback` fell back to a plain letter badge for it).

**`scripts/block_entity_models.go` already catalogs every box's `(u, v, sizeX, sizeY, sizeZ)` for
every such block/item in Minecraft 1.21.11** (`chest`, `bed`, `banner`, `shield`, `trident`,
`shulker_box`, `sign`, `hanging_sign`, `skull`, `conduit`, `decorated_pot` — 11 types, each with a
`SourceFile`/`SourceMethod` citing exactly which decompiled Java file and method it was transcribed
from; `copper_golem_statue` and the dragon/piglin skull variants were explicitly skipped as
low-relevance/oversized, not silently omitted — see that file's own top comment). Run
`go run ./scripts -list` to see the full catalog, or `go run ./scripts -list -model bed` for one
type's own parts/boxes/notes/texture paths. For anything already in that file, skip straight to
["Turning catalog data into crops"](#turning-catalog-data-into-crops) below — the "read the Java
source" step this doc originally walked through for chest is already done.

## The method: read it, don't guess it

The first pass at fixing `chest`/`trapped_chest`/`ender_chest` (see git history around when
`cmd/rsi-train/itemicons/chest.png` was added) found the right pixels by eye: zoom the 64×64 atlas
10x with a pixel grid overlaid, scan for the bounding box of opaque content per row to find face
boundaries, crop each candidate segment, and look at which one had the recognizable detail (the
latch, for chest). That's a real, working method — but it's trial and error, redone from scratch
for every new block type, with real risk of a subtly wrong crop.

There's a better way, and it turns out to already be sitting on disk: **this machine already has a
fully decompiled, readably-named Minecraft client source tree**, via the `mc-data-gen` tool this
ecosystem already uses to extract block/item/entity data
(`.agent/cache/mc-data-gen/` has the tool; the actual decompiled sources are at
`../mc-data-gen/extractedSrc/<version>/net/minecraft/...`, one subtree per Minecraft version this
project supports). No decompiling needs to happen — it's just reading `.java` files that are
already there.

Every block-entity's visual model is one small, readable file under
`extractedSrc/<version>/net/minecraft/client/render/block/entity/model/` (e.g.
`ChestBlockModel.java`). Each box of the model is one call like:

```java
ModelPartBuilder.create().uv(0, 19).cuboid(1.0F, 0.0F, 1.0F, 14.0F, 10.0F, 14.0F)
                          ^^^^^^^^^                          ^^^^   ^^^^   ^^^^
                          (u, v): UV origin in the atlas      sizeX  sizeY  sizeZ
```

(The three leading `cuboid()` args are the box's position *within the model* — irrelevant for
texture extraction, only useful if you're also replicating the 3D shape, the way
`dashboard.html`'s `CHEST_SHAPES`/`CHEST_BODY_H`/`CHEST_LID_H` do for chest.)

For chest specifically, `ChestBlockModel.getSingleTexturedModelData()` has three boxes:

```java
addChild("bottom", ...uv(0, 19).cuboid(1, 0, 1, 14, 10, 14), ...)   // the body
addChild("lid",    ...uv(0, 0).cuboid(1, 0, 0, 14, 5, 14), origin(0, 9, 1))
addChild("lock",   ...uv(0, 0).cuboid(7, -2, 14, 2, 4, 1), origin(0, 9, 1))  // the latch, a real tiny 3D box - not just a texture detail
```

## Turning catalog data into crops

`scripts/extract-block-entity-face.go` does this mechanically — no eyeballing. Given a box's
`(u, v, sizeX, sizeY, sizeZ)`, it crops all 6 faces (`down`, `up`, `west`, `north`, `east`,
`south`) to separate PNGs. The face-rect math is transcribed exactly from the real engine source
(`ModelPart.java`'s `Cuboid` constructor, same `extractedSrc` tree, lines ~280-333), not
reverse-engineered — and it was cross-checked against the original hand-measured chest crop: both
land on the pixel-identical rectangle for chest's front face (`x=42–55, y=33–42` for the body).

Preferred: look the box up by name in `scripts/block_entity_models.go` directly (run as `go run
./scripts`, the directory form — a single-file `go run scripts/extract-block-entity-face.go` won't
see that file's data):

```bash
go run ./scripts -list -model chest      # see its parts/boxes first
go run ./scripts -model chest -part 0 -box bottom \
  path/to/extracted_client/assets/minecraft/textures/entity/chest/normal.png \
  /tmp/chest_body
# -> /tmp/chest_body_down.png, _up.png, _west.png, _north.png, _east.png, _south.png
```

For a box that isn't cataloged yet, the legacy explicit-number form still works (see "Recipe for a
new block-entity type" below — the last step there is adding a permanent entry to
`block_entity_models.go` instead of leaving it as a one-off command):

```bash
go run ./scripts \
  path/to/extracted_client/assets/minecraft/textures/entity/chest/normal.png \
  0 19 14 10 14 \
  /tmp/chest_body
```

One thing the formula *can't* tell you: which of the four side faces (`west`/`north`/`east`/`south`)
is the visually-interesting one (the one with the latch, the glowing eye, the trip-hook, ...) —
those are model-space directions, not a promise about which one looks distinctive. Open each
`_west`/`_north`/`_east`/`_south.png` and look; for chest it was `south`. This is the one step that
still needs a human (or Claude) to glance at the output.

It's also worth remembering this doesn't solve block *facing*: a real in-world chest can face any
of 4 directions, and `VisibleBlockInfo`/`worldStreamBlock` carry no facing/rotation data at all, so
the 3D view always shows the "front" face pointing the same fixed world direction (+Z) regardless of
which way the real chest is actually rotated. That's a separate, orthogonal gap from texture
extraction — the formula above doesn't change it, and fixing it would mean plumbing block state
(not just StateID→name) through `FindAllVisibleBlocksInSphere`'s whole pipeline.

## Recipe for a block-entity type not yet in block_entity_models.go

(`copper_golem_statue` and the dragon/piglin skull variants are the only known gaps as of this
writing — see `block_entity_models.go`'s top comment. Everything else, skip to step 4.)

1. Find its model file: `grep -ril <BlockName> ../mc-data-gen/extractedSrc/1.21.11/net/minecraft/client/render/block/entity/model/` (or `.../render/item/model/special/` for the item-icon-specific renderer, which sometimes wraps the same block model rather than defining its own).
2. Read off each box's `uv(u, v).cuboid(x, y, z, sizeX, sizeY, sizeZ)` call, and the
   `TexturedModelData.of(modelData, W, H)` call giving the canvas size.
3. Add a `blockEntityModel` entry for it in `scripts/block_entity_models.go`, citing the exact
   `SourceFile`/`SourceMethod` you just read (follow the existing entries' shape — most types are
   one `blockEntityPart`; a type with several independent `getXyzTexturedModelData()` methods, like
   `bed` or `conduit`, gets one `blockEntityPart` per method). Note any geometry that doesn't fit
   the simple one-box-per-call pattern (see `decorated_pot`'s entry for an example) rather than
   forcing it to look clean.
4. Run `go run ./scripts -model <name> -part <n> -box <name-or-index> <real-texture.png>
   <out-prefix>`, pointed at that type's real texture under `textures/entity/<name>/...` (find it
   the way chest's was found: `find <jar-extraction-dir>/textures/entity -iname '*<name>*'`).
5. Look at the 6 output crops, pick whichever face(s) are visually distinctive vs. plain.
6. Save the ones you want into `cmd/rsi-train/itemicons/<name>.png` (front/distinctive,
   doubles as the 2D inventory icon - `itemIconFallback` serves an exact-match file with zero code
   changes) and `cmd/rsi-train/itemicons/<name>_side.png` (plain, for the 3D view's non-front
   faces).
7. If the shape is meaningfully non-cube (a bed's L-shape, a banner's flat hanging cloth), extend
   `dashboard.html`'s `CHEST_SHAPES`-style special-casing with the new box dimensions (the same
   `(sizeX, sizeY, sizeZ)` triples now sitting in `block_entity_models.go`, divided by 16 to
   convert to block-fraction units) rather than forcing it through the generic single-cube path.
