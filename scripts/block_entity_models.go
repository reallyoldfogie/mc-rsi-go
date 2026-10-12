package main

// blockEntityModels is a direct transcription of every `uv(u, v).cuboid(x,
// y, z, sizeX, sizeY, sizeZ)` call (and equivalents) found in the
// decompiled Minecraft 1.21.11 client source already on this machine at
// ../mc-data-gen/extractedSrc/1.21.11/net/minecraft/... (see
// docs/block-entity-textures.md for how that tree got there and why no
// decompiling was needed). Every SourceFile/SourceMethod below was read
// directly from that tree, not retyped from memory or guessed - this file
// exists so extract-block-entity-face.go (and anyone extending it to a
// new block type) never has to go back and re-read the Java source for
// anything already covered here.
//
// Minecraft's own box-UV convention: (U, V) is the UV-atlas origin for
// that one box; SizeX/SizeY/SizeZ are the box's dimensions in texture
// pixels (not its position in the model, which matters for replicating
// the 3D shape - see dashboard.html's CHEST_SHAPES for that - but not for
// texture extraction). Feed (U, V, SizeX, SizeY, SizeZ) straight into
// this file's own faceRects (shared with the CLI's legacy explicit-number
// path) to get that box's 6 face crop rectangles.
type blockEntityBox struct {
	// Name is the literal child name Mojang's own addChild("name", ...)
	// call uses, where the source gives one - purely documentation, not
	// looked up by the CLI (box selection is by index; see -box-index).
	Name                string
	U, V                int
	SizeX, SizeY, SizeZ int
	// Mirrored is true when the source chains .mirrored() onto this
	// box's builder (seen on trident's right_spike) - the face rects
	// are identical to the unmirrored box, just horizontally flipped on
	// the model (not in the texture), so it doesn't change what to
	// crop, only how the box renders in 3D.
	Mirrored bool
}

// blockEntityPart is one getXyzTexturedModelData()-style method - some
// block-entities (chest) have exactly one; others (bed, banner, conduit,
// decorated_pot) build several independent parts, each against its own
// texture canvas size, from separate methods. TextureWidth/Height is the
// `TexturedModelData.of(modelData, W, H)` call's own W/H - NOT
// necessarily the real texture file's actual pixel dimensions (compare
// against the file itself; they're expected to match, but this file only
// transcribes what the Java source declares).
type blockEntityPart struct {
	SourceMethod                string
	TextureWidth, TextureHeight int
	// TexturePaths are real files under textures/entity/ in the client
	// jar (relative to assets/minecraft/), confirmed to exist on disk at
	// mc-protocol-go's .cache/metadata/1.21.11/extracted_client/assets/
	// minecraft/ - not every variant is listed for a block with many
	// (e.g. all 16 bed colors), just enough to show the naming pattern.
	TexturePaths []string
	Boxes        []blockEntityBox
	// Note flags anything about this part that doesn't fit the simple
	// "one static texture, crop it" model this tool was built for -
	// e.g. a zero-depth/zero-height quad (not a real 6-faced box), a
	// Dilation-adjusted box, or geometry defined by chaining more than
	// one .uv().cuboid() call on a single builder. Non-empty Note means
	// "read this before trusting a crop from this part blindly."
	Note string
}

// blockEntityModel is one block/item's full set of parts, plus where
// they came from.
type blockEntityModel struct {
	// SourceFile is relative to extractedSrc/<Version>/net/minecraft/.
	SourceFile string
	Version    string
	Parts      []blockEntityPart
	// Note flags a whole-model caveat - most commonly "the real in-game
	// texture is a runtime composite (dye color, pattern layers, skin),
	// not a single static file" - the TexturePaths given are the base
	// layer only, which is a real, existing file, just not the full
	// final appearance.
	Note string
}

// blockEntityModels covers every block/item Minecraft 1.21.11 renders
// through its "special model" system instead of a standard flat block/
// item texture (the full list: client/render/item/model/special/*.java
// in the same extractedSrc tree) - except copper_golem_statue (a 26.1-era
// mob statue, ~325 lines across 5 pose variants, and low relevance to a
// mine/craft/goto training bot - skipped, not guessed) and the
// dragon/piglin skull variants (SkullEntityModel below only covers the
// 5 skull types - skeleton/wither_skeleton/player/zombie/creeper - that
// share its geometry; dragon and piglin heads use their own distinct,
// untranscribed model classes).
var blockEntityModels = map[string]blockEntityModel{
	"chest": {
		SourceFile: "net/minecraft/client/render/block/entity/model/ChestBlockModel.java",
		Version:    "1.21.11",
		Parts: []blockEntityPart{{
			SourceMethod:  "getSingleTexturedModelData",
			TextureWidth:  64,
			TextureHeight: 64,
			TexturePaths: []string{
				"textures/entity/chest/normal.png", "textures/entity/chest/trapped.png", "textures/entity/chest/ender.png",
				"textures/entity/chest/christmas.png", "textures/entity/chest/copper.png", "textures/entity/chest/copper_exposed.png",
				"textures/entity/chest/copper_weathered.png", "textures/entity/chest/copper_oxidized.png",
			},
			Boxes: []blockEntityBox{
				{Name: "bottom", U: 0, V: 19, SizeX: 14, SizeY: 10, SizeZ: 14},
				{Name: "lid", U: 0, V: 0, SizeX: 14, SizeY: 5, SizeZ: 14},
				{Name: "lock", U: 0, V: 0, SizeX: 2, SizeY: 4, SizeZ: 1},
			},
		}},
	},

	"bed": {
		SourceFile: "net/minecraft/client/render/block/entity/BedBlockEntityRenderer.java",
		Version:    "1.21.11",
		Note:       "head and foot are two independent parts (not two boxes of one model) - a real bed block needs both, placed per the block's own two-block footprint, which this data doesn't carry.",
		Parts: []blockEntityPart{
			{
				SourceMethod:  "getHeadTexturedModelData",
				TextureWidth:  64,
				TextureHeight: 64,
				TexturePaths:  []string{"textures/entity/bed/red.png", "textures/entity/bed/white.png", "textures/entity/bed/blue.png"},
				Boxes: []blockEntityBox{
					{Name: "main", U: 0, V: 0, SizeX: 16, SizeY: 16, SizeZ: 6},
					{Name: "left_leg", U: 50, V: 6, SizeX: 3, SizeY: 3, SizeZ: 3},
					{Name: "right_leg", U: 50, V: 18, SizeX: 3, SizeY: 3, SizeZ: 3},
				},
			},
			{
				SourceMethod:  "getFootTexturedModelData",
				TextureWidth:  64,
				TextureHeight: 64,
				TexturePaths:  []string{"textures/entity/bed/red.png", "textures/entity/bed/white.png", "textures/entity/bed/blue.png"},
				Boxes: []blockEntityBox{
					{Name: "main", U: 0, V: 22, SizeX: 16, SizeY: 16, SizeZ: 6},
					{Name: "left_leg", U: 50, V: 0, SizeX: 3, SizeY: 3, SizeZ: 3},
					{Name: "right_leg", U: 50, V: 12, SizeX: 3, SizeY: 3, SizeZ: 3},
				},
			},
		},
	},

	"banner": {
		SourceFile: "net/minecraft/client/render/block/entity/model/BannerBlockModel.java (pole/bar) + BannerFlagBlockModel.java (flag)",
		Version:    "1.21.11",
		Note:       "the real texture is textures/entity/banner_base.png composited at runtime with a dye color plus a stack of grayscale pattern-layer masks (textures/entity/banner/*.png) - cropping banner_base.png alone gives an undyed, pattern-free approximation only, same simplification already accepted for dashboard.html's grass-tint gap.",
		Parts: []blockEntityPart{
			{
				SourceMethod:  "BannerBlockModel.getTexturedModelData(standing=true)",
				TextureWidth:  64,
				TextureHeight: 64,
				TexturePaths:  []string{"textures/entity/banner_base.png"},
				Boxes: []blockEntityBox{
					{Name: "pole", U: 44, V: 0, SizeX: 2, SizeY: 42, SizeZ: 2},
					{Name: "bar", U: 0, V: 42, SizeX: 20, SizeY: 2, SizeZ: 2},
				},
				Note: "wall-mounted banners (standing=false) use the same \"bar\" box/UV, just repositioned - no pole.",
			},
			{
				SourceMethod:  "BannerFlagBlockModel.getTexturedModelData",
				TextureWidth:  64,
				TextureHeight: 64,
				TexturePaths:  []string{"textures/entity/banner_base.png"},
				Boxes: []blockEntityBox{
					{Name: "flag", U: 0, V: 0, SizeX: 20, SizeY: 40, SizeZ: 1},
				},
			},
		},
	},

	"shield": {
		SourceFile: "net/minecraft/client/render/entity/model/ShieldEntityModel.java",
		Version:    "1.21.11",
		Note:       "textures/entity/shield_base_nopattern.png is the plain base; a real shield also layers an optional banner-pattern overlay at runtime, same compositing family as banner.",
		Parts: []blockEntityPart{{
			SourceMethod:  "getTexturedModelData",
			TextureWidth:  64,
			TextureHeight: 64,
			TexturePaths:  []string{"textures/entity/shield_base_nopattern.png"},
			Boxes: []blockEntityBox{
				{Name: "plate", U: 0, V: 0, SizeX: 12, SizeY: 22, SizeZ: 1},
				{Name: "handle", U: 26, V: 0, SizeX: 2, SizeY: 6, SizeZ: 6},
			},
		}},
	},

	"trident": {
		SourceFile: "net/minecraft/client/render/entity/model/TridentEntityModel.java",
		Version:    "1.21.11",
		Parts: []blockEntityPart{{
			SourceMethod:  "getTexturedModelData",
			TextureWidth:  32,
			TextureHeight: 32,
			TexturePaths:  []string{"textures/entity/trident.png"},
			Boxes: []blockEntityBox{
				{Name: "pole", U: 0, V: 6, SizeX: 1, SizeY: 25, SizeZ: 1},
				{Name: "base", U: 4, V: 0, SizeX: 3, SizeY: 2, SizeZ: 1},
				{Name: "left_spike", U: 4, V: 3, SizeX: 1, SizeY: 4, SizeZ: 1},
				{Name: "middle_spike", U: 0, V: 0, SizeX: 1, SizeY: 4, SizeZ: 1},
				{Name: "right_spike", U: 4, V: 3, SizeX: 1, SizeY: 4, SizeZ: 1, Mirrored: true},
			},
		}},
	},

	"shulker_box": {
		SourceFile: "net/minecraft/client/render/entity/model/ShulkerEntityModel.java",
		Version:    "1.21.11",
		Parts: []blockEntityPart{{
			SourceMethod:  "getShulkerBoxTexturedModelData",
			TextureWidth:  64,
			TextureHeight: 64,
			TexturePaths: []string{
				"textures/entity/shulker/shulker.png", "textures/entity/shulker/shulker_white.png", "textures/entity/shulker/shulker_black.png",
			},
			Boxes: []blockEntityBox{
				{Name: "lid", U: 0, V: 0, SizeX: 16, SizeY: 12, SizeZ: 16},
				{Name: "base", U: 0, V: 28, SizeX: 16, SizeY: 8, SizeZ: 16},
			},
			Note: "17 dyed colors + 1 undyed default, each its own complete, static PNG (unlike banner/shield, no runtime color compositing) - straightforward, same tier as chest.",
		}},
	},

	"sign": {
		SourceFile: "net/minecraft/client/render/block/entity/SignBlockEntityRenderer.java",
		Version:    "1.21.11",
		Note:       "identical geometry reused for every wood type (EntityModels.java builds one TexturedModelData and shares it across WoodType.stream()) - only the texture file differs.",
		Parts: []blockEntityPart{{
			SourceMethod:  "getTexturedModelData(standing)",
			TextureWidth:  64,
			TextureHeight: 32,
			TexturePaths: []string{
				"textures/entity/signs/oak.png", "textures/entity/signs/spruce.png", "textures/entity/signs/birch.png",
				"textures/entity/signs/crimson.png", "textures/entity/signs/warped.png", "textures/entity/signs/bamboo.png",
			},
			Boxes: []blockEntityBox{
				{Name: "sign", U: 0, V: 0, SizeX: 24, SizeY: 12, SizeZ: 2},
				{Name: "stick", U: 0, V: 14, SizeX: 2, SizeY: 14, SizeZ: 2},
			},
			Note: "\"stick\" only exists when standing=true (a post sign); a wall sign has no stick.",
		}},
	},

	"hanging_sign": {
		SourceFile: "net/minecraft/client/render/block/entity/HangingSignBlockEntityRenderer.java",
		Version:    "1.21.11",
		Note:       "same wood-type texture-sharing as sign; chain boxes vary by AttachmentType (WALL/CEILING/CEILING_MIDDLE) - only \"board\" (the actual sign face) is universal and the one worth cropping for an icon.",
		Parts: []blockEntityPart{{
			SourceMethod:  "getTexturedModelData(AttachmentType)",
			TextureWidth:  64,
			TextureHeight: 32,
			TexturePaths:  []string{"textures/entity/signs/hanging/oak.png", "textures/entity/signs/hanging/spruce.png", "textures/entity/signs/hanging/bamboo.png"},
			Boxes: []blockEntityBox{
				{Name: "board", U: 0, V: 12, SizeX: 14, SizeY: 10, SizeZ: 2},
				{Name: "plank (WALL only)", U: 0, V: 0, SizeX: 16, SizeY: 2, SizeZ: 4},
				{Name: "chain_left/right (WALL/CEILING)", U: 0, V: 6, SizeX: 3, SizeY: 6, SizeZ: 0},
				{Name: "vChains (CEILING_MIDDLE)", U: 14, V: 6, SizeX: 12, SizeY: 6, SizeZ: 0},
			},
			Note: "chain boxes have SizeZ=0 (flat quads, not real 6-faced boxes) - faceRects' DOWN/UP/WEST/EAST faces degenerate to zero-width rects for these; only board/plank are meaningful crop targets.",
		}},
	},

	"skull": {
		SourceFile: "net/minecraft/client/render/entity/model/SkullEntityModel.java",
		Version:    "1.21.11",
		Note:       "covers skeleton, wither_skeleton, player, zombie, creeper (SkullBlockEntityRenderer.getModels() maps all 5 to this one class/geometry - confirmed PlayerHeadModelRenderer is not a separate geometry, just this same path called with SkullBlock.Type.PLAYER). dragon and piglin skulls use their own distinct, untranscribed model classes - not covered here.",
		Parts: []blockEntityPart{{
			SourceMethod:  "getSkullTexturedModelData",
			TextureWidth:  64,
			TextureHeight: 32,
			TexturePaths: []string{
				"textures/entity/skeleton/skeleton.png", "textures/entity/skeleton/wither_skeleton.png",
				"textures/entity/zombie/zombie.png", "textures/entity/creeper/creeper.png",
			},
			Boxes: []blockEntityBox{
				{Name: "head", U: 0, V: 0, SizeX: 8, SizeY: 8, SizeZ: 8},
			},
			Note: "player heads use a dynamic per-player skin texture (session-server skin or DefaultSkinHelper), not a static asset file - no TexturePaths entry for it.",
		}},
	},

	"conduit": {
		SourceFile: "net/minecraft/client/render/block/entity/ConduitBlockEntityRenderer.java",
		Version:    "1.21.11",
		Note:       "4 fully independent parts/canvases (not variants of one model) - eye, wind, shell (active), and a separate plain/inactive shell.",
		Parts: []blockEntityPart{
			{
				SourceMethod: "getEyeTexturedModelData", TextureWidth: 16, TextureHeight: 16,
				TexturePaths: []string{"textures/entity/conduit/open_eye.png", "textures/entity/conduit/closed_eye.png"},
				Boxes:        []blockEntityBox{{Name: "eye", U: 0, V: 0, SizeX: 8, SizeY: 8, SizeZ: 0}},
				Note:         "SizeZ=0 (flat quad, Dilation(0.01) applied) - not a real 6-faced box.",
			},
			{
				SourceMethod: "getWindTexturedModelData", TextureWidth: 64, TextureHeight: 32,
				TexturePaths: []string{"textures/entity/conduit/wind.png", "textures/entity/conduit/wind_vertical.png"},
				Boxes:        []blockEntityBox{{Name: "wind", U: 0, V: 0, SizeX: 16, SizeY: 16, SizeZ: 16}},
			},
			{
				SourceMethod: "getShellTexturedModelData", TextureWidth: 32, TextureHeight: 16,
				TexturePaths: []string{"textures/entity/conduit/cage.png"},
				Boxes:        []blockEntityBox{{Name: "shell", U: 0, V: 0, SizeX: 6, SizeY: 6, SizeZ: 6}},
			},
			{
				SourceMethod: "getPlainTexturedModelData", TextureWidth: 32, TextureHeight: 16,
				TexturePaths: []string{"textures/entity/conduit/base.png"},
				Boxes:        []blockEntityBox{{Name: "shell", U: 0, V: 0, SizeX: 8, SizeY: 8, SizeZ: 8}},
				Note:         "the inactive/plain conduit - probably the right one for a static dashboard icon (no eye/wind animation state to pick otherwise).",
			},
		},
	},

	"decorated_pot": {
		SourceFile: "net/minecraft/client/render/block/entity/DecoratedPotBlockEntityRenderer.java",
		Version:    "1.21.11",
		Note: "the most irregular model here: the \"neck\" box chains TWO .uv().cuboid() calls on one builder (both listed, as neck_outer/neck_inner); " +
			"\"top\"/\"bottom\" share one zero-height (SizeY=0) builder repositioned per-instance; \"sides\" are four zero-depth (SizeZ=0), " +
			"single-face (NORTH only) quads sharing one builder, rotated per side. None of these fit the simple box-crop model cleanly - " +
			"read DecoratedPotBlockEntityRenderer.java directly before trusting a crop from this entry. Real side textures also come from " +
			"per-sherd-pattern compositing (getTextureIdFromSherd), not one static file.",
		Parts: []blockEntityPart{
			{
				SourceMethod: "getTopBottomNeckTexturedModelData", TextureWidth: 32, TextureHeight: 32,
				TexturePaths: []string{"textures/entity/decorated_pot/decorated_pot_base.png"},
				Boxes: []blockEntityBox{
					{Name: "neck_outer", U: 0, V: 0, SizeX: 8, SizeY: 3, SizeZ: 8},
					{Name: "neck_inner", U: 0, V: 5, SizeX: 6, SizeY: 1, SizeZ: 6},
					{Name: "top_and_bottom (shared builder)", U: -14, V: 13, SizeX: 14, SizeY: 0, SizeZ: 14},
				},
			},
			{
				SourceMethod: "getSidesTexturedModelData", TextureWidth: 16, TextureHeight: 16,
				TexturePaths: []string{"textures/entity/decorated_pot/decorated_pot_base.png"},
				Boxes: []blockEntityBox{
					{Name: "back_left_right_front (shared builder, NORTH face only)", U: 1, V: 0, SizeX: 14, SizeY: 16, SizeZ: 0},
				},
			},
		},
	},
}
