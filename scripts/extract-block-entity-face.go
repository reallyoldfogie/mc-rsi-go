// Command extract-block-entity-face crops the 6 individual cube faces out
// of a block-entity's UV-atlas texture (e.g. textures/entity/chest/*.png),
// given the exact (u, v, sizeX, sizeY, sizeZ) a vanilla *Model.java file
// passes to ModelPartBuilder.create().uv(u, v).cuboid(x, y, z, sizeX,
// sizeY, sizeZ) for one box of that model.
//
// This exists because block-entity-rendered blocks (chest, trapped_chest,
// ender_chest, bed, banner, shulker_box, skull, sign, ...) have no flat
// block/item texture at all - unlike every other block, which the
// dashboard's icon/3D pipeline already handles by filename convention
// (see cmd/rsi-train/dashboard.go's itemIconFallback and
// dashboard.html's resolveBlockMaterials). Figuring out WHERE in a
// block-entity's UV atlas a given face actually lives used to mean
// eyeballing a zoomed, grid-overlaid screenshot and guessing which
// segment looked like the right face (see this tool's own origin: the
// chest/trapped_chest/ender_chest crop done by hand in an earlier
// session). That guessing is no longer necessary - see the HOW section
// below.
//
// Usage (two modes - run as `go run ./scripts ...`, the directory form,
// not `go run scripts/extract-block-entity-face.go ...`, a single-file
// invocation that would never see block_entity_models.go's data):
//
//	go run ./scripts -list
//	go run ./scripts -list -model chest
//	go run ./scripts -model chest -part 0 -box bottom <src.png> <out-prefix>
//	go run ./scripts <src.png> <u> <v> <sizeX> <sizeY> <sizeZ> <out-prefix>
//
// The first form (-model, looking up block_entity_models.go's own
// transcribed data) is preferred whenever the block type is already
// cataloged there - no Java source to go re-read. The last (legacy
// explicit-number) form still exists for a block type that ISN'T
// cataloged yet; see docs/block-entity-textures.md's "recipe for a new
// block-entity type" for going from a fresh *Model.java read to a
// block_entity_models.go entry.
//
// Writes <out-prefix>_down.png, _up.png, _west.png, _north.png,
// _east.png, _south.png - crop one or two of these (plus an opposite
// "side" face from a different box) into a block-type's
// cmd/rsi-train/itemicons/ entry; see docs/block-entity-textures.md for
// the full walkthrough, including which face is a chest's actual front
// (south - see that doc for how this was confirmed).
//
// HOW: this isn't reverse-engineered or approximated. mc-data-gen (see
// .agent/cache/mc-data-gen/ and
// ../mc-data-gen/extractedSrc/<version>/net/minecraft/...) already
// ships a fully decompiled, Fabric-mapped Minecraft client source tree
// on this machine - no decompiling needed, just reading files already
// on disk. Every block-entity's visual model is a small, readable Java
// file under extractedSrc/<version>/net/minecraft/client/render/block/
// entity/model/ (e.g. ChestBlockModel.java), each box defined by one
// call like:
//
//	ModelPartBuilder.create().uv(0, 19).cuboid(1.0F, 0.0F, 1.0F, 14.0F, 10.0F, 14.0F)
//	                          ^^^^^^^^^                          ^^^^   ^^^^   ^^^^
//	                          (u, v)                             sizeX  sizeY  sizeZ
//
// (the three leading cuboid() args are the box's position offset within
// the model, irrelevant to texture extraction - only the trailing three
// sizes matter here.) Those (u, v, sizeX, sizeY, sizeZ) feed straight
// into this program.
//
// The face-rect math itself is transcribed exactly from the real engine
// source (ModelPart.java, inner class Cuboid's constructor, in that
// same extractedSrc tree) - not reverse-engineered, not approximated:
// given box size (w=sizeX, h=sizeY, d=sizeZ) at uv origin (u, v), the
// six faces occupy (confirmed by cross-checking against a hand-measured
// crop of the real chest texture - both land on the identical pixel
// rectangles):
//
//	DOWN:  x=[u+d,   u+d+w)    y=[v,   v+d)
//	UP:    x=[u+d+w, u+d+2w)   y=[v,   v+d)
//	WEST:  x=[u,     u+d)      y=[v+d, v+d+h)
//	NORTH: x=[u+d,   u+d+w)    y=[v+d, v+d+h)
//	EAST:  x=[u+d+w, u+2d+w)   y=[v+d, v+d+h)
//	SOUTH: x=[u+2d+w,u+2d+2w)  y=[v+d, v+d+h)
//
// Which face is actually the "front" (e.g. SOUTH carries the chest's
// latch) still has to be confirmed per model - WEST/NORTH/EAST/SOUTH are
// model-space directions, not a guarantee about which one a real
// in-world chest happens to face, and this dashboard has no block-facing
// data to orient it correctly anyway (same limitation the hand-cropped
// chest/trapped_chest/ender_chest textures already accepted - see
// dashboard.html's own CHEST_SHAPES doc comment). Open the cropped
// _south.png (or whichever) and look, the same way the chest's was
// confirmed.
package main

import (
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"sort"
	"strconv"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "extract-block-entity-face:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) > 0 && args[0] == "-list" {
		if len(args) >= 3 && args[1] == "-model" {
			return listModel(args[2])
		}
		listModels()
		return nil
	}
	if len(args) > 0 && args[0] == "-model" {
		return runFromCatalog(args[1:])
	}
	return runExplicit(args)
}

// runExplicit is the legacy path: every number given on the command
// line, for a block type block_entity_models.go doesn't have an entry
// for yet (see docs/block-entity-textures.md's "recipe for a new
// block-entity type" for going from a *Model.java read to here, and
// then to a permanent catalog entry).
func runExplicit(args []string) error {
	if len(args) != 7 {
		return fmt.Errorf("usage: extract-block-entity-face <src.png> <u> <v> <sizeX> <sizeY> <sizeZ> <out-prefix>\n   or: extract-block-entity-face -model <name> [-part N] [-box name-or-index] <src.png> <out-prefix>\n   or: extract-block-entity-face -list [-model <name>]")
	}
	srcPath, outPrefix := args[0], args[6]
	u, err := strconv.Atoi(args[1])
	if err != nil {
		return fmt.Errorf("u: %w", err)
	}
	v, err := strconv.Atoi(args[2])
	if err != nil {
		return fmt.Errorf("v: %w", err)
	}
	w, err := strconv.Atoi(args[3])
	if err != nil {
		return fmt.Errorf("sizeX: %w", err)
	}
	h, err := strconv.Atoi(args[4])
	if err != nil {
		return fmt.Errorf("sizeY: %w", err)
	}
	d, err := strconv.Atoi(args[5])
	if err != nil {
		return fmt.Errorf("sizeZ: %w", err)
	}
	return cropFaces(srcPath, u, v, w, h, d, outPrefix)
}

// runFromCatalog looks the box up in block_entity_models.go instead of
// taking raw numbers on the command line - the preferred path for
// anything already cataloged there.
func runFromCatalog(args []string) error {
	modelName := ""
	partIdx := 0
	boxSel := ""
	var rest []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-part":
			i++
			if i >= len(args) {
				return fmt.Errorf("-part needs a value")
			}
			n, err := strconv.Atoi(args[i])
			if err != nil {
				return fmt.Errorf("-part: %w", err)
			}
			partIdx = n
		case "-box":
			i++
			if i >= len(args) {
				return fmt.Errorf("-box needs a value")
			}
			boxSel = args[i]
		default:
			if modelName == "" {
				modelName = args[i]
			} else {
				rest = append(rest, args[i])
			}
		}
	}
	if len(rest) != 2 {
		return fmt.Errorf("usage: extract-block-entity-face -model <name> [-part N] [-box name-or-index] <src.png> <out-prefix>")
	}
	srcPath, outPrefix := rest[0], rest[1]

	model, ok := blockEntityModels[modelName]
	if !ok {
		return fmt.Errorf("no catalog entry for %q - run -list to see what's available", modelName)
	}
	if partIdx < 0 || partIdx >= len(model.Parts) {
		return fmt.Errorf("%q has %d part(s) (0..%d) - got -part %d", modelName, len(model.Parts), len(model.Parts)-1, partIdx)
	}
	part := model.Parts[partIdx]
	if boxSel == "" {
		if len(part.Boxes) != 1 {
			return fmt.Errorf("%q part %d has %d boxes - pick one with -box (name or index)", modelName, partIdx, len(part.Boxes))
		}
		boxSel = "0"
	}
	box, err := selectBox(part.Boxes, boxSel)
	if err != nil {
		return fmt.Errorf("%q part %d: %w", modelName, partIdx, err)
	}
	if part.Note != "" {
		fmt.Fprintf(os.Stderr, "note (part): %s\n", part.Note)
	}
	if model.Note != "" {
		fmt.Fprintf(os.Stderr, "note (model): %s\n", model.Note)
	}
	return cropFaces(srcPath, box.U, box.V, box.SizeX, box.SizeY, box.SizeZ, outPrefix)
}

func selectBox(boxes []blockEntityBox, sel string) (blockEntityBox, error) {
	if i, err := strconv.Atoi(sel); err == nil {
		if i < 0 || i >= len(boxes) {
			return blockEntityBox{}, fmt.Errorf("box index %d out of range (0..%d)", i, len(boxes)-1)
		}
		return boxes[i], nil
	}
	for _, b := range boxes {
		if b.Name == sel {
			return b, nil
		}
	}
	return blockEntityBox{}, fmt.Errorf("no box named %q", sel)
}

func listModels() {
	names := make([]string, 0, len(blockEntityModels))
	for name := range blockEntityModels {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		m := blockEntityModels[name]
		fmt.Printf("%-14s %d part(s)  %s\n", name, len(m.Parts), m.SourceFile)
	}
}

func listModel(name string) error {
	m, ok := blockEntityModels[name]
	if !ok {
		return fmt.Errorf("no catalog entry for %q", name)
	}
	fmt.Printf("%s (%s)\n", name, m.SourceFile)
	if m.Note != "" {
		fmt.Printf("  note: %s\n", m.Note)
	}
	for i, part := range m.Parts {
		fmt.Printf("  part %d: %s (%dx%d texture)\n", i, part.SourceMethod, part.TextureWidth, part.TextureHeight)
		if part.Note != "" {
			fmt.Printf("    note: %s\n", part.Note)
		}
		for bi, b := range part.Boxes {
			fmt.Printf("    box %d %q: u=%d v=%d size=%dx%dx%d%s\n", bi, b.Name, b.U, b.V, b.SizeX, b.SizeY, b.SizeZ, mirroredSuffix(b.Mirrored))
		}
		if len(part.TexturePaths) > 0 {
			fmt.Printf("    textures: %v\n", part.TexturePaths)
		}
	}
	return nil
}

func mirroredSuffix(m bool) string {
	if m {
		return " (mirrored)"
	}
	return ""
}

// cropFaces writes srcPath's 6 face crops for a box at UV origin (u, v)
// with dimensions (w, h, d) to <outPrefix>_<face>.png.
func cropFaces(srcPath string, u, v, w, h, d int, outPrefix string) error {
	f, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		return fmt.Errorf("decoding %s: %w", srcPath, err)
	}

	// Transcribed exactly from ModelPart.java's Cuboid constructor (see
	// this file's own package doc comment for the derivation and the
	// cross-check against a hand-measured crop).
	faces := map[string]image.Rectangle{
		"down":  rect(u+d, v, u+d+w, v+d),
		"up":    rect(u+d+w, v, u+d+2*w, v+d),
		"west":  rect(u, v+d, u+d, v+d+h),
		"north": rect(u+d, v+d, u+d+w, v+d+h),
		"east":  rect(u+d+w, v+d, u+2*d+w, v+d+h),
		"south": rect(u+2*d+w, v+d, u+2*d+2*w, v+d+h),
	}

	for name, r := range faces {
		out := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
		draw.Draw(out, out.Bounds(), src, r.Min, draw.Src)
		outPath := fmt.Sprintf("%s_%s.png", outPrefix, name)
		of, err := os.Create(outPath)
		if err != nil {
			return err
		}
		if err := png.Encode(of, out); err != nil {
			of.Close()
			return fmt.Errorf("encoding %s: %w", outPath, err)
		}
		if err := of.Close(); err != nil {
			return err
		}
		fmt.Printf("wrote %s (%dx%d from %s)\n", outPath, r.Dx(), r.Dy(), r)
	}
	return nil
}

func rect(x0, y0, x1, y1 int) image.Rectangle {
	return image.Rect(x0, y0, x1, y1)
}
