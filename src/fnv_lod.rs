//! Terrain LOD generation for FNV worldspaces: NIF meshes + DDS texture
//! atlases + a `.dlodsettings` file, written as loose files alongside the
//! `.esm` (see `generate_fnv_esm` in `fnv_esm.rs`).
//!
//! NIF block layout below is a direct port of the real (open-source) C#
//! reference implementation in xLODGen (github.com/TES5Edit/xLODGen,
//! `NifMain/*.cs`), specialized to the exact version/userVersion/userVersion2
//! triple real FNV LOD meshes use (335675399 / 11 / 34) — confirmed by
//! decoding real `wastelandnv.level*.nif` files from `Fallout - Meshes.bsa`.
//! Each block-content builder below corresponds 1:1 to one C# class's
//! `Write()` method, with version-conditional branches collapsed to the
//! single concrete path that applies for our fixed version triple (rather
//! than porting the fully generic conditional logic, since we only ever
//! target one NIF version).

use crate::coordinate_system::cartesian::XZPoint;
use crate::ground::Ground;
use std::collections::BTreeMap;

/// Quads per side of every LOD tile's uniform sampling grid, regardless of
/// level (Phase 1 — see this module's plan notes: adaptive/variable-density
/// meshing matching real triangle budgets is an explicit follow-up phase).
/// 10x10 quads = 11x11 = 121 vertices, 200 top-surface + 80 skirt = 280
/// triangles per tile — chosen to land close to the real per-tile average
/// (10,936 triangles / 44 tiles ~= 248 for Rome's actual bounds) as a cheap
/// diagnostic test of whether our uniform (non-adaptive) grid being ~2.6x
/// real's total triangle count across the worldspace is what causes the
/// in-game load hang, before committing to real adaptive simplification.
/// Previously 16 (640 tris/tile, ~2.6x real's per-worldspace total).
const LOD_TILE_QUADS: usize = 10;

/// One LOD tile: `level` cells wide/tall, anchored at FNV cell coordinate
/// (tile_cell_x, tile_cell_y) — matches the real naming convention where
/// tile X/Y step by `level` in cell units.
struct LodTile {
    level: u32,
    tile_cell_x: i32,
    tile_cell_y: i32,
}

/// Convert an FNV cell coordinate to the arnis-internal block index (see
/// `game_x`/`game_y` derivation in this function's caller) — inverse of
/// `cell_x = col - x_offset`.
fn cell_to_block(cell: i32, offset: i32) -> i32 {
    (cell + offset) * 32
}

/// Size (in cells) of the square LOD tile grid the engine expects, from the
/// `.dlodsettings` `grid_size` field's own formula (see `build_dlodsettings`).
/// Shared here because `generate_lod_assets` must generate tiles spanning
/// this *entire* square, not just the area overlapping authored cells — see
/// its doc comment for why.
fn compute_grid_size(min_cell_x: i32, max_cell_x: i32, min_cell_y: i32, max_cell_y: i32) -> i32 {
    const MAX_LEVEL: i32 = 32;
    let tile_min_x = min_cell_x.div_euclid(MAX_LEVEL) * MAX_LEVEL;
    let tile_max_x = (max_cell_x.div_euclid(MAX_LEVEL) + 1) * MAX_LEVEL;
    let tile_min_y = min_cell_y.div_euclid(MAX_LEVEL) * MAX_LEVEL;
    let tile_max_y = (max_cell_y.div_euclid(MAX_LEVEL) + 1) * MAX_LEVEL;
    ((tile_max_x - tile_min_x).max(tile_max_y - tile_min_y)).max(MAX_LEVEL)
}

/// Block-space (AX, AZ) range covering the worldspace's actual authored
/// cells (min_cell_x..max_cell_x, min_cell_y..max_cell_y) — the only region
/// `ground.level()`/`cover_class()` hold real generated terrain for. Used to
/// clamp queries for tiles (or the parts of tiles) outside this range —
/// see `generate_lod_assets`'s doc comment.
fn valid_block_range(
    min_cell_x: i32,
    max_cell_x: i32,
    min_cell_y: i32,
    max_cell_y: i32,
    x_offset: i32,
    y_offset: i32,
    num_rows: i32,
) -> (i32, i32, i32, i32) {
    let ax_min = cell_to_block(min_cell_x, x_offset);
    let ax_max = cell_to_block(max_cell_x + 1, x_offset);
    let az_min = cell_to_block((num_rows - 1 - y_offset) - max_cell_y, 0);
    let az_max = cell_to_block((num_rows - 1 - y_offset) - min_cell_y + 1, 0);
    (ax_min, ax_max, az_min, az_max)
}

/// Sample world-space terrain height at an arbitrary arnis block coordinate
/// (spanning multiple cells, unlike `sample_heights` in fnv_esm.rs which is
/// scoped to one cell). Phase 1 uses the raw `ground.level()` value directly
/// (no cross-cell smoothing pass) — an acceptable simplification for
/// low-detail LOD terrain, unlike the smoothed near-view LAND mesh.
fn sample_height_game_z(ground: &Ground, ax: i32, az: i32, global_min: i32, effective_scale: i32) -> f32 {
    let raw_h = ground.level(XZPoint::new(ax, az));
    ((raw_h - global_min) * effective_scale + crate::fnv_esm::HEIGHT_MARGIN) as f32 * 8.0
}

/// Build every LOD tile (across all 4 levels) covering the generated
/// worldspace, plus the `.dlodsettings` file. Returns `(relative_path,
/// bytes)` pairs ready to write under the output directory root (paths use
/// forward slashes; caller should convert to the platform separator).
///
/// Tiles are generated only for the area overlapping authored cells
/// (`min_cell_x..max_cell_x`, `min_cell_y..max_cell_y`), anchored at
/// `min_cell_x`/`min_cell_y` and stepping by `level` — not the entire
/// `grid_size x grid_size` square real xLODGen populates (confirmed via its
/// tile counts: 4/16/64/256 for level32/16/8/4, spanning far beyond Rome's
/// actual terrain). Generating the full square was an earlier theory for a
/// load hang that turned out to be caused by something else entirely (a
/// missing NIF footer — see `build_lod_nif`); once that was fixed, the
/// full-grid tiles proved unnecessary, so this reverts to only the tiles
/// that actually carry real terrain. A tile at the worldspace's edge can
/// still individually extend past authored cells (see `tile_block_range` —
/// tiles are always the full nominal square, never clipped), so
/// `valid_block_range`-clamped queries (edge-extrapolated, not garbage or a
/// placeholder path) are still needed for those partial-overlap tiles.
#[allow(clippy::too_many_arguments)]
pub fn generate_lod_assets(
    ground: &Ground,
    worldspace_edid: &str,
    min_cell_x: i32,
    max_cell_x: i32,
    min_cell_y: i32,
    max_cell_y: i32,
    x_offset: i32,
    y_offset: i32,
    num_rows: i32,
    global_min: i32,
    effective_scale: i32,
) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let edid_lower = worldspace_edid.to_lowercase();
    let valid_range = valid_block_range(
        min_cell_x, max_cell_x, min_cell_y, max_cell_y, x_offset, y_offset, num_rows,
    );

    for &level in &[4i32, 8, 16, 32] {
        let mut tx = min_cell_x;
        while tx <= max_cell_x {
            let mut ty = min_cell_y;
            while ty <= max_cell_y {
                let tile = LodTile {
                    level: level as u32,
                    tile_cell_x: tx,
                    tile_cell_y: ty,
                };
                let (ax_min, ax_max, az_min, az_max) =
                    tile_block_range(&tile, x_offset, y_offset, num_rows);
                let (verts, tris) = build_tile_geometry(
                    ground, ax_min, ax_max, az_min, az_max,
                    x_offset, y_offset, num_rows, global_min, effective_scale, valid_range,
                );
                // ".N." (capital), matching the real reference tile's embedded
                // texture path exactly (a real, if likely harmless given
                // Windows' case-insensitive filesystem, discrepancy).
                let diffuse_path = format!(
                    "Data\\Textures\\Landscape\\LOD\\{}\\Diffuse\\{}.N.Level{}.X{}.Y{}.dds",
                    worldspace_edid, worldspace_edid, level, tx, ty
                );
                let normal_path = format!(
                    "Data\\Textures\\Landscape\\LOD\\{}\\Normals\\{}.N.Level{}.X{}.Y{}.dds",
                    worldspace_edid, worldspace_edid, level, tx, ty
                );
                let tile_origin = (
                    tx as f32 * crate::fnv_esm::CELL_GAME_UNITS,
                    ty as f32 * crate::fnv_esm::CELL_GAME_UNITS,
                );
                let nif = build_lod_nif(&verts, &tris, &diffuse_path, &normal_path, tile_origin);
                out.push((
                    format!(
                        "meshes/landscape/lod/{}/{}.level{}.x{}.y{}.nif",
                        edid_lower, edid_lower, level, tx, ty
                    ),
                    nif,
                ));

                let (diffuse_dds, normal_dds) =
                    bake_tile_textures(ground, ax_min, ax_max, az_min, az_max, valid_range);
                out.push((
                    format!(
                        "textures/landscape/lod/{}/diffuse/{}.n.level{}.x{}.y{}.dds",
                        edid_lower, edid_lower, level, tx, ty
                    ),
                    diffuse_dds,
                ));
                out.push((
                    format!(
                        "textures/landscape/lod/{}/normals/{}.n.level{}.x{}.y{}.dds",
                        edid_lower, edid_lower, level, tx, ty
                    ),
                    normal_dds,
                ));
                ty += level;
            }
            tx += level;
        }
    }

    out.push((
        format!("lodsettings/{}.dlodsettings", edid_lower),
        build_dlodsettings(min_cell_x, max_cell_x, min_cell_y, max_cell_y),
    ));

    out
}

/// Real average RGB colors extracted from the actual vanilla FNV diffuse
/// textures referenced by arnis's 5 known texture-set FormIDs
/// (`TEXTURE_GRASS`/`ASPHALT`/`SNOW`/`SAND`/`DIRT` in fnv_esm.rs). Resolved
/// by decoding each FormID's `LTEX` -> `TNAM` -> `TXST` -> `TX00` chain in
/// the real `FalloutNV.esm`, then averaging the referenced `.dds` (BC1)
/// texture's block reference colors:
///   GRASS   Landscape\GrassGreenSuburb01.dds       -> (115,136,53)
///   ASPHALT Landscape\Asphalt02.dds                -> (70,81,76)
///   SNOW    Landscape\DLCAnch\..._softsnow.dds      -> (190,194,190)
///   SAND    Landscape\WaterGravelSandNV01.dds       -> (139,137,125)
///   DIRT    Landscape\DirtWasteland01.dds           -> (94,89,77)
/// Hardcoded rather than read from the user's BSA at runtime: these are
/// fixed vanilla assets that never change, so a one-time extraction avoids
/// adding BSA-reading fragility (install path detection, archive format
/// edge cases) for a value that's already a constant.
fn texture_color_for_cover(lc: u8) -> [u8; 3] {
    match lc {
        crate::land_cover::LC_SNOW_ICE => [190, 194, 190],
        crate::land_cover::LC_BUILT_UP => [70, 81, 76],
        crate::land_cover::LC_BARE => [139, 137, 125],
        crate::land_cover::LC_CROPLAND
        | crate::land_cover::LC_GRASSLAND
        | crate::land_cover::LC_SHRUBLAND
        | crate::land_cover::LC_TREE_COVER => [115, 136, 53],
        _ => [94, 89, 77], // dirt/default
    }
}

const ATLAS_SIZE: usize = 256;

/// Bake this tile's diffuse (land-cover average color per pixel) and normal
/// (flat up-vector, Phase 1) texture atlases, BC1-compressed with a full
/// mip chain, matching real LOD atlases' 256x256/9-mip/DXT1 format.
fn bake_tile_textures(
    ground: &Ground,
    ax_min: i32,
    ax_max: i32,
    az_min: i32,
    az_max: i32,
    valid_range: (i32, i32, i32, i32),
) -> (Vec<u8>, Vec<u8>) {
    let (valid_ax_min, valid_ax_max, valid_az_min, valid_az_max) = valid_range;
    // px=0 -> ax_max (east), py=0 -> az_max (south) — matches real xLODGen's
    // UV convention exactly (re-derived from decoding a real tile's own
    // corner UVs: NW->(1,1), NE->(0,1), SW->(1,0), SE->(0,0), i.e. u
    // decreases west->east and v increases south->north). An earlier
    // version used the naive px=0->west/py=0->north mapping, which was
    // internally self-consistent with its own (also naive) UV formula but
    // didn't match the proven-working reference — this is what the user
    // saw as the mesh's texture looking rotated 180 degrees.
    let mut diffuse_rgba = vec![0u8; ATLAS_SIZE * ATLAS_SIZE * 4];
    for py in 0..ATLAS_SIZE {
        let az = (az_max - (az_max - az_min) * py as i32 / ATLAS_SIZE as i32)
            .clamp(valid_az_min, valid_az_max);
        for px in 0..ATLAS_SIZE {
            let ax = (ax_max - (ax_max - ax_min) * px as i32 / ATLAS_SIZE as i32)
                .clamp(valid_ax_min, valid_ax_max);
            let lc = ground.cover_class(XZPoint::new(ax, az));
            let [r, g, b] = texture_color_for_cover(lc);
            let i = (py * ATLAS_SIZE + px) * 4;
            diffuse_rgba[i] = r;
            diffuse_rgba[i + 1] = g;
            diffuse_rgba[i + 2] = b;
            diffuse_rgba[i + 3] = 255;
        }
    }
    // Flat up-vector normal map (Phase 1 — real gradient-derived normals are
    // a reasonable follow-up, not required for a working, correct-format
    // Phase 1; see this module's doc comment).
    let mut normal_rgba = vec![0u8; ATLAS_SIZE * ATLAS_SIZE * 4];
    for px in normal_rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&[128, 128, 255, 255]);
    }

    (
        encode_dds_bc1(&diffuse_rgba, ATLAS_SIZE, ATLAS_SIZE),
        encode_dds_bc1(&normal_rgba, ATLAS_SIZE, ATLAS_SIZE),
    )
}

/// Downsample an RGBA8 image by exactly half (simple 2x2 box filter),
/// for mip chain generation.
fn downsample_half(rgba: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let nw = (width / 2).max(1);
    let nh = (height / 2).max(1);
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let sx = (x * 2 + dx).min(width - 1);
                let sy = (y * 2 + dy).min(height - 1);
                let si = (sy * width + sx) * 4;
                for c in 0..4 {
                    acc[c] += rgba[si + c] as u32;
                }
            }
            let oi = (y * nw + x) * 4;
            for c in 0..4 {
                out[oi + c] = (acc[c] / 4) as u8;
            }
        }
    }
    (out, nw, nh)
}

/// Encode an RGBA8 image as a full-mip-chain BC1 (DXT1) `.dds` file, matching
/// the header layout of real FNV LOD texture atlases.
fn encode_dds_bc1(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut mips = Vec::new();
    let (mut cur, mut w, mut h) = (rgba.to_vec(), width, height);
    loop {
        mips.push((cur.clone(), w, h));
        if w == 1 && h == 1 {
            break;
        }
        let (next, nw, nh) = downsample_half(&cur, w, h);
        cur = next;
        w = nw;
        h = nh;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"DDS ");
    pu32(&mut out, 124); // header size
    pu32(&mut out, 0x000A1007); // CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT|LINEARSIZE
    pu32(&mut out, height as u32);
    pu32(&mut out, width as u32);
    let first_mip_size = texpresso::Format::Bc1.compressed_size(width, height);
    pu32(&mut out, first_mip_size as u32); // pitchOrLinearSize
    pu32(&mut out, 1); // depth (unused since DDSD_DEPTH isn't in flags, but real files write 1)
    pu32(&mut out, mips.len() as u32); // mipMapCount
    for _ in 0..11 {
        pu32(&mut out, 0); // reserved1
    }
    // pixel format (32 bytes)
    pu32(&mut out, 32); // size
    pu32(&mut out, 0x4); // DDPF_FOURCC
    out.extend_from_slice(b"DXT1");
    for _ in 0..5 {
        pu32(&mut out, 0); // RGBBitCount + 4 masks
    }
    pu32(&mut out, 0x00401008); // caps: TEXTURE|MIPMAP|COMPLEX
    pu32(&mut out, 0); // caps2
    pu32(&mut out, 0); // caps3
    pu32(&mut out, 0); // caps4
    pu32(&mut out, 0); // reserved2

    for (mip_rgba, mw, mh) in &mips {
        let size = texpresso::Format::Bc1.compressed_size(*mw, *mh);
        let mut block = vec![0u8; size];
        texpresso::Format::Bc1.compress(
            mip_rgba,
            *mw,
            *mh,
            texpresso::Params::default(),
            &mut block,
        );
        out.extend_from_slice(&block);
    }
    out
}

/// Compute this tile's arnis block-space (AX, AZ) range: always the full
/// nominal `level`-cell square, never clipped to the authored cell range.
/// Confirmed against real xLODGen output: a tile at the worldspace's edge
/// (e.g. Rome's X4.Y7, spanning cells 7-10 when the worldspace's own
/// max_cell_y is 9) is still a full 16384x16384 square in the reference
/// file, not a clipped rectangle — xLODGen samples straight through the
/// authored-cell boundary rather than truncating. An earlier version of
/// this function clipped to min/max_cell_x/y, producing a rectangular tile
/// exactly at these edge positions, which the user caught by direct
/// comparison against the reference file. `ground.level()`/`cover_class()`
/// are safe to query somewhat past the authored range (the terrain data
/// already carries smoothing padding for this reason — see SMOOTH_PAD in
/// fnv_esm.rs), and every tile visited by the generation loop overlaps the
/// authored range by construction, so no clipping/None case is needed here.
fn tile_block_range(tile: &LodTile, x_offset: i32, y_offset: i32, num_rows: i32) -> (i32, i32, i32, i32) {
    let level = tile.level as i32;
    let cell_x_start = tile.tile_cell_x;
    let cell_x_end = tile.tile_cell_x + level - 1;
    let cell_y_start = tile.tile_cell_y;
    let cell_y_end = tile.tile_cell_y + level - 1;
    let ax_min = cell_to_block(cell_x_start, x_offset);
    let ax_max = cell_to_block(cell_x_end + 1, x_offset); // exclusive end -> east edge
    let az_min = cell_to_block((num_rows - 1 - y_offset) - cell_y_end, 0); // north edge (largest cell_y = smallest AZ)
    let az_max = cell_to_block((num_rows - 1 - y_offset) - cell_y_start + 1, 0); // south edge, exclusive end
    (ax_min, ax_max, az_min, az_max)
}

/// Sample this tile's uniform `LOD_TILE_QUADS` x `LOD_TILE_QUADS` grid over
/// the given arnis block-space range (see `tile_block_range`). `valid_range`
/// is the worldspace's authored cell range in block-space (see
/// `valid_block_range`) — height/cover queries are clamped into it, since
/// tiles (or parts of tiles) outside the authored range don't have real
/// generated terrain and must not query `ground` outside where it's valid;
/// clamping edge-extrapolates the boundary terrain outward instead of
/// needing a separate placeholder code path.
#[allow(clippy::too_many_arguments)]
fn build_tile_geometry(
    ground: &Ground,
    ax_min: i32,
    ax_max: i32,
    az_min: i32,
    az_max: i32,
    x_offset: i32,
    y_offset: i32,
    num_rows: i32,
    global_min: i32,
    effective_scale: i32,
    valid_range: (i32, i32, i32, i32),
) -> (Vec<LodVertex>, Vec<LodTriangle>) {
    let (valid_ax_min, valid_ax_max, valid_az_min, valid_az_max) = valid_range;
    let n = LOD_TILE_QUADS;
    let mut verts = Vec::with_capacity((n + 1) * (n + 1));
    for r in 0..=n {
        let az = az_min + (az_max - az_min) * r as i32 / n as i32;
        for c in 0..=n {
            let ax = ax_min + (ax_max - ax_min) * c as i32 / n as i32;
            let game_x = ax as f32 * 128.0 - x_offset as f32 * crate::fnv_esm::CELL_GAME_UNITS;
            let game_y = (num_rows - y_offset) as f32 * crate::fnv_esm::CELL_GAME_UNITS
                - az as f32 * 128.0;
            let sample_ax = ax.clamp(valid_ax_min, valid_ax_max);
            let sample_az = az.clamp(valid_az_min, valid_az_max);
            let game_z = sample_height_game_z(ground, sample_ax, sample_az, global_min, effective_scale);
            verts.push((game_x, game_y, game_z));
        }
    }

    // r indexes AZ (increasing r -> increasing az -> DECREASING world Y, so
    // r=0 is the tile's north edge, r=n is south); c indexes AX (increasing
    // c -> increasing world X, so c=0 is west, c=n is east). So vidx(r,c) is
    // NW at (0,0) and the two triangles below are NW,SE,NE and NW,SW,SE —
    // verified CCW viewed from above (+Z) by explicit cross-product check,
    // unlike an earlier version of this code which put SE before NE (CW,
    // culled as invisible from above — exactly the "transparent from above,
    // wrong from below" symptom seen when testing against real xLODGen
    // output).
    let vidx = |r: usize, c: usize| -> u16 { (r * (n + 1) + c) as u16 };
    let mut tris = Vec::with_capacity(n * n * 2);
    for r in 0..n {
        for c in 0..n {
            tris.push((vidx(r, c), vidx(r + 1, c + 1), vidx(r, c + 1)));
            tris.push((vidx(r, c), vidx(r + 1, c), vidx(r + 1, c + 1)));
        }
    }

    add_skirt(&mut verts, &mut tris, n, vidx, compute_skirt_floor_z());
    (verts, tris)
}

/// Margin (game units) the global skirt floor sits below the worldspace's
/// own lowest possible terrain point. Still not derived from real data (real
/// xLODGen's own floor, -14098, is presumably tied to its internal defaults
/// rather than our procedural terrain range) — but the *shape* of the fix
/// (one absolute floor shared by the whole worldspace) is confirmed: every
/// real tile across all 4 levels and different positions shares the
/// identical skirt-bottom Z (-14098.0), never a per-tile-relative depth.
const SKIRT_FLOOR_MARGIN: f32 = 8192.0;

/// The single Z value every tile's skirt hangs down to, computed once for
/// the whole worldspace (not per-tile) — see `SKIRT_FLOOR_MARGIN`'s doc
/// comment for why this must be a shared constant rather than each tile
/// computing its own relative depth (an earlier version of this code did
/// exactly that, `tile_min_z - SKIRT_DEPTH`, which could leave gaps between
/// adjacent tiles' skirts whenever their local terrain heights differed).
/// `sample_height_game_z`'s lowest possible output (at `raw_h == global_min`)
/// is always `HEIGHT_MARGIN * 8.0`, independent of `global_min` and
/// `effective_scale` (both cancel out at that point), so this needs no
/// worldspace-specific inputs.
fn compute_skirt_floor_z() -> f32 {
    (crate::fnv_esm::HEIGHT_MARGIN as f32) * 8.0 - SKIRT_FLOOR_MARGIN
}

/// Extend the tile mesh with vertical "skirt" wall geometry hanging down
/// from all 4 outer edges, matching real xLODGen output (its
/// `BSSegmentedTriShape` blocks serve the same gap-hiding purpose, though we
/// fold the skirt into the same shared NiTriShapeData rather than a separate
/// block/segment system). Without this, a flat top-only tile shows a gap
/// into the void wherever it doesn't exactly meet neighboring geometry.
///
/// Winding for each edge was derived independently per edge (not assumed
/// symmetric) via explicit 3D cross-product sign checks, since the same
/// "forward" vertex order gives an outward-pointing normal on the
/// north/east edges but an inward-pointing one on south/west — verified
/// against the already cross-product-validated top-surface winding fix.
fn add_skirt(
    verts: &mut Vec<LodVertex>,
    tris: &mut Vec<LodTriangle>,
    n: usize,
    vidx: impl Fn(usize, usize) -> u16,
    skirt_z: f32,
) {
    // North edge (r=0, outward = +Y): "forward" winding.
    let base = verts.len() as u16;
    for c in 0..=n {
        let (x, y, _) = verts[vidx(0, c) as usize];
        verts.push((x, y, skirt_z));
    }
    let skirt = |i: usize| -> u16 { base + i as u16 };
    for c in 0..n {
        tris.push((vidx(0, c), vidx(0, c + 1), skirt(c + 1)));
        tris.push((vidx(0, c), skirt(c + 1), skirt(c)));
    }

    // South edge (r=n, outward = -Y): reversed winding.
    let base = verts.len() as u16;
    for c in 0..=n {
        let (x, y, _) = verts[vidx(n, c) as usize];
        verts.push((x, y, skirt_z));
    }
    let skirt = |i: usize| -> u16 { base + i as u16 };
    for c in 0..n {
        tris.push((vidx(n, c), skirt(c + 1), vidx(n, c + 1)));
        tris.push((vidx(n, c), skirt(c), skirt(c + 1)));
    }

    // West edge (c=0, outward = -X): reversed winding.
    let base = verts.len() as u16;
    for r in 0..=n {
        let (x, y, _) = verts[vidx(r, 0) as usize];
        verts.push((x, y, skirt_z));
    }
    let skirt = |i: usize| -> u16 { base + i as u16 };
    for r in 0..n {
        tris.push((vidx(r, 0), skirt(r + 1), vidx(r + 1, 0)));
        tris.push((vidx(r, 0), skirt(r), skirt(r + 1)));
    }

    // East edge (c=n, outward = +X): "forward" winding.
    let base = verts.len() as u16;
    for r in 0..=n {
        let (x, y, _) = verts[vidx(r, n) as usize];
        verts.push((x, y, skirt_z));
    }
    let skirt = |i: usize| -> u16 { base + i as u16 };
    for r in 0..n {
        tris.push((vidx(r, n), vidx(r + 1, n), skirt(r + 1)));
        tris.push((vidx(r, n), skirt(r + 1), skirt(r)));
    }
}

// --- byte helpers (mirrors fnv_esm.rs's pu16/pu32/pi32/pf32) ---

fn pu16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn pu32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn pi32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn pf32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn pbool(buf: &mut Vec<u8>, v: bool) {
    buf.push(if v { 1 } else { 0 });
}
/// NIF "sized string": u32 length prefix + raw bytes (no null terminator).
fn sized_string(buf: &mut Vec<u8>, s: &str) {
    pu32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

const NIF_VERSION: u32 = 335_675_399; // "20.2.0.7" packed
const NIF_USER_VERSION: u32 = 11;
const NIF_USER_VERSION2: u32 = 34;

/// One vertex: position only (Phase 1 has no normals/vertex colors, matching
/// `hasNormals=false` seen in real `wastelandnv` LOD tiles at this scale).
pub type LodVertex = (f32, f32, f32);
pub type LodTriangle = (u16, u16, u16);

/// Build the content bytes for a `NiObjectNET`-family object's shared prefix
/// (nameIdx/extraData/controller). `name_idx` references the header's global
/// string table (-1 = no name). extraData/controller are always empty/absent
/// — none of our blocks need those.
///
/// Root-caused via a real NifSkope repro: running every "Sanitize" spell on
/// our hanging tile made it load successfully; diffing the sanitized file
/// against ours byte-for-byte (after fixing an offset bug in the diff
/// script itself, which hadn't accounted for a populated string table) showed
/// literally one difference in the entire file — `NiTriShape`'s nameIdx was
/// -1 (ours) vs 0 (sanitized, referencing a real header string). Every other
/// block, including another nameIdx=-1 on `BSMultiBoundNode`, was identical.
/// So an anonymous `NiTriShape` is specifically what the engine chokes on;
/// the exact name string doesn't appear to matter (NifSkope's sanitize just
/// used its own default), so any real non-empty header string works.
fn ni_object_net_prefix(buf: &mut Vec<u8>, name_idx: i32) {
    pi32(buf, name_idx);
    pu32(buf, 0); // numExtraData
    pi32(buf, -1); // controller (none)
}

/// `NiAVObject` fields (flags/flags2/translation/rotation/scale/properties/
/// collisionObject) shared by `NiNode`/`NiGeometry`-derived blocks.
/// `properties` is a list of block indices (e.g. the shader property
/// attached to a `NiTriShape`). `translation` places this node in its
/// parent's space — see `build_bs_multi_bound_node`'s doc comment for why
/// this matters for the root node specifically. `flags` is caller-supplied
/// (not a shared constant) since real reference tiles show it differs by
/// block type: `BSMultiBoundNode` needs 2062 (0x080E), `NiTriShape` needs 14
/// (0x000E) — confirmed by decoding all 4 levels at the origin, uniformly.
/// `flags2` is 8 in every real sample for both block types (we previously
/// hardcoded 0 here, an uncaught bug present in every tile regardless of
/// level).
fn ni_av_object_fields(buf: &mut Vec<u8>, properties: &[u32], translation: (f32, f32, f32), flags: u16) {
    pu16(buf, flags);
    pu16(buf, 8); // flags2 (present: userVersion>=11 && userVersion2>=26)
    pf32(buf, translation.0);
    pf32(buf, translation.1);
    pf32(buf, translation.2);
    // rotation: 3x3 identity matrix, row-major f32
    for row in 0..3 {
        for col in 0..3 {
            pf32(buf, if row == col { 1.0 } else { 0.0 });
        }
    }
    pf32(buf, 1.0); // scale
    // properties list (present: userVersion<=11)
    pu32(buf, properties.len() as u32);
    for &p in properties {
        pi32(buf, p as i32);
    }
    pi32(buf, -1); // collisionObject (none)
}

/// `BSMultiBoundNode` (extends `NiNode` extends `NiAVObject`): the root node
/// of the tile, with the terrain `NiTriShape` as its one child and a
/// `BSMultiBound` bounding-volume reference.
///
/// `translation` places the whole tile in world space: `(tile_cell_x,
/// tile_cell_y) * CELL_GAME_UNITS`. Real xLODGen tiles always bake
/// tile-LOCAL vertex data (X/Y in ~0..16384 regardless of where the tile
/// actually sits) and carry the tile's world position here instead —
/// confirmed by decoding real tiles' translation directly: X0.Y-1 ->
/// (0,-4096,0), X0.Y-9 -> (0,-36864,0), X4.Y-1 -> (16384,-4096,0), each
/// exactly (tileX,tileY)*4096. An earlier version of this generator baked
/// already-absolute world coordinates into the vertices with translation
/// left at (0,0,0) instead — harmless for the one tile at the world origin
/// (where local and absolute coincide, which is why it looked fine in
/// isolated NifSkope inspection) but wrong for every other tile, handing
/// the engine's LOD spatial system out-of-expected-range "local" bounding
/// data for nearly the entire worldspace. Height (Z) is NOT translated —
/// real tiles keep translation.z=0 and bake absolute Z directly into each
/// vertex, since height varies per-vertex rather than being a per-tile
/// constant offset.
fn build_bs_multi_bound_node(child_tri_shape_idx: u32, multi_bound_idx: u32, translation: (f32, f32, f32)) -> Vec<u8> {
    let mut buf = Vec::new();
    ni_object_net_prefix(&mut buf, -1);
    ni_av_object_fields(&mut buf, &[], translation, 2062); // flags: real value, not the generic NiAVObject default
    // NiNode: children list
    pu32(&mut buf, 1);
    pi32(&mut buf, child_tri_shape_idx as i32);
    // numEffects (present: userVersion2 < 130)
    pu32(&mut buf, 0);
    // BSMultiBoundNode: multiBound (userVersion < 12, so no cullMode field)
    pi32(&mut buf, multi_bound_idx as i32);
    buf
}

/// `NiTriShape` (extends `NiTriBasedGeom` extends `NiGeometry` extends
/// `NiAVObject`): the mesh instance, referencing its shader property and its
/// `NiTriShapeData` geometry block.
fn build_ni_tri_shape(shader_property_idx: u32, data_idx: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    ni_object_net_prefix(&mut buf, 0); // nameIdx=0 -> header string table entry 0 (see doc comment above)
    ni_av_object_fields(&mut buf, &[shader_property_idx], (0.0, 0.0, 0.0), 14);
    // NiGeometry (version > 335544325, so the "numMaterials" branch applies;
    // userVersion != 12, so no bsProperties):
    pi32(&mut buf, data_idx as i32); // data -> NiTriShapeData
    pi32(&mut buf, -1); // skinInstance (none)
    pu32(&mut buf, 0); // numMaterials
    pi32(&mut buf, 0); // activeMaterial
    pbool(&mut buf, false); // dirtyFlag
    buf
}

/// `BSShaderPPLightingProperty` (extends `NiProperty` extends
/// `NiObjectNET`): references the texture set; defaults corrected against a
/// real reference tile's raw bytes (shaderType=1 "Lighting", envmapScale=1.0,
/// etc). Two fields were previously wrong, caught by direct NifSkope
/// side-by-side inspection: `shaderFlags` was 8192 (0x2000, bit 13 only) but
/// real is 12288 (0x3000, bits 12+13 — NifSkope labels bit 12 "Unknown_3");
/// and the second flags field was 4 ("LOD_Building") but real is 2
/// ("LOD_Landscape") — telling the engine this is building rather than
/// terrain LOD is a very plausible contributor to the load hang.
fn build_bs_shader_pp_lighting_property(texture_set_idx: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    ni_object_net_prefix(&mut buf, -1);
    pu16(&mut buf, 1); // Flags
    pu32(&mut buf, 1); // shaderType = Lighting
    pu32(&mut buf, 12288); // shaderFlags (bits 12+13: Unknown_3 | ZBuffer_Test or similar)
    pi32(&mut buf, 2); // shaderFlags2 = LOD_Landscape
    pf32(&mut buf, 1.0); // envmapScale
    pi32(&mut buf, 0); // unknownInt3
    pi32(&mut buf, texture_set_idx as i32); // textureSet
    pf32(&mut buf, 0.0); // unknownFloat2
    pi32(&mut buf, 0); // refractionPeriod
    pf32(&mut buf, 8.0); // unknownFloat4
    pf32(&mut buf, 1.0); // unknownFloat5
    buf
}

/// `BSShaderTextureSet` (extends `NiObject` directly — no NiObjectNET
/// prefix). Slot 0 = diffuse, slot 1 = normal map, matching real LOD tiles'
/// `Diffuse`/`Normals` texture path pair.
fn build_bs_shader_texture_set(diffuse_path: &str, normal_path: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    let textures = [diffuse_path, normal_path, "", "", "", ""];
    pi32(&mut buf, textures.len() as i32); // numTextures
    for t in textures {
        sized_string(&mut buf, t);
    }
    buf
}

/// `NiAdditionalGeometryData` (extends `NiObject` directly). Reverse-engineered
/// byte-for-byte from two real reference tiles (270 and 210 vertices) since
/// no source exists for this block in xLODGen (it's GECK-native): the block
/// is a fixed 56-byte header template with the vertex count and
/// `vertexCount*4` substituted in at 3 positions, followed by exactly one
/// f32 per vertex. That per-vertex float was confirmed to be an *exact*
/// (zero-diff, every vertex) copy of the vertex's own Z height from the
/// sibling `NiTriShapeData` — cross-checked against both samples with no
/// discrepancy. The header's other fields have no established semantic
/// meaning (no schema found), but since both real samples match this
/// template byte-for-byte outside the 3 substituted fields, reproducing it
/// verbatim is safe.
fn build_ni_additional_geometry_data(verts: &[LodVertex]) -> Vec<u8> {
    let mut buf = Vec::new();
    let n = verts.len() as u16;
    let n4 = verts.len() as u32 * 4;
    pu16(&mut buf, n);
    pu16(&mut buf, 1);
    pu16(&mut buf, 0);
    pu16(&mut buf, 1);
    pu16(&mut buf, 0);
    pu16(&mut buf, 4);
    pu16(&mut buf, 0);
    pu32(&mut buf, n4);
    pu16(&mut buf, 4);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 258);
    pu16(&mut buf, 0);
    pu16(&mut buf, 256);
    pu32(&mut buf, n4);
    pu16(&mut buf, 1);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 0);
    pu16(&mut buf, 1);
    pu16(&mut buf, 0);
    pu16(&mut buf, 4);
    pu16(&mut buf, 0);
    for &(_, _, z) in verts {
        pf32(&mut buf, z);
    }
    buf
}

/// `NiTriShapeData` (extends `NiTriBasedGeomData` extends `NiGeometryData`
/// extends `NiObject` directly). `additional_data_idx` references the
/// sibling `NiAdditionalGeometryData` block (see its doc comment) — every
/// real reference tile has this set, matching `userVersion == 11`
/// (skyrimMaterial field is version==12-only, so absent here).
///
/// `top_surface_count` is how many of `verts`' leading entries are the
/// actual top surface, with the rest being skirt geometry — the bounding
/// center/radius are computed from the top surface only, matching real
/// tiles exactly (decoded directly: a real tile's center.z and radius match
/// its top surface's own Z range precisely, excluding the skirt entirely,
/// which hangs an arbitrary/unverified depth below and would otherwise
/// wildly inflate the radius and drag the center down).
fn build_ni_tri_shape_data(
    verts: &[LodVertex],
    tris: &[LodTriangle],
    additional_data_idx: u32,
    top_surface_count: usize,
) -> Vec<u8> {
    let mut buf = Vec::new();
    let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
    for &(x, y, z) in verts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        max_z = max_z.max(z);
    }
    let top = &verts[..top_surface_count];
    let (mut sx, mut sy, mut sz) = (0.0f32, 0.0f32, 0.0f32);
    for &(x, y, z) in top {
        sx += x;
        sy += y;
        sz += z;
    }
    let n = top.len().max(1) as f32;
    let center = (sx / n, sy / n, sz / n);
    let radius = top
        .iter()
        .map(|&(x, y, z)| {
            let (dx, dy, dz) = (x - center.0, y - center.1, z - center.2);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max);

    pi32(&mut buf, 0); // unknownInt
    pu16(&mut buf, verts.len() as u16); // numVertices
    buf.push(0); // keepFlags
    buf.push(0); // compressFlags
    pbool(&mut buf, true); // hasVertices
    for &(x, y, z) in verts {
        pf32(&mut buf, x);
        pf32(&mut buf, y);
        pf32(&mut buf, z);
    }
    buf.push(1); // numUVSets = 1 — required: this shape has an assigned
                 // diffuse/normal texture set (BSShaderTextureSet), and a
                 // textured mesh with zero UV data is invalid. An earlier
                 // version of this generator shipped numUVSets=0 here,
                 // which is almost certainly what caused the "sunken
                 // terrain" rendering and the load hang on a larger
                 // worldspace reported during testing — every real
                 // GECK/xLODGen-generated tile has numUVSets=1.
    buf.push(0); // extraVectorFlags
    pbool(&mut buf, false); // hasNormals
    pf32(&mut buf, center.0);
    pf32(&mut buf, center.1);
    pf32(&mut buf, center.2);
    pf32(&mut buf, radius);
    pbool(&mut buf, false); // hasVertexColors
    // numUVSets & 1 == 1: one planar UV coordinate per vertex, mapping the
    // tile's own world-space extent onto the single 256x256 diffuse/normal
    // atlas baked for this tile. u decreases west->east, v increases
    // south->north, matching bake_tile_textures's px=0->east/py=0->south
    // raster and real xLODGen's own UV convention exactly (re-derived from
    // a real tile's corner UVs: NW->(1,1), NE->(0,1), SW->(1,0), SE->(0,0)).
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    for &(x, y, _z) in verts {
        let u = (max_x - x) / span_x;
        let v = (y - min_y) / span_y;
        pf32(&mut buf, u);
        pf32(&mut buf, v);
    }
    pu16(&mut buf, 0); // consistencyFlags
    pi32(&mut buf, additional_data_idx as i32); // additionalData -> NiAdditionalGeometryData
    // NiTriBasedGeomData:
    pu16(&mut buf, tris.len() as u16); // numTriangles
    // NiTriShapeData:
    pu32(&mut buf, tris.len() as u32 * 3); // numTrianglePoints
    pbool(&mut buf, true); // hasTriangles
    for &(a, b, c) in tris {
        pu16(&mut buf, a);
        pu16(&mut buf, b);
        pu16(&mut buf, c);
    }
    pu16(&mut buf, 0); // numMatchGroups
    buf
}

/// `BSMultiBound` (extends `NiObject` directly): references the bounding
/// volume data block (`BSMultiBoundAABB`).
fn build_bs_multi_bound(aabb_idx: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    pi32(&mut buf, aabb_idx as i32);
    buf
}

/// `BSMultiBoundAABB` (extends `BSMultiBoundData` extends `NiObject`
/// directly): axis-aligned bounding box as center + half-extent.
fn build_bs_multi_bound_aabb(verts: &[LodVertex]) -> Vec<u8> {
    let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
    for &(x, y, z) in verts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        max_z = max_z.max(z);
    }
    let mut buf = Vec::new();
    pf32(&mut buf, (min_x + max_x) / 2.0);
    pf32(&mut buf, (min_y + max_y) / 2.0);
    pf32(&mut buf, (min_z + max_z) / 2.0);
    pf32(&mut buf, (max_x - min_x) / 2.0);
    pf32(&mut buf, (max_y - min_y) / 2.0);
    pf32(&mut buf, (max_z - min_z) / 2.0);
    buf
}

/// Build a full terrain LOD `.nif` file: `BSMultiBoundNode` root containing
/// one textured `NiTriShape`, plus its `BSMultiBound`/`BSMultiBoundAABB`
/// bounding volume. Block order matches real files' convention (data blocks
/// after their owning shape).
///
/// `verts` are in absolute world space (X/Y/Z); `tile_origin` is this tile's
/// world position (`(tile_cell_x, tile_cell_y) * CELL_GAME_UNITS`). Vertices
/// are re-based to tile-local X/Y here (Z stays absolute) and `tile_origin`
/// becomes the root node's translation — see `build_bs_multi_bound_node`'s
/// doc comment for why this split (not raw absolute vertices) is required.
///
/// `BSMultiBoundAABB` is the one exception to the local-space rule: decoding
/// a real tile showed its AABB position is in ABSOLUTE world space (matching
/// the tile's true world-space center exactly), unlike every other
/// position/bounding value in the file, which is local. It's also computed
/// from the top-surface vertices only (the leading `(LOD_TILE_QUADS+1)^2` of
/// `verts` — skirt vertices excluded), matching `NiTriShapeData`'s own
/// center/radius fix for the same reason (see `build_ni_tri_shape_data`'s
/// doc comment).
pub fn build_lod_nif(
    verts: &[LodVertex],
    tris: &[LodTriangle],
    diffuse_path: &str,
    normal_path: &str,
    tile_origin: (f32, f32),
) -> Vec<u8> {
    let top_surface_count = ((LOD_TILE_QUADS + 1) * (LOD_TILE_QUADS + 1)).min(verts.len());
    let aabb_verts: Vec<LodVertex> = verts[..top_surface_count].to_vec();

    let local_verts: Vec<LodVertex> = verts
        .iter()
        .map(|&(x, y, z)| (x - tile_origin.0, y - tile_origin.1, z))
        .collect();
    let verts = &local_verts[..];

    // Fixed block order; indices below are hand-resolved to match (no
    // generic block-reference remapping needed since we build the block
    // list directly in final order).
    const IDX_ROOT: u32 = 0; // BSMultiBoundNode, the scene graph's root
    const IDX_TRI_SHAPE: u32 = 1;
    const IDX_SHADER_PROPERTY: u32 = 2;
    const IDX_TEXTURE_SET: u32 = 3;
    const IDX_TRI_SHAPE_DATA: u32 = 4;
    const IDX_ADDITIONAL_GEOMETRY_DATA: u32 = 5;
    const IDX_MULTI_BOUND: u32 = 6;
    const IDX_MULTI_BOUND_AABB: u32 = 7;

    let blocks: Vec<(&'static str, Vec<u8>)> = vec![
        (
            "BSMultiBoundNode",
            build_bs_multi_bound_node(IDX_TRI_SHAPE, IDX_MULTI_BOUND, (tile_origin.0, tile_origin.1, 0.0)),
        ),
        (
            "NiTriShape",
            build_ni_tri_shape(IDX_SHADER_PROPERTY, IDX_TRI_SHAPE_DATA),
        ),
        (
            "BSShaderPPLightingProperty",
            build_bs_shader_pp_lighting_property(IDX_TEXTURE_SET),
        ),
        (
            "BSShaderTextureSet",
            build_bs_shader_texture_set(diffuse_path, normal_path),
        ),
        (
            "NiTriShapeData",
            build_ni_tri_shape_data(verts, tris, IDX_ADDITIONAL_GEOMETRY_DATA, top_surface_count),
        ),
        (
            "NiAdditionalGeometryData",
            build_ni_additional_geometry_data(verts),
        ),
        ("BSMultiBound", build_bs_multi_bound(IDX_MULTI_BOUND_AABB)),
        ("BSMultiBoundAABB", build_bs_multi_bound_aabb(&aabb_verts)),
    ];

    // --- header ---
    let mut header = Vec::new();
    header.extend_from_slice(b"Gamebryo File Format, Version 20.2.0.7\n");
    pu32(&mut header, NIF_VERSION);
    header.push(1); // endianType (little-endian)
    pu32(&mut header, NIF_USER_VERSION);
    pu32(&mut header, blocks.len() as u32); // numBlocks
    pu32(&mut header, NIF_USER_VERSION2);
    // creator / exportInfo1 / exportInfo2: short strings (u8 length + bytes),
    // but null-TERMINATED C-strings, not raw byte content — the length byte
    // includes the trailing \0. Confirmed against the real xLODGen reference
    // (creator="LODGen ...Sheson\0", export1="\0", export2="\0" — i.e. even
    // an "empty" string is length=1, a lone terminator, not length=0). Empty
    // (length=0, no bytes at all) was what we wrote before; that was the
    // last remaining difference between our output and a NifSkope-sanitized
    // file that loads successfully in-game (every block was already
    // byte-identical).
    for s in ["arnis", "", ""] {
        header.push(s.len() as u8 + 1);
        header.extend_from_slice(s.as_bytes());
        header.push(0);
    }
    // exportInfo3 only present when version==335675399 && userVersion2==130 (ours is 34, so absent)

    // Distinct block type name table, in first-seen order (matches how
    // NiHeader.Write emits `blockTypes`/`blockTypeIndices`).
    let mut block_type_order: Vec<&str> = Vec::new();
    let mut block_type_lookup: BTreeMap<&str, u16> = BTreeMap::new();
    for (t, _) in &blocks {
        if !block_type_lookup.contains_key(t) {
            block_type_lookup.insert(t, block_type_order.len() as u16);
            block_type_order.push(t);
        }
    }
    pu16(&mut header, block_type_order.len() as u16); // numBlockTypes
    for t in &block_type_order {
        sized_string(&mut header, t); // NIF block-type strings ARE sized (u32-prefixed), unlike header short-strings
    }
    for (t, _) in &blocks {
        pu16(&mut header, block_type_lookup[*t]);
    }
    for (_, content) in &blocks {
        pu32(&mut header, content.len() as u32); // block size
    }
    // Header string table: NiTriShape's nameIdx (0) references entry 0 here.
    // "NiTransformController" (entry 1) is NOT referenced by any block —
    // confirmed by an exhaustive byte diff: with only entry 0 present, the
    // file still hung; adding this second, unreferenced string (exactly
    // replicating what NifSkope's "Sanitize" spells produced) was the entire
    // remaining difference in a file that then loaded successfully. The
    // real xLODGen reference has zero header strings at all and (as part of
    // its full dataset) works, so this isn't a generally-required NIF
    // convention — more likely a specific parser quirk tied to our exact
    // block content. Not fully understood; kept because it's empirically
    // validated via direct A/B against a working reference, twice.
    let strings: &[&str] = &["ArnisLODMesh", "NiTransformController"];
    pu32(&mut header, strings.len() as u32); // numStrings
    pu32(&mut header, strings.iter().map(|s| s.len()).max().unwrap_or(0) as u32); // maxStringLength
    for s in strings {
        sized_string(&mut header, s);
    }
    pu32(&mut header, 0); // unknownInt2 (groups)

    let mut out = header;
    for (_, content) in &blocks {
        out.extend_from_slice(content);
    }

    // Footer: numRoots + roots[] (block indices of the scene graph's root
    // objects) — a standard NIF trailer we were omitting entirely. Found by
    // noticing a real, working reference file was exactly 8 bytes longer
    // than ours even after every declared block matched byte-for-byte: the
    // extra bytes decode as numRoots=1, roots=[0] (our BSMultiBoundNode, the
    // only top-level object). This — not the earlier header-string
    // changes — may be the actual fix; those are kept since they're
    // harmless, but this missing footer is a well-documented NIF feature we
    // simply never wrote.
    pu32(&mut out, 1); // numRoots
    pi32(&mut out, IDX_ROOT as i32); // roots[0] -> BSMultiBoundNode

    out
}

/// Build a `Data/lodsettings/<worldspace>.dlodsettings` file. 24 bytes,
/// decoded from two real samples (`NukaLand0.dlodsettings`,
/// `LBWorld.dlodsettings`): `min_level`(u32)=4 and `max_level`(u32)=32 are
/// constant across both regardless of worldspace size; `grid_size`(u32) is
/// 32 for a tiny worldspace (cell span well under 32) and 128 for one
/// spanning exactly 128 cells — consistent with "size of the level32 tile
/// grid covering the worldspace, rounded up to a multiple of 32" (i.e. at
/// least one full level32 tile). Cell bounds are written as i16 min/max
/// x/y. Not independently confirmed beyond these 2 samples — low risk given
/// the file's small size, easy to adjust from real GECK/game load testing.
pub fn build_dlodsettings(min_cell_x: i32, max_cell_x: i32, min_cell_y: i32, max_cell_y: i32) -> Vec<u8> {
    const MIN_LEVEL: u32 = 4;
    const MAX_LEVEL: u32 = 32;
    // grid_size shared with generate_lod_assets — see compute_grid_size's
    // doc comment. generate_lod_assets must tile this entire square, not
    // just the worldspace's own raw cell span.
    let grid_size = compute_grid_size(min_cell_x, max_cell_x, min_cell_y, max_cell_y) as u32;

    let mut buf = Vec::new();
    pu32(&mut buf, MIN_LEVEL);
    pu32(&mut buf, MAX_LEVEL);
    pu32(&mut buf, grid_size);
    buf.extend_from_slice(&(min_cell_x as i16).to_le_bytes());
    buf.extend_from_slice(&(min_cell_y as i16).to_le_bytes());
    buf.extend_from_slice(&(max_cell_x as i16).to_le_bytes());
    buf.extend_from_slice(&(max_cell_y as i16).to_le_bytes());
    pu32(&mut buf, MIN_LEVEL);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a solid-red test image encoded as our DDS/BC1 writer produces,
    /// for external structural verification (DDS header parses, dimensions/
    /// mip count correct, decompressed pixels are approximately red).
    #[test]
    fn write_sample_dds_for_inspection() {
        let mut rgba = vec![0u8; 64 * 64 * 4];
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[200, 40, 30, 255]);
        }
        let dds = encode_dds_bc1(&rgba, 64, 64);
        let out_path = std::env::var("ARNIS_TEST_DDS_OUT")
            .unwrap_or_else(|_| "test_sample.dds".to_string());
        std::fs::write(&out_path, &dds).expect("failed to write test DDS");
        println!("wrote {} bytes to {}", dds.len(), out_path);
    }

    /// Writes a small synthetic tile (3x3 vertex / 2x2 quad grid, 8
    /// triangles) to the scratchpad for external structural verification
    /// (decode_nif.py) against real xLODGen-generated files.
    #[test]
    fn write_sample_nif_for_inspection() {
        let mut verts = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                verts.push((col as f32 * 512.0, row as f32 * 512.0, 100.0 + row as f32 * 10.0));
            }
        }
        let vidx = |r: u16, c: u16| -> u16 { r * 3 + c };
        let mut tris = Vec::new();
        for r in 0..2u16 {
            for c in 0..2u16 {
                tris.push((vidx(r, c), vidx(r, c + 1), vidx(r + 1, c + 1)));
                tris.push((vidx(r, c), vidx(r + 1, c + 1), vidx(r + 1, c)));
            }
        }
        let nif = build_lod_nif(
            &verts,
            &tris,
            "Data\\Textures\\Landscape\\LOD\\ArnisWorldspace\\Diffuse\\ArnisWorldspace.n.Level4.X0.Y0.dds",
            "Data\\Textures\\Landscape\\LOD\\ArnisWorldspace\\Normals\\ArnisWorldspace.n.Level4.X0.Y0.dds",
            (0.0, 0.0),
        );
        let out_path = std::env::var("ARNIS_TEST_NIF_OUT")
            .unwrap_or_else(|_| "test_sample.nif".to_string());
        std::fs::write(&out_path, &nif).expect("failed to write test NIF");
        println!("wrote {} bytes to {}", nif.len(), out_path);
    }
}
