# 6. Models, Textures and Animation

[← Levels and maps](05-levels-and-maps.md) · [Index](README.md) · [Next: Player moves →](07-player-moves.md)

The decomp has **no model importer or exporter**. `bk_asset_tool` packs models,
animations and sprites as opaque binaries. What it does have is the full
format, as C structs in `include/core2/model.h`, `include/structs.h` and
`include/core2/animationfile.h`, plus the code that reads them
(`src/core2/modelRender.c`, `src/core2/animationfile.c`, sprite code in
`src/core1/sprite.c`). This chapter maps that out so you can write a converter
or check one.

## 6.1 Model file (`BKModelBin`)

Magic number `0x0000000B` (that's how `bk_asset_tool` detects models). All
offsets are from the start of the file, and 0 means the section is absent.

```c
typedef struct bk_model_header_s {
    u32 magic_number;                // 0x0000000B
    s32 geo_list_offset;             // geometry command script (see 6.2)
    s16 texture_list_offset;         // BKTextureList
    s16 geo_type;                    // BK_GEO_TYPE_MIPMAP_TRILINEAR_BIT 0x02, BK_GEO_TYPE_ENV_MAP_BIT 0x04
    s32 gfx_list_offset;             // BKGfxList: F3DEX display lists
    s32 vtx_list_offset;             // BKVertexList: bounds + Vtx[]
    s32 unk14_list_offset;           // BKModelUnk14List (hitbox-like shapes; partially understood)
    s32 animation_list_offset;       // BKAnimationList: bone pivots
    s32 collision_list_offset;       // BKCollisionList (see 6.3)
    s32 camera_area_list_offset;     // BKCameraAreaList: boxes that toggle rendering by camera position
    s32 mesh_list_offset;            // BKMeshList: named vertex groups (scripted geometry, e.g. moving parts)
    s32 anim_vertices_list_offset;   // BKAnimVerticesList: per-animation vertex sets
    s32 animated_texture_list_offset;// BKAnimTextureList: up to 4 scrolling/flipbook textures
    u16 tri_count;
    u16 vertex_count;
    f32 unk34;
} BKModelBin;
```

Key sub-structures:

| Section | Struct | Notes |
|---|---|---|
| Textures | `BKTextureList { s32 size; s16 count; u16 pad; BKTextureInfo[] }` then texel data | `BKTextureInfo { s32 offset; s16 type; ...; u8 width, height; }`. Type bits: CI4 `0x1`, CI8 `0x2`, RGBA16 `0x4`, RGBA32 `0x8`. |
| Display lists | `BKGfxList { u32 size; u32 pad; Gfx list[]; }` | Standard **F3DEX** (`-DF3DEX_GBI`). Texture addresses use a segment the renderer sets up. |
| Vertices | `BKVertexList { s16 min[3], max[3], center[3]; s16 local_norm, count, global_norm; Vtx[]; }` | Standard N64 `Vtx` (s16 position, flags, s16 tc, rgba/normal) |
| Bones | `BKAnimationList { f32 scale; s16 count; BKAnimation { f32 translation[3]; s16 bone_id; s16 mtx_id; }[] }` | Pivot points per bone |
| Meshes | `BKMeshList { s16 count; BKMesh { s16 uid; s16 vtx_count; s16 vertices[]; } }` | Vertex groups that code can move (e.g. GV water, doors in level models) |

## 6.2 Geometry commands

The model's draw "script" (`geo_list_offset`) is a linked list of commands run
by `modelRender_draw`. Each command starts with `u32 cmd; s32 next_offset;`:

| Cmd | Name | Purpose |
|---|---|---|
| `0x00` | UNK0 | Call sublist (billboard-ish, optional pitch rotate) |
| `0x01` | SORT | Draw two sublists in order of distance between two points (transparency sorting) |
| `0x02` | BONE | Apply animation matrix `anim_matrix_id`, then call sublist |
| `0x03` | LOADDL | Draw display list `gfx_index` |
| `0x05` | SKINNING | Draw several DLs with pre-applied matrices |
| `0x06` | CALL | Call sublist |
| `0x07` | LOADDL2 | Draw display list |
| `0x08` | LOD | Call sublist if camera distance is within [min, max] |
| `0x0A` | REFPOINT | Record a point (attached to a bone) back to the object, e.g. where to spawn particles |
| `0x0C` | SELECTOR | Choose one of N sublists (used for swappable parts) |
| `0x0D` | DRAWDIST | Frustum-cull a sublist by bounding box |
| `0x0E` | UNKE | Distance-based, bone-attached |
| `0x0F` | CAMERA | Call sublist depending on whether the camera is inside/outside listed camera areas |
| `0x10` | TEXWRAP | Set texture wrap mode |
| `0x04`, `0x09`, `0x0B` | NOP | |

Rigged characters are BONE commands wrapping LOADDL/SKINNING commands.
Level models mostly use LOADDL, DRAWDIST, CAMERA and SORT.

## 6.3 Collision data

Level geometry collision and solid-object collision come from the model's
collision list:

```c
typedef struct bk_collision_list_s {
    s16 min[3], max[3];   // grid of cells, in cell units
    s16 y_stride, z_stride;
    s16 geo_count;        // number of cells
    s16 scale;            // cell size in world units (0 = single cell)
    s16 tri_count;
    u8  pad[2];
    // BKCollisionGeometry { s16 start_tri_index; s16 tri_count; } [geo_count]
    // BKCollisionTriangle  { s16 vtx_indx[3]; s16 unk6; s32 flags; } [tri_count]
} BKCollisionList;
```

Triangles reference the model's **vertex list**. A triangle spanning several
cells is stored once per cell.

Known **collision flag bits** (from the comment above `BKCollisionTriangle`):

| Bit(s) | Meaning |
|---|---|
| `0x00000080` | Knocks Banjo back when flying into it |
| `0x00001F00` | Surface type, picks footstep sounds. With `0x80000000` set it indexes a global table, otherwise bits `0x200`–`0x1000` index a per-map table (`func_803246B4`). |
| `0x00004000` | Hurts on contact (only checked in Gobi's Valley, used by the cactus) |
| `0x00010000` | Two-sided |
| `0x001E0000` | Water surface |
| `0x00400000` | Always included in the player's collision filter |
| `0x08000000 << n` | Toggled by actors (see `baMarker_8028D694`) |
| `unk6 & 0x2` | Standing here doesn't tint the player |

> When building a level, the collision triangles and their flags are as
> important as the visuals. Water, slippery slopes and damaging floors all
> come from these bits.

## 6.4 Sprites

Sprites (`BKSprite`, `include/structs.h`) are used for HUD, particles, props
(notes, eggs, feathers) and some billboards:

```c
typedef struct {
    s16 frameCnt;
    s16 type;          // SPRITE_TYPE_* bit, see include/enums.h
    s16 unk4..unkA;
    u32 unkC (bitfield);
    s32 offsets[];     // one per frame → BKSpriteFrame
} BKSprite;
// BKSpriteFrame { s16 unk0, unk2, w, h, chunkCnt, ...; palette (CI formats), BKSpriteTextureBlock[] }
// BKSpriteTextureBlock { s16 x, y, w, h; texels[] }   ← frames are split into chunks
```

| Type bit | Format |
|---|---|
| `0x001` | CI4 (16-color palette, 0x20 bytes) |
| `0x004` | CI8 (256-color palette, 0x200 bytes) |
| `0x020` | I4 |
| `0x040` | I8 |
| `0x080` | IA4 |
| `0x100` | IA8 |
| `0x400` | RGBA16 |
| `0x800` | RGBA32 |

`bk_asset_tool -e` writes each sprite's frames as PNGs plus a
`XXXX.sprite.yaml` descriptor **for viewing only**. To change a sprite you
must write the `.bin` (header, frame table, chunked texels in N64 format). The
tool's Rust decoder (`tools/bk_asset_tool/src/banjo_kazooie/asset.rs`,
`Sprite`/`Texture`) is the best reference for the layout.

Easy wins without a sprite encoder:

- **Recolor** a CI4/CI8 sprite by editing only its palette bytes (RGBA5551).
- **Swap** two sprites by swapping asset files in `assets.yaml`.

## 6.5 Animations

Skeletal animations are separate assets (`ASSET_xx_ANIM_*`) in
`BKAnimationFileBin` format (`include/core2/animationfile.h`):

```c
typedef struct {
    s16 unk0;            // first frame
    s16 unk2;            // last frame
    s16 element_count;
    u16 pad;
    BKAnimationFileElement elements[];
} BKAnimationFileBin;

typedef struct {
    u16 bone_id : 12;
    u16 unk0_3  : 4;     // likely the channel (which rotation/scale/translation component)
    s16 data_count;
    BKAnimationFileElementData data[];   // { u16 unk0_15:1, unk0_14:1, unk0_13:14; s16 unk2; }
} BKAnimationFileElement;                // likely keyframes: 2 flag bits, a 14-bit frame, a value
```

The field names are still `unk` upstream. `animationfilebin_getBoneTransformList()`
in `src/core2/animationfile.c` shows how keyframes are interpolated.

An animation targets **bone IDs**, so it only works on models whose skeleton
(the `BKAnimationList` bones) matches. `anctrl_*` (`src/core2/anctrl.c`)
controls playback: index, duration, start, subrange, loop or once,
transitions.

Model-level vertex animation (water waves, flags) uses the
`anim_vertices_list` and `animated_texture_list` sections instead.

## 6.6 Practical workflows

### Recolor or retexture an existing model

1. Locate the texture list (`texture_list_offset`) in `assets/model/XXXX.model.bin`.
2. Edit the texels/palette in place (same format and size) with a hex editor
   or a small script. CI palettes are RGBA5551 big-endian.
3. Rebuild. Size and layout don't change, so nothing else needs fixing.

### Replace a model with a custom one

You need a converter from a modern format (glTF/OBJ/FBX) to `BKModelBin`. That
means writing F3DEX display lists, a texture list, the vertex list, a
geometry script (BONE wrapping per bone, or LOADDL for static meshes) and a
collision list. Options:

- **Banjo's Backpack** (community tool) imports models and builds levels for
  BK. Use it to produce the `.bin` files, then drop them into `assets/`.
- Write your own exporter against `include/core2/model.h`. The renderer
  (`src/core2/modelRender.c`) and collision code (`src/core2/code_*.c` that use
  `BKCollisionList`) tell you exactly which fields matter.

### Use a model in the game

- As an **actor**: set `ActorInfo.modelId` (ID ≤ `0x1FFF`).
- As **level geometry**: an entry in `sMapModelDescriptions`.
- As a **static decoration**: a model prop in a setup file (ID ≤ `0x12D0`, see
  [3.5](03-assets.md#35-adding-a-new-asset)), or a tiny actor with no logic if
  the ID is higher.
- As a **skybox**: `D_8036BD40` in `src/core2/gc/sky.c`.

## 6.7 Memory notes

- Models and animations load into the heap through the asset cache. Big
  textures (RGBA32) eat heap fast. Prefer CI4/CI8.
- The N64 texture cache (TMEM) is 4 KB per load. Large textures must be split
  into tiles in the display list, as vanilla models do.
- Keep triangle counts in line with vanilla assets of the same role. The game
  targets 30/20 fps on original hardware.
