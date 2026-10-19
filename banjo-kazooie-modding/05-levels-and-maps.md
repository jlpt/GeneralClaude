# 5. Levels and Maps

[← Custom enemies](04-custom-enemies.md) · [Index](README.md) · [Next: Models, textures, animation →](06-models-textures-animation.md)

This chapter explains what a map is made of, every table you need to touch to
change or add one, the binary layout of setup files, warps and exits, and how
far you can push it (new maps, new worlds).

## 5.1 What happens when a map loads

`gsworld_set(map, exit, reload)` in `src/core2/gsworld.c` (simplified order):

```
musicTrack_load(core2_9B650_getMusicTrackFromMap(map))   ← map music table
mapModel_init()                  ← loads the OPA/XLU level models (mapModel.c)
dialogBin_initialize(), spawnQueue_malloc(), camera + particle + light init
sky_reset()                      ← skybox table (gc/sky.c)
cubeList_init()                  ← cube grid sized from the OPA model's bounds
gsworld_load(map)                ← parses the SETUP file: cubes/nodes/props, cameras, lights
gcdialog_init(); mapSpecificFlags_clearAll()
spawnQueue_reset()               ← builds the spawnable actor list, spawns every cube's actors
leveloverlay_initCallback_NotFP()← level overlay "init" hook
player_spawnAtMapExit()          ← puts Banjo at the requested exit
leveloverlay_initCallback_OnlyFP()
```

If the new map belongs to a **different level** than the old one, the game
first calls `level_unload()` / `level_load()` (`src/core2/code_9A740.c`). That
swaps the code overlay and resets the per-level state: notes, the level's
jiggy count, level-specific flags, timers.

To change maps from code:

```c
transitionToMap(MAP_2_MM_MUMBOS_MOUNTAIN, WARP_MM_1_MUMBOS_HUT, 0 /* transition */);
```

(`src/core2/code_5C870.c`). Exit numbers have per-map enums such as
`enum map_warp_MM` at the end of `include/enums.h`.

## 5.2 Every table keyed by map

When you **add** a map, give it an entry in each table that applies. When you
**retheme** one, these are the knobs.

| Table | File | Per-map data | If missing |
|---|---|---|---|
| `sGcSectionMapInfoList[0x9B]` | `src/core2/gc/section.c` | **Level** (`level_e`) and debug name | `map_getLevel()` dereferences NULL → **crash** |
| `sMapModelDescriptions[]` | `src/core2/mapModel.c` | OPA model, XLU model (or 0), cube-bound offsets, scale | NULL description → **crash** |
| `sCore2_9B650MapMusicList[0x84]` | `src/core2/code_9B650.c` | Music track, secondary track, flags | Falls through to the terminator entry |
| `D_8036BD40[]` (`MapSkyInfo`) | `src/core2/gc/sky.c` | Up to 3 sky layers: model, scale, rotation speed | Terminator entry (no sky) |
| `sGcSectionLevelEntrances[]` | `src/core2/gc/section.c` | Per **level**: main map and its entry exit | Used for level entry and resets |
| `cubeList_init()` margins | `src/core2/gccube.c` | Prop visibility margin (200 default, 250–700 for big maps) | Default 200 |
| `sCore2_12F30StrongTintMaps[]` | `src/core2/code_12F30.c` | Floor tint can darken the player strongly | Normal tint |
| player model `switch` | `src/core2/playermodel.c` | Which Banjo model (low-poly in cutscenes, ...) | Default model |
| Camera "type 4" vectors | `src/core2/code_35520.c` | Per-overlay camera tuning tables | |
| `func_802D6948` / `func_802D6A38` | `src/core2/code_4C020.c` | "Has entered" file-progress flags (used by lair enemy lists) | |
| Positional music channels | `src/core1/code_CE60.c` | MIDI channels faded by player position | |
| `D_80371F00[]`, `D_8036E420[]` | `src/core2/code_C4320.c`, `src/core2/code_9D640.c` | Per-map parameter tables, not fully documented | |
| Bold font texture per map | `src/core2/print.c` (`MapToBoldFontTextureMap`) | HUD font texture variant | |

Quick way to see every place a map is special-cased:

```sh
grep -rn "MAP_2_MM_MUMBOS_MOUNTAIN" src | sort
```

## 5.3 The setup file format

Asset `map_id + 0x71C`, parsed by `gsworld_load()`. The game reads it with
`file_isNextByteExpected()` (`src/core2/file.c`). That function reads one byte
and, if it isn't the expected one, **remembers** it for the next check. So
the file is a stream of tag bytes followed by payloads. All values are
**big-endian**.

```
setup := section* 0x00
section :=
    0x01 cubes          (GS_WORLD_START_INDICATOR_1_CUBES)
  | 0x02                (unused, no payload)
  | 0x03 cameras        (GS_WORLD_START_INDICATOR_3_CAMERAS)
  | 0x04 lights         (GS_WORLD_START_INDICATOR_4_LIGHTING)

cubes := 0x01 s32 min[3] s32 max[3]      ; cube index range (1 cube = 1000 units)
         cube{for x in min..max, y in min..max, z in min..max}
         0x00
cube  := ( 0x03 cube_contents
         | 0x02 s32[3]                    ; skipped
         | 0x00 s32[3] s32[3]             ; skipped
         )* 0x01                          ; 0x01 ends the cube
cube_contents :=
         [ 0x0A u8 n 0x0B NodeProp[n]       ; 20 bytes each
         | 0x06 u8 n 0x07 OtherNode[n] ]    ; 12 bytes each (rare)
         [ 0x08 u8 n 0x09 Prop[n] ]         ; 12 bytes each

cameras := ( 0x01 s16 index 0x02 u8 type camera_body )* 0x00
    type 1 pivot / 3 zoom: (0x01 f32[3] pos | 0x02 f32 h,v speed | 0x03 f32 rot,accel
                           | 0x04 f32[3] pitch,yaw,roll | 0x05 s32 flag
                           | 0x06 f32 close,far  ; zoom only)* 0x00
    type 2 static:  (0x01 f32[3] pos | 0x02 f32[3] pitch,yaw,roll)* 0x00
    type 4 random:  (0x01 s32 flag)* 0x00

lights := ( 0x01 0x02 f32[3] pos 0x03 f32 min,max fade radius 0x04 s32 r,g,b )* 0x00
```

Sources: `src/core2/gsworld.c`, `src/core2/gccube.c` (`cubeList_fromFile`,
`__code7AF80_initCubeFromFile`), `src/core2/code_A5BC0.c`
(`code7AF80_initCubeFromFile`), `src/core2/nc/cameranodelist.c`,
`src/core2/code_33250.c`/`33310.c`/`336F0.c`/`33AB0.c`, `src/core2/gclights.c`.

### `NodeProp` (20 bytes), from `include/prop.h`

```
off  size  field
0x00 s16   x
0x02 s16   y
0x04 s16   z
0x06 u16   bits 15..7 selector_or_radius (9)   ← trigger half-size, or a per-actor parameter
           bits  6..1 category (6)
           bit      0 bit0
0x08 u16   actorId      ← actor to spawn (cat 6) / index into a callback table (cat 3, 4)
0x0A u8    markerId     ← trigger nodes: which marker type can set them off (0 = player)
0x0B u8    pad
0x0C u32   bits 31..23 yaw (9, degrees) | bits 22..0 scale (23, percent; 0 = 100%)
0x10 u32   bits 31..20 unk10_31 (12) → Actor::secondaryId
           bits 19..8  unk10_19 (12) | bit 7 | bit 6 unk10_6 | bits 5..2 | bits 1..0 unk10_0
```

Categories (dispatch in `func_80334448` in `src/core2/code_AD110.c`, spawning
in `cube_spawnActors` and `func_803303B8` in `src/core2/code_A5BC0.c`):

| Category | Meaning |
|---|---|
| 0–5 | **Trigger volumes**, a box of ±`selector_or_radius` around the position. **3 = warp** → `sWarpFunctions[actorId](node, marker)`. **4 = radius trigger** → `sRadiusTriggers[actorId]`. 0, 1, 2 and 5 do nothing on contact. (The `PROP_1_CATEGORY_*` enum names in `code_A5BC0.c` for 2 and 3 don't match this dispatch code.) |
| **6** | **Actor spawn**: `actorId`, yaw, scale, `secondaryId`, parameter |
| 7 | Enemy boundary (keeps walkers in an area; see `ACTOR_FLAG_UNKNOWN_0`) |
| 8 | Path node (`unk10_31` / `unk10_19` link nodes) |
| 9 | Camera trigger volume, grouped by `actorId` (registered by `func_8030688C` in `gccube.c`; ties a region to a camera node) |
| 10 (`0xA`) | Flag node |

Exits are actor nodes too. `player_spawnAtMapExit()` finds the node whose
`actorId` is `nodeprop_getExitActorId(exit)` (`src/core2/gccube.c`: exit 1 →
actor `0x1`, 2 → `0x2`, 3 → `0x15`, 4 → `0x76` …) and puts Banjo there, facing
its yaw.

### `Prop` (12 bytes)

The last two bits of byte `0x0B` say what it is: bit 0 `isActorProp`, bit 1
`isModelProp`, neither = sprite.

```
Sprite prop:  u32 [spriteId:12 | unk:1 | rgb remove r:3 g:3 b:3 | scale:8 | mirrored:1 | pad:1]
              s16 x, y, z
              u16 [frame:5 | unk:5 | unk:1 | isNotFeatherEggOrNote:1 | unk:1 | collisionResolved:1 | model:1 | actor:1]
              asset = spriteId + 0x572   (notes, eggs, feathers, sparkles...)

Model prop:   u16 [modelId:12 | pad:4]  u8 yaw  u8 roll
              s16 x, y, z
              u8 scale   u8 [pad:2 | unk:1 | unk:1 | unk:1 | collisionResolved:1 | model:1 | actor:1]
              asset = modelId + 0x2D1
```

Music notes and blue eggs in levels are **sprite props**, not actors. That's
how a world holds 100 notes cheaply.

### Limits that come from these structures

| Limit | Why |
|---|---|
| Cube indices −16..15 per axis, so the world spans ±16,000 units | `Cube.x/y/z` are signed 5-bit fields |
| ≤ 63 nodes per cube | `Cube.prop1Cnt:6` |
| ≤ 31 trigger nodes (categories 0–5) per cube | `Cube.unk0_4:5` |
| ≤ 63 props per cube | `Cube.prop2Cnt:6` |
| Positions are `s16` | |
| Trigger half-size ≤ 511 | 9-bit field |

Spread dense content over several cubes, or the counts silently overflow.

### Inspecting and patching setups

[`examples/bk_setup_tool.py`](examples/bk_setup_tool.py) implements the parser
above:

```sh
python3 examples/bk_setup_tool.py dump assets/lvl_setup/071E.lvl_setup.bin          # MM (map 2)
python3 examples/bk_setup_tool.py dump assets/lvl_setup/071E.lvl_setup.bin --props  # include notes/eggs/models
python3 examples/bk_setup_tool.py replace-actor assets/lvl_setup/071E.lvl_setup.bin 0x6 0x3CC --limit 3
python3 examples/bk_setup_tool.py selftest
```

It also contains a small writer (`build_setup()`, `encode_node()`). You can
generate a minimal setup for a new map from Python with it.

## 5.4 Warps and doors

A warp is a **category 3 node** in the source map. Its `actorId` indexes
`sWarpFunctions[]` in `src/core2/code_AD110.c`. Most entries are small
functions in `src/core2/code_956B0.c`:

```c
void warp_mmEnterMumbosHut(s32 arg0, s32 arg1) {
    _func_8031CC8C(arg0, MAP_E_MM_MUMBOS_SKULL, WARP_MM_MUMBOS_HUT_1_ENTRANCE);
}
```

`_func_8031CC8C(node, map, exit)` packs `(map << 8) + exit` and starts the
transition. Warps can have **conditions**:

```c
void warp_gvEnterMazePyramid(s32 arg0, s32 arg1) {
    if (fileProgressFlag_getN(FILEPROG_F8_KING_SANDYBUTT_PYRAMID_STATE, 2) == 3)
        _func_8031CC8C(arg0, MAP_14_GV_SANDYBUTTS_MAZE, WARP_GV_MAZE_PYRAMID_1_ENTRANCE);
}
```

To make a warp go somewhere else, **edit the function** (easiest), or append a
new function to `sWarpFunctions[]` and point a warp node's `actorId` at its
index. There are leftover functions for the cut **Lava Level**
(`MAP_32_STUB_LL_LAVA_LEVEL`, `WARP_LL_*`) in `code_956B0.c`.

The destination needs an exit node (an actor node with the matching exit actor
ID, see 5.3) or Banjo spawns at the default position.

## 5.5 Cameras and lighting

- **Camera nodes** (setup section `0x03`) are numbered 0–`0x45`
  (`NC_CAMERA_NODE_LIST_CAPACITY`). Types: 1 pivot, 2 static, 3 zoom, 4
  random. **Camera trigger nodes** (category 9) in cubes define the regions
  where a camera node takes over. Code: `src/core2/nc/` and the camera
  trigger lists in `src/core2/gccube.c`.
- **Lights** (section `0x04`, max 16) are point lights that recolor vertices
  near them (`gclights_recolor_vertices` in `src/core2/gclights.c`). Only a few
  maps use them.

## 5.6 Level geometry models

Each map draws an **OPA** (opaque) model and optionally an **XLU**
(translucent: water surfaces, glass) model. Both are normal model assets in
the `0x146B`–`0x1515` range ([chapter 6](06-models-textures-animation.md)).

- **Collision comes from the model itself** (its collision list). There's no
  separate collision mesh asset.
- The **cube grid** is computed from the OPA model's vertex bounds
  (`mapModel_getCubeBounds()`: bounds ÷ 1000, plus `cube_min_offset`/`cube_max_offset`
  from the table). If setup content lies outside the model, widen it with
  those offsets.
- `scale` is usually `1.0f`. The intro cutscene maps use `1.6666666f`.
- The same model can serve several maps (all of CCW's Nabnut house variants
  share one).

## 5.7 Editing an existing level (the practical path)

Most level mods are "remix" mods: new layout, new enemies, new puzzles in an
existing world slot. That keeps every table, music, overlay and save-file
mapping intact.

1. **Geometry**: replace the OPA/XLU model assets (`assets/model/14AA.model.bin`
   etc.). That needs a model exporter (chapter 6) or a level editor.
2. **Contents**: edit the setup (`assets/lvl_setup/(map+0x71C).lvl_setup.bin`):
   move, add or remove actor nodes and sprite props, adjust warps and cameras.
3. **Logic**: change the level overlay (`src/MM/...`), which holds level-only
   actors and controllers.
4. **Collectible totals**: keep **exactly 100 notes** and **10 jiggies** per
   world. The pause menu, note scores and the lair note doors assume them (see
   [chapter 10](10-game-rules-and-tweaks.md)).

## 5.8 Adding a brand-new map

### Pick an ID

Best: **reuse a stub map ID**. `enum map_e` contains about 25 IDs Rare left
empty: test maps (`MAP_3_STUB_TEST_TEMPLE`, `MAP_8/9/19_STUB_TEST_CS_*`,
`MAP_F_STUB_TEST_PM`, `MAP_17_STUB_TEST_GS`, `MAP_18_STUB_TEST_RL`), cut levels
(`MAP_32_STUB_LL_LAVA_LEVEL`, `MAP_56_STUB_ML_MINE_LEVEL` and their sub-rooms
`MAP_42/4E/51/52_STUB_LL_*`, `MAP_49/54/55/57/58/59_STUB_ML_*`) and others
(`MAP_33_STUB_DEMO_INTRO_ROOM`, `MAP_5D_STUB_CCW_WINTER_ZUBBA_HIVE`,
`MAP_73_STUB_GL_ORIGINAL_FP_LOBBY`). None of them are in `section.c`, and their
setup ID `id + 0x71C` falls inside the vanilla setup range. Check that slot in
`assets.yaml`: it may hold a tiny stub file or be empty.

Otherwise you can append `0x9A`–`0x9C` (see [3.6](03-assets.md#36-level-setups-are-tied-to-map-ids)),
bump `MAP_NUM_MAPS`, and grow the arrays sized from it (`sGcSectionMapInfoList[0x9B]`,
`sCore2_9B650MapMusicList[0x84]`, …).

### Steps (using `MAP_32_STUB_LL_LAVA_LEVEL` as the example)

1. **Assets**
   - Add OPA (and optional XLU) models at new asset IDs ([3.5](03-assets.md#35-adding-a-new-asset)).
   - Write the setup to `assets/lvl_setup/074E.lvl_setup.bin` (`0x32 + 0x71C = 0x74E`)
     and make sure `assets.yaml` lists it. The minimum is one cube section with
     an exit node, a camera section, and the terminating `0x00`.
2. **`src/core2/gc/section.c`**: add before the terminator:
   ```c
   {MAP_32_STUB_LL_LAVA_LEVEL, LEVEL_1_MUMBOS_MOUNTAIN, "Lava - Main"},
   ```
   The level decides which code overlay loads and which note and jiggy counters
   it uses. Reusing an existing level is far simpler than adding one (5.9).
3. **`src/core2/mapModel.c`**: add
   `{MAP_32_STUB_LL_LAVA_LEVEL, ASSET_xxxx_MODEL_LAVA_OPA, ASSET_xxxx_MODEL_LAVA_XLU, {0,0,0}, {0,0,0}, 1.0f},`
4. **`src/core2/code_9B650.c`**: add a music row (the array is
   `[0x84]`; increase the size if you add rows).
5. **`src/core2/gc/sky.c`**: add a sky row, or leave it out for no sky.
6. **Get there**: point a warp function at it
   (`_func_8031CC8C(arg0, MAP_32_STUB_LL_LAVA_LEVEL, 1)`) and place an exit node
   for exit 1 (actor `0x1`) in its setup. For quick testing, call
   `transitionToMap()` from a debug button handler.
7. **Actors**: anything placed must be in the spawnable list for the level's
   overlay ([4.7](04-custom-enemies.md#47-registering-the-actor-spawnable-actor-lists)).

## 5.9 Adding a whole new world (level)

Possible, but it touches many systems. Prefer replacing a world. If you go
ahead, here's the scope:

| Area | What to change |
|---|---|
| `enum level_e` | New `LEVEL_E_...` (after `LEVEL_D_CUTSCENE`) |
| Level-indexed arrays | `sItemscoreNoteScores[0xB]`, `sItemscoreTimeScores[0xE]` (`code_BEF20.c`), jiggy/honeycomb/token ranges per level, pause menu totals, save-file layout. All sized for the vanilla levels. |
| Code overlay (optional) | New `src/NEW/` dir, a segment in `decompressed.us.v10.yaml` (copy an overlay's header with `follows_vram: core2`, `exclusive_ram_id: overlay`), add to `OVERLAYS` in the `Makefile`, `enum overlay_e`, `sOverlayAdressMap` in `src/core1/overlaymanager.c`, `sLevelOverlayMap` + `sLevelOverlayCallbacks` in `src/core2/code_9B990.c`, a `case` in `spawnQueue_reset()`, and the hard-coded `overlay_names` list in `tools/bk_rom_compressor/src/comp/main.rs`. Or **share an existing overlay**: map your level to `OVERLAY_6_JUNGLE` and put your code there. |
| Entry | `sGcSectionLevelEntrances[]`, a lair door / world entry warp, note door / puzzle gating |
| Save | New file-progress flags (few spare bits; see [10.4](10-game-rules-and-tweaks.md#104-the-save-file)) |

## 5.10 Tools

Building geometry and placing hundreds of objects by hand is slow. The
community tool **Banjo's Backpack** is a visual level editor for Banjo-Kazooie
(and Tooie) that builds levels and patches ROMs. The decomp doesn't integrate
it, but you can use it to *produce* model and setup binaries and drop them into
`assets/`. See [chapter 11](11-debugging-limits-references.md#115-external-tools-and-resources).
