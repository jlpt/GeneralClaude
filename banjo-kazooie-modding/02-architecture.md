# 2. Engine Architecture

[← Setup](01-setup-and-building.md) · [Index](README.md) · [Next: Assets →](03-assets.md)

You can't mod confidently without a mental model of how the game is put
together. This chapter is that model.

## 2.1 Code segments and overlays

The game is split into segments. Some are always in RAM; others share one RAM
window and are swapped in per level.

| Segment | Source dir | RAM (US 1.0) | Size (ROM, uncompressed) | Contents |
|---|---|---|---|---|
| boot | `src/boot` | `0x80000450` | ~20 KB | Bootstrap, rarezip decompressor, overlay table |
| core1 | `src/core1` | `0x8023DA20` | ~242 KB | Low-level engine: memory/heap, audio (`n_audio`), controllers, EEPROM, gfx helpers, overlay manager |
| core2 | `src/core2` | `0x80286F90` | ~971 KB | The game engine: player, actors, camera, collision, UI, dialog, items, and every shared enemy/object |
| **one level overlay** | `src/<LEVEL>` | `0x803863F0` | 14–60 KB each | Level-specific actors and controllers |

Level overlays (`enum overlay_e` in `include/enums.h`, table in
`src/core1/overlaymanager.c`):

| Overlay | Dir | Codename | Used by |
|---|---|---|---|
| `OVERLAY_2_WHALE` | `src/CC` | whale | Clanker's Cavern |
| `OVERLAY_3_HAUNTED` | `src/MMM` | haunted | Mad Monster Mansion |
| `OVERLAY_4_DESERT` | `src/GV` | desert | Gobi's Valley |
| `OVERLAY_5_BEACH` | `src/TTC` | beach | Treasure Trove Cove |
| `OVERLAY_6_JUNGLE` | `src/MM` | jungle | Mumbo's Mountain |
| `OVERLAY_7_SWAMP` | `src/BGS` | swamp | Bubblegloop Swamp |
| `OVERLAY_8_SHIP` | `src/RBB` | ship | Rusty Bucket Bay |
| `OVERLAY_9_SNOW` | `src/FP` | snow | Freezeezy Peak |
| `OVERLAY_A_TREE` | `src/CCW` | tree | Click Clock Wood |
| `OVERLAY_B_TRAINING` | `src/SM` | training | Spiral Mountain |
| `OVERLAY_C_INTRO` | `src/cutscenes` | intro | Intro/ending cutscene maps |
| `OVERLAY_D_WITCH` | `src/lair` | witch | Gruntilda's Lair (+ Furnace Fun) |
| `OVERLAY_E_BATTLE` | `src/fight` | battle | Final battle |
| `OVERLAY_1_COSHOW` | `src/emptyLvl` | coshow | Empty dummy |

Only **one** level overlay is loaded at a time. `level_load()` in
`src/core2/code_9A740.c` runs when you enter a map from a different level. It
calls `map_getLevel(map)` and then `overlayManager_load(leveloverlay_getOverlayFromLevel(level))`.

**Consequences for modders:**

- Code that must work in several levels (a generic new enemy, say) belongs in
  **core2**.
- Code that's only used in one world can go in that world's overlay, which
  saves resident RAM.
- A core2 function **can't call** an overlay function unless that overlay is
  loaded. The vanilla code checks `overlayManager_getLoadedID()` or
  `level_get()` first (see `spawnQueue_reset()` in `src/core2/spawnqueue.c`).

## 2.2 RAM budget (important!)

```
0x80000400  boot / entry
0x8002D500  gHeapBase ── heap, HEAP_SIZE = 0x210520 (~2.06 MB) ── src/core1/memory.c
0x8023DA20  core1 (.text/.data/.bss)
0x80286F90  core2 (.text/.data/.bss)
0x803863F0  level overlay (follows_vram: core2)
   ...      free gap
0x803A5D00  gFramebuffers (fixed in manual_syms.us.v10.txt)
```

- The level overlay window is about **0x803A5D00 − 0x803863F0 ≈ 126 KB**, and
  the largest vanilla overlay (lair, ~60 KB plus bss) already uses about half
  of it.
- Every overlay segment in the yaml has `follows_vram: core2`. **Growing core2
  pushes every level overlay up** and shrinks that window for all of them.
- `src/core1/overlaymanager.c` computes the gap between the largest overlay
  and the framebuffers, but it only calls an empty `overlayManager_debug()`
  when the gap goes negative. **Overflowing it doesn't fail the build; it
  corrupts the framebuffers at runtime.** Check `build/us.v10/banjo.us.v10.map`
  after big additions.
- Assets (models, animations, textures, setups) are loaded into the **heap**
  through `assetcache_get()`. The asset cache holds at most 150 (`0x96`) assets
  at once. Heavy custom models cost heap, not code space.
- The game doesn't use the Expansion Pak. Using it (moving the framebuffers or
  adding a second heap) is an advanced project nobody has documented in the
  decomp yet.

## 2.3 The world: maps, levels, cubes

- A **map** (`enum map_e`, `MAP_1_SM_SPIRAL_MOUNTAIN` … `MAP_99_...`) is one
  loadable area: a room, a world hub, a cutscene stage. `MAP_NUM_MAPS` is `0x9A`.
- A **level** (`enum level_e`, `LEVEL_1_MUMBOS_MOUNTAIN` … `LEVEL_D_CUTSCENE`)
  is a world made of maps. It decides the code overlay, item scores (notes,
  jiggies), and level-scoped flags.
- Space is partitioned into **cubes**, 1000×1000×1000 world-unit cells
  (`src/core2/gccube.c`). Each cube holds:
  - **prop1 = `NodeProp`s**, the setup "nodes": actor spawn points, warps and
    triggers, camera triggers, enemy boundaries, paths and flags. See
    `include/prop.h`.
  - **prop2 = `Prop`s**, lightweight placed objects: **sprite props** (notes,
    eggs, feathers, small sprites), **model props** (static decoration models)
    and **actor props** (live actors).

Everything about a map's contents comes from its **setup file** (asset ID
`map_id + 0x71C`). See [chapter 5](05-levels-and-maps.md).

## 2.4 Actors, markers, props

These three are the core of every object in the game:

```
ActorInfo  (static, const data you write)        include/prop.h
  │ markerId, actorId, modelId, animations, update/draw funcs, distances
  ▼ actor_new() / spawn
Actor      (live instance; position, state, yaw, velocity, anim ctrl, local[] scratch)
  │ ->marker
  ▼
ActorMarker (the "physical" handle: id, collision callbacks, propPtr, model)
  │ ->propPtr
  ▼
ActorProp  (the entry in a cube's prop list: s16 position, flags)
```

- **Actor ID** (`enum actor_e`, `ACTOR_6_GRUBLIN`) is what **setup files**
  reference. It's looked up in the *spawnable actor list* to find the
  `ActorInfo`.
- **Marker ID** (`enum marker_e`, `MARKER_5_GRUBLIN`) is the runtime **type**.
  Collision rules, the enemy hit table and the player's interaction `switch`
  key off it. `ActorMarker.id` is a 10-bit field.
- **Model/asset ID** (`enum asset_e`, `ASSET_3C5_MODEL_GRUBLIN`) is the 3D model
  drawn for the actor.

Every frame the engine calls the actor's `update_func`. Collisions call the
marker's callbacks (`collisionFunc` / `collision2Func` / `dieFunc`, set with
`marker_setCollisionScripts()`). Full details in [chapter 4](04-custom-enemies.md).

### Actor scratch memory

`Actor` has a union at offset `0x7C`: `u8 local[...]` / `unk7C[0x40]` +
`unkBC[0x30]`, about **0x70 bytes** of per-instance storage. Actors define a
private struct and cast `&this->local` (see `Humanoid_Baddies_Actor` in
`include/actor.h`). Keep your struct within that size.

## 2.5 The player

The player ("ba" = Banjo) is not an `Actor`. It's a set of subsystems in
`src/core2/ba/` (physics, model, anim, marker/collision, carry, drone) driven
by a **state machine** ("bs" = Banjo state) in `src/core2/bs/`, with the state
table in `src/core2/bsmethods.c`. See [chapter 7](07-player-moves.md).

## 2.6 Source layout and naming prefixes

| Prefix / dir | Meaning | Example |
|---|---|---|
| `ch*` / `*/ch/` | **Ch**aracter = actor implementation | `src/MM/ch/grublin.c` |
| `bs*` / `core2/bs/` | **B**anjo **s**tate (player move) | `src/core2/bs/bFlip.c` |
| `ba*` / `core2/ba/` | **Ba**njo subsystems | `src/core2/ba/physics.c` |
| `ma*` / `*/ma/` | **Ma**p controllers (level logic, minigames) | `src/TTC/ma/castle.c` (sandcastle) |
| `gc*` / `core2/gc/` | **G**ame **c**ontrol: dialog, transitions, sky, pause, sections | `src/core2/gc/dialog.c` |
| `nc*` / `core2/nc/` | Camera system | `src/core2/nc/dynamicCamera.c` |
| `fx*` / `core2/fx/` | HUD counters and effects | `src/core2/fx/healthscore.c` |
| `<lvl>spawnqueue.c` | Registers the level's spawnable actors | `src/MM/mmspawnqueue.c` |
| `code_XXXXX.c` | File not yet given a name (hex = ROM offset) | `src/core2/code_B62B0.c` |
| `func_80XXXXXX` / `D_80XXXXXX` | Unnamed function / data at that RAM address | |

The upstream team often leaves comments describing what a table or function
does. Reading the file header and comments first saves a lot of time.

## 2.7 The four flag systems

| System | Storage | Lifetime | API | Enum |
|---|---|---|---|---|
| **File progress** | 0x25-byte bitfield | **Saved** to EEPROM | `fileProgressFlag_get/set/getN/setN` | `enum file_progress_e` (`FILEPROG_*`) |
| **Volatile** | bitfield + checksum | Whole session, not saved | `volatileFlag_get/set` | `enum volatile_flags_e` |
| **Level-specific** | 8 bytes + CRCs | Cleared on entering a new level (`level_load`) | `levelSpecificFlags_get/set` | `enum level_flags_e` (`LEVEL_FLAG_*`) |
| **Map-specific** | one `u32` + CRCs | Per map, cleared on map change | `mapSpecificFlags_get/set` | per-level enums such as `enum mm_specific_flags` |

Jiggies, honeycombs and Mumbo tokens have their own per-ID collected bitfields
(`enum jiggy_e`, `honeycomb_e`, `mumbotoken_e`) in their score modules.

> Most flag stores keep CRCs next to the data. Always use the accessor
> functions. Writing the arrays directly trips the anti-tamper checks.

## 2.8 Frame flow (simplified)

`src/core2/gsworld.c`:

- `gsworld_update()` updates particles, cameras, actors, props, sky, the asset
  cache and level overlay callbacks.
- `gsworld_draw()` draws the sky, the opaque level model, then (unless frozen)
  the level overlay draw callback, player, props/actors, jiggies, particles,
  translucent level model, and HUD.

Level overlays plug into the frame through `sLevelOverlayCallbacks` in
`src/core2/code_9B990.c`: `init`, `update`, `release`, `draw`, `unk14`.
That's the hook for "level-wide" logic (Clanker himself, the sandcastle, GV
water levels, the MMM organ).

## 2.9 IDs at a glance (all in `include/enums.h`)

| Enum | Range used by vanilla | Notes |
|---|---|---|
| `map_e` | `0x01`–`0x99` | `MAP_NUM_MAPS = 0x9A` |
| `level_e` | `0x1`–`0xD` | |
| `actor_e` | up to `0x3CB` | Setup files reference these |
| `marker_e` | up to `0x29E` | 10-bit field; hit table lookup array sized `0x2A9` |
| `asset_e` | up to `0x15C2` | Actor model ID field is 13 bits (`≤ 0x1FFF`) |
| `comusic_e` | `0x00`–`0xAC` | Music asset = `0x1516 + id` |
| `sfx_e` | many | Sound effects |
| `bs_e` | `0x00`–`0xA5` | Player states; `bsList` array is 166 entries |
| `item_e` | `0x00`–`0x2B` | Item counters (`D_80385F30[0x2C]`) |
| `file_progress_e` | up to `0x123` | 0x25 bytes = 296 bits of storage |
| `ability_e` | `0x00`–`0x13` | Stored in a `u32`: 12 spare bits |
