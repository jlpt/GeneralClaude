# 11. Debugging, Limits and References

[← Game rules](10-game-rules-and-tweaks.md) · [Index](README.md)

## 11.1 Debugging techniques

### On-screen text (the easiest debugger)

`print_bold_spaced()` / `print_dialog()` (`src/core2/print.c`, declared in
`include/core2/print.h`) queue a string for this frame. Call them **every
frame** from an update function. The buffer stores the **pointer** (32 slots
per frame), so the text must live in static memory:

```c
#include "core2/print.h"

static char sDebugPosText[0x40];

void debugPos_update(void) {             // call from e.g. leveloverlay_MM_update()
    f32 pos[3];
    player_getPosition(pos);
    sDebugPosText[0] = '\0';
    bk_strcat(sDebugPosText, "X ");  bk_strIToA(sDebugPosText, (s32) pos[0]);
    bk_strcat(sDebugPosText, " Y "); bk_strIToA(sDebugPosText, (s32) pos[1]);
    bk_strcat(sDebugPosText, " Z "); bk_strIToA(sDebugPosText, (s32) pos[2]);
    print_bold_spaced(0x10, 0x10, (u8 *) sDebugPosText);
}
```

This is the quickest way to find coordinates for spawning actors and placing
warps. `gcsection_getName(gsworld_getMap())` returns the current map's debug
name.

### Crash-time debug output

`src/core1/debugtext.c` (`gcdebugtext_*`) draws pixels and characters
**directly into the framebuffer**. It works even when the renderer is dead,
so use it to mark how far execution got before a hang.

### Emulator debuggers

- Build once, then open **`build/us.v10/banjo.us.v10.map`**. It has the RAM
  address of every function and variable in *your* build.
- Project64 and some Mupen64Plus front-ends have breakpoints and memory
  viewers. Break on your function's address and watch `Actor` fields.
- Remember that addresses shift as soon as you add code. Always use the map
  file from the build you're running.

### Bisecting

The matching vanilla build is a perfect baseline. When something breaks,
`git stash` your changes, confirm vanilla works, then reapply them one by one.

## 11.2 Hard limits cheat sheet

| Limit | Value | Where it comes from |
|---|---|---|
| Level overlay RAM window | `0x803863F0`–`0x803A5D00` (~126 KB, shared) | `manual_syms.us.v10.txt` (`gFramebuffers`), overlay VRAM in the splat yaml |
| Heap | `0x210520` bytes (~2.06 MB) | `HEAP_SIZE`, `src/core1/memory.c` |
| Assets cached at once | 150 | `assetcache_get()`, `src/core2/code_B3A80.c` |
| Actor scratch memory | ~0x70 bytes per actor | `Actor.local` union, `include/prop.h` |
| Actor states | 0–63 | `Actor.state:6` |
| Marker IDs with hit-table support | ≤ `0x2A8` | `D_80383F10[0x2A9]`, `src/core2/code_B62B0.c` |
| Marker ID field | 10 bits | `ActorMarker.id` |
| Actor model asset ID | ≤ `0x1FFF` | `ActorMarker.modelId:13` |
| Model prop asset ID | ≤ `0x12D0` | 12 bits + `0x2D1` |
| Sprite prop asset ID | ≤ `0x1571` | 12 bits + `0x572` |
| Hit table rows | `COLLISION_TABLE_SIZE` (187) | bump it when adding rows |
| Player states | 166 slots, 9 unused | `D_8037C700[166]`, `src/core2/bsList.c` |
| Learned abilities | 32 bits, 12 unused | `learnedAbilities`, `src/core2/abilityprogress.c` |
| File progress flags | 296 bits, ~4 unused | `0x25` bytes, `src/core2/code_98CB0.c` |
| Map IDs | `MAP_NUM_MAPS = 0x9A`, ~25 unused stubs | `enum map_e` |
| Setup ID | always `map + 0x71C` | `file_openMap()`, `src/core2/file.c` |
| World extent | cube index −16..15 → about ±16,000 units | `Cube.x/y/z:5` |
| Nodes per cube | 63 (31 trigger volumes) | `Cube.prop1Cnt:6`, `unk0_4:5` |
| Props per cube | 63 | `Cube.prop2Cnt:6` |
| Camera nodes per map | 70 (`0x46`) | `NC_CAMERA_NODE_LIST_CAPACITY` |
| Lights per map | 16 | `NUM_LIGHTING_ELEM` |
| Dialog string length | 255 bytes | `u8 len` |
| Print buffer | 32 strings per frame | `PRINT_BUFFER_COUNT` |
| Music asset ID | `0x1516 + comusic_id` | `MUSIC_TRACK_ASSET_BASE_ID` |
| Max health | 8, or 16 with double health | `item_adjustByDiff()` |

## 11.3 Finding things in the source

| You want… | Try |
|---|---|
| The code for an enemy or object | `grep -rli <name> src/*/ch src/core2/ch`, or grep its `ACTOR_`/`MARKER_` enum |
| Who spawns an actor | `grep -rn ACTOR_xx src` + the level's `*spawnqueue.c` |
| Everything special about a map | `grep -rn MAP_xx src` |
| Where a dialog is shown | `grep -rn ASSET_xxx_DIALOG src` |
| Where a flag is set | `grep -rn FILEPROG_xx src` |
| What a `func_80XXXXXX` does | Read it, then search callers. Upstream may have renamed it in a newer commit, so check `git log -S func_80XXXXXX`. |
| A table of something | The tables in [README.md](README.md#five-minute-tour) |

`tools/m2ctx.py` builds a single preprocessed context file. That's handy for
pasting into decomp.me or giving an IDE full type information.

## 11.4 Common mistakes

| Mistake | Result |
|---|---|
| Forgetting `ANTI_TAMPER=0 ANTI_PIRACY=0` | Odd failures far from your change |
| No `cargo` installed | Asset edits silently ignored |
| Registering an overlay actor in core2's list | Crash after changing worlds |
| Adding a hit-table row without bumping `COLLISION_TABLE_SIZE` | Compile error (too many initializers) |
| New `.c` file not in the yaml and not `#include`d | Compiles but the symbol is missing at link time |
| Map added to the model table but not `section.c` | NULL dereference on load |
| Non-static string passed to `print_*` | Garbage on screen |
| Writing `gVolatileFlags` directly | Checksum mismatch → save wipe |
| More than 63 props in one 1000-unit cube | Silent corruption |
| Huge RGBA32 textures | Heap exhaustion, crashes on real hardware |
| Growing core2 a lot | Level overlays overlap the framebuffers |

## 11.5 External tools and resources

| Resource | What it is |
|---|---|
| [banjo.decomp/banjo-kazooie (GitLab)](https://gitlab.com/banjo.decomp/banjo-kazooie) | The decompilation (primary repo, CI, issues) |
| [n64decomp/banjo-kazooie (GitHub mirror)](https://github.com/n64decomp/banjo-kazooie) | Mirror of the same history |
| [MittenzHugg/bk_asset_tool](https://github.com/MittenzHugg/bk_asset_tool) | Asset extractor/constructor used by the build |
| [MittenzHugg/bk_rom_compressor](https://github.com/MittenzHugg/bk_rom_compressor) | ROM (de)compressor + checksum fixer used by the build |
| [ethteck/splat](https://github.com/ethteck/splat) | The ROM splitter that generates the linker script |
| [decompals/ido-static-recomp](https://github.com/decompals/ido-static-recomp) | The IDO compiler the build downloads |
| Banjo's Backpack | Community visual level editor and model importer for BK/BT. Its official site is no longer linked; look for archived builds and its Discord. ([beginner's guide](https://www.publish0x.com/romhacks/banjos-backpack-beginners-guide-xqejrry); also listed on the r/n64homebrew modding wiki) |
| [Skelux wiki: Banjo-Tooie model data](https://wiki.skelux.root.sx/wiki/banjo_tooie/model_data) / [geometry layouts](https://wiki.skelux.root.sx/wiki/_export/raw/banjo_tooie/geometry_layouts) | Notes on Tooie's model format. Closely related to BK's, but verify against `include/core2/model.h`. |
| Banjo: Recompiled | A PC port built with N64: Recompiled from the US 1.0 ROM, with built-in mod support ([Time Extension](https://www.timeextension.com/news/2026/01/banjo-recompiled-adds-better-frame-rates-widescreen-support-and-more-to-rares-n64-classic-and-you-can-play-it-on-steam-deck), [VGC](https://www.videogameschronicle.com/news/banjo-kazooie-now-has-a-pc-port-with-4k-120fps-and-ultrawide-support)). If your audience plays there, check its mod format. Recomp mods are generally written against decomp symbols, so the knowledge in this guide carries over. |
| decomp.me | Online decompilation scratchpad (useful if you want to help finish naming functions upstream) |

### Contributing back

Every `func_80XXXXXX` you figure out while modding is worth a merge request
upstream. Better names make everyone's mods easier, and the project welcomes
documentation.
