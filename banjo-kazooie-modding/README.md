# Banjo-Kazooie Decomp Modding Guide

A practical guide to modifying **Banjo-Kazooie (N64)** with the community
decompilation. It covers custom enemies, levels and maps, assets, player moves,
text, music, game rules, and the engine limits you will run into.

The decomp is **100% decompiled and matching**. Every function in the game
exists as C source that rebuilds a byte-identical ROM. That means you can mod
the game by editing C code and rebuilding, rather than hex-editing a ROM.

| | |
|---|---|
| Upstream repo (primary) | <https://gitlab.com/banjo.decomp/banjo-kazooie> |
| GitHub mirror | <https://github.com/n64decomp/banjo-kazooie> |
| Written against commit | `e8d31fa5` (2026-09-28, "Merge branch 'documentation'") |
| Target version | US v1.0 (`baserom.us.v10.z64`) unless noted |

> **About names in this guide.** The decomp is still being documented, so
> functions get renamed over time (`func_8030E6D4` may become something
> readable later). Every function, file and table named here was checked
> against the commit above. If a name is missing in your checkout, `grep` for
> the surrounding code or the table contents.

## Chapters

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 1 | [Setup and building](01-setup-and-building.md) | Toolchain, building a ROM, modding build flags, testing, workflow |
| 2 | [Engine architecture](02-architecture.md) | Memory map, overlays, actors vs markers vs props, IDs, flags, source layout |
| 3 | [Assets](03-assets.md) | The asset table, `bk_asset_tool`, ID ranges, replacing and adding assets |
| 4 | [Custom enemies and actors](04-custom-enemies.md) | Full walkthrough: `ActorInfo`, states, spawn lists, hit tables, placement |
| 5 | [Levels and maps](05-levels-and-maps.md) | Map lifecycle, every per-map table, setup file format, warps, new maps and worlds |
| 6 | [Models, textures, animation](06-models-textures-animation.md) | Model binary format, geo commands, collision flags, sprites, tools |
| 7 | [Player moves and abilities](07-player-moves.md) | The `bs` state machine, adding moves, abilities, physics tuning |
| 8 | [Text and dialog](08-text-and-dialog.md) | Dialog YAML, speaker portraits, control codes, showing text from code |
| 9 | [Music and sound](09-music-and-sound.md) | Map music table, MIDI assets, sound effects |
| 10 | [Game rules and tweaks](10-game-rules-and-tweaks.md) | Items, caps, costs, note doors, flags and saves, cheats, quick-win recipes |
| 11 | [Debugging, limits, references](11-debugging-limits-references.md) | Debug tools, hard limits cheat sheet, where to find things, external tools |

### Examples

| File | What it is |
|---|---|
| [`examples/chmyenemy.c`](examples/chmyenemy.c) | A complete custom enemy (states, facing, taunts, dialog, hit/death callbacks) |
| [`examples/bsdoublejump.c`](examples/bsdoublejump.c) | A new player move (double jump) using an unused state slot |
| [`examples/bk_setup_tool.py`](examples/bk_setup_tool.py) | Dumps level setup files (actors, warps, props, cameras, lights) and retargets actor placements |

Both C examples compile cleanly with the decomp's IDO flags (`-fullwarn`)
against the commit above. The Python tool's parser is a port of the game's
own setup reader and passes its built-in round-trip self-test
(`python3 bk_setup_tool.py selftest`).

## Five-minute tour

If you only read one section, read this.

1. **Build vanilla first.** Get `make` to print `banjo.us.v10.z64: OK`
   ([chapter 1](01-setup-and-building.md)). Then build mods with
   `make ANTI_TAMPER=0 ANTI_PIRACY=0`.
2. **Everything is a table.** The engine is data-driven. Maps, models, music,
   skyboxes, warps, spawnable actors, enemy hit rules and player moves are all
   C arrays you can edit:

   | What | Where |
   |------|-------|
   | Map → level and debug name | `src/core2/gc/section.c` (`sGcSectionMapInfoList`) |
   | Map → level geometry models | `src/core2/mapModel.c` (`sMapModelDescriptions`) |
   | Map → music | `src/core2/code_9B650.c` (`sCore2_9B650MapMusicList`) |
   | Map → skybox | `src/core2/gc/sky.c` |
   | Level → code overlay | `src/core2/code_9B990.c` (`sLevelOverlayMap`) |
   | Which actors can spawn | `src/core2/spawnqueue.c` + `src/<LEVEL>/<level>spawnqueue.c` |
   | Enemy damage / hits-to-kill / drops | `src/core2/code_B62B0.c` (`collision_table`) |
   | Warp destinations | `src/core2/code_AD110.c` (`sWarpFunctions`) + `src/core2/code_956B0.c` |
   | Player move states | `src/core2/bsmethods.c` (`bsMethodsTable`) |
   | Item caps and starting inventory | `src/core2/code_BEF20.c` |
   | Note door costs | `src/lair/lairspawnqueue.c` (`note_door_cost_data`) |

3. **An enemy is an `ActorInfo` + an update function**, registered with
   `spawnableActorList_add()` and placed in a map's setup file by its actor ID
   ([chapter 4](04-custom-enemies.md)).
4. **A map is a setup asset + models + table entries.** The setup asset ID is
   always `map_id + 0x71C` ([chapter 5](05-levels-and-maps.md)).
5. **Assets live in one big table** that `bk_asset_tool` extracts to
   `assets/` and rebuilds automatically during `make`, as long as `cargo` is
   installed ([chapter 3](03-assets.md)).

## Legal note

You need your own legally dumped US v1.0 ROM. The decomp doesn't include game
assets, and neither does this guide. Share mods as patches (`.bps`/`.xdelta`)
or as source diffs, never as ROMs.
