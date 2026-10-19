# 10. Game Rules and Tweaks

[← Music and sound](09-music-and-sound.md) · [Index](README.md) · [Next: Debugging, limits, references →](11-debugging-limits-references.md)

Small edits with big effects: items, costs, progression, flags, saves, and
the built-in cheat system.

## 10.1 Items and counters

All counters live in `s32 D_80385F30[0x2C]`, indexed by `enum item_e`, in
`src/core2/code_BEF20.c`:

| API | Does |
|---|---|
| `item_getCount(ITEM_x)` | Read |
| `item_inc(ITEM_x)` / `item_dec(ITEM_x)` | ±1, shows the HUD |
| `item_adjustByDiffWithHud(ITEM_x, n)` / `item_adjustByDiffWithoutHud(ITEM_x, n)` | ±n |
| `item_set(ITEM_x, n)` | Set |
| `item_setMaxCount(ITEM_x)` | Fill to the cap (refill pillows, cheats) |

Caps are enforced in `item_adjustByDiff()`:

| Item | Cap | Raised by |
|---|---|---|
| `ITEM_D_EGGS` | 100 | 200 with `FILEPROG_BE_CHEATO_BLUEEGGS` |
| `ITEM_F_RED_FEATHER` | 50 | 100 with `FILEPROG_BF_CHEATO_REDFEATHERS` |
| `ITEM_10_GOLD_FEATHER` | 10 | 20 with `FILEPROG_C0_CHEATO_GOLDFEATHERS` |
| `ITEM_15_HEALTH_TOTAL` | 8 | 16 with `FILEPROG_B9_DOUBLE_HEALTH` |
| `ITEM_16_LIFE` | 255 | |
| `ITEM_17_AIR` | 3600 (US) | `VER_SELECT(3600, 3000, …)` |

Starting inventory (`item_setItemsStartCounts()`): health 5, max health 5,
lives 3, air full, everything else 0.

Max health is recalculated from honeycombs in `func_8034789C()`:
`5 + min(3, empty_honeycombs / 6)`, or 16 with double health. That's "six
empty honeycombs = one extra health".

Per-level counters reset on `level_load()` → `itemscore_levelReset()`: notes,
jinjos, the level's jiggy count, oranges, gold bullions, acorns, presents,
caterpillars, timers.

### Recipes

```c
// Start with 10 lives and 8 health (code_BEF20.c, item_setItemsStartCounts)
D_80385F30[ITEM_14_HEALTH] = D_80385F30[ITEM_15_HEALTH_TOTAL] = 8;
D_80385F30[ITEM_16_LIFE] = 10;

// Double all ammo caps (item_adjustByDiff)
case ITEM_D_EGGS:          sp38 = (fileProgressFlag_get(FILEPROG_BE_CHEATO_BLUEEGGS))? 400 : 200; break;
```

The HUD draws a fixed number of digits, so check `src/core2/fx/` before going
past 3 digits.

## 10.2 Progression costs

| What | Where | Vanilla |
|---|---|---|
| **Note doors** | `note_door_cost_data[]` in `src/lair/lairspawnqueue.c` | `{50, 180, 260, 350, 450, 640, 765, 810, 828, 846, 864, 882}` |
| **Jigsaw puzzles** (jiggies to complete each world's picture) | `PICTURE_INFO[0xB]` in `src/lair/jigsawpicture.c` (`{cost, sizeBits, progressFlag}`) | MM 1, TTC 2, CC 5, BGS 7, FP 8, GV 9, MMM 10, RBB 12, CCW 15, Door of Grunty 25, double-health picture 4 |
| **Mumbo transformations** (tokens) | `__transformation_getCost()` in `src/core2/ch/mumbo.c` | Termite 5, Croc 10, Walrus 15, Pumpkin 20, Bee 25 |
| **Bottles' bonus / honeycomb health** | `func_8034789C()` in `src/core2/code_BEF20.c` | 6 empty honeycombs per extra health |

> `sizeBits` in `PICTURE_INFO` is how many **file-progress bits** store that
> puzzle's placed-piece count. Raising a puzzle's cost past `2^sizeBits − 1`
> needs more bits, which means re-laying out the `FILEPROG_*` range for
> puzzle pieces.

## 10.3 Flags reference

The four systems are summarized in [2.7](02-architecture.md#27-the-four-flag-systems).
Useful ones:

| Flag | Effect |
|---|---|
| `FILEPROG_B9_DOUBLE_HEALTH` | Max health 16 |
| `FILEPROG_BE/BF/C0_CHEATO_*` | Double ammo caps |
| `FILEPROG_31_MM_OPEN` and similar `*_OPEN` | World entrances opened |
| `FILEPROG_B0_HAS_ENTERED_MM` … | "Has entered world X" (used for lair enemy lists, `func_802D6948`) |
| `VOLATILE_FLAG_73..76/94/96_SANDCASTLE_INFINITE_*` | Infinite lives, eggs, red and gold feathers, health, air |
| `VOLATILE_FLAG_65_CHEAT_ENTERED` | A sandcastle cheat was used |

Grep the enums: `grep FILEPROG_ include/enums.h`.

## 10.4 The save file

| Piece | Where |
|---|---|
| Slot struct | `SaveData` in `include/save.h`: `u8 magic; u8 slotIndex; u8 data[0x70]; u8 pad[2]; u32 checksum;` |
| Slots | `gameFile_saveData[4]` (3 games + 1 rotating spare) in `src/core2/code_B5E00.c`. `gameFile_save/load/clear(gamenum)`. |
| Packing | `src/core2/code_B5040.c`: jiggies, honeycombs, Mumbo tokens, best note scores, best times, **file-progress flags**, saved items, **learned abilities**, at offsets computed into `jiggyOffset`, `honeycombOffset`, … |
| EEPROM I/O | `src/core1/eeprom.c` |
| Global data | `GlobalData` (Stop 'n' Swop items) |

There's very little spare space:

- File progress is `0x25` bytes = 296 bits, used up to `FILEPROG_123`, so
  only `0x124`–`0x127` are free.
- `learnedAbilities` has 12 free bits.
- `data[0x70]` is nearly full.

For a mod that needs a lot of new persistent state, repurpose flags of content
you removed, or widen the save layout. That breaks compatibility with vanilla
saves, so change the `magic` value too.

## 10.5 Sandcastle cheats

The Treasure Trove Cove sandcastle cheats live in `src/TTC/ma/castle.c`:

- `sSecretsCheatCodes[]`: `{code, progress, id, unused}`. `id` is a
  `VOLATILE_FLAG_*` plus `VOLATILE_FLAG_CHEAT_OFFSET` (0x14), or a small number
  for the "secret" Stop 'n' Swop codes.
- Codes are stored as strings of **floor-tile byte values**, not letters
  (`enum floor_letters_e` at the top of the file):

  | Letter | Byte | Letter | Byte | Letter | Byte | Letter | Byte |
  |---|---|---|---|---|---|---|---|
  | A | `n` (0x6E) | H | `m` (0x6D) | O | `6` (0x36) | U | `g` (0x67) |
  | B | `k` (0x6B) | I | `4` (0x34) | P | `r` (0x72) | V | `a` (0x61) |
  | C | `0` (0x30) | J | `p` (0x70) | R | `9` (0x39) | W | `7` (0x37) |
  | D | `l` (0x6C) | K | `8` (0x38) | S | `2` (0x32) | X | `h` (0x68) |
  | E | `j` (0x6A) | L | `b` (0x62) | T | `d` (0x64) | Y | `e` (0x65) |
  | F | `c` (0x63) | M | `1` (0x31) | | | Z | `3` (0x33) |
  | G | `5` (0x35) | N | `i` (0x69) | | | | |

  There's no Q on the floor. `"CHEAT"` → `"0mjnd"`. Spaces are dropped:
  `"NOTE DOOR THREE GET IN FOR FREE"` → `"i6djl669dm9jj5jd4ic69c9jj"`.
- What a cheat does is coded in `__maCastle_checkSecretCheatCodeIndex()` and
  its helpers (set a volatile flag, set file progress, give items).
- A runtime hash of the table (`__maCastle_scrambleAddressForSecretCheatCode`)
  is compared every few frames in TTC to catch memory poking. It's computed
  from the live table, so source edits are fine.

**Adding a cheat**: append an entry before the `NULL` terminator with your
encoded string and a new `id`, then handle that `id` in
`__maCastle_checkSecretCheatCodeIndex()` (e.g.
`if (secret_cheat_code->id == MY_ID) ability_setAllLearned(0xFFFFF);`).
Volatile-flag cheats are copied into file progress by the table in
`src/core2/code_B5040.c` (`D_80370A20`) so they persist.

## 10.6 Quick-win recipes

| Goal | Edit |
|---|---|
| All moves from the start | Add `ability_setAllLearned(0xFFFFF);` at the end of `clearScoreStates()` in `src/core2/code_956B0.c`. That function resets abilities, items, jiggies and flags for a fresh game. |
| Higher jump / backflip | [7.2](07-player-moves.md#72-tuning-existing-moves-quick-wins) constants |
| Faster Banjo | `bsWalkMaxFastWalkVelocity`, `gBsbTrotMaxWalkVelocity` |
| Cheaper note doors | `note_door_cost_data[]` |
| Any map's music | `sCore2_9B650MapMusicList` |
| Swap a level's skybox | `D_8036BD40` in `gc/sky.c` |
| Make an enemy tougher | Its `collision_table` row: raise `hitsToTrigger` (bits 4..2) and `damageToPlayer` (bits 6..5) |
| Enemies drop more honeycombs | `dropBundleNum` (bits 1..0) in the hit table |
| Change Bottles/Mumbo text | `assets/dialog/*.dialog` |
| Skip intro | `cutscene_skipIntroCutsceneCheck()` and friends in `src/core2/code_956B0.c` |
| Infinite eggs always | Make `item_adjustByDiff()` ignore negative `diff` for `ITEM_D_EGGS` |
| Cheaper transformations | `__transformation_getCost()` |

## 10.7 Attract-mode demos

The title-screen demos are recorded controller inputs (asset range
`0x09A2`–`0x0A0A`, `assets/demo/*.demo` YAML):

```yaml
type: DemoInput
flag: 0x00
inputs:
  - { x: 0, y: 0, buttons: 0x0000, frames: 2 }   # stick x/y, button mask, duration
```

Demo playback lives in `src/core2/demo.c`. If you change a level used by a
demo, its recording desyncs. Re-record it, or remove that demo.
