# 3. Assets

[← Architecture](02-architecture.md) · [Index](README.md) · [Next: Custom enemies →](04-custom-enemies.md)

All game data that isn't code lives in a single **asset table**: models,
animations, sprites, level setups, dialog, quiz questions, demo inputs and
music. The game addresses everything by a 16-bit **asset ID** (`enum asset_e`
in `include/enums.h`).

## 3.1 How the game reads assets

`assetcache_get(asset_id)` in `src/core2/code_B3A80.c`:

1. Reads the meta table entry for `asset_id` and `asset_id + 1`. The size is
   the difference of their offsets.
2. If the entry's compressed flag (`compFlag & 1`) is set, it DMAs the data and
   inflates it with **rarezip** (Rare's deflate variant). Otherwise it DMAs it
   raw.
3. Keeps it in a reference-counted cache of at most **150 (`0x96`) assets**.
   Free assets with `assetcache_release()`.

The game only cares about the offset and the compressed flag. Asset **type** is
decided by the code that asks for the ID.

## 3.2 Extracting and rebuilding: `bk_asset_tool`

`tools/bk_asset_tool` (Rust, by MittenzHugg) is wired into the Makefile:

```sh
# extract (make does this for you after splitting)
tools/bk_asset_tool/target/release/bk_asset_tool -e bin/assets.bin assets/
# construct (make does this when anything under assets/ changes)
tools/bk_asset_tool/target/release/bk_asset_tool -c assets/assets.yaml build/us.v10/assets.bin
```

> **Remember:** construction only happens when `cargo` is installed. See
> [1.5](01-setup-and-building.md#15-gotcha-assets-are-only-rebuilt-when-cargo-is-installed).

After extraction you get:

```
assets/
  assets.yaml          ← the table: one line per asset
  anim/      XXXX.anim.bin
  model/     XXXX.model.bin
  sprite/    XXXX.sprite.<fmt>.bin  (+ XXXX.sprite.yaml and XXXX/NN.<fmt>.png previews)
  lvl_setup/ XXXX.lvl_setup.bin
  dialog/    XXXX.dialog            ← editable YAML
  quiz_q/    XXXX.quiz_q            ← editable YAML (Furnace Fun questions)
  grunty_q/  XXXX.grunty_q          ← editable YAML (Grunty's questions)
  demo/      XXXX.demo              ← editable YAML (attract-mode controller input)
  midi/      XXXX.midi.bin
  bin/       XXXX.bin
```

`assets.yaml` looks like:

```yaml
tbl_len: 0x15C4     # (illustrative) table length
files:
  - {uid: 0x0001, type: Animation, compressed: true , flags: 0x0000, relative_path: "anim/0001.anim.bin"}
  - {uid: 0x03C5, type: Model    , compressed: true , flags: 0x0000, relative_path: "model/03C5.model.bin"}
  ...
```

| Field | Meaning |
|---|---|
| `uid` | The asset ID |
| `type` | Used only on extraction and for choosing a parser on construct |
| `compressed` | Whether to rarezip it. Keep `true` for big assets. |
| `flags` | The table's type flags. `4` marks an empty slot. The tool uses them to guess asset types on extraction. Copy the flags of a similar asset. |
| `relative_path` | The file to pack |

### What round-trips as editable text, and what doesn't

From `tools/bk_asset_tool/src/banjo_kazooie/mod.rs` (`AssetFolder::read`):

| Type | On construct |
|---|---|
| `Dialog`, `QuizQuestion`, `GruntyQuestion`, `DemoInput` | **Parsed from YAML.** Edit the text files directly. |
| `Model`, `Animation`, `LevelSetup`, `Midi`, `Sprite_*` | **Packed as raw binary** (`Binary::read`). The parsers are commented out. Edit the `.bin` with other tools, or replace the file. |

The sprite PNGs are **preview-only**. Editing a PNG does nothing; the `.bin`
is what gets packed. To change a sprite you must re-encode the `.bin` (header
and N64 texture format).

## 3.3 Asset ID map

From `tools/generate_asset_enums.py`, the code, and `enum asset_e`:

| Range (approx.) | Type | How the game finds it |
|---|---|---|
| `0x0001`–`0x02C9` | Animations | `ASSET_xx_ANIM_*`, in `ActorAnimationInfo` tables and `anctrl_setIndex()` |
| `0x02D1`–`~0x08A0` | Models **and** sprites, interleaved | `ASSET_xxx_MODEL_*` / `ASSET_xxx_SPRITE_*`. In setup files, model props store `asset_id − 0x2D1` (`MODEL_ASSET_OFFSET`, 12 bits) and sprite props store `asset_id − 0x572` (`SPRITE_ASSET_OFFSET`, 12 bits). |
| `0x071D`–`0x07B5` | **Level setups** (inside the range above) | `file_openMap(map) = file_open(map + 0x71C)` (`src/core2/file.c`) |
| `0x07BD`–… | Skyboxes, transitions, misc | e.g. `ASSET_7BD_MODEL_SKYBOX_MM` |
| `0x09A2`–`0x0A0A` | Demo inputs | Attract-mode recordings |
| `0x0A0B`–`~0x1212` | Dialog | `ASSET_xxx_DIALOG_*`, `gcdialog_showDialog(id, …)` |
| `0x1213`–`0x13D5` | Furnace Fun quiz questions | |
| `0x1407`–`0x1424` | Grunty questions | |
| `0x146B`–`0x1515` | **Level geometry models** (`*_OPA`, `*_XLU`) | `sMapModelDescriptions` in `src/core2/mapModel.c` |
| `0x1516`–`0x15C2` | **Music** (MIDI) | `assetcache_get(0x1516 + comusic_id)` (`src/core1/code_11AC0.c`) |

The exact boundaries come from `tools/generate_asset_enums.py` and the named
entries in `enum asset_e`. Your extracted `assets/assets.yaml` is the ground
truth.

Use the `ASSET_*` enum names in code. They're self-describing, and
`grep ASSET_ include/enums.h | grep -i <name>` finds most things.

## 3.4 Replacing an asset

The easy path:

1. Find the ID (`grep -i grublin include/enums.h`, giving
   `ASSET_3C5_MODEL_GRUBLIN`).
2. Replace `assets/model/03C5.model.bin` with your file (same format).
3. `make ANTI_TAMPER=0 ANTI_PIRACY=0`.

Size doesn't matter. Offsets are recomputed and the ROM grows if needed.
Only the heap at runtime limits you.

## 3.5 Adding a new asset

1. **Pick an ID after the last one** (`tbl_len − 1` and up). Appending is
   safest because nothing in the game computes IDs relative to the end of the
   table. Don't insert into the middle; that shifts the IDs of everything after.
   One exception: music is addressed as `0x1516 + track_id`, so if you plan to
   add songs, leave the IDs right after `0x15C2` free for them (see
   [9.4](09-music-and-sound.md#94-replacing-or-adding-songs)).
2. Add a line to `assets.yaml`:

   ```yaml
     - {uid: 0x1600, type: Model, compressed: true, flags: 0x0000, relative_path: "model/1600.model.bin"}
   ```

   The tool grows the table to `max(tbl_len, max_uid + 1)` automatically and
   fills any gap with empty slots. This example uses `0x1600` to leave
   `0x15C3`–`0x15FF` free for new music.
   Copy `flags` from an existing asset of the same kind.
3. Add a name to `enum asset_e` so your code reads well:

   ```c
   ASSET_1600_MODEL_MY_ENEMY = 0x1600,
   ```

4. Reference it, e.g. as `modelId` in your enemy's `ActorInfo`.

**Field-width limits for new IDs:**

| Used as | Field | Max ID |
|---|---|---|
| Actor model (`ActorInfo.modelId` → `ActorMarker.modelId:13`) | 13 bits | `0x1FFF` |
| Animation (`Actor.anctrl_asset_id:15`) | 15 bits | `0x7FFF` |
| Model prop in a setup | 12 bits + `0x2D1` | `0x12D0` |
| Sprite prop in a setup | 12 bits + `0x572` | `0x1571` |

So a *new* model **placed as a static model prop** must have an ID ≤ `0x12D0`,
which is already used by vanilla assets. Either replace an unused vanilla model
or spawn it as an actor instead (actors can use IDs up to `0x1FFF`).

## 3.6 Level setups are tied to map IDs

The setup for map `M` is **always** asset `M + 0x71C`. The comment at the end
of `enum map_e` says maps `0x9A`–`0x9C` "were allocated somewhere at some point
(e.g. the asset enum)". Their setup IDs would be `0x7B6`–`0x7B8`, between the
last vanilla setup (`0x7B5`) and the first skybox (`0x7BD`), and no
`enum asset_e` entry names them. Check those slots in your extracted
`assets.yaml`. If they're empty (missing, or `flags: 0x0004`), they're the
natural home for [new maps](05-levels-and-maps.md#58-adding-a-brand-new-map).
Past `0x9C` you'd collide with the skybox assets, so a bigger map count means
changing `file_openMap()` too.

## 3.7 Sound data

Sound banks aren't in the asset table. They're separate ROM segments in the
yaml: `soundfont1ctl/tbl` and `soundfont2ctl/tbl` (instrument and sample
banks for libultra's `n_audio`). Replacing samples means rebuilding those
banks with N64 audio tools. See [chapter 9](09-music-and-sound.md).

## 3.8 Useful scripts in the decomp

| Script | What it does |
|---|---|
| `tools/generate_asset_enums.py` | Cross-references `assets.yaml` with `enum asset_e` to find unnamed assets |
| `tools/rareunzip.py` | Standalone rarezip decompressor |
| `tools/splat_inputs.py` | Lists files referenced by the splat yaml |
| `tools/m2ctx.py`, `decomp_me_ctx.sh` | Build a context header (for decomp.me, m2c) |
