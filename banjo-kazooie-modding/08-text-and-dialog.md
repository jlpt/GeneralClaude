# 8. Text and Dialog

[← Player moves](07-player-moves.md) · [Index](README.md) · [Next: Music and sound →](09-music-and-sound.md)

Text is the easiest thing to mod: dialog assets round-trip through
`bk_asset_tool` as YAML, so you edit text files and rebuild.

## 8.1 Where text lives

| Kind | Asset IDs | Extracted to | Editable |
|---|---|---|---|
| Dialog (conversations, signs, tutorials) | `0x0A0B`–`~0x1212` (`ASSET_xxx_DIALOG_*`) | `assets/dialog/XXXX.dialog` | YAML |
| Furnace Fun quiz questions | `0x1213`–`0x13D5` | `assets/quiz_q/XXXX.quiz_q` | YAML |
| Grunty's questions | `0x1407`–`0x1424` | `assets/grunty_q/XXXX.grunty_q` | YAML |
| Debug/map names | C strings in `src/core2/gc/section.c` | | C |

Find the dialog for a scene by name: `grep -i "DIALOG.*BOTTLES" include/enums.h`,
then open `assets/dialog/<id>.dialog`.

## 8.2 Dialog file format

Binary (`tools/bk_asset_tool/src/banjo_kazooie/asset.rs`, `Dialog`; parsed in
game by `loadDialogStrings()` in `src/core2/gc/dialog.c`):

```
01 03 00                    header
u8  bottom_count            strings for the BOTTOM box
    { u8 cmd; u8 len; u8 text[len] }   (len includes the trailing 0)
u8  top_count               strings for the TOP box
    { u8 cmd; u8 len; u8 text[len] }
```

As YAML:

```yaml
type: Dialog
bottom:
  - { cmd: 0x83, string: "HELLO THERE, BANJO!"}
  - { cmd: 0x80, string: "WHO ARE YOU?"}
  - { cmd: 0x04, string: ""}
top:
  - { cmd: 0x04, string: ""}
```

The two lists drive the two speech boxes. The engine alternates between them
as each box finishes or yields.

### The `cmd` byte

From `loadDialogStrings()` and the state machine below it:

| `cmd` | Meaning |
|---|---|
| `0x80`–`0xFF` | **Speaker portrait.** Portrait = `ZOOMBOX_SPRITE_*` index `(cmd − 0x80) + 0xC` (`include/core2/gc/zoombox.h`). |
| `0x01` | Player **choice** (yes/no) |
| `0x02` | Minimize this box and hand over to the other |
| `0x03` | Wait for a player choice |
| `0x04` | **Close** this box |
| `0x05` | Minimize, advance |
| `0x06` | Conditional minimize |
| `0x07` | **Trigger**: call the caller's callback with the first byte of the string (scripted events mid-conversation) |
| `0x08` | **Conditional text**: shown only if the caller's conditional callback agrees |
| `0x09` | **Substitute integer** (`~` in the text is replaced) |

Speaker portraits (`cmd` → portrait):

| cmd | Portrait | cmd | Portrait | cmd | Portrait |
|---|---|---|---|---|---|
| `0x80` | Banjo | `0x87` | Blubber | `0x8E` | Flibbit |
| `0x81` | Kazooie | `0x88` | Nipper | `0x8F` | Trunker |
| `0x82` | Kazooie (alt) | `0x89` | Clanker | `0x90` | Rubee |
| `0x83` | Bottles | `0x8A` | Snippet | `0x91` | Gobi |
| `0x84` | Mumbo | `0x8B` | Mr. Vile | `0x92` | Grabba |
| `0x85` | Chimpy | `0x8C` | Tiptup | `0x93` | Tee-Hee |
| `0x86` | Conga | `0x8D` | Tanktup | `0x94`… | Jinjos, items, … (see `zoombox.h`) |

The full list continues through `ZOOMBOX_SPRITE_45_MOTZHAND` in `include/core2/gc/zoombox.h`.

### Text encoding and formatting codes

- Vanilla dialog is all **uppercase ASCII**. Stick to the characters the
  vanilla text uses; the dialog font doesn't cover everything.
- `bk_asset_tool` writes bytes below `0x20` and non-ASCII bytes as `\xNN`.
  When reading, YAML turns `\xNN` into a Unicode character, and the tool only
  converts **`\xFD`** back to a single byte. Use only ASCII, `\x01`–`\x1F`
  and `\xFD`.
- **`\xFD` + letter** is a formatting escape (handled in `src/core2/print.c`;
  dialog goes through `print_dialog()`):

  | Code | Effect |
  |---|---|
  | `\xFDh` / `\xFDl` | Shaky text on / off |
  | `\xFDp` | Toggle monospace |
  | `\xFDf` | Toggle font |
  | `\xFDb` | Toggle background |
  | `\xFDj` / `\xFDe` | "Broken" font on / off |
  | `\xFDv`, `\xFDd` | Gradient modes |
  | `\xFDq` | Toggle texture filtering |

  Example: `string: "I AM \xFDhVERY SCARY\xFDl!"`
- Lines wrap automatically at spaces (`_gczoombox_findLineBreak`). Keep each
  string about as long as similar vanilla strings.
- `len` is one byte, so a single string is at most 255 bytes including its
  terminator.

## 8.3 Showing dialog from code

```c
bool gcdialog_showDialog(s32 text_id, s32 flags, f32 *pos, ActorMarker *marker,
                         void (*callback)(ActorMarker *, enum asset_e, s32),
                         void (*trigger_callback)(ActorMarker *, enum asset_e, s32));
```

It returns TRUE if the dialog started (FALSE if another is active, unless
queued). Observed `flags` bits (`gcdialog_showDialogConditional()` in
`src/core2/gc/dialog.c`):

| Bit | Effect |
|---|---|
| `0x04` | If a dialog is already open, **queue** this one instead of failing |
| `0x08` | Put Banjo into a talking state, facing `pos` if given |
| `0x20` | Interrupt the current dialog, then queue this one |
| `0x02`, `0x80` | Passed to the box setup (affect skipping/box behavior) |

Common values in vanilla: `0`, `4`, `6`, `0xE` (`2|4|8`), `0x2A`.

Typical NPC pattern (from `src/core2/ch/beehive.c`):

```c
if (!fileProgressFlag_get(FILEPROG_D_BEEHIVE_TEXT)
    && subaddie_playerIsWithinCylinder(this, 250, 300)
    && player_getTransformation() == TRANSFORM_1_BANJO
    && gcdialog_showDialog(ASSET_D96_DIALOG_BEEHIVE_MEET, 0, NULL, NULL, NULL, 0)) {
    fileProgressFlag_set(FILEPROG_D_BEEHIVE_TEXT, TRUE);
}
```

Callbacks:

- `callback(marker, text_id, choice)` runs when the conversation ends, with
  the player's choice for yes/no prompts. Use it to continue a scene ("if
  they said yes, open the door").
- `trigger_callback(marker, text_id, value)` runs at each `cmd 0x07` entry,
  for events in the middle of a conversation.

Other helpers: `gcdialog_getCurrentTextId()`, `gcdialog_hasCurrentTextId()`,
and `func_803115C4(text_id)` (closes the dialog if it's the given one; used by
`chBeehive_die`).

## 8.4 Adding new dialog

1. Copy an existing `.dialog` file to a new asset ID at the end of the table
   ([3.5](03-assets.md#35-adding-a-new-asset)) with `type: Dialog`.
2. Name it in `enum asset_e`, e.g. `ASSET_1601_DIALOG_MY_NPC_HELLO`.
3. Call `gcdialog_showDialog(ASSET_1601_DIALOG_MY_NPC_HELLO, 0xE, this->position, this->marker, NULL, NULL)`.

Version note: some call sites use `VER_SELECT(usa_id, pal_id, ...)` because
dialog IDs differ between versions. For a US-only mod, use the US ID.

## 8.5 Quiz and Grunty questions

`QuizQuestion` (Furnace Fun) and `GruntyQuestion` YAML have a `question:` list
of strings and an `options:` list of three answers, in the same
`{cmd, string}` form. Edit them in place. The quiz logic (which question is
asked, which answer is correct) is in the lair overlay (`src/lair/`) and
`src/core2/quizquestionaskedbitfield.c`.

## 8.6 Fonts

- Dialog font: `FONTS_0_DIALOG` (`enum fonts_e`). Bold HUD font textures can
  vary per map (`MapToBoldFontTextureMap` in `src/core2/print.c`).
- Font glyphs are sprites in the asset table. Changing a glyph means
  re-encoding that sprite ([6.4](06-models-textures-animation.md#64-sprites)).
