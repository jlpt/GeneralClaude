# Banjo-Kazooie: "Turning Point 2" boot scene

A small Banjo-Kazooie (USA v1.0) mod. On power-on the game skips the Rareware logo, title and file select. It boots straight into Mad Monster Mansion at night and plays an original scripted scene. The scene is a homage to the plot of Mushoku Tensei's "Turning Point 2", acted out by the Banjo-Kazooie cast:

| Role in the homage | Played by |
|---|---|
| Young mage on the road home | Banjo |
| Hot-headed companion | Kazooie |
| Feared, stoic protector | Mumbo Jumbo |
| Unbeatable stranger who hates the protector's kind | Gruntilda |
| Stranger's companion who begs for mercy | Brentilda |
| Smiling voice in your dreams | Cheato |

Beats: night camp → the stranger arrives (thunder and camera shake) → Mumbo is floored in one hit → the stranger interrogates Banjo about the voice in his dreams → Kazooie's eggs are swatted away → the "fatal" blow (big shake, Banjo cries out, health drops to 1) → Brentilda talks her sister into mending him → dream conversation with Cheato → Banjo wakes at full health with Kazooie in tears → "TO BE CONTINUED", then you have control of Banjo.

All dialogue is newly written for this mod (see `tools/scene.py`).

Previews (from a headless mupen64plus run): `preview-1.png`, `preview-2.png`, `preview-3.png`.

## Applying the patch

You need a clean **Banjo-Kazooie (USA) v1.0** `.z64` with SHA-1 `1fe1632098865f639e22c11b9a81ee8f29c75d7a`.

```bash
xdelta3 -d -s Banjo-Kazooie_USA.z64 patch/banjo-kazooie-turning-point-2.xdelta Banjo-Kazooie_TurningPoint2.z64
```

The output should have SHA-1 `4e9904b6a18cd37af591b3c55b277dbf0696a98a`. GUI patchers such as Delta Patcher also work.

## Controls

Press **A/B** to advance the dialogue, just like any other Banjo-Kazooie conversation.

## How it's built

The mod is built from the [Banjo-Kazooie decompilation](https://gitlab.com/banjo.decomp/banjo-kazooie) at the commit in `patch/decomp-base-commit.txt`:

- `src/core1/code_0.c`: the default boot map is `MAP_1B_MMM_MAD_MONSTER_MANSION` instead of the Rareware intro.
- `src/core2/gc/dialog.c`:
  - Text IDs `0x7E00`–`0x7E02` load from in-code dialogue binaries (`include/core2/tp2_text.h`) instead of the asset table.
  - Adds `tp2_update()`, a small state machine for the title card, the three dialogues, and the mid-dialogue trigger effects (sound, camera shake, health).
- `src/core2/gsworld.c`: calls `tp2_update()` once per frame.

To rebuild:

```bash
python3 tools/gen_dialog.py <decomp>/include/core2/tp2_text.h
```

```bash
cd <decomp> && git apply <this dir>/patch/decomp-changes.diff
```

```bash
make ANTI_TAMPER=0 ANTI_PIRACY=0
```

The decomp's "differs" message at the end is expected, because this is a modified ROM.

## Limitations

- Gruntilda, Brentilda, Mumbo and Cheato appear as dialogue portraits only. No new 3D actors are spawned.
- Tested only in mupen64plus (Rice video plugin). Not tested on real hardware or other emulators.
- Booting skips file select, so nothing is saved.
