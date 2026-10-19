# 1. Setup and Building

[← Index](README.md) · [Next: Engine architecture →](02-architecture.md)

## 1.1 What you need

| Requirement | Notes |
|---|---|
| Linux x86_64 (Ubuntu 18.04+), WSL2, or Docker | Native Windows is rejected by the Makefile. macOS and ARM work through Docker. |
| A US v1.0 ROM | Name it `baserom.us.v10.z64`. SHA-1 `1fe1632098865f639e22c11b9a81ee8f29c75d7a`. |
| `apt` packages from `packages.txt` | MIPS binutils/gcc, cmake, python3-venv, and others |
| Rust / `cargo` | Builds `bk_asset_tool` and `bk_rom_compressor`. **Required for modding assets** (see 1.5). |
| Python venv | `requirements.txt` (splat, spimdisasm, rabbitizer, and more) |

The IDO 5.3 compiler (the one Rare used) is downloaded automatically as a
static recompilation (`ido-static-recomp`) the first time you run `make`.

## 1.2 First build (vanilla)

```sh
git clone https://gitlab.com/banjo.decomp/banjo-kazooie.git
# or: git clone https://github.com/n64decomp/banjo-kazooie.git
cd banjo-kazooie

sudo apt-get update && sudo apt-get install -y $(cat packages.txt)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt

cp /path/to/your/rom.z64 baserom.us.v10.z64
sha1sum baserom.us.v10.z64     # must match the checksum above

make                           # add -j$(nproc) on later builds for speed
```

A good build ends with a little jiggy drawn in ASCII and:

```
banjo.us.v10.z64: OK
```

Your ROM is at **`build/us.v10/banjo.us.v10.z64`**.

### Using Docker instead

```sh
docker build -t banjo-kazooie .        # ARM: add --platform linux/amd64
docker run -it --rm -v $(pwd):/banjo banjo-kazooie
# then run make inside the container
```

## 1.3 What `make` does

Understanding the pipeline tells you where your edits go:

```
baserom.us.v10.z64
   │ bk_rom_decompress
   ▼
decompressed.us.v10.z64 ──splat (decompressed.us.v10.yaml)──► asm/, bin/, banjo.ld
                                                              │
   bin/assets.bin ──bk_asset_tool -e──► assets/ (assets.yaml + files)    ← you edit these
   assets/ ──bk_asset_tool -c──► build/us.v10/assets.bin                 (needs cargo)
   src/**/*.c ──IDO cc (+asm-processor)──► build/us.v10/src/**/*.o        ← you edit these
   *.o + banjo.ld + libultra ──ld──► banjo.us.v10.elf
   elf ──objcopy──► banjo.us.v10.uncompressed.z64
   ──bk_rom_compress──► build/us.v10/banjo.us.v10.z64                    (final ROM)
```

Key files:

| File | Role |
|---|---|
| `Makefile` | Everything above. Overlay list is in `OVERLAYS :=`. |
| `decompressed.us.v10.yaml` | **The splat config.** Lists every segment (boot, core1, core2, each level overlay) and every object file in it, in link order. It generates `banjo.ld`. |
| `symbol_addrs.us.v10.txt` | Names for addresses splat needs |
| `manual_syms.us.v10.txt` | Hard-coded addresses (heap base, framebuffers, ...) passed to the linker |
| `include/enums.h` | Map, actor, marker, asset, sound, music, flag and item IDs |
| `include/functions.h` | Most shared function prototypes |

## 1.4 Building a mod: turn off anti-tamper

Rare shipped Banjo-Kazooie with deliberate anti-tamper and anti-piracy checks.
They checksum code and data, and quietly break the game when something looks
modified. Examples from the source:

- `src/core2/gccube.c`, `cubeList_spawnActors()`: if a ROM checksum read fails,
  it swaps the **World Exit Pad** actor for an empty actor.
- `src/core2/gccube.c`, `__actor_spawnWithYaw_s32()`: makes **every actor spawn
  as Bigbutt** if `volatileflag_stub2()` returns false. That function is
  stubbed to `return TRUE` in the shipped game, but it shows how Rare thought.
- `src/core2/gsworld.c`: copies a random block of RAM over the EEPROM (wiping
  saves) if the volatile-flag checksum doesn't match. Always change volatile
  flags with `volatileFlag_set()`, never by writing `gVolatileFlags` directly,
  so the checksum stays valid. While in Treasure Trove Cove it also
  busy-waits 150,000,000 iterations (a massive stutter) if the sandcastle cheat
  table's runtime hash changes. That one catches GameShark-style memory pokes.
  The hash is computed from the table at runtime, so editing cheat strings in
  source doesn't trip it.
- Per-overlay `crc.c` / `checksums.c` files and code guarded by `#if ANTI_TAMPER`
  in about 17 source files (`grep -rn "#if ANTI_TAMPER" src`).

The Makefile exposes two switches, both on by default:

```make
ANTI_TAMPER ?= 1
ANTI_PIRACY ?= 1
```

For modding, always build with:

```sh
make clean            # once, because make doesn't notice CFLAGS changes
make ANTI_TAMPER=0 ANTI_PIRACY=0
```

`bk_rom_compressor` recalculates the overlay checksums it knows about, so
small edits often work with the checks on. Don't rely on that. Not every check
is a simple CRC, and the failures show up as weird bugs much later.

> Tip: put `ANTI_TAMPER ?= 0` and `ANTI_PIRACY ?= 0` in your fork's Makefile so
> you can't forget.

After your first change, `make` prints `baserom.us.v10.z64 differs`. **That is
expected.** The verification target compares your ROM with the original. The
modded ROM is still written to `build/us.v10/banjo.us.v10.z64`.

## 1.5 Gotcha: assets are only rebuilt when `cargo` is installed

From the Makefile:

```make
else ifneq (,$(shell which cargo))
$(ASSET_BIN): $(ASSET_ROOT)/assets.yaml $(BK_ASSET_TOOL) $(ALL_ASSET_FILES)
	$(BK_ASSET_TOOL) -c $< $@
else
$(ASSET_BIN): $(BIN_ROOT)/assets.bin
	@$(CP) $< $@          # "install cargo to construct instead"
```

Without `cargo` on `PATH`, `make` **silently copies the original asset blob**
and ignores every change you made under `assets/`. If your text or model edits
"don't show up", check this first. (PAL builds always copy, so mod US v1.0.)

## 1.6 Testing

| Option | Notes |
|---|---|
| **ares** | Most accurate emulator. Best for catching hardware-only bugs. |
| **Mupen64Plus** (RMG, simple64) | Fast, good debugger integration via some front-ends |
| **Project64** | Popular on Windows. Has a debugger and memory viewer. |
| Flashcart (EverDrive 64, SummerCart64) | The real test. Save type is EEPROM. |

Emulators are more forgiving than hardware about memory overruns and timing.
Test on ares or real hardware before you release.

## 1.7 Recommended workflow

```sh
git checkout -b my-mod          # keep master pristine for reference
make ANTI_TAMPER=0 ANTI_PIRACY=0 -j$(nproc)
# edit src/ or assets/
make ANTI_TAMPER=0 ANTI_PIRACY=0 -j$(nproc)
# test build/us.v10/banjo.us.v10.z64
```

- **Distribute patches, not ROMs.** Make a BPS patch against the vanilla ROM
  with Flips or `floating ips`, or share your git branch.
- **Rebase often.** The decomp keeps renaming `func_XXXXXXXX` symbols to real
  names. Small, focused commits rebase much more easily.
- **Diff against vanilla** with `git diff master` when something breaks. The
  matching build is your known-good reference.
- Version-specific code uses `VER_SELECT(usa10, pal, usa11, jp)` and
  `#if VERSION == ...`. When you change a value wrapped in `VER_SELECT`, change
  the first argument (US 1.0).

## 1.8 Adding new source files

Every object file must appear in `decompressed.us.v10.yaml`, because that's what
generates the linker script. A `.c` file that isn't listed gets compiled but
**never linked**. There are two ways to add code.

### Option A (simplest, no yaml edits): `#include` your file

Put your code in a new file and include it from an existing file in the right
overlay:

```c
// at the bottom of src/MM/mmspawnqueue.c
#include "ch/myenemy.inc.c"
```

Use an extension other than `.c` (such as `.inc.c` or `.h`) so the Makefile
doesn't also try to compile it on its own. The Makefile compiles every `*.c`
under `src/`, but that's harmless since unlisted objects aren't linked. This
is the most robust approach.

### Option B: list it in the splat yaml

Add a subsegment to the overlay you want, next to the files that are already
there. For example, in the `- name: MM` segment:

```yaml
  subsegments:
  - [0x1064AE0, c, ch/orangepad]
  ...
  - [auto, c, ch/myenemy]        # new file: src/MM/ch/myenemy.c
```

Editing the yaml makes `make` re-run splat (the yaml is a dependency of
`SPLAT_TIMESTAMP`). splat won't overwrite C files that already exist. Check
the generated `banjo.ld` to confirm your object and its
`.data`/`.rodata`/`.bss` sections are placed where you expect, since splat's
handling of `auto` entries and section ordering depends on its version.

## 1.9 Other versions

`make VERSION=pal`, `us.v11` and `jp` exist, but some data differs between
versions (heap size, air meter, dialog asset IDs) and PAL never rebuilds
assets. Target **US v1.0** for mods unless you have a specific reason not to.
