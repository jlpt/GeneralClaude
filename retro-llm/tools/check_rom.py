"""
Run a ROM headless in a libretro core and verify its output against train/ref.py.

    python3 tools/check_rom.py nes  --seconds 60
    python3 tools/check_rom.py snes --seconds 60

The ROMs keep a log of every generated token (plus the RNG seed and the
temperature) in RAM; this script reads it and replays the integer reference
with the same seed. Any arithmetic difference shows up as a token mismatch.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "train"), HERE]

import ref  # noqa: E402
import spec as S  # noqa: E402
from retro import Retro  # noqa: E402

CORES = "/usr/lib/x86_64-linux-gnu/libretro/"
PLATFORMS = {
    # core, rom, memory kind, token log offset, log size, info offset
    "nes": ("nestopia_libretro.so", "retro-llm.nes", Retro.SAVE_RAM, 0x1000, 0x800, 0x1800),
    "snes": ("snes9x_libretro.so", "retro-llm.sfc", Retro.SYSTEM_RAM, 0x2000, 0x1000, 0x3000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=PLATFORMS)
    ap.add_argument("--model", default=ref.MODEL_PT)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--start-frame", type=int, default=90)
    ap.add_argument("--shots", default=os.path.join(ROOT, "screenshots"))
    a = ap.parse_args()
    core, rom, kind, log_off, log_size, info = PLATFORMS[a.platform]
    os.makedirs(a.shots, exist_ok=True)

    emu = Retro(CORES + core, os.path.join(ROOT, "roms", rom))
    emu.run(a.start_frame)
    emu.save_png(os.path.join(a.shots, f"{a.platform}_title.png"))
    emu.press("start", 3)
    emu.run(int(a.seconds * 60))
    emu.save_png(os.path.join(a.shots, f"{a.platform}_running.png"))

    mem = emu.memory(kind)
    magic = mem[info:info + 4]
    seed = mem[info + 4] | mem[info + 5] << 8
    ntok = mem[info + 6] | mem[info + 7] << 8
    temp = mem[info + 8]
    toks = list(mem[log_off:log_off + min(ntok, log_size)])
    print(f"[{a.platform}] magic={magic!r} seed=0x{seed:04X} temp={temp} tokens={ntok} "
          f"-> {ntok / a.seconds:.2f} tokens/s (emulated time)")
    if magic != b"LLM!":
        print("ROM never started generating")
        return 1
    want = ref.generate(ref.export(a.model), len(toks), seed, temp)
    bad = next((i for i, (x, y) in enumerate(zip(toks, want)) if x != y), None)
    print("console :", repr(S.decode(toks)))
    print("ref.py  :", repr(S.decode(want)))
    print(f"bit-exact: {bad is None} ({len(toks)} tokens compared"
          + ("" if bad is None else f", first mismatch at {bad}") + ")")
    return 0 if bad is None and toks else 1


if __name__ == "__main__":
    sys.exit(main())
