"""Record an animated GIF of a ROM generating text (libretro, headless)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from retro import Retro  # noqa: E402
from check_rom import CORES, PLATFORMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=PLATFORMS)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--every", type=int, default=20, help="capture every N frames")
    ap.add_argument("--out")
    a = ap.parse_args()
    core, rom = PLATFORMS[a.platform][:2]
    emu = Retro(CORES + core, os.path.join(ROOT, "roms", rom))
    frames = []
    emu.run(60)
    for _ in range(4):
        frames.append(emu.image())
        emu.run(15)
    emu.press("start", 3)
    for _ in range(int(a.seconds * 60 / a.every)):
        emu.run(a.every)
        frames.append(emu.image())
    out = a.out or os.path.join(ROOT, "screenshots", f"{a.platform}.gif")
    frames = [f.convert("P", palette=0, colors=16) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=a.every * 1000 // 60,
                   loop=0, optimize=True)
    print("wrote", out, len(frames), "frames")


if __name__ == "__main__":
    main()
