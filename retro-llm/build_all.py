"""
Build every ROM from build/model.pt and verify them.

    python3 build_all.py            # build NES, SNES, N64 ROMs
    python3 build_all.py --verify   # ...and check NES/SNES in emulators, N64 engine on MIPS
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(args, cwd=ROOT):
    print("$", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seconds", default="60")
    a = ap.parse_args()
    py = sys.executable
    run([py, "build.py"], cwd=os.path.join(ROOT, "nes"))
    run([py, "build.py"], cwd=os.path.join(ROOT, "snes"))
    run([py, "build.py"] + (["--test"] if a.verify else []), cwd=os.path.join(ROOT, "n64"))
    if a.verify:
        run([py, "tools/check_rom.py", "nes", "--seconds", a.seconds])
        run([py, "tools/check_rom.py", "snes", "--seconds", a.seconds])


if __name__ == "__main__":
    main()
