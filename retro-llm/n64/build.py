"""
Build retro-llm.z64 with Ubuntu's mips-linux-gnu-gcc (no N64 SDK needed) and
libdragon's prebuilt open-source IPL3 (compat build, public domain;
vendored here as ipl3_compat.z64 from DragonMinded/libdragon boot/bin/).

    python3 build.py [--ipl3 /path/to/libdragon/boot/bin/ipl3_compat.z64] [--test]
"""
import argparse
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "train"), os.path.join(ROOT, "tools")]

import spec as S  # noqa: E402
import ref  # noqa: E402
import font  # noqa: E402

GEN = os.path.join(HERE, "gen")
CC = "mips-linux-gnu-gcc"
CFLAGS = ["-march=vr4300", "-mabi=32", "-EB", "-msoft-float", "-mno-abicalls", "-fno-pic",
          "-G0", "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib", "-mno-check-zero-division",
          "-Wall", "-I", HERE]


def carr(ctype, name, data, per=24):
    lines = [f"const {ctype} {name}[{len(data)}] = {{"]
    for i in range(0, len(data), per):
        lines.append("    " + ",".join(str(int(v)) for v in data[i:i + per]) + ",")
    return lines + ["};"]


def gen_sources(model):
    os.makedirs(GEN, exist_ok=True)
    m = model
    h = ["#pragma once", "#include <stdint.h>",
         f"#define LLM_D {S.D}", f"#define LLM_L {S.L}", f"#define LLM_H {S.H}",
         f"#define LLM_HD {S.HD}", f"#define LLM_F {S.F}", f"#define LLM_T {S.T}",
         f"#define LLM_V {S.V}", f"#define LLM_SS {S.SS}", f"#define LLM_SHIFT_OUT {m['wout.s']}",
         "extern const int8_t LLM_LAYERS[LLM_L][8192];", "extern const int8_t LLM_EMB[];",
         "extern const int8_t LLM_WOUT[];", "extern const uint8_t LLM_SHIFTS[];",
         "extern const uint8_t LLM_EXP_ATT[];", "extern const uint8_t LLM_EXPS[];",
         "extern const uint8_t LLM_ALIBI[];"]
    with open(os.path.join(GEN, "model.h"), "w") as f:
        f.write("\n".join(h) + "\n")
    c = ['#include "model.h"', "const int8_t LLM_LAYERS[LLM_L][8192] = {"]
    for l in range(S.L):
        flat = [int(v) for n in ref.MATS for v in m[f"l{l}.{n}"].reshape(-1)]
        c += ["{"] + [",".join(map(str, flat[i:i + 32])) + "," for i in range(0, len(flat), 32)] + ["},"]
    c += ["};"]
    c += carr("int8_t", "LLM_EMB", m["emb"].reshape(-1))
    c += carr("int8_t", "LLM_WOUT", m["wout"].reshape(-1))
    c += carr("uint8_t", "LLM_SHIFTS", [m[f"l{l}.{n}.s"] for l in range(S.L) for n in ref.MATS])
    c += carr("uint8_t", "LLM_EXP_ATT", S.EXP_ATT)
    c += carr("uint8_t", "LLM_EXPS", [v for t in S.TEMPS for v in S.exp_sample_table(t)])
    c += carr("uint8_t", "LLM_ALIBI", [s * d for s in S.SLOPES for d in range(S.T)])
    vocab = "".join(ch if ch != "\n" else " " for ch in S.VOCAB)
    c += [f'const char LLM_VOCAB[] = "{vocab}";']
    with open(os.path.join(GEN, "model.c"), "w") as f:
        f.write("\n".join(c) + "\n")
    glyphs = font.ascii_glyphs()
    fh = ["#pragma once", "#include <stdint.h>"]
    fh += ["static " + line if i == 0 else line for i, line in
           enumerate(carr("uint8_t", "FONT", [v for ch in range(0x20, 0x60) for v in glyphs[ch]]))]
    with open(os.path.join(GEN, "font.h"), "w") as f:
        f.write("\n".join(fh) + "\n")


def build_rom(ipl3_path):
    srcs = ["crt0.S", "main.c", "engine.c", "gen/model.c"]
    elf = os.path.join(GEN, "llm.elf")
    subprocess.check_call([CC] + CFLAGS + ["-T", "n64.ld", "-o", elf] + srcs, cwd=HERE)
    binf = os.path.join(GEN, "llm.bin")
    subprocess.check_call(["mips-linux-gnu-objcopy", "-O", "binary", elf, binf])
    code = open(binf, "rb").read()
    code += b"\0" * (-len(code) % 16)
    ipl3 = bytearray(open(ipl3_path, "rb").read())
    assert len(ipl3) == 4096, "expected the 4 KiB ipl3_compat.z64"
    hdr = ipl3
    struct.pack_into(">I", hdr, 0x08, 0x80000400)        # entry point
    struct.pack_into(">I", hdr, 0x10, len(code))         # compat IPL3: bytes to load
    hdr[0x20:0x34] = b"RETRO-LLM N64".ljust(20)
    hdr[0x3B:0x40] = b"NRLE\0"
    rom = bytes(hdr) + code
    rom += b"\xFF" * (-len(rom) % (1024 * 1024))           # pad to 1 MiB
    out = os.path.join(ROOT, "roms", "retro-llm.z64")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(rom)
    print("wrote", out, len(rom), "bytes (code+data", len(code), "bytes)")


def host_test(model, n=300, seed=0x1234, temp=2):
    """Compile the engine for x86 and (big-endian) MIPS/qemu; compare with ref.py."""
    want = ref.generate(model, n, seed, temp)
    srcs = ["host_test.c", "engine.c", "gen/model.c"]
    ok = True
    for name, cc, run in [("x86-64", "gcc", []),
                          ("mips-be (qemu)", CC, ["qemu-mips", "-L", "/usr/mips-linux-gnu"])]:
        exe = os.path.join(GEN, "host_test_" + name.split()[0])
        extra = ["-static"] if cc == CC else []
        subprocess.check_call([cc, "-O2", "-Wall", "-I", HERE] + extra + ["-o", exe] + srcs, cwd=HERE)
        out = subprocess.check_output(run + [exe, str(n), str(seed), str(temp)]).split()
        got = [int(v) for v in out]
        same = got == want
        ok &= same
        print(f"engine.c on {name}: {'bit-exact' if same else 'MISMATCH'} over {n} tokens")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipl3", default=os.path.join(HERE, "ipl3_compat.z64"))
    ap.add_argument("--model", default=ref.MODEL_PT)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    model = ref.export(a.model)
    gen_sources(model)
    if a.test and not host_test(model):
        sys.exit(1)
    build_rom(a.ipl3)
