"""Generate SNES includes from the trained model and assemble retro-llm.sfc."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "train"), os.path.join(ROOT, "tools")]

import spec as S  # noqa: E402
import ref  # noqa: E402
import font  # noqa: E402

GEN = os.path.join(HERE, "gen")


def bytes_lines(label, data, per=32):
    out = [f"{label}:"] if label else []
    for i in range(0, len(data), per):
        out.append("    .byte " + ",".join(f"${b & 0xFF:02X}" for b in data[i:i + per]))
    return out


def main(model=None):
    os.makedirs(GEN, exist_ok=True)
    m = model or ref.export()
    L = ['.segment "RODATA"', "LAYERS:"]
    for l in range(S.L):
        for name in ref.MATS:
            L += bytes_lines(None, [int(v) for v in m[f"l{l}.{name}"].reshape(-1)])
    shifts = [m[f"l{l}.{n}.s"] for l in range(S.L) for n in ref.MATS]
    L += bytes_lines("EMB", [int(v) for v in m["emb"].reshape(-1)])
    L += bytes_lines("WOUT", [int(v) for v in m["wout"].reshape(-1)])
    L += bytes_lines("SHIFTS", shifts + [0])
    L += [f"SHIFT_OUT = {m['wout.s']}", f"SCORE_SHIFT = {S.SS}"]
    with open(os.path.join(GEN, "model.inc"), "w") as f:
        f.write("\n".join(L) + "\n")

    T = bytes_lines("EXP_ATT", S.EXP_ATT + [0])
    T += ["EXPS:"]
    for t in S.TEMPS:
        T += bytes_lines(None, S.exp_sample_table(t))
    T += [".byte 0"]
    T += bytes_lines("ALIBI", [s * d for s in S.SLOPES for d in range(S.T)] + [0])
    T += bytes_lines("VOCAB_ASCII", [ord(c) if c != "\n" else 0x20 for c in S.VOCAB] + [0])
    glyphs = font.ascii_glyphs()
    T += bytes_lines("FONT", [v for c in range(0x20, 0x60) for v in glyphs[c]] + [0])
    with open(os.path.join(GEN, "tables.inc"), "w") as f:
        f.write("\n".join(T) + "\n")

    out = os.path.join(ROOT, "roms", "retro-llm.sfc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.check_call(["ca65", "--cpu", "65816", "llm.s", "-o", "gen/llm.o", "-g"], cwd=HERE)
    subprocess.check_call(["ld65", "-C", "snes.cfg", "gen/llm.o", "-o", out,
                           "-m", "gen/llm.map", "-Ln", "gen/llm.lbl"], cwd=HERE)
    rom = bytearray(open(out, "rb").read())
    rom[0x7FDC:0x7FE0] = b"\xFF\xFF\x00\x00"
    csum = sum(rom) & 0xFFFF
    rom[0x7FDC:0x7FE0] = bytes([~csum & 0xFF, (~csum >> 8) & 0xFF, csum & 0xFF, csum >> 8])
    open(out, "wb").write(rom)
    print("wrote", out, len(rom), "bytes")


if __name__ == "__main__":
    main()
