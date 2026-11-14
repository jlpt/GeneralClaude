"""Generate NES includes from the trained model and assemble retro-llm.nes."""
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


def main():
    os.makedirs(GEN, exist_ok=True)
    m = ref.export()

    # ---- model.inc
    L = [".segment \"WEIGHTS0\"", "LAYERS:"]
    for l in range(S.L):
        for name in ref.MATS:
            w = m[f"l{l}.{name}"]
            L += bytes_lines(None, [(int(v) + 128) for v in w.reshape(-1)])
    shifts = [m[f"l{l}.{n}.s"] for l in range(S.L) for n in ref.MATS]
    L += [".segment \"RODATA\""]
    L += bytes_lines("EMB", [int(v) for v in m["emb"].reshape(-1)])
    L += bytes_lines("WOUT", [(int(v) + 128) for v in m["wout"].reshape(-1)])
    L += bytes_lines("SHIFTS", shifts)
    L += [f"SHIFT_OUT = {m['wout.s']}", f"SCORE_SHIFT = {S.SS}"]
    with open(os.path.join(GEN, "model.inc"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- tables.inc  (TABLES segment, page aligned, in this order)
    q = [((n - 256) ** 2) // 4 for n in range(512)]
    T = bytes_lines("QLO", [v & 0xFF for v in q])
    T += bytes_lines("QHI", [v >> 8 for v in q])
    T += bytes_lines("EXP_ATT", S.EXP_ATT)
    T += ["EXPS:"]
    for t in S.TEMPS:
        T += bytes_lines(None, S.exp_sample_table(t))
    T += bytes_lines("ALIBI", [s * d for s in S.SLOPES for d in range(S.T)])
    T += bytes_lines("VOCAB_ASCII", [ord(c) if c != "\n" else 0x20 for c in S.VOCAB])
    with open(os.path.join(GEN, "tables.inc"), "w") as f:
        f.write("\n".join(T) + "\n")

    # ---- font.chr: tile n = ASCII n, colour 1
    chr_data = bytearray(8192)
    for code, rows in font.ascii_glyphs().items():
        for r, v in enumerate(rows):
            chr_data[code * 16 + r] = v
    with open(os.path.join(GEN, "font.chr"), "wb") as f:
        f.write(chr_data)

    out = os.path.join(ROOT, "roms", "retro-llm.nes")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.check_call(["ca65", "llm.s", "-o", "gen/llm.o", "-g"], cwd=HERE)
    subprocess.check_call(["ld65", "-C", "nes.cfg", "gen/llm.o", "-o", out,
                           "-m", "gen/llm.map", "-Ln", "gen/llm.lbl"], cwd=HERE)
    print("wrote", out, os.path.getsize(out), "bytes; shifts", shifts, "out", m["wout.s"])


if __name__ == "__main__":
    main()
