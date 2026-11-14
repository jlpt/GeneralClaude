"""
Bit-exact integer reference of the console inference engine.

Every ROM (NES / SNES / N64) must produce exactly the same tokens as this
file given the same seed. It uses only integer arithmetic.

    python3 ref.py --n 400 --seed 1234     # sample text
    python3 ref.py --check                 # compare against the PyTorch QAT model
"""
import argparse
import os

import numpy as np

import spec as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PT = os.path.join(ROOT, "build", "model.pt")

MATS = ["wq", "wk", "wv", "wo", "w1", "w2"]


def export(pt_path=MODEL_PT):
    """Turn the trained checkpoint into int8 tensors + shifts."""
    import torch
    ck = torch.load(pt_path, map_location="cpu")
    st, shifts = ck["state"], ck["shifts"]
    m = {"emb": np.clip(np.floor(np.clip(st["emb"].numpy(), -127, 127) + 0.5), -127, 127).astype(np.int64)}
    names = [f"l{l}.{n}" for l in range(S.L) for n in MATS] + ["wout"]
    keys = [f"blocks.{l}.{n}.w" for l in range(S.L) for n in MATS] + ["wout.w"]
    for name, key, s in zip(names, keys, shifts):
        w = st[key].numpy().astype(np.float64)
        m[name] = np.clip(np.floor(w * (1 << s) + 0.5), -127, 127).astype(np.int64)
        m[name + ".s"] = int(s)
    return m


def rr(a, s):
    """round-half-up arithmetic right shift (works on ints and int arrays)."""
    if s == 0:
        return a
    return (a + (1 << (s - 1))) >> s


def clamp8(a):
    return np.clip(a, -127, 127)


def mm(x, m, name, clamp=True):
    acc = x.astype(np.int64) @ m[name]
    y = rr(acc, m[name + ".s"])
    return clamp8(y) if clamp else y


class Engine:
    def __init__(self, model):
        self.m = model
        self.reset()

    def reset(self):
        self.t = 0
        self.K = np.zeros((S.L, S.T, S.D), dtype=np.int64)   # slot-major
        self.V = np.zeros((S.L, S.T, S.D), dtype=np.int64)

    def step(self, tok):
        m, t = self.m, self.t
        slot = t % S.T
        n = min(t + 1, S.T)
        slots = np.arange(n)
        dist = (slot - slots) % S.T
        x = m["emb"][tok].copy()
        for l in range(S.L):
            p = f"l{l}."
            q = mm(x, m, p + "wq")
            self.K[l, slot] = mm(x, m, p + "wk")
            self.V[l, slot] = mm(x, m, p + "wv")
            o = np.zeros(S.D, dtype=np.int64)
            for h in range(S.H):
                c = slice(h * S.HD, (h + 1) * S.HD)
                dot = self.K[l, :n, c] @ q[c]
                sc = rr(dot, S.SS) - S.SLOPES[h] * dist
                d = np.minimum(sc.max() - sc, 255)
                e = np.array(S.EXP_ATT, dtype=np.int64)[d]
                E = int(e.sum())
                pr = np.minimum((e << S.P_BITS) // E, (1 << S.P_BITS) - 1)
                o[c] = clamp8(rr(pr @ self.V[l, :n, c], S.P_BITS))
            x = clamp8(x + mm(o, m, p + "wo"))
            hdn = np.maximum(mm(x, m, p + "w1"), 0)
            x = clamp8(x + mm(hdn, m, p + "w2"))
        self.t += 1
        return mm(x, m, "wout")


class Rng:
    """16-bit xorshift (7, 9, 8)."""

    def __init__(self, seed):
        self.x = seed & 0xFFFF or 1

    def next(self):
        x = self.x
        x ^= (x << 7) & 0xFFFF
        x ^= x >> 9
        x ^= (x << 8) & 0xFFFF
        self.x = x
        return x


def sample(logits, rng, temp_idx=S.DEFAULT_TEMP):
    tab = S.exp_sample_table(S.TEMPS[temp_idx])
    lmax = int(logits.max())
    e = [tab[lmax - int(v)] for v in logits]
    E = sum(e)
    mask = (1 << E.bit_length()) - 1
    while True:
        r = rng.next() & mask
        if r < E:
            break
    acc = 0
    for i, ev in enumerate(e):
        acc += ev
        if r < acc:
            return i
    raise AssertionError


def generate(model, n, seed, temp_idx=S.DEFAULT_TEMP, start=S.NEWLINE):
    eng, rng = Engine(model), Rng(seed)
    tok, out = start, []
    for _ in range(n):
        tok = sample(eng.step(tok), rng, temp_idx)
        out.append(tok)
    return out


def check_against_torch(model):
    import torch
    import train as TR
    net = TR.TinyLM()
    ck = torch.load(MODEL_PT, map_location="cpu")
    net.load_state_dict(ck["state"])
    for ql, s in zip(net.qlinears(), ck["shifts"]):
        ql.frozen_shift = s
    text = open(os.path.join(ROOT, "data", "tinyshakespeare.txt")).read()[5000:5000 + S.T]
    ids = S.encode(text)[:S.T]
    with torch.no_grad():
        tl = (net(torch.tensor([ids]), True)[0] * S.LOGIT_TAU).round().long().numpy()
    eng = Engine(model)
    rl = np.stack([eng.step(i) for i in ids])
    bad = int((tl != rl).sum())
    print(f"torch-vs-integer logits mismatches: {bad} / {tl.size}")
    return bad


def val_loss(model, nchars=4000):
    """Cross-entropy (nats/char) of the integer engine with a sliding window."""
    text = open(os.path.join(ROOT, "data", "tinyshakespeare.txt")).read()
    ids = S.encode(text)
    ids = ids[int(len(ids) * 0.95):][:nchars]
    eng = Engine(model)
    tot = 0.0
    for a, b in zip(ids[:-1], ids[1:]):
        lg = eng.step(a).astype(np.float64) / S.LOGIT_TAU
        lg -= lg.max()
        tot += np.log(np.exp(lg).sum()) - lg[b]
    return tot / (len(ids) - 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0x1234)
    ap.add_argument("--temp", type=int, default=S.DEFAULT_TEMP)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    model = export()
    if a.check:
        check_against_torch(model)
        print(f"integer engine val loss (sliding window): {val_loss(model):.3f} nats/char")
    print(S.decode(generate(model, a.n, a.seed, a.temp)))
