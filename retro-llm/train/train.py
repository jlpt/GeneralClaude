"""
Quantization-aware training of the tiny integer transformer.

The forward pass reproduces the console arithmetic exactly (int8 weights,
int8 activations, shift-based requantisation, table-based softmax). Rounding
uses straight-through estimators so gradients still flow.

    python3 train.py --steps 30000        # writes ../build/model.pt
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fn

import spec as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# straight-through helpers
def ste(x, xq):
    return x + (xq - x).detach()


def ste_floor(x):
    return ste(x, torch.floor(x))


def ste_round(x):
    return ste(x, torch.floor(x + 0.5))


def rshift_round(acc, s):
    """(acc + 2^(s-1)) >> s, arithmetic. acc holds exact integers."""
    if s == 0:
        return acc
    return ste_floor((acc + (1 << (s - 1))) / float(1 << s))


def clamp8(x):
    return torch.clamp(x, -127, 127)


EXP_ATT = torch.tensor(S.EXP_ATT, dtype=torch.float32)


class QLinear(nn.Module):
    """y = clamp8(rshift_round(x @ Wint, s)), Wint = clamp(round(W * 2^s))."""

    def __init__(self, n_in, n_out, gain=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out) * gain / math.sqrt(n_in))
        self.frozen_shift = None

    def shift(self):
        if self.frozen_shift is not None:
            return self.frozen_shift
        m = self.w.detach().abs().max().item()
        return int(max(0, min(14, math.floor(math.log2(127.0 / max(m, 1e-6))))))

    def wint(self):
        s = self.shift()
        return clamp8(ste_round(self.w * (1 << s))), s

    def forward(self, x, quant=True, clamp=True):
        if not quant:
            y = x @ self.w
        else:
            wi, s = self.wint()
            y = rshift_round(x @ wi, s)
        return clamp8(y) if clamp else y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = QLinear(S.D, S.D)
        self.wk = QLinear(S.D, S.D)
        self.wv = QLinear(S.D, S.D)
        self.wo = QLinear(S.D, S.D, gain=0.5)
        self.w1 = QLinear(S.D, S.F, gain=1.4)
        self.w2 = QLinear(S.F, S.D, gain=0.5)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(S.V, S.D) * 32.0)
        self.blocks = nn.ModuleList([Block() for _ in range(S.L)])
        self.wout = QLinear(S.D, S.V)
        dist = torch.arange(S.T)[:, None] - torch.arange(S.T)[None, :]       # i - j
        self.register_buffer("dist", dist.float())
        self.register_buffer("causal", dist >= 0)
        self.register_buffer("slopes", torch.tensor(S.SLOPES, dtype=torch.float32))

    def embed(self, idx, quant):
        e = clamp8(self.emb)
        if quant:
            e = ste_round(e)
        return e[idx]

    def attention(self, blk, x, quant):
        B, Tn, _ = x.shape
        q = blk.wq(x, quant).view(B, Tn, S.H, S.HD).transpose(1, 2)    # B H T hd
        k = blk.wk(x, quant).view(B, Tn, S.H, S.HD).transpose(1, 2)
        v = blk.wv(x, quant).view(B, Tn, S.H, S.HD).transpose(1, 2)
        dot = q @ k.transpose(-1, -2)                                     # B H T T (exact ints)
        sc = rshift_round(dot, S.SS) if quant else dot / (1 << S.SS)
        sc = sc - self.slopes[None, :, None, None] * self.dist[None, None, :Tn, :Tn]
        causal = self.causal[:Tn, :Tn]
        sc = sc.masked_fill(~causal, -1e9)
        m = sc.max(dim=-1, keepdim=True).values.detach()
        dlt = torch.clamp(m - sc, 0, 255)
        e = 255.0 * torch.exp(-dlt / S.ATT_TAU)
        if quant:
            e = ste(e, EXP_ATT.to(x.device)[dlt.detach().long()])
        e = e * causal
        E = e.sum(-1, keepdim=True)
        p = e * (1 << S.P_BITS) / E
        if quant:
            p = ste(p, torch.clamp(torch.floor(p.detach()), max=(1 << S.P_BITS) - 1))
        o = p @ v
        o = rshift_round(o, S.P_BITS) if quant else o / (1 << S.P_BITS)
        o = clamp8(o)
        o = o.transpose(1, 2).reshape(B, Tn, S.D)
        return blk.wo(o, quant)

    def forward(self, idx, quant=True):
        x = self.embed(idx, quant)
        for blk in self.blocks:
            x = clamp8(x + self.attention(blk, x, quant))
            h = torch.relu(blk.w1(x, quant))
            x = clamp8(x + blk.w2(h, quant))
        logits = self.wout(x, quant)
        return logits / S.LOGIT_TAU

    def qlinears(self):
        out = []
        for b in self.blocks:
            out += [b.wq, b.wk, b.wv, b.wo, b.w1, b.w2]
        return out + [self.wout]


# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_data():
    path = os.path.join(ROOT, "data", "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path) as f:
        ids = S.encode(f.read())
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * 0.95)
    return data[:n], data[n:]


def batch(data, bs, gen):
    ix = torch.randint(0, len(data) - S.T - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + S.T] for i in ix])
    y = torch.stack([data[i + 1:i + S.T + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, quant, n=40, bs=256):
    gen = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(n):
        x, y = batch(data, bs, gen)
        logits = model(x, quant)
        tot += Fn.cross_entropy(logits.reshape(-1, S.V), y.reshape(-1)).item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--float-steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "model.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count())
    train, val = load_data()
    print(f"train {len(train)} chars, val {len(val)} chars, vocab {S.V}")
    model = TinyLM()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # embeddings live in integer units, so give them a proportionally larger lr
    opt = torch.optim.AdamW([
        {"params": [model.emb], "lr": args.lr * 32},
        {"params": [p for n, p in model.named_parameters() if n != "emb"]},
    ], lr=args.lr, weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps + 1):
        quant = step >= args.float_steps
        frac = step / args.steps
        warm = min(1.0, step / 300)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        if step == int(args.steps * 0.9):
            for ql in model.qlinears():          # freeze the shifts for the last 10%
                ql.frozen_shift = ql.shift()
        x, y = batch(train, args.bs, gen)
        logits = model(x, quant)
        loss = Fn.cross_entropy(logits.reshape(-1, S.V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            vl = evaluate(model, val, quant, n=8)
            print(f"step {step:6d} quant={int(quant)} train {loss.item():.3f} val {vl:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"final val: float {evaluate(model, val, False):.3f}  int {evaluate(model, val, True):.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shifts = [ql.frozen_shift for ql in model.qlinears()]
    torch.save({"state": model.state_dict(), "shifts": shifts}, args.out)
    print("saved", args.out, "shifts", shifts)


if __name__ == "__main__":
    main()
