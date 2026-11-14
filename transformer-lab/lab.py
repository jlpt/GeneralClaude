"""Transformer lab: one configurable decoder-only model, so components can be
switched on one at a time and measured on the same data and training budget.

Usage:  python lab.py '{"name": "baseline"}'  [seed]
Appends one JSON line per run to results.jsonl.
"""
import json, math, os, sys, time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    name: str = "baseline"
    # size
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    n_kv_head: int = 0          # 0 = same as n_head (no grouped-query attention)
    ctx: int = 128
    # components
    norm: str = "layer"         # layer | rms
    prenorm: bool = False       # False = original post-norm
    pos: str = "sin"            # sin | learned | rope | none
    ffn: str = "relu"           # relu | gelu | swiglu | relu2
    ffn_mult: float = 4.0       # hidden width = ffn_mult * d_model
    qk_norm: bool = False
    tie: bool = False           # tie input embedding and output projection
    conv: int = 0               # >0: short causal depthwise conv (kernel size) before attention
    moe_experts: int = 0        # >0: mixture-of-experts FFN
    moe_topk: int = 2
    mtp: int = 0                # extra heads predicting tokens t+2 .. t+1+mtp (training aux loss)
    mtp_weight: float = 0.3
    value_residual: bool = False  # mix layer-1 values into every layer's values
    loop: int = 1               # run the whole stack this many times (shared weights)
    registers: int = 0          # learned "memory" tokens prepended to every sequence
    attn_temp: bool = False     # learned per-head attention temperature
    gate_attn: bool = False     # per-head sigmoid output gate on attention
    dropout: float = 0.0
    # training
    steps: int = 1500
    batch: int = 32
    lr: float = 3e-3
    warmup: int = 100
    wd: float = 0.1
    eval_batches: int = 100


# ---------------------------------------------------------------- data
def load_data():
    text = open(os.path.join(HERE, "data", "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, cfg, gen, extra=0):
    L = cfg.ctx + extra
    ix = torch.randint(len(data) - L - 1, (cfg.batch,), generator=gen)
    x = torch.stack([data[i:i + cfg.ctx] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + L] for i in ix])  # y[:, k] = token after x[:, k]; longer if MTP
    return x, y


# ---------------------------------------------------------------- parts
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def make_norm(cfg, d=None):
    d = d or cfg.d_model
    return RMSNorm(d) if cfg.norm == "rms" else nn.LayerNorm(d)


def rope_cache(T, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_head
        self.kvh = cfg.n_kv_head or cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.kvh * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        if cfg.attn_temp:
            self.temp = nn.Parameter(torch.zeros(self.h))
        if cfg.gate_attn:
            self.gate = nn.Linear(cfg.d_model, self.h, bias=True)
        if cfg.value_residual:
            self.vmix = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, rope, v1=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.kvh, self.hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if self.cfg.value_residual:
            if v1 is None:
                v1 = v
            else:
                lam = torch.sigmoid(self.vmix)
                v = lam * v + (1 - lam) * v1
        if self.cfg.qk_norm:
            q, k = self.qn(q), self.kn(k)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        if self.kvh != self.h:
            rep = self.h // self.kvh
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scale = None
        if self.cfg.attn_temp:
            q = q * torch.exp(self.temp).view(1, -1, 1, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        if self.cfg.gate_attn:
            g = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)  # B,H,T,1
            y = y * g
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o(y), v1


class FFN(nn.Module):
    def __init__(self, cfg, hidden=None):
        super().__init__()
        self.kind = cfg.ffn
        d = cfg.d_model
        hdn = hidden or int(cfg.ffn_mult * d)
        if self.kind == "swiglu":
            hdn = int(hdn * 2 / 3)  # keep parameter count equal to a 2-matrix FFN
            self.w1 = nn.Linear(d, 2 * hdn, bias=False)
        else:
            self.w1 = nn.Linear(d, hdn, bias=False)
        self.w2 = nn.Linear(hdn, d, bias=False)

    def forward(self, x):
        h = self.w1(x)
        if self.kind == "swiglu":
            a, b = h.chunk(2, -1)
            h = F.silu(a) * b
        elif self.kind == "gelu":
            h = F.gelu(h)
        elif self.kind == "relu2":
            h = F.relu(h).square()
        else:
            h = F.relu(h)
        return self.w2(h)


class MoE(nn.Module):
    """Top-k routed experts. Each expert is 1/topk the width of the dense FFN,
    so active parameters per token match the dense model."""

    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.moe_topk
        self.n = cfg.moe_experts
        hidden = int(cfg.ffn_mult * cfg.d_model / self.k)
        self.router = nn.Linear(cfg.d_model, self.n, bias=False)
        self.experts = nn.ModuleList(FFN(cfg, hidden) for _ in range(self.n))
        self.aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.router(xf)
        probs = logits.softmax(-1)
        w, idx = probs.topk(self.k, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e in range(self.n):
            mask = idx == e
            rows = mask.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            we = (w * mask)[rows].sum(-1, keepdim=True)
            out.index_add_(0, rows, we * self.experts[e](xf[rows]))
        # Switch-style load-balancing loss
        frac = F.one_hot(idx, self.n).float().sum(1).mean(0) / self.k
        self.aux = self.n * (frac * probs.mean(0)).sum()
        return out.view(B, T, D)


class CausalConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.conv
        self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, cfg.conv, groups=cfg.d_model)

    def forward(self, x):
        h = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(h).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n1, self.n2 = make_norm(cfg), make_norm(cfg)
        self.attn = Attention(cfg)
        self.ffn = MoE(cfg) if cfg.moe_experts else FFN(cfg)
        if cfg.conv:
            self.nc, self.conv = make_norm(cfg), CausalConv(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, v1):
        c = self.cfg
        if c.conv:
            x = x + self.conv(self.nc(x)) if c.prenorm else self.nc(x + self.conv(x))
        if c.prenorm:
            a, v1 = self.attn(self.n1(x), rope, v1)
            x = x + self.drop(a)
            x = x + self.drop(self.ffn(self.n2(x)))
        else:
            a, v1 = self.attn(x, rope, v1)
            x = self.n1(x + self.drop(a))
            x = self.n2(x + self.drop(self.ffn(x)))
        return x, v1


class Model(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab, cfg.d_model)
        T = cfg.ctx + cfg.registers
        if cfg.pos == "learned":
            self.pos = nn.Parameter(torch.randn(T, cfg.d_model) * 0.02)
        elif cfg.pos == "sin":
            p = torch.arange(T).float()[:, None]
            i = torch.arange(0, cfg.d_model, 2).float()
            ang = p / (10000 ** (i / cfg.d_model))
            pe = torch.zeros(T, cfg.d_model)
            pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
            self.register_buffer("pos", pe)
        if cfg.pos == "rope":
            cos, sin = rope_cache(T, cfg.d_model // cfg.n_head)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        if cfg.registers:
            self.reg = nn.Parameter(torch.randn(cfg.registers, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.nf = make_norm(cfg) if cfg.prenorm else nn.Identity()
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        if cfg.tie:
            self.head.weight = self.emb.weight
        self.mtp_heads = nn.ModuleList(
            nn.Sequential(make_norm(cfg), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                          nn.Linear(cfg.d_model, vocab, bias=False))
            for _ in range(cfg.mtp))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        c = self.cfg
        B, T = x.shape
        h = self.emb(x) * (math.sqrt(c.d_model) if c.pos == "sin" else 1.0)
        if c.registers:
            h = torch.cat([self.reg.expand(B, -1, -1), h], 1)
        if c.pos in ("learned", "sin"):
            h = h + self.pos[:h.shape[1]]
        rope = (self.cos, self.sin) if c.pos == "rope" else None
        for _ in range(c.loop):
            v1 = None
            for b in self.blocks:
                h, v1 = b(h, rope, v1)
        h = self.nf(h)
        if c.registers:
            h = h[:, c.registers:]
        return self.head(h), h

    def aux_loss(self):
        return sum(b.ffn.aux for b in self.blocks if isinstance(b.ffn, MoE))


# ---------------------------------------------------------------- train
def run(cfg, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    train, val, vocab = load_data()
    model = Model(cfg, vocab)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg.moe_experts:
        per_expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
        n_active = n_params - cfg.n_layer * per_expert * (cfg.moe_experts - cfg.moe_topk)
    else:
        n_active = n_params
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    gen = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    model.train()
    for s in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = get_batch(train, cfg, gen, extra=cfg.mtp)
        logits, h = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y[:, :cfg.ctx].reshape(-1))
        total = loss
        for j, head in enumerate(model.mtp_heads):
            lj = head(h)
            tgt = y[:, j + 1:j + 1 + cfg.ctx]
            total = total + cfg.mtp_weight * F.cross_entropy(lj.reshape(-1, vocab), tgt.reshape(-1))
        if cfg.moe_experts:
            total = total + 0.01 * model.aux_loss()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not math.isfinite(loss.item()):
            break
    train_time = time.time() - t0

    model.eval()
    vgen = torch.Generator().manual_seed(12345)  # same val batches for every run
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(val, cfg, vgen)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).item())
    val_loss = sum(losses) / len(losses)
    res = {"name": cfg.name, "seed": seed, "val_loss": round(val_loss, 4),
           "bpc": round(val_loss / math.log(2), 4), "train_loss_last": round(loss.item(), 4),
           "params": n_params, "active_params": n_active, "time_s": round(train_time, 1),
           "cfg": asdict(cfg)}
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "cfg"}), flush=True)
    return res


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(Config(**overrides), seed)
