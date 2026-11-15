"""Evolve a network's wiring: nodes ("neurons") are math functions, edges
("synapses") say which earlier node feeds which. The evolved graph is one
block; the model stacks it n_layer times with a residual: x <- x + block(x).

Search: regularized (aging) evolution. Each candidate is scored by briefly
training it on Tiny Shakespeare and measuring validation loss.

Usage:
  python evolve.py scratch  [evals=300]   # start from random graphs
  python evolve.py seeded   [evals=300]   # start from a transformer block
"""
import json, math, os, random, sys, time
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# op name -> arity
OPS = {
    "linear": 1, "norm": 1, "relu": 1, "gelu": 1, "silu": 1, "sigmoid": 1,
    "tanh": 1, "square": 1, "neg": 1, "shift": 1, "cummean": 1,
    "add": 2, "mul": 2, "max": 2,
    "mix": 3,  # causal softmax attention: softmax(rope(a) rope(b)^T) c, 4 heads
}
MAX_NODES = 20

# A genome is a list of nodes [op, [inputs...]] plus an output index.
# Index 0 is the block input x; node i (1-based) may read any index < i.
# The standard pre-norm transformer block, written in this language:
TRANSFORMER = {
    "nodes": [
        ["norm", [0]],          # 1  n = norm(x)
        ["linear", [1]],        # 2  q
        ["linear", [1]],        # 3  k
        ["linear", [1]],        # 4  v
        ["mix", [2, 3, 4]],     # 5  attention
        ["linear", [5]],        # 6  attn out projection = a
        ["add", [0, 6]],        # 7  h = x + a
        ["norm", [7]],          # 8
        ["linear", [8]],        # 9  ffn up
        ["relu", [9]],          # 10
        ["linear", [10]],       # 11 ffn down = f
        ["add", [6, 11]],       # 12 out = a + f  (so x + out = h + f)
    ],
    "out": 12,
}


# ---------------------------------------------------------------- genome ops
def reachable(g):
    need, stack = set(), [g["out"]]
    while stack:
        i = stack.pop()
        if i == 0 or i in need:
            continue
        need.add(i)
        stack.extend(g["nodes"][i - 1][1])
    return need


def canonical(g):
    """String of the live subgraph only, so dead nodes don't make duplicates."""
    live = sorted(reachable(g))
    remap = {0: 0, **{old: new + 1 for new, old in enumerate(live)}}
    return json.dumps([[g["nodes"][i - 1][0], [remap[j] for j in g["nodes"][i - 1][1]]] for i in live])


def render(g):
    live = reachable(g)
    lines = []
    for i in sorted(live):
        op, ins = g["nodes"][i - 1]
        args = ", ".join("x" if j == 0 else f"n{j}" for j in ins)
        lines.append(f"n{i} = {op}({args})")
    lines.append(f"block(x) = n{g['out']}   (model does x <- x + block(x))")
    return "\n".join(lines)


def random_inputs(op, pos, rng):
    return [rng.randrange(pos) for _ in range(OPS[op])]


def random_genome(rng):
    n = rng.randint(2, 6)
    nodes = []
    for i in range(1, n + 1):
        op = rng.choice(list(OPS))
        nodes.append([op, random_inputs(op, i, rng)])
    return {"nodes": nodes, "out": n}


def mutate(g, rng):
    g = json.loads(json.dumps(g))
    nodes = g["nodes"]
    kind = rng.choice(["op", "op", "wire", "wire", "insert", "insert", "delete", "out"])
    if kind == "op" and nodes:
        i = rng.randrange(len(nodes))
        op = rng.choice(list(OPS))
        ins = nodes[i][1]
        if OPS[op] != len(ins):
            ins = (ins + random_inputs(op, i + 1, rng))[:OPS[op]]
        nodes[i] = [op, ins]
    elif kind == "wire" and nodes:
        i = rng.randrange(len(nodes))
        k = rng.randrange(len(nodes[i][1]))
        nodes[i][1][k] = rng.randrange(i + 1)
    elif kind == "insert" and len(nodes) < MAX_NODES:
        p = rng.randint(1, len(nodes) + 1)          # new node gets index p
        for nd in nodes:
            nd[1] = [j + 1 if j >= p else j for j in nd[1]]
        if g["out"] >= p:
            g["out"] += 1
        op = rng.choice(list(OPS))
        nodes.insert(p - 1, [op, random_inputs(op, p, rng)])
        consumers = [i for i in range(p, len(nodes))]  # 0-based positions after p
        if consumers and rng.random() < 0.8:          # splice it into the live path
            c = rng.choice(consumers)
            k = rng.randrange(len(nodes[c][1]))
            nodes[c][1][k] = p
        elif rng.random() < 0.5:
            g["out"] = p
    elif kind == "delete" and len(nodes) > 1:
        p = rng.randint(1, len(nodes))
        repl = nodes[p - 1][1][0]
        del nodes[p - 1]
        for nd in nodes:
            nd[1] = [repl if j == p else (j - 1 if j > p else j) for j in nd[1]]
        g["out"] = repl if g["out"] == p else (g["out"] - 1 if g["out"] > p else g["out"])
        if g["out"] == 0:
            g["out"] = len(nodes)
    else:
        g["out"] = rng.randint(1, len(nodes))
    return g


# ---------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class GraphBlock(nn.Module):
    def __init__(self, g, d, n_head, ctx):
        super().__init__()
        self.g, self.d, self.h = g, d, n_head
        self.live = sorted(reachable(g))
        self.mods = nn.ModuleDict()
        for i in self.live:
            op = g["nodes"][i - 1][0]
            if op == "linear":
                self.mods[str(i)] = nn.Linear(d, d, bias=False)
            elif op == "norm":
                self.mods[str(i)] = RMSNorm(d)
        hd = d // n_head
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(ctx).float(), inv)
        self.register_buffer("cos", f.cos())
        self.register_buffer("sin", f.sin())

    def rope(self, x):
        T = x.shape[2]
        c, s = self.cos[:T], self.sin[:T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)

    def forward(self, x):
        vals = {0: x}
        B, T, D = x.shape
        for i in self.live:
            op, ins = self.g["nodes"][i - 1]
            a = [vals[j] for j in ins]
            if op == "linear" or op == "norm":
                y = self.mods[str(i)](a[0])
            elif op == "relu": y = F.relu(a[0])
            elif op == "gelu": y = F.gelu(a[0])
            elif op == "silu": y = F.silu(a[0])
            elif op == "sigmoid": y = torch.sigmoid(a[0])
            elif op == "tanh": y = torch.tanh(a[0])
            elif op == "square": y = a[0].square()
            elif op == "neg": y = -a[0]
            elif op == "shift": y = F.pad(a[0], (0, 0, 1, 0))[:, :-1]
            elif op == "cummean":
                y = a[0].cumsum(1) / torch.arange(1, T + 1, device=x.device).view(1, T, 1)
            elif op == "add": y = a[0] + a[1]
            elif op == "mul": y = a[0] * a[1]
            elif op == "max": y = torch.maximum(a[0], a[1])
            elif op == "mix":
                q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in a)
                y = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
                y = y.transpose(1, 2).reshape(B, T, D)
            vals[i] = y
        return vals[self.g["out"]]


class GraphModel(nn.Module):
    def __init__(self, g, vocab, d, n_layer, n_head, ctx):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(GraphBlock(g, d, n_head, ctx) for _ in range(n_layer))
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(self.nf(h))


# ---------------------------------------------------------------- fitness
PROXY = dict(d=64, n_layer=2, n_head=4, ctx=64, batch=32, steps=400, lr=3e-3, eval_batches=20)
_DATA = {}


def _load():
    if not _DATA:
        text = open(os.path.join(HERE, "data", "input.txt")).read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        _DATA.update(train=data[:n], val=data[n:], vocab=len(chars))
    return _DATA


def batch(data, P, gen):
    ix = torch.randint(len(data) - P["ctx"] - 1, (P["batch"],), generator=gen)
    x = torch.stack([data[i:i + P["ctx"]] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + P["ctx"]] for i in ix])
    return x, y


def fitness(g, P=PROXY, seed=0):
    """Train briefly, return (val_loss, params, seconds). Broken graphs get inf."""
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    torch.manual_seed(seed)
    D = _load()
    t0 = time.time()
    try:
        model = GraphModel(g, D["vocab"], P["d"], P["n_layer"], P["n_head"], P["ctx"])
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], betas=(0.9, 0.95), weight_decay=0.1)
        gen = torch.Generator().manual_seed(1000 + seed)
        for s in range(P["steps"]):
            lr = P["lr"] * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / P["steps"])))
            for gr in opt.param_groups:
                gr["lr"] = lr
            x, y = batch(D["train"], P, gen)
            loss = F.cross_entropy(model(x).reshape(-1, D["vocab"]), y.reshape(-1))
            if not torch.isfinite(loss):
                return float("inf"), n_params, time.time() - t0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vgen = torch.Generator().manual_seed(12345)
        tot = 0.0
        with torch.no_grad():
            for _ in range(P["eval_batches"]):
                x, y = batch(D["val"], P, vgen)
                tot += F.cross_entropy(model(x).reshape(-1, D["vocab"]), y.reshape(-1)).item()
        v = tot / P["eval_batches"]
        return (v if math.isfinite(v) else float("inf")), n_params, time.time() - t0
    except Exception:
        return float("inf"), 0, time.time() - t0


def _worker(args):
    g, seed = args
    return g, fitness(g, seed=seed)


# ---------------------------------------------------------------- evolution
def evolve(mode, total_evals, pop_size=24, sample=6, workers=4):
    rng = random.Random(7 if mode == "scratch" else 8)
    log_path = os.path.join(HERE, f"evolution_{mode}.jsonl")
    open(log_path, "w").close()
    pop, seen, history = [], {}, []
    t_start = time.time()

    def record(g, res, parent):
        v, n_params, secs = res
        key = canonical(g)
        seen[key] = v
        entry = {"i": len(history), "val_loss": round(v, 4) if math.isfinite(v) else None,
                 "params": n_params, "live_nodes": len(reachable(g)), "secs": round(secs, 1),
                 "parent": parent, "genome": g}
        history.append(entry)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        pop.append((v, entry["i"], g))
        if len(pop) > pop_size:
            pop.pop(0)                              # aging: the oldest dies
        best = min(h["val_loss"] or 9 for h in history)
        print(f"[{mode}] eval {entry['i']:3d}  loss {v:7.4f}  nodes {entry['live_nodes']:2d}  "
              f"best {best:.4f}  t={time.time() - t_start:5.0f}s", flush=True)

    def make_child():
        for _ in range(50):
            contenders = rng.sample(pop, min(sample, len(pop)))
            parent = min(contenders, key=lambda p: p[0])
            child = parent[2]
            for _ in range(rng.choice([1, 1, 2, 3])):
                child = mutate(child, rng)
            if canonical(child) not in seen:
                return child, parent[1]
        return random_genome(rng), None

    with mp.Pool(workers) as pool:
        # initial population
        if mode == "seeded":
            init = [TRANSFORMER] + [mutate(TRANSFORMER, rng) for _ in range(pop_size - 1)]
        else:
            init = [random_genome(rng) for _ in range(pop_size)]
        for g, res in pool.imap(_worker, [(g, 0) for g in init]):
            record(g, res, None)
        # steady-state evolution with `workers` children in flight
        pending = []
        while len(history) < total_evals:
            while len(pending) < workers and len(history) + len(pending) < total_evals:
                child, parent = make_child()
                seen[canonical(child)] = None
                pending.append((pool.apply_async(_worker, ((child, 0),)), parent))
            done = [p for p in pending if p[0].ready()]
            if not done:
                time.sleep(0.2)
                continue
            for p in done:
                pending.remove(p)
                g, res = p[0].get()
                record(g, res, p[1])

    best = min((h for h in history if h["val_loss"] is not None), key=lambda h: h["val_loss"])
    print(f"\nBEST [{mode}] loss {best['val_loss']} (eval {best['i']}, {best['live_nodes']} live nodes)")
    print(render(best["genome"]))
    return best


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scratch"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    evolve(mode, n)
