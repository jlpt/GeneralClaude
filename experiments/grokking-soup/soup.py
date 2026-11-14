"""
Grokking Soup: does averaging the weights of N memorizing networks
(shared init, disjoint data shards) produce a network that generalizes?

All N models are trained in parallel as one batched tensor program
(leading dim N on every parameter), so AdamW stays exactly per-model.
"""
import argparse, json, math, time
import torch, torch.nn.functional as F

p_ = argparse.ArgumentParser()
p_.add_argument("--p", type=int, default=97)
p_.add_argument("--op", default="add")           # add | sub | mul | x2y2
p_.add_argument("--frac", type=float, default=0.4)  # union train fraction
p_.add_argument("--n", type=int, default=4)          # number of shards/workers
p_.add_argument("--steps", type=int, default=10000)
p_.add_argument("--eval_every", type=int, default=100)
p_.add_argument("--lr", type=float, default=1e-3)
p_.add_argument("--wd", type=float, default=1.0)
p_.add_argument("--d", type=int, default=128)
p_.add_argument("--heads", type=int, default=4)
p_.add_argument("--dmlp", type=int, default=512)
p_.add_argument("--seed", type=int, default=0)
p_.add_argument("--merge_every", type=int, default=0)  # 0 = never (one-shot soup only)
p_.add_argument("--out", default="run.json")
args = p_.parse_args()
torch.manual_seed(args.seed)
torch.set_num_threads(4)
P, D, H, DM = args.p, args.d, args.heads, args.dmlp
DH = D // H

# ---------------- data ----------------
a = torch.arange(P).repeat_interleave(P); b = torch.arange(P).repeat(P)
if args.op == "add": y = (a + b) % P
elif args.op == "sub": y = (a - b) % P
elif args.op == "mul": y = (a * b) % P
elif args.op == "x2y2": y = (a * a + b * b) % P
X = torch.stack([a, b, torch.full_like(a, P)], 1)
perm = torch.randperm(P * P)
n_train = int(args.frac * P * P) // args.n * args.n
tr, te = perm[:n_train], perm[n_train:]
shards = tr.view(args.n, -1)            # [N, n_train/N] disjoint
Xte, yte = X[te], y[te]

# ---------------- batched model ----------------
def init(N, gen):
    def r(*s, std): return torch.randn(N, *s, generator=gen) * std
    ps = dict(
        WE=r(P + 1, D, std=1 / math.sqrt(D)), Wpos=r(3, D, std=1 / math.sqrt(D)),
        WQ=r(H, D, DH, std=1 / math.sqrt(D)), WK=r(H, D, DH, std=1 / math.sqrt(D)),
        WV=r(H, D, DH, std=1 / math.sqrt(D)), WO=r(H, DH, D, std=1 / math.sqrt(D)),
        Win=r(D, DM, std=1 / math.sqrt(D)), bin=torch.zeros(N, DM),
        Wout=r(DM, D, std=1 / math.sqrt(DM)), bout=torch.zeros(N, D),
        WU=r(D, P, std=1 / math.sqrt(D)))
    return ps

def forward(ps, x):  # x: [N, B, 3] -> logits [N, B, P]
    # Causal 1-layer transformer; only the '=' position is read out, so only its query is needed.
    N, B, _ = x.shape
    h = torch.gather(ps["WE"], 1, x.reshape(N, -1, 1).expand(-1, -1, D)).view(N, B, 3, D)
    h = h + ps["Wpos"][:, None]
    WQ = ps["WQ"].permute(0, 2, 1, 3).reshape(N, D, D)   # [N, D, H*DH]
    WK = ps["WK"].permute(0, 2, 1, 3).reshape(N, D, D)
    WV = ps["WV"].permute(0, 2, 1, 3).reshape(N, D, D)
    hf = h.reshape(N, B * 3, D)
    k = torch.bmm(hf, WK).view(N, B, 3, H, DH)
    v = torch.bmm(hf, WV).view(N, B, 3, H, DH)
    hl = h[:, :, 2]
    q = torch.bmm(hl, WQ).view(N, B, H, DH)
    att = (torch.einsum("nbhe,nbthe->nbht", q, k) / math.sqrt(DH)).softmax(-1)
    z = torch.einsum("nbht,nbthe->nbhe", att, v).reshape(N, B, D)
    hl = hl + torch.bmm(z, ps["WO"].reshape(N, D, D))
    m = F.relu(torch.baddbmm(ps["bin"][:, None], hl, ps["Win"]))
    hl = hl + torch.baddbmm(ps["bout"][:, None], m, ps["Wout"])
    return torch.bmm(hl, ps["WU"])

def acc(logits, yy): return (logits.argmax(-1) == yy).float().mean(-1)  # per model

def soup(ps):  # average over model dim -> single model with N=1
    return {k: v.mean(0, keepdim=True) for k, v in ps.items()}

g = torch.Generator().manual_seed(1000 + args.seed)
base = init(1, g)
# Group S: N workers, SAME init, disjoint shards
S = {k: v.repeat(args.n, *[1] * (v.dim() - 1)).clone().requires_grad_() for k, v in base.items()}
# Group R: N workers, DIFFERENT inits, same shards (control: soup should fail)
gR = torch.Generator().manual_seed(2000 + args.seed)
R = {k: v.clone().requires_grad_() for k, v in init(args.n, gR).items()}
R["WE"].data[0] = base["WE"][0]  # irrelevant; keep R independent
# Group U: one model, same init, trained on the UNION of shards (centralized baseline)
U = {k: v.clone().requires_grad_() for k, v in base.items()}

def mkopt(ps): return torch.optim.AdamW(ps.values(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
oS, oR, oU = mkopt(S), mkopt(R), mkopt(U)

xs, ys = X[shards], y[shards]              # [N, Bs, 3]
xu, yu = X[tr][None], y[tr][None]          # [1, Bu, 3]
xte, yte_ = Xte[None], yte[None]

log = []
t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        with torch.no_grad():
            lS = forward(S, xte.expand(args.n, -1, -1))
            row = dict(
                step=step,
                member_test=acc(lS, yte_).mean().item(),
                member_test_max=acc(lS, yte_).max().item(),
                member_train=acc(forward(S, xs), ys).mean().item(),
                soup_test=acc(forward(soup(S), xte), yte_).item(),
                soup_train=acc(forward(soup(S), xu), yu).item(),
                ens_test=acc(lS.softmax(-1).mean(0, keepdim=True), yte_).item(),
                diffinit_soup_test=acc(forward(soup(R), xte), yte_).item(),
                diffinit_member_test=acc(forward(R, xte.expand(args.n, -1, -1)), yte_).mean().item(),
                union_test=acc(forward(U, xte), yte_).item(),
                union_train=acc(forward(U, xu), yu).item(),
            )
            log.append(row)
            if step % (args.eval_every * 10) == 0:
                print(f"[{time.time()-t0:6.0f}s] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    if step == args.steps: break
    for ps, opt, xx, yy in ((S, oS, xs, ys), (R, oR, xs, ys), (U, oU, xu, yu)):
        loss = F.cross_entropy(forward(ps, xx).flatten(0, 1), yy.flatten(), reduction="sum") / yy.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
    if args.merge_every and (step + 1) % args.merge_every == 0:
        with torch.no_grad():
            for k in S: S[k].copy_(S[k].mean(0, keepdim=True).expand_as(S[k]))

json.dump(dict(args=vars(args), log=log), open(args.out, "w"))
