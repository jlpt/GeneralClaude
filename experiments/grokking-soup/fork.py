"""
Fork-point sweep: train one model on the full training set; at fork step T_f, clone it into
N workers that each continue on a DISJOINT shard (1/N of the training data) for K steps.
Then weight-average the workers ("soup"). Question: from which T_f on is the soup a
generalizing model, and how does that compare to when the parent model itself generalizes?
"""
import argparse, json, math, time, copy
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=97); ap.add_argument("--op", default="add")
ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--n", type=int, default=4)
ap.add_argument("--parent_steps", type=int, default=8000); ap.add_argument("--k", type=int, default=1500)
ap.add_argument("--forks", default="0,100,200,400,700,1000,1500,2000,3000,4000")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
ap.add_argument("--d", type=int, default=64); ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--dmlp", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="fork.json")
args = ap.parse_args()
torch.manual_seed(args.seed); torch.set_num_threads(4)
P, D, H, DM = args.p, args.d, args.heads, args.dmlp; DH = D // H

a = torch.arange(P).repeat_interleave(P); b = torch.arange(P).repeat(P)
y = {"add": (a + b) % P, "sub": (a - b) % P, "mul": (a * b) % P, "x2y2": (a * a + b * b) % P}[args.op]
X = torch.stack([a, b, torch.full_like(a, P)], 1)
perm = torch.randperm(P * P)
n_train = int(args.frac * P * P) // args.n * args.n
tr, te = perm[:n_train], perm[n_train:]
shards = tr.view(args.n, -1)

def init(gen):
    def r(*s, std): return torch.randn(1, *s, generator=gen) * std
    s = 1 / math.sqrt(D)
    return dict(WE=r(P + 1, D, std=s), Wpos=r(3, D, std=s), WQ=r(H, D, DH, std=s), WK=r(H, D, DH, std=s),
                WV=r(H, D, DH, std=s), WO=r(H, DH, D, std=s), Win=r(D, DM, std=s), bin=torch.zeros(1, DM),
                Wout=r(DM, D, std=1 / math.sqrt(DM)), bout=torch.zeros(1, D), WU=r(D, P, std=s))

def forward(ps, x):  # batched over leading model dim N; x [N,B,3]
    N, B, _ = x.shape
    h = torch.gather(ps["WE"], 1, x.reshape(N, -1, 1).expand(-1, -1, D)).view(N, B, 3, D) + ps["Wpos"][:, None]
    cat = lambda W: W.permute(0, 2, 1, 3).reshape(N, D, D)
    hf = h.reshape(N, B * 3, D)
    k = torch.bmm(hf, cat(ps["WK"])).view(N, B, 3, H, DH); v = torch.bmm(hf, cat(ps["WV"])).view(N, B, 3, H, DH)
    hl = h[:, :, 2]; q = torch.bmm(hl, cat(ps["WQ"])).view(N, B, H, DH)
    att = (torch.einsum("nbhe,nbthe->nbht", q, k) / math.sqrt(DH)).softmax(-1)
    hl = hl + torch.bmm(torch.einsum("nbht,nbthe->nbhe", att, v).reshape(N, B, D), ps["WO"].reshape(N, D, D))
    m = F.relu(torch.baddbmm(ps["bin"][:, None], hl, ps["Win"]))
    hl = hl + torch.baddbmm(ps["bout"][:, None], m, ps["Wout"])
    return torch.bmm(hl, ps["WU"])

acc = lambda lg, yy: (lg.argmax(-1) == yy).float().mean(-1)
soup = lambda ps: {k: v.mean(0, keepdim=True) for k, v in ps.items()}
mkopt = lambda ps: torch.optim.AdamW(ps.values(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
def step(ps, opt, xx, yy):
    loss = F.cross_entropy(forward(ps, xx).flatten(0, 1), yy.flatten(), reduction="sum") / yy.shape[1]
    opt.zero_grad(); loss.backward(); opt.step()

xte, yte = X[te][None], y[te][None]; xu, yu = X[tr][None], y[tr][None]
xs, ys = X[shards], y[shards]
def ev(ps):
    with torch.no_grad(): return acc(forward(ps, xte), yte).item(), acc(forward(ps, xu), yu).item()

# ---- Phase 1: parent trajectory ----
forks = sorted(int(f) for f in args.forks.split(","))
U = {k: v.clone().requires_grad_() for k, v in init(torch.Generator().manual_seed(1000 + args.seed)).items()}
oU = mkopt(U); parent, snaps = [], {}
t0 = time.time()
for s in range(args.parent_steps + 1):
    if s % 100 == 0:
        te_, tr_ = ev(U); parent.append(dict(step=s, test=te_, train=tr_))
    if s in forks:
        snaps[s] = ({k: v.detach().clone() for k, v in U.items()}, copy.deepcopy(oU.state_dict()))
    if s == args.parent_steps: break
    step(U, oU, xu, yu)
print(f"parent done {time.time()-t0:.0f}s; test@end={parent[-1]['test']:.3f}", flush=True)

# ---- Phase 2: fork into N shard workers ----
results = []
for tf in forks:
    w, ost = snaps[tf]
    W = {k: v.repeat(args.n, *[1] * (v.dim() - 1)).clone().requires_grad_() for k, v in w.items()}
    oW = mkopt(W)
    # carry the parent's Adam moments into each worker
    sd = copy.deepcopy(ost)
    for i, st in sd["state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            st[key] = st[key].repeat(args.n, *[1] * (st[key].dim() - 1))
    if sd["state"]: oW.load_state_dict(sd)
    traj = []
    for s in range(args.k + 1):
        if s % 250 == 0:
            with torch.no_grad():
                lW = forward(W, xte.expand(args.n, -1, -1))
                st_, sr_ = ev(soup(W))
                traj.append(dict(k=s, soup_test=st_, soup_train=sr_,
                                 member_test=acc(lW, yte).mean().item(),
                                 member_train=acc(forward(W, xs), ys).mean().item(),
                                 ens_test=acc(lW.softmax(-1).mean(0, keepdim=True), yte).item()))
        if s == args.k: break
        step(W, oW, xs, ys)
    par_at = lambda t: min(parent, key=lambda r: abs(r["step"] - t))["test"]
    res = dict(fork=tf, parent_test_at_fork=par_at(tf), parent_test_at_fork_plus_k=par_at(tf + args.k), traj=traj)
    results.append(res); e = traj[-1]
    print(f"fork={tf:5d} parent@fork={res['parent_test_at_fork']:.3f} parent@fork+K={res['parent_test_at_fork_plus_k']:.3f} | "
          f"soup_test={e['soup_test']:.3f} soup_train={e['soup_train']:.3f} member_test={e['member_test']:.3f} "
          f"member_train={e['member_train']:.3f} ens_test={e['ens_test']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

json.dump(dict(args=vars(args), parent=parent, forks=results), open(args.out, "w"))
