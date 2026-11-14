import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
forks = np.array([f["fork"] for f in runs[0]["forks"]])
def get(fn): return np.array([[fn(f) for f in r["forks"]] for r in runs])
S = {
 "Parent test acc at fork": (get(lambda f: f["parent_test_at_fork"]), "#1f4e9c", "-", "o"),
 "Parent test acc at fork+K (kept training on ALL data)": (get(lambda f: f["parent_test_at_fork_plus_k"]), "#1f4e9c", ":", None),
 "Soup of N shard forks (weight average)": (get(lambda f: f["traj"][-1]["soup_test"]), "#d1495b", "-", "s"),
 "Ensemble of N shard forks (output average)": (get(lambda f: f["traj"][-1]["ens_test"]), "#edae49", "--", "^"),
 "Single shard fork (mean)": (get(lambda f: f["traj"][-1]["member_test"]), "#66a182", "-", "v"),
}
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
ax = axs[0]
for lab, (v, c, ls, mk) in S.items():
    ax.plot(forks, v.mean(0), color=c, ls=ls, marker=mk, lw=2, label=lab)
    if len(runs) > 1: ax.fill_between(forks, v.min(0), v.max(0), color=c, alpha=0.12, lw=0)
a = runs[0]["args"]
ax.set_title(f"Test accuracy after forking into N={a['n']} shards for K={a['k']} steps")
ax.set_xlabel("fork step (parent training step at which it is split)"); ax.set_ylabel("test accuracy")
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axs[1]
gap_s = S["Soup of N shard forks (weight average)"][0] - S["Single shard fork (mean)"][0]
gap_e = S["Ensemble of N shard forks (output average)"][0] - S["Soup of N shard forks (weight average)"][0]
gap_p = S["Soup of N shard forks (weight average)"][0] - S["Parent test acc at fork"][0]
for lab, v, c in [("soup − single fork", gap_s, "#d1495b"), ("ensemble − soup", gap_e, "#edae49"), ("soup − parent@fork", gap_p, "#1f4e9c")]:
    ax.plot(forks, v.mean(0), marker="o", color=c, lw=2, label=lab)
    if len(runs) > 1: ax.fill_between(forks, v.min(0), v.max(0), color=c, alpha=0.12, lw=0)
ax.axhline(0, color="k", lw=0.8); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title("Gaps (positive = first term better)"); ax.set_xlabel("fork step")
fig.suptitle(f"{a['op']} mod {a['p']}, train frac {a['frac']}, lr {a['lr']}, wd {a['wd']}, {len(runs)} seeds (band = min/max)")
fig.tight_layout(); fig.savefig(out, dpi=130)
print("fork  " + "  ".join(f"{k[:28]:>28s}" for k in S))
for i, f in enumerate(forks):
    print(f"{f:5d} " + "  ".join(f"{v[:, i].mean():>20.3f} ±{(v[:, i].max()-v[:, i].min())/2:.3f}" for v, *_ in S.values()))
