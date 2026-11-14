import json, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# usage: plot.py out.png run1.json [run2.json ...]  (multiple runs = seeds; mean ± range)
out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
steps = np.array([r["step"] for r in runs[0]["log"]])
def series(key): return np.array([[r[key] for r in run["log"]] for run in runs])
curves = [
    ("union_test", "Centralized: 1 model on ALL data (test)", "#1f4e9c", "-"),
    ("soup_test", "Soup: weight-average of N shard models (test)", "#d1495b", "-"),
    ("ens_test", "Ensemble: average of N models' outputs (test)", "#edae49", "--"),
    ("member_test", "Single shard model (test, mean)", "#66a182", ":"),
    ("diffinit_soup_test", "Control: soup of different-init models (test)", "#888888", "-."),
]
fig, ax = plt.subplots(figsize=(9, 5.2))
for key, label, c, ls in curves:
    s = series(key); m = s.mean(0)
    ax.plot(steps, m, color=c, ls=ls, lw=2, label=label)
    if len(runs) > 1: ax.fill_between(steps, s.min(0), s.max(0), color=c, alpha=0.15, lw=0)
ax.plot(steps, series("member_train").mean(0), color="#66a182", lw=1, alpha=0.5, label="Single shard model (train)")
a = runs[0]["args"]
ax.set_title(f"Grokking Soup — {a['op']} mod {a['p']}, N={a['n']} shards, union frac={a['frac']}, wd={a['wd']}, {len(runs)} seed(s)")
ax.set_xlabel("training step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=8.5)
fig.tight_layout(); fig.savefig(out, dpi=130)

def first(key, thr=0.95):
    s = series(key); res = []
    for row in s:
        idx = np.where(row >= thr)[0]; res.append(int(steps[idx[0]]) if len(idx) else None)
    return res
for k in ["union_test", "soup_test", "ens_test", "member_test", "diffinit_soup_test"]:
    print(f"{k:22s} first step >=95%: {first(k)}   final: {series(k)[:, -1].round(3).tolist()}")
