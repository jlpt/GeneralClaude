import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    r = json.load(open(f))
    for fk in r["forks"]:
        e = fk["traj"][-1]
        rows.append(dict(seed=r["args"]["seed"], fork=fk["fork"], parent=fk["parent_test_at_fork"],
                         parentK=fk["parent_test_at_fork_plus_k"], soup=e["soup_test"], ens=e["ens_test"],
                         member=e["member_test"], soup_train=e["soup_train"]))
x = np.array([r["parent"] for r in rows])
fig, ax = plt.subplots(figsize=(7.5, 6))
for key, lab, c, mk in [("parentK", "parent after K more steps on ALL data", "#1f4e9c", "o"),
                        ("soup", "soup (weight avg of N shard forks)", "#d1495b", "s"),
                        ("ens", "ensemble (output avg of N shard forks)", "#edae49", "^"),
                        ("member", "single shard fork", "#66a182", "v")]:
    ax.scatter(x, [r[key] for r in rows], c=c, marker=mk, s=36, alpha=0.8, label=lab, edgecolors="none")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="y = parent at fork")
ax.set_xlabel("parent test accuracy at the moment of forking"); ax.set_ylabel("test accuracy after K steps")
ax.set_title("Every fork point, all seeds (aligned by parent progress)")
ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper left"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout(); fig.savefig(out, dpi=130)

def frac(cond, sub): s = [cond(r) for r in sub]; return f"{sum(s)}/{len(s)}"
for name, lo, hi in [("pre-grok (parent<0.3)", 0, 0.3), ("transition (0.3-0.95)", 0.3, 0.95), ("grokked (>=0.95)", 0.95, 1.01)]:
    sub = [r for r in rows if lo <= r["parent"] < hi]
    if not sub: continue
    m = lambda k: np.mean([r[k] for r in sub])
    print(f"{name:24s} n={len(sub):2d}  parent={m('parent'):.3f} parent+K={m('parentK'):.3f} soup={m('soup'):.3f} "
          f"ens={m('ens'):.3f} member={m('member'):.3f} | soup>member {frac(lambda r: r['soup'] > r['member'], sub)} "
          f"soup>parent@fork {frac(lambda r: r['soup'] > r['parent'], sub)} soup>=parent+K {frac(lambda r: r['soup'] >= r['parentK'], sub)} "
          f"ens>soup {frac(lambda r: r['ens'] > r['soup'], sub)}")
