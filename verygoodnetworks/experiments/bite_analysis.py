#!/usr/bin/env python3
"""Poison share of the n-th bite of each episode, for a trained policy in a random-poison world.

  bite_analysis.py SCRATCH_DIR [SETTINGS]

Use harmless poison (difficulty=0) to avoid survivorship effects (see experiments/README.md).
"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/user/GeneralClaude/verygoodnetworks/experiments")
from train_rl import Bridge, Policy, Envs, with_prev
torch.manual_seed(5); torch.set_num_threads(1)
S = sys.argv[1]
b = Bridge(S + "/libnworks_bridge_v5.so")
def run(path, settings, steps=15000, n=64):
    ck = torch.load(path)
    pol = Policy(ck["obs_dim"], ck["n_actions"], ck["hidden"], prev=ck.get("prev", False)); pol.load_state_dict(ck["state"])
    env = Envs(b, n, 4242, settings, 3000)
    obs = torch.from_numpy(env.observe()); h = torch.zeros(1, n, ck["hidden"])
    pa = torch.zeros(n, dtype=torch.long); pr = torch.zeros(n)
    count = np.zeros(n, int); bins = np.zeros((6, 2))
    def binof(k): return 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 3 if k <= 5 else 4 if k <= 10 else 5
    for t in range(steps):
        with torch.no_grad():
            logits, _, h = pol.step(with_prev(pol, obs, pa, pr), h)
            a = torch.distributions.Categorical(logits=logits).sample()
        o, r, d, st = env.step(a.numpy())
        for i in np.nonzero(st[:, 0] + st[:, 1])[0]:
            for kind, k in ((1, st[i, 1]), (0, st[i, 0])):
                for _ in range(k):
                    count[i] += 1; bb = binof(count[i]); bins[bb, 1] += 1; bins[bb, 0] += kind
        keep = torch.from_numpy(1 - d.astype(np.float32))
        h = h * keep.view(1, -1, 1); pa = a * keep.long(); pr = torch.from_numpy(r) * keep
        count[d.astype(bool)] = 0
        obs = torch.from_numpy(o)
    return bins
labels = ["1st", "2nd", "3rd", "4-5th", "6-10th", "11th+"]
for name, path in [("learning-to-learn agent", S + "/round7/l2l_init/policy.pt"),
                   ("memory agent (never trained on random poison)", S + "/gym/memory_near/policy.pt")]:
    bins = run(path, sys.argv[2] if len(sys.argv) > 2 else "maxFood=40 initialFood=40 difficulty=0.3 random_poison=1")
    print(name + ": poison share of the n-th bite in an episode (bites counted)", flush=True)
    print("   " + "  ".join(f"{l}: {100*p/max(1,n):4.1f}% ({int(n)})" for l, (p, n) in zip(labels, bins)), flush=True)
