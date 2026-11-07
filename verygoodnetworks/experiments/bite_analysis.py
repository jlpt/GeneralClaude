#!/usr/bin/env python3
"""Poison share of the n-th bite of each episode, for trained policies.

  bite_analysis.py --lib LIB --policy A.pt [--policy B.pt ...]
                   [--settings "maxFood=40 initialFood=40 difficulty=0 random_poison=1"]

With random_poison=1 the poisonous colour is random for each episode, so an
agent that learns within its life starts near 50% and gets lower with later
bites. Use harmless poison (difficulty=0): with damaging poison, episodes
where an agent sticks to the poisonous colour end early, which makes even a
non-learning agent look like it improves (see experiments/README.md).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rl import Bridge, Envs, Policy, with_prev  # noqa: E402

LABELS = ["1st", "2nd", "3rd", "4-5th", "6-10th", "11th+"]


def bin_of(k):
    return 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 3 if k <= 5 else 4 if k <= 10 else 5


def run(bridge, path, settings, steps, n, seed):
    ck = torch.load(path)
    pol = Policy(ck["obs_dim"], ck["n_actions"], ck["hidden"], prev=ck.get("prev", False))
    pol.load_state_dict(ck["state"])
    env = Envs(bridge, n, seed, settings, 3000)
    obs = torch.from_numpy(env.observe())
    h = torch.zeros(1, n, ck["hidden"])
    prev_a, prev_r = torch.zeros(n, dtype=torch.long), torch.zeros(n)
    count = np.zeros(n, int)
    bins = np.zeros((len(LABELS), 2))  # poison bites, all bites
    for _ in range(steps):
        with torch.no_grad():
            logits, _, h = pol.step(with_prev(pol, obs, prev_a, prev_r), h)
            a = torch.distributions.Categorical(logits=logits).sample()
        o, r, d, st = env.step(a.numpy() % 18)
        for i in np.nonzero(st[:, 0] + st[:, 1])[0]:
            for poison, k in ((1, st[i, 1]), (0, st[i, 0])):
                for _ in range(k):
                    count[i] += 1
                    b = bin_of(count[i])
                    bins[b, 1] += 1
                    bins[b, 0] += poison
        keep = torch.from_numpy(1 - d.astype(np.float32))
        h = h * keep.view(1, -1, 1)
        prev_a, prev_r = a * keep.long(), torch.from_numpy(r) * keep
        count[d.astype(bool)] = 0
        obs = torch.from_numpy(o)
    return bins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True)
    p.add_argument("--policy", action="append", required=True)
    p.add_argument("--settings", default="maxFood=40 initialFood=40 difficulty=0 random_poison=1")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()
    torch.manual_seed(5)
    torch.set_num_threads(1)
    bridge = Bridge(args.lib)
    for path in args.policy:
        bins = run(bridge, path, args.settings, args.steps, args.envs, args.seed)
        print(f"{path}: poison share of the n-th bite in an episode (bites counted)")
        print("   " + "  ".join(f"{label}: {100 * pz / max(1, n):4.1f}% ({int(n)})" for label, (pz, n) in zip(LABELS, bins)),
              flush=True)


if __name__ == "__main__":
    main()
