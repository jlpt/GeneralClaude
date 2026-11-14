"""A tiny policy that learns to reconstruct 3D primitives from photos, trained only on held-out-view reward.

This is the RL loop in miniature. The "policy" is a linear softmax model rather than an LLM,
so it trains on a CPU in a couple of minutes, but the loop is the same one you'd run on an LLM:

    1. sample a target, show the policy its input photos
    2. sample G candidate scene programs from the policy
    3. render each candidate from HELD-OUT cameras and score it against the target
    4. GRPO: advantage = (reward - group mean) / group std; push up log-prob of above-average candidates
       RFT (rejection-sampling fine-tuning): keep only the best candidate and do a supervised step on it

The policy's "tokens" are: shape type, x position, z position, size, color.
Every evaluation uses targets the policy never trained on, scored from random new angles.
"""
import argparse
import csv
import os
import time

import numpy as np

from ig.episode import INPUT_CAMERAS, random_heldout_cams
from ig.render import render, to_png
from ig.reward import scene_score
from ig.scenes import PALETTE

TYPES = ["sphere", "box", "cylinder"]
XS = np.linspace(-0.6, 0.6, 7)
ZS = np.linspace(-0.6, 0.6, 7)
SIZES = np.linspace(0.15, 0.4, 5)
COLORS = ["red", "blue", "green", "yellow"]
HEADS = [("type", len(TYPES)), ("x", len(XS)), ("z", len(ZS)), ("size", len(SIZES)), ("color", len(COLORS))]
SLICES, _o = [], 0
for _, k in HEADS:
    SLICES.append(slice(_o, _o + k))
    _o += k
N_LOGITS = _o

IN_RES, OUT_RES, N_HELDOUT = 24, 32, 4


def build_scene(action):
    t, xi, zi, si, ci = (int(a) for a in action)
    s = float(SIZES[si])
    prim = {"type": TYPES[t], "center": [float(XS[xi]), s, float(ZS[zi])], "color": PALETTE[COLORS[ci]]}
    if TYPES[t] == "sphere":
        prim["radius"] = s
    elif TYPES[t] == "box":
        prim["size"] = [2 * s] * 3
    else:
        prim.update(radius=s, height=2 * s, axis="y")
    return [prim]


class ToyEpisode:
    def __init__(self, rng):
        self.answer = [int(rng.integers(k)) for _, k in HEADS]
        self.target = build_scene(self.answer)
        imgs = [render(self.target, c, IN_RES)[0] for c in INPUT_CAMERAS]
        self.features = np.concatenate([np.concatenate([i.ravel() for i in imgs]), [1.0]])
        self.cams = random_heldout_cams(rng, N_HELDOUT)
        self.target_renders = [render(self.target, c, OUT_RES) for c in self.cams]

    def reward(self, action):
        return scene_score(build_scene(action), self.target_renders, self.cams, OUT_RES)


def softmaxes(logits):
    out = []
    for sl in SLICES:
        z = logits[..., sl]
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        out.append(z / z.sum(axis=-1, keepdims=True))
    return out


def sample(probs_row, rng, greedy=False):
    return [int(p.argmax()) if greedy else int(rng.choice(len(p), p=p)) for p in probs_row]


class Adam:
    def __init__(self, shape, lr):
        self.m, self.v, self.t, self.lr = np.zeros(shape), np.zeros(shape), 0, lr

    def ascend(self, w, g, b1=0.9, b2=0.999):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mh, vh = self.m / (1 - b1 ** self.t), self.v / (1 - b2 ** self.t)
        w += self.lr * mh / (np.sqrt(vh) + 1e-8)


def evaluate(W, test_eps, rng):
    X = np.stack([e.features for e in test_eps])
    probs = softmaxes(X @ W)
    greedy, exact = [], 0
    for i, e in enumerate(test_eps):
        a = sample([p[i] for p in probs], rng, greedy=True)
        greedy.append(e.reward(a))
        exact += a == e.answer
    return float(np.mean(greedy)), exact / len(test_eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["grpo", "rft"], default="grpo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16, help="targets per step")
    ap.add_argument("--group", type=int, default=8, help="candidates sampled per target (G)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    test_rng = np.random.default_rng(10_000 + args.seed)      # disjoint targets AND angles
    test_eps = [ToyEpisode(test_rng) for _ in range(64)]
    dim = len(test_eps[0].features)
    W = np.zeros((dim, N_LOGITS))                            # all-zero = uniform random policy
    opt = Adam(W.shape, args.lr)

    os.makedirs("out", exist_ok=True)
    log_path = f"out/toy_{args.algo}_curve.csv"
    log = csv.writer(open(log_path, "w", newline=""))
    log.writerow(["step", "train_mean_sampled_reward", "test_greedy_heldout_reward", "test_exact_match"])

    rand = np.mean([e.reward([int(rng.integers(k)) for _, k in HEADS]) for e in test_eps for _ in range(4)])
    print(f"algo={args.algo}  policy params={W.size}  test set=64 unseen targets x {N_HELDOUT} unseen angles")
    print(f"random-guess baseline on the test set: {rand:.3f}   (perfect = 1.000)")
    print(f"{'step':>5} {'train reward (sampled)':>23} {'test reward (held-out)':>23} {'test exact':>11}")
    t0 = time.time()
    train_hist = []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            test_r, exact = evaluate(W, test_eps, rng)
            tr = np.mean(train_hist[-args.eval_every:]) if train_hist else float("nan")
            print(f"{step:5d} {tr:23.3f} {test_r:23.3f} {exact:11.0%}")
            log.writerow([step, tr, test_r, exact])
        if step == args.steps:
            break

        eps = [ToyEpisode(rng) for _ in range(args.batch)]
        X = np.stack([e.features for e in eps])
        probs = softmaxes(X @ W)
        grad_logits = np.zeros((args.batch, N_LOGITS))
        for i, e in enumerate(eps):
            row = [p[i] for p in probs]
            actions = [sample(row, rng) for _ in range(args.group)]
            r = np.array([e.reward(a) for a in actions])
            train_hist.append(r.mean())
            if args.algo == "grpo":
                weights = (r - r.mean()) / (r.std() + 1e-6) if r.std() > 1e-6 else np.zeros_like(r)
            else:  # rft: imitate the single best sample (only if it is decent)
                weights = np.zeros_like(r)
                if r.max() > 0.3:
                    weights[r.argmax()] = 1.0
            for a, wgt in zip(actions, weights):
                if wgt == 0:
                    continue
                for (sl, p, ai) in zip(SLICES, row, a):
                    onehot = np.zeros(len(p)); onehot[ai] = 1
                    grad_logits[i, sl] += wgt * (onehot - p)     # d log pi(a) / d logits
        grad_W = X.T @ grad_logits / (args.batch * args.group)
        opt.ascend(W, grad_W)

    print(f"done in {time.time() - t0:.0f}s, curve -> {log_path}")

    # picture: 4 unseen test targets vs the trained policy's reconstructions, from held-out angles
    probs = softmaxes(np.stack([e.features for e in test_eps[:4]]) @ W)
    rows = []
    for i, e in enumerate(test_eps[:4]):
        pred = build_scene(sample([p[i] for p in probs], rng, greedy=True))
        tgt = np.concatenate([render(e.target, c, 64)[0] for c in e.cams[:3]], axis=1)
        prd = np.concatenate([render(pred, c, 64)[0] for c in e.cams[:3]], axis=1)
        rows.append(np.concatenate([tgt, np.full((64, 6, 3), 0.5), prd], axis=1))
    to_png(np.concatenate(rows, axis=0), f"out/toy_{args.algo}_examples.png", scale=2)
    print(f"examples -> out/toy_{args.algo}_examples.png (left: target, right: prediction; held-out angles)")


if __name__ == "__main__":
    main()
