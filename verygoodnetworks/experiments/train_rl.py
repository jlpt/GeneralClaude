#!/usr/bin/env python3
"""Train a brain with reinforcement learning instead of evolving it.

A small recurrent network (GRU) is trained with PPO to control one creature in
the Nworks world, through the C bridge (libnworks_bridge.so). It sees the same
inputs an evolved creature's sensors produce and picks one of 18 actions:
stay or move in one of 8 directions, each with eat / don't eat.

Reward: +1 per good bite, -1 per poison bite, -2 for dying.

  train_rl.py train --lib LIB --out DIR [--steps N] [--settings "..."]
  train_rl.py train --lib LIB --out DIR --multi 8 --envs 8 --settings "... share=0.5"
  train_rl.py eval  --lib LIB --policy DIR/policy.pt [--trials 400]

With --multi K, every world has K creatures driven by the same network, which
also gets a call action (36 actions); "share=S" in --settings mixes each
creature's reward with its world's average.

--settings are Nworks Config overrides ("key=value ..."), plus difficulty=D and
random_poison=1 (poisonous colour random per episode) for training. The test
battery is the same C++ code nworks_iq uses.
"""
import argparse
import ctypes
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn


class Bridge:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        vp, fp, ip, bp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8)
        lib.nwb_create.restype = vp
        lib.nwb_create.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_int]
        lib.nwb_obs_dim.argtypes = [vp]
        lib.nwb_observe.argtypes = [vp, fp]
        lib.nwb_step.argtypes = [vp, ip, fp, fp, bp, ip]
        lib.nwb_destroy.argtypes = [vp]
        lib.nws_create.restype = vp
        lib.nws_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        lib.nws_obs_dim.argtypes = [vp]
        lib.nws_observe.argtypes = [vp, fp]
        lib.nws_step.argtypes = [vp, ip, bp]
        lib.nws_results.argtypes = [vp, bp, fp, bp]
        lib.nws_learning_results.argtypes = [vp, ip]
        lib.nws_destroy.argtypes = [vp]
        lib.nw_test_name.restype = ctypes.c_char_p
        lib.nwm_create.restype = vp
        lib.nwm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_int]
        lib.nwm_obs_dim.argtypes = [vp]
        lib.nwm_observe.argtypes = [vp, fp]
        lib.nwm_step.argtypes = [vp, ip, fp, fp, bp, ip]
        lib.nwm_destroy.argtypes = [vp]
        self.lib = lib

    @staticmethod
    def ptr(a, t):
        return a.ctypes.data_as(ctypes.POINTER(t))


class Envs:
    """n training episodes stepped together."""

    def __init__(self, bridge, n, seed, settings, episode_steps):
        self.b, self.n = bridge, n
        self.h = bridge.lib.nwb_create(n, seed, settings.encode(), episode_steps)
        if not self.h:
            raise SystemExit(f"bad settings: {settings}")
        self.dim = bridge.lib.nwb_obs_dim(self.h)
        self.obs = np.zeros((n, self.dim), np.float32)
        self.rew = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.uint8)
        self.stats = np.zeros((n, 3), np.int32)

    def observe(self):
        self.b.lib.nwb_observe(self.h, Bridge.ptr(self.obs, ctypes.c_float))
        return self.obs.copy()

    def step(self, actions):
        a = np.ascontiguousarray(actions, np.int32)
        self.b.lib.nwb_step(self.h, Bridge.ptr(a, ctypes.c_int32), Bridge.ptr(self.obs, ctypes.c_float),
                            Bridge.ptr(self.rew, ctypes.c_float), Bridge.ptr(self.done, ctypes.c_uint8),
                            Bridge.ptr(self.stats, ctypes.c_int32))
        return self.obs.copy(), self.rew.copy(), self.done.copy(), self.stats.copy()


class MultiEnvs(Envs):
    """n worlds with `per` creatures each, all driven by one network: n * per agents."""

    def __init__(self, bridge, n, per, seed, settings, episode_steps):
        self.b, self.n = bridge, n * per
        self.h = bridge.lib.nwm_create(n, per, seed, settings.encode(), episode_steps)
        if not self.h:
            raise SystemExit(f"bad settings: {settings}")
        self.dim = bridge.lib.nwm_obs_dim(self.h)
        self.obs = np.zeros((self.n, self.dim), np.float32)
        self.rew = np.zeros(self.n, np.float32)
        self.done = np.zeros(self.n, np.uint8)
        self.stats = np.zeros((self.n, 4), np.int32)

    def observe(self):
        self.b.lib.nwm_observe(self.h, Bridge.ptr(self.obs, ctypes.c_float))
        return self.obs.copy()

    def step(self, actions):
        a = np.ascontiguousarray(actions, np.int32)
        self.b.lib.nwm_step(self.h, Bridge.ptr(a, ctypes.c_int32), Bridge.ptr(self.obs, ctypes.c_float),
                            Bridge.ptr(self.rew, ctypes.c_float), Bridge.ptr(self.done, ctypes.c_uint8),
                            Bridge.ptr(self.stats, ctypes.c_int32))
        return self.obs.copy(), self.rew.copy(), self.done.copy(), self.stats[:, :3].copy()


# With the default body, inputs 40-47 are "predator in this direction" (1 = touching,
# 0 = nothing within 300 px). Above 0.38 means a predator within about 200 px.
PREDATOR_INPUTS = slice(40, 48)
PREDATOR_NEAR = 0.38


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def step(self, obs, h):
        y, h = self.gru(self.enc(obs).unsqueeze(0), h)
        return self.pi(y[0]), self.v(y[0]).squeeze(-1), h

    def unroll(self, obs, h, dones):
        """obs (T, N, D); the hidden state resets after every step where dones is 1."""
        x = self.enc(obs)
        outs = []
        for t in range(obs.shape[0]):
            y, h = self.gru(x[t:t + 1], h)
            outs.append(y[0])
            h = h * (1.0 - dones[t]).view(1, -1, 1)
        y = torch.stack(outs)
        return self.pi(y), self.v(y).squeeze(-1)


def strip_difficulty(settings):
    """The battery takes plain Config overrides: drop the training-only options."""
    return " ".join(s for s in settings.split() if not s.startswith(("difficulty=", "random_poison=")))


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    bridge = Bridge(args.lib)
    if args.multi:
        envs = MultiEnvs(bridge, args.envs, args.multi, args.seed, args.settings, args.episode_steps)
        n_actions = bridge.lib.nwm_num_actions()
    else:
        envs = Envs(bridge, args.envs, args.seed, args.settings, args.episode_steps)
        n_actions = bridge.lib.nw_num_actions()
    policy = Policy(envs.dim, n_actions, args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | {"obs_dim": envs.dim, "n_actions": n_actions}, f, indent=1)
    log = open(os.path.join(args.out, "log.txt"), "a")

    N, T = envs.n, args.rollout
    calls = np.zeros(4)  # calls with a predator near, steps with a predator near, calls otherwise, other steps
    obs = torch.from_numpy(envs.observe())
    h = torch.zeros(1, N, args.hidden)
    ep_ret = np.zeros(N)
    finished, totals = [], np.zeros(3)  # finished episode returns; good, poison, deaths since last log
    steps, start, update = 0, time.time(), 0
    while steps < args.steps:
        h0 = h.clone()
        buf_obs = torch.zeros(T, N, envs.dim)
        buf_act = torch.zeros(T, N, dtype=torch.long)
        buf_logp, buf_val, buf_rew, buf_done = (torch.zeros(T, N) for _ in range(4))
        for t in range(T):
            with torch.no_grad():
                logits, v, h = policy.step(obs, h)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            nobs, r, d, st = envs.step(a.numpy())
            buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = obs, a, dist.log_prob(a), v
            buf_rew[t], buf_done[t] = torch.from_numpy(r), torch.from_numpy(d.astype(np.float32))
            h = h * (1.0 - buf_done[t]).view(1, -1, 1)
            obs = torch.from_numpy(nobs)
            ep_ret += r
            totals += st.sum(0)
            for i in np.nonzero(d)[0]:
                finished.append(ep_ret[i])
                ep_ret[i] = 0.0
        steps += N * T
        if args.multi:
            near = (buf_obs[:, :, PREDATOR_INPUTS].max(-1).values > PREDATOR_NEAR).numpy()
            called = (buf_act >= 18).numpy()
            calls += [(called & near).sum(), near.sum(), (called & ~near).sum(), (~near).sum()]
        with torch.no_grad():
            _, v_last, _ = policy.step(obs, h)
        adv = torch.zeros(T, N)
        last = torch.zeros(N)
        for t in reversed(range(T)):
            nxt = v_last if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * nxt * nonterm - buf_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + buf_val

        chunk = N // args.minibatches
        for _ in range(args.epochs):
            perm = torch.randperm(N)
            for m in range(args.minibatches):
                idx = perm[m * chunk:(m + 1) * chunk]
                logits, v = policy.unroll(buf_obs[:, idx], h0[:, idx], buf_done[:, idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(buf_act[:, idx])
                a_mb = adv[:, idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                ratio = torch.exp(logp - buf_logp[:, idx])
                pg = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_mb).mean()
                vloss = 0.5 * ((v - ret[:, idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vcoef * vloss - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        update += 1

        if update % args.log_every == 0 or steps >= args.steps:
            good, poison, deaths = totals
            per = 1000.0 / (N * T * args.log_every)
            line = (f"steps {steps:>10} | episodes {len(finished):>6} | return {np.mean(finished[-200:]) if finished else 0:7.2f}"
                    f" | per 1000 steps: good {good * per:5.2f} poison {poison * per:5.2f} deaths {deaths * per:5.3f}"
                    f" | poison share {100 * poison / max(1, good + poison):5.1f}% | {steps / (time.time() - start):6.0f} steps/s")
            if args.multi:
                line += (f" | calls: predator near {100 * calls[0] / max(1, calls[1]):5.1f}%"
                         f", otherwise {100 * calls[2] / max(1, calls[3]):5.1f}%")
                calls[:] = 0
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            totals[:] = 0
            torch.save({"state": policy.state_dict(), "obs_dim": envs.dim, "n_actions": n_actions, "multi": args.multi,
                        "hidden": args.hidden, "settings": args.settings, "steps": steps}, os.path.join(args.out, "policy.pt"))


def evaluate(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    bridge = Bridge(args.lib)
    ck = torch.load(args.policy)
    policy = Policy(ck["obs_dim"], ck["n_actions"], ck["hidden"])
    policy.load_state_dict(ck["state"])
    settings = strip_difficulty(args.settings if args.settings is not None else ck["settings"])
    settings = " ".join(s for s in settings.split() if not s.startswith("share=")).encode()
    print(f"trained policy {args.policy} ({ck['steps']} training steps), {args.trials} trials per test:")
    lib = bridge.lib
    for t in range(lib.nw_num_tests()):
        s = lib.nws_create(t, args.trials, settings)
        dim = lib.nws_obs_dim(s)
        obs = np.zeros((args.trials, dim), np.float32)
        dones = np.zeros(args.trials, np.uint8)
        h = torch.zeros(1, args.trials, ck["hidden"])
        running = args.trials
        while running:
            lib.nws_observe(s, Bridge.ptr(obs, ctypes.c_float))
            with torch.no_grad():
                logits, _, h = policy.step(torch.from_numpy(obs), h)
                a = torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int32)
                a %= 18  # a policy with a call action plays the battery without it
            running = lib.nws_step(s, Bridge.ptr(a, ctypes.c_int32), Bridge.ptr(dones, ctypes.c_uint8))
        succ = np.zeros(args.trials, np.uint8)
        score = np.zeros(args.trials, np.float32)
        bit = np.zeros(args.trials, np.uint8)
        lib.nws_results(s, Bridge.ptr(succ, ctypes.c_uint8), Bridge.ptr(score, ctypes.c_float), Bridge.ptr(bit, ctypes.c_uint8))
        learn = np.zeros((args.trials, 4), np.int32)
        lib.nws_learning_results(s, Bridge.ptr(learn, ctypes.c_int32))
        lib.nws_destroy(s)
        name = lib.nw_test_name(t).decode()
        line = f"    {name:<15} success {100 * succ.mean():5.1f}%"
        if name.startswith("flee"):
            line += f"  heading {score.mean():+.2f}"
        if name == "poison-learning":
            ate, safe, lb, ls = learn.sum(0)
            line += (f"  (round 1: ate in {100 * ate / args.trials:.1f}%, safe in {100 * safe / max(1, ate):.1f}% of those;"
                     f" rounds 2-6: {lb / args.trials:.1f} bites per trial, {100 * ls / max(1, lb):.1f}% safe)")
        if name == "poison-choice":
            line += f"  (ate one in {100 * bit.mean():.1f}%; picked purple in {100 * succ.sum() / max(1, bit.sum()):.1f}% of those)"
        print(line, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--lib", required=True)
    p.add_argument("--out", default="rl_run")
    p.add_argument("--policy")
    p.add_argument("--settings", default=None)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--episode-steps", type=int, default=3000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--multi", type=int, default=0, help="creatures per world (0 = one creature, no calls)")
    args = p.parse_args()
    if args.mode == "train":
        if args.settings is None:
            args.settings = "foodVisibleSteps=60 maxFood=40 initialFood=40 difficulty=0.3"
        train(args)
    else:
        if not args.policy:
            p.error("eval needs --policy")
        evaluate(args)


if __name__ == "__main__":
    main()
