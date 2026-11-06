"""Check that the engine's batched propagation equals kernel.cu-style event propagation.

kernel.cu's propagateKernel does, for every neuron that fires:

    for j in connections:  dest = packed >> 8;  w = int8(packed & 0xFF) / 127
                           atomicAdd(&act[dest], sourceAct * w)

This test runs a token through the network in spike mode, takes the real spike
payloads of a few populations, pushes them through their packed synapses
exactly that way (np.add.at standing in for atomicAdd), and compares the
dendrite currents with what the engine computed with matrix products.

    python test_kernel_style.py --snn qwen3.5-0.8b.snn
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snn_format as fmt  # noqa: E402
from engine import Engine  # noqa: E402


def propagate_events(eng, pop, payload):
    """payload: [n] numpy (spike count * theta, or graded value) for one batch row."""
    act = np.zeros(eng.header["n_slots"], dtype=np.float64)       # d_activations
    off = eng._offsets
    firing = np.nonzero(payload)[0]
    for i in firing:                                                # one thread per firing neuron
        g = pop["neuron_offset"] + i
        dest, w = fmt.unpack(eng._packed[off[g]:off[g + 1]])
        np.add.at(act, dest, payload[i] * (w.astype(np.float64) / 127.0))   # atomicAdd
    return act * eng._slot_gain                                      # per-slot gain (int8 scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snn", required=True)
    ap.add_argument("--T", type=int, default=16)
    args = ap.parse_args()
    eng = Engine(args.snn, mode="spike", T=args.T)
    tok = eng.tokenizer()
    ids = tok("The quick brown fox jumps over the lazy dog")["input_ids"]
    for t in ids[:-1]:
        eng.step([t])
    captured = {}
    names = ["L0.mlp", "L3.attn", "L5.gdn", "L10.attn_norm"]
    orig = eng._fire

    def spy(p, drive):
        out = orig(p, drive)
        if p["name"] in names:
            captured[p["name"]] = out[0].double().numpy()
        return out

    eng._fire = spy
    eng.step([ids[-1]])
    worst = 0.0
    for name in names:
        pop = eng.pop[name]
        payload = captured[name]
        act = propagate_events(eng, pop, payload)
        for pr in eng.out[name]:
            mat = (torch.from_numpy(payload).float() @ pr.weight.float().T).double().numpy()
            ev = act[pr.slot_offset:pr.slot_offset + pr.n_dst]
            err = np.abs(mat - ev).max() / (np.abs(ev).max() + 1e-12)
            worst = max(worst, err)
            print(f"{name:14s} -> {pr.dst}.{pr.port:5s} {np.count_nonzero(payload):5d} firing neurons, "
                  f"max rel diff {err:.2e}")
    assert worst < 1e-4, worst
    print("event-driven propagation matches the engine")


if __name__ == "__main__":
    main()
