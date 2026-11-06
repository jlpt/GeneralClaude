"""Compare the spiking network with the original Qwen3.5-0.8B (transformers, fp32).

    python verify.py --snn qwen3.5-0.8b.snn --model /path/to/Qwen3.5-0.8B --text a.txt b.txt ...

Each text gives one sequence (first --tokens tokens).  The original model sees
the whole sequence at once; the SNN gets it one token per window, carrying
only its recurrent state between tokens.  Reported per configuration:

  top1   how often the SNN's most likely next token equals the original's
  KL     mean KL(original || SNN) of the next-token distributions (nats)
  ppl    perplexity of the actual next tokens (original model's in brackets)
  spikes spikes per token (all spiking neurons), synaptic events per token
"""
import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine  # noqa: E402

DEFAULT_TEXTS = ["/usr/share/common-licenses/GPL-3", "/usr/local/lib/python3.13/LICENSE.txt",
                 "/usr/share/common-licenses/Apache-2.0", "/usr/share/common-licenses/MPL-2.0"]


def load_sequences(model_dir, paths, n_tokens):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    seqs = []
    for p in paths:
        text = open(p, encoding="utf-8", errors="replace").read()
        text = " ".join(text.split())          # collapse the license files' hard wrapping
        ids = tok(text)["input_ids"][:n_tokens]
        assert len(ids) == n_tokens, f"{p} has fewer than {n_tokens} tokens"
        seqs.append(ids)
    return torch.tensor(seqs)


@torch.no_grad()
def reference_logprobs(model_dir, seqs):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    lp = torch.log_softmax(m(seqs).logits.float(), -1).half()
    del m
    gc.collect()
    return lp


def use_float_weights(eng, model_dir):
    """Swap the int8 synapses for the original fp32 weights (checks the wiring, not the quantization)."""
    from convert import build_graph, float_weight
    _, st, _, projs, _, _ = build_graph(model_dir)
    for gp in projs:
        if gp.kind == "identity":
            continue
        pr = eng.projection(gp.src, gp.dst, gp.port)
        w = float_weight(st, gp)
        if pr.columns is not None:
            pr.columns, pr.col_gain = w, torch.ones(w.shape[0])
            p = eng.pop[gp.src]
            eng.range[p["neuron_offset"]:p["neuron_offset"] + p["n"]] = 1.0
        else:
            pr.weight = w.to(eng.dtype)


@torch.no_grad()
def run(eng, seqs, ref_lp, label):
    B, L = seqs.shape
    eng.reset(B)
    eng.stats = {}
    t0 = time.time()
    agree = kl = nll = 0.0
    for t in range(L):
        logits = eng.step(seqs[:, t])
        lp = torch.log_softmax(logits.float(), -1)
        r = ref_lp[:, t].float()
        agree += (lp.argmax(-1) == r.argmax(-1)).float().sum().item()
        kl += (r.exp() * (r - lp)).sum(-1).sum().item()
        if t + 1 < L:
            nll += -lp.gather(1, seqs[:, t + 1:t + 2]).sum().item()
    n = B * L
    ppl = torch.tensor(nll / (B * (L - 1))).exp().item()
    spikes = sum(v[0] for k, v in eng.stats.items() if k != "token") / n
    events = sum(v[1] for v in eng.stats.values()) / n
    dt = time.time() - t0
    print(f"{label:34s} top1 {100 * agree / n:6.2f}%   KL {kl / n:8.4f}   ppl {ppl:8.3f}   "
          f"spikes/token {spikes:12,.0f}   syn events/token {events:14,.0f}   {dt / L:5.2f}s/step", flush=True)
    return {"label": label, "top1": agree / n, "kl": kl / n, "ppl": ppl, "spikes": spikes, "events": events}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snn", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", nargs="*", default=[p for p in DEFAULT_TEXTS if os.path.exists(p)])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--T", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--leak", type=float, nargs="*", default=[1.0])
    ap.add_argument("--refractory", type=int, default=None)
    ap.add_argument("--float-check", action="store_true", help="also run with the original fp32 weights")
    ap.add_argument("--no-graded", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    seqs = load_sequences(args.model, args.text, args.tokens)
    print(f"{seqs.shape[0]} sequences x {seqs.shape[1]} tokens", flush=True)
    ref = reference_logprobs(args.model, seqs)
    nll = -ref[:, :-1].float().gather(2, seqs[:, 1:, None]).mean()
    print(f"original model ppl {nll.exp().item():.3f}", flush=True)

    eng = Engine(args.snn, mode="graded", refractory=args.refractory)
    results = []
    if args.float_check:
        saved = [(pr, pr.weight, pr.columns, pr.col_gain) for prs in eng.out.values() for pr in prs]
        saved_range = eng.range.copy()
        use_float_weights(eng, args.model)
        results.append(run(eng, seqs, ref, "graded, fp32 weights (wiring check)"))
        for pr, w, c, g in saved:
            pr.weight, pr.columns, pr.col_gain = w, c, g
        eng.range[:] = saved_range
    if not args.no_graded:
        results.append(run(eng, seqs, ref, "graded spikes, int8 synapses"))
    eng.mode = "spike"
    for leak in args.leak:
        for p in eng.pops:
            if "leak" in eng.params[p["name"]] and p["type"] != "token":
                eng.params[p["name"]]["leak"].fill_(leak)
        for T in args.T:
            eng.T = T
            results.append(run(eng, seqs, ref, f"LIF spikes T={T} leak={leak}"))
    return results


if __name__ == "__main__":
    main()
