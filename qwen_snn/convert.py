"""Convert Qwen3.5-0.8B (text part) into a spiking neuron network file (.snn).

    python convert.py --model /path/to/Qwen3.5-0.8B --out qwen3.5-0.8b.snn

Every weight matrix becomes packed int8 synapses (see snn_format.py).  Every
neuron belongs to a population; a population's type says what its neurons do
with the current their dendrites collect:

  token       sensory neurons, one per vocabulary entry; the input token's
              neuron fires once and its synapses are that token's embedding
  integrator  residual-stream neurons: membrane = sum of everything received
  rmsnorm     LIF neurons driven by divisive normalisation of their input
  swiglu      LIF neurons with two dendrites, drive = silu(gate) * up
  gdn         LIF output neurons of a Gated DeltaNet block: 4-tap delay lines
              on the input and a fast plastic synapse matrix (delta rule with
              decay) that carries memory from token to token
  attention   LIF output neurons of a softmax attention block with a
              fixed-size key/value ring buffer
  readout     motor neurons, one per vocabulary entry; their membrane is the
              logit of that token

After writing the file the converter runs the network on calibration text and
stores each spiking neuron's drive range, from which its threshold is set
(threshold = range / spikes per token window).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snn_format as fmt  # noqa: E402

LM = "model.language_model."
CHUNK_SYNAPSES = 1 << 24


class Port:
    def __init__(self, name, n):
        self.name, self.n, self.slot_offset = name, n, None


class Pop:
    def __init__(self, name, type_, n, ports=(), params=None, arrays=None):
        self.name, self.type, self.n = name, type_, n
        self.ports = list(ports)
        self.params = params or {}
        self.arrays = arrays or {}
        self.neuron_offset = None

    def port(self, name):
        return next(p for p in self.ports if p.name == name)


class Proj:
    """src population -> dst population's port.

    kind: dense (tensor rows = dst slots, cols = src neurons), identity,
    embed (token -> residual), lmhead (final norm -> vocab readout)."""

    def __init__(self, src, dst, port, kind, tensor=None):
        self.src, self.dst, self.port, self.kind, self.tensor = src, dst, port, kind, tensor


def quantize_rows(w):
    """Per-row symmetric int8: w ~= q / 127 * scale[:, None]."""
    w = w.float()
    scale = w.abs().amax(dim=1)
    scale[scale == 0] = 1.0
    q = torch.round(w / scale[:, None] * 127).clamp_(-127, 127).to(torch.int8)
    return q.numpy(), scale.numpy().astype(np.float32)


def build_graph(model_dir, window=8192):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    H, V, I = cfg["hidden_size"], cfg["vocab_size"], cfg["intermediate_size"]
    st = safe_open(os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors"), "pt")
    get = lambda k: st.get_tensor(k).float()

    pops, projs = [], []
    add = lambda p: (pops.append(p), p)[1]
    eps = cfg["rms_norm_eps"]

    add(Pop("token", "token", V))
    add(Pop("resid.0", "integrator", H, [Port("embed", H)]))
    projs.append(Proj("token", "resid.0", "embed", "embed", LM + "embed_tokens.weight"))
    prev = "resid.0"

    def norm(name, src, weight_key):
        add(Pop(name, "rmsnorm", H, [Port("x", H)], {"eps": eps},
                {"gain": (1.0 + get(weight_key)).numpy()}))
        projs.append(Proj(src, name, "x", "identity"))

    for l, kind in enumerate(cfg["layer_types"]):
        L, W = f"L{l}", f"{LM}layers.{l}."
        norm(f"{L}.attn_norm", prev, W + "input_layernorm.weight")
        if kind == "linear_attention":
            nh, dk, dv = cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
            assert cfg["linear_num_key_heads"] == nh
            mix = f"{L}.gdn"
            add(Pop(mix, "gdn", nh * dv,
                    [Port("qkv", nh * (2 * dk + dv)), Port("z", nh * dv), Port("b", nh), Port("a", nh)],
                    {"heads": nh, "dk": dk, "dv": dv, "conv": cfg["linear_conv_kernel_dim"], "eps": eps},
                    {"conv_w": get(W + "linear_attn.conv1d.weight")[:, 0, :].numpy(),
                     "A_log": get(W + "linear_attn.A_log").numpy(),
                     "dt_bias": get(W + "linear_attn.dt_bias").numpy(),
                     "norm_w": get(W + "linear_attn.norm.weight").numpy()}))
            for port, t in (("qkv", "in_proj_qkv"), ("z", "in_proj_z"), ("b", "in_proj_b"), ("a", "in_proj_a")):
                projs.append(Proj(f"{L}.attn_norm", mix, port, "dense", f"{W}linear_attn.{t}.weight"))
            out_t = W + "linear_attn.out_proj.weight"
        else:
            nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
            rp = cfg["rope_parameters"]
            mix = f"{L}.attn"
            add(Pop(mix, "attention", nh * hd,
                    [Port("q", nh * hd * 2), Port("k", nkv * hd), Port("v", nkv * hd)],
                    {"heads": nh, "kv_heads": nkv, "head_dim": hd, "eps": eps,
                     "rotary_dim": int(hd * rp["partial_rotary_factor"]), "rope_theta": rp["rope_theta"],
                     "window": window},
                    {"q_gain": (1.0 + get(W + "self_attn.q_norm.weight")).numpy(),
                     "k_gain": (1.0 + get(W + "self_attn.k_norm.weight")).numpy()}))
            for port in "qkv":
                projs.append(Proj(f"{L}.attn_norm", mix, port, "dense", f"{W}self_attn.{port}_proj.weight"))
            out_t = W + "self_attn.o_proj.weight"
        add(Pop(f"{L}.resid_a", "integrator", H, [Port("skip", H), Port("proj", H)]))
        projs.append(Proj(prev, f"{L}.resid_a", "skip", "identity"))
        projs.append(Proj(mix, f"{L}.resid_a", "proj", "dense", out_t))

        norm(f"{L}.mlp_norm", f"{L}.resid_a", W + "post_attention_layernorm.weight")
        add(Pop(f"{L}.mlp", "swiglu", I, [Port("gate", I), Port("up", I)]))
        projs.append(Proj(f"{L}.mlp_norm", f"{L}.mlp", "gate", "dense", W + "mlp.gate_proj.weight"))
        projs.append(Proj(f"{L}.mlp_norm", f"{L}.mlp", "up", "dense", W + "mlp.up_proj.weight"))
        add(Pop(f"{L}.resid_m", "integrator", H, [Port("skip", H), Port("proj", H)]))
        projs.append(Proj(f"{L}.resid_a", f"{L}.resid_m", "skip", "identity"))
        projs.append(Proj(f"{L}.mlp", f"{L}.resid_m", "proj", "dense", W + "mlp.down_proj.weight"))
        prev = f"{L}.resid_m"

    norm("final_norm", prev, LM + "norm.weight")
    add(Pop("vocab", "readout", V, [Port("logits", V)]))
    projs.append(Proj("final_norm", "vocab", "logits", "lmhead", LM + "embed_tokens.weight"))

    n_neurons = n_slots = 0
    for p in pops:
        p.neuron_offset = n_neurons
        n_neurons += p.n
        for port in p.ports:
            port.slot_offset = n_slots
            n_slots += port.n
    assert n_slots - 1 <= fmt.MAX_DEST, "dendrite slots exceed the 24-bit destination field"
    return cfg, st, pops, projs, n_neurons, n_slots


def float_weight(st, proj):
    """The unquantized weight of a projection as [dst slots, src neurons] (for verification)."""
    if proj.kind == "identity":
        return None
    w = st.get_tensor(proj.tensor).float()
    return w.t().contiguous() if proj.kind == "embed" else w


def convert(model_dir, out, leak=1.0, refractory=0, window=8192):
    t0 = time.time()
    cfg, st, pops, projs, n_neurons, n_slots = build_graph(model_dir, window)
    by_name = {p.name: p for p in pops}
    out_projs = {p.name: [] for p in pops}
    for pr in projs:
        out_projs[pr.src].append(pr)
    for lst in out_projs.values():
        lst.sort(key=lambda pr: by_name[pr.dst].port(pr.port).slot_offset)

    def fanout(pr):
        return 1 if pr.kind == "identity" else by_name[pr.dst].port(pr.port).n

    fan = np.zeros(n_neurons, dtype=np.int64)
    for p in pops:
        fan[p.neuron_offset:p.neuron_offset + p.n] = sum(fanout(pr) for pr in out_projs[p.name])
    offsets = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(fan, out=offsets[1:])
    n_syn = int(offsets[-1])

    slot_gain = np.ones(n_slots, dtype=np.float32)
    rng = np.zeros(n_neurons, dtype=np.float32)
    embed_q = {}

    def embed_quant(key):
        if key not in embed_q:
            embed_q.clear()
            embed_q[key] = quantize_rows(st.get_tensor(key))
        return embed_q[key]

    def synapses():
        for p in pops:
            plist = out_projs[p.name]
            if not plist:
                continue
            blocks = []  # (dest slots, int8 [n_dst, n_src] or None for identity)
            for pr in plist:
                port = by_name[pr.dst].port(pr.port)
                dest = np.arange(port.slot_offset, port.slot_offset + port.n, dtype=np.uint32)
                if pr.kind == "identity":
                    blocks.append(("id", dest, None))
                elif pr.kind == "embed":
                    q, s = embed_quant(pr.tensor)      # q: [V, H] rows = source tokens
                    rng[p.neuron_offset:p.neuron_offset + p.n] = s
                    blocks.append(("rows", dest, q))
                elif pr.kind == "lmhead":
                    q, s = embed_quant(pr.tensor)      # q: [V, H] rows = dst vocab slots
                    slot_gain[port.slot_offset:port.slot_offset + port.n] = s
                    blocks.append(("cols", dest, q))
                else:
                    q, s = quantize_rows(st.get_tensor(pr.tensor))
                    assert q.shape == (port.n, p.n), (pr.tensor, q.shape, port.n, p.n)
                    slot_gain[port.slot_offset:port.slot_offset + port.n] = s
                    blocks.append(("cols", dest, q))
            f = int(fan[p.neuron_offset])
            step = max(1, CHUNK_SYNAPSES // f)
            for i0 in range(0, p.n, step):
                i1 = min(p.n, i0 + step)
                parts = []
                for how, dest, q in blocks:
                    if how == "id":
                        parts.append(fmt.pack(dest[i0:i1, None], np.full((i1 - i0, 1), 127, np.int8)))
                    elif how == "rows":
                        parts.append(fmt.pack(dest[None, :], q[i0:i1]))
                    else:
                        parts.append(fmt.pack(dest[None, :], q[:, i0:i1].T))
                yield np.hstack(parts).ravel()
            print(f"  synapses from {p.name:16s} {p.n:>7d} neurons x {f:>6d}  ({time.time() - t0:.0f}s)", flush=True)

    spiking = {"rmsnorm", "swiglu", "gdn", "attention"}
    leak_a = np.ones(n_neurons, dtype=np.float32)
    refr_a = np.zeros(n_neurons, dtype=np.uint8)
    for p in pops:
        if p.type in spiking:
            sl = slice(p.neuron_offset, p.neuron_offset + p.n)
            leak_a[sl] = leak
            refr_a[sl] = refractory
            rng[sl] = 1.0  # placeholder until calibration

    w = fmt.Writer()
    w.add("syn.offsets", np.int64, offsets.shape, lambda: [offsets])
    w.add("syn.packed", np.uint32, (n_syn,), synapses)
    w.add("slot.gain", np.float32, (n_slots,), lambda: [slot_gain])
    w.add("neuron.range", np.float32, (n_neurons,), lambda: [rng])
    w.add("neuron.leak", np.float32, (n_neurons,), lambda: [leak_a])
    w.add("neuron.refractory", np.uint8, (n_neurons,), lambda: [refr_a])
    pop_meta = []
    for p in pops:
        arrays = {}
        for k, a in p.arrays.items():
            name = f"pop.{p.name}.{k}"
            a = np.ascontiguousarray(a, dtype=np.float32)
            w.add(name, np.float32, a.shape, (lambda a=a: [a]))
            arrays[k] = name
        pop_meta.append({"name": p.name, "type": p.type, "n": p.n, "neuron_offset": p.neuron_offset,
                         "ports": [{"name": q.name, "n": q.n, "slot_offset": q.slot_offset} for q in p.ports],
                         "params": p.params, "arrays": arrays})
    header = {
        "format": "QWENSNN1",
        "source_model": "Qwen/Qwen3.5-0.8B (text decoder)",
        "n_neurons": n_neurons, "n_slots": n_slots, "n_synapses": n_syn,
        "populations": pop_meta,
        "calibrated": False,
        "tokenizer_dir": os.path.abspath(model_dir),
        "eos_token_ids": [cfg["eos_token_id"], 248046],
    }
    print(f"{n_neurons:,} neurons, {n_slots:,} dendrite slots, {n_syn:,} synapses "
          f"({n_syn * 4 / 2**30:.2f} GiB packed)", flush=True)
    w.write(out, header)
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


def calibration_text(model_dir):
    """Chat-formatted exchanges written by the original model (make_calibration.py)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    chats = json.load(open(os.path.join(here, "data", "calibration_chats.json")))
    ids = []
    for c in chats:
        ids += tok.apply_chat_template(c, tokenize=True, enable_thinking=False, return_dict=True)["input_ids"]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Qwen3.5-0.8B directory (safetensors + tokenizer)")
    ap.add_argument("--out", required=True, help="output .snn file")
    ap.add_argument("--leak", type=float, default=1.0, help="membrane decay per timestep stored for spiking neurons")
    ap.add_argument("--refractory", type=int, default=0, help="refractory timesteps after a spike")
    ap.add_argument("--window", type=int, default=8192, help="attention ring-buffer length (tokens)")
    ap.add_argument("--calib-tokens", type=int, default=4096)
    ap.add_argument("--quantile", type=float, default=0.999,
                    help="per-neuron drive range = this quantile of |drive| over calibration tokens")
    ap.add_argument("--skip-write", action="store_true", help="only recalibrate an existing file")
    args = ap.parse_args()
    if not args.skip_write:
        convert(args.model, args.out, args.leak, args.refractory, args.window)
    from engine import calibrate
    ids = calibration_text(args.model)[:args.calib_tokens]
    calibrate(args.out, ids, quantile=args.quantile)


if __name__ == "__main__":
    main()
