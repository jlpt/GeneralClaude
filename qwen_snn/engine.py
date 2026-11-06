"""Run a .snn network: leaky integrate-and-fire neurons wired by packed int8 synapses.

The engine reads only the .snn file (plus a tokenizer for text in/out).  It
does not import transformers or know anything about Qwen layers: the file's
populations, their neuron types and the packed synapses are all it uses.

One token = one "window" of T timesteps.  For every population, in the order
the file lists them:

  1. its dendrite slots hold the current that arrived this window
  2. its neurons turn that into a drive (sum, normalisation, gate, ...)
  3. spiking neurons fire: LIF dynamics for T steps, the drive arriving as a
     constant current of drive/T per step (so a window's input totals drive),
         v <- leak * v + drive / T
         spike +1 if v >= +theta/2, -1 if v <= -theta/2 (and not refractory);
         a membrane past several thresholds fires that many spikes at once
         (a burst; max_burst=1 forbids it)
         v <- v - spikes * theta           (reset by subtraction)
         after a spike: refractory for r steps (v keeps integrating, as in kernel.cu)
     each spike pushes +-theta * w/127 * gain into every target slot
  4. analog neurons (residual stream, readout) push their membrane value
     instead (a graded event, the same as kernel.cu's sourceAct * weight)

mode="graded" makes the spiking neurons do what kernel.cu's neurons do: fire
once per window and push their whole activation (sourceAct * weight).  With
int8 weights that reproduces the original network's arithmetic.
mode="spike" uses the binary LIF dynamics above; accuracy rises with T.

Memory across tokens (the recurrent part) lives in:
  gdn        4-tap delay lines + fast plastic synapse matrix S (per head),
             S <- S*decay; S <- S + k (x) beta*(v - S^T k)   (delta rule)
  attention  key/value ring buffer of fixed length (`window` tokens)
  membranes  optionally carried between windows (carry_membrane=True)
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snn_format as fmt  # noqa: E402

SPIKING = {"rmsnorm", "swiglu", "gdn", "attention"}


class Projection:
    """All synapses from one source population into one destination port."""

    def __init__(self, src, dst, port, slot_offset, n_dst, n_src):
        self.src, self.dst, self.port = src, dst, port
        self.slot_offset, self.n_dst, self.n_src = slot_offset, n_dst, n_src
        self.weight = None   # [n_dst, n_src] real weights (int8/127*gain), dense
        self.diag = None     # [n] if the synapses are one-to-one
        self.columns = None  # int8 [n_dst, n_src] kept for event-driven (one-hot) sources
        self.col_gain = None


class Engine:
    def __init__(self, path, device="cpu", dtype=torch.float32, mode="graded", T=32,
                 leak=None, refractory=None, fire_level=0.5, carry_membrane=False, max_burst=None,
                 verbose=True):
        t0 = time.time()
        self.path, self.device, self.dtype = path, torch.device(device), dtype
        self.header = h = fmt.read_header(path)
        self.pops = h["populations"]
        self.pop = {p["name"]: p for p in self.pops}
        self.mode, self.T, self.fire_level, self.carry_membrane = mode, T, fire_level, carry_membrane
        self.max_burst = max_burst
        self.verbose = verbose
        arr = lambda name: fmt.open_array(path, h, name)
        self._offsets = arr("syn.offsets")
        self._packed = arr("syn.packed")
        self._slot_gain = np.array(arr("slot.gain"))
        self.range = np.array(arr("neuron.range"))
        leak_a = np.array(arr("neuron.leak"))
        refr_a = np.array(arr("neuron.refractory")).astype(np.int64)
        dev = self.device
        self.params = {}
        for p in self.pops:
            P = dict(p["params"])
            for k, name in p["arrays"].items():
                P[k] = torch.from_numpy(np.array(arr(name))).to(dev)
            sl = slice(p["neuron_offset"], p["neuron_offset"] + p["n"])
            if p["type"] in SPIKING or p["type"] == "token":
                P["range"] = torch.from_numpy(self.range[sl].copy()).to(dev)
                P["leak"] = torch.from_numpy(leak_a[sl].copy()).to(dev)
                P["refractory"] = torch.from_numpy(refr_a[sl].copy()).to(dev)
                if leak is not None:
                    P["leak"].fill_(leak)
                if refractory is not None:
                    P["refractory"].fill_(refractory)
            self.params[p["name"]] = P
        self._decode_synapses()
        self.stats = {}
        self.reset(1)
        if verbose:
            print(f"loaded {h['n_neurons']:,} neurons / {h['n_synapses']:,} synapses "
                  f"in {time.time() - t0:.0f}s ({mode}, T={T})", flush=True)

    # ------------------------------------------------------------------ synapses
    def _slot_owner(self):
        starts, owners = [], []
        for p in self.pops:
            for q in p["ports"]:
                starts.append(q["slot_offset"])
                owners.append((p["name"], q["name"], q["slot_offset"], q["n"]))
        return np.array(starts), owners

    def _decode_synapses(self):
        """Unpack every population's outgoing packed synapses into per-port matrices.

        A population whose neurons all have the same fan-out to the same slots
        (a dense layer) is decoded row-block by row-block; anything else goes
        through a generic scatter."""
        starts, owners = self._slot_owner()
        self.out = {p["name"]: [] for p in self.pops}
        off = self._offsets
        for p in self.pops:
            a, n = p["neuron_offset"], p["n"]
            s0, s1 = int(off[a]), int(off[a + n])
            if s0 == s1:
                continue
            fan = np.diff(np.asarray(off[a:a + n + 1]))
            mats = {}
            uniform = bool((fan == fan[0]).all())
            f = int(fan[0])
            step = max(1, (1 << 24) // max(f, 1)) if uniform else n
            for i0 in range(0, n, step):
                i1 = min(n, i0 + step)
                c0, c1 = int(off[a + i0]), int(off[a + i1])
                dest, w = fmt.unpack(self._packed[c0:c1])
                if uniform:
                    dest, w = dest.reshape(i1 - i0, f), w.reshape(i1 - i0, f)
                    if (dest == dest[0]).all():          # same targets for every source: dense block
                        pid = np.searchsorted(starts, dest[0], side="right") - 1
                        for u in np.unique(pid):
                            name, port, so, pn = owners[u]
                            ks = np.nonzero(pid == u)[0]
                            m = mats.setdefault(u, np.zeros((pn, n), np.int8))
                            m[dest[0, ks] - so, i0:i1] = w[:, ks].T
                        continue
                    dest, w = dest.ravel(), w.ravel()
                src = np.repeat(np.arange(i0, i1), fan[i0:i1])
                pid = np.searchsorted(starts, dest, side="right") - 1
                for u in np.unique(pid):
                    name, port, so, pn = owners[u]
                    sel = pid == u
                    m = mats.setdefault(u, np.zeros((pn, n), np.int8))
                    m[dest[sel] - so, src[sel]] = w[sel]
            for u, m in mats.items():
                name, port, so, pn = owners[u]
                pr = Projection(p["name"], name, port, so, pn, n)
                gain = self._slot_gain[so:so + pn]
                if p["type"] == "token":
                    pr.columns = torch.from_numpy(m).to(self.device)
                    pr.col_gain = torch.from_numpy(gain / 127.0).float().to(self.device)
                elif pn == n and np.count_nonzero(m) == np.count_nonzero(np.diag(m)):
                    pr.diag = torch.from_numpy(np.diag(m) / 127.0 * gain).float().to(self.device)
                else:
                    wt = torch.from_numpy(m).to(self.device).float()
                    wt.mul_(torch.from_numpy(gain / 127.0).float().to(self.device)[:, None])
                    pr.weight = wt.to(self.dtype)
                self.out[p["name"]].append(pr)
            del mats

    def projection(self, src, dst, port):
        return next(pr for pr in self.out[src] if pr.dst == dst and pr.port == port)

    # ------------------------------------------------------------------ state
    def reset(self, batch=1):
        """Clear all recurrent state (new conversation)."""
        self.batch, self.pos = batch, 0
        self.state = {}
        dev = self.device
        for p in self.pops:
            P = self.params[p["name"]]
            if p["type"] == "gdn":
                h, dk, dv = P["heads"], P["dk"], P["dv"]
                self.state[p["name"]] = {
                    "conv": torch.zeros(batch, P["conv_w"].shape[0], P["conv"] - 1, device=dev),
                    "S": torch.zeros(batch, h, dk, dv, device=dev)}
            elif p["type"] == "attention":
                self.state[p["name"]] = {"k": None, "v": None}
            if p["type"] in SPIKING and self.mode == "spike":
                self.state.setdefault(p["name"], {})["mem"] = torch.zeros(batch, p["n"], device=dev)

    # ------------------------------------------------------------------ neurons
    @staticmethod
    def _rms(x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def _drive(self, p, inp):
        """What each neuron of population p computes from its dendrites this window."""
        t, P, B = p["type"], self.params[p["name"]], self.batch
        if t in ("integrator", "readout"):
            return sum(inp.values())
        if t == "rmsnorm":
            return self._rms(inp["x"], P["eps"]) * P["gain"]
        if t == "swiglu":
            return F.silu(inp["gate"]) * inp["up"]
        if t == "gdn":
            return self._gdn(p, P, inp)
        if t == "attention":
            return self._attention(p, P, inp)
        raise ValueError(t)

    def _gdn(self, p, P, inp):
        st, B = self.state[p["name"]], self.batch
        h, dk, dv = P["heads"], P["dk"], P["dv"]
        # 4-tap delay line on every qkv dendrite: out = sum_k w_k * x[t-3+k]
        win = torch.cat([st["conv"], inp["qkv"][..., None]], dim=-1)
        st["conv"] = win[..., 1:]
        x = F.silu((win * P["conv_w"]).sum(-1))
        q, k, v = torch.split(x, [h * dk, h * dk, h * dv], dim=-1)
        q = q.view(B, h, dk)
        k = k.view(B, h, dk)
        v = v.view(B, h, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(inp["b"])
        decay = torch.exp(-P["A_log"].exp() * F.softplus(inp["a"] + P["dt_bias"]))
        # fast plastic synapses: leak, then delta-rule update
        S = st["S"] * decay[..., None, None]
        delta = (v - torch.einsum("bhkv,bhk->bhv", S, k)) * beta[..., None]
        S = S + k[..., :, None] * delta[..., None, :]
        st["S"] = S
        o = torch.einsum("bhkv,bhk->bhv", S, q)
        o = self._rms(o, P["eps"]) * P["norm_w"] * F.silu(inp["z"].view(B, h, dv))
        return o.reshape(B, h * dv)

    def _attention(self, p, P, inp):
        st, B = self.state[p["name"]], self.batch
        nh, nkv, hd, rot = P["heads"], P["kv_heads"], P["head_dim"], P["rotary_dim"]
        qg = inp["q"].view(B, nh, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(B, nh * hd)
        q = self._rms(q, P["eps"]) * P["q_gain"]
        k = self._rms(inp["k"].view(B, nkv, hd), P["eps"]) * P["k_gain"]
        v = inp["v"].view(B, nkv, hd)
        inv = 1.0 / (P["rope_theta"] ** (torch.arange(0, rot, 2, device=q.device, dtype=torch.float32) / rot))
        fr = self.pos * inv
        cos, sin = torch.cat([fr, fr]).cos(), torch.cat([fr, fr]).sin()

        def rope(x):
            r, rest = x[..., :rot], x[..., rot:]
            rh = torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], dim=-1)
            return torch.cat([r * cos + rh * sin, rest], dim=-1)

        q, k = rope(q), rope(k)
        W = P["window"]
        if st["k"] is None or (st["k"].shape[2] <= self.pos % W and st["k"].shape[2] < W):
            cap = min(W, max(256, 2 * (0 if st["k"] is None else st["k"].shape[2])))
            nk = torch.zeros(B, nkv, cap, hd, device=q.device)
            nv = torch.zeros_like(nk)
            if st["k"] is not None:
                nk[:, :, :st["k"].shape[2]] = st["k"]
                nv[:, :, :st["v"].shape[2]] = st["v"]
            st["k"], st["v"] = nk, nv
        slot = self.pos % W
        st["k"][:, :, slot] = k
        st["v"][:, :, slot] = v
        n = min(self.pos + 1, W)
        K = st["k"][:, :, :n].repeat_interleave(nh // nkv, dim=1)
        Vv = st["v"][:, :, :n].repeat_interleave(nh // nkv, dim=1)
        a = torch.softmax(torch.einsum("bhd,bhnd->bhn", q, K) / math.sqrt(hd), dim=-1)
        o = torch.einsum("bhn,bhnd->bhd", a, Vv).reshape(B, nh * hd)
        return o * torch.sigmoid(gate)

    def _fire(self, p, drive):
        """Spiking neurons turn their drive into output payload (what synapses carry)."""
        P = self.params[p["name"]]
        if self.mode == "graded":
            self._count(p, (drive != 0).sum().item())
            return drive
        T = self.T
        r = P["refractory"]
        nmax = torch.ceil(T / (r + 1).float())
        theta = (P["range"] / nmax).clamp_min(1e-12)
        leak = P["leak"]
        st = self.state.setdefault(p["name"], {})
        v = st["mem"] if self.carry_membrane and "mem" in st else torch.zeros_like(drive)
        ref = torch.zeros_like(drive, dtype=torch.long)
        count = torch.zeros_like(drive)
        current = drive / T
        for _ in range(T):
            v = leak * v + current
            ok = ref <= 0
            # number of thresholds crossed this step: 1 normally, more = a burst
            s = torch.sign(v) * torch.floor(v.abs() / theta + (1.0 - self.fire_level))
            if self.max_burst:
                s = s.clamp(-self.max_burst, self.max_burst)
            s = s * ok
            v = v - s * theta
            count += s
            ref = torch.where(s != 0, r.expand_as(ref), ref - 1)
        st["mem"] = v * leak
        self._count(p, count.abs().sum().item())
        return count * theta

    def _count(self, p, spikes):
        fan = int(self._offsets[p["neuron_offset"] + 1] - self._offsets[p["neuron_offset"]])
        s = self.stats.setdefault(p["type"], [0, 0])
        s[0] += spikes
        s[1] += spikes * fan

    # ------------------------------------------------------------------ one token
    @torch.no_grad()
    def step(self, tokens, record=None, observe=None):
        """Feed one token per batch row; returns readout membranes (logits) [B, vocab].

        record(pop, drive) is called for spiking populations before they fire,
        observe(pop, drive, payload) for every non-sensory population after."""
        tokens = torch.as_tensor(tokens, device=self.device).view(-1)
        assert tokens.numel() == self.batch
        B = self.batch
        buf = {}
        out = None
        for p in self.pops:
            name, t = p["name"], p["type"]
            if t == "token":
                for pr in self.out[name]:
                    amp = torch.from_numpy(self.range[p["neuron_offset"] + tokens.cpu().numpy()]).to(self.device)
                    cur = pr.columns[:, tokens].T.float() * pr.col_gain * amp[:, None]
                    buf[(pr.dst, pr.port)] = buf.get((pr.dst, pr.port), 0) + cur
                self._count(p, B)
                continue
            inp = {q["name"]: buf.pop((name, q["name"]), None) for q in p["ports"]}
            for k, val in inp.items():
                if val is None:
                    inp[k] = torch.zeros(B, next(q["n"] for q in p["ports"] if q["name"] == k), device=self.device)
            drive = self._drive(p, inp)
            if record is not None and t in SPIKING:
                record(p, drive)
            payload = self._fire(p, drive) if t in SPIKING else drive
            if observe is not None:
                observe(p, drive, payload)
            if t == "readout":
                out = payload
            for pr in self.out[name]:
                if pr.diag is not None:
                    cur = payload * pr.diag
                else:
                    cur = (payload.to(self.dtype) @ pr.weight.T).float()
                key = (pr.dst, pr.port)
                buf[key] = buf[key] + cur if key in buf else cur
        self.pos += 1
        return out

    # ------------------------------------------------------------------ text
    def tokenizer(self, path=None):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path or self.header["tokenizer_dir"])

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=64, temperature=0.0, top_p=0.9, stream=None):
        """ids: prompt token ids. Runs the prompt through the network token by token, then samples."""
        self.reset(1)
        logits = None
        for t in ids:
            logits = self.step([t])
        out = []
        eos = set(self.header.get("eos_token_ids", []))
        for _ in range(max_new_tokens):
            if temperature <= 0:
                nxt = int(logits[0].argmax())
            else:
                pr = torch.softmax(logits[0].float() / temperature, -1)
                sp, si = pr.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p
                nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])
            if nxt in eos:
                break
            out.append(nxt)
            if stream:
                stream(nxt)
            logits = self.step([nxt])
        return out


# ---------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(path, ids, quantile=0.999, streams=8, verbose=True):
    """Run the network (graded mode) on calibration tokens and store every
    spiking neuron's drive range (a high quantile of |drive|) in the file."""
    t0 = time.time()
    eng = Engine(path, mode="graded", verbose=verbose)
    L = len(ids) // streams
    seqs = torch.tensor(ids[:L * streams]).view(streams, L)
    eng.reset(streams)
    rec = {}

    def record(p, drive):
        rec.setdefault(p["name"], []).append(drive.abs().bfloat16().cpu())

    for t in range(L):
        eng.step(seqs[:, t], record=record)
        if verbose and (t + 1) % 64 == 0:
            print(f"  calibration {t + 1}/{L} steps x {streams} streams ({time.time() - t0:.0f}s)", flush=True)
    h = eng.header
    rng = fmt.open_array(path, h, "neuron.range", mode="r+")
    for p in eng.pops:
        if p["name"] not in rec:
            continue
        a = torch.cat(rec[p["name"]]).float()          # [tokens, n]
        k = max(1, int(round(quantile * a.shape[0])))
        r = a.kthvalue(k, dim=0).values
        r = torch.where(r > 0, r, a.max(0).values.clamp_min(1e-6))
        rng[p["neuron_offset"]:p["neuron_offset"] + p["n"]] = r.numpy()
    rng.flush()
    h["calibrated"] = True
    h["calibration"] = {"tokens": int(L * streams), "quantile": quantile}
    fmt.update_header(path, h)
    if verbose:
        print(f"calibrated {sum(len(v) for v in rec.values()) // len(rec)} steps in {time.time() - t0:.0f}s", flush=True)
