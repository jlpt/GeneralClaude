"""3D viewer for the spiking network: fly around the neurons, click one to see
its synapses, type a prompt and watch the spikes.

    python viewer.py --snn qwen3.5-0.8b.snn            # then open http://127.0.0.1:8765

Browsing reads the .snn file directly (neuron parameters and packed synapses,
decoded on demand); the engine is loaded the first time a prompt is run.
"""
import argparse
import base64
import bisect
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snn_format as fmt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKING = {"rmsnorm", "swiglu", "gdn", "attention"}


class Network:
    """Read-only view of a .snn file for browsing."""

    def __init__(self, path):
        self.path = path
        self.h = h = fmt.read_header(path)
        self.pops = h["populations"]
        self.pop = {p["name"]: p for p in self.pops}
        self.starts = [p["neuron_offset"] for p in self.pops]
        arr = lambda n: fmt.open_array(path, h, n)
        self.off, self.packed = arr("syn.offsets"), arr("syn.packed")
        self.gain = np.array(arr("slot.gain"))
        self.range = np.array(arr("neuron.range"))
        self.leak = np.array(arr("neuron.leak"))
        self.refr = np.array(arr("neuron.refractory"))
        self.ports = []  # (slot_offset, n, pop, port)
        for p in self.pops:
            for q in p["ports"]:
                self.ports.append((q["slot_offset"], q["n"], p, q))
        self.port_starts = [x[0] for x in self.ports]
        self._sources = None
        self._tok = None

    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.h["tokenizer_dir"])
        return self._tok

    def find(self, gid):
        p = self.pops[bisect.bisect_right(self.starts, gid) - 1]
        return p, gid - p["neuron_offset"]

    def slot_owner(self, slot):
        so, n, p, q = self.ports[bisect.bisect_right(self.port_starts, slot) - 1]
        return p, q, slot - so

    # which neuron of the target population a dendrite slot belongs to (for drawing)
    def slot_neuron(self, p, q, j):
        t, P = p["type"], p["params"]
        if t == "gdn":
            hd = P["dv"]
            if q["name"] == "qkv":
                return (j % (P["heads"] * P["dk"]))
            if q["name"] in ("b", "a"):
                return j * hd + hd // 2
            return j
        if t == "attention":
            hd = P["head_dim"]
            if q["name"] == "q":
                return (j // (2 * hd)) * hd + (j % hd)
            g = j // hd
            return g * (P["heads"] // P["kv_heads"]) * hd + j % hd
        return j

    def neuron_slots(self, p, i):
        """Dendrite slots that feed neuron i of population p, with a label each."""
        t, P, out = p["type"], p["params"], []
        port = {q["name"]: q for q in p["ports"]}
        if t == "gdn":
            hd, nh = P["dv"], P["heads"]
            h, d = divmod(i, hd)
            for k, part in enumerate("qkv"):
                out.append((port["qkv"]["slot_offset"] + k * nh * hd + i, f"{part} (head {h}, dim {d})"))
            out.append((port["z"]["slot_offset"] + i, f"z gate (head {h}, dim {d})"))
            if d == hd // 2:
                out.append((port["b"]["slot_offset"] + h, f"beta (head {h})"))
                out.append((port["a"]["slot_offset"] + h, f"decay (head {h})"))
        elif t == "attention":
            hd, rep = P["head_dim"], P["heads"] // P["kv_heads"]
            h, d = divmod(i, hd)
            out.append((port["q"]["slot_offset"] + h * 2 * hd + d, f"query (head {h}, dim {d})"))
            out.append((port["q"]["slot_offset"] + h * 2 * hd + hd + d, f"output gate (head {h}, dim {d})"))
            if h % rep == 0:
                out.append((port["k"]["slot_offset"] + (h // rep) * hd + d, f"key (kv head {h // rep}, dim {d})"))
                out.append((port["v"]["slot_offset"] + (h // rep) * hd + d, f"value (kv head {h // rep}, dim {d})"))
        else:
            for q in p["ports"]:
                out.append((q["slot_offset"] + i, q["name"]))
        return out

    def sources(self):
        """port slot_offset -> [(source pop, kind, column)], read off each population's first two neurons.

        dense:    every source neuron reaches slot j of the port at row position column + j
        identity: source neuron j reaches slot j, at row position column"""
        if self._sources is None:
            src = {}
            for p in self.pops:
                a = p["neuron_offset"]
                if self.off[a] == self.off[a + 1]:
                    continue
                d0, _ = fmt.unpack(self.packed[self.off[a]:self.off[a + 1]])
                d1, _ = fmt.unpack(self.packed[self.off[a + 1]:self.off[a + 2]])
                ident = len(d1) == len(d0) and np.array_equal(d1, d0 + 1)
                seen = set()
                for col, d in enumerate(d0):
                    tp, tq, j = self.slot_owner(int(d))
                    if tq["slot_offset"] in seen:
                        continue
                    seen.add(tq["slot_offset"])
                    src.setdefault(tq["slot_offset"], []).append(
                        (p, "identity", col) if ident else (p, "dense", col - j))
            self._sources = src
        return self._sources

    def neuron(self, gid, k=24):
        p, i = self.find(gid)
        T = 32
        info = {"gid": gid, "pop": p["name"], "type": p["type"], "index": i, "n": p["n"]}
        if p["type"] in SPIKING:
            info.update(range=float(self.range[gid]), threshold_T32=float(self.range[gid] / T),
                        leak=float(self.leak[gid]), refractory=int(self.refr[gid]))
        if p["type"] == "token":
            info["token"] = self.tokenizer().decode([i])
            info["spike_amplitude"] = float(self.range[gid])
        if p["type"] == "readout":
            info["token"] = self.tokenizer().decode([i])
        a, b = int(self.off[gid]), int(self.off[gid + 1])
        info["fan_out"] = b - a
        outs = []
        if b > a:
            dest, w = fmt.unpack(self.packed[a:b])
            real = w.astype(np.float32) / 127.0 * self.gain[dest]
            top = np.argsort(-np.abs(real))[:k]
            for t in top:
                tp, tq, j = self.slot_owner(int(dest[t]))
                outs.append({"slot": int(dest[t]), "target": tp["neuron_offset"] + self.slot_neuron(tp, tq, j),
                             "label": f"{tp['name']}.{tq['name']}[{j}]", "int8": int(w[t]), "weight": float(real[t])})
        info["out"] = outs
        ins, fan_in = [], 0
        srcs = self.sources()
        for slot, label in self.neuron_slots(p, i):
            tp, tq, j = self.slot_owner(slot)
            for sp, kind, col in srcs.get(tq["slot_offset"], []):
                s0 = sp["neuron_offset"]
                if kind == "identity":
                    src_ids = np.array([s0 + j])
                    pos = np.asarray(self.off[src_ids]) + col
                else:
                    src_ids = np.arange(s0, s0 + sp["n"])
                    pos = np.asarray(self.off[s0:s0 + sp["n"]]) + col + j
                _, w = fmt.unpack(self.packed[pos])
                fan_in += len(src_ids)
                real = w.astype(np.float32) / 127.0 * self.gain[slot]
                for t in np.argsort(-np.abs(real))[:max(4, k // 3)]:
                    ins.append({"source": int(src_ids[t]), "label": f"{sp['name']}[{int(src_ids[t]) - s0}] -> {label}",
                                "int8": int(w[t]), "weight": float(real[t])})
        ins.sort(key=lambda e: -abs(e["weight"]))
        info["in"] = ins[:k]
        info["fan_in"] = fan_in
        return info


class Runner:
    """Runs prompts through the engine in a thread and keeps per-token activity frames."""

    def __init__(self, net, device):
        self.net, self.device = net, device
        self.engine = None
        self.lock = threading.Lock()
        self.runs = {}
        p0 = net.pop["resid.0"]["neuron_offset"]
        self.lo, self.hi = p0, net.pop["vocab"]["neuron_offset"]

    def start(self, prompt, mode, T, max_new, chat):
        rid = uuid.uuid4().hex[:8]
        run = self.runs[rid] = {"frames": [], "done": False, "status": "loading engine", "error": None}
        threading.Thread(target=self._run, args=(run, prompt, mode, T, max_new, chat), daemon=True).start()
        return rid

    def _run(self, run, prompt, mode, T, max_new, chat):
        import torch
        try:
            with self.lock:
                if self.engine is None:
                    from engine import Engine
                    self.engine = Engine(self.net.path, device=self.device, mode=mode, T=T)
                eng = self.engine
                eng.mode, eng.T = mode, T
                tok = self.net.tokenizer()
                if chat:
                    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                  enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]
                else:
                    ids = tok(prompt)["input_ids"]
                eng.reset(1)
                run["status"] = "running"
                eos = set(eng.header.get("eos_token_ids", []))
                feed, phase = list(ids), "prompt"
                n_gen = 0
                while feed:
                    t = feed.pop(0)
                    act = np.zeros(self.hi - self.lo, dtype=np.uint8)
                    spikes = [0]

                    def observe(p, drive, payload):
                        a = p["neuron_offset"] - self.lo
                        if p["type"] == "readout":
                            return
                        x = payload[0].abs().float()
                        if p["type"] in SPIKING:
                            r = eng.params[p["name"]]["range"]
                            x = x / r
                            if eng.mode == "spike":
                                nmax = torch.ceil(eng.T / (eng.params[p["name"]]["refractory"] + 1).float())
                                spikes[0] += int(torch.round(payload[0].abs() / (r / nmax)).sum())
                        else:
                            x = x / x.max().clamp_min(1e-9)
                        act[a:a + p["n"]] = (x.clamp(0, 1).sqrt() * 255).to(torch.uint8).cpu().numpy()

                    t0 = time.time()
                    logits = eng.step([t], observe=observe)[0]
                    top = torch.topk(logits, 8)
                    probs = torch.softmax(logits.float(), -1)[top.indices]
                    nxt = int(top.indices[0])
                    run["frames"].append({
                        "index": len(run["frames"]), "token": t, "text": tok.decode([t]), "phase": phase,
                        "activity": base64.b64encode(act.tobytes()).decode(), "spikes": spikes[0],
                        "seconds": round(time.time() - t0, 3),
                        "top": [{"gid": self.net.pop["vocab"]["neuron_offset"] + int(i), "text": tok.decode([int(i)]),
                                 "p": float(pv)} for i, pv in zip(top.indices, probs)]})
                    if not feed:
                        if n_gen >= max_new or nxt in eos:
                            break
                        feed.append(nxt)
                        phase = "gen"
                        n_gen += 1
                run["status"] = "done"
        except Exception as e:  # surface errors to the page
            run["error"] = repr(e)
            run["status"] = "error"
        finally:
            run["done"] = True


def make_handler(net, runner):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, open(os.path.join(HERE, "viewer", "index.html"), "rb").read(),
                                      "text/html; charset=utf-8")
                if u.path == "/api/layout":
                    return self._send(200, {
                        "populations": [{k: p[k] for k in ("name", "type", "n", "neuron_offset", "ports", "params")}
                                        for p in net.pops],
                        "n_neurons": net.h["n_neurons"], "n_synapses": net.h["n_synapses"],
                        "calibrated": net.h.get("calibrated", False),
                        "activity_range": [runner.lo, runner.hi]})
                if u.path.startswith("/api/neuron/"):
                    return self._send(200, net.neuron(int(u.path.rsplit("/", 1)[1]), int(q.get("k", 24))))
                if u.path == "/api/frames":
                    run = runner.runs.get(q.get("run"))
                    if run is None:
                        return self._send(404, {"error": "no such run"})
                    since = int(q.get("since", 0))
                    return self._send(200, {"frames": run["frames"][since:], "done": run["done"],
                                            "status": run["status"], "error": run["error"]})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": repr(e)})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/run":
                return self._send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            rid = runner.start(body.get("prompt", "Hello!"), body.get("mode", "spike"), int(body.get("T", 32)),
                               int(body.get("max_new_tokens", 24)), bool(body.get("chat", True)))
            return self._send(200, {"run": rid})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snn", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    net = Network(args.snn)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(net, Runner(net, args.device)))
    print(f"{net.h['n_neurons']:,} neurons, {net.h['n_synapses']:,} synapses")
    print(f"open http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
