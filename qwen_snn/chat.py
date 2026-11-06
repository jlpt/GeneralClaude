"""Talk to the spiking network.

    python chat.py --snn qwen3.5-0.8b.snn                      # interactive chat
    python chat.py --snn qwen3.5-0.8b.snn --prompt "Hi!"       # one answer
    python chat.py --snn qwen3.5-0.8b.snn --mode spike --T 32  # binary LIF spikes

--mode graded  each spiking neuron fires once per token and pushes its whole
               activation (sourceAct * weight, like kernel.cu)
--mode spike   binary +-1 spikes from leaky integrate-and-fire neurons over T
               timesteps per token
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snn", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer directory (default: the one recorded in the .snn file)")
    ap.add_argument("--mode", choices=["graded", "spike"], default="spike")
    ap.add_argument("--T", type=int, default=32, help="timesteps per token (spike mode)")
    ap.add_argument("--leak", type=float, default=None, help="override membrane leak (1.0 = no leak)")
    ap.add_argument("--refractory", type=int, default=None, help="override refractory timesteps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float16 if args.device.startswith("cuda") else torch.float32)

    eng = Engine(args.snn, device=args.device, dtype=dtype, mode=args.mode, T=args.T,
                 leak=args.leak, refractory=args.refractory)
    if args.mode == "spike" and not eng.header.get("calibrated"):
        print("warning: this .snn file has no calibrated thresholds; spike mode will be poor")
    tok = eng.tokenizer(args.tokenizer)
    history = []

    def answer(text):
        history.append({"role": "user", "content": text})
        ids = tok.apply_chat_template(history, add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True, return_dict=True)["input_ids"]
        eng.stats = {}
        t0 = time.time()
        printed = []

        def stream(t):
            printed.append(t)
            sys.stdout.write(tok.decode(printed)[len(tok.decode(printed[:-1])):])
            sys.stdout.flush()

        out = eng.generate(ids, args.max_new_tokens, args.temperature, stream=stream)
        n = len(ids) + len(out)
        spikes = sum(v[0] for k, v in eng.stats.items() if k != "token") / max(n, 1)
        events = sum(v[1] for v in eng.stats.values()) / max(n, 1)
        print(f"\n[{len(out)} tokens, {time.time() - t0:.1f}s, {spikes:,.0f} spikes/token, "
              f"{events:,.0f} synaptic events/token]")
        history.append({"role": "assistant", "content": tok.decode(out)})

    if args.prompt:
        answer(args.prompt)
        return
    print("type a message (empty line to quit)")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        answer(text)


if __name__ == "__main__":
    main()
