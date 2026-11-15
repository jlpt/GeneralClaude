# Transformer lab: growing a transformer one part at a time

Start from a 2017-style transformer, then add one component at a time.
A component stays only if it lowers validation loss by more than the
run-to-run noise.

## Setup

- **Task:** character-level language modeling on Tiny Shakespeare
  (1.1M characters, 90/10 train/validation split, 65-character vocabulary).
- **Model size:** 4 layers, width 128, 4 heads, context 128, about 0.8M parameters.
- **Training:** 2,000 steps × 32 sequences, AdamW, cosine learning-rate schedule, 4-core CPU.
- **Metric:** validation cross-entropy on the same 100 held-out batches for every run
  (lower is better). Each config was run with 2–3 seeds; noise is about ±0.005.

Reproduce a run with `python lab.py '{"name":"x", ...overrides}' seed`.
Reproduce a round with `python sweep.py sweeps/roundN.json 0,1`.
Print results with `python summarize.py rN`. All runs are in `results.jsonl`.

## Results

### Round 1: modernize the 2017 design (cumulative)

| Config | Val loss | Verdict |
|---|---|---|
| 2017 baseline: post-LayerNorm, sinusoidal positions, ReLU | 1.542 ± 0.007 | |
| + pre-norm | 1.549 ± 0.007 | neutral (matters for deep nets, not 4 layers) |
| + RMSNorm | 1.543 ± 0.006 | neutral, cheaper |
| + RoPE | 1.524 ± 0.007 | **keep** |
| + SwiGLU | 1.516 ± 0.004 | **keep** |
| + QK-norm | 1.507 ± 0.005 | **keep** |
| + tied embeddings | 1.505 ± 0.003 | keep (neutral, fewer params) |

### Round 2: each candidate tested alone on the round-1 base (1.505)

| Candidate | Val loss | Verdict |
|---|---|---|
| Mixture-of-Experts (4 experts, top-2, same active params) | 1.486 | win |
| Per-head attention output gate | 1.493 | win |
| Multi-token prediction (1 extra head, training only) | 1.496 | win |
| Grouped-query attention (2 KV heads) | 1.496 | win, 8% fewer params |
| 4 register tokens | 1.506 | neutral |
| Dropout 0.1 | 1.506 | neutral |
| Value residual | 1.512 | worse |
| Short causal depthwise conv (k=4) | 1.537 | clearly worse |

### Round 3: stack the winners

| Config | Val loss | Note |
|---|---|---|
| base + gate + MoE | 1.504 | gains did not add |
| base + gate + MoE + MTP | **1.482** | |
| … + GQA | 1.489 | no longer helps |
| … + 8 experts | 1.494 | more capacity → more overfitting |
| … + recurrent depth ×2 | 1.478 | 2× compute |

### Round 4: probe the plateau

| Config | Val loss | Compute |
|---|---|---|
| base + gate + MoE + MTP (from round 3) | 1.482 | 1× |
| … + dropout 0.1 | 1.494 | 1× |
| … with 2 MTP heads | 1.483 | 1× |
| **… without MoE (dense)** | **1.483** | **1×, fastest** |
| … + recurrent depth ×2 | 1.482 | 2× |
| … with 8 real layers (control) | 1.487 | 2×, 2× params |

## Final architecture

**Dense decoder-only transformer.** About 823k parameters; the extra prediction head is used only during training.

```
tokens → embedding (tied with output) ──┐
                                        ▼
  ┌──────────── block × 4 ────────────────────────┐
  │ RMSNorm → attention                           │
  │     Q,K: RMSNorm per head (QK-norm) + RoPE    │
  │     output × sigmoid gate (per head)          │
  │ + residual                                    │
  │ RMSNorm → SwiGLU feed-forward                 │
  │ + residual                                    │
  └───────────────────────────────────────────────┘
                                        ▼
  RMSNorm → output head (next token)
          → extra head (token after next; training-only auxiliary loss)
```

**Validation loss 1.542 → 1.483 (−3.9%; 2.225 → 2.139 bits per character)** with the
same parameter count and training budget.

## What we learned

1. **The 2017 design can be improved cheaply.** RoPE, SwiGLU, QK-norm, an attention
   output gate and multi-token prediction together give about a 4% loss reduction for free.
2. **Gains don't add up.** MoE and the gate each helped alone and did nothing together.
   Each component has to be tested again in combination.
3. **At this scale, data is the wall.** Train loss falls well below validation loss.
   Everything that added capacity hit the same plateau of about 1.48: more experts,
   8 layers, looping the stack. That held whether the capacity came from more parameters
   or more compute. MoE dropped out of the final design for the same reason.
4. **Some popular components didn't help here:** pre-norm, value residual,
   short convolutions and register tokens. Results like these are scale-dependent.
   Pre-norm and value residual are known to matter mostly in much deeper networks.

## Part 2: letting the architecture evolve itself (`evolve.py`)

### How it works

**Genome.** One block is a wiring diagram:
- **Nodes ("neurons")** are math functions: `linear`, `norm`, `relu`, `gelu`, `silu`,
  `sigmoid`, `tanh`, `square`, `neg`, `shift` (previous token), `cummean`
  (running average of the past), `add`, `mul`, `max`, and `mix`.
  `mix` is causal softmax attention as a 3-input function: `softmax(rope(a)·rope(b)ᵀ)·c`.
- **Edges ("synapses")** say which earlier node feeds each node.

The model stacks the block with a residual, `x ← x + block(x)`, and each copy has its own weights.

**Mutations.** A child can get:
- a node's function changed;
- an input rewired;
- a node inserted (and spliced into the live path);
- a node deleted;
- a different output node.

**Search.** Regularized (aging) evolution:
- population 24, tournament of 6;
- the oldest member dies when a child is added;
- 4 candidates are evaluated in parallel.

**Fitness.** Validation loss after a 14-second training run on the "proxy" setup:
2 layers, width 64, context 64, 400 steps.

### Runs

| Run | Evaluations | Best proxy loss | Hand-designed transformer block |
|---|---|---|---|
| From random graphs | 400 | 2.122 | 2.026 |
| From a transformer block | 300 | 1.995 | 2.026 |

**Evolved from scratch.** It did not find attention.
It found a *token-shift gate*, similar to the token-shift idea in RWKV-style models:
```
block(x) = linear(x) * shift(norm(linear(x)))
```

**Evolved from the transformer block (#1).** Key changes:
- the feed-forward network is removed;
- the value input is `norm(x) + relu(norm(x))`, with no value projection;
- the attention output is gated by the query;
- a parallel linear branch is added.

```
n1 = norm(x);  n2 = relu(n1)
att = mix(linear(n2), linear(n2), n1 + n2)
block(x) = linear(n2) + linear(silu(tanh(att * linear(n2))))
```

**Evolved #2.** Standard attention, but the feed-forward network lost its up-projection:
`ffn(h) = linear(relu(silu(norm(h))))`.

### Re-validation on fresh seeds

| Design | Proxy (3 seeds) | Full size: 4 layers, d=128, 2000 steps (2 seeds) | Params at full size | Time |
|---|---|---|---|---|
| Transformer block (hand-designed) | 2.024 ± 0.015 | 1.555 ± 0.008 | 403k | 310s |
| Evolved from transformer #1 | 1.998 ± 0.002 | 1.554 ± 0.003 | **271k (−33%)** | 274s |
| Evolved from transformer #2 | 1.993 ± 0.004 | 1.553 ± 0.000 | 337k (−16%) | 286s |
| Evolved from scratch | 2.109 ± 0.011 | 1.631 ± 0.006 | 140k (−65%) | 83s |

### What this showed

1. **Evolution beat the hand design on the task it was trained on (the proxy).**
   Most of that edge disappeared at full size. Both evolved-from-transformer blocks
   *tie* the original at full size, but with 16–33% fewer parameters.
   This is the classic proxy gap: designs tuned for a small, short run don't
   automatically win in a bigger, longer one.
2. **From scratch, 400 tries were not enough to assemble attention.**
   Attention needs three separate projections wired into `mix`, which is a lot of
   coordinated mutation. Instead, evolution found a cheap local-mixing trick that is
   3.7× faster but 0.08 worse.
3. **Evolution's favorite move was deletion.** It mostly kept quality while
   removing parts. Given a fixed budget, that is a sensible way to improve fitness.

The graph language uses width-d linears only: no 4× feed-forward expansion and no
SwiGLU. That is why these full-size numbers (about 1.55) are worse than Part 1's best (1.483).

Reproduce with:
- `python evolve.py scratch 400`
- `python evolve.py seeded 300`
- `python validate_evolved.py proxy|full`

## Caveats

- The model is tiny, the dataset is tiny, and each config has only 2–3 seeds.
  Differences under about 0.007 are within noise.
- No hyperparameter retuning was done per config. The learning rate was fixed at 3e-3.
- MoE runs used a simple loop over experts, so their wall-clock times are pessimistic.
