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

## Caveats

- The model is tiny, the dataset is tiny, and each config has only 2–3 seeds.
  Differences under about 0.007 are within noise.
- No hyperparameter retuning was done per config. The learning rate was fixed at 3e-3.
- MoE runs used a simple loop over experts, so their wall-clock times are pessimistic.
