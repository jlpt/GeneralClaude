# Grokking Soup: can averaging memorizers produce a generalizer?

## The question

Grokking networks (here, a 1-layer transformer learning `a + b mod 97`) first **memorize**
the training set, then thousands of steps later suddenly **generalize**. Prior work
(Nanda et al. 2023) shows the generalizing circuit forms gradually *inside* the memorizing
network, and is then "cleaned up" by weight decay.

**Hypothesis.** Memorization is idiosyncratic to the exact examples seen; the generalizing
circuit is not. So if N copies of a network share an initialization but train on
*disjoint shards* of the data, weight-averaging them ("soup") should cancel the
memorization and keep the shared circuit, short-circuiting grokking's slow cleanup phase.
Concretely, the soup should (H1) generalize even when every member only memorizes,
(H2) beat output-ensembling, and (H3) generalize *earlier* than one model trained on all the data.

Novelty check (Nov 2026): grokking + low-rank truncation is studied
([Subspace Crystallization, 2610.35112](https://arxiv.org/abs/2610.35112)); one-shot merging in
decentralized vision training is studied ([Zhu et al., ICLR 2026, 2507.06542](https://arxiv.org/abs/2507.06542)).
I found no published test of weight-averaging disjoint-shard memorizers as a route to
algorithmic generalization.

## Setup

- Task `a+b mod 97`, 1-layer causal transformer (d=64, 4 heads, MLP 256), AdamW, wd=1.0, full-batch.
- N = 4 disjoint shards of the training set; all N workers trained as one batched tensor program.
- Controls in every run: output-ensemble of the same workers, soup of *different-init* workers,
  and a centralized model trained on the union of the shards.

| file | what |
|---|---|
| `soup.py` | Exp 1: N workers from a shared random init, each on its own shard |
| `fork.py` | Exp 2: train one parent on all data; at step T_f fork into N shard workers for K steps; soup them |
| `plot*.py` | figures + summary tables |
| `results/` | raw JSON logs and figures |

## Results

### Exp 1: Soup of shard models from a shared init fails completely ✗

`results/explore_d64.png` (train frac 0.4, lr 1e-3)

| at step 8000 | test acc |
|---|---|
| centralized model on all data | **1.000** (grokked at ~2000) |
| soup of 4 shard models | 0.008 (chance = 0.010) |
| ensemble of 4 shard models | 0.006 |
| single shard model | 0.006 (train acc 1.000) |
| soup of different-init models | 0.010 |

The soup can't even fit the *training* data (train acc ≈ 0.02 at every checkpoint from step 100).
Mergeability is gone within the first 100 steps, long before memorization is complete.
**H1 rejected.**

### Exp 2: Fork-point sweep, 3 seeds, 33 forks ✗ (with one nuance)

`results/grok_fork_aligned.png`, `results/grok_fork.png` (train frac 0.3, lr 3e-3, K = 1000).
The parents grok at steps 3500, 1900, and not by 6000 for the three seeds, so the forks are grouped by
the parent's test accuracy at the fork:

| parent phase at fork | n | parent@fork | parent after K more steps | **soup** | ensemble | single fork |
|---|---|---|---|---|---|---|
| pre-grok (<0.3) | 22 | 0.097 | 0.240 | 0.051 | 0.065 | 0.045 |
| transition (0.3–0.95) | 2 | 0.384 | 1.000 | 0.527 | 0.542 | 0.330 |
| grokked (≥0.95) | 9 | 0.998 | 1.000 | 0.901 | 0.993 | 0.902 |

- **Soup ≥ centralized training: 0 / 33 forks.** Splitting the data and averaging never beat simply
  continuing training on all of it. **H3 rejected.**
- **Ensemble > soup: 26 / 33.** Averaging outputs is almost always better than averaging weights. **H2 rejected.**
- Pre-grok, the soup is *worse than the parent it was forked from* in 21 / 22 cases: shard training
  destroys the partial circuit faster than averaging recovers it.
- **The one nuance:** in the brief transition window (2 forks, both seed 0), the soup beat both its
  members (0.53 vs 0.33) and the parent at the fork (0.38). Each fork alone got worse, yet their
  average got better. That fits "the circuit is shared, memorization isn't", but n = 2 and
  the parent trained on all data reached 1.000 over the same K steps. It's suggestive, not a result.

### Side observation: grokked solutions aren't stable under subset training

All 9 forks of an already-grokked parent (test ≥ 0.99), trained on a 7.5% shard that is perfectly
consistent with the true rule, lost generalization steadily while staying at 100% train accuracy:
e.g. 1.000 → 0.989 → 0.947 → 0.881 → 0.849 over 1000 steps. Mean after K: 0.90. With wd = 1.0 and
fewer examples, the optimizer drifts back toward a memorizing solution. This is related to reported
"anti-grokking"; I haven't checked whether this exact form is known.

## Verdict

No breakthrough: the main hypothesis is cleanly falsified. In this setting, the generalizing circuit can't be pulled out of
memorizing networks by weight averaging. Networks trained on different data diverge out
of a shared linear basin within ~100 steps, and once grokked, they "un-grok" when the data
shrinks. Grokking's slow cleanup phase is not something you can skip by averaging.

**Limitations:** one task (modular addition), one small architecture, N = 4, K = 1000, 3 seeds, no sweep
over N/K/learning rate. Smaller K (closer to linear-regime local SGD) would very likely merge better,
but that converges toward ordinary FedAvg, which is known to work.

## Reproduce

```bash
pip install torch matplotlib numpy
python3 soup.py --d 64 --dmlp 256 --steps 8000 --out results/explore_d64.json
for s in 0 1 2; do python3 fork.py --frac 0.3 --lr 3e-3 --parent_steps 6000 --k 1000 \
  --forks 0,250,500,1000,1500,2000,2500,3000,3500,4000,5000 --seed $s --out results/grok_fork_s$s.json; done
python3 plot_fork_aligned.py results/grok_fork_aligned.png results/grok_fork_s{0,1,2}.json
```
Runtime: ~11 min (Exp 1) + ~4 min per seed (Exp 2) on 4 CPU cores.
