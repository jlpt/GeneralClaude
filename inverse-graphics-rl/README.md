# Inverse-graphics RL prototype

Can a model learn to build 3D models from a few photos when you score it on angles it never saw?
This folder is a small, CPU-only test of that idea:

- A hidden **target** scene is made of spheres, boxes and cylinders.
- The **policy** sees 3 photos and writes a **scene program** (JSON list of primitives).
- The program is rendered from **held-out cameras** (new random angles each episode) and compared
  with the target. That comparison is the reward.

Everything uses only `numpy` and `Pillow`, except `llm_attempt.py`, which also needs the `anthropic` package.

## Files

| File | What it does |
|---|---|
| `ig/render.py` | Numpy ray caster: scene program + camera -> RGB, silhouette mask, depth |
| `ig/scenes.py` | Validates programs; generates random targets at 3 difficulty levels (the endless dataset) |
| `ig/reward.py` | Per-view score = silhouette IoU x (0.5 + 0.3 depth accuracy + 0.2 color accuracy), 0..1 |
| `ig/episode.py` | Episode = target + input cameras + held-out cameras; `score()` returns input-view and held-out scores |
| `demo_cheat.py` | Shows why held-out views are needed: a flat disc beats an honest guess on the input view, then fails |
| `train_toy_grpo.py` | A tiny policy trained with GRPO or rejection sampling (RFT) on held-out reward; runs in a few minutes |
| `llm_attempt.py` | Claude attempts targets (best-of-N + multi-turn refinement) and logs prompts, programs and rewards as JSONL |

```bash
python3 demo_cheat.py
python3 train_toy_grpo.py --algo grpo --steps 600 --eval-every 50
python3 train_toy_grpo.py --algo rft  --steps 600 --eval-every 50
python3 llm_attempt.py --dry-run                                   # pipeline check, no API call
python3 llm_attempt.py --episodes 5 --samples 4 --refine 2 --level 2   # needs an Anthropic API key
```

## How the model gets better

The reward tells you how good one attempt is. Training turns many scored attempts into weight updates.
Every method below repeats the same loop:

1. Sample a target and render its input photos.
2. Have the current model write **several** programs for the same target (a "group", e.g. 8).
3. Render each program from the held-out cameras and score it.
4. Update the model so that higher-scoring programs become more likely.

The methods differ in step 4.

### 1. Rejection-sampling fine-tuning (RFT, "best-of-N then imitate")

Keep only the best program in each group, if it clears a quality bar, and fine-tune on it as if it
were a human-written answer. Repeat with the improved model. It is the simplest method, it is stable,
and it uses ordinary supervised fine-tuning. It ignores the failed attempts, so it learns slowly
when good samples are rare.

### 2. GRPO (group-relative policy optimization)

For each program, compute advantage = (its reward - group mean) / group std. Raise the probability
of every token in above-average programs and lower it for below-average ones. No separate value
model is needed, because the group is the baseline. DeepSeek-style reasoning models are trained this
way on math and code. The bad attempts teach as well: "putting the sphere there made it worse."

### 3. Multi-turn refinement (agentic RL)

Let the model see renders of its own program **from the input cameras only**, next to the target
photos, and revise. Score the final turn on the held-out views. The model learns a skill (compare,
diagnose, edit) instead of having to get everything right in one shot. Never show held-out renders
during refinement, or they stop being a test.

### Making it work in practice

- **Curriculum:** start with single primitives (`--level 1`), move to compositions, then to real
  meshes (Objaverse/ShapeNet) rendered the same way. If nearly all samples in a group get the same
  reward, there is no learning signal, so keep tasks at the edge of what the model can do.
- **Cold start:** before RL, do a little supervised fine-tuning on (photos -> known program) pairs,
  which are free to make by generating random programs and rendering them. The model then starts
  out writing valid programs.
- **Shaped reward early, strict reward late:** silhouette IoU gives partial credit; add depth, color
  and perceptual metrics as the model improves. In training you know the true shape, so you can also
  use 3D metrics such as Chamfer distance. With real photos you only have images.
- **Guard against reward hacking:** use random held-out angles, include top and low views, and
  check the outputs by eye from time to time. `demo_cheat.py` shows the failure this prevents.
- **Watch the train/test gap:** log the input-view score and the held-out score separately. When
  the input score rises while the held-out score stalls, the model is fitting the photos instead of
  the object.

### Which model can you actually train?

Claude's weights can't be fine-tuned through the public API, so in this setup Claude works as a
**strong data generator and baseline**. `llm_attempt.py` produces exactly the rows that RFT and GRPO
consume. For real weight updates, take an open-weights vision-language model and an RL library with
GRPO support (for example Hugging Face TRL). The only project-specific part is the reward function,
which is `Episode.score(scene)["heldout"]`. A sketch, untested here (no GPU in this sandbox):

```python
from ig.episode import make_episode
from llm_attempt import parse_scene

def reward_fn(completions, episode_seed, **kwargs):      # one entry per sampled completion
    rewards = []
    for text, seed in zip(completions, episode_seed):
        scene = parse_scene(text)
        rewards.append(0.0 if scene is None else make_episode(seed, level=2).score(scene)["heldout"])
    return rewards
```

## Toy results

`train_toy_grpo.py` replaces the LLM with a linear softmax policy over 5 "tokens" (shape, x, z,
size, color) so the full loop runs on a CPU. It sees only pixels and learns only from the
held-out reward; it is never shown the answer. Test targets and angles are disjoint from training.

Results from one 600-step run of each method (16 targets x 8 samples per step, ~3.5 min on 4 CPU cores):

| | Held-out reward on unseen test targets | Exact match |
|---|---|---|
| Random guessing | 0.08 | 0% |
| RFT (imitate best of 8) | 0.48 | 9% |
| **GRPO** | **0.71** | **27%** |

GRPO learns faster because every sample carries signal, including the bad ones. RFT throws away 7 of
8 samples. `out/toy_grpo_examples.png` shows unseen targets (left) next to the policy's
reconstructions (right) from held-out angles. Positions, sizes and colors come out right; the
remaining errors are mostly the shape type (box vs sphere vs cylinder look alike in 24-pixel photos).
`demo_cheat.py` output:

```
candidate                             score on input view  score on held-out views
exact answer (sphere r=0.40)                        1.000                    1.000
honest but rough (sphere r=0.33)                    0.563                    0.562
honest but rough (cube 0.8)                         0.470                    0.428
CHEAT: flat disc facing camera                      0.807                    0.237
```

The toy policy can't reason and can't compose multiple objects. An LLM brings strong priors about
shapes and spatial reasoning, so it starts far above random guessing. RL then improves what it
already does reasonably well, which is why the curriculum and the cold start matter.
