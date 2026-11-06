# Nworks experiments: when does lifetime learning emerge?

Nworks creatures never evolve lifetime learning in the default world: they bite
the poisonous colour 50% of the time and the learning-rate gene drifts to about
zero. These experiments work out why, and what it takes for learning to appear.

## Summary

1. **The plasticity rule works.** A hand-built brain with direct
   sense-to-movement wiring learns which colour is poisonous within one life:
   safe picks go from 51% to 87% over 12 choices (96% with 4 vision sectors).
2. **Evolution from random brains doesn't find a structure that uses it.** In
   evolved brains, plasticity mostly scrambles what evolution built (forcing the
   learning rate up makes them forage less). Brains with only direct wiring and
   no hidden neurons don't evolve learning either.
3. **With innate direct wiring, learning is strongly selected.** In a shared
   world, learners displaced identical non-learners in 8 of 8 runs, from 50% to
   100% within 2-3 generations. Started at zero, the learning-rate gene rose
   above a neutral-drift control.
4. **What learning buys depends on the innate wiring.** With weak innate
   wiring, learning mostly makes up for it (food strengthens the approach).
   With well-tuned innate wiring, the payoff is learning which colour to avoid.
5. **Why it rarely pays off in the default world:**
   - The poisonous colour swaps every 10,000 steps, about once per lifetime, so
     what a creature learned is often wrong after a swap. Its first bites in a
     new period are worse than chance, and that cancels the gain.
   - Creatures eat only a few times per period.
   - Poison is mild at the difficulties reached.
   - Populations are small, so drift is strong.
   - In dense food, a simple "move toward food" circuit stalls, because pulls
     from every direction cancel out.

So learning in Nworks behaves like learning in animals: it builds on innate
wiring and doesn't replace it. Evolution here can keep and tune learning once
some innate scaffold exists. It has not been seen to invent that scaffold from
random wiring.

## Tools

Built by CMake alongside the headless runner (`nworks_exp`, `nworks_assay`).

**`nworks_exp SEED STEPS REPORT [tests=A,B] [key=value ...]`** evolves a world
with Config overrides (any field listed in `ApplyOverride`) and prints a row
every REPORT steps. Each row includes:
- poison share by fifth of the swap period;
- poison share by the creature's bite number within the current poison period
  (1st, 2nd, 3rd-4th, 5th-8th, 9th+), counted exactly per creature;
- learning-rate statistics.

At each `tests=` step it runs a **within-life learning test** on copies of every
creature. Each copy gets one life of 12 rounds, with one food of each colour on
opposite sides each round. It runs with plasticity on, frozen, and with the rate
forced to 0.1.

Seeding options start the population from hand-built brains:
- `seed_circuit=direct`
- `seed_rate=R`
- `seed_mix=1` (every other founder gets learning rate 0)
- `seed_wmotor=W`
- `seed_count=N`

`nworks_exp handbuilt circuit=direct|gated [sectors=4] [...]` runs the
capacity test: the within-life test on hand-built brains at learning rates 0 to
0.5.

**`nworks_assay SEED CHECKPOINTS [TRIALS]`** runs behaviour tests on evolved
creatures. Each one is placed alone in an empty arena and tested on:
- foraging, by colour, against a no-food control;
- heading away from a predator, against a no-predator control;
- reaching a spring when out of air;
- colour choice.

At the end it measures what calls correlate with in the evolved world.

### New Config switches (defaults keep the original behaviour)
| Setting | Effect |
|---|---|
| `poisonScalesWithDifficulty` | `false`: every poison bite costs the full `poisonDamage` |
| `colourSmell` | +2 inputs: how close the nearest food of each colour is, from any direction |
| `maxAge` | lifespan limit in steps (0 = none) |
| `learningHasEffect` | `false`: the learning-rate gene still mutates but nothing is learned (neutral control) |
| `learnFromReward`, `learnFromPunishment` | switch off one of the two learning signals |

Also new: `NeuralNetwork::AddHiddenNeuron`, `Connect` and `SetLearningRate`
for hand-building brains, plus per-creature bite counters.

### Fix
Mutations per birth were `int(synapses x 0.02 x rate)`, which is 0 for brains
under 50 synapses, so small brains never mutated. They are now rounded randomly
(`rng::RoundRandomly`). Default 600-synapse brains still get 12 mutations, but
the extra random draw means a given `--seed` no longer reproduces runs from
before this change.

## Experiments

Common focused setup unless noted: predators, calls and big food off.

### 1. Baseline (default world)
| What was measured | Result |
|---|---|
| Within-life test, safe picks | ~50%, with learning on or frozen |
| Learning rate | drifts to ~0 |
| Behaviour tests (`nworks_assay`, 3 seeds × 10M steps) | vision-guided foraging with a fixed colour habit, unrelated to which colour is poisonous; a flee reflex in one or two of three populations; no alarm or food calls |

### 2. Strong fixed poison (50% of health per bite), with and without colour smell
No learning in any run. The population lives on seed-bank refills the whole
time, so selection is weak. Forcing the learning rate to 0.1 in evolved brains
halves their bite rate without improving choices.

### 3. Capacity test: can the rule learn colour avoidance?
The hand-built "direct" circuit sends food of each colour in each sector to the
movement outputs toward that sector. There are 40 brains with ±20% random
innate biases, 2 lives each.

| Learning rate | Safe picks, rounds 1-4 / 5-8 / 9-12 |
|---|---|
| 0 (frozen) | 51% / 54% / 54% |
| 0.02 | 61% / 82% / 87% |
| 0.02, 4 sectors | 69% / 88% / 96% |

Higher rates also learn, but suppress foraging.

### 4. Hand-built learners in the world
Sparse food (10 per colour), fixed poison 0.3, unlimited air. Poison share by
bite number within a poison period (swaps every 10,000 steps):

| | 1st | 2nd | 3rd-4th | 5th-8th | 9th+ |
|---|---|---|---|---|---|
| learners (rate 0.02) | 60% | 51% | 44% | 38% | 35% |
| frozen | 48% | 47% | 46% | 46% | 46% |

The overall poison share is still ~50%, because of the bad first bites after
each swap. Learners are fitter anyway. With swaps every 30,000 steps over 200k
steps:
- 1,354 births against 898;
- 64 seed-bank refills against 154;
- 40% more bites.

In dense food (up to 200 items) the direct circuit stalls.

### 5. Does evolution turn learning on? (learning rate starts at 0)
Hand-built founders. The learning-rate gene mutates 10x more often than default
(`learningRateChance=0.2`). Runs last 2M steps.

| | Average learning rate over the run |
|---|---|
| real learning (4 runs) | 0.013, 0.016, 0.016, 0.021 |
| neutral control, `learningHasEffect=0` (2 runs) | 0.011, 0.010 |

The control matters because the rate can't go below 0, so drift alone pushes
the average up. The difference points the right way but is small.

### 6. Head-to-head: learners vs frozen in one world
There are 30 founders with the same circuit: half with learning rate 0.02, half
with 0, and the rate is not allowed to mutate. Learners took over in 8 of 8 runs
(chance odds: 1 in 256). They went from ~50% to 100% within ~15,000 steps, with
no seed-bank refills in that window.

### 7. What is the advantage?
Learner share after 30k steps, 6 seeds each:

| Learning signal | Motor weights 0.1 | Motor weights 0.3 |
|---|---|---|
| good food only | 93, 97, 100, 100, 100, 100% | 93, 71, 75, 100, 87, 27% |
| poison only | 67, 63, 87, 80, 100, 41% | 100, 100, 87, 90, 100, 100% |
| both | 100, 96, 100, 100, 93, 100% | 95, 100, 100, 100, 100, 100% |

### 8. Random brains in the learner-friendly world
In sparse food they never sustain themselves: the population sits at 15, with
~3,500 refills per million steps. That's a chicken-and-egg problem:
- In dense food, random brains survive but learning doesn't pay.
- In sparse food, learning pays but random brains can't get started.

### 9. Random brains with only direct wiring (no hidden neurons)
| Food | Result |
|---|---|
| dense | One seed sustains itself, but its learning rate falls to ~0.003 (below neutral) and poison share by bite stays flat. The other never sustains itself. |
| 25 per colour | Neither seed sustains itself. |

## Ideas not yet tried
- Put the innate learning circuit into the full default world. It would need an
  innate air-seeking reflex as well.
- A curriculum from dense to sparse food, so random brains can get started and
  then face pressure to learn.
- Let the forgetting rate evolve, so creatures can unlearn quickly after a swap.
- Larger populations (a bigger world) to reduce drift.
