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

Round 2 (experiments 10-13) tested three ways around that:

6. **A food curriculum doesn't help random brains learn.** Thinning the food
   as the population copes lets random brains survive at medium density. But
   real plasticity is selected against: the learning rate is held at 0.007,
   against 0.25 for a control where the gene has no effect. Those populations
   also do worse.
7. **Learning only on sensory synapses makes random brains able to learn but
   not willing.** Forced to rate 0.1, evolved brains learn colours (55% -> 74%
   safe picks), but they bite less, and evolution keeps the rate low.
8. **A hand-built scaffold thrives in the full default world, and learning
   wins there.** The scaffold has colour-specific "nearest food wins" relays and
   reflexes for air, predators and big food.
   - It sustains 60-90 creatures with no refills.
   - It reaches 70-75% difficulty within 700k steps. Random-start populations
     reached 38-60% after 10M steps.
   - It still learns the poisonous colour after 1.5M steps of evolution.
   - Learners displaced non-learners in 6 of 6 runs at low difficulty and 6 of 6
     at 70% difficulty.

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

## Round 2

New in `nworks_exp`:
- `plasticInputsOnly=1`: only synapses from sensor neurons learn.
- `food_curriculum=1`: every `food_window` steps (default 20,000), food is
  thinned (x0.9, down to `food_min`) if the population needed no refills, and
  thickened (+5) if it needed 10 or more.
- `start_difficulty=D`.
- The `full` hand-built circuit (see below).

### 10. Food curriculum with random brains
Focused world, fixed poison 0.3, unlimited air, learning-rate gene mutating 10x
faster, 4M steps.

| | real learning | neutral control | sensory-only learning |
|---|---|---|---|
| learning rate at the end (average over the run) | 0.007 (~0.01) | 0.25 (drifts up freely) | 0.009 (~0.02) |
| food reached (per colour) | 45-72 | 24-27 | 82-100 |
| seed-bank refills | 4,572 | 2,015 | 3,133 |
| difficulty | 12% | 40% | 0% |
| within-life test, own rate | no learning | (learning off) | no learning |
| within-life test, rate forced to 0.1 | 50% -> 60% -> 46%, bites in 23% of rounds | (learning off) | 55% -> 68% -> 74%, bites in 43% of rounds |

The curriculum lets random brains survive at medium food density. But real
plasticity is held about 35x below where the gene drifts when it does nothing,
and those populations cope worse. For random-start brains, learning is a
handicap. Restricting learning to sensory synapses makes evolved brains capable
of learning when the rate is forced, but it still cuts how often they eat, and
evolution keeps the rate low.

### 11. The `full` scaffold
A hand-built brain for the full default world (86 neurons):

| part | wiring |
|---|---|
| food | One relay per colour and vision sector, driven by that sector's food input. Relays inhibit each other, so the nearest food wins instead of all pulls cancelling out. Each relay drives movement toward its sector. |
| air | A "migrate" neuron fires when breathlessness is high and pushes up-left, which from every food patch leads to a spring. Per-sector spring relays fire only when migrating and a spring is in view, and steer straight to it. |
| predators | Direct wiring away from each sector. |
| big food | Direct wiring toward each sector. |
| calls | None. |

Results:
- In the default world, with everything on, it sustains 62-87 creatures with
  no refills. Random-start populations sustain about 15-40.
- Within-life test: safe picks 54% -> 55% -> 54% frozen, and 64% -> 80% -> 92%
  at learning rate 0.02.
- Learning doesn't reduce biting (82% of rounds vs 78%).

### 12. Learners vs frozen with the scaffold, full world
Half the founders learn (rate 0.02), half don't; the rate does not mutate; 60k
steps; 6 seeds each.

| start | learner share over time | result |
|---|---|---|
| difficulty 0 (reached 8%) | e.g. 65 -> 90 -> 97 -> 100% | learners win 6 of 6 (96-100%) |
| difficulty 70% (predators, poison ~28% of health per bite) | e.g. 46 -> 65 -> 90 -> 100% | learners win 6 of 6 (96-100%), mostly by 12-18k steps |

### 13. Long evolution from the scaffold, full world
1.5M steps, learning-rate gene mutating 10x faster, neutral control alongside.

| | real (2 seeds) | neutral |
|---|---|---|
| difficulty reached | 70-75% by 700k steps | 70-72% |
| within-life test at 1.5M (frozen ~50%) | 60% -> 78% -> 81%; 57% -> 74% -> 84% | no learning |
| learning rate | 0.02-0.14 | 0.01-0.16 (drift) |
| poison share by bite # in period | 57 / 52 / 48 / 45 / 39% | flat ~49% |
| neurons | 86 -> 92 | 86 -> 84 |

What this shows:
- Learning is kept and keeps working as difficulty rises, and it isn't pushed
  toward 0 as it is in random brains.
- The learning rate doesn't rise above neutral drift either. Once learning is
  present, its exact level looks roughly neutral.
- The overall poison share stays ~50%, because of the cost after each colour
  swap.
- In ~500 generations, evolution added little to the scaffold.

## Round 3: no lifetime learning, aiming for evolved instincts

### 14. A fixed poisonous colour isn't enough
Learning off, red always poisonous, 2M steps, full world.

| poison | food | poison share of bites |
|---|---|---|
| damage scaled by difficulty (default then) | default | 46-50% |
| 30% of health per bite, every time | default | 46-49% |
| 30% per bite | half (50 per colour) | 45-47% |
| 30% per bite, focused world | default | 46-49% |

Most bites happen by bumping into food. In a dense world, colour choice barely
changes what gets eaten, and mild poison is soon made up by the next good bite.

### 15. Lethal poison
Red kills; 3M steps. Poison share fell slowly, 48.5% -> 46.3% and 45.5% -> 40.9%.
The population lives on refills (40,000-52,000 over 3M steps).

### 16. Eating as a decision (`eatAction`)
A new output: touched food is eaten only while it is >= 0.5. 3M steps.

| poison | poison share of bites |
|---|---|
| lethal | 48% -> 33%, and 48% -> 34% (still falling) |
| 50% of health | 44% -> 43%, and 47% -> 47% |
| lethal, only 25% of food poisonous | 22-25%, against 22-32% bad food in the world: no avoidance |

### 17. The new defaults
Eating is a decision, red always kills, no learning. Behaviour tests at 6M
steps, 3 seeds (see the main README for the table):
- Two of three populations evolve vision-guided foraging for purple and avoid
  red (72% and 83% purple picks, against 39-48% for random brains).
- One also heads for springs when out of air.
- The population stays at the minimum of 15, so difficulty never rises and
  predators never arrive.

One bug along the way: the first "one bite per step" commit compared against an
iterator invalidated by `erase`, so every bite ate all the food stored after the
nearest item. It was fixed in the next commit. The behaviour tests in 17 use the
fixed build.

## Round 4: letting the population grow

### 18. What keeps the population at the minimum?
Each run 1.5-3M steps, eating as a choice, red poisonous.

| change | population | difficulty | predator kills |
|---|---|---|---|
| no random brains in refills (lethal) | 15 | 0% | 0 |
| poison 60% of health | 15-25 | 0% | 0 |
| both, plus 50% more food | 15-33 | 0% | 0 |
| unlimited air (60% poison) | 15-19 | 0% | 0 |
| no poison at all | 28-50 | 100% | 1,671 |

Poison is the bottleneck, not air or refills.

There was also an ecology bug. Creatures that avoided red let it build up
until red filled the shared food cap (60-80% of the food lying around), so
purple had no room to regrow. Each colour now has its own cap
(`separateFoodCaps`).

### 19. Poison that grows with difficulty
Damage = poisonDamage x difficulty, so poison is mild while a population gets
going and lethal at 100% difficulty. With `poisonDamage` 1.0, populations hold
at 15-55, difficulty rises to 20-30%, and predators arrive. This is now the
default. Behaviour tests at 6M steps are in the main README:
- All three populations prefer purple: 86-100% picks against 39-63% for
  random brains.
- Seed 1 eats purple 74% of the time and red 2%.
- Seed 2 evolved a flee response: heading +0.20 against +0.05 with no
  predator.

## Round 5: intelligence tests, and training instead of evolving

### Tools
- **Hidden food** (`foodVisibleSteps`): food can only be seen for N steps
  after it appears, so remembering where it was pays.
- **Test battery** (`scenarios.cpp`), shared by evolved and trained brains.
  Each test puts one creature alone in an empty arena; trial k of a test always
  has the same layout. All food tests start the creature hungry. Every test
  except poison-choice and poison-learning has a control.

  | test | setup |
  |---|---|
  | forage | safe food 150 px away, eaten within 600 steps; control: no food there |
  | poison-choice | one safe and one poisonous food, opposite sides |
  | flee | a predator 100 px away; heading in the first 30 steps and survival over 600; control: no predator |
  | memory | food 150 or 250 px away is visible only for the first 20 steps; control: food never visible |
  | air | 20% air, a spring 200 px away; control: no spring |
  | poison-learning | the poisonous colour is random; 6 rounds of one of each colour; are rounds 2-6 safer than round 1? |
- **`nworks_iq`**: evolves a world and scores random and evolved brains.
- **Training bridge** (`libnworks_bridge.so`, `train_rl.py`):
  - A recurrent network (128-unit GRU) is trained with PPO to control one
    creature in the same world through a C interface.
  - It gets the same 56 inputs as an evolved creature and picks one of 18
    actions: stay or move in 8 directions, eat or not.
  - Reward: +1 good bite, -1 poison bite, -2 for dying.
- **`replay.py`**: records an episode of a trained brain as a web page that
  animates it.

### Results
**Evolution:** hidden-food world, food thinning, 6M steps, 20-60 creatures,
about 200M creature-steps. **Training:** one creature, 40 food per colour, 30%
difficulty, 20M steps (about 40 minutes on 2 cores).

| test (control) | random brains | evolved (best seed) | trained, food visible 60 steps | trained, food visible 15 steps |
|---|---|---|---|---|
| forage | 7% (5%) | 28% (0%) | 97% (6%) | 59% (5%) |
| poison-choice: purple picks | 49% | 88% | 100% | 100% |
| flee: survival / heading | 14% / -0.02 | 0% / +0.02 | 77% / +0.94 | 80% / +0.93 |
| memory 150 px | 2% (1%) | 0% (0%) | 8% (6%) | 8% (4%) |
| air | 7% (4%) | 18% (0%) | 13% (2%) | 45% (6%) |
| poison-learning: rounds 2-6 safe | about 50% | - | 50% | 51% |

What this shows:
- **Training beats evolution on every reflex,** from roughly a tenth of the
  experience:
  - it goes straight for food and never picks red;
  - it runs almost directly away from predators (heading +0.94);
  - the 15-step agent also learned to head for springs.
- **Neither method learned memory.**
- **Neither learns which colour is poisonous within a life.** Agents trained
  with a fixed poisonous colour pick purple every time, so in poison-learning
  they are right only half the time.

Evolution with food visible for 15 steps also showed no memory. One seed
evolved a flee response (heading +0.23, 31% survival).

## Round 6: memory, learning to learn, many agents

New in `train_rl.py`:

| option | what it does |
|---|---|
| `--prev` | the network also gets its previous action and reward |
| `--vis-curriculum START:END` | food visibility shrinks during training |
| `--multi K` | K creatures per world with a call action; `share=S` mixes each creature's reward with its world's average |
| `--gym` | trains directly on the memory skill |
| `--init` | starts from trained weights |

New in the bridge: `random_poison=1` (poisonous colour random per episode), a
`poison-learning` test, and `replay.py` (animated web page of an episode).

### Results (400 trials per test)

| trained agent | forage | purple picks | flee: survival / heading | memory 150 px (control) | poison-learning, rounds 2-6 safe |
|---|---|---|---|---|---|
| memory curriculum, food visible 150 -> 15 steps, previous action and reward, 25M steps | 50% | 100% | 99% / +0.70 | 5% (7%) | 50% |
| learning to learn (random poisonous colour), previous action and reward, 20M steps | 23% | 64% | 69% / +0.11 | 5% (3%) | 48% |
| 8 creatures per world, half of reward shared, 20M steps | 100% | 100% | 15% / +0.94 | 8% (6%) | 50% |

- **Memory:** as visibility shrank, the agent did not learn to remember. It
  just foraged worse (training return +0.5 -> -0.3).
- **Learning to learn:** not learned. The agent mostly stopped eating.
- **Calls:** with 8 agents and shared reward, calls stayed unrelated to
  predators (49.6% of steps with a predator near vs 48.4% otherwise). These
  agents are the best foragers so far.

### Memory gym
Trained from scratch, the gym agent collapsed. Its success rate fell from ~5%
(random) to 0%, and it ended up moving in one direction 99.9% of the time
without eating. A success every ~20 episodes of 600 steps is too sparse a
reward for PPO to start from. Starting from the memory-world agent instead
(`--init`), it succeeds 40% of the time straight away.

## Ideas not yet tried
- Fix the after-swap cost so colour learning pays overall. For example, let
  the forgetting rate evolve, or give creatures a cue that the swap happened.
- Find out why plasticity costs random brains foraging: run random brains with
  reward-only and with punishment-only learning.
- Much longer runs from the scaffold, to see whether evolution elaborates it
  (e.g. adds calls or memory of spring locations).
- Larger populations (a bigger world) to reduce drift.
