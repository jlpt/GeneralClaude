# Neurogenesis Walker

The walking version of [`dna-neural-ca`](../dna-neural-ca). A DNA string grows a spiking neural network on a
48 × 48 grid (same genome format and growth automaton), and the network drives the hip and knee muscles of a
two-legged walker seen from the side. Evolution on the DNA selects for distance walked in 10 seconds.

Open `index.html` in a browser. It's a single file with no dependencies; training runs in a Web Worker.

## The body

Nine point masses joined by rigid bones, simulated with position-based dynamics at 60 steps a second: a box-shaped
torso with a head, and a rear and a front leg, each with a hip, knee and foot. Four muscles span the joints:

| Motor neuron | Muscle | Firing does |
|---|---|---|
| rear hip | front of torso → rear knee | swings the rear thigh forward |
| rear knee | rear hip → rear foot | folds the rear leg |
| front hip | head → front knee | swings the front thigh forward |
| front knee | front hip → front foot | folds the front leg |

A muscle is a distance constraint whose target length shrinks (by up to 30% at the hips and 25% at the knees) with
its motor neuron's recent firing rate. When the motor neuron is silent, the muscle holds the standing pose.

## Senses

Five sensor neurons, built by gene 1:

- **beat A / beat B**: a steady rhythm that alternates between the two, once a second
- **rear foot / front foot**: whether that foot is on the ground
- **tilt**: how far the body is pitched nose-up or nose-down

Gene 2 builds the four motor neurons and gene 3 onward the interneurons, as in the forager. Sensors may also connect
straight to motor neurons (reflex arcs), which made training much more reliable.

## The encoding

Each gene's first 29 codons are read. Codons 1–24 are as in the forager: growth (divisions, axons, which types to
connect to), spiking settings, and a base weight onto each target type. The rest make it richer:

- **Weight map (codons 25–28, on by default).** A synapse's weight is its type-pair base weight, shifted by where its
  two neurons sit: how far toward the front the target is, how far toward the front the source is, whether the two
  are on the same side or crossed, and how long the synapse is. So the four motor neurons can each get their own
  weights from the same type, which a type-pair table alone can't do.
- **Learning synapses (codon 29, off by default).** Reward-modulated plasticity: a synapse whose source fired just
  before its target builds an eligibility trace, which becomes a weight change when the walker moves faster than its
  recent average. The lower half of codon values means no learning.
- **DNA sets the beat (off by default).** The sensor gene never divides, so its division codon is reused to set the
  beat's speed, from 0.5 to 3 times a second.

The three can be switched in the page's training panel. They change how the same DNA is read, so switching them
regrows and rescores the walker shown.

**Small DNA changes make small brain changes.** Axon growth has a little randomness. Originally it was seeded from all
the decoded rules, so a mutation that only nudged a weight re-rolled every axon path and rewired the whole brain. Now
each type's axons are seeded only by that type's growth rules, and each axon tip draws its own noise from which axon
it is and how far it has grown. Of single mutations that change a rule, the share that leaves the wiring untouched
went from 0–4% to 61–63%.

## Rules that keep it honest

Evolution is very good at finding holes in a physics simulation, and the first versions were exploited:

1. **Sliding.** Simple velocity-damping friction let a creature hold still, vibrate, and slide 15 m in 10 s
   with both feet down. Fixed with Coulomb friction: a loaded point sticks unless pushed past its static limit.
2. **Buzzing.** Even with proper friction, vibrating fast while shifting weight between the feet still moved
   it 4 m. Fixed by making muscles lag their motor neuron by about 0.2 s, as real muscle does, so they can't buzz.
3. **Kneeling.** Some evolved gaits scooted along on a knee. Now anything but a foot touching the ground counts as
   a fall and ends the run.

## Results

Distance in 10 seconds on flat ground. Every run starts from the same pose, so a genome always scores the same.

| Controller | Metres |
|---|---:|
| no muscle activity (stands still) | 0.05 |
| best hand-made rhythm (each muscle on one of 4 phases of a 1 Hz sine) | 4.83 |
| best rhythm found by searching muscle drives directly, no brain, about 1 beat per second | 7.4–8.7 |
| the same search allowed up to 3 beats per second (a fast scurry) | 18.0 |
| evolved brains, 150 generations (44 runs, both encodings with the rewiring fix) | median 3.6, best 7.5 |
| evolved brains, shipped encoding: the 4 best of the 22 weight-map runs below, continued to 500 generations (the examples in the page) | 6.41, 6.83, 7.27, 8.80 |

## Does the richer encoding help?

Not on this task, as far as I could measure. Each row is a separate set of evolution runs, 48 genomes per
generation, scored by metres walked in 10 seconds:

| Encoding | Runs | Mean after 150 gens |
|---|---:|---:|
| old | 6 | 4.22 |
| + weight map | 6 | 4.58 |
| + learning, first rule | 6 | 3.30 |
| + weight map + learning, first rule | 6 | 2.77 |
| + learning, fixed rule | 6 | 3.31 |
| + weight map + learning, fixed rule | 6 | 3.66 |
| + weight map + DNA beat | 6 | 3.10 |
| old, with the rewiring fix | 22 | 4.27 |
| + weight map, with the rewiring fix | 22 | 3.93 |

After 500 generations (4 runs each): old 5.24, weight map 4.75, weight map + DNA beat 3.75.

- The weight map makes no measurable difference: with 22 runs each, the gap is 0.34 m and a permutation test gives
  p = 0.54. Run-to-run spread (about 2 m) swamps it.
- Learning made things worse in every version. The first rule rewarded raw forward speed, so weights ran away; the
  fixed rule rewards speed above the recent average and lets genes switch learning off, and it was still worse. On
  flat ground with identical runs, evolution can set the weights directly, so learning has nothing to adapt to.
- Letting the DNA set the beat made things worse: most random genomes start with an unworkable rhythm.
- The rewiring fix made mutations far more local without measurably changing results.

What limits these walkers is the search, not the encoding: muscle drives searched directly reach 7.4–8.7 m at the
same pace, and evolution through a grown brain gets a median of about 3.6 m.

## Ideas for next steps

- A stronger search: bigger populations (one Web Worker per core), crossover between genomes, or novelty search.
- Uneven ground and a push now and then, so gaits must use the foot and tilt sensors, and so learning has something
  to adapt to.
- Let the DNA build the body too: number of legs, segment lengths, muscle placement.
