# Neurogenesis Automaton

A DNA string (A, C, G, T) is decoded into growth rules. A cellular automaton on a 48 × 48 grid runs those rules
and grows a network of neurons and synapses. That network is a recurrent spiking neural network, and it steers a
swimmer that has to collect food. Training is evolution on the DNA itself.

Open `index.html` in a browser. It's a single file with no build step and no dependencies; training runs in a
Web Worker (falling back to the main thread if workers are unavailable).

## From DNA to rules

1. **Genes.** A gene starts at `ATG` (in any reading frame) and runs codon by codon to the first stop codon
   (`TAA`, `TAG`, `TGA`). If there's no `ATG`, the whole strand is read as one gene. Up to 6 genes are used.
2. **Each gene is one neuron type, with a fixed role by position:**
   - gene 1: the **sensors**, 5 neurons in an arc that smell food at −60°, −30°, 0°, 30° and 60° from the heading
   - gene 2: the **motors**, 2 neurons that turn the swimmer left and right
   - gene 3: the **zygote**, the interneuron that every other interneuron grows from
   - genes 4–6: further **interneuron** types

   A genome needs at least 3 genes to build a working brain.
3. **Codons are numbers.** Each codon is read in base 4 (`A=0 C=1 G=2 T=3`), giving 0–63. A gene's first 24
   codons set its rules. Shorter genes wrap around, with an offset so repeats still give new values. Codons past
   the 24th change nothing (silent mutations).

| Codon | Rule | Range |
|------:|------|-------|
| 1 | steps between divisions | 1–6 |
| 2 | number of divisions | 1–4 |
| 3 | distance to daughter cell | 2–5 cells |
| 4 | first division direction | 0–337.5° in 22.5° steps |
| 5 | turn added per division | 0–315° in 45° steps |
| 6 | daughter type on odd divisions | an interneuron type |
| 7 | generation cap: divides only in generations below this | 2–5 |
| 8 | number of axons | 1–3 |
| 9 | delay before sprouting axons | 0–5 steps |
| 10 | which types its axons connect to | bitmask (0 = all allowed) |
| 11 | chemotaxis toward/away from target signal | −1…+1 |
| 12 | max axon length | 8–35 cells |
| 13 | straightness of axon growth | 0…1 |
| 14 | growth signal strength released | 0.25…1 |
| 15 | spike threshold | 0.5…1.5 |
| 16 | daughter type on even divisions | an interneuron type |
| 17 | leak: fraction of charge kept per tick | 0.5…0.95 |
| 18 | bias current | −0.2…+0.4 |
| 19–24 | synapse weight onto sensor, motor, interneuron 1–4 | −1.6…+1.6 |

Only interneurons divide; sensors and motors are placed at fixed cells and just grow axons. Sensors and motors
connect only to interneurons, and interneurons connect to interneurons and motors, so every path from smell to
movement runs through the grown brain. A synapse's weight is set by its source type's weight onto its target type.

## Development

Each step of the automaton:

- **Signals** diffuse across the grid, one channel per neuron type, decaying as they spread.
- **Interneurons** divide on schedule into a free cell (no other neuron in its 3 × 3 neighbourhood). Daughters
  inherit the direction they were placed in, so the turn rule builds spirals, fans and branches. Once a neuron
  stops dividing, it waits, then sprouts its axons.
- **Axon tips** move one cell per step. Each picks the neighbouring cell that best combines climbing (or fleeing)
  its targets' signal, keeping its heading, and some noise. A tip that touches a target neuron becomes a synapse.
  A tip that runs out of length is pruned.

The noise is seeded from the decoded rules, so the same rules always grow the same brain.

## The spiking brain

Leaky integrate-and-fire, one update per world tick: `v ← v·leak + bias + input`. When `v` reaches the threshold
the neuron spikes and resets to 0. Spikes arrive at their targets on the next tick, so loops in the wiring make it
recurrent. Sensor neurons get an extra input current proportional to how much food they smell.

## The world

A 100 × 100 wrap-around square with 12 food pellets; an eaten pellet reappears somewhere random. The swimmer moves
forward at a constant speed and turns by the difference between the recent firing rates of its turn-left and
turn-right motor neurons. An episode is 500 ticks, and the score is the food eaten.

## Training

Evolution on the DNA strings: 48 genomes per generation, each scored on the same 6 random worlds. The top 4 are
kept unchanged, and the rest are mutated copies of tournament winners (about 1.5 point mutations per genome, plus
occasional codon insertions and deletions). Each generation's best genome is re-tested on 8 fixed worlds, and the
best of those becomes the champion. In headless Chromium this runs at about 1.2 generations per second.

What to expect, in food per episode, averaged over 40 fresh worlds that played no part in training or picking
the champion:

| Brain | Food |
|------|-----:|
| swims straight (no working motors) | 4.3 |
| hand-wired reflex: left sensors drive the left turn, right sensors the right | 9.4 |
| evolved, 3 runs of 120 generations | 9.5, 10.8, 11.2 |
| evolved, 3 runs of 400 generations (the examples in the page) | 14.3, 14.8, 15.2 |

The page reports scores on the 8 test worlds used to pick champions, which run about 15% higher (16.4–17.9 for
the examples).

Most random brains have no path from sensors to motors, so runs often sit at the straight-swimming level for a
while before jumping.

## Ideas for next steps

- Learning within a lifetime as well as across generations, e.g. reward-modulated STDP on the evolved weights.
- Tasks that need memory (moving food, food you have to remember after it goes out of range), where the recurrent
  wiring would matter more.
- Export the champion network as JSON (neurons, positions, weighted edges).
