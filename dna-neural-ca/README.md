# Neurogenesis Automaton

A DNA string (A, C, G, T) is decoded into growth rules, and a cellular automaton on a 48 × 48 grid runs those
rules starting from one cell. Neurons (circles) divide, release chemical signals and grow axons. When an axon tip
touches a neuron of a type it targets, it forms a synapse (a line).

Open `index.html` in a browser. It's a single file with no build step and no dependencies.

## From DNA to rules

1. **Genes.** A gene starts at `ATG` (in any reading frame) and runs codon by codon to the first stop codon
   (`TAA`, `TAG`, `TGA`). If there's no `ATG`, the whole strand is read as one gene. Up to 6 genes are used.
2. **One gene = one neuron type.** Gene 1 defines the zygote, the single cell everything grows from.
3. **Codons are numbers.** Each codon is read in base 4 (`A=0 C=1 G=2 T=3`), giving 0–63. A gene's first 16 codons
   set its type's rules. Shorter genes wrap around, with an offset so repeats still give new values. Codons past
   the 16th change nothing (silent mutations).

| Codon | Rule | Range |
|------:|------|-------|
| 1 | steps between divisions | 1–6 |
| 2 | number of divisions | 1–4 |
| 3 | distance to daughter cell | 2–5 cells |
| 4 | first division direction | 0–337.5° in 22.5° steps |
| 5 | turn added per division | 0–315° in 45° steps |
| 6 | daughter type on odd divisions | any type |
| 7 | generation cap: divides only in generations below this | 2–5 |
| 8 | number of axons | 1–3 |
| 9 | delay before sprouting axons | 0–5 steps |
| 10 | which types its axons connect to | bitmask (0 = any) |
| 11 | chemotaxis toward/away from target signal | −1…+1 |
| 12 | max axon length | 8–35 cells |
| 13 | straightness of axon growth | 0…1 |
| 14 | signal strength released | 0.25…1 |
| 15 | synapse weight (negative = inhibitory) | −1…+1 |
| 16 | daughter type on even divisions | any type |

## The automaton

Each step:

- **Signals** diffuse across the grid. Each neuron type has its own channel, decaying slowly as it spreads.
- **Neurons** divide on schedule into a free cell (no other neuron in its 3 × 3 neighbourhood). Daughters inherit
  the direction they were placed in, so the turn rule builds spirals, fans and branches. Once a neuron stops
  dividing, it waits, then sprouts its axons.
- **Axon tips** move one cell per step. Each picks the neighbouring cell that best combines climbing (or fleeing)
  its targets' signal, keeping its heading, and some noise. A tip that touches a target neuron becomes a synapse.
  A tip that runs out of length is pruned.

The noise is seeded from the decoded rules, so the same rules always grow the same network. A mutation in
non-coding DNA changes nothing unless it creates or breaks a start or stop codon.

## Ideas for next steps

- Run activity through the grown network using the synapse weights (e.g. leaky integrate-and-fire).
- Export the network as JSON (neurons, positions, weighted edges).
- Evolve genomes toward a target behaviour by mutating and selecting.
