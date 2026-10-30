# Neurogenesis (JAX)

A creature grows from a single cell. One cellular automaton grows its **body and its brain together**:
every cell becomes bone, muscle or an eye, and every cell is also a spiking neuron wired to its
neighbours. The creature lives in a side-view world with fruit on the ground, has to find fruit to keep
its energy up, and starves if it doesn't. Evolution improves the numbers that define the growth rule.

Written in JAX, so a whole population is grown and simulated in parallel on a GPU.

## Setup

**Windows:** JAX's GPU support runs through WSL2. Install WSL2 with Ubuntu (`wsl --install` in an
administrator PowerShell) and keep your normal Windows NVIDIA driver. Don't install a Linux NVIDIA driver
inside WSL2. Then, inside Ubuntu:

```bash
cd neurogenesis-jax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -U "jax[cuda13]"
```

On Linux, do the same without WSL2. For a CPU-only install, skip the last line. Check the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the current CUDA extra if
`cuda13` doesn't suit your driver. `python3 -c "import jax; print(jax.devices())"` should list your GPU.

## Use

```bash
python evolve.py --out runs/first                # sized for a GPU: 512 creatures per generation, 500 generations
python evolve.py --out runs/check --small        # a quick CPU check: 32 creatures, 15 generations, 8 s lives
python evolve.py --out runs/more --resume runs/first/mean.npy    # carry on from where a run stopped
python render.py runs/first/best.npy --out creature.gif          # watch it grow and live
```

The physics works on square cells, but `render.py` paints each frame's cells as one soft, rounded body:
it blurs the outline and cuts it back out, and blends the tissue colours.

`evolve.py` prints one line per generation and writes `best.npy`, `mean.npy`, `log.csv` and
`config.json` into the run folder. Useful options: `--pop`, `--gens`, `--seconds` (length of each
life), `--sigma` (size of the random changes), `--lr`, `--seed`.

## How it works

The code is in three files, each with notes at the top:
- **`creature.py`**: the growth rule, and how a grown grid becomes a body and a brain
- **`sim.py`**: physics, brain, senses and survival
- **`evolve.py`**: the starter body and evolution

### The genome
A flat list of about 2,100 numbers. Almost all of them are the weights of the growth rule, a tiny neural
network. The last 8 are a wiring bias for each of the 8 neighbour directions. There are no DNA letters.

### Growth (`creature.py`)
The grid is 10 × 14 cells, and each cell holds 16 numbers. Growth starts with one living cell. On each of
32 steps, every cell at once:
1. senses its own numbers and how its neighbourhood's numbers change left to right and top to bottom
2. runs those through the growth rule
3. adds the result to its own numbers

A cell lives if it or a neighbour is alive (channel 0 above 0.1). This is a neural cellular automaton, as
in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). Growing robots this way
has also been studied ([Horibe, Walker & Risi, 2021](https://arxiv.org/abs/2102.02579)).

When growth ends, each cell's numbers are read off:

| Channel | Becomes |
|---|---|
| 0 | alive or empty (only the piece joined to the seed is kept) |
| 1–3 | tissue, whichever is largest: bone (stiff), muscle (contracts) or eye (sees fruit) |
| 4–5 | the neuron's spiking threshold and bias |
| 6–9 | how strongly the neuron responds to the beat, touch, sight and hunger |
| 10–11 | how strongly it sends to and receives from neighbours |
| 12–15 | no fixed meaning; cells use them to signal each other while growing |

The synapse from cell i to neighbour j has weight 1.5 · tanh(send_i · receive_j + direction bias). So the
same growth rule shapes the limbs, decides where the muscles and eyes go, and wires the brain.

### Body (`sim.py`)
Each living cell is a 10 cm square whose corners are point masses. Neighbouring cells share corners.
Edges and diagonals are distance constraints, solved with position-based dynamics:
- 3 substeps per tick, each with 4 passes over 8 batches of constraints that share no corners, so every
  batch is solved in parallel
- a muscle cell shrinks its edges by up to 30% as it contracts
- the ground pushes back, and Coulomb friction makes a loaded point stick unless pushed hard, so a
  creature must lift and push to travel

### Brain
One leaky integrate-and-fire neuron per cell. Each tick (1/30 s):

```
charge = 0.8 · charge + bias + senses + spikes @ weights
```

A neuron spikes and resets when its charge reaches its threshold. A muscle cell follows its own neuron's
recent firing rate, with about a 0.2 s lag; real muscle is slow too, and this stops evolution winning
with tiny fast vibrations.

### Senses
Each sense is a current into a cell's neuron, scaled by that cell's gain for it:
- **beat:** a steady rhythm, sin(2π · 1.5 · t), into every cell
- **touch:** 1 for cells touching the ground
- **sight:** eye cells only. Fruit is one colour; an eye looks away from the middle of the body and reads
  how much fruit colour lies ahead, with nearer fruit brighter. It's computed straight from the fruit
  positions, which gives the same numbers as reading a strip of pixels.
- **hunger:** 1 − energy, into every cell

### Survival and score
- **Settling.** Before the clock starts, each creature gets a tiny random sideways nudge and 3 s to
  settle with its muscles off. Fruit is then placed 0.9–2 m away on either side of where it came to rest.
  Without this, evolution grew tall bodies that simply fell over onto fruit.
- **Energy.** It starts at 1 and lasts 12 s for an idle creature the size of the starter (27 cells).
  Bigger bodies drain faster, but smaller ones never drain slower; otherwise evolution shrank creatures
  until they outlived the clock without eating. Working muscles make it drain up to 1.5× faster.
- **Food.** Each fruit eaten adds 0.4. A fruit is eaten when any body point touches it near the ground,
  and it regrows 1–3 m from the creature on a random side.
- At zero energy the creature dies and stops.

The score is fruit eaten + the fraction of its life it stayed alive + 0.2 per metre travelled (up to
5 m). The travel credit lets creatures that merely move beat ones that don't, before any can reach fruit.
A body of fewer than 3 cells scores 0; a growth rule that lets its seed die once beat everything by
having no body to feed.

### Evolution (`evolve.py`)
1. **Starter body.** Gradient descent trains the growth rule to grow a round bone body with one leg (a
   column of muscle beside a column of bone, so contracting bends it) and an eye on each side. This
   takes about 10 seconds. Evolution starts from that simple creature rather than from shapeless
   random growth.
2. **OpenAI-ES.** Each generation adds small random noise to the genome to make the population,
   grows and runs every member in the same world, and moves the genome toward the noise of the members
   that scored best. Noise comes in mirrored pairs (+ε and −ε), and scores are converted to ranks.
   Each generation's best is re-tested on 4 fixed worlds, and the best of those is saved as
   `best.npy`.

Growth rules are sensitive: noise of 0.01 changes about 4 cells of the starter body, and 0.05 can wipe
the body out, so `--sigma` defaults to 0.01.

## Speed and results so far

On one CPU core here, an 8-second life takes about 35 ms. Generations of 128 creatures with 20 s lives
took about 13 s each on this 4-core CPU. A GPU runs the population in parallel, which is what the
default of 512 creatures is for.

I could only test on a CPU. In a run of 128 creatures per generation with 20 s lives, the champion's
score on the fixed test worlds rose from 0.92 to 1.19 in 40 generations (the starter scores 0.54). That
is slow progress, and why the defaults assume a GPU. Along the way, evolution found three loopholes, all
now closed:
- toppling onto fruit
- shrinking to outlive the clock
- having no body at all

## Ideas for next steps

- More kinds of fruit (colours that are poisonous), so eyes have to tell them apart.
- Let fruit hang at different heights, so reaching matters.
- MAP-Elites (for example with [QDax](https://github.com/instadeepai/QDax)) to keep many different body
  plans instead of one.
- A bigger grid and more growth steps, for bigger bodies.
