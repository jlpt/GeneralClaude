# Neurogenesis (JAX)

A creature grows from a single cell. One cellular automaton grows its **body and, inside it, a nervous
system**. Every cell becomes bone or muscle, and some cells also grow a nerve cell inside them. Nerve
cells are the only neurons, and a signal can only pass between nerve cells that touch. So a muscle only
contracts if a nerve runs through it, a foot only feels the ground if a nerve carries the feeling away,
and a brain is just a place where nerve cells cluster. Nothing in the code says where the brain goes or
what it connects to: the growth rule lays down nerves the same way it lays down bone and muscle.

The creature is a soft body in a side-view world, and its job is to walk forward. Evolution improves
the numbers that define the growth rule. Written in JAX, so a whole population is grown and simulated
in parallel on a GPU.

## Setup

**Windows:** JAX's GPU support runs through WSL2. Install WSL2 with Ubuntu (`wsl --install` in an
administrator PowerShell) and keep your normal Windows NVIDIA driver. Don't install a Linux NVIDIA driver
inside WSL2.

Work inside Ubuntu's own home folder, not under `/mnt/c`. Windows folders are slow from WSL, and a
OneDrive folder will try to sync the thousands of files in the Python environment. So copy the project
over first, for example
`cp -r /mnt/c/Users/<you>/Downloads/neurogenesis-jax ~/`. Then:

```bash
cd ~/neurogenesis-jax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -U "jax[cuda12]"
```

Creating and activating the environment prints nothing; `(.venv)` at the start of the prompt shows
it worked. On Linux, do the same without WSL2. For a CPU-only install, skip the last line.

Use `cuda12`, not `cuda13`. Ubuntu 22.04 comes with Python 3.10, and the newest JAX for Python 3.10
(0.6.2) has no `cuda13` option, so asking for it silently installs JAX without GPU support. `cuda12`
works with any recent NVIDIA driver. The code is tested on JAX 0.6.2 and newer. Run `python benchmark.py`:
the first line should name your GPU. If it says CPU, check that `nvidia-smi` works inside Ubuntu, and see
the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Use

```bash
python benchmark.py                              # is the GPU being used, and how fast is a generation?
python evolve.py --out runs/first                # sized for a GPU: 512 creatures per generation, 500 generations
python evolve.py --out runs/check --small        # a quick CPU check: 32 creatures, 15 generations, 8 s lives
python evolve.py --out runs/more --resume runs/first/mean.npy    # carry on from where a run stopped
python render.py runs/first/best.npy --out creature.gif          # watch it grow and walk
```

`render.py` draws the body as one soft shape: bone is pale, and muscle turns from pink to deep red as it
contracts. On top is the nervous system. Each nerve cell is a small dot inside its cell, joined by thin
fibres to the nerve cells it's wired to, and it flashes yellow when it fires, so you can watch signals
travel from the brain down to the legs.

On a GPU, time per generation is mostly a fixed cost for the long chain of physics steps, so bigger
populations come almost free. Pick `--pop` from what `benchmark.py` reports.

`evolve.py` prints one line per generation (metres walked) and writes `best.npy`, `mean.npy`, `log.csv`
and `config.json` into the run folder. Useful options: `--pop`, `--gens`, `--seconds` (length of each
life), `--sigma` (size of the random changes), `--lr`, `--seed`.

## How it works

The code is in three files, each with notes at the top:
- **`creature.py`**: the growth rule, and how a grown grid becomes a body and a nervous system
- **`sim.py`**: physics, the nervous system, senses and the score
- **`evolve.py`**: the starter creature and evolution

### The genome
A flat list of about 2,100 numbers. Almost all of them are the weights of the growth rule, a tiny
neural network. The last 8 are a wiring bias for each of the 8 neighbour directions. There are no DNA
letters.

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
| 1–2 | tissue, whichever is larger: bone (stiff) or muscle (contracts) |
| 3 | above 0: a nerve cell grows inside this cell too |
| 4–5 | the nerve cell's spiking threshold and bias |
| 6–7 | how strongly it feels the pacemaker rhythm and touch |
| 8–9 | how strongly it sends to and receives from neighbouring nerve cells |
| 10 | how hard it drives the muscle it sits in |
| 11–15 | no fixed meaning; cells use them to signal each other while growing |

The synapse from nerve cell i to a touching nerve cell j has weight 1.5 · tanh(send_i · receive_j +
direction bias). Cells without a nerve have no neuron at all.

### Body (`sim.py`)
Each living cell is a 10 cm square whose corners are point masses. Neighbouring cells share corners.
Edges and diagonals are distance constraints, solved with position-based dynamics:
- 3 substeps per tick, each with 4 passes over 8 batches of constraints that share no corners, so every
  batch is solved in parallel
- a muscle cell shrinks its edges by up to 30% as it contracts
- the ground pushes back, and friction follows Coulomb's law: the body's weight is shared among the
  points touching the ground, and a loaded point sticks unless pushed hard, so a creature must lift and
  push to travel

### Nervous system
Every nerve cell is a leaky integrate-and-fire neuron. Each tick (1/30 s):

```
charge = 0.8 · charge + bias + senses + spikes @ weights
```

A nerve cell fires and resets when its charge reaches its threshold. Signals spread only along chains of
touching nerve cells, one cell per tick, so a long nerve has a delay, as real ones do.

**Muscles** contract only when the nerve cell inside them fires. A muscle cell follows that nerve cell's
recent firing rate times its junction strength, with about a 0.2 s lag; real muscle is slow too, and
this stops evolution winning with tiny fast vibrations.

**Senses** are currents into the nerve cells that can feel them:
- **touch:** nerve cells in cells touching the ground
- **pacemaker:** a steady rhythm, sin(2π · 1.5 · t), felt by every nerve cell (each with its own
  strength), like the pacemaker cells that keep a heart beating

### The score
- Before the clock starts, each creature gets a tiny random sideways nudge and 3 s to settle with its
  muscles off, so a body that would topple over has already done so.
- The score is how far the creature's middle moves to the right in 20 s.
- **Falling over ends the run.** Once the body turns more than 45° from the pose it settled in, the
  score stays at the distance reached before the fall. Without this, evolution grew tall bodies that
  threw themselves forward and lay there.
- A body of fewer than 3 cells scores 0.

### Evolution (`evolve.py`)
1. **Starter creature.** Gradient descent trains the growth rule to grow a long, low body with a
   bending leg at each end (a column of muscle beside a column of bone), and inside it a brain of six
   nerve cells with a nerve running from it through each leg's muscle. Some random starts miss a
   detail, so it tries again until the creature grows exactly right; each try takes a few seconds.

   ```
   ..BBBBbbbBBB..     B bone, M muscle; lower case where a nerve runs
   ..BbbbbbbbBB..     (the brain is the cluster in the middle)
   ..mB......mB..
   ..mB......mB..
   ..mB......mB..
   ..mB......mB..
   ```
2. **OpenAI-ES.** Each generation adds small random noise to the genome to make the population, grows
   and runs every member, and moves the genome toward the noise of the members that walked farthest.
   Noise comes in mirrored pairs (+ε and −ε), and scores are converted to ranks. Each generation's best
   is re-tested 4 times with different nudges, and the best of those is saved as `best.npy`.

Growth rules are sensitive: noise of 0.01 changes a handful of cells of the starter, and 0.05 can wipe
the body out, so `--sigma` defaults to 0.01.

RESULTS_PLACEHOLDER

## Ideas for next steps

- Breeding: crossover between close relatives, with species kept apart by genetic distance (see above).
- Uneven ground or pushes, so that touch feedback through the nerves matters more.
- An energy cost per nerve cell or per spike, as real brains are expensive.
- A bigger grid and more growth steps, for bigger bodies and bigger brains.

Earlier versions are in the git history: the same soft bodies with a neuron in every cell and fruit to
find, and a version with jointed skeletons.
