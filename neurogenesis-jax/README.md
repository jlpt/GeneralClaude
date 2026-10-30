# Neurogenesis (JAX)

A creature grows from a single cell. One cellular automaton grows a **jointed body and, inside it, a
nervous system**. The growth runs on a fine grid, and every 3 × 3 patch of living cells becomes a joint of
a skeleton, joined by a bone to the patch it grew from, so thin lines of patches become limbs. Bone
joints are rigid and muscle joints are hinges. Any fine cell can also grow a nerve cell, so nerve cells
are a third the size of joints and a body can hold up to 1,260 of them. Nerve cells are the only
neurons, and a signal can only pass between nerve cells that touch, so **a joint only bends if a nerve
reaches it and fires**. The growth rule decides limb lengths, where the joints are, where the nerves run
and where they pile up into a brain.

Nothing is designed in advance: evolution starts from a single cell and a random growth rule, and its
job is to walk forward. Written in JAX, so a whole population is grown and simulated in parallel on a
GPU.

**Watch it in a browser:** `web/neurogenesis-walker.html` is the same simulation in JavaScript, with
evolution running in background workers while the best creature so far walks. Open it in any browser.

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

`render.py` draws bones as pale, rounded limbs and fills in solid patches of bone. A muscle joint is a
dark dot, with its muscle drawn as a red band on the side it pulls; the band darkens and thickens as it
contracts, and stays dull if no nerve reaches the joint. On top is the nervous system: a small violet dot
for each nerve cell, at its place inside its joint, violet fibres between wired nerve cells, and a green
flash when a nerve cell fires.

`evolve.py` prints one line per generation (metres walked) and writes `best.npy`, `mean.npy`, `log.csv`
and `config.json` into the run folder. Useful options: `--pop`, `--gens`, `--seconds` (length of each
life), `--sigma` (size of the random changes), `--lr`, `--seed` (which random growth rule it starts
from), `--init-scale` (how big that rule's weights are: bigger grows bigger random bodies). A genome copied out of the browser
version can be drawn here too; the page shows how.

## How it works

The code is in three files, each with notes at the top:
- **`creature.py`**: the growth rule, and how a grown grid becomes a skeleton and a nervous system
- **`sim.py`**: physics, the nervous system, senses and the score
- **`evolve.py`**: the starter creature and evolution

### The genome
A flat list of about 2,100 numbers. Almost all of them are the weights of the growth rule, a tiny
neural network. The last 8 are a wiring bias for each of the 8 neighbour directions. There are no DNA
letters.

### Growth (`creature.py`)
Growth runs on a 30 × 42 grid of fine cells, and each cell holds 16 numbers. It starts with one living
cell. On each of 40 steps, every cell at once:
1. senses its own numbers and how its neighbourhood's numbers change left to right and top to bottom
2. runs those through the growth rule
3. adds the result to its own numbers

A cell lives if it or a neighbour is alive (channel 0 above 0.1). This is a neural cellular automaton, as
in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). Growing robots this way
has also been studied ([Horibe, Walker & Risi, 2021](https://arxiv.org/abs/2102.02579)).

When growth ends, the grid is read off. Each 3 × 3 patch is one joint of the body, a 10 × 14 grid of
joints:

| Channel | Becomes |
|---|---|
| 0 | alive or empty; a patch with at least 5 of its 9 cells alive is a joint (only joints joined to the seed are kept) |
| 1–2 | the patch's joint: bone (rigid) or muscle (bends), whichever its cells add up to more of |
| 11 | for a muscle joint: which way it bends, and how far (averaged over the patch) |
| 3 | for each fine cell: above 0, a nerve cell grows there |
| 4–5 | the nerve cell's spiking threshold and bias |
| 6–7 | how strongly it feels the pacemaker rhythm and touch |
| 8–9 | how strongly it sends to and receives from neighbouring nerve cells |
| 10 | how hard it drives the joint whose patch it sits in |
| 12–15 | no fixed meaning; cells use them to signal each other while growing |

The synapse from nerve cell i to a touching nerve cell j has weight 1.5 · tanh(send_i · receive_j +
direction bias). Cells without a nerve have no neuron at all.

### Body (`sim.py`)
Every living cell is a joint: a point mass at its place on the grid, 10 cm from its neighbours.
- **Bones.** Each cell is joined by a bone to its *parent*, the neighbour one step nearer the seed, so the
  skeleton is a tree branching out from the seed. Neighbouring bone cells are also joined, so a solid
  patch of bone is solid.
- **Joints.** At each cell, the angle between the bone to its parent and the bone to each child is held
  where it grew. A bone cell holds it rigidly. A muscle cell is a hinge: when the nerve cell inside it
  fires, the joint bends by up to 70°, the way that cell's bend number says, and it straightens again
  when the nerve goes quiet. A muscle joint with no nerve never moves.
- **Solving.** Position-based dynamics: 6 substeps per tick, each with 2 passes over every bone and joint.
  Bones and joints are split into batches that share no points, so each batch is solved in parallel. A
  joint's angle is held with a linear constraint: with both arms one bone long, a joint at angle θ puts
  its cell at a fixed offset from the midpoint of its two neighbours.
- **Ground.** The ground pushes joints up. Friction follows Coulomb's law: the body's weight is shared
  among the joints touching the ground, and a loaded joint sticks unless pushed hard, so a creature must
  lift a foot to move it.

### Nervous system
The nerve cells form a spiking recurrent neural network. Every nerve cell is a leaky integrate-and-fire
neuron with weighted synapses to the nerve cells touching it (in both directions, so signals can loop).
Each tick (1/30 s):

```
charge = 0.8 · charge + bias + senses + Σ weight · spike, over the touching nerve cells that just fired
```

A nerve cell fires and resets when its charge reaches its threshold. Spikes spread only along chains of
touching nerve cells, one cell per tick, so a long nerve has a delay, as real ones do. The weights are
not trained by gradient descent: they come from growth, and evolution tunes the growth rule.

A muscle joint follows the nerve cells in its patch: their recent firing rates, each times its junction
strength, added up and divided by 3, so one nerve running straight through a joint can drive it fully.
It follows with about a 0.2 s lag; real muscle is slow too, and this stops evolution winning with tiny
fast vibrations.

**Senses** are currents into the nerve cells that can feel them:
- **touch:** nerve cells in the patch of a joint touching the ground
- **pacemaker:** a steady rhythm, sin(2π · 1.5 · t), felt by every nerve cell (each with its own
  strength), like the pacemaker cells that keep a heart beating

### The score
- Before the clock starts, each creature gets a tiny random sideways nudge and 3 s to settle with its
  muscles off, so a body that would topple over has already done so.
- Its middle is then tracked for 20 s as it moves right. **The slower half of the 20 s counts three times
  as much as the faster half**, so a steady walker keeps its full distance and one big lunge keeps only
  half. Evolution's first walkers lunged forward once and stopped.
- **Falling over scores 0**: turning more than 45° from the pose it settled in. Without this, evolution
  grew bodies that threw themselves forward and lay there.
- A body of fewer than 3 cells scores 0.

### Evolution (`evolve.py`)
1. **A random start.** Evolution starts from a single cell and a random growth rule. About half of random
   rules grow nothing, so rules are drawn until one grows some kind of body (at least 3 joints). Most are
   blobs with hundreds of nerve cells that twitch at random.
2. **OpenAI-ES.** Each generation adds small random noise to the genome to make the population, grows
   and runs every member, and moves the genome toward the noise of the members that walked farthest.
   Noise comes in mirrored pairs (+ε and −ε), and scores are converted to ranks. Each generation's best
   is re-tested 4 times with different nudges, and the best of those is saved as `best.npy`.

Growth rules are sensitive: noise of 0.01 changes a handful of cells, and 0.05 can wipe the body out, so
`--sigma` defaults to 0.01.

## Results so far

I could only test on CPUs, and this version (fine nerve cells, no starter) is new:
- About half of random growth rules grow a body. They are mostly blobs with dozens to hundreds of nerve
  cells; a quarter of them twitch enough to move a little.
- Starting from big blobs is slow: one that filled the whole grid never moved, and a 68-joint blob crept
  10 cm in 25 generations of 64. Small random starts are being tested now.
- An earlier version that started from a trained walker (with one nerve cell per joint) reached a steady
  0.70 m in 20 s within a minute of evolution in a browser, so walking is within reach of this body.

The JavaScript port was checked against this Python code: the same genomes grow identical bodies, nerve
cells and skeletons, and walk the same distance to within a few millimetres.

## Could creatures breed?

Measured on an earlier soft-bodied version of the same growth rule: two parents' genomes were mixed
(each number taken from one parent or the other) and the child's body compared with both parents'.

| How far apart the parents' numbers are (median) | Children that grow normally |
|---|---|
| about 6% (close relatives) | 98–100% |
| about 13% (two members of one generation) | 60–80% |
| about 20% | 50–80% |
| about 27% | 15% |
| about 40% | none |
| two separate lineages that grow the *same* body | none: about 40 cells come out wrong |

What matters is shared ancestry, not a percentage. Two growth rules trained separately to grow exactly
the same body use their numbers in completely different ways, so their children grow nonsense, like
crossing two species. Breeding works within a lineage, and lineages drift apart into species.

## Ideas for next steps

- Breeding: crossover between close relatives, with species kept apart by genetic distance.
- Uneven ground or pushes, so that touch feedback through the nerves matters more.
- An energy cost per nerve cell or per spike, as real brains are expensive.
- A bigger grid and more growth steps, for bigger bodies and bigger brains.

Earlier versions are in the git history: soft bodies with a neuron in every cell and fruit to find, the
same soft bodies with a grown nervous system, and jointed skeletons with a neuron in every cell.
