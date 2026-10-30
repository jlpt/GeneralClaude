# Neurogenesis (JAX)

A creature grows from a single cell. One cellular automaton grows a **jointed body and, inside it, a
nervous system**. The growth runs on a fine grid, and every 3 × 3 patch of living cells becomes a joint of
a skeleton, joined by a bone to the patch it grew from, so thin lines of patches become limbs. Bone
joints are rigid and muscle joints are hinges. Any fine cell can also grow a nerve cell, so nerve cells
are a third the size of joints and a body can hold up to 1,260 of them. Nerve cells are the only
neurons, and a signal can only pass between nerve cells that touch, so **a joint only bends if a nerve
reaches it and fires**. The growth rule decides limb lengths, where the joints are, where the nerves run
and where they pile up into a brain.

Nothing is designed in advance: evolution starts from a single cell and a random growth rule. To stay
alive, a creature has to **find fruit**: some joints grow into eyes that see fruit ahead, every nerve
cell can feel hunger, and energy drains faster when muscles work. Written in JAX, so a whole population
is grown and simulated in parallel on a GPU.

**Watch it in a browser:** `web/neurogenesis-forager.html` is the same simulation in JavaScript, with
evolution running in background workers while the best creature so far looks for fruit. It also has a
genome bench: every one of the genome's numbers as a heatmap you can click and edit, a dial per channel,
and a Grow button that grows and runs your edited genome. Open it in any browser.
`web/neurogenesis-thrower.html` and `web/neurogenesis-balltoss.html` are the same for the throwing and
five-ball tasks (see below).
(`web/neurogenesis-walker.html` is the earlier walking version; its genomes don't fit the current code.)

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
python render.py runs/first/best.npy --out creature.gif          # watch it grow and look for fruit
python render.py runs/first/best.npy --world 7                   # in another world (fruit elsewhere)
python evolve.py --out runs/throw --task throw                   # another task: throw a ball
python render.py runs/throw/best.npy --task throw
python evolve.py --out runs/balls --task balls                   # and another: five balls, total air time
python evolve.py --out runs/balls2 --task balls --resume examples/thrower.npy   # starting from a thrower
python render.py runs/balls/best.npy --task balls
```

`render.py` draws bones as pale, rounded limbs and fills in solid patches of bone. A muscle joint is a
dark dot, with its muscle drawn as a red band on the side it pulls; the band darkens and thickens as it
contracts, and stays dull if no nerve reaches the joint. An eye is a white eyeball looking away from the
body's middle. On top is the nervous system: a small violet dot for each nerve cell, at its place inside
its joint, violet fibres between wired nerve cells, and a green flash when a nerve cell fires. Fruit are
orange dots on the ground, with an arrow at the edge for fruit off screen, and the bar at the top right
is the creature's energy.

`evolve.py` prints one line per generation (scores) and writes `best.npy`, `mean.npy`, `log.csv`
and `config.json` into the run folder. Useful options: `--pop`, `--gens`, `--seconds` (length of each
life), `--sigma` (size of the random changes), `--lr`, `--seed` (which random growth rule it starts
from), `--init-scale` (how big that rule's weights are: bigger grows bigger random bodies). A genome copied out of the browser
version can be drawn here too; the page shows how.

## How it works

The code is in five files, each with notes at the top:
- **`creature.py`**: the growth rule, and how a grown grid becomes a skeleton and a nervous system
- **`sim.py`**: physics, the nervous system, senses, and the food task and its score
- **`throw.py`**: the throwing task
- **`balls.py`**: the five-ball task
- **`evolve.py`**: the random start and evolution

### The genome
A flat list of 2,344 numbers: a wiring bias for each of the 8 neighbour directions, then the weights of
the growth rule, a tiny neural network (54 inputs → 32 hidden units → 18 outputs, plus the hidden units'
biases). There are no DNA letters.

### Growth (`creature.py`)
Growth runs on a 30 × 42 grid of fine cells, and each cell holds 18 numbers. It starts with one living
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
| 1, 2, 12 | the patch's joint: bone (rigid), muscle (bends) or eye (rigid, sees), whichever its cells add up to most of |
| 11 | for a muscle joint: which way it bends, and how far (averaged over the patch) |
| 3 | for each fine cell: above 0, a nerve cell grows there |
| 4–5 | the nerve cell's spiking threshold and bias |
| 6–7 | how strongly it feels the pacemaker rhythm and touch |
| 13–14 | how strongly it feels sight (in an eye) and hunger |
| 8–9 | how strongly it sends to and receives from neighbouring nerve cells |
| 10 | how hard it drives the joint whose patch it sits in |
| 15–17 | no fixed meaning; cells use them to signal each other while growing |

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
- **Damping.** Each bone damps the relative motion of its two ends. Without it, a floppy body made of
  nothing but muscle joints built up a jiggle under its own weight, and friction turned the jiggle into
  sliding: evolution found a tangle of limbs that slid 9 m in 20 s with every muscle switched off.
- **No free turning.** Bones and joints are forces inside the body, so they can't turn it as a whole.
  But when a joint can't reach its angle because the rest of the body won't let it, each pass over the
  bones and joints went round a small loop of shapes and came back turned a little, every substep.
  Evolution found this free motor at once: a 10-joint body held its muscles still and rolled along at up
  to 6 m/s, eating every fruit. Each pass's turn of the whole body is now taken back out; hand-made legs
  walk just as far as before.
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

**Senses** are currents into the nerve cells that can feel them, each nerve cell with its own strength:
- **touch:** nerve cells in the patch of a joint touching the ground
- **pacemaker:** a steady rhythm, sin(2π · 1.5 · t), felt by every nerve cell, like the pacemaker cells
  that keep a heart beating
- **sight:** nerve cells in an eye. Each eye looks away from the body's middle and sees fruit ahead of
  it, nearer fruit brighter (1.5 m makes it 1/e as bright), so an eye on each end tells left from right
- **hunger:** 1 − energy, felt by every nerve cell

### The world and the score
- Before the clock starts, each creature gets a tiny sideways nudge and 3 s to settle with its muscles
  off, so a body that would topple over has already done so.
- Four fruit then lie beyond its left or right end, farther away than the creature is tall plus 0.2–1.2 m,
  so falling over can't reach one. A joint near the ground within 8 cm of a fruit eats it, and a new fruit
  appears 0.5–2.5 m beyond a random end. Fruit more than 6 m away is moved back near.
- Energy starts full and drains with time: a full stomach lasts 12 s, less for bodies over 20 joints and
  for busy muscles (working every muscle flat out drains 1.25 times as fast). A fruit gives 0.4. At zero
  energy the creature dies where it lies.
- **Score = fruit eaten + the fraction of the 20 s it stayed alive + 0.25 per metre travelled (at most
  1).** The distance part gives evolution a first step: moving at all, before it ever reaches a fruit.
- A world is a number. Fruit positions and the nudge come from an integer hash of the world number, the
  same in Python and JavaScript, so a world looks the same in both. A body of fewer than 3 joints scores 0.

### The throwing task (`throw.py`, `--task throw`)
A ball 10 cm across, half a joint's weight. After the creature settles, the ball is dropped onto the top
of its body (above which joint depends on the world number) and given a second to settle too. The
creature has no hands, so it has to fling the ball off with its body.
- **Score: the longest throw in 10 s**: how far the ball travels through the air (more than 1 cm off the
  ground) after it last left the creature, in either direction. Carrying it while walking earns nothing,
  and so does rolling, kicking or shoving it along the ground.
- **Why the ball starts on the creature.** With the ball on the ground and ground distance counting at
  all, evolution's best "throwers" kicked it along the ground at 7 m/s and never lifted it. Counting only
  flight stopped that, but then nothing scored at all, since random creatures never lift a ball, and
  evolution had nothing to start from. On the creature's back, even a ball that rolls off flies a little,
  and flinging it sends it farther. Picking the ball up off the ground could come back once throwers exist.
- **No free speed.** Joints hold their angles with unlimited force, so pressing the ball into the ground
  or pinching it between joints could shoot it out at any speed. The ball can leave the joints it touches
  at most twice as fast as the fastest of them moved (a light ball hit by a heavy limb), and never faster
  than 8 m/s. It slides to a stop on the ground.
- Senses: touch includes touching the ball, eyes see the ball, and hunger stays at 0.

### The five-ball task (`balls.py`, `--task balls`)
One ball is dropped onto the creature's back, as in the throwing task, and four more lie on the ground
0.14–1.44 m beyond either end. Lives are 30 s.
- **Score: each ball's longest time in the air** (more than 1 cm up) after it last left the creature,
  added up over the five balls, in seconds, plus 0.1 for each ball on the ground the creature reaches.
  Each ball counts only its best throw, so juggling one ball over and over can't add up: to score more,
  the creature has to get to another ball and get it up.
- Why the credit for reaching: without it, four runs of 60 generations only ever threw the ball on their
  back, since reaching a ball earned nothing until it was in the air.
- The ball on its back is the first step, for the same reason as in the throwing task: no random
  creature ever lifts a ball off the ground.
- Balls pass through each other. Eyes see all the balls ahead of them, added up like fruit, and touch
  includes touching any ball.

### Evolution (`evolve.py`)
1. **A random start.** Evolution starts from a single cell and a random growth rule. About half of random
   rules grow nothing, so rules are drawn until one grows a small body of 3 to 30 joints with nerves
   reaching at least two of its muscle joints: usually a lump with a few dozen nerve cells that twitches
   at random.
2. **OpenAI-ES.** Each generation adds small random noise to the genome to make the population, grows
   and runs every member in one world, and moves the genome toward the noise of the members that scored
   best. Noise comes in mirrored pairs (+ε and −ε), and scores are converted to ranks. Each generation's
   best is re-tested in 4 fixed worlds (1000–1003), and the best of those is saved as `best.npy`.
3. **Jumping to champions.** When a generation's best beats the best so far on that re-test, the search
   jumps to it and carries on from there. From a random start, good creatures are rare lucky finds, and
   one lucky member barely moves the population's average genome: without the jump, runs stalled.

Growth rules are sensitive: noise of 0.05 can wipe a body out. From a random start, `--sigma` 0.02 found
walkers fastest in the walking version, so that is the default.

## Results so far

I could only test on CPUs, using the browser version's code (which matches the Python).

Food version (runs of 64 creatures per generation, 120 generations, 20 s lives):
- From random starts that only had to grow 3 to 30 joints, three runs never ate a fruit: the best moved
  0.3–0.4 m once and then sat still. Two of the three starts had zero or one muscle joint, so they could
  hardly move whatever the score rewarded. Evolution now starts from a random rule whose body also has
  nerves reaching at least two muscle joints.
- With that start, one run of three found a forager at generation 6, shown below; the others crept
  0.3–0.4 m. It's worth trying a few `--seed`s, and a GPU can afford far bigger populations.
- The forager: 12 joints (4 of them muscle), 10 nerve cells, 18 synapses, no eyes. In 20 worlds it was
  never tested on, it eats 2.15 fruit on average and stays alive 91% of the time (resting creatures
  manage 60%). Switching parts off shows how it works: with its muscles off it doesn't move, blinding it
  changes nothing (it has no eyes), and **with hunger switched off it never moves at all**. It rests
  until it gets hungry, walks, and eats the fruit lying in its path; eating calms it down. That's a real,
  if simple, behaviour, a bit like a worm that roams when it's hungry. Draw it with
  `python render.py examples/forager.npy`, or load it in the browser page ("Load an evolved forager").

Throwing task (three runs of 64 creatures per generation, 100 generations, 10 s lives): all three
learned to throw. Their bodies grew from 3–30 joints to 80–130, almost all muscle, and they fling the
ball by whipping the whole body, timed by the pacemaker, with limb tips reaching 5–8 m/s. Their best
throws in the test worlds were 3.7, 4.0 and 5.6 m; over 20 worlds they had never seen, they averaged
0.9, 2.5 and 1.4 m. With their muscles switched off, all three score 0. The 2.5 m one also uses its
eyes (blinded, it averages 2.0 m). It's `examples/thrower.npy`
(`python render.py examples/thrower.npy --task throw`), and the browser page can load it. Since muscles
cost nothing and have unlimited force here, long chains of muscle make very good whips; a cost for muscle
work would push toward more economical throws.

Five-ball task (runs of 64 creatures per generation, 60 generations, 30 s lives):
- Without credit for reaching balls, four runs (three from random starts, one from the thrower) only
  ever flung the ball on their back, 0.6–1.2 s in the air, and barely moved (0.1–0.25 m).
- With 0.1 for each ground ball reached, both runs from random starts learned to crawl 2.5–3 m in 30 s
  and touch 2–4 of the four ground balls, still flinging the ball on their back. One briefly got two
  ground balls up (0.13 s each, a couple of centimetres) at generation 45, then traded it away. None yet
  reliably gets a ball up off the ground: that's the gap a GPU-sized run has to close. A run starting
  from the thrower lost even its crawl and reached at most one ball.
- The best: 140 joints (116 muscle, 24 eyes), crawling (its body turns less than 30°) with a rhythm
  from the pacemaker (without it, it crawls 0.5 m instead of 2.3 m). It's `examples/ball_chaser.npy`
  (`python render.py examples/ball_chaser.npy --task balls`), and the browser page can load it.

Walking version (before food, the score was distance walked right):
- About half of random growth rules grow a body; small ones are lumps that twitch.
- Without jumping to champions, three runs from random starts stalled near zero for 50 generations.
- With the jump, two of three runs of 64 creatures per generation found walkers within 25 to 35
  generations. One grew from a 6-joint lump into a 29-joint creature with 66 nerve cells that rows
  itself along with a limb at the back, a steady 0.69 m every 2 s (6.8 m in 20 s), and doesn't move at
  all with its muscles switched off. The third run was still creeping (0.2 m) after 15 generations, so
  it's worth trying a few `--seed`s.
- Replayed after the fix for the free solver motor (see **No free turning**), that walker still walks
  6.4 m, and another 12 m walker still walks 10 m. A third, 21 m walker turned out to depend on the motor:
  with the fix it falls over.

On a GPU, `benchmark.py` measures how many creatures per generation your card can test. My estimate for
an RTX 4080, not measured: 4,096 creatures per generation in roughly 10–20 s, so about 200–400 creatures
per second.

The JavaScript port was checked against this Python code: the same genomes grow identical bodies, nerve
cells and skeletons, see the same fruit in the same places, eat the same fruit and get the same scores.
In the throwing task they agree to within a fraction of a millimetre, except where a flailing body's
chaos amplifies the tiny rounding differences between the two (one test throw of twelve: 0.51 vs 0.62 m).

## Where is the brain?

There is no central brain unless evolution builds one. Wiring is purely local (a nerve cell connects only
to the nerve cells touching it), and walking can be done by local rhythms, the way a jellyfish's nerve
net or a worm's spinal circuits work. A brain pays when information from far apart has to be combined:
with an eye on each end, turning toward food means comparing the two sides, which needs a long nerve or a
hub where nerves meet. Making spikes or nerve cells cost energy would push the same way.

Without learning during a life (the synapse weights are fixed at birth), the best to hope for is reactive
and rhythmic behaviour: walking, and steering toward what the eyes see, roughly what the 302 neurons of
the worm *C. elegans* manage. Bigger runs on a GPU are the way to find out how far it gets.

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
- Synapses that change during a life (Hebbian plasticity), so a creature can learn, not only react.
- A bigger grid and more growth steps, for bigger bodies and bigger brains.

Earlier versions are in the git history: soft bodies with a neuron in every cell and fruit to find, the
same soft bodies with a grown nervous system, jointed skeletons with a neuron in every cell, and the
jointed walker with fine nerves that this food version grew from.
