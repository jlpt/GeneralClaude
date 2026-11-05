# Nworks

An artificial-life sandbox. Small creatures with **spiking neural-network brains**
evolve by natural selection in a 2D world. There is no fitness function:
creatures that find food and air live long enough to reproduce, and their
children inherit a mutated copy of their brain.

The world is built so that survival rewards memory, cooperation, escape and
learning, and brains cost energy. A brain only grows if the extra neurons pay
for themselves.

## How it works

**Survival.** Each creature loses 1 health and 1 air per step, plus an energy
cost for its brain: 0.002 per neuron and 0.0002 per synapse.
- **Food** refills health.
- **Springs** (dark blue) refill air.
- A creature that eats 3 food has a child.

**Springs.** Three fixed air sources, placed further apart than a creature can see.
- Each holds a limited amount of air and refills slowly. The bright centre shrinks as it drains.
- A crowd can empty one, so creatures do better if they remember where the others are.

**Seasons.** Food blooms in one of three green patches, and the bloom moves to the next patch every 8,000 steps.
- Uneaten food rots after 6,000 steps.
- Every patch is out of sight of every spring, so creatures have to travel between them without seeing where they're going.

**Calls.** One output broadcasts a call. Calling costs energy, and a calling creature shows a yellow halo.
Four inputs hear the loudest nearby call from each direction. Calls start out
meaningless; evolution decides whether they turn into alarm calls, food calls or nothing.

**Big food** (gold). Each one is worth 2 food to every creature touching it. It only opens when two or more touch it at the same time.

**Predators** (dark red, with an eye showing their heading).
- They chase the nearest creature within 200 px and kill on contact, then rest.
- They arrive with the curriculum: the first at 25% difficulty, the second at 75%.
- They get faster as difficulty rises, but creatures can still outrun them.

**Poison lookalikes.** Food comes in red and purple. One colour is poisonous, and which one swaps every 10,000 steps, about once per lifetime.
- That is too often for evolution to hard-wire which colour to avoid.
- Poison damage grows with difficulty, up to 40% of health per bite.
- Bad bites always *taste* bad, through two taste inputs.

**Lifetime learning.** Synapses can change during a creature's life by a reward-modulated Hebbian rule:
- A synapse whose spike helped its target fire becomes *eligible*.
- Good food then strengthens recently eligible synapses, and poison weakens them.
- Each brain's learning rate is a gene, and it can evolve down to zero.
- What a creature learns is not inherited.

**Curriculum.** A difficulty level (0–100%) controls three things:
- how much food grows in the blooming patch versus scattered anywhere
- how many predators hunt, and how fast
- how strong poison is

Difficulty rises every 20,000 steps in which the population sustains itself, and falls when it collapses.

**Seed bank.** If the population drops below 15, it's topped up, mostly
with mutated copies of the last 50 creatures that reproduced (80%), plus some
brand-new random brains (20%). A crash doesn't throw away what evolution has found.

**Senses and body.** Creatures see in 8 directions. Each direction has an input for six kinds of thing: red food, purple food, air, other creatures, big food and predators. That makes 48 inputs, each saying how close the nearest one is, within 300 px.

On top of vision there are 4 hearing inputs, 2 taste inputs and 2 urgency inputs (hunger, breathlessness). Five outputs control movement (right, left, down, up) and calling.

**Brains.** Brains are recurrent networks of leaky integrate-and-fire neurons:
- Input neurons fire at a rate proportional to their sensor value.
- Hidden neurons leak, add up the weights of incoming spikes, and fire when they reach their threshold. They then reset and stay silent for a few steps.
- Output neurons smooth the spikes they receive into a 0–1 level.
- Each synapse takes 4 bytes: a 24-bit target index and an 8-bit weight.

**Evolution.** Each child gets about 2% of its parent's synapses mutated. A mutation can:
- tweak a weight, a threshold or the learning rate
- add or remove a synapse
- add a neuron, by splitting an existing synapse so the old pathway keeps working
- remove a neuron

The mutation rate is inherited and drifts, so it evolves too. One birth in 250
is a "macro mutation", 15× stronger, which also shifts the creature's colour.
Related creatures therefore look alike.

Every number above is a setting in [`Nworks/config.h`](Nworks/config.h), and
every feature can be switched off there. To get the original world back, set the
brain costs to 0 and turn off vision, springs, seasons and everything in stage 4.

## What the experiments show so far

These are headless runs of 8–12 million steps (5–10 minutes each).

- **The world is survivable and gets harder.** Lineages last 2,000+ generations, and difficulty climbs to 45–60% with every feature on. It reaches about 100% with stage 4 off.
- **Brains grow despite the energy cost,** from about 105 to about 140 neurons. A lesion test suggests this is mostly chance, not need. Evolved worlds were copied, and up to 50% of every creature's hidden neurons were removed. The copies with fewer neurons did as well or better. A neuron's cost is too small for selection to remove it in a population of 20–50.
- **Lifetime learning has not evolved yet.** The learning-rate gene drifts toward zero, and creatures eat poison at chance (about 50% of bites). Creatures live only about 5,000 steps and eat a handful of times, which is very little experience to learn from.
- **Big food does get opened together,** thousands of times per run, though that may be creatures crowding the same food rather than coordinating.

Ways to push further: longer runs, larger populations (more food or a bigger
world), longer lifetimes, or a stronger penalty for poison.

## Window app controls

| Control | Action |
|---|---|
| **W A S D** | Pan the camera |
| **Click** a creature | Show its brain and stats on the right; brain size is printed to the console |
| **Hover** a creature | Tooltip |
| **Target FPS** slider | Frame-rate cap |
| **Steps per frame** slider | Run evolution up to 100× faster while watching |

When nothing is selected, the creature with the highest generation glows, and its
brain is shown. The faint circle around the focused creature is its vision range.
The bottom right shows:
- difficulty
- season
- which colour is poisonous right now
- the biggest brain so far

Brain diagram key:
- **Inputs** are on the left, in grey by sensor value.
- **Outputs** are on the right.
- **Hidden neurons** go from red to green as they approach their threshold.
- **Anything that spikes** flashes yellow, along with its outgoing synapses.
- **Synapse colours** run from red (inhibitory) to green (excitatory).

## Building

### Visual Studio (Windows)

Open `Nworks.sln` and build **Release | x64**. The project looks for raylib 5.5 at
`C:\Users\kcjan\OneDrive\Desktop\c++ libraries\raylib-5.5_win64_msvc16`. If yours
is somewhere else, change *C/C++ → Additional Include Directories* and *Linker →
Additional Library Directories* in the project properties.

### CMake (Windows, macOS, Linux)

```sh
cmake -S . -B build
cmake --build build --config Release
```

If raylib 5.5 isn't installed, CMake downloads it. Add
`-DNWORKS_BUILD_WINDOW_APP=OFF` to build only the headless runner, which has no
graphics dependency.

## Headless mode

Without drawing, the simulation runs at about 10,000–50,000 steps per second on
one core. Learning and bigger populations make it slower.

```sh
Nworks.exe --headless --steps 10000000 --report 1000000 --seed 1   # Visual Studio build
build/nworks_headless --steps 10000000 --report 1000000 --seed 1   # CMake build
```

Options:
- `--stage3` turns off everything from stage 4.
- `--no-signals`, `--no-coop-food`, `--no-predators`, `--no-poison` and `--no-plasticity` turn off single stage-4 features.
- `--free-brains`, `--no-vision`, `--no-springs` and `--no-seasons` turn off the earlier features.
- `--organism-sensor`, `--position-sensor` and `--slices` turn on the extras.
- A fixed `--seed` repeats a run exactly.

Columns:

| Column | Meaning |
|---|---|
| `alive` | Population |
| `max gen` | Highest generation alive |
| `diff` | Difficulty |
| `neurons`, `synapses` | Average brain size |
| `biggest` | Largest brain alive |
| `births` | Total births so far |
| `spawned` | Total seed-bank and random refills so far |
| `poison%` | Share of bites since the last row that were poisonous; 50% means no better than chance |
| `learn` | Average learning rate |
| `shared` | Big food opened together |
| `eaten` | Killed by predators |

## Code layout

| File | What it does |
|---|---|
| `config.h` | Every tunable setting |
| `neuron.h`, `neural_network.*` | Spiking neurons, network stepping, mutations, lifetime learning |
| `module.h`, `sensors.*`, `actuators.*` | Body parts that connect a brain to the world: vision, hearing, taste, vitals, movement, calls |
| `organism.*` | A creature: body state, brain, modules, reproduction |
| `world.*` | Food, springs, seasons, big food, predators, poison, curriculum, seed bank, one simulation step |
| `random.*` | Shared, seedable random number generator |
| `headless.*`, `headless_main.cpp` | Window-free runner |
| `main.cpp`, `render.*`, `slider.h` | raylib window app (the only code that draws) |
