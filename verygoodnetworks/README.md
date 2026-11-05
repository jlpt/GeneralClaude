# Nworks

An artificial-life sandbox. Small creatures with **spiking neural-network brains**
evolve by natural selection in a 2D world. There is no fitness function:
creatures that find food and air live long enough to reproduce, and their
children inherit a mutated copy of their brain.

The world is built so that survival rewards memory and planning, and brains
cost energy. A brain only grows if the extra neurons pay for themselves.

## How it works

**Survival.** Each creature loses 1 health and 1 air per step, plus an energy
cost for its brain: 0.002 per neuron and 0.0002 per synapse. The starting brain
adds about 25%.
- **Food** (red) refills health.
- **Springs** (dark blue) refill air.
- A creature that eats 3 food has a child.

**Springs.** Three fixed air sources, placed further apart than a creature can see.
- Each holds a limited amount of air and refills slowly. The bright centre shrinks as it drains.
- A crowd can empty one, so creatures do better if they remember where the others are.

**Seasons.** Food blooms in one of three green patches, and the bloom moves to the next patch every 8,000 steps.
- Uneaten food rots after 6,000 steps.
- Every patch is out of sight of every spring, so creatures have to travel between them without seeing where they're going.

**Curriculum.** A difficulty level (0–100%) sets how much food grows in the
blooming patch versus scattered anywhere.
- It rises every 20,000 steps in which the population sustains itself.
- It falls when the population collapses.

**Seed bank.** If the population drops below 15, it's topped up, mostly
with mutated copies of the last 50 creatures that reproduced (80%), plus some
brand-new random brains (20%). A crash doesn't throw away what evolution has found.

**Senses and body.** Creatures see in 8 directions. For each direction, three inputs report how close the nearest food,
air source and other creature is, within 300 px. Two more inputs report hunger
and breathlessness. Four outputs push the creature right, left, down and up.

**Brains.** Brains are recurrent networks of leaky integrate-and-fire neurons:
- Input neurons fire at a rate proportional to their sensor value.
- Hidden neurons leak, add up the weights of incoming spikes, and fire when they reach their threshold. They then reset and stay silent for a few steps.
- Output neurons smooth the spikes they receive into a 0–1 level that drives movement.
- Each synapse takes 4 bytes: a 24-bit target index and an 8-bit weight.

**Evolution.** Each child gets about 2% of its parent's synapses mutated. A mutation can:
- tweak a weight or a threshold
- add or remove a synapse
- add a neuron, by splitting an existing synapse so the old pathway keeps working
- remove a neuron

The mutation rate is inherited and drifts, so it evolves too. One birth in 250
is a "macro mutation", 15× stronger, which also shifts the creature's colour.
Related creatures therefore look alike.

Every number above is a setting in [`Nworks/config.h`](Nworks/config.h). To get
the original world back, set the two brain costs to 0 and turn off `vision`,
`springs` and `seasons`. Without seasons, the curriculum moves food left and air
pockets right, as before.

Optional extras, off by default:
- the original nearest-creature sensor
- a position sensor
- "slices": lines creatures can cut that kill whatever they touch

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
The bottom right shows difficulty, season and the biggest brain so far.

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

Without drawing, the simulation runs at about 20,000–50,000 steps per second on
one core. The speed depends on population size.

```sh
Nworks.exe --headless --steps 8000000 --report 1000000 --seed 1   # Visual Studio build
build/nworks_headless --steps 8000000 --report 1000000 --seed 1   # CMake build
```

Options:
- `--free-brains`, `--no-vision`, `--no-springs` and `--no-seasons` turn the new features off, for comparison.
- `--organism-sensor`, `--position-sensor` and `--slices` turn the extras on.
- A fixed `--seed` repeats a run exactly.

Each line shows:
- the population, the highest generation alive and the difficulty
- the average and biggest brain (neurons and synapses)
- totals for births, refills and deaths
- the food in the world

Example (seed 1, 8M steps, about 6 minutes):

```
       step alive max gen  diff | neurons biggest synapses  biggest |   births  spawned   deaths  food
    1000000    43     231   67% |    76.1      77    579.0      581 |     6075      375     6407    17
    4000000    37    1381   80% |    92.3      93    492.6      496 |    28696     1283    29942    18
    8000000    47    2774   97% |   114.0     114    474.3      476 |    58799     2536    61288     2
```

## Code layout

| File | What it does |
|---|---|
| `config.h` | Every tunable setting |
| `neuron.h`, `neural_network.*` | Spiking neurons, network stepping, mutations |
| `module.h`, `sensors.*`, `actuators.*` | Body parts that connect a brain to the world (vision, vitals, movement, ...) |
| `organism.*` | A creature: body state, brain, modules, reproduction |
| `world.*` | Food, springs, seasons, curriculum, seed bank, one simulation step |
| `random.*` | Shared, seedable random number generator |
| `headless.*`, `headless_main.cpp` | Window-free runner |
| `main.cpp`, `render.*`, `slider.h` | raylib window app (the only code that draws) |
