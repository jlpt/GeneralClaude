# Nworks

An artificial-life sandbox. Small creatures with **spiking neural-network brains**
evolve by natural selection in a 2D world. There is no fitness function:
creatures that find food and air live long enough to reproduce, and their
children inherit a mutated copy of their brain.

## How it works

**Survival.** Each creature loses 1 health and 1 air per step.
- **Food** (red) refills health. A creature can go 5,000 steps without it.
- **Air pockets** (blue) refill air. A creature can go 10,000 steps without them.
- A creature that eats 3 food has a child.
- If the population drops below 15, random creatures are added.

**Senses and body.** Each creature gets 12 inputs and 4 outputs:
- 5 inputs for food direction: left, right, up, down and "none in range".
- 5 inputs for air-pocket direction.
- 2 urgency inputs that reach 1 as health or air runs low.
- 4 outputs that push it right, left, down and up.

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

**Curriculum.** As generations pass, new food spawns further left and new air
pockets further right. Eventually the gap between them is wider than the
creatures can sense, so they have to head toward air they can't see yet.

Every number above is a setting in [`Nworks/config.h`](Nworks/config.h).
Three optional features are off by default:
- sensing other creatures
- sensing your own position
- "slices": lines creatures can cut that kill whatever they touch

## Window app controls

| Control | Action |
|---|---|
| **W A S D** | Pan the camera |
| **Click** a creature | Show its brain and stats on the right; brain size is printed to the console |
| **Hover** a creature | Tooltip |
| **Target FPS** slider | Frame-rate cap |
| **Steps per frame** slider | Run evolution up to 100× faster while watching |

When nothing is selected, the creature with the highest generation glows, and its brain is shown.

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

Without drawing, the simulation runs at about 60,000 steps per second on one core:

```sh
Nworks.exe --headless --steps 1000000 --seed 1   # Visual Studio build
build/nworks_headless --steps 1000000 --seed 1   # CMake build
```

Other options:
- `--report N` prints a line every N steps.
- `--organism-sensor`, `--position-sensor` and `--slices` turn on the optional features.

Each line shows:
- the population and the highest generation alive
- average neurons and synapses per creature
- totals for births, random spawns and deaths
- food and air left in the world

A fixed `--seed` repeats a run exactly.

## Code layout

| File | What it does |
|---|---|
| `config.h` | Every tunable setting |
| `neuron.h`, `neural_network.*` | Spiking neurons, network stepping, mutations |
| `module.h`, `sensors.*`, `actuators.*` | Body parts that connect a brain to the world |
| `organism.*` | A creature: body state, brain, modules, reproduction |
| `world.*` | Resources, slices, one simulation step, spawning and the curriculum |
| `random.*` | Shared, seedable random number generator |
| `headless.*`, `headless_main.cpp` | Window-free runner |
| `main.cpp`, `render.*`, `slider.h` | raylib window app (the only code that draws) |
