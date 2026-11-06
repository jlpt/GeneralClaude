# GPU network: weight-space programming

This is the CUDA + raylib spiking network from `GPU-NETWORK-GOOD`: 250,000 neurons per organism, 500 synapses each, which is 125 million synapses. It has been put through the two steps of *weight-space programming*:

1. **Read** the weights and work out what the network actually does.
2. **Write** new weights by hand, with no training, so the network does something on purpose. Here, the organism follows the mouse.

The original code is in the first commit that adds this folder. The diff after it is the whole change.

## 1. Reading the weights: what the original network does

**How it is built.** Every neuron sends 500 synapses to random neurons in the same organism. Each weight is drawn uniformly from [-1, 1] and stored as an `int8`. Each neuron has a random refractory period of 1–60 frames, and every neuron starts active. Neurons 0–15 are read as four motor groups: left, right, up and down. The organism moves by `(right - left, down - up)` pixels per frame.

**What the weights show:**

- **The network has no inputs.** Nothing from the world is ever written into it, so it cannot react to anything. It can only produce its own internal activity.
- **The motor neurons are ordinary random neurons.** Each one receives about 500 random synapses from the rest of the network, like every other neuron, and nothing about them is special.
- **Activations are effectively binary.** Every frame, each neuron receives dozens of random ±weights, so the sum is far outside [0, 1]. The clamp then pins it to exactly 0 or exactly 1. In the headless runs, about 98% of motor-group sums were whole numbers. About 50% of neurons are active at any moment, and about 5.5% fire each frame.
- **So the behaviour is a random walk.** Each motor group is the sum of 4 random on/off bits. The organism jitters around and drifts until it hits a wall. That matches `screenshot000.png`, where it sits near the bottom-left corner. In one 60-second run it was pinned to a wall 34% of the time.
- **No trained knowledge is stored in the weights.** Everything in them is random. This is the big difference from the MIT paper, which read the weights of *trained* networks.

**Hazards found while reading the code:**

1. **Race in `propagateKernel`.** A firing neuron sets its own activation to 0 *after* its loop. Other threads may `atomicAdd` into that same neuron during the same launch, and those inputs are lost. How many get lost depends on GPU thread scheduling, so runs are not reproducible. I left this unchanged, because the program below does not depend on it.
2. **24-bit destination field.** `TOTAL_NEURONS` must stay at or below 16,777,216. At 250k neurons per organism, that means at most 67 organisms. I added a `static_assert` so a bigger setting fails to compile instead of silently corrupting connections. Your 2M-neuron build in the screenshot is within the limit.
3. **Weights are never saved.** The network is seeded from `std::random_device` and nothing is written to disk. Every run is a brand-new random network, so you can't re-read last run's weights.
4. **`cudaMalloc` results are not checked.** The connection table is 500 MB in this configuration, and 8 GB in the 2M × 1000 build.
5. Minor: `(int8_t)(w * 127)` truncates toward zero, so weight 0 is twice as likely as any other value. Only about 0.8% of synapses are exactly 0, so this is harmless.

## 2. Writing the weights: the "follow the mouse" program

The random network is generated exactly as before. Then a small patch is written straight into its weights. Neurons 16–20 of each organism are taken over as inputs:

```
mouse ──► host computes how far the target is in each direction
          (0..1, where 1 means 200 px or more) and writes it into 4 sensor neurons

  neuron 16 "target is LEFT"  ──(+1.0 × 100 synapses each)──►  motor neurons 0-3   (LEFT)
  neuron 17 "target is RIGHT" ──(+1.0 × 100 synapses each)──►  motor neurons 4-7   (RIGHT)
  neuron 18 "target is UP"    ──(+1.0 × 100 synapses each)──►  motor neurons 8-11  (UP)
  neuron 19 "target is DOWN"  ──(+1.0 × 100 synapses each)──►  motor neurons 12-15 (DOWN)
  neuron 20  bias = 1.0       ──(-1.0 × 2 / 10 / 25 / 50 synapses)──► motor neuron 1st/2nd/3rd/4th of every group

  the other 249,979 neurons   ──(weight set to 0)──X  neurons 0-20
```

Each frame, motor neuron *k* of a group computes

```
a  ←  clamp(a + 100 × (sensor − threshold_k), 0, 1)        threshold = 0.02, 0.10, 0.25, 0.50
```

The result is about 1 when the target is more than 4, 20, 50 or 100 px away on that side, and 0 otherwise. A motor group therefore outputs 0–4, which sets the speed:

| distance to target (per axis) | speed |
|---|---|
| ≥ 100 px | 4 px/frame |
| 50–100 px | 3 px/frame |
| 20–50 px | 2 px/frame |
| 4–20 px | 1 px/frame |
| < 4 px | stop |

The speed is always lower than the remaining distance, so the organism never overshoots. It slows down as it gets close and stops within 4 px.

**What was edited:**

- About **13,000 of 125,000,000 synapses (≈ 0.0104%)**. Of these, about 10,500 random synapses into neurons 0–20 were set to weight 0, and about 2,500 sensor and bias synapses were rewired.
- The exact count changes a little from run to run, because the random network does. The program prints it at startup.
- The refractory periods of the same 21 neurons. Motor neurons never fire, so they act as readouts. Sensor and bias neurons can fire every frame.

The `propagateKernel` simulation step itself is untouched. The whole behaviour lives in data.

**Running it:** build as before (`build.bat` or the VS project). The program starts in programmed mode. Press **P** to swap between the programmed weights and the original random ones live. The patch stores each edited slot's original value, so switching back restores the exact original network.

**What this is and isn't.** This is the "write a network from scratch" half of weight-space programming, done by hand rather than by an LLM. It is a hand-designed 21-neuron controller placed inside the weights. The other 249,979 neurons keep running, but they are cut off from the motors and contribute nothing. Nothing was learned.

## 3. Results (headless test, CPU)

These runs used the same mouse path for both versions: five positions, 300 frames (5 s) each.

| | original weights (2 runs) | programmed weights |
|---|---|---|
| mean distance to mouse | 347 px / 343 px | 74 px (most of that is travel time) |
| got within 10 px of the mouse | never (0 of 10 positions) | every time (5 of 5), after 77–140 frames (1.3–2.3 s) |
| distance when the mouse moved on | 143–605 px | 2.8–4.2 px |

In the P-toggle run (P pressed at frames 600 and 1200), the organism went straight back to random jitter after the first press. It never reached the next two positions. After the second press it resumed following and reached its target in 60 frames.

## Headless test harness (`test/`)

There is no GPU here, so `test/run_headless.sh` compiles `kernel.cu` with g++ on the CPU:

- `shim/cuda_runtime.h` runs kernels one thread at a time.
- `shim/raylib.h` replaces the window with a scripted mouse path and writes a per-frame CSV.

```bash
SHIM_FRAMES=1500 SHIM_TARGETS="150,120;650,480;650,120;150,480;400,300" test/run_headless.sh CudaRuntime5/kernel.cu programmed.csv
```

```bash
python3 test/summarize.py programmed.csv
```

Add `SHIM_KEYS="600:P,1200:P"` to press P on given frames. A full 250k-neuron run takes about 30 s and 1 GB of RAM.

Caveat: kernels run sequentially here, so the reservoir's race condition (hazard 1) plays out differently than on a GPU. The programmed sensor → motor circuit gives the same result in any order. It has not yet been run on a real GPU.
