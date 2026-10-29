# Neurogenesis Walker

The walking version of [`dna-neural-ca`](../dna-neural-ca). A DNA string grows a spiking neural network on a
48 × 48 grid (same genome format and growth automaton), and the network drives the hip and knee muscles of a
two-legged walker seen from the side. Evolution on the DNA selects for distance walked in 10 seconds.

Open `index.html` in a browser. It's a single file with no dependencies; training runs in a Web Worker.

## The body

Nine point masses joined by rigid bones, simulated with position-based dynamics at 60 steps a second: a box-shaped
torso with a head, and a rear and a front leg, each with a hip, knee and foot. Four muscles span the joints:

| Motor neuron | Muscle | Firing does |
|---|---|---|
| rear hip | front of torso → rear knee | swings the rear thigh forward |
| rear knee | rear hip → rear foot | folds the rear leg |
| front hip | head → front knee | swings the front thigh forward |
| front knee | front hip → front foot | folds the front leg |

A muscle is a distance constraint whose target length shrinks (by up to 30% at the hips and 25% at the knees) with
its motor neuron's recent firing rate. When the motor neuron is silent, the muscle holds the standing pose.

## Senses

Five sensor neurons, built by gene 1:

- **beat A / beat B**: a steady rhythm, once a second, that alternates between the two
- **rear foot / front foot**: whether that foot is on the ground
- **tilt**: how far the body is pitched nose-up or nose-down

Gene 2 builds the four motor neurons and gene 3 onward the interneurons, as in the forager. In this version sensors
may also connect straight to motor neurons (reflex arcs), which made training much more reliable.

## Rules that keep it honest

Evolution is very good at finding holes in a physics simulation, and the first versions were exploited:

1. **Sliding.** Simple velocity-damping friction let a creature hold still, vibrate, and slide 15 m in 10 s
   with both feet down. Fixed with Coulomb friction: a loaded point sticks unless pushed past its static limit.
2. **Buzzing.** Even with proper friction, vibrating fast while shifting weight between the feet still moved
   it 4 m. Fixed by making muscles lag their motor neuron by about 0.2 s, as real muscle does, so they can't buzz.
3. **Kneeling.** Some evolved gaits scooted along on a knee. Now anything but a foot touching the ground counts as
   a fall and ends the run.

## Results

Distance in 10 seconds on flat ground. Every run starts from the same pose, so a genome always scores the same.

| Controller | Metres |
|---|---:|
| no muscle activity (stands still) | 0.05 |
| best hand-made rhythm (each muscle on one of 4 phases of a 1 Hz sine) | 4.83 |
| evolved, 4 runs of 150 generations | 0.65, 3.05, 5.33, 6.17 |
| evolved, 4 runs of 500 generations (the best three are the examples in the page) | 3.31, 5.62, 5.88, 6.15 |

Most runs find a real stepping gait (level body, front foot lifting and swinging forward about once a second),
and the best beat the best hand-made rhythm. About 1 run in 4 gets stuck early on a gait that falls.

## Ideas for next steps

- Let the DNA build the body too: number of legs, segment lengths, muscle placement.
- Uneven ground and a push now and then, so gaits must use the foot and tilt sensors to stay up.
- Drop the beat inputs and see whether the recurrent network evolves its own rhythm (a central pattern generator).
