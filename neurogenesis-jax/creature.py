"""Growing a creature: one cellular automaton grows the body and the brain together.

Every cell of a small grid carries C numbers (its state). The genome is the growth rule: a tiny neural
network that every cell applies on every step, reading its own state and its 8 neighbours' states and
writing a change to its own. Growth starts from a single living cell. When growth ends, each cell's
numbers are read off as:

  - alive or empty, and which tissue it is: bone (a rigid joint), muscle (a joint that bends) or eye
  - for a muscle, which way and how far it bends its joint
  - its neuron: spiking threshold, bias, and how strongly it responds to the beat, touch, sight, hunger
  - its wiring: how strongly it sends to and receives from its neighbours

Every living cell becomes a joint of the skeleton, joined by a bone to its parent: the neighbour one
step nearer the seed. So the bones form a tree branching out from the seed, and a thin line of cells
is a limb. The same numbers shape the body, place its joints and muscles, and wire the brain. Everything is a fixed-size array, which is what lets
JAX grow and test thousands of creatures in parallel on a GPU.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

H, W = 10, 14            # the grid the creature grows on (rows, columns); row 0 is the top
N = H * W                # one cell, one joint of the skeleton, one neuron
C = 16                   # numbers each cell carries
HIDDEN = 32              # size of the growth rule's hidden layer
GROW_STEPS = 32
SEED = (4, 7)            # where the first living cell is (row, column)

# What each channel means once growth has finished.
ALIVE = 0
BONE, MUSCLE, EYE = 1, 2, 3                  # tissue: the largest of these three wins
THRESH, BIAS = 4, 5                          # spiking settings
BEAT, TOUCH, SIGHT, HUNGER = 6, 7, 8, 9      # how strongly the neuron responds to each sense
SEND, RECV = 10, 11                          # wiring: strength out of / into this cell
BEND = 12                                    # a muscle joint: which way it bends, and how far
# Channels 13–15 have no fixed meaning: cells use them to signal each other while growing.

TISSUES = ('bone', 'muscle', 'eye')
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right: the order a cell picks its parent in


def _neighbour_pairs():
    """Every (cell i, neighbour j, which offset) pair inside the grid, for wiring."""
    pairs = []
    for r in range(H):
        for c in range(W):
            for o, (dy, dx) in enumerate(OFFSETS):
                rr, cc = r + dy, c + dx
                if 0 <= rr < H and 0 <= cc < W:
                    pairs.append((r * W + c, rr * W + cc, o))
    return tuple(np.array(a, dtype=np.int32) for a in zip(*pairs))


PAIR_I, PAIR_J, PAIR_O = _neighbour_pairs()


# ---------- the genome ----------

def init_params(key, scale=0.1):
    """A random growth rule, plus a wiring bias for each of the 8 neighbour directions."""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'W1': jax.random.normal(k1, (3 * C, HIDDEN)) * scale,
        'b1': jnp.zeros(HIDDEN),
        'W2': jax.random.normal(k2, (HIDDEN, C)) * scale,
        'K': jax.random.normal(k3, (len(OFFSETS),)) * scale,
    }


_template = init_params(jax.random.PRNGKey(0))
_, unflatten = ravel_pytree(_template)
GENOME_SIZE = ravel_pytree(_template)[0].size


def flatten(params):
    return ravel_pytree(params)[0]


# ---------- growth ----------

def _shifted(x):
    """The state of each cell's 8 neighbours and itself, with empty space beyond the grid edge."""
    p = jnp.pad(x, ((1, 1), (1, 1), (0, 0)))
    return {(dy, dx): p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def _alive_nearby(n):
    """A cell can live if it, or any neighbour, is alive."""
    return jnp.max(jnp.stack([v[..., ALIVE] for v in n.values()]), axis=0) > 0.1


def grow_step(params, s):
    """One step of the automaton: every cell updates at once from what it and its neighbours hold."""
    n = _shifted(s)
    # Each cell senses its neighbourhood: its own state, and how the state changes left-right and up-down.
    sobel_x = (n[(-1, 1)] + 2 * n[(0, 1)] + n[(1, 1)] - n[(-1, -1)] - 2 * n[(0, -1)] - n[(1, -1)]) / 8
    sobel_y = (n[(1, -1)] + 2 * n[(1, 0)] + n[(1, 1)] - n[(-1, -1)] - 2 * n[(-1, 0)] - n[(-1, 1)]) / 8
    sensed = jnp.concatenate([s, sobel_x, sobel_y], axis=-1)
    change = jax.nn.relu(sensed @ params['W1'] + params['b1']) @ params['W2']
    s_new = jnp.clip(s + change, -3.0, 3.0)
    keep = _alive_nearby(n) & _alive_nearby(_shifted(s_new))
    return s_new * keep[..., None]


def seed_state():
    s = jnp.zeros((H, W, C))
    s = s.at[SEED[0], SEED[1], ALIVE].set(1.0)
    return s.at[SEED[0], SEED[1], 12:].set(1.0)


def grow(params, steps=GROW_STEPS):
    return jax.lax.fori_loop(0, steps, lambda _, s: grow_step(params, s), seed_state())


def grow_history(params, steps=GROW_STEPS):
    """Every intermediate state, for drawing the growth."""
    def step(s, _):
        s = grow_step(params, s)
        return s, s
    _, states = jax.lax.scan(step, seed_state(), None, length=steps)
    return states


# ---------- reading off the body and the brain ----------

class Body(NamedTuple):
    alive: jnp.ndarray     # (N,) bool: part of the creature
    tissue: jnp.ndarray    # (N,) int: 0 bone, 1 muscle, 2 eye
    parent: jnp.ndarray    # (N,) int: the neighbour one step nearer the seed (the seed is its own parent)
    bend: jnp.ndarray      # (N,) −1 to 1: which way and how far a muscle bends its joint


class Brain(NamedTuple):
    weights: jnp.ndarray   # (N, N): weights[i, j] is the synapse from cell i to cell j
    threshold: jnp.ndarray # (N,)
    bias: jnp.ndarray      # (N,)
    gains: jnp.ndarray     # (N, 4): response to beat, touch, sight, hunger


def read_body(s):
    alive = s[..., ALIVE] > 0.1
    # Steps from the seed to every living cell, moving side by side (not corner to corner). Cells that
    # can't be reached are not part of the creature.
    far = N + 1
    start = jnp.where(alive & (jnp.zeros((H, W), bool).at[SEED].set(True)), 0, far)

    def neighbours(d):
        p = jnp.pad(d, 1, constant_values=far)
        return [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in DIRS]

    def spread(_, d):
        return jnp.where(alive, jnp.minimum(d, jnp.min(jnp.stack(neighbours(d)), axis=0) + 1), far)

    steps = jax.lax.fori_loop(0, N, spread, start)
    connected = steps < far
    # Each cell's parent is its first neighbour (up, down, left, right) one step nearer the seed.
    nearer = jnp.stack([n == steps - 1 for n in neighbours(steps)])          # (4, H, W)
    shift = jnp.array([dy * W + dx for dy, dx in DIRS])[jnp.argmax(nearer, axis=0)]
    index = jnp.arange(N).reshape(H, W)
    parent = jnp.where(connected & (steps > 0), index + shift, index)
    tissue = jnp.argmax(s[..., [BONE, MUSCLE, EYE]], axis=-1)
    return Body(alive=connected.reshape(N), tissue=tissue.reshape(N), parent=parent.reshape(N),
                bend=jnp.tanh(s[..., BEND]).reshape(N))


def read_brain(s, body, params):
    cell = s.reshape(N, C)
    m = body.alive.astype(jnp.float32)
    w = 1.5 * jnp.tanh(cell[PAIR_I, SEND] * cell[PAIR_J, RECV] + params['K'][PAIR_O])
    w = w * m[PAIR_I] * m[PAIR_J]
    return Brain(
        weights=jnp.zeros((N, N)).at[PAIR_I, PAIR_J].set(w),
        threshold=0.5 + jax.nn.sigmoid(cell[:, THRESH]),
        bias=0.3 * jnp.tanh(cell[:, BIAS]) * m,
        gains=1.5 * jnp.tanh(cell[:, [BEAT, TOUCH, SIGHT, HUNGER]]) * m[:, None],
    )


def develop(genome):
    """Genome (a flat array of numbers) → grown body and brain."""
    params = unflatten(genome)
    s = grow(params)
    body = read_body(s)
    return body, read_brain(s, body, params)


def describe(body):
    """A text picture of a body: . empty, B bone, M muscle (a bending joint), E eye."""
    alive = np.asarray(body.alive).reshape(H, W)
    tissue = np.asarray(body.tissue).reshape(H, W)
    return '\n'.join(''.join('BME'[tissue[r, c]] if alive[r, c] else '.' for c in range(W)) for r in range(H))
