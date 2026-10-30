"""Growing a creature: one cellular automaton grows a jointed body and, inside it, a nervous system.

Every cell of a small grid carries C numbers (its state). The genome is the growth rule: a tiny neural
network that every cell applies on every step, reading its own state and its 8 neighbours' states and
writing a change to its own. Growth starts from a single living cell. When growth ends, each cell's
numbers are read off as:

  - alive or empty, and which tissue it is: bone (a rigid joint) or muscle (a joint that bends), and for
    a muscle which way and how far it bends
  - whether a nerve cell also grows inside it, and if so its spiking threshold and bias, how strongly
    it feels touch and the pacemaker rhythm, how strongly it sends to and receives from neighbouring
    nerve cells, and how hard it drives the joint it sits in

Every living cell is a joint of the skeleton, joined by a bone to its parent: the neighbour one step
nearer the seed. So the bones form a tree branching out from the seed, and a thin line of cells is a limb
whose length and joints the growth rule decides.

Only nerve cells are neurons, and a signal can only travel from a nerve cell to a nerve cell touching it.
So a muscle joint bends only if a nerve reaches it and fires, a foot's touch matters only if a nerve
carries it away, and a brain is simply a place where many nerve cells cluster and connect. Nothing says
where the brain goes or what connects to what: the growth rule lays it all down, the same way it lays
down bone and muscle.

Everything is a fixed-size array, which is what lets JAX grow and test thousands of creatures in
parallel on a GPU.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

H, W = 10, 14            # the grid the creature grows on (rows, columns); row 0 is the top
N = H * W                # cells: each is one joint of the skeleton, and may also hold one nerve cell
C = 16                   # numbers each cell carries
HIDDEN = 32              # size of the growth rule's hidden layer
GROW_STEPS = 32
SEED = (4, 7)            # where the first living cell is (row, column)

# What each channel means once growth has finished.
ALIVE = 0
BONE, MUSCLE = 1, 2                          # tissue: the larger of these two wins (rigid or bending joint)
NERVE = 3                                    # above 0: a nerve cell grows here too
THRESH, BIAS = 4, 5                          # the nerve cell's spiking settings
PACE, TOUCH = 6, 7                           # how strongly it feels the pacemaker and touch
SEND, RECV = 8, 9                            # wiring: strength out to / in from neighbouring nerve cells
JUNCTION = 10                                # how hard it drives the joint it sits in
BEND = 11                                    # a muscle joint: which way it bends, and how far
FREE = 12                                    # channels 12–15 have no fixed meaning: cells use them to
                                             # signal each other while growing

TISSUES = ('bone', 'muscle')
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
    return s.at[SEED[0], SEED[1], FREE:].set(1.0)


def grow(params, steps=GROW_STEPS):
    return jax.lax.fori_loop(0, steps, lambda _, s: grow_step(params, s), seed_state())


def grow_history(params, steps=GROW_STEPS):
    """Every intermediate state, for drawing the growth."""
    def step(s, _):
        s = grow_step(params, s)
        return s, s
    _, states = jax.lax.scan(step, seed_state(), None, length=steps)
    return states


# ---------- reading off the body and the nervous system ----------

class Body(NamedTuple):
    alive: jnp.ndarray     # (N,) bool: part of the creature
    tissue: jnp.ndarray    # (N,) int: 0 bone, 1 muscle
    nerve: jnp.ndarray     # (N,) bool: a nerve cell grows here too
    parent: jnp.ndarray    # (N,) int: the neighbour one step nearer the seed (the seed is its own parent)
    bend: jnp.ndarray      # (N,) −1 to 1: which way and how far a muscle bends its joint


class Brain(NamedTuple):
    weights: jnp.ndarray   # (N, N): weights[i, j] is the synapse from nerve cell i to nerve cell j
    threshold: jnp.ndarray # (N,)
    bias: jnp.ndarray      # (N,)
    gains: jnp.ndarray     # (N, 2): response to the pacemaker and to touch
    junction: jnp.ndarray  # (N,) 0 to 1: how hard a nerve cell drives the joint it sits in


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
    tissue = jnp.argmax(s[..., [BONE, MUSCLE]], axis=-1)
    nerve = connected & (s[..., NERVE] > 0)
    return Body(alive=connected.reshape(N), tissue=tissue.reshape(N), nerve=nerve.reshape(N),
                parent=parent.reshape(N), bend=jnp.tanh(s[..., BEND]).reshape(N))


def read_brain(s, body, params):
    cell = s.reshape(N, C)
    m = body.nerve.astype(jnp.float32)
    w = 1.5 * jnp.tanh(cell[PAIR_I, SEND] * cell[PAIR_J, RECV] + params['K'][PAIR_O])
    w = w * m[PAIR_I] * m[PAIR_J]                 # synapses only join nerve cells that touch
    return Brain(
        weights=jnp.zeros((N, N)).at[PAIR_I, PAIR_J].set(w),
        threshold=0.5 + jax.nn.sigmoid(cell[:, THRESH]),
        bias=0.3 * jnp.tanh(cell[:, BIAS]) * m,
        gains=1.5 * jnp.tanh(cell[:, [PACE, TOUCH]]) * m[:, None],
        junction=jax.nn.sigmoid(2 * cell[:, JUNCTION]) * m,
    )


def develop(genome):
    """Genome (a flat array of numbers) → grown body and nervous system."""
    params = unflatten(genome)
    s = grow(params)
    body = read_body(s)
    return body, read_brain(s, body, params)


def describe(body):
    """A text picture of a body: . empty, B bone, M muscle joint; lower case where a nerve runs."""
    alive = np.asarray(body.alive).reshape(H, W)
    tissue = np.asarray(body.tissue).reshape(H, W)
    nerve = np.asarray(body.nerve).reshape(H, W)
    letter = lambda r, c: ('bm' if nerve[r, c] else 'BM')[tissue[r, c]] if alive[r, c] else '.'
    return '\n'.join(''.join(letter(r, c) for c in range(W)) for r in range(H))
