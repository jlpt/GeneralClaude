"""Growing a creature: one cellular automaton grows a jointed body and, inside it, a nervous system.

Every cell of a fine grid carries C numbers (its state). The genome is the growth rule: a tiny neural
network that every cell applies on every step, reading its own state and its 8 neighbours' states and
writing a change to its own. Growth starts from a single living cell. The body is built from 3 × 3
patches of fine cells, so nerve cells are a third the size of the body's joints and hundreds of them fit
in a body. When growth ends, the cells' numbers are read off as:

  - for each 3 × 3 patch: whether it is part of the body (at least 5 of its 9 cells alive), and which
    tissue it is: bone (a rigid joint), muscle (a joint that bends) or eye (a rigid joint that sees
    food), and for a muscle which way and how far it bends
  - for each fine cell: whether a nerve cell grows there, and if so its spiking threshold and bias, how
    strongly it feels touch, the pacemaker rhythm, sight (in an eye) and hunger, how strongly it sends to
    and receives from neighbouring nerve cells, and how hard it drives the joint whose patch it sits in

Every patch in the body is a joint of the skeleton, joined by a bone to its parent: the neighbouring
patch one step nearer the seed. So the bones form a tree branching out from the seed, and a thin line of
patches is a limb whose length and joints the growth rule decides.

Only nerve cells are neurons, and a signal can only travel from a nerve cell to a nerve cell touching it.
So a muscle joint bends only if a nerve reaches it and fires, a foot's touch or an eye's sight matters
only if a nerve carries it away, and a brain is simply a place where many nerve cells cluster and connect. Nothing says
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

H, W = 10, 14            # the body's grid of joints (rows, columns); row 0 is the top
N = H * W
FINE = 3                 # each joint is a FINE × FINE patch of the grid that grows
HF, WF = H * FINE, W * FINE
NF = HF * WF             # fine cells: any of them can grow a nerve cell
C = 18                   # numbers each cell carries
HIDDEN = 32              # size of the growth rule's hidden layer
GROW_STEPS = 40
SEED = (4, 7)            # the joint the creature grows from
SEED_FINE = (SEED[0] * FINE + FINE // 2, SEED[1] * FINE + FINE // 2)   # the first living cell

# What each channel means once growth has finished.
ALIVE = 0
BONE, MUSCLE = 1, 2                          # tissue: the largest of these and EYE wins
NERVE = 3                                    # above 0: a nerve cell grows here too
THRESH, BIAS = 4, 5                          # the nerve cell's spiking settings
PACE, TOUCH = 6, 7                           # how strongly it feels the pacemaker and touch
SEND, RECV = 8, 9                            # wiring: strength out to / in from neighbouring nerve cells
JUNCTION = 10                                # how hard it drives the joint it sits in
BEND = 11                                    # a muscle joint: which way it bends, and how far
EYE = 12                                     # tissue: an eye, a rigid joint that sees food
SIGHT, HUNGER = 13, 14                       # how strongly a nerve cell feels sight (in an eye) and hunger
FREE = 15                                    # channels 15–17 have no fixed meaning: cells use them to
                                             # signal each other while growing

TISSUES = ('bone', 'muscle', 'eye')
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right: the order a cell picks its parent in


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
    return {(dy, dx): p[1 + dy:1 + dy + HF, 1 + dx:1 + dx + WF] for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


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
    s = jnp.zeros((HF, WF, C))
    s = s.at[SEED_FINE[0], SEED_FINE[1], ALIVE].set(1.0)
    return s.at[SEED_FINE[0], SEED_FINE[1], FREE:].set(1.0)


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
    alive: jnp.ndarray     # (N,) bool: this joint is part of the creature
    tissue: jnp.ndarray    # (N,) int: 0 bone, 1 muscle, 2 eye
    parent: jnp.ndarray    # (N,) int: the neighbouring joint one step nearer the seed (the seed is its own parent)
    bend: jnp.ndarray      # (N,) −1 to 1: which way and how far a muscle bends its joint
    nerve: jnp.ndarray     # (HF, WF) bool: a nerve cell grows in this fine cell


class Brain(NamedTuple):
    weights: jnp.ndarray   # (HF, WF, 8): synapse from each nerve cell to its neighbour in each of the 8 directions
    threshold: jnp.ndarray # (HF, WF)
    bias: jnp.ndarray      # (HF, WF)
    gains: jnp.ndarray     # (HF, WF, 4): response to the pacemaker, touch, sight and hunger
    junction: jnp.ndarray  # (HF, WF) 0 to 1: how hard a nerve cell drives the joint whose patch it sits in


def patches(x):
    """(HF, WF, ...) → (H, W, FINE * FINE, ...): the fine cells of each joint's patch."""
    rest = x.shape[2:]
    x = x.reshape((H, FINE, W, FINE) + rest)
    return jnp.moveaxis(x, 2, 1).reshape((H, W, FINE * FINE) + rest)


def spread(x):
    """(H, W, ...) → (HF, WF, ...): each joint's value copied to every fine cell of its patch."""
    return jnp.repeat(jnp.repeat(x, FINE, axis=0), FINE, axis=1)


def read_body(s):
    alive_fine = s[..., ALIVE] > 0.1
    alive = jnp.sum(patches(alive_fine), axis=-1) >= 5          # (H, W): a joint needs 5 of its 9 cells
    # Steps from the seed to every living joint, moving side by side (not corner to corner). Joints that
    # can't be reached are not part of the creature.
    far = N + 1
    start = jnp.where(alive & (jnp.zeros((H, W), bool).at[SEED].set(True)), 0, far)

    def neighbours(d):
        p = jnp.pad(d, 1, constant_values=far)
        return [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in DIRS]

    def step(_, d):
        return jnp.where(alive, jnp.minimum(d, jnp.min(jnp.stack(neighbours(d)), axis=0) + 1), far)

    steps = jax.lax.fori_loop(0, N, step, start)
    connected = steps < far
    # Each joint's parent is its first neighbour (up, down, left, right) one step nearer the seed.
    nearer = jnp.stack([n == steps - 1 for n in neighbours(steps)])          # (4, H, W)
    shift = jnp.array([dy * W + dx for dy, dx in DIRS])[jnp.argmax(nearer, axis=0)]
    index = jnp.arange(N).reshape(H, W)
    parent = jnp.where(connected & (steps > 0), index + shift, index)
    tissue = jnp.argmax(jnp.sum(patches(s[..., [BONE, MUSCLE, EYE]]), axis=2), axis=-1)
    bend = jnp.tanh(jnp.mean(patches(s[..., BEND]), axis=-1))
    nerve = alive_fine & (s[..., NERVE] > 0) & spread(connected)
    return Body(alive=connected.reshape(N), tissue=tissue.reshape(N), parent=parent.reshape(N),
                bend=bend.reshape(N), nerve=nerve)


def _neighbour(x, dy, dx):
    """Each cell's neighbour at (dy, dx), with 0 beyond the grid edge."""
    p = jnp.pad(x, ((1, 1), (1, 1)))
    return p[1 + dy:1 + dy + HF, 1 + dx:1 + dx + WF]


def read_brain(s, body, params):
    m = body.nerve.astype(jnp.float32)
    send, recv = s[..., SEND], s[..., RECV]
    # synapses only join nerve cells that touch
    weights = jnp.stack([1.5 * jnp.tanh(send * _neighbour(recv, dy, dx) + params['K'][o]) * m * _neighbour(m, dy, dx)
                         for o, (dy, dx) in enumerate(OFFSETS)], axis=-1)
    return Brain(
        weights=weights,
        threshold=0.5 + jax.nn.sigmoid(s[..., THRESH]),
        bias=0.3 * jnp.tanh(s[..., BIAS]) * m,
        gains=1.5 * jnp.tanh(s[..., [PACE, TOUCH, SIGHT, HUNGER]]) * m[..., None],
        junction=jax.nn.sigmoid(2 * s[..., JUNCTION]) * m,
    )


def synapse_input(brain, spikes):
    """The current each nerve cell gets from the spikes of the nerve cells touching it."""
    total = jnp.zeros((HF, WF))
    for o, (dy, dx) in enumerate(OFFSETS):
        out = jnp.pad(spikes * brain.weights[..., o], 1)       # sent toward the neighbour at (dy, dx)
        total = total + out[1 - dy:1 - dy + HF, 1 - dx:1 - dx + WF]
    return total


def develop(genome):
    """Genome (a flat array of numbers) → grown body and nervous system."""
    params = unflatten(genome)
    s = grow(params)
    body = read_body(s)
    return body, read_brain(s, body, params)


def describe(body):
    """A text picture of a body: . empty, B bone, M muscle joint, E eye; lower case where a nerve runs."""
    alive = np.asarray(body.alive).reshape(H, W)
    tissue = np.asarray(body.tissue).reshape(H, W)
    nerves = np.asarray(patches(body.nerve)).sum(-1)
    letter = lambda r, c: ('bme' if nerves[r, c] else 'BME')[tissue[r, c]] if alive[r, c] else '.'
    return '\n'.join(''.join(letter(r, c) for c in range(W)) for r in range(H))


def describe_nerves(body):
    """The fine grid: . empty, o a nerve cell."""
    nerve = np.asarray(body.nerve)
    return '\n'.join(''.join('o' if nerve[r, c] else '.' for c in range(WF)) for r in range(HF))
