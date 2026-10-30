"""The world: a side view with flat ground and fruit lying on it.

Body physics. Each living cell is a square voxel; its four corners are point masses. Neighbouring
voxels share corners. Every voxel edge and diagonal is a distance constraint, solved with position-
based dynamics. The constraints are split into 8 batches in which no two share a corner, so each batch
is solved fully in parallel (good for a GPU) and later batches see earlier fixes, which converges fast.
A muscle voxel shrinks its edges by up to MUSCLE_AMP as its neuron fires. The ground pushes points up, and Coulomb friction makes a
loaded point stick unless it is pushed hard, so to travel a creature has to lift and push.

Brain. One leaky integrate-and-fire neuron per cell, wired to its neighbours. Each tick:
    charge = LEAK * charge + bias + senses + spikes @ weights
and a neuron spikes (and resets) when its charge reaches its threshold. A muscle follows its own
neuron's recent firing rate, with about a 0.2 s lag, as real muscle does.

Senses, as currents into each cell's neuron (scaled by that cell's gain for the sense):
    beat    a steady rhythm, sin(2π · BEAT_HZ · t), into every cell
    touch   1 if the cell touches the ground
    sight   eye cells only: food colour seen in the direction the eye faces (away from the body's
            middle), nearer fruit counting more. This is the same as reading a strip of pixels ahead
            where fruit pixels are one colour, computed straight from the fruit positions.
    hunger  1 − energy, into every cell

Survival. Before the clock starts, the creature settles for SETTLE_SECONDS with its muscles off, so a
body that would topple over has already done so (otherwise evolution grows tall bodies that fall onto
fruit). Then energy starts at 1 and drains every tick, faster for bigger bodies and while muscles work;
each fruit eaten adds FRUIT_ENERGY. At zero the creature dies and stops. Eaten fruit regrows nearby.

Score. Fruit eaten, plus the fraction of the time it stayed alive, plus a little for distance travelled
(so early creatures that merely move beat ones that don't, before any can reach fruit).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from creature import H, W, N, Body, Brain

# body physics
VOXEL = 0.1                 # metres per cell
DT = 1 / 30                 # seconds per tick (brain and physics)
GRAVITY = 9.8
DAMPING = 0.99
SUBSTEPS = 3                # physics steps per tick
ITERATIONS = 4              # passes over all 8 constraint batches per substep
MUSCLE_AMP = 0.3            # a fully active muscle voxel shrinks its edges by 30%
MUSCLE_RESPONSE = 0.15      # fraction of the way a muscle moves toward its drive each tick
STATIC_FRICTION, KINETIC_FRICTION = 1.2, 0.8

# brain and senses
LEAK = 0.8
BEAT_HZ = 1.5
SIGHT_RANGE = 1.5           # metres at which fruit looks 1/e as bright

# survival
FRUIT = 4                   # fruit lying in the world at any time
FRUIT_ENERGY = 0.4
EAT_REACH = 0.08            # metres: a body point this close to a fruit (and near the ground) eats it
IDLE_LIFE = 12.0            # seconds an idle starter-sized creature survives on a full stomach
STARTER_CELLS = 27          # bigger bodies burn energy faster in proportion (half of the drain scales with size)
MOVE_COST = 0.5             # working every muscle flat out drains energy 1.5 times as fast
SETTLE_SECONDS = 3.0        # time to fall into a resting pose before the clock starts
NUDGE = 0.09                # m/s sideways push at the start of settling, so balanced-on-a-knife-edge bodies fall too
EPISODE_SECONDS = 20.0

P = (H + 1) * (W + 1)       # corner points


def _lattice():
    """Constraints (edges and diagonals) and corner tables for the full grid; masks pick the live ones."""
    point = lambda r, c: r * (W + 1) + c
    voxel = lambda r, c: r * W + c if 0 <= r < H and 0 <= c < W else N   # N stands for "no voxel"
    a, b, v1, v2, rest, batch = [], [], [], [], [], []
    for r in range(H + 1):                       # horizontal edges, shared by the voxels above and below
        for c in range(W):
            a.append(point(r, c)); b.append(point(r, c + 1)); v1.append(voxel(r - 1, c)); v2.append(voxel(r, c))
            rest.append(1.0); batch.append(c % 2)
    for r in range(H):                           # vertical edges, shared by the voxels left and right
        for c in range(W + 1):
            a.append(point(r, c)); b.append(point(r + 1, c)); v1.append(voxel(r, c - 1)); v2.append(voxel(r, c))
            rest.append(1.0); batch.append(2 + r % 2)
    for r in range(H):                           # each voxel's two diagonals stop it shearing flat
        for c in range(W):
            for p1, p2 in ((point(r, c), point(r + 1, c + 1)), (point(r, c + 1), point(r + 1, c))):
                a.append(p1); b.append(p2); v1.append(voxel(r, c)); v2.append(N)
                rest.append(np.sqrt(2)); batch.append(4 + 2 * (r % 2) + c % 2)
    corners = [[point(r, c), point(r, c + 1), point(r + 1, c), point(r + 1, c + 1)] for r in range(H) for c in range(W)]
    owners = [[voxel(r - 1, c - 1), voxel(r - 1, c), voxel(r, c - 1), voxel(r, c)] for r in range(H + 1) for c in range(W + 1)]
    home = [[c * VOXEL, (H - r) * VOXEL] for r in range(H + 1) for c in range(W + 1)]
    arr = lambda x, t=np.int32: np.array(x, dtype=t)
    batches = [np.flatnonzero(np.array(batch) == k).astype(np.int32) for k in range(8)]
    for idx in batches:                          # within a batch, no corner appears twice
        pts = np.concatenate([arr(a)[idx], arr(b)[idx]])
        assert len(np.unique(pts)) == len(pts)
    return (arr(a), arr(b), arr(v1), arr(v2), arr(rest, np.float32) * VOXEL, arr(corners), arr(owners),
            arr(home, np.float32), batches)


SPRING_A, SPRING_B, SPRING_V1, SPRING_V2, SPRING_REST, CORNERS, OWNERS, HOME, BATCHES = _lattice()


class State(NamedTuple):
    x: jnp.ndarray          # (P, 2) corner positions
    px: jnp.ndarray         # (P, 2) positions one tick ago (velocity = x − px)
    charge: jnp.ndarray     # (N,)
    spikes: jnp.ndarray     # (N,)
    rate: jnp.ndarray       # (N,) recent firing rate
    act: jnp.ndarray        # (N,) how contracted each muscle is
    energy: jnp.ndarray
    fruit: jnp.ndarray      # (FRUIT,) x positions on the ground
    key: jnp.ndarray
    eaten: jnp.ndarray
    alive_ticks: jnp.ndarray
    tick: jnp.ndarray


class Setup(NamedTuple):
    """Everything about a creature that stays fixed during its life."""
    point_on: jnp.ndarray   # (P,) corner belongs to the body
    spring_on: jnp.ndarray  # (S,) constraint belongs to the body
    voxel_on: jnp.ndarray   # (N+1,) padded with a dead voxel at index N
    muscle: jnp.ndarray     # (N+1,)
    eye: jnp.ndarray        # (N,)
    size: jnp.ndarray       # how much bigger than the starter body (in cells), for the energy cost
    start_x: jnp.ndarray


def _setup(body: Body):
    voxel_on = jnp.concatenate([body.alive, jnp.zeros(1, bool)])
    point_on = jnp.any(voxel_on[OWNERS], axis=1)
    spring_on = voxel_on[SPRING_V1] | voxel_on[SPRING_V2]
    muscle = jnp.concatenate([body.alive & (body.tissue == 1), jnp.zeros(1, bool)])
    return voxel_on, point_on, spring_on, muscle


def _centre_x(x, point_on):
    w = point_on.astype(jnp.float32)
    return jnp.sum(x[:, 0] * w) / jnp.maximum(jnp.sum(w), 1.0)


def start(body: Body, key):
    """Stand the creature on the ground, nudge it, let it settle with muscles off, then scatter fruit
    0.9–2 m away on either side of where it came to rest."""
    voxel_on, point_on, spring_on, muscle = _setup(body)
    big = 1e3
    lowest = jnp.min(jnp.where(point_on, HOME[:, 1], big))
    x = jnp.asarray(HOME) - jnp.array([0.0, lowest])
    x = x - jnp.array([_centre_x(x, point_on), 0.0])
    size = jnp.sum(body.alive) / STARTER_CELLS
    setup = Setup(point_on, spring_on, voxel_on, muscle, body.alive & (body.tissue == 2), size, jnp.float32(0.0))
    k0, k1, k2, k3 = jax.random.split(key, 4)
    push = jnp.where(jax.random.bernoulli(k0), 1.0, -1.0) * NUDGE * DT
    relaxed = jnp.zeros(N)
    x, px = jax.lax.fori_loop(0, ticks_for(SETTLE_SECONDS), lambda _, xs: physics(setup, *xs, relaxed),
                              (x, x - jnp.array([push, 0.0])))
    rest_x = _centre_x(x, point_on)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (FRUIT,)), 1.0, -1.0)
    fruit = rest_x + side * jax.random.uniform(k2, (FRUIT,), minval=0.9, maxval=2.0)
    zeros = jnp.zeros(N)
    state = State(x, px, zeros, zeros, zeros, zeros, jnp.float32(1.0), fruit, k3,
                  jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0))
    return setup._replace(start_x=rest_x), state


# ---------- one tick ----------

def senses(setup: Setup, state: State):
    x = state.x
    corner_y = x[CORNERS, 1]                                       # (N, 4)
    touch = jnp.any(corner_y < 0.01, axis=1) & setup.voxel_on[:N]
    centre = jnp.mean(x[CORNERS], axis=1)                          # (N, 2) voxel centres
    middle = _centre_x(x, setup.point_on)
    facing = jnp.where(centre[:, 0] >= middle, 1.0, -1.0)          # eyes look away from the body's middle
    ahead = (state.fruit[None, :] - centre[:, :1]) * facing[:, None]
    brightness = jnp.sum(jnp.where(ahead > 0, jnp.exp(-ahead / SIGHT_RANGE), 0.0), axis=1)
    sight = (1 - jnp.exp(-2 * brightness)) * setup.eye
    beat = jnp.sin(2 * jnp.pi * BEAT_HZ * state.tick * DT)
    return beat, touch.astype(jnp.float32), sight, 1.0 - state.energy


def think(brain: Brain, state: State, alive, beat, touch, sight, hunger):
    g = brain.gains
    current = brain.bias + g[:, 0] * beat + g[:, 1] * touch + g[:, 2] * sight + g[:, 3] * hunger
    charge = LEAK * state.charge + current + state.spikes @ brain.weights
    spikes = ((charge >= brain.threshold) & alive).astype(jnp.float32)
    charge = jnp.maximum(jnp.where(spikes > 0, 0.0, charge), -3.0)
    rate = 0.7 * state.rate + 0.3 * spikes
    act = state.act + (rate - state.act) * MUSCLE_RESPONSE
    return charge, spikes, rate, act


def physics(setup: Setup, x, px, act):
    # each constraint's length: muscle voxels shrink their edges as they contract
    a = jnp.concatenate([act, jnp.zeros(1)]) * setup.muscle
    on = setup.voxel_on.astype(jnp.float32)
    shrink = (a[SPRING_V1] + a[SPRING_V2]) / jnp.maximum(on[SPRING_V1] + on[SPRING_V2], 1.0)
    length = SPRING_REST * (1 - MUSCLE_AMP * shrink)
    live = setup.spring_on.astype(jnp.float32)
    return jax.lax.fori_loop(0, SUBSTEPS, lambda _, xs: _substep(setup, *xs, length, live), (x, px))


def _substep(setup: Setup, x, px, length, live):
    dt = DT / SUBSTEPS
    v = (x - px) * DAMPING ** (1 / SUBSTEPS)
    px = x
    x = x + v + jnp.array([0.0, -GRAVITY * dt * dt])
    x = jnp.where(setup.point_on[:, None], x, px)

    def solve(_, carry):
        x, depth = carry
        for idx in BATCHES:                     # no two constraints in a batch share a corner
            a, b = SPRING_A[idx], SPRING_B[idx]
            d = x[b] - x[a]
            dist = jnp.linalg.norm(d, axis=1) + 1e-9
            fix = (0.5 * (dist - length[idx]) / dist * live[idx])[:, None] * d   # move both ends half the error
            x = x.at[a].add(fix).at[b].add(-fix)
        below = jnp.minimum(x[:, 1], 0.0)
        return x.at[:, 1].set(x[:, 1] - below), depth - below

    x, depth = jax.lax.fori_loop(0, ITERATIONS, solve, (x, jnp.zeros(P)))
    # Coulomb friction, with how hard the ground pushed back standing in for the load on each point
    slide = x[:, 0] - px[:, 0]
    stick = (depth > 0) & (jnp.abs(slide) < STATIC_FRICTION * depth)
    skid = (depth > 0) & ~stick
    new_x0 = jnp.where(stick, px[:, 0], jnp.where(skid, x[:, 0] - jnp.sign(slide) * KINETIC_FRICTION * depth, x[:, 0]))
    x = x.at[:, 0].set(new_x0)
    return jnp.where(setup.point_on[:, None], x, px), px


def feed(setup: Setup, state: State, x, act):
    """Eat fruit within reach, pay for living and moving, regrow eaten fruit nearby."""
    near_ground = (x[:, 1] < 0.12) & setup.point_on
    reach = jnp.abs(x[:, None, 0] - state.fruit[None, :]) < EAT_REACH                  # (P, FRUIT)
    eaten = jnp.any(reach & near_ground[:, None], axis=0)
    middle = _centre_x(x, setup.point_on)
    lost = jnp.abs(state.fruit - middle) > 6.0                                         # wandered far away
    key, k1, k2 = jax.random.split(state.key, 3)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (FRUIT,)), 1.0, -1.0)
    fresh = middle + side * jax.random.uniform(k2, (FRUIT,), minval=1.0, maxval=3.0)
    fruit = jnp.where(eaten | lost, fresh, state.fruit)
    working = jnp.sum(act * setup.muscle[:N]) / jnp.maximum(jnp.sum(setup.muscle), 1.0)
    drain = DT / IDLE_LIFE * ((0.5 + 0.5 * setup.size) + MOVE_COST * working)
    energy = jnp.minimum(state.energy + FRUIT_ENERGY * jnp.sum(eaten) - drain, 1.0)
    return fruit, energy, jnp.sum(eaten).astype(jnp.float32), key


def tick(setup: Setup, brain: Brain, state: State):
    beat, touch, sight, hunger = senses(setup, state)
    charge, spikes, rate, act = think(brain, state, setup.voxel_on[:N], beat, touch, sight, hunger)
    x, px = physics(setup, state.x, state.px, act)
    fruit, energy, ate, key = feed(setup, state, x, act)
    new = State(x, px, charge, spikes, rate, act, energy, fruit, key,
                state.eaten + ate, state.alive_ticks + 1, state.tick + 1)
    living = state.energy > 0
    # a dead creature stays where it fell
    return jax.tree_util.tree_map(lambda n, o: jnp.where(living, n, o), new, state._replace(tick=state.tick + 1))


# ---------- a whole life ----------

def ticks_for(seconds):
    return int(round(seconds / DT))


def life(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    setup, state = start(body, key)
    final, _ = jax.lax.scan(lambda s, _: (tick(setup, brain, s), None), state, None, length=ticks_for(seconds))
    return setup, final


def score(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    setup, final = life(body, brain, key, seconds)
    travelled = jnp.abs(_centre_x(final.x, setup.point_on) - setup.start_x)
    survived = final.alive_ticks / ticks_for(seconds)
    return final.eaten + survived + 0.2 * jnp.minimum(travelled, 5.0)


def record(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, key)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.fruit, s.energy, s.eaten)

    _, frames = jax.lax.scan(step, state, None, length=ticks_for(seconds))
    return setup, frames
