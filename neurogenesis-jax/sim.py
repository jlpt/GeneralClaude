"""The world: a side view with flat ground and fruit lying on it.

Body. Every living cell is a joint: a point mass at its place on the grid. Bones (rigid links one
cell long) join each cell to its parent, the neighbour one step nearer the seed, so the skeleton is a
tree branching out from the seed. Neighbouring cells that are both bone or eye are joined too, so a
solid patch of them is solid. At each cell, the angle between the bone to its parent and the bone to
each of its children is held where it grew:
    - a bone cell holds it rigidly
    - a muscle cell is a hinge: as its neuron fires, its muscle bends the joint by up to BEND_RANGE,
      the way the cell's bend number says, and when the neuron goes quiet the joint straightens again
So a thin line of cells is a limb, stiff along its bone cells and bending at its muscle cells.

Bones and joints are solved with position-based dynamics. They're split into batches in which no two
share a point, so each batch is solved fully in parallel (good for a GPU) and later batches see earlier
fixes, which converges fast. A joint's angle is held with a linear constraint (see _hold_angle), which
settles much faster than correcting the angle directly. The ground pushes joints up. Friction follows Coulomb's law: the body's
weight is shared among the joints touching the ground, and a loaded joint sticks unless pushed hard, so
to travel a creature has to lift and push.

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
fruit). Then energy starts at 1 and drains every tick, faster for bodies bigger than the starter (but
never slower for smaller ones, or evolution shrinks creatures until they outlive the clock) and while
muscles work; each fruit eaten adds FRUIT_ENERGY. At zero the creature dies and stops. Fruit is placed,
and regrows when eaten, a distance beyond either end of the body, so it can't be reached by falling or
lunging (evolution found both).

Score. Fruit eaten, plus the fraction of the time it stayed alive, plus a little for distance travelled
(so early creatures that merely move beat ones that don't, before any can reach fruit). A body of fewer
than 3 cells scores 0.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from creature import DIRS, H, W, N, SEED, Body, Brain

# body physics
CELL = 0.1                  # metres between neighbouring joints: the length of every bone
RADIUS = 0.04               # limbs are this thick (half-width): joints rest this far above the ground
DT = 1 / 30                 # seconds per tick (brain and physics)
GRAVITY = 9.8
DAMPING = 0.99
SUBSTEPS = 6                # physics steps per tick; joints get stiffer with smaller steps, and with
ITERATIONS = 2              # passes over all bones and joints per substep; fewer, the starter's hips buckle
BEND_RANGE = 1.2            # radians (about 70°) a fully active muscle bends its joint
MUSCLE_RESPONSE = 0.15      # fraction of the way a muscle moves toward its drive each tick
STATIC_FRICTION, KINETIC_FRICTION = 1.2, 0.8

# brain and senses
LEAK = 0.8
BEAT_HZ = 1.5
SIGHT_RANGE = 1.5           # metres at which fruit looks 1/e as bright

# survival
FRUIT = 4                   # fruit lying in the world at any time
FRUIT_ENERGY = 0.4
EAT_REACH = 0.08            # metres: a joint this close to a fruit (and near the ground) eats it
IDLE_LIFE = 12.0            # seconds an idle starter-sized creature survives on a full stomach
STARTER_CELLS = 32          # each cell beyond this many adds half a starter's worth of drain per starter size
MOVE_COST = 0.5             # working every muscle flat out drains energy 1.5 times as fast
SETTLE_SECONDS = 3.0        # time to fall into a resting pose before the clock starts
NUDGE = 0.09                # m/s sideways push at the start of settling, so balanced-on-a-knife-edge bodies fall too
EPISODE_SECONDS = 20.0

ROOT = SEED[0] * W + SEED[1]


def _tables():
    """Every possible bone and joint on the grid; each creature's masks pick the ones it has."""
    cell = lambda r, c: r * W + c
    inside = lambda r, c: 0 <= r < H and 0 <= c < W
    link_a, link_b, link_batch = [], [], []
    for r in range(H):                           # side-by-side neighbours
        for c in range(W - 1):
            link_a.append(cell(r, c)); link_b.append(cell(r, c + 1)); link_batch.append(c % 2)
    for r in range(H - 1):                       # neighbours above and below
        for c in range(W):
            link_a.append(cell(r, c)); link_b.append(cell(r + 1, c)); link_batch.append(2 + r % 2)
    # A joint is a cell p and one of its children c; its other arm, to p's parent, is only known once the
    # creature has grown, but it is always one of p's 4 neighbours. So a joint only touches the plus
    # shape around p, and plus shapes around cells with the same (row + 2·column) mod 5 never overlap.
    joint_p, joint_c, joint_batch = [], [], []
    for r in range(H):
        for c in range(W):
            for d, (dy, dx) in enumerate(DIRS):
                if inside(r + dy, c + dx):
                    joint_p.append(cell(r, c)); joint_c.append(cell(r + dy, c + dx))
                    joint_batch.append(5 * d + (r + 2 * c) % 5)
    arr = lambda x: np.array(x, dtype=np.int32)
    link_a, link_b, joint_p, joint_c = arr(link_a), arr(link_b), arr(joint_p), arr(joint_c)
    link_batches = [np.flatnonzero(np.array(link_batch) == k).astype(np.int32) for k in range(4)]
    joint_batches = [np.flatnonzero(np.array(joint_batch) == k).astype(np.int32) for k in range(20)]
    for idx in link_batches:                     # within a batch, no joint appears twice
        pts = np.concatenate([link_a[idx], link_b[idx]])
        assert len(np.unique(pts)) == len(pts)
    for idx in joint_batches:
        plus = [cell(p // W + dy, p % W + dx) for p in joint_p[idx] for dy, dx in ((0, 0),) + DIRS
                if inside(p // W + dy, p % W + dx)]
        assert len(np.unique(plus)) == len(plus)
    root_kids = arr([cell(SEED[0] + dy, SEED[1] + dx) for dy, dx in DIRS if inside(SEED[0] + dy, SEED[1] + dx)])
    home = np.array([[c * CELL, (H - 1 - r) * CELL] for r in range(H) for c in range(W)], np.float32)
    return link_a, link_b, link_batches, joint_p, joint_c, joint_batches, root_kids, home


LINK_A, LINK_B, LINK_BATCHES, JOINT_P, JOINT_C, JOINT_BATCHES, ROOT_KIDS, HOME = _tables()


def _angle(u, v):
    """Signed angle from u to v, anticlockwise positive."""
    return jnp.arctan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], jnp.sum(u * v, axis=-1))


class State(NamedTuple):
    x: jnp.ndarray          # (N, 2) joint positions
    px: jnp.ndarray         # (N, 2) positions one substep ago (velocity = x − px)
    charge: jnp.ndarray     # (N,)
    spikes: jnp.ndarray     # (N,)
    rate: jnp.ndarray       # (N,) recent firing rate
    act: jnp.ndarray        # (N,) how hard each muscle is pulling
    energy: jnp.ndarray
    fruit: jnp.ndarray      # (FRUIT,) x positions on the ground
    key: jnp.ndarray
    eaten: jnp.ndarray
    alive_ticks: jnp.ndarray
    tick: jnp.ndarray


class Skeleton(NamedTuple):
    link_on: jnp.ndarray    # (L,) this bone exists
    joint_on: jnp.ndarray   # (J,) this joint exists
    joint_a: jnp.ndarray    # (J,) the joint's other arm: its cell's parent (or, at the seed, another child)
    joint_rest: jnp.ndarray # (J,) the angle it grew with
    joint_bend: jnp.ndarray # (J,) radians it bends at full muscle activity (0 for bone)


def skeleton(body: Body):
    """Which bones and joints a grown body has."""
    alive, parent = body.alive, body.parent
    muscle = alive & (body.tissue == 1)
    i, j = LINK_A, LINK_B
    family = (parent[i] == j) | (parent[j] == i)
    link_on = alive[i] & alive[j] & (family | (~muscle[i] & ~muscle[j]))
    # The seed has no parent, so its children's angles are held relative to its first child.
    kids = jnp.asarray(ROOT_KIDS)
    first = kids[jnp.argmax(alive[kids] & (parent[kids] == ROOT))]
    p, c = JOINT_P, JOINT_C
    a = jnp.where(p == ROOT, first, parent[p])
    on = alive[p] & alive[c] & (parent[c] == p) & ~((p == ROOT) & (c == first))
    home = jnp.asarray(HOME)
    rest = jnp.where(on, _angle(home[a] - home[p], home[c] - home[p]) % (2 * jnp.pi), jnp.pi)
    bend = jnp.where(muscle[p], BEND_RANGE * body.bend[p], 0.0)
    return Skeleton(link_on, on, a, rest, bend)


class Setup(NamedTuple):
    """Everything about a creature that stays fixed during its life."""
    point_on: jnp.ndarray   # (N,) the joint belongs to the body
    bones: Skeleton
    muscle: jnp.ndarray     # (N,)
    eye: jnp.ndarray        # (N,)
    size: jnp.ndarray       # how much bigger than the starter body (in cells), for the energy cost
    start_x: jnp.ndarray


def _centre_x(x, point_on):
    w = point_on.astype(jnp.float32)
    return jnp.sum(x[:, 0] * w) / jnp.maximum(jnp.sum(w), 1.0)


def _beyond(x, point_on, k1, k2, near, far):
    """Fruit positions a random distance beyond the creature's left or right end, so that no body,
    however long, can reach food just by falling or lunging toward it."""
    left = jnp.min(jnp.where(point_on, x[:, 0], jnp.inf))
    right = jnp.max(jnp.where(point_on, x[:, 0], -jnp.inf))
    gap = jax.random.uniform(k2, (FRUIT,), minval=near, maxval=far)
    return jnp.where(jax.random.bernoulli(k1, 0.5, (FRUIT,)), right + gap, left - gap)


def start(body: Body, key):
    """Stand the creature on the ground, nudge it, let it settle with muscles off, then scatter fruit
    0.8–2 m beyond either end of where it came to rest."""
    point_on = body.alive
    big = 1e3
    lowest = jnp.min(jnp.where(point_on, HOME[:, 1], big))
    x = jnp.asarray(HOME) - jnp.array([0.0, lowest - RADIUS])
    x = x - jnp.array([_centre_x(x, point_on), 0.0])
    size = jnp.sum(body.alive) / STARTER_CELLS
    setup = Setup(point_on, skeleton(body), body.alive & (body.tissue == 1), body.alive & (body.tissue == 2),
                  size, jnp.float32(0.0))
    k0, k1, k2, k3 = jax.random.split(key, 4)
    push = jnp.where(jax.random.bernoulli(k0), 1.0, -1.0) * NUDGE * DT / SUBSTEPS   # x − px is one substep
    relaxed = jnp.zeros(N)
    x, px = jax.lax.fori_loop(0, ticks_for(SETTLE_SECONDS), lambda _, xs: physics(setup, *xs, relaxed),
                              (x, x - jnp.array([push, 0.0])))
    rest_x = _centre_x(x, point_on)
    fruit = _beyond(x, point_on, k1, k2, 0.8, 2.0)
    zeros = jnp.zeros(N)
    state = State(x, px, zeros, zeros, zeros, zeros, jnp.float32(1.0), fruit, k3,
                  jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0))
    return setup._replace(start_x=rest_x), state


# ---------- one tick ----------

def senses(setup: Setup, state: State):
    x = state.x
    touch = (x[:, 1] < RADIUS + 0.01) & setup.point_on
    middle = _centre_x(x, setup.point_on)
    facing = jnp.where(x[:, 0] >= middle, 1.0, -1.0)              # eyes look away from the body's middle
    ahead = (state.fruit[None, :] - x[:, :1]) * facing[:, None]
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
    bones = setup.bones
    target = bones.joint_rest + bones.joint_bend * act[JOINT_P]      # muscles bend their joints
    return jax.lax.fori_loop(0, SUBSTEPS, lambda _, xs: _substep(setup, *xs, target), (x, px))


def _hold_angle(x, bones: Skeleton, target, idx):
    """Bend each joint toward its target angle (one batch; no two joints share a point).

    With both arms one bone long, a joint at angle θ puts its cell p at a fixed offset from the midpoint
    of its two neighbours a and c:  p = (a + c) / 2 + k · turn90(c − a),  where k = cot(θ/2) / 2.
    Holding that is a linear constraint, which settles much faster than correcting the angle directly.
    """
    a, p, c = bones.joint_a[idx], JOINT_P[idx], JOINT_C[idx]
    xa, xp, xc = x[a], x[p], x[c]
    k = 0.5 / jnp.tan(jnp.clip(target[idx], 0.2, 2 * jnp.pi - 0.2) / 2)[:, None]
    turn = lambda w: jnp.stack([-w[:, 1], w[:, 0]], axis=1)
    error = xp - (xa + xc) / 2 - k * turn(xc - xa)
    step = error * bones.joint_on[idx][:, None] / (1.5 + 2 * k * k)
    return x.at[a].add(step / 2 + k * turn(step)).at[p].add(-step).at[c].add(step / 2 - k * turn(step))


def _substep(setup: Setup, x, px, target):
    dt = DT / SUBSTEPS
    bones = setup.bones
    v = (x - px) * DAMPING ** (1 / SUBSTEPS)
    px = x
    x = x + v + jnp.array([0.0, -GRAVITY * dt * dt])
    x = jnp.where(setup.point_on[:, None], x, px)
    live = bones.link_on.astype(jnp.float32)
    joint_live = bones._replace(joint_on=bones.joint_on.astype(jnp.float32))

    def solve(_, carry):
        x, depth = carry
        for idx in LINK_BATCHES:                # bones keep their length
            a, b = LINK_A[idx], LINK_B[idx]
            d = x[b] - x[a]
            dist = jnp.linalg.norm(d, axis=1) + 1e-9
            fix = (0.5 * (dist - CELL) / dist * live[idx])[:, None] * d   # move both ends half the error
            x = x.at[a].add(fix).at[b].add(-fix)
        for idx in JOINT_BATCHES:               # joints keep (or, for muscles, bend to) their angle
            x = _hold_angle(x, joint_live, target, idx)
        below = jnp.minimum(x[:, 1] - RADIUS, 0.0)
        return x.at[:, 1].set(x[:, 1] - below), depth - below

    x, depth = jax.lax.fori_loop(0, ITERATIONS, solve, (x, jnp.zeros(N)))
    # Coulomb friction. The body's weight is shared among the joints touching the ground, in proportion to
    # how hard the ground pushed back on each; friction can stop a joint sliding up to the share it carries.
    depth = depth * setup.point_on
    weight = jnp.sum(setup.point_on) * GRAVITY * dt * dt
    load = weight * depth / (jnp.sum(depth) + 1e-12)
    slide = x[:, 0] - px[:, 0]
    stick = (depth > 0) & (jnp.abs(slide) < STATIC_FRICTION * load)
    skid = (depth > 0) & ~stick
    new_x0 = jnp.where(stick, px[:, 0], jnp.where(skid, x[:, 0] - jnp.sign(slide) * KINETIC_FRICTION * load, x[:, 0]))
    x = x.at[:, 0].set(new_x0)
    return jnp.where(setup.point_on[:, None], x, px), px


def feed(setup: Setup, state: State, x, act):
    """Eat fruit within reach, pay for living and moving, regrow eaten fruit 1–3 m beyond an end."""
    near_ground = (x[:, 1] < RADIUS + 0.08) & setup.point_on
    reach = jnp.abs(x[:, None, 0] - state.fruit[None, :]) < EAT_REACH                  # (N, FRUIT)
    eaten = jnp.any(reach & near_ground[:, None], axis=0)
    middle = _centre_x(x, setup.point_on)
    lost = jnp.abs(state.fruit - middle) > 6.0                                         # wandered far away
    key, k1, k2 = jax.random.split(state.key, 3)
    fresh = _beyond(x, setup.point_on, k1, k2, 1.0, 3.0)
    fruit = jnp.where(eaten | lost, fresh, state.fruit)
    working = jnp.sum(act * setup.muscle) / jnp.maximum(jnp.sum(setup.muscle), 1.0)
    drain = DT / IDLE_LIFE * (1 + 0.5 * jnp.maximum(setup.size - 1, 0.0) + MOVE_COST * working)
    energy = jnp.minimum(state.energy + FRUIT_ENERGY * jnp.sum(eaten) - drain, 1.0)
    return fruit, energy, jnp.sum(eaten).astype(jnp.float32), key


def tick(setup: Setup, brain: Brain, state: State):
    beat, touch, sight, hunger = senses(setup, state)
    charge, spikes, rate, act = think(brain, state, setup.point_on, beat, touch, sight, hunger)
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
    has_body = jnp.sum(body.alive) >= 3         # a growth rule that kills its seed gets nothing
    return jnp.where(has_body, final.eaten + survived + 0.2 * jnp.minimum(travelled, 5.0), 0.0)


def record(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, key)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.fruit, s.energy, s.eaten)

    _, frames = jax.lax.scan(step, state, None, length=ticks_for(seconds))
    return setup, frames
