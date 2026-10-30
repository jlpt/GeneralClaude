"""The world: a side view with flat ground. The creature's job is to walk forward (to the right).

Body. Every living cell is a joint: a point mass at its place on the grid. Bones (rigid links one
cell long) join each cell to its parent, the neighbour one step nearer the seed, so the skeleton is a
tree branching out from the seed. Neighbouring cells that are both bone are joined too, so a solid patch
of bone is solid. At each cell, the angle between the bone to its parent and the bone to each of its
children is held where it grew:
    - a bone cell holds it rigidly
    - a muscle cell is a hinge: when the nerve cells in its patch fire, the joint bends by up to
      BEND_RANGE, the way the cell's bend number says, and it straightens again when they go quiet. A
      muscle joint with no nerve in it never moves.
So a thin line of cells is a limb, stiff along its bone cells and bending at its muscle cells.

Bones and joints are solved with position-based dynamics. They're split into batches in which no two
share a point, so each batch is solved fully in parallel (good for a GPU) and later batches see earlier
fixes, which converges fast. A joint's angle is held with a linear constraint (see _hold_angle), which
settles much faster than correcting the angle directly. The ground pushes joints up. Friction follows
Coulomb's law: the body's weight is shared among the joints touching the ground, and a loaded joint
sticks unless pushed hard, so to travel a creature has to lift and push.

Nervous system. Only nerve cells are neurons: leaky integrate-and-fire, each wired to the nerve cells
touching it. They live on the fine grid, 9 to a joint. Each tick:
    charge = LEAK * charge + bias + senses + input from the spikes of touching nerve cells
and a nerve cell spikes (and resets) when its charge reaches its threshold. A muscle joint follows the
recent firing rate of the nerve cells in its patch, each times its junction strength, summed and divided
by FIBRE (so one nerve running straight through a joint can drive it fully), with about a 0.2 s lag, as
real muscle does.

Senses. Each is a current into the nerve cells that can feel it, scaled by each one's gain for it:
    touch      nerve cells in the patch of a joint touching the ground
    pacemaker  every nerve cell: a steady rhythm, sin(2π · PACE_HZ · t), like the heart's pacemaker
So a foot's touch only matters if a nerve carries the signal on to something.

Score. Before the clock starts, the creature settles for SETTLE_SECONDS with its muscles off, after a
tiny random sideways nudge, so a body that would topple over has already done so. Then its middle is
tracked for EPISODE_SECONDS as it moves to the right. The worse half of that time counts three times as
much as the better half, so a steady walker keeps its full distance and one big lunge keeps only half
(evolution's first walkers lunged once and stopped). Falling over, turning more than MAX_TILT from the
pose it settled in, scores 0 (otherwise evolution grows bodies that throw themselves forward). A body of
fewer than 3 cells scores 0.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from creature import DIRS, FINE, H, HF, W, WF, N, SEED, Body, Brain, patches, spread, synapse_input

# body physics
CELL = 0.1                  # metres between neighbouring joints: the length of every bone
RADIUS = 0.04               # limbs are this thick (half-width): joints rest this far above the ground
DT = 1 / 30                 # seconds per tick (nervous system and physics)
GRAVITY = 9.8
DAMPING = 0.99
SUBSTEPS = 6                # physics steps per tick; joints get stiffer with smaller steps, and with
ITERATIONS = 2              # passes over all bones and joints per substep; fewer, the starter's hips buckle
BEND_RANGE = 1.2            # radians (about 70°) a fully active muscle bends its joint
MUSCLE_RESPONSE = 0.15      # fraction of the way a muscle moves toward its drive each tick
FIBRE = 3                   # firing nerve cells in a joint's patch needed to drive it fully
STATIC_FRICTION, KINETIC_FRICTION = 1.2, 0.8

# nervous system
LEAK = 0.8
PACE_HZ = 1.5

# the test
SETTLE_SECONDS = 3.0        # time to fall into a resting pose before the clock starts
NUDGE = 0.09                # m/s sideways push at the start of settling, so balanced-on-a-knife-edge bodies fall too
MAX_TILT = np.radians(45)   # turning further than this from the settled pose counts as falling over
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
    charge: jnp.ndarray     # (HF, WF)
    spikes: jnp.ndarray     # (HF, WF)
    rate: jnp.ndarray       # (HF, WF) recent firing rate
    act: jnp.ndarray        # (N,) how hard each muscle is pulling
    walked: jnp.ndarray     # metres walked to the right so far
    fallen: jnp.ndarray
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
    nerve: jnp.ndarray      # (HF, WF) a nerve cell grows in this fine cell
    start_x: jnp.ndarray
    pose: jnp.ndarray       # (N, 2) the settled shape around its middle, to tell how far the body has turned


def centre_x(x, point_on):
    w = point_on.astype(jnp.float32)
    return jnp.sum(x[:, 0] * w) / jnp.maximum(jnp.sum(w), 1.0)


def _middle(x, point_on):
    w = point_on.astype(jnp.float32)[:, None]
    return jnp.sum(x * w, axis=0) / jnp.maximum(jnp.sum(w), 1.0)


def tilt(setup: Setup, x):
    """How far the body has turned from its settled pose: the rotation that best lines the two up."""
    now = (x - _middle(x, setup.point_on)) * setup.point_on[:, None]
    then = setup.pose
    return jnp.arctan2(jnp.sum(then[:, 0] * now[:, 1] - then[:, 1] * now[:, 0]), jnp.sum(then * now))


def start(body: Body, key):
    """Stand the creature on the ground, nudge it, and let it settle with its muscles off."""
    point_on = body.alive
    lowest = jnp.min(jnp.where(point_on, HOME[:, 1], 1e3))
    x = jnp.asarray(HOME) - jnp.array([0.0, lowest - RADIUS])
    x = x - jnp.array([centre_x(x, point_on), 0.0])
    setup = Setup(point_on, skeleton(body), body.nerve, jnp.float32(0.0), jnp.zeros((N, 2)))
    push = jnp.where(jax.random.bernoulli(key), 1.0, -1.0) * NUDGE * DT / SUBSTEPS   # x − px is one substep
    relaxed = jnp.zeros(N)
    x, px = jax.lax.fori_loop(0, ticks_for(SETTLE_SECONDS), lambda _, xs: physics(setup, *xs, relaxed),
                              (x, x - jnp.array([push, 0.0])))
    pose = (x - _middle(x, point_on)) * point_on[:, None]
    setup = setup._replace(start_x=centre_x(x, point_on), pose=pose)
    fine = jnp.zeros((HF, WF))
    return setup, State(x, px, fine, fine, fine, jnp.zeros(N), jnp.float32(0.0), jnp.bool_(False), jnp.int32(0))


# ---------- one tick ----------

def senses(setup: Setup, state: State):
    touch = (state.x[:, 1] < RADIUS + 0.01) & setup.point_on
    pace = jnp.sin(2 * jnp.pi * PACE_HZ * state.tick * DT)
    return pace, spread(touch.reshape(H, W)).astype(jnp.float32)       # felt by every fine cell of the patch


def think(brain: Brain, state: State, nerve, pace, touch):
    g = brain.gains
    current = brain.bias + g[..., 0] * pace + g[..., 1] * touch
    charge = LEAK * state.charge + current + synapse_input(brain, state.spikes)
    spikes = ((charge >= brain.threshold) & nerve).astype(jnp.float32)
    charge = jnp.maximum(jnp.where(spikes > 0, 0.0, charge), -3.0)
    rate = 0.7 * state.rate + 0.3 * spikes
    # a muscle joint follows the nerve cells in its patch (junction is 0 where there's no nerve)
    drive = jnp.minimum(jnp.sum(patches(rate * brain.junction), axis=-1) / FIBRE, 1.0).reshape(N)
    act = state.act + (drive - state.act) * MUSCLE_RESPONSE
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


def tick(setup: Setup, brain: Brain, state: State):
    pace, touch = senses(setup, state)
    charge, spikes, rate, act = think(brain, state, setup.nerve, pace, touch)
    x, px = physics(setup, state.x, state.px, act)
    fallen = state.fallen | (jnp.abs(tilt(setup, x)) > MAX_TILT)
    walked = centre_x(x, setup.point_on) - setup.start_x
    return State(x, px, charge, spikes, rate, act, walked, fallen, state.tick + 1)


# ---------- a whole life ----------

def ticks_for(seconds):
    return int(round(seconds / DT))


def life(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """The whole life; also returns the distance walked by halfway."""
    setup, state = start(body, key)
    half = ticks_for(seconds) // 2
    step = lambda s, _: (tick(setup, brain, s), None)
    middle, _ = jax.lax.scan(step, state, None, length=half)
    final, _ = jax.lax.scan(step, middle, None, length=ticks_for(seconds) - half)
    return setup, final, middle.walked


def score(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """Metres walked to the right, the worse half of the life counting three times as much as the better
    half; 0 for falling over."""
    setup, final, first = life(body, brain, key, seconds)
    second = final.walked - first
    steady = 1.5 * jnp.minimum(first, second) + 0.5 * jnp.maximum(first, second)
    has_body = jnp.sum(body.alive) >= 3         # a growth rule that kills its seed gets nothing
    return jnp.where(has_body & ~final.fallen, steady, 0.0)


def record(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, key)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.walked, s.fallen)

    _, frames = jax.lax.scan(step, state, None, length=ticks_for(seconds))
    return setup, frames
