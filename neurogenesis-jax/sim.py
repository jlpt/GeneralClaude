"""The world: a side view with flat ground and fruit lying on it. The creature has to find fruit to keep
its energy up, and starves if it doesn't.

Body. Every living cell is a joint: a point mass at its place on the grid. Bones (rigid links one
cell long) join each cell to its parent, the neighbour one step nearer the seed, so the skeleton is a
tree branching out from the seed. Neighbouring cells that are both bone are joined too, so a solid patch
of bone is solid. At each cell, the angle between the bone to its parent and the bone to each of its
children is held where it grew:
    - a bone cell holds it rigidly, and so does an eye
    - a muscle cell is a hinge: when the nerve cells in its patch fire, the joint bends by up to
      BEND_RANGE, the way the cell's bend number says, and it straightens again when they go quiet. A
      muscle joint with no nerve in it never moves.
So a thin line of cells is a limb, stiff along its bone cells and bending at its muscle cells.

Bones and joints are solved with position-based dynamics. They're split into batches in which no two
share a point, so each batch is solved fully in parallel (good for a GPU) and later batches see earlier
fixes, which converges fast. Each bone also damps the relative motion of its two ends; without that,
floppy bodies build up a jiggle under their own weight, and evolution used it to slide along. A joint's
angle is held with a linear constraint (see _hold_angle), which settles much faster than correcting the
angle directly. The ground pushes joints up. Friction follows
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
    sight      nerve cells in an eye's patch: how much fruit colour the eye sees in the direction it
               faces (away from the body's middle), nearer fruit counting more. This is the same as
               reading a strip of pixels ahead where fruit is one colour, computed from the fruit positions.
    hunger     every nerve cell, like a blood-sugar signal: 1 − energy
    pacemaker  every nerve cell: a steady rhythm, sin(2π · PACE_HZ · t), like the heart's pacemaker
So an eye or a foot only matters if a nerve carries its signal on to something.

The world. Before the clock starts, the creature settles for SETTLE_SECONDS with its muscles off, after
a tiny random sideways nudge, so a body that would topple over has already done so. Then FRUIT fruit are
placed beyond its left or right end, farther away than the body is tall, so no body can reach food by
falling or lunging toward it. Energy starts at 1 and drains every tick, faster for bodies bigger than
SIZE_REF joints (never slower for smaller ones, or evolution shrinks creatures until they outlive the
clock) and while muscles work. Touching a fruit near the ground eats it: energy + FRUIT_ENERGY, and a new
fruit appears beyond a random end. At zero energy the creature dies and stops. Where fruit appears, and
which way the nudge goes, come from chance(world, n), a small hash both this file and the browser version
compute identically, so a world number gives the same world in both.

Score. Fruit eaten, plus the fraction of the time it stayed alive, plus TRAVEL_CREDIT per metre it ended
up from where it started (up to TRAVEL_MAX metres), so that creatures that merely move beat ones that
don't, before any can reach food. A body of fewer than 3 joints scores 0.
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
BONE_DAMPING = 0.3          # fraction of the relative motion of a bone's two ends removed each substep
STATIC_FRICTION, KINETIC_FRICTION = 1.2, 0.8
# a ball, for the throwing task (throw.py)
BALL_RADIUS = 0.05
BALL_WEIGHT = 0.5           # a joint weighs 1
BALL_FRICTION = 0.8         # on the ground: a pushed ball stops within centimetres, a thrown one rolls a little
BALL_MAX_SPEED = 8.0        # m/s, in any case
BALL_KICK = 2.0             # the ball can leave a joint at most this many times as fast as the joint moves

# nervous system and senses
LEAK = 0.8
PACE_HZ = 1.5
SIGHT_RANGE = 1.5           # metres at which fruit looks 1/e as bright

# the world
SETTLE_SECONDS = 3.0        # time to fall into a resting pose before the clock starts
NUDGE = 0.09                # m/s sideways push at the start of settling, so balanced-on-a-knife-edge bodies fall too
FRUIT = 4                   # fruit lying in the world at any time
FRUIT_ENERGY = 0.4
EAT_REACH = 0.08            # metres: a joint this close to a fruit (and near the ground) eats it
IDLE_LIFE = 12.0            # seconds an idle creature of SIZE_REF joints survives on a full stomach
SIZE_REF = 20               # each joint beyond this many adds half as much drain again per SIZE_REF joints
MOVE_COST = 0.25            # working every muscle flat out drains energy 1.25 times as fast
TRAVEL_CREDIT, TRAVEL_MAX = 0.25, 4.0
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
    energy: jnp.ndarray
    fruit: jnp.ndarray      # (FRUIT,) x positions on the ground
    placed: jnp.ndarray     # how many fruit have been placed so far (numbers the draws from chance())
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
    nerve: jnp.ndarray      # (HF, WF) a nerve cell grows in this fine cell
    eye: jnp.ndarray        # (N,) this joint is an eye
    muscle: jnp.ndarray     # (N,) this joint is a muscle
    size: jnp.ndarray       # joints / SIZE_REF, for the energy cost
    height: jnp.ndarray     # how tall it is: how far it could reach just by falling over
    world: jnp.ndarray      # the world number
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


def chance(world, n):
    """A number in [0, 1) for draw n of a world: a 32-bit integer hash, so the browser version can compute
    exactly the same one."""
    h = (jnp.uint32(world) * jnp.uint32(2654435761) + jnp.uint32(n) * jnp.uint32(2246822519) + jnp.uint32(374761393))
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x7FEB352D)
    h = h ^ (h >> 15)
    h = h * jnp.uint32(0x846CA68B)
    h = h ^ (h >> 16)
    return (h >> 8).astype(jnp.float32) / 16777216.0


NUDGE_DRAW = 1_000_000      # chance(world, NUDGE_DRAW) decides the nudge; fruit k uses draws 2k and 2k + 1


def _place(x, point_on, height, world, k, near, far):
    """Fruit number k: beyond the creature's left or right end, farther away than it is tall."""
    left = jnp.min(jnp.where(point_on, x[:, 0], jnp.inf))
    right = jnp.max(jnp.where(point_on, x[:, 0], -jnp.inf))
    gap = height + near + (far - near) * chance(world, 2 * k + 1)
    return jnp.where(chance(world, 2 * k) < 0.5, right + gap, left - gap)


def start(body: Body, world):
    """Stand the creature on the ground, nudge it, let it settle with its muscles off, then place fruit
    beyond either end: farther than its height, plus 0.2–1.2 m."""
    point_on = body.alive
    lowest = jnp.min(jnp.where(point_on, HOME[:, 1], 1e3))
    x = jnp.asarray(HOME) - jnp.array([0.0, lowest - RADIUS])
    x = x - jnp.array([centre_x(x, point_on), 0.0])
    grown_height = jnp.max(jnp.where(point_on, x[:, 1], 0.0))
    setup = Setup(point_on, skeleton(body), body.nerve, body.alive & (body.tissue == 2), body.alive & (body.tissue == 1),
                  jnp.sum(body.alive) / SIZE_REF, grown_height, jnp.uint32(world), jnp.float32(0.0), jnp.zeros((N, 2)))
    push = jnp.where(chance(world, NUDGE_DRAW) < 0.5, 1.0, -1.0) * NUDGE * DT / SUBSTEPS   # x − px is one substep
    relaxed = jnp.zeros(N)
    x, px = jax.lax.fori_loop(0, ticks_for(SETTLE_SECONDS), lambda _, xs: physics(setup, *xs, relaxed),
                              (x, x - jnp.array([push, 0.0])))
    pose = (x - _middle(x, point_on)) * point_on[:, None]
    height = jnp.maximum(grown_height, jnp.max(jnp.where(point_on, x[:, 1], 0.0)))
    setup = setup._replace(start_x=centre_x(x, point_on), pose=pose, height=height)
    fruit = jax.vmap(lambda k: _place(x, point_on, height, world, k, 0.2, 1.2))(jnp.arange(FRUIT))
    fine = jnp.zeros((HF, WF))
    state = State(x, px, fine, fine, fine, jnp.zeros(N), jnp.float32(1.0), fruit, jnp.int32(FRUIT),
                  jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0))
    return setup, state


# ---------- one tick ----------

def senses(setup: Setup, state: State):
    x = state.x
    touch = (x[:, 1] < RADIUS + 0.01) & setup.point_on
    middle = centre_x(x, setup.point_on)
    facing = jnp.where(x[:, 0] >= middle, 1.0, -1.0)              # eyes look away from the body's middle
    ahead = (state.fruit[None, :] - x[:, :1]) * facing[:, None]
    brightness = jnp.sum(jnp.where(ahead > 0, jnp.exp(-ahead / SIGHT_RANGE), 0.0), axis=1)
    sight = (1 - jnp.exp(-2 * brightness)) * setup.eye
    pace = jnp.sin(2 * jnp.pi * PACE_HZ * state.tick * DT)
    # touch and sight are felt by every fine cell of the joint's patch
    return (pace, spread(touch.reshape(H, W)).astype(jnp.float32), spread(sight.reshape(H, W)),
            1.0 - state.energy)


def think(brain: Brain, state: State, nerve, pace, touch, sight, hunger):
    g = brain.gains
    current = brain.bias + g[..., 0] * pace + g[..., 1] * touch + g[..., 2] * sight + g[..., 3] * hunger
    charge = LEAK * state.charge + current + synapse_input(brain, state.spikes)
    spikes = ((charge >= brain.threshold) & nerve).astype(jnp.float32)
    charge = jnp.maximum(jnp.where(spikes > 0, 0.0, charge), -3.0)
    rate = 0.7 * state.rate + 0.3 * spikes
    # a muscle joint follows the nerve cells in its patch (junction is 0 where there's no nerve)
    drive = jnp.minimum(jnp.sum(patches(rate * brain.junction), axis=-1) / FIBRE, 1.0).reshape(N)
    act = state.act + (drive - state.act) * MUSCLE_RESPONSE
    return charge, spikes, rate, act


def physics(setup: Setup, x, px, act, balls=None):
    """One tick of physics. balls, if given, is (positions, positions one substep ago), each (B, 2), and
    comes back moved too. Balls pass through each other."""
    bones = setup.bones
    target = bones.joint_rest + bones.joint_bend * act[JOINT_P]      # muscles bend their joints
    if balls is None:
        return jax.lax.fori_loop(0, SUBSTEPS, lambda _, xs: _substep(setup, *xs, target), (x, px))
    return jax.lax.fori_loop(0, SUBSTEPS, lambda _, s: _substep(setup, s[0], s[1], target, s[2]), (x, px, balls))


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


def _no_turn(x, before, point_on):
    """Take back any turn of the whole body that one pass over the bones and joints made.

    Bones and joints are forces inside the body, and can't turn it as a whole. But when a joint can't reach
    its angle because the rest of the body won't let it, each pass over the bones and joints goes round a
    small loop of shapes and comes back turned a little, every substep: a free motor (evolution found it and
    rolled along at 6 m/s). This removes the pass's rotation about the body's middle.
    """
    w = point_on.astype(jnp.float32)[:, None]
    r = (before - jnp.sum(before * w, axis=0) / jnp.maximum(jnp.sum(w), 1.0)) * w
    d = (x - before) * w
    turned = jnp.sum(r[:, 0] * d[:, 1] - r[:, 1] * d[:, 0]) / (jnp.sum(r * r) + 1e-12)
    return x - turned * jnp.stack([-r[:, 1], r[:, 0]], axis=1)


def _push_balls(x, balls, point_on):
    """Joints and balls can't overlap: push each touching pair apart along the line between them, the
    lighter ball moving more. A ball moves by the average of its pushes. Returns which joints touch which
    balls, (N, B)."""
    d = balls[None, :, :] - x[:, None, :]                                   # (N, B, 2)
    dist = jnp.linalg.norm(d, axis=-1) + 1e-9
    overlap = jnp.maximum(RADIUS + BALL_RADIUS - dist, 0.0) * point_on[:, None]
    push = (overlap / dist)[..., None] * d
    share = (1 / BALL_WEIGHT) / (1 / BALL_WEIGHT + 1)
    balls = balls + share * jnp.sum(push, axis=0) / jnp.maximum(jnp.sum(overlap > 0, axis=0), 1)[:, None]
    return x - (1 - share) * jnp.sum(push, axis=1), balls, overlap > 0


def _substep(setup: Setup, x, px, target, balls=None):
    dt = DT / SUBSTEPS
    bones = setup.bones
    v = (x - px) * DAMPING ** (1 / SUBSTEPS)
    px = x
    x = x + v + jnp.array([0.0, -GRAVITY * dt * dt])
    x = jnp.where(setup.point_on[:, None], x, px)
    live = bones.link_on.astype(jnp.float32)
    joint_live = bones._replace(joint_on=bones.joint_on.astype(jnp.float32))
    if balls is not None:
        bx, bpx = balls
        bv = (bx - bpx) * DAMPING ** (1 / SUBSTEPS)
        bpx = bx
        bx = bx + bv + jnp.array([0.0, -GRAVITY * dt * dt])
        coming = jnp.linalg.norm(bx - bpx, axis=1)  # how far each would go this substep untouched
    else:
        bx = jnp.zeros((0, 2))

    def solve(_, carry):
        x, depth, bx, contact = carry
        before = x
        for idx in LINK_BATCHES:                # bones keep their length
            a, b = LINK_A[idx], LINK_B[idx]
            d = x[b] - x[a]
            dist = jnp.linalg.norm(d, axis=1) + 1e-9
            fix = (0.5 * (dist - CELL) / dist * live[idx])[:, None] * d   # move both ends half the error
            x = x.at[a].add(fix).at[b].add(-fix)
        for idx in JOINT_BATCHES:               # joints keep (or, for muscles, bend to) their angle
            x = _hold_angle(x, joint_live, target, idx)
        x = _no_turn(x, before, setup.point_on)
        if balls is not None:                   # balls push from outside, so they can turn the body
            x, bx, touch = _push_balls(x, bx, setup.point_on)
            bx = bx.at[:, 1].set(jnp.maximum(bx[:, 1], BALL_RADIUS))
            contact = contact | touch
        below = jnp.minimum(x[:, 1] - RADIUS, 0.0)
        return x.at[:, 1].set(x[:, 1] - below), depth - below, bx, contact

    x, depth, bx, contact = jax.lax.fori_loop(0, ITERATIONS, solve, (x, jnp.zeros(N), bx, jnp.zeros((N, len(bx)), bool)))
    if balls is not None:
        # Joints hold their angles with unlimited force, so pressing a ball into the ground or pinching it
        # between joints could shoot it out at any speed, and evolution found that at once. So a ball can
        # leave the joints it touched at most BALL_KICK times as fast as the fastest of them moved (a light
        # ball hit by a heavy limb), unless it was already going faster.
        hit = jnp.max(jnp.where(contact, jnp.linalg.norm(x - px, axis=1)[:, None], 0.0), axis=0)    # (B,)
        step = bx - bpx
        limit = jnp.maximum(coming, BALL_KICK * hit)
        scale = jnp.minimum(1.0, limit / (jnp.linalg.norm(step, axis=1) + 1e-9))
        bx = jnp.where(jnp.any(contact, axis=0)[:, None], bpx + step * scale[:, None], bx)
    # Every bone damps how fast its two ends move relative to each other, which stops the jiggling that
    # floppy bodies otherwise build up under their own weight (evolution found it and slid along on it).
    for idx in LINK_BATCHES:
        a, b = LINK_A[idx], LINK_B[idx]
        fix = (BONE_DAMPING / 2 * live[idx])[:, None] * ((x[b] - px[b]) - (x[a] - px[a]))
        px = px.at[a].add(-fix).at[b].add(fix)
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
    x = jnp.where(setup.point_on[:, None], x, px)
    if balls is None:
        return x, px
    # balls slide to a stop on the ground, and never move faster than BALL_MAX_SPEED
    slide = bx[:, 0] - bpx[:, 0]
    slowed = jnp.sign(slide) * jnp.maximum(jnp.abs(slide) - BALL_FRICTION * GRAVITY * dt * dt, 0.0)
    bx = bx.at[:, 0].set(jnp.where(bx[:, 1] <= BALL_RADIUS + 1e-6, bpx[:, 0] + slowed, bx[:, 0]))
    step = bx - bpx
    bx = bpx + step * jnp.minimum(1.0, BALL_MAX_SPEED * dt / (jnp.linalg.norm(step, axis=1) + 1e-9))[:, None]
    return x, px, (bx, bpx)


def feed(setup: Setup, state: State, x, act):
    """Eat fruit within reach, pay for living and moving, and put new fruit beyond a random end."""
    near_ground = (x[:, 1] < RADIUS + 0.08) & setup.point_on
    reach = jnp.abs(x[:, None, 0] - state.fruit[None, :]) < EAT_REACH                  # (N, FRUIT)
    eaten = jnp.any(reach & near_ground[:, None], axis=0)
    lost = jnp.abs(state.fruit - centre_x(x, setup.point_on)) > 6.0                    # wandered far away
    renew = eaten | lost
    number = state.placed + jnp.cumsum(renew) - 1                                     # each new fruit's draw
    fresh = jax.vmap(lambda k: _place(x, setup.point_on, setup.height, setup.world, k, 0.5, 2.5))(number)
    fruit = jnp.where(renew, fresh, state.fruit)
    working = jnp.sum(act * setup.muscle) / jnp.maximum(jnp.sum(setup.muscle), 1.0)
    drain = DT / IDLE_LIFE * (1 + 0.5 * jnp.maximum(setup.size - 1, 0.0) + MOVE_COST * working)
    energy = jnp.minimum(state.energy + FRUIT_ENERGY * jnp.sum(eaten) - drain, 1.0)
    return fruit, state.placed + jnp.sum(renew).astype(jnp.int32), energy, jnp.sum(eaten).astype(jnp.float32)


def tick(setup: Setup, brain: Brain, state: State):
    pace, touch, sight, hunger = senses(setup, state)
    charge, spikes, rate, act = think(brain, state, setup.nerve, pace, touch, sight, hunger)
    x, px = physics(setup, state.x, state.px, act)
    fruit, placed, energy, ate = feed(setup, state, x, act)
    new = State(x, px, charge, spikes, rate, act, energy, fruit, placed, state.eaten + ate,
                state.alive_ticks + 1, state.tick + 1)
    living = state.energy > 0
    # a dead creature stays where it fell
    return jax.tree_util.tree_map(lambda n, o: jnp.where(living, n, o), new, state._replace(tick=state.tick + 1))


# ---------- a whole life ----------

def ticks_for(seconds):
    return int(round(seconds / DT))


def life(body: Body, brain: Brain, world, seconds=EPISODE_SECONDS):
    setup, state = start(body, world)
    final, _ = jax.lax.scan(lambda s, _: (tick(setup, brain, s), None), state, None, length=ticks_for(seconds))
    return setup, final


def score(body: Body, brain: Brain, world, seconds=EPISODE_SECONDS):
    """Fruit eaten + the fraction of the time alive + a little for distance travelled."""
    setup, final = life(body, brain, world, seconds)
    travelled = jnp.abs(centre_x(final.x, setup.point_on) - setup.start_x)
    survived = final.alive_ticks / ticks_for(seconds)
    has_body = jnp.sum(body.alive) >= 3         # a growth rule that kills its seed gets nothing
    return jnp.where(has_body, final.eaten + survived + TRAVEL_CREDIT * jnp.minimum(travelled, TRAVEL_MAX), 0.0)


def record(body: Body, brain: Brain, world, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, world)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.fruit, s.energy, s.eaten)

    _, frames = jax.lax.scan(step, state, None, length=ticks_for(seconds))
    return setup, frames
