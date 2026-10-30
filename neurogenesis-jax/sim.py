"""The world: a side view with flat ground. The creature's job is to walk forward (to the right).

Body physics. Each living cell is a square voxel; its four corners are point masses. Neighbouring
voxels share corners. Every voxel edge and diagonal is a distance constraint, solved with position-
based dynamics. The constraints are split into 8 batches in which no two share a corner, so each batch
is solved fully in parallel (good for a GPU) and later batches see earlier fixes, which converges fast.
A muscle voxel shrinks its edges by up to MUSCLE_AMP as it contracts. The ground pushes points up.
Friction follows Coulomb's law: the body's weight is shared among the points touching the ground, and a
loaded point sticks unless pushed hard, so to travel a creature has to lift and push.

Nervous system. Only nerve cells are neurons: leaky integrate-and-fire, each wired to the nerve cells
touching it. Each tick:
    charge = LEAK * charge + bias + senses + spikes @ weights
and a nerve cell spikes (and resets) when its charge reaches its threshold. A muscle voxel contracts
only when the nerve cell inside it fires: it follows that nerve cell's recent firing rate times its
junction strength, with about a 0.2 s lag, as real muscle does. A muscle with no nerve never moves.

Senses. Each is a current into the nerve cells that can feel it, scaled by each one's gain for it:
    touch      nerve cells in voxels touching the ground
    pacemaker  every nerve cell: a steady rhythm, sin(2π · PACE_HZ · t), like the heart's pacemaker
So a foot's touch only matters if a nerve carries the signal on to something.

Score. Before the clock starts, the creature settles for SETTLE_SECONDS with its muscles off, after a
tiny random sideways nudge, so a body that would topple over has already done so. The score is how far
its middle then moves to the right in EPISODE_SECONDS. Falling over ends the run: once the body has
turned more than MAX_TILT from the pose it settled in, the score stays at the distance reached before
the fall (otherwise evolution grows tall bodies that fall forward and lie there). A body of fewer than 3
cells scores 0.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from creature import H, W, N, Body, Brain

# body physics
VOXEL = 0.1                 # metres per cell
DT = 1 / 30                 # seconds per tick (nervous system and physics)
GRAVITY = 9.8
DAMPING = 0.99
SUBSTEPS = 3                # physics steps per tick
ITERATIONS = 4              # passes over all 8 constraint batches per substep
MUSCLE_AMP = 0.3            # a fully active muscle voxel shrinks its edges by 30%
MUSCLE_RESPONSE = 0.15      # fraction of the way a muscle moves toward its drive each tick
STATIC_FRICTION, KINETIC_FRICTION = 1.2, 0.8

# nervous system
LEAK = 0.8
PACE_HZ = 1.5

# the test
SETTLE_SECONDS = 3.0        # time to fall into a resting pose before the clock starts
NUDGE = 0.09                # m/s sideways push at the start of settling, so balanced-on-a-knife-edge bodies fall too
MAX_TILT = np.radians(45)   # turning further than this from the settled pose counts as falling over
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
    px: jnp.ndarray         # (P, 2) positions one substep ago (velocity = x − px)
    charge: jnp.ndarray     # (N,)
    spikes: jnp.ndarray     # (N,)
    rate: jnp.ndarray       # (N,) recent firing rate
    act: jnp.ndarray        # (N,) how contracted each muscle is
    walked: jnp.ndarray     # metres walked to the right, up to now or up to a fall
    fallen: jnp.ndarray
    tick: jnp.ndarray


class Setup(NamedTuple):
    """Everything about a creature that stays fixed during its life."""
    point_on: jnp.ndarray   # (P,) corner belongs to the body
    spring_on: jnp.ndarray  # (S,) constraint belongs to the body
    voxel_on: jnp.ndarray   # (N+1,) padded with a dead voxel at index N
    muscle: jnp.ndarray     # (N+1,)
    nerve: jnp.ndarray      # (N,) a nerve cell grows in this voxel
    start_x: jnp.ndarray
    pose: jnp.ndarray       # (P, 2) the settled shape around its middle, to tell how far the body has turned


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
    voxel_on = jnp.concatenate([body.alive, jnp.zeros(1, bool)])
    point_on = jnp.any(voxel_on[OWNERS], axis=1)
    spring_on = voxel_on[SPRING_V1] | voxel_on[SPRING_V2]
    muscle = jnp.concatenate([body.alive & (body.tissue == 1), jnp.zeros(1, bool)])
    setup = Setup(point_on, spring_on, voxel_on, muscle, body.nerve, jnp.float32(0.0), jnp.zeros((P, 2)))
    lowest = jnp.min(jnp.where(point_on, HOME[:, 1], 1e3))
    x = jnp.asarray(HOME) - jnp.array([0.0, lowest])
    x = x - jnp.array([centre_x(x, point_on), 0.0])
    push = jnp.where(jax.random.bernoulli(key), 1.0, -1.0) * NUDGE * DT / SUBSTEPS   # x − px is one substep
    relaxed = jnp.zeros(N)
    x, px = jax.lax.fori_loop(0, ticks_for(SETTLE_SECONDS), lambda _, xs: physics(setup, *xs, relaxed),
                              (x, x - jnp.array([push, 0.0])))
    zeros = jnp.zeros(N)
    pose = (x - _middle(x, point_on)) * point_on[:, None]
    setup = setup._replace(start_x=centre_x(x, point_on), pose=pose)
    return setup, State(x, px, zeros, zeros, zeros, zeros, jnp.float32(0.0), jnp.bool_(False), jnp.int32(0))


# ---------- one tick ----------

def senses(setup: Setup, state: State):
    touch = jnp.any(state.x[CORNERS, 1] < 0.01, axis=1) & setup.voxel_on[:N]
    pace = jnp.sin(2 * jnp.pi * PACE_HZ * state.tick * DT)
    return pace, touch.astype(jnp.float32)


def think(brain: Brain, state: State, nerve, pace, touch):
    g = brain.gains
    current = brain.bias + g[:, 0] * pace + g[:, 1] * touch
    charge = LEAK * state.charge + current + state.spikes @ brain.weights
    spikes = ((charge >= brain.threshold) & nerve).astype(jnp.float32)
    charge = jnp.maximum(jnp.where(spikes > 0, 0.0, charge), -3.0)
    rate = 0.7 * state.rate + 0.3 * spikes
    # a muscle follows the nerve cell inside it (junction is 0 where there's no nerve)
    act = state.act + (rate * brain.junction - state.act) * MUSCLE_RESPONSE
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
    # Coulomb friction. The body's weight is shared among the points touching the ground, in proportion to
    # how hard the ground pushed back on each; friction can stop a point sliding up to the share it carries.
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
    walked = jnp.where(fallen, state.walked, centre_x(x, setup.point_on) - setup.start_x)
    return State(x, px, charge, spikes, rate, act, walked, fallen, state.tick + 1)


# ---------- a whole life ----------

def ticks_for(seconds):
    return int(round(seconds / DT))


def life(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    setup, state = start(body, key)
    final, _ = jax.lax.scan(lambda s, _: (tick(setup, brain, s), None), state, None, length=ticks_for(seconds))
    return setup, final


def score(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """Metres walked to the right before the time ran out or it fell over."""
    setup, final = life(body, brain, key, seconds)
    has_body = jnp.sum(body.alive) >= 3         # a growth rule that kills its seed gets nothing
    return jnp.where(has_body, final.walked, 0.0)


def record(body: Body, brain: Brain, key, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, key)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.walked, s.fallen)

    _, frames = jax.lax.scan(step, state, None, length=ticks_for(seconds))
    return setup, frames
