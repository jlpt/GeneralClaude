"""A third task: keep five balls in the air as long as possible, one throw at a time.

After the creature has settled, one ball is dropped onto the top of its body, as in the throwing task
(throw.py), and four more lie on the ground 0.14–1.44 m beyond its left or right end (which end and how
far depending on the world number). The creature has no hands: to throw a ball it has to fling it with
its body, and to throw the ones on the ground it has to get to them and get them up.

Score: for each ball, its longest time in the air (more than AIR_HEIGHT above the ground) after it last
left the creature, added up over the five balls, in seconds, over a 30 s life, plus FETCH_CREDIT for each
ball on the ground the creature has touched.
    - each ball counts only its best throw, so juggling one ball over and over can't add up: to score
      more, the creature has to fetch another ball
    - the fetch credit is a first step toward the balls on the ground. Without it, four runs of 60
      generations only ever threw the ball on their back: reaching a ball earned nothing until it was
      in the air
    - carrying a ball, or shoving it along the ground, earns nothing
    - the ball on its back gives evolution a first step: even a ball that rolls off spends a moment in
      the air, while no random creature ever lifts a ball off the ground
A body of fewer than 3 joints scores 0.

Balls pass through each other. Senses: touch is a joint touching the ground or any ball; eyes see the
balls ahead of them, nearer brighter, added up like fruit; the pacemaker is the same; hunger stays at 0.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import sim
from creature import H, W, spread
from sim import BALL_RADIUS, RADIUS
from throw import AIR_HEIGHT, BALL_DRAW, BALL_SETTLE_SECONDS

EPISODE_SECONDS = 30.0
BALLS = 5
GROUND_DRAW = 3_000_000     # ground ball k (from 1) uses draws GROUND_DRAW + 2k (side) and + 2k + 1 (distance)
FETCH_CREDIT = 0.1          # seconds of score for touching each ball that started on the ground


class Juggle(NamedTuple):
    x: jnp.ndarray          # (N, 2) joint positions
    px: jnp.ndarray
    charge: jnp.ndarray     # (HF, WF)
    spikes: jnp.ndarray
    rate: jnp.ndarray
    act: jnp.ndarray        # (N,)
    balls: jnp.ndarray      # (BALLS, 2)
    balls_px: jnp.ndarray   # (BALLS, 2) positions one substep ago
    air: jnp.ndarray        # (BALLS,) seconds each has been in the air since it last left the creature
    best: jnp.ndarray       # (BALLS,) each one's longest time in the air
    reached: jnp.ndarray    # (BALLS,) the creature has touched it (the first ball starts on its back)
    tick: jnp.ndarray


def start(body, world):
    """Settle the creature as in the food task, drop one ball onto its top and lay the rest on the ground
    beyond its ends, then let the balls settle."""
    setup, s = sim.start(body, world)
    on = setup.point_on
    k = jnp.floor(sim.chance(world, BALL_DRAW) * jnp.sum(on))
    column = jnp.argmax(on & (jnp.cumsum(on) == k + 1))
    top = jnp.max(jnp.where(on, s.x[:, 1], 0.0))
    left = jnp.min(jnp.where(on, s.x[:, 0], jnp.inf))
    right = jnp.max(jnp.where(on, s.x[:, 0], -jnp.inf))
    n = jnp.arange(1, BALLS)
    gap = RADIUS + BALL_RADIUS + 0.05 + 1.3 * sim.chance(world, GROUND_DRAW + 2 * n + 1)
    ground_x = jnp.where(sim.chance(world, GROUND_DRAW + 2 * n) < 0.5, right + gap, left - gap)
    balls = jnp.concatenate([jnp.array([[s.x[column, 0], top + RADIUS + BALL_RADIUS + 0.02]]),
                             jnp.stack([ground_x, jnp.full(BALLS - 1, BALL_RADIUS)], axis=1)])
    relaxed = jnp.zeros_like(s.act)
    x, px, (balls, balls_px) = jax.lax.fori_loop(
        0, sim.ticks_for(BALL_SETTLE_SECONDS), lambda _, c: sim.physics(setup, c[0], c[1], relaxed, c[2]),
        (s.x, s.px, (balls, balls)))
    zeros = jnp.zeros(BALLS)
    return setup, Juggle(x, px, s.charge, s.spikes, s.rate, s.act, balls, balls_px, zeros, zeros, jnp.zeros(BALLS, bool),
                         jnp.int32(0))


def touching(setup, x, balls):
    """(N, BALLS): which joints touch which balls."""
    near = jnp.linalg.norm(x[:, None, :] - balls[None, :, :], axis=-1) < RADIUS + BALL_RADIUS + 0.01
    return near & setup.point_on[:, None]


def senses(setup, state: Juggle):
    x = state.x
    touch = (x[:, 1] < RADIUS + 0.01) & setup.point_on | jnp.any(touching(setup, x, state.balls), axis=1)
    middle = sim.centre_x(x, setup.point_on)
    facing = jnp.where(x[:, 0] >= middle, 1.0, -1.0)             # eyes look away from the body's middle
    ahead = (state.balls[None, :, 0] - x[:, :1]) * facing[:, None]
    brightness = jnp.sum(jnp.where(ahead > 0, jnp.exp(-ahead / sim.SIGHT_RANGE), 0.0), axis=1)
    sight = (1 - jnp.exp(-2 * brightness)) * setup.eye
    pace = jnp.sin(2 * jnp.pi * sim.PACE_HZ * state.tick * sim.DT)
    return pace, spread(touch.reshape(H, W)).astype(jnp.float32), spread(sight.reshape(H, W)), 0.0


def tick(setup, brain, state: Juggle):
    pace, touch, sight, hunger = senses(setup, state)
    charge, spikes, rate, act = sim.think(brain, state, setup.nerve, pace, touch, sight, hunger)
    x, px, (balls, balls_px) = sim.physics(setup, state.x, state.px, act, (state.balls, state.balls_px))
    held = jnp.any(touching(setup, x, balls), axis=0)
    airborne = balls[:, 1] > BALL_RADIUS + AIR_HEIGHT
    air = jnp.where(held, 0.0, state.air + jnp.where(airborne, sim.DT, 0.0))
    best = jnp.maximum(state.best, air)
    return Juggle(x, px, charge, spikes, rate, act, balls, balls_px, air, best, state.reached | held, state.tick + 1)


def life(body, brain, world, seconds=EPISODE_SECONDS):
    setup, state = start(body, world)
    final, _ = jax.lax.scan(lambda s, _: (tick(setup, brain, s), None), state, None, length=sim.ticks_for(seconds))
    return setup, final


def score(body, brain, world, seconds=EPISODE_SECONDS):
    """Each ball's longest time in the air, added up, in seconds, plus a little for each ground ball reached."""
    _, final = life(body, brain, world, seconds)
    total = jnp.sum(final.best) + FETCH_CREDIT * jnp.sum(final.reached[1:])
    return jnp.where(jnp.sum(body.alive) >= 3, total, 0.0)


def record(body, brain, world, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, world)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.balls, s.best)

    _, frames = jax.lax.scan(step, state, None, length=sim.ticks_for(seconds))
    return setup, frames
