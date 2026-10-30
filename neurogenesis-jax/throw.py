"""A second task for the same creatures: throw a ball as far as it will go.

After the creature has settled, a ball is dropped onto the top of its body (above which joint depends on
the world number) and given a second to come to rest, usually in a dip between two joints. The creature
has no hands, so to throw it has to fling the ball off with a limb or with its whole body. The score is
the longest throw over a 10 s life: the horizontal distance the ball travels through the air (more than
AIR_HEIGHT above the ground) after it last left the creature, in either direction.
    - carrying the ball while walking earns nothing, since a throw counts only once the ball has left
    - rolling, kicking or shoving the ball along the ground earns nothing. Evolution tried both: with the
      ball starting on the ground and ground distance counting at all, its best "throwers" kicked the ball
      along the ground at 7 m/s and never lifted it, and with only flight counting, nothing scored at all
      (no random creature ever lifted the ball), so evolution had nothing to start from. Starting with the
      ball on the creature fixes both: even a ball that rolls off travels a little through the air
    - only the best throw counts, so the creature can pick the ball up and try again
A body of fewer than 3 joints scores 0.

The body, nervous system and physics are the food task's (sim.py), with the ball added to the physics.
Senses here: touch is a joint touching the ground or the ball; eyes see the ball ahead of them, nearer
brighter; the pacemaker is the same; there is nothing to eat, so hunger stays at 0.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import sim
from creature import H, W, spread
from sim import BALL_RADIUS, RADIUS

EPISODE_SECONDS = 10.0
BALL_DRAW = 2_000_000       # chance(world, BALL_DRAW) picks the joint the ball is dropped above
AIR_HEIGHT = 0.01           # metres the ball must be above the ground for its travel to count
BALL_SETTLE_SECONDS = 1.0


class Throw(NamedTuple):
    x: jnp.ndarray          # (N, 2) joint positions
    px: jnp.ndarray
    charge: jnp.ndarray     # (HF, WF)
    spikes: jnp.ndarray
    rate: jnp.ndarray
    act: jnp.ndarray        # (N,)
    ball: jnp.ndarray       # (2,) ball position
    ball_px: jnp.ndarray    # (2,) its position one substep ago
    release: jnp.ndarray    # the ball's x when it last left the creature
    air: jnp.ndarray        # metres it has travelled through the air since then
    best: jnp.ndarray       # the longest throw so far
    tick: jnp.ndarray


def start(body, world):
    """Settle the creature as in the food task, drop the ball onto its top, and let the ball settle too."""
    setup, s = sim.start(body, world)
    on = setup.point_on
    # the ball falls from above the whole body, in the column of one of its joints
    k = jnp.floor(sim.chance(world, BALL_DRAW) * jnp.sum(on))
    column = jnp.argmax(on & (jnp.cumsum(on) == k + 1))
    top = jnp.max(jnp.where(on, s.x[:, 1], 0.0))
    ball = jnp.array([s.x[column, 0], top + RADIUS + BALL_RADIUS + 0.02])
    relaxed = jnp.zeros_like(s.act)
    x, px, (balls, balls_px) = jax.lax.fori_loop(
        0, sim.ticks_for(BALL_SETTLE_SECONDS), lambda _, c: sim.physics(setup, c[0], c[1], relaxed, c[2]),
        (s.x, s.px, (ball[None], ball[None])))
    ball, ball_px = balls[0], balls_px[0]
    zero = jnp.float32(0.0)
    return setup, Throw(x, px, s.charge, s.spikes, s.rate, s.act, ball, ball_px, ball[0], zero, zero, jnp.int32(0))


def touching(setup, x, ball):
    return (jnp.linalg.norm(x - ball, axis=1) < RADIUS + BALL_RADIUS + 0.01) & setup.point_on


def senses(setup, state: Throw):
    x = state.x
    touch = (x[:, 1] < RADIUS + 0.01) & setup.point_on | touching(setup, x, state.ball)
    middle = sim.centre_x(x, setup.point_on)
    facing = jnp.where(x[:, 0] >= middle, 1.0, -1.0)             # eyes look away from the body's middle
    ahead = (state.ball[0] - x[:, 0]) * facing
    sight = (1 - jnp.exp(-2 * jnp.where(ahead > 0, jnp.exp(-ahead / sim.SIGHT_RANGE), 0.0))) * setup.eye
    pace = jnp.sin(2 * jnp.pi * sim.PACE_HZ * state.tick * sim.DT)
    return pace, spread(touch.reshape(H, W)).astype(jnp.float32), spread(sight.reshape(H, W)), 0.0


def tick(setup, brain, state: Throw):
    pace, touch, sight, hunger = senses(setup, state)
    charge, spikes, rate, act = sim.think(brain, state, setup.nerve, pace, touch, sight, hunger)
    x, px, (balls, balls_px) = sim.physics(setup, state.x, state.px, act, (state.ball[None], state.ball_px[None]))
    ball, ball_px = balls[0], balls_px[0]
    held = jnp.any(touching(setup, x, ball))
    airborne = ball[1] > BALL_RADIUS + AIR_HEIGHT
    release = jnp.where(held, ball[0], state.release)
    air = jnp.where(held, 0.0, state.air + jnp.where(airborne, jnp.abs(ball[0] - state.ball[0]), 0.0))
    best = jnp.maximum(state.best, air)
    return Throw(x, px, charge, spikes, rate, act, ball, ball_px, release, air, best, state.tick + 1)


def life(body, brain, world, seconds=EPISODE_SECONDS):
    setup, state = start(body, world)
    final, _ = jax.lax.scan(lambda s, _: (tick(setup, brain, s), None), state, None, length=sim.ticks_for(seconds))
    return setup, final


def score(body, brain, world, seconds=EPISODE_SECONDS):
    """The longest throw, in metres."""
    _, final = life(body, brain, world, seconds)
    return jnp.where(jnp.sum(body.alive) >= 3, final.best, 0.0)


def record(body, brain, world, seconds=EPISODE_SECONDS):
    """A life tick by tick, for drawing."""
    setup, state = start(body, world)

    def step(s, _):
        s = tick(setup, brain, s)
        return s, (s.x, s.act, s.spikes, s.ball, s.release, s.best)

    _, frames = jax.lax.scan(step, state, None, length=sim.ticks_for(seconds))
    return setup, frames
