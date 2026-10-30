"""Evolve creatures that grow a body and a nervous system, and walk forward.

    python evolve.py --out runs/first            # sized for a GPU
    python evolve.py --out runs/check --small    # a quick check on a CPU

Two stages:

1. Starter body. Gradient descent trains the growth rule to grow a simple walker: a long, low body with a
   leg near each end, each leg with a hip and a knee joint, and inside it a small brain with a nerve
   running from it down each leg through the hip to the knee. The pacemaker reaches the two legs'
   nerves with opposite signs, so the legs take turns. Evolution starts from there rather than from
   shapeless random growth.

2. Evolution strategy (OpenAI-ES). Each generation, make POP copies of the current genome with a little
   random noise added, grow and test every copy, and nudge the genome toward the noise of the copies
   that walked farthest. All copies are grown and simulated in parallel (jax.vmap), which is where the
   GPU earns its keep.

Scores are metres walked to the right. Each generation's farthest walker is re-tested 4 times (with
different starting nudges), and the best of those is saved. The run folder gets best.npy (that best
creature), mean.npy (the current genome, to resume from), log.csv and config.json. Draw one with
render.py.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import creature as cr
import sim

TEST_WORLDS = jnp.arange(1000, 1004)      # the re-test runs: each starts with a different nudge


# ---------- stage 1: the starter body ----------

def starter_target():
    """A long, low body (bone) with a leg near each end. Each leg is a line of cells from the body to the
    foot: a muscle joint at the hip, bone, a muscle joint at the knee, then bone. Hips bend one way and
    knees the other. Inside, a nervous system: a brain of six nerve cells in the middle of the body, and a
    nerve from it along the body and down each leg through the hip to the knee. The pacemaker reaches the
    two legs' nerves with opposite signs, so the legs take turns; evolution can change all of it."""
    alive, nerve, bend, pace = (np.zeros((cr.H, cr.W), np.float32) for _ in range(4))
    tissue = np.zeros((cr.H, cr.W), np.int32)
    r0, c0 = cr.SEED
    alive[r0 - 1:r0 + 1, c0 - 5:c0 + 6] = 1                  # body
    brain = [(r, c) for r in (r0 - 1, r0) for c in (c0 - 1, c0, c0 + 1)]
    wiring = brain
    for c, side, sign in ((c0 - 4, -1, 1), (c0 + 4, 1, -1)):  # back leg, front leg
        alive[r0 + 1:, c] = 1
        tissue[r0 + 1, c] = tissue[r0 + 3, c] = 1             # hip and knee
        bend[r0 + 1, c], bend[r0 + 3, c] = 1, -1
        leg = [(r, c) for r in (r0 + 1, r0 + 2, r0 + 3)]
        wiring = wiring + [(r0, c0 + side * k) for k in (2, 3, 4)] + leg
        for r, cc in leg:
            pace[r, cc] = sign
    for r, c in wiring:
        nerve[r, c] = 1
    return (jnp.asarray(alive), jnp.asarray(tissue), jnp.asarray(nerve), jnp.asarray(bend), jnp.asarray(pace))


def adam(grad, m, v, t, lr, b1=0.9, b2=0.999):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    step = lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
    return step, m, v


def train_starter(key, steps=1500, lr=2e-3, attempts=8):
    alive_t, tissue_t, nerve_t, bend_t, pace_t = starter_target()
    onehot = jax.nn.one_hot(tissue_t, 2)
    muscle_t = onehot[..., 1] * alive_t
    paced = (pace_t != 0).astype(jnp.float32)

    def loss(genome):
        s = cr.grow(cr.unflatten(genome))
        shape = jnp.mean((jnp.clip(s[..., cr.ALIVE], 0, 1) - alive_t) ** 2)
        logp = jax.nn.log_softmax(s[..., [cr.BONE, cr.MUSCLE]])
        kind = -jnp.sum(jnp.sum(logp * onehot, -1) * alive_t) / jnp.sum(alive_t)
        wiring = jnp.sum((jnp.tanh(s[..., cr.NERVE]) - 0.8 * (2 * nerve_t - 1)) ** 2 * alive_t) / jnp.sum(alive_t)
        rhythm = jnp.sum((jnp.tanh(s[..., cr.PACE]) - 0.8 * pace_t) ** 2 * paced) / jnp.sum(paced)
        turn = jnp.sum((jnp.tanh(s[..., cr.BEND]) - 0.8 * bend_t) ** 2 * muscle_t) / jnp.sum(muscle_t)
        return 10 * shape + 0.3 * kind + 0.3 * wiring + 0.1 * rhythm + 0.1 * turn

    def grows_right(genome):
        body = cr.read_body(cr.grow(cr.unflatten(genome)))
        alive = np.asarray(body.alive).reshape(cr.H, cr.W)
        tissue = np.asarray(body.tissue).reshape(cr.H, cr.W)
        nerve = np.asarray(body.nerve).reshape(cr.H, cr.W)
        return (np.array_equal(alive, np.asarray(alive_t) > 0) and np.array_equal(nerve, np.asarray(nerve_t) > 0)
                and np.array_equal(tissue[alive], np.asarray(tissue_t)[alive]))

    grad_fn = jax.jit(jax.value_and_grad(loss))
    # Some random starts miss a detail (a nerve cell too few, a muscle grown as bone), so try again from
    # another start until one grows exactly the target.
    best, best_loss = None, np.inf
    for attempt in range(1, attempts + 1):
        key, k = jax.random.split(key)
        # Start with a rule that changes nothing, so the seed survives and the gradient has something to work on.
        params = cr.init_params(k)
        params['W2'] = jnp.zeros_like(params['W2'])
        genome = cr.flatten(params)
        m = v = jnp.zeros_like(genome)
        mine, mine_loss = genome, np.inf
        for t in range(1, steps + 1):
            value, g = grad_fn(genome)
            if value < mine_loss:
                mine, mine_loss = genome, float(value)
            g = g * jnp.minimum(1.0, 1.0 / (jnp.linalg.norm(g) + 1e-9))     # clipped, or thin nerves get lost
            step, m, v = adam(g, m, v, t, lr)
            genome = genome - step
        right = grows_right(mine)
        print(f'  starter body, attempt {attempt}: loss {mine_loss:.4f}, {"grows the target" if right else "not quite right"}')
        if right:
            return mine
        if mine_loss < best_loss:
            best, best_loss = mine, mine_loss
    return best


# ---------- stage 2: evolution ----------

def centered_ranks(x):
    ranks = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    return ranks / (x.size - 1) - 0.5


def evaluate(genome, key, seconds):
    body, brain = cr.develop(genome)
    return sim.score(body, brain, key, seconds)


def make_generation(pop, sigma, lr, seconds, decay=0.005):
    half = pop // 2
    many = jax.vmap(evaluate, in_axes=(0, None, None))

    @jax.jit
    def generation(mean, m, v, t, key):
        k_noise, k_world = jax.random.split(key)
        eps = jax.random.normal(k_noise, (half, cr.GENOME_SIZE))
        eps = jnp.concatenate([eps, -eps])                     # mirrored pairs cancel out noise
        genomes = mean + sigma * eps
        fitness = many(genomes, k_world, seconds)
        grad = -(centered_ranks(fitness) @ eps) / (pop * sigma) + decay * mean
        step, m, v = adam(grad, m, v, t, lr)
        return mean - step, m, v, fitness, genomes[jnp.argmax(fitness)]

    return generation


def make_test(seconds):
    @jax.jit
    def test(genome):
        return jnp.mean(jax.vmap(lambda k: evaluate(genome, jax.random.PRNGKey(k), seconds))(TEST_WORLDS))
    return test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='runs/first')
    ap.add_argument('--pop', type=int, default=512, help='creatures per generation (even)')
    ap.add_argument('--gens', type=int, default=500)
    ap.add_argument('--sigma', type=float, default=0.01, help='size of the random changes (0.01 changes about 4 cells of the starter body)')
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--seconds', type=float, default=sim.EPISODE_SECONDS, help='length of each life')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', help='a mean.npy to continue from')
    ap.add_argument('--small', action='store_true', help='tiny settings for a CPU check')
    args = ap.parse_args()
    if args.small:
        args.pop, args.gens, args.seconds = 32, 15, 8.0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'JAX devices: {jax.devices()}')
    key = jax.random.PRNGKey(args.seed)

    if args.resume:
        mean = jnp.asarray(np.load(args.resume))
    else:
        print('Training the starter body…')
        key, k = jax.random.split(key)
        mean = train_starter(k)
    body, _ = cr.develop(mean)
    print('Starter body (B bone, M muscle joint; lower case where a nerve runs):\n' + cr.describe(body))

    generation = make_generation(args.pop, args.sigma, args.lr, args.seconds)
    test = make_test(args.seconds)
    m = v = jnp.zeros_like(mean)
    best_score = -np.inf
    log = open(os.path.join(args.out, 'log.csv'), 'a')
    log.write('generation,best,population_mean,champion_test,best_test,seconds\n')
    print(f'Evolving {args.pop} creatures per generation, {args.seconds:g} s lives. First generation compiles, so it is slow.')
    for gen in range(1, args.gens + 1):
        t0 = time.time()
        key, k = jax.random.split(key)
        mean, m, v, fitness, champion = generation(mean, m, v, gen, k)
        champion_test = float(test(champion))
        if champion_test > best_score:
            best_score = champion_test
            np.save(os.path.join(args.out, 'best.npy'), np.asarray(champion))
        np.save(os.path.join(args.out, 'mean.npy'), np.asarray(mean))
        took = time.time() - t0
        log.write(f'{gen},{float(fitness.max()):.4f},{float(fitness.mean()):.4f},{champion_test:.4f},{best_score:.4f},{took:.2f}\n')
        log.flush()
        print(f'gen {gen:4d}  metres walked: best {float(fitness.max()):5.2f}  mean {float(fitness.mean()):5.2f}  '
              f'best re-tested {champion_test:5.2f}  best so far {best_score:5.2f}  ({took:.1f}s)')
    print(f'Done. Draw the best creature with:  python render.py {os.path.join(args.out, "best.npy")}')


if __name__ == '__main__':
    main()
