"""Evolve creatures that grow a body and a nervous system, and must find fruit to stay alive.

    python evolve.py --out runs/first                  # sized for a GPU
    python evolve.py --out runs/check --small          # a quick check on a CPU
    python evolve.py --out runs/throw --task throw     # another task: throw a ball as far as possible
    python evolve.py --out runs/balls --task balls     # and another: keep five balls in the air longest

Evolution starts from nothing designed: a single cell and a random growth rule. A random rule grows
nothing at all about half the time, so random rules are drawn until one grows a small body (3 to 30
joints) with nerves reaching at least two of its muscle joints. From there, an evolution strategy (OpenAI-ES): each generation, make POP copies of the
current genome with a little random noise added, grow and test every copy, and nudge the genome toward
the noise of the copies that scored best. When a copy beats the best walker so far on a re-test, the
search jumps to it: from a random start, good walkers are rare lucky finds, and one lucky copy barely
moves the average. All copies are grown and simulated in parallel (jax.vmap), which is where the GPU
earns its keep.

Scores are fruit eaten + the fraction of the time alive + a little for distance travelled (see sim.py),
with --task throw, the longest throw of a ball in metres (see throw.py), or with --task balls, five balls'
best times in the air added up, in seconds (see balls.py).
Each generation's best is re-tested in 4 fixed worlds, and the best of those is saved. The run folder gets best.npy (that best
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
import balls
import sim
import throw

TASKS = {'food': sim, 'throw': throw, 'balls': balls}

TEST_WORLDS = jnp.arange(1000, 1004)      # the re-test worlds: fruit placed differently in each


# ---------- the start: a random growth rule ----------

def random_start(key, scale=0.1, smallest=3, largest=30, attempts=5000):
    """A random growth rule that grows a small body (3 to 30 joints) from the single seed cell, with at
    least two muscle joints that a nerve reaches. A body that can't move gives evolution nothing to work
    with: in three test runs from starts without that, none moved more than 0.4 m in 120 generations."""
    def grows(g):
        body = cr.develop(g)[0]
        reached = jnp.any(cr.patches(body.nerve).reshape(cr.N, -1), axis=1)
        return jnp.sum(body.alive), jnp.sum(body.alive & (body.tissue == 1) & reached)
    grows = jax.jit(grows)
    for attempt in range(1, attempts + 1):
        key, k = jax.random.split(key)
        genome = cr.flatten(cr.init_params(k, scale))
        joints, wired = (int(n) for n in grows(genome))
        if smallest <= joints <= largest and wired >= 2:
            print(f'  random growth rule number {attempt} grows a small body with nerves reaching its muscles')
            return genome
    raise RuntimeError('no random growth rule grew a small body; try another --seed or --init-scale')


def adam(grad, m, v, t, lr, b1=0.9, b2=0.999):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    step = lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
    return step, m, v


# ---------- evolution ----------

def centered_ranks(x):
    ranks = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    return ranks / (x.size - 1) - 0.5


def evaluate(genome, world, seconds, task='food'):
    body, brain = cr.develop(genome)
    return TASKS[task].score(body, brain, world, seconds)


def make_generation(pop, sigma, lr, seconds, task='food', decay=0.005):
    half = pop // 2
    many = jax.vmap(lambda g, w: evaluate(g, w, seconds, task), in_axes=(0, None))

    @jax.jit
    def generation(mean, m, v, t, key):
        k_noise, k_world = jax.random.split(key)
        world = jax.random.randint(k_world, (), 0, 1_000_000)      # every member lives in the same world
        eps = jax.random.normal(k_noise, (half, cr.GENOME_SIZE))
        eps = jnp.concatenate([eps, -eps])                     # mirrored pairs cancel out noise
        genomes = mean + sigma * eps
        fitness = many(genomes, world)
        grad = -(centered_ranks(fitness) @ eps) / (pop * sigma) + decay * mean
        step, m, v = adam(grad, m, v, t, lr)
        return mean - step, m, v, fitness, genomes[jnp.argmax(fitness)]

    return generation


def make_test(seconds, task='food'):
    @jax.jit
    def test(genome):
        return jnp.mean(jax.vmap(lambda w: evaluate(genome, w, seconds, task))(TEST_WORLDS))
    return test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='runs/first')
    ap.add_argument('--pop', type=int, default=512, help='creatures per generation (even)')
    ap.add_argument('--gens', type=int, default=500)
    ap.add_argument('--sigma', type=float, default=0.02, help='size of the random changes')
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--task', choices=sorted(TASKS), default='food', help='find fruit, throw a ball, or keep five balls in the air')
    ap.add_argument('--seconds', type=float, help='length of each life (default: 20 for food, 10 for throw, 30 for balls)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', help='a mean.npy to continue from')
    ap.add_argument('--init-scale', type=float, default=0.1, help='size of the random starting growth rule\'s weights')
    ap.add_argument('--small', action='store_true', help='tiny settings for a CPU check')
    args = ap.parse_args()
    if args.seconds is None:
        args.seconds = TASKS[args.task].EPISODE_SECONDS
    if args.small:
        args.pop, args.gens, args.seconds = 32, 15, min(args.seconds, 8.0)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'JAX devices: {jax.devices()}')
    key = jax.random.PRNGKey(args.seed)

    if args.resume:
        mean = jnp.asarray(np.load(args.resume))
    else:
        print('Drawing a random growth rule…')
        key, k = jax.random.split(key)
        mean = random_start(k, args.init_scale)
    body, _ = cr.develop(mean)
    print('Where evolution starts (B bone, M muscle joint, E eye; lower case where a nerve runs):\n' + cr.describe(body))
    print(f'{int(body.alive.sum())} joints, {int(body.nerve.sum())} nerve cells')

    generation = make_generation(args.pop, args.sigma, args.lr, args.seconds, args.task)
    test = make_test(args.seconds, args.task)
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
            mean = champion                     # a new champion: carry on searching from it
        np.save(os.path.join(args.out, 'mean.npy'), np.asarray(mean))
        took = time.time() - t0
        log.write(f'{gen},{float(fitness.max()):.4f},{float(fitness.mean()):.4f},{champion_test:.4f},{best_score:.4f},{took:.2f}\n')
        log.flush()
        print(f'gen {gen:4d}  score: best {float(fitness.max()):5.2f}  mean {float(fitness.mean()):5.2f}  '
              f'best re-tested {champion_test:5.2f}  best so far {best_score:5.2f}  ({took:.1f}s)')
    task = f' --task {args.task}' if args.task != 'food' else ''
    print(f'Done. Draw the best creature with:  python render.py {os.path.join(args.out, "best.npy")}{task}')


if __name__ == '__main__':
    main()
