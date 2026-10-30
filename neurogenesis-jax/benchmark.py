"""Is JAX using the GPU, and how fast does evolution run here? Times one generation at several sizes.

    python benchmark.py

Each life is tens of thousands of small physics steps that must happen one after another, and a GPU
pays a fixed cost for each step no matter how many creatures share it. So on a GPU the time per
generation is mostly that fixed cost, and bigger populations come almost free: use the largest --pop
whose time per generation you're happy with.
"""

import time

import jax
import jax.numpy as jnp

import creature as cr
import evolve
import sim


def main():
    device = jax.devices()[0]
    print(f'JAX {jax.__version__} is using: {device.platform.upper()} ({device.device_kind})')
    on_cpu = device.platform == 'cpu'
    if on_cpu:
        print('No GPU found, so this runs on the CPU. See the GPU install steps in the README.')
    mean = cr.flatten(cr.init_params(jax.random.PRNGKey(0)))
    zeros = jnp.zeros_like(mean)
    for pop in ([32, 64] if on_cpu else [256, 1024, 4096, 8192]):
        generation = evolve.make_generation(pop, 0.01, 0.005, sim.EPISODE_SECONDS)
        try:
            t0 = time.time()
            jax.block_until_ready(generation(mean, zeros, zeros, 1, jax.random.PRNGKey(1)))
            first = time.time() - t0
            t0 = time.time()
            jax.block_until_ready(generation(mean, zeros, zeros, 2, jax.random.PRNGKey(2)))
            took = time.time() - t0
        except Exception as e:                      # most likely out of GPU memory
            print(f'{pop:5d} creatures: failed ({type(e).__name__}); use a smaller --pop')
            break
        print(f'{pop:5d} creatures per generation: {took:6.2f} s each, {pop / took:7.0f} creatures/s '
              f'(the first one also compiles: {first:.1f} s)')


if __name__ == '__main__':
    main()
