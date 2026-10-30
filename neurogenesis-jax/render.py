"""Draw a creature growing from its seed cell and then living, as an animated GIF.

    python render.py runs/first/best.npy                 # writes creature.gif
    python render.py runs/first/best.npy --out walk.gif --seconds 30 --world 7

Colours: bone is pale, muscle turns from pink to deep red as it contracts, eyes are blue, a yellow dot
means that cell's neuron fired on this tick, and fruit is green. Fruit off the edge of the view is shown
as a green tick at the edge.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import creature as cr
import sim

SIZE = (640, 360)
SKY = (245, 243, 237)
GRID = (226, 222, 213)
GROUND = (110, 96, 80)
INK = (38, 34, 46)
MUTED = (120, 114, 128)
BONE, BONE_EDGE = (228, 219, 199), (128, 116, 98)
MUSCLE_SOFT, MUSCLE_HARD = (222, 160, 150), (190, 34, 26)
EYE, PUPIL = (52, 128, 206), (16, 24, 40)
FRUIT = (58, 166, 72)
SPIKE = (255, 206, 0)
FONT = ImageFont.load_default(size=14)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def tissue_colour(kind, act=0.0):
    return [BONE, mix(MUSCLE_SOFT, MUSCLE_HARD, act * 1.6), EYE][kind]


def growth_frames(genome):
    states = np.asarray(cr.grow_history(cr.unflatten(genome)))
    cell = 22
    ox, oy = (SIZE[0] - cr.W * cell) // 2, (SIZE[1] - cr.H * cell) // 2 + 10
    frames = []
    for k, s in enumerate(states):
        img = Image.new('RGB', SIZE, SKY)
        d = ImageDraw.Draw(img)
        for r in range(cr.H + 1):
            d.line([(ox, oy + r * cell), (ox + cr.W * cell, oy + r * cell)], fill=GRID)
        for c in range(cr.W + 1):
            d.line([(ox + c * cell, oy), (ox + c * cell, oy + cr.H * cell)], fill=GRID)
        alive = s[..., cr.ALIVE] > 0.1
        kind = np.argmax(s[..., [cr.BONE, cr.MUSCLE, cr.EYE]], axis=-1)
        for r in range(cr.H):
            for c in range(cr.W):
                if alive[r, c]:
                    box = [ox + c * cell + 1, oy + r * cell + 1, ox + (c + 1) * cell - 1, oy + (r + 1) * cell - 1]
                    d.rectangle(box, fill=tissue_colour(kind[r, c], 0.4), outline=BONE_EDGE)
        d.text((16, 14), f'growing  step {k + 1} / {len(states)}', fill=INK, font=FONT)
        d.text((16, 30), 'pale = bone   red = muscle   blue = eye', fill=MUTED, font=FONT)
        frames.append(img)
    return frames


def life_frames(genome, world, seconds, every=2):
    body, brain = cr.develop(jnp.asarray(genome))
    setup, frames = jax.jit(sim.record, static_argnums=(3,))(body, brain, jax.random.PRNGKey(world), seconds)
    x, act, spikes, fruit, energy, eaten = (np.asarray(a) for a in frames)
    alive = np.asarray(body.alive)
    kind = np.asarray(body.tissue)
    on = np.asarray(setup.point_on)
    scale = 200.0                      # pixels per metre
    ground_y = SIZE[1] - 60
    cam = float(x[0, on, 0].mean())
    images = []
    for t in range(0, len(x), every):
        cam += (float(x[t, on, 0].mean()) - cam) * 0.2
        px = lambda X, Y: (SIZE[0] / 2 + (X - cam) * scale, ground_y - Y * scale)
        img = Image.new('RGB', SIZE, SKY)
        d = ImageDraw.Draw(img)
        # ground with a mark every half metre
        left = cam - SIZE[0] / 2 / scale
        for k in range(int(np.floor(left * 2)), int(np.ceil((left + SIZE[0] / scale) * 2)) + 1):
            gx = px(k / 2, 0)[0]
            d.line([(gx, ground_y), (gx, ground_y + (8 if k % 2 == 0 else 4))], fill=GROUND)
            if k % 2 == 0:
                d.text((gx - 8, ground_y + 12), f'{k // 2} m', fill=MUTED, font=FONT)
        d.line([(0, ground_y), (SIZE[0], ground_y)], fill=GROUND, width=2)
        # fruit, or a tick at the edge of the view for fruit out of sight
        for f in fruit[t]:
            fx, fy = px(f, 0.04)
            if 0 <= fx <= SIZE[0]:
                d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=FRUIT)
            else:
                ex = 6 if fx < 0 else SIZE[0] - 6
                d.line([(ex, ground_y - 16), (ex, ground_y - 2)], fill=FRUIT, width=4)
        # the body: one quad per living voxel
        for i in np.flatnonzero(alive):
            tl, tr, bl, br = x[t, sim.CORNERS[i]]
            quad = [px(*tl), px(*tr), px(*br), px(*bl)]
            d.polygon(quad, fill=tissue_colour(kind[i], act[t, i]), outline=BONE_EDGE)
            cx = sum(p[0] for p in quad) / 4
            cy = sum(p[1] for p in quad) / 4
            if kind[i] == 2:
                facing = 1 if (tl[0] + tr[0]) / 2 >= x[t, on, 0].mean() else -1
                d.ellipse([cx + 3 * facing - 3, cy - 3, cx + 3 * facing + 3, cy + 3], fill=PUPIL)
            if spikes[t, i] > 0:
                d.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=SPIKE)
        # energy, fruit eaten, time
        e = max(float(energy[t]), 0.0)
        d.rectangle([16, 16, 176, 28], outline=INK)
        d.rectangle([17, 17, 17 + int(158 * e), 27], fill=FRUIT if e > 0.3 else (210, 120, 40))
        d.text((186, 15), 'energy', fill=MUTED, font=FONT)
        d.text((16, 36), f'{int(eaten[t])} fruit eaten    {t * sim.DT:4.1f} s', fill=INK, font=FONT)
        if e <= 0:
            d.text((16, 54), 'starved', fill=(190, 34, 26), font=FONT)
        images.append(img)
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('genome', help='a .npy file saved by evolve.py')
    ap.add_argument('--out', default='creature.gif')
    ap.add_argument('--world', type=int, default=1000, help='which world to live in (fruit placement)')
    ap.add_argument('--seconds', type=float, default=sim.EPISODE_SECONDS)
    ap.add_argument('--no-growth', action='store_true', help='skip the growing part')
    args = ap.parse_args()
    genome = np.load(args.genome)
    body, _ = cr.develop(jnp.asarray(genome))
    print('Body (B bone, M muscle, E eye):\n' + cr.describe(body))
    frames, durations = [], []
    if not args.no_growth:
        grown = growth_frames(genome)
        frames += grown
        durations += [70] * (len(grown) - 1) + [900]
    lived = life_frames(genome, args.world, args.seconds)
    frames += lived
    durations += [int(2 * sim.DT * 1000)] * len(lived)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f'Wrote {args.out} ({len(frames)} frames)')


if __name__ == '__main__':
    main()
