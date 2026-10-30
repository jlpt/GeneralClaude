"""Draw a creature growing from its seed cell and then living, as an animated GIF.

    python render.py runs/first/best.npy                 # writes creature.gif
    python render.py runs/first/best.npy --out walk.gif --seconds 30 --nudge 1

The physics works on square cells, but each frame paints the cells as one soft, rounded body: the
outline is blurred and cut back out, and tissue colours blend into each other. Bone is pale and muscle
turns from pink to deep red as it contracts. On top is the nervous system: each nerve cell is a small
dot inside its voxel, joined by thin fibres to the nerve cells it's wired to, and it flashes yellow on
the tick it fires. A green line on the ground marks where the creature started. Falling over (turning
more than 45° from the starting pose) stops the distance counting.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

import creature as cr
import sim

SIZE = (640, 360)
SUPERSAMPLE = 2
SKY = (245, 243, 237)
GRID = (228, 224, 216)
GROUND = (110, 96, 80)
INK = (38, 34, 46)
MUTED = (120, 114, 128)
OUTLINE = (92, 80, 70)
BONE = (232, 222, 200)
MUSCLE_SOFT, MUSCLE_HARD = (224, 158, 146), (186, 36, 30)
START = (58, 166, 72)
FIBRE = (196, 150, 52)
NERVE_CELL = (140, 96, 24)
SPIKE = (255, 214, 0)
FONT = ImageFont.load_default(size=14)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def tissue_colour(kind, act=0.0):
    return [BONE, mix(MUSCLE_SOFT, MUSCLE_HARD, act * 1.6)][kind]


def paint_body(img, quads, colours, cell_px):
    """Draw cells as one soft body: blur the silhouette and threshold it, and blur the tissue colours."""
    if not quads:
        return
    pad = int(cell_px * 1.5)
    xs = [p[0] for q in quads for p in q]
    ys = [p[1] for q in quads for p in q]
    x0, y0 = max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0)
    x1, y1 = min(int(max(xs)) + pad, SIZE[0]), min(int(max(ys)) + pad, SIZE[1])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    w, h = (x1 - x0) * SUPERSAMPLE, (y1 - y0) * SUPERSAMPLE
    mask = Image.new('L', (w, h), 0)
    colour = Image.new('RGB', (w, h), BONE)
    dm, dc = ImageDraw.Draw(mask), ImageDraw.Draw(colour)
    for q, c in zip(quads, colours):
        pts = [((x - x0) * SUPERSAMPLE, (y - y0) * SUPERSAMPLE) for x, y in q]
        dm.polygon(pts, fill=255)
        dc.polygon(pts, fill=c)
    blur = cell_px * SUPERSAMPLE * 0.3
    mask = mask.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: 255 if v >= 120 else 0)
    colour = colour.filter(ImageFilter.GaussianBlur(blur * 0.7))
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask)
    body = Image.composite(colour, Image.new('RGB', (w, h), OUTLINE), mask)
    body.paste(OUTLINE, mask=edge)
    alpha = ImageChops.lighter(mask, edge)
    size = (x1 - x0, y1 - y0)
    img.paste(body.resize(size, Image.LANCZOS), (x0, y0), alpha.resize(size, Image.LANCZOS))


def paint_nerves(d, centres, nerve, weights, spikes, cell_px):
    """The nervous system: fibres between wired nerve cells, a small dot for each nerve cell, and a bright
    flash for each one firing."""
    wired = (weights != 0) | (weights.T != 0)
    for i, j in zip(cr.PAIR_I, cr.PAIR_J):
        if i < j and wired[i, j]:
            d.line([tuple(centres[i]), tuple(centres[j])], fill=FIBRE, width=2)
    for i in np.flatnonzero(nerve):
        cx, cy = centres[i]
        if spikes is not None and spikes[i] > 0:
            r = cell_px * 0.28
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SPIKE, outline=NERVE_CELL)
        else:
            r = cell_px * 0.14
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NERVE_CELL)


def growth_frames(genome):
    params = cr.unflatten(genome)
    history = cr.grow_history(params)
    bodies = jax.vmap(cr.read_body)(history)
    wiring = np.asarray(jax.vmap(lambda s, b: cr.read_brain(s, b, params).weights)(history, bodies))
    nerves = np.asarray(bodies.nerve)
    states = np.asarray(history)
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
        kind = np.argmax(s[..., [cr.BONE, cr.MUSCLE]], axis=-1)
        cells = [(r, c) for r in range(cr.H) for c in range(cr.W) if alive[r, c]]
        quads = [[(ox + c * cell, oy + r * cell), (ox + (c + 1) * cell, oy + r * cell),
                  (ox + (c + 1) * cell, oy + (r + 1) * cell), (ox + c * cell, oy + (r + 1) * cell)] for r, c in cells]
        paint_body(img, quads, [tissue_colour(kind[r, c], 0.4) for r, c in cells], cell)
        centres = np.array([(ox + (c + 0.5) * cell, oy + (r + 0.5) * cell) for r in range(cr.H) for c in range(cr.W)])
        paint_nerves(d, centres, nerves[k], wiring[k], None, cell)
        d.text((16, 14), f'growing  step {k + 1} / {len(states)}', fill=INK, font=FONT)
        d.text((16, 32), 'pale = bone   red = muscle   yellow = nerves', fill=MUTED, font=FONT)
        frames.append(img)
    return frames


def life_frames(genome, nudge, seconds, every=2):
    body, brain = cr.develop(jnp.asarray(genome))
    setup, frames = jax.jit(sim.record, static_argnums=(3,))(body, brain, jax.random.PRNGKey(nudge), seconds)
    x, act, spikes, walked, fallen = (np.asarray(a) for a in frames)
    start_x = float(setup.start_x)
    alive = np.flatnonzero(np.asarray(body.alive))
    kind = np.asarray(body.tissue)
    nerve = np.asarray(body.nerve)
    weights = np.asarray(brain.weights)
    on = np.asarray(setup.point_on)
    scale = 200.0                      # pixels per metre
    cell_px = sim.VOXEL * scale
    ground_y = SIZE[1] - 60
    cam = float(x[0, on, 0].mean())
    images = []
    for t in range(0, len(x), every):
        middle = float(x[t, on, 0].mean())
        cam += (middle - cam) * 0.2
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
        sx = px(start_x, 0)[0]
        d.line([(sx, ground_y - 24), (sx, ground_y)], fill=START, width=3)
        # the body, then the nervous system on top
        quads = []
        for i in alive:
            tl, tr, bl, br = x[t, sim.CORNERS[i]]
            quads.append([px(*tl), px(*tr), px(*br), px(*bl)])
        paint_body(img, quads, [tissue_colour(kind[i], act[t, i]) for i in alive], cell_px)
        centres = np.zeros((cr.N, 2))
        for q, i in zip(quads, alive):
            centres[i] = sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4
        fired = np.maximum(spikes[t], spikes[t - 1]) if t > 0 else spikes[t]    # frames skip every other tick
        paint_nerves(d, centres, nerve, weights, fired, cell_px)
        d.text((16, 14), f'walked {walked[t]:+.2f} m    {(t + 1) * sim.DT:4.1f} s', fill=INK, font=FONT)
        d.text((16, 32), 'yellow = nerves, flashing when they fire', fill=MUTED, font=FONT)
        if fallen[t]:
            d.text((16, 50), 'fell over: the distance stops counting', fill=(190, 34, 26), font=FONT)
        images.append(img)
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('genome', help='a .npy file saved by evolve.py')
    ap.add_argument('--out', default='creature.gif')
    ap.add_argument('--nudge', type=int, default=1000, help='which random starting nudge to use')
    ap.add_argument('--seconds', type=float, default=sim.EPISODE_SECONDS)
    ap.add_argument('--no-growth', action='store_true', help='skip the growing part')
    args = ap.parse_args()
    genome = np.load(args.genome)
    body, _ = cr.develop(jnp.asarray(genome))
    print('Body (B bone, M muscle; lower case where a nerve runs):\n' + cr.describe(body))
    frames, durations = [], []
    if not args.no_growth:
        grown = growth_frames(genome)
        frames += grown
        durations += [70] * (len(grown) - 1) + [900]
    lived = life_frames(genome, args.nudge, args.seconds)
    frames += lived
    durations += [int(2 * sim.DT * 1000)] * len(lived)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f'Wrote {args.out} ({len(frames)} frames)')


if __name__ == '__main__':
    main()
