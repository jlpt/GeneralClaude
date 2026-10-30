"""Draw a creature growing from its seed cell and then living, as an animated GIF.

    python render.py runs/first/best.npy                 # writes creature.gif
    python render.py runs/first/best.npy --out walk.gif --seconds 30 --world 7

Bones are pale, rounded limbs, and solid patches of bone are filled in. A muscle joint is a dark dot,
with its muscle drawn as a red band on the side it pulls: the band turns deeper red, thickens and
shortens as the muscle contracts. Eyes are eyeballs looking the way they face, a yellow dot means that
cell's neuron fired on this tick, and fruit is green. Fruit off the edge of the view is shown as a
green tick at the edge.
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
SCALE = 200.0                       # pixels per metre
SKY = (245, 243, 237)
GRID = (228, 224, 216)
GROUND = (110, 96, 80)
INK = (38, 34, 46)
MUTED = (120, 114, 128)
OUTLINE = (92, 80, 70)
BONE = (232, 222, 200)
JOINT = (120, 70, 62)
MUSCLE_SOFT, MUSCLE_HARD = (226, 150, 138), (182, 30, 26)
EYEBALL, PUPIL = (255, 255, 255), (24, 30, 46)
FRUIT = (58, 166, 72)
SPIKE = (255, 206, 0)
FONT = ImageFont.load_default(size=14)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def draw_creature(img, x, to_px, body, bones, act, spikes=None):
    """Draw one creature. x is (N, 2) joint positions in metres; to_px turns a position into pixels."""
    alive = np.asarray(body.alive)
    tissue = np.asarray(body.tissue)
    link_on = np.asarray(bones.link_on)
    joint_on = np.asarray(bones.joint_on)
    if not alive.any():
        return
    pts = np.array([to_px(*p) for p in x])
    cell_px = sim.CELL * SCALE
    radius = sim.RADIUS * SCALE
    # 1. the bones and solid patches, as one silhouette with an outline, drawn large and shrunk to smooth it
    live = pts[alive]
    pad = int(cell_px * 2)
    x0, y0 = max(int(live[:, 0].min()) - pad, 0), max(int(live[:, 1].min()) - pad, 0)
    x1, y1 = min(int(live[:, 0].max()) + pad, SIZE[0]), min(int(live[:, 1].max()) + pad, SIZE[1])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    S = SUPERSAMPLE
    local = lambda p: ((p[0] - x0) * S, (p[1] - y0) * S)
    mask = Image.new('L', ((x1 - x0) * S, (y1 - y0) * S), 0)
    dm = ImageDraw.Draw(mask)
    linked = {(int(a), int(b)) for a, b, on in zip(sim.LINK_A, sim.LINK_B, link_on) if on}
    for r in range(cr.H - 1):                   # fill each square whose four sides are bones
        for c in range(cr.W - 1):
            tl, tr, bl, br = r * cr.W + c, r * cr.W + c + 1, (r + 1) * cr.W + c, (r + 1) * cr.W + c + 1
            if {(tl, tr), (bl, br), (tl, bl), (tr, br)} <= linked:
                dm.polygon([local(pts[i]) for i in (tl, tr, br, bl)], fill=255)
    for a, b in linked:
        dm.line([local(pts[a]), local(pts[b])], fill=255, width=int(2 * radius * S))
    for i in np.flatnonzero(alive):
        cx, cy = local(pts[i])
        dm.ellipse([cx - radius * S, cy - radius * S, cx + radius * S, cy + radius * S], fill=255)
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask)
    body_img = Image.composite(Image.new('RGB', mask.size, BONE), Image.new('RGB', mask.size, OUTLINE), mask)
    body_img.paste(OUTLINE, mask=edge)
    alpha = ImageChops.lighter(mask, edge)
    size = (x1 - x0, y1 - y0)
    img.paste(body_img.resize(size, Image.LANCZOS), (x0, y0), alpha.resize(size, Image.LANCZOS))
    # 2. muscles: a band across each muscle joint, on the side that shortens as it pulls
    d = ImageDraw.Draw(img)
    for j in np.flatnonzero(joint_on & (np.asarray(bones.joint_bend) != 0)):
        a, p, c = int(np.asarray(bones.joint_a)[j]), int(sim.JOINT_P[j]), int(sim.JOINT_C[j])
        u, v = x[a] - x[p], x[c] - x[p]
        side = np.sign(float(np.asarray(bones.joint_bend)[j]))
        cw = lambda w: np.array([w[1], -w[0]])
        ccw = lambda w: np.array([-w[1], w[0]])
        k = float(act[p])
        off = sim.RADIUS * 0.75
        m1 = x[p] + 0.55 * u + off * side * cw(unit(u))
        m2 = x[p] + 0.55 * v + off * side * ccw(unit(v))
        width = max(int(radius * (0.7 + 0.8 * k)), 2)
        colour = mix(MUSCLE_SOFT, MUSCLE_HARD, k * 1.5)
        q1, q2 = to_px(*m1), to_px(*m2)
        d.line([q1, q2], fill=colour, width=width)
        for q in (q1, q2):
            d.ellipse([q[0] - width / 2, q[1] - width / 2, q[0] + width / 2, q[1] + width / 2], fill=colour)
    # 3. joints, eyes and firing neurons
    middle = x[alive, 0].mean()
    for i in np.flatnonzero(alive):
        cx, cy = pts[i]
        if tissue[i] == 1:
            r = radius * 0.45
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=JOINT)
        elif tissue[i] == 2:
            r, facing = cell_px * 0.36, 1 if x[i, 0] >= middle else -1
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=EYEBALL, outline=OUTLINE)
            pr, off = cell_px * 0.17, cell_px * 0.14 * facing
            d.ellipse([cx + off - pr, cy - pr, cx + off + pr, cy + pr], fill=PUPIL)
        if spikes is not None and spikes[i] > 0 and tissue[i] != 2:
            d.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=SPIKE)


def growth_frames(genome):
    states = cr.grow_history(cr.unflatten(genome))
    bodies = jax.vmap(cr.read_body)(states)
    skeletons = jax.vmap(sim.skeleton)(bodies)
    cell = sim.CELL * SCALE
    ox, oy = (SIZE[0] - (cr.W - 1) * cell) / 2, (SIZE[1] - (cr.H - 1) * cell) / 2 + 10
    to_px = lambda X, Y: (ox + X * SCALE, oy + ((cr.H - 1) * sim.CELL - Y) * SCALE)
    frames = []
    for k in range(len(states)):
        img = Image.new('RGB', SIZE, SKY)
        d = ImageDraw.Draw(img)
        for r in range(cr.H):
            d.line([to_px(0, r * sim.CELL), to_px((cr.W - 1) * sim.CELL, r * sim.CELL)], fill=GRID)
        for c in range(cr.W):
            d.line([to_px(c * sim.CELL, 0), to_px(c * sim.CELL, (cr.H - 1) * sim.CELL)], fill=GRID)
        body = jax.tree_util.tree_map(lambda a: a[k], bodies)
        bones = jax.tree_util.tree_map(lambda a: a[k], skeletons)
        draw_creature(img, sim.HOME, to_px, body, bones, np.full(cr.N, 0.3))
        d.text((16, 14), f'growing  step {k + 1} / {len(states)}', fill=INK, font=FONT)
        d.text((16, 32), 'pale = bone   red = muscle at a joint   eyes see fruit', fill=MUTED, font=FONT)
        frames.append(img)
    return frames


def life_frames(genome, world, seconds, every=2):
    body, brain = cr.develop(jnp.asarray(genome))
    setup, frames = jax.jit(sim.record, static_argnums=(3,))(body, brain, jax.random.PRNGKey(world), seconds)
    x, act, spikes, fruit, energy, eaten = (np.asarray(a) for a in frames)
    on = np.asarray(setup.point_on)
    ground_y = SIZE[1] - 60
    cam = float(x[0, on, 0].mean())
    images = []
    for t in range(0, len(x), every):
        middle = float(x[t, on, 0].mean())
        cam += (middle - cam) * 0.2
        px = lambda X, Y: (SIZE[0] / 2 + (X - cam) * SCALE, ground_y - Y * SCALE)
        img = Image.new('RGB', SIZE, SKY)
        d = ImageDraw.Draw(img)
        # ground with a mark every half metre
        left = cam - SIZE[0] / 2 / SCALE
        for k in range(int(np.floor(left * 2)), int(np.ceil((left + SIZE[0] / SCALE) * 2)) + 1):
            gx = px(k / 2, 0)[0]
            d.line([(gx, ground_y), (gx, ground_y + (8 if k % 2 == 0 else 4))], fill=GROUND)
            if k % 2 == 0:
                d.text((gx - 8, ground_y + 12), f'{k // 2} m', fill=MUTED, font=FONT)
        d.line([(0, ground_y), (SIZE[0], ground_y)], fill=GROUND, width=2)
        # fruit, or a tick at the edge of the view for fruit out of sight
        for f in fruit[t]:
            fx, fy = px(f, 0.035)
            if 0 <= fx <= SIZE[0]:
                d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=FRUIT)
            else:
                ex = 6 if fx < 0 else SIZE[0] - 6
                d.line([(ex, ground_y - 16), (ex, ground_y - 2)], fill=FRUIT, width=4)
        draw_creature(img, x[t], px, body, setup.bones, act[t], spikes[t])
        # energy, fruit eaten, time
        e = max(float(energy[t]), 0.0)
        d.rectangle([16, 16, 176, 28], outline=INK)
        d.rectangle([17, 17, 17 + int(158 * e), 27], fill=FRUIT if e > 0.3 else (210, 120, 40))
        d.text((186, 14), 'energy', fill=MUTED, font=FONT)
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
    print('Body (B bone, M muscle joint, E eye):\n' + cr.describe(body))
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
