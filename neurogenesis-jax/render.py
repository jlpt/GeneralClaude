"""Draw a creature growing from its seed cell and then living, as an animated GIF.

    python render.py runs/first/best.npy                 # writes creature.gif
    python render.py runs/first/best.npy --out life.gif --seconds 30 --world 7

Bones are pale, rounded limbs, and solid patches of bone are filled in. A muscle joint is a dark dot,
with its muscle drawn as a red band on the side it pulls: the band turns deeper red, thickens and
shortens as the muscle contracts, and stays dull if no nerve reaches the joint. An eye is a white
eyeball looking away from the body's middle. On top is the nervous system: violet fibres between wired
nerve cells, a violet dot for each nerve cell, and a green flash when one fires. Fruit are orange dots
on the ground (an arrow at the edge points to fruit off screen), and the bar at the top right is the
creature's energy.
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
MUSCLE_IDLE = (220, 179, 171)
NERVE = (90, 61, 176)
FIRE = (18, 181, 83)
EYE_WHITE, PUPIL = (255, 255, 255), (30, 28, 36)
FRUIT, LEAF = (236, 120, 30), (60, 150, 60)
ENERGY_FULL, ENERGY_LOW = (40, 160, 80), (200, 50, 40)
FONT = ImageFont.load_default(size=14)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def nerve_positions(x, angle):
    """Where each fine cell sits: its joint's position plus its place in the 3 × 3 patch, turned with the body."""
    f = np.arange(cr.NF)
    fr, fc = f // cr.WF, f % cr.WF
    joint = (fr // cr.FINE) * cr.W + fc // cr.FINE
    ox, oy = (fc % cr.FINE - 1) * sim.CELL / cr.FINE, -(fr % cr.FINE - 1) * sim.CELL / cr.FINE
    c, s = np.cos(angle), np.sin(angle)
    return x[joint] + np.stack([c * ox - s * oy, s * ox + c * oy], axis=1)


def draw_creature(img, x, to_px, body, bones, act, brain, spikes=None, angle=0.0):
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
    wired = np.asarray(cr.patches(body.nerve)).reshape(cr.N, -1).any(axis=1)      # a nerve reaches this joint
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
        colour = mix(MUSCLE_SOFT, MUSCLE_HARD, k * 1.5) if wired[p] else MUSCLE_IDLE
        q1, q2 = to_px(*m1), to_px(*m2)
        d.line([q1, q2], fill=colour, width=width)
        for q in (q1, q2):
            d.ellipse([q[0] - width / 2, q[1] - width / 2, q[0] + width / 2, q[1] + width / 2], fill=colour)
    # 3. muscle joints and eyes (looking away from the body's middle, as they see), then the nervous system
    for i in np.flatnonzero(alive & (tissue == 1)):
        cx, cy = pts[i]
        r = radius * 0.45
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=JOINT)
    middle = x[alive, 0].mean()
    for i in np.flatnonzero(alive & (tissue == 2)):
        cx, cy = pts[i]
        r = radius * 0.95
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=EYE_WHITE, outline=OUTLINE)
        px = cx + (1 if x[i, 0] >= middle else -1) * r * 0.45
        q = r * 0.45
        d.ellipse([px - q, cy - q, px + q, cy + q], fill=PUPIL)
    nerve = np.asarray(body.nerve).reshape(-1)
    weights = np.asarray(brain.weights).reshape(cr.NF, -1)
    spots = np.array([to_px(*p) for p in nerve_positions(x, angle)])
    for i in np.flatnonzero(nerve):
        r, c = i // cr.WF, i % cr.WF
        for o, (dy, dx) in enumerate(cr.OFFSETS):
            rr, cc = r + dy, c + dx
            j = rr * cr.WF + cc
            if 0 <= rr < cr.HF and 0 <= cc < cr.WF and j > i and nerve[j] and (weights[i, o] != 0 or weights[j, 7 - o] != 0):
                d.line([tuple(spots[i]), tuple(spots[j])], fill=NERVE, width=1)
    fine_px = cell_px / cr.FINE
    fired = np.asarray(spikes).reshape(-1) if spikes is not None else np.zeros(cr.NF)
    for i in np.flatnonzero(nerve):
        cx, cy = spots[i]
        if fired[i] > 0:
            r = fine_px * 0.55
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FIRE)
        r = fine_px * 0.22
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NERVE)


def growth_frames(genome):
    params = cr.unflatten(genome)
    states = cr.grow_history(params)
    bodies = jax.vmap(cr.read_body)(states)
    skeletons = jax.vmap(sim.skeleton)(bodies)
    brains = jax.vmap(lambda s, b: cr.read_brain(s, b, params))(states, bodies)
    cell = sim.CELL * SCALE
    ox, oy = (SIZE[0] - (cr.W - 1) * cell) / 2, (SIZE[1] - (cr.H - 1) * cell) / 2 + 10
    to_px = lambda X, Y: (ox + X * SCALE, oy + ((cr.H - 1) * sim.CELL - Y) * SCALE)
    frames = []
    pick = lambda tree, k: jax.tree_util.tree_map(lambda a: a[k], tree)
    for k in range(len(states)):
        img = Image.new('RGB', SIZE, SKY)
        d = ImageDraw.Draw(img)
        for r in range(cr.H):
            d.line([to_px(0, r * sim.CELL), to_px((cr.W - 1) * sim.CELL, r * sim.CELL)], fill=GRID)
        for c in range(cr.W):
            d.line([to_px(c * sim.CELL, 0), to_px(c * sim.CELL, (cr.H - 1) * sim.CELL)], fill=GRID)
        draw_creature(img, sim.HOME, to_px, pick(bodies, k), pick(skeletons, k), np.full(cr.N, 0.3), pick(brains, k))
        d.text((16, 14), f'growing  step {k + 1} / {len(states)}', fill=INK, font=FONT)
        d.text((16, 32), 'pale = bone   red = muscle joint   white = eye   violet = nerves', fill=MUTED, font=FONT)
        frames.append(img)
    return frames


def draw_fruit(d, px, fruit, ground_y):
    r = 0.035 * SCALE
    for f in fruit:
        cx, cy = px(f, 0)
        cy -= r
        if cx < -r or cx > SIZE[0] + r:          # off screen: an arrow at the edge, with the distance
            side = 1 if cx > SIZE[0] else -1
            ex = SIZE[0] - 14 if side > 0 else 14
            d.polygon([(ex + side * 8, ground_y - 20), (ex - side * 4, ground_y - 27), (ex - side * 4, ground_y - 13)], fill=FRUIT)
            d.text((ex - (40 if side > 0 else -6), ground_y - 46), f'{abs(f - cam_of(px)):.1f} m', fill=MUTED, font=FONT)
            continue
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FRUIT)
        d.ellipse([cx, cy - r - 5, cx + 7, cy - r + 1], fill=LEAF)


def cam_of(px):
    """The world x at the middle of the picture."""
    return (SIZE[0] / 2 - px(0, 0)[0]) / SCALE


def draw_energy(d, energy):
    x0, y0, w, h = SIZE[0] - 150, 16, 130, 12
    d.text((x0 - 56, y0 - 2), 'energy', fill=MUTED, font=FONT)
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=OUTLINE)
    e = float(np.clip(energy, 0, 1))
    if e > 0:
        d.rectangle([x0 + 1, y0 + 1, x0 + 1 + (w - 2) * e, y0 + h - 1], fill=mix(ENERGY_LOW, ENERGY_FULL, e * 1.6))


def life_frames(genome, world, seconds, every=2):
    body, brain = cr.develop(jnp.asarray(genome))
    setup, frames = jax.jit(sim.record, static_argnums=(3,))(body, brain, world, seconds)
    x, act, spikes, fruit, energy, eaten = (np.asarray(a) for a in frames)
    angles = np.asarray(jax.vmap(lambda xt: sim.tilt(setup, xt))(frames[0]))
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
        draw_fruit(d, px, fruit[t], ground_y)
        fired = np.maximum(spikes[t], spikes[t - 1]) if t > 0 else spikes[t]    # frames skip every other tick
        draw_creature(img, x[t], px, body, setup.bones, act[t], brain, fired, angles[t])
        d.text((16, 14), f'fruit eaten {int(eaten[t])}    {(t + 1) * sim.DT:4.1f} s', fill=INK, font=FONT)
        d.text((16, 32), 'violet = nerves, flashing green when they fire', fill=MUTED, font=FONT)
        draw_energy(d, energy[t])
        if energy[t] <= 0:
            d.text((16, 50), 'ran out of energy', fill=ENERGY_LOW, font=FONT)
        images.append(img)
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('genome', help='a .npy file saved by evolve.py')
    ap.add_argument('--out', default='creature.gif')
    ap.add_argument('--world', type=int, default=1000, help='which world: where the fruit lie and which way the start nudge goes')
    ap.add_argument('--seconds', type=float, default=sim.EPISODE_SECONDS)
    ap.add_argument('--no-growth', action='store_true', help='skip the growing part')
    args = ap.parse_args()
    genome = np.load(args.genome)
    body, _ = cr.develop(jnp.asarray(genome))
    print('Body (B bone, M muscle joint, E eye; lower case where a nerve runs):\n' + cr.describe(body))
    print(f'{int(body.alive.sum())} joints, {int(body.nerve.sum())} nerve cells')
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
