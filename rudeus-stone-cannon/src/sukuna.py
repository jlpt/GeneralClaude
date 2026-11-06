"""Sukuna: wide-shot sprite, face close-up, pixel font, the Malevolent Shrine and its slashes."""
import math
import numpy as np
from canvas import (Scene, C, L, XS, YS, W, H, M, CW, CH, v2, rot, unit, disc, ellipse, polygon, capsule, nb,
                    outline_of, dilate4, line_pts, smooth01, lerp)

# ---------------------------------------------------------------- wide-shot head (drawn facing right, mirrored)
SK_KEY = {
    "o": "ol", "1": "ph0", "2": "ph1", "3": "ph2", "4": "ph3", "5": "ph4",
    "k": "sk0", "s": "sk1", "S": "sk2", "L": "sk3",
    "e": "ol", "r": "re1", "R": "re2", "t": "ol", "w": "white", "m": "re0",
}
SK_HEAD = [
    "o.....o....o......",
    "o4o..o4o..o4o.....",
    ".o4o.o45o.o45o....",
    ".o34oo455o4554o...",
    "..o345545555443o..",
    ".o2345555554443o..",
    "o22344444444433o..",
    "o223333333333333o.",
    "o2233SSSSSSSSSS3o.",
    "o223SSSSSSSSSSSSo.",
    "o22kSoooSSSSoooSo.",
    "o22kLSeRSSSSSeRSo.",
    "o12kSSSSSSSSSSSSo.",
    ".o2kStrtSSSsStrto.",
    ".o21kSttSSSSSttSo.",
    "..o2kSSwwwwwwSSo..",
    "...okkSmwmwmSSko..",
    "....o.kkSSSSSko...",
    ".......kkkkkk.....",
]
assert all(len(r) == 18 for r in SK_HEAD), [len(r) for r in SK_HEAD]
SK_ANCHOR = (10, 19)


def _head(sc, neck, layer, eyes_only=False):
    ox = int(round(neck[0])) - SK_ANCHOR[0]
    oy = int(round(neck[1])) - SK_ANCHOR[1]
    for y, row in enumerate(SK_HEAD):
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            if eyes_only and ch not in "rR":
                continue
            sc.put_px([ox + x], [oy + y], SK_KEY[ch], layer)


def _kimono(sc, hip, pose, layer):
    """Facing right. pose: 'stand' (hands in sleeves) or 'sign' (hands joined in front of the chest)."""
    hx, hy = hip

    def p(u, v):
        return v2((hx + u, hy - v))

    body = polygon([p(-2.5, 21.5), p(-6.5, 19.5), p(-7.0, 12), p(-6.0, 2), p(-6.8, -10), p(-7.6, -21),
                    p(7.0, -21), p(6.2, -10), p(5.2, 1), p(6.3, 11), p(6.0, 19.5), p(2.6, 21.5)])
    dx, dy = XS - hx, hy - YS          # u, v of every pixel
    sc.put(outline_of(body), "ki0", layer)
    sc.put(body, "ki3", layer)
    sc.put(body & ~nb(body, -1, 0), "ki4", layer)
    sc.put(body & (dx > 3.2), "ki2", layer)
    sc.put(body & ~nb(body, 1, 0), "ki1", layer)
    for fu in (-3.5, 0.5, 3.8):        # long folds down the skirt
        sc.put(body & (dy < -3) & (np.abs(dx - fu - (dy + 3) * 0.06 * fu) < 0.5), "ki2", layer)
    sc.put(body & (dy < -19.5), "ki1", layer)
    # crossed collar (left over right) with the dark inner layer
    col = polygon([p(-1.6, 21.8), p(0.6, 21.8), p(3.6, 3.0), p(2.0, 3.0)])
    inner = polygon([p(0.6, 21.8), p(2.8, 21.8), p(1.4, 14.0)])
    sc.put(inner & body, "ob1", layer)
    sc.put(col & body, "ki1", layer)
    sc.put(outline_of(col) & body & (dx < 1.5), "ki0", layer)
    # obi
    obi = body & (dy > -0.5) & (dy < 3.5)
    sc.put(obi, "ob0", layer)
    sc.put(obi & (np.abs(dy - 1.5) < 0.5), "ob2", layer)
    knot = ellipse(p(-6.5, 1.5), 1.8, 2.4)
    sc.put(outline_of(knot), "ki0", layer)
    sc.put(knot, "ob1", layer)
    # sandals
    for fx_, tone in ((-4.5, "ob1"), (3.0, "ob0")):
        ft = ellipse(p(fx_, -21.6), 3.0, 1.1)
        sc.put(ft, tone, layer)
    # neck
    nk = capsule(p(0.5, 20.5), p(0.8, 24.0), 1.5)
    sc.put(outline_of(nk) & ~body, "sk0", layer)
    sc.put(nk & ~body, "sk2", layer)
    # sleeves
    if pose == "stand":
        sl = polygon([p(-5.8, 19.0), p(-1.2, 17.6), p(0.8, 6.0), p(1.6, -3.0), p(-8.6, -2.6), p(-8.2, 8.0)])
        sc.put(outline_of(sl), "ki0", layer)
        sc.put(sl, "ki3", layer)
        sc.put(sl & ~nb(sl, -1, 0), "ki4", layer)
        sc.put(sl & ~nb(sl, 0, 1), "ki1", layer)
        sc.put(sl & (np.abs(dx + 3.6) < 0.5) & (dy < 12), "ki2", layer)
        hand = disc(p(1.8, 1.0), 1.4)     # one hand just showing at the cuff
        sc.put(outline_of(hand) & ~sl, "sk0", layer)
        sc.put(hand & ~sl, "sk2", layer)
    else:
        hands_c = p(9.5, 13.0)
        for shoulder, elbow, tone in ((p(4.8, 18.0), p(5.5, 9.0), "ki2"), (p(-4.8, 18.5), p(-1.5, 9.0), "ki3")):
            arm = capsule(shoulder, elbow, 2.4) | capsule(elbow, hands_c + v2((-2.0, 1.0)), 2.2)
            drape = polygon([elbow + v2((-2.5, -1)), hands_c + v2((-2.5, 1.5)), hands_c + v2((-3.5, 8.5)),
                             elbow + v2((-3.0, 6.0))])
            sl = arm | drape
            sc.put(outline_of(sl), "ki0", layer)
            sc.put(sl, tone, layer)
            sc.put(sl & ~nb(sl, 0, 1), "ki1", layer)
            sc.put(sl & ~nb(sl, -1, -1), "ki4" if tone == "ki3" else "ki3", layer)
        # joined hands, fingers up (the seal)
        hm = disc(hands_c, 2.0) | capsule(hands_c, hands_c + v2((0.5, -4.2)), 0.7) | \
            capsule(hands_c + v2((1.2, 0)), hands_c + v2((2.0, -3.4)), 0.6)
        sc.put(outline_of(hm), "sk0", layer)
        sc.put(hm, "sk2", layer)
        sc.put(hm & ~nb(hm, -1, -1), "sk3", layer)
        sc.put(hm & ~nb(hm, 1, 1), "sk1", layer)
    return p(0.8, 23.5)


def draw_sukuna(sc, x, feet_y, pose="stand", sil=0.0, layer="Sukuna", eye_glow=0.0):
    """Facing left. sil 0..1 blends him into a dark silhouette (only the red eyes stay lit)."""
    tmp = Scene()
    hip = v2((W - 1 - x, feet_y - 22))
    neck = _kimono(tmp, hip, pose, layer)
    _head(tmp, neck, layer)
    m = tmp.owner >= 0
    rgb = tmp.rgb[:, ::-1]
    m = m[:, ::-1]
    if sil > 0:
        dark = np.array(C["ob0"]) * 0.8 + np.array(C["dm1"]) * 0.2
        rgb = rgb * (1 - sil) + dark * sil
    sc.put_rgb(m, rgb, layer)
    if sil > 0.05 or eye_glow > 0:
        eyes = Scene()
        _head(eyes, neck, layer, eyes_only=True)
        em = (eyes.owner >= 0)[:, ::-1]
        sc.put(em, "re2", layer)
        g = 0.0
        for yy, xx in zip(*np.nonzero(em)):
            g = g + np.exp(-((XS - (xx - M)) ** 2 + (YS - (yy - M)) ** 2) / 6.0)
        sc.light(g, C["re2"], 0.5 * max(sil, eye_glow), dither=30)
    return m


# ---------------------------------------------------------------- pixel font
FONT = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    " ": ["....."] * 7,
}


def text_mask(s, x, y, k=1, spacing=1):
    m = np.zeros((CH, CW), bool)
    cx = x
    for ch in s:
        g = FONT[ch]
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == "#":
                    y0, x0 = y + r * k + M, cx + c * k + M
                    if 0 <= y0 < CH - k and 0 <= x0 < CW - k:
                        m[y0:y0 + k, x0:x0 + k] = True
        cx += (5 + spacing) * k
    return m


def text_width(s, k=1, spacing=1):
    return len(s) * (5 + spacing) * k - spacing * k


def draw_text(sc, s, x, y, k=1, fill="white", edge="ob0", shade=None, layer="Text", outline=True):
    m = text_mask(s, x, y, k)
    if outline:
        o = dilate4(m)
        o = o | nb(o, -1, -1)
        sc.put(o & ~m, edge, layer)
    sc.put(m, fill, layer)
    if shade:
        sc.put(m & ~nb(m, 0, k), shade, layer)     # bottom row of each stroke darker
    return m


# ---------------------------------------------------------------- face close-up
FACE_C = 160


def _mirror_pts(pts):
    return [(2 * FACE_C - x, y) for (x, y) in reversed(pts)]


def portrait(sc, f, talk=0, lower_eyes=1.0, grin=0.0, glow=1.0, sway=0.0):
    """Frontal close-up of Sukuna. talk: 0 closed .. 2 wide open; lower_eyes 0 shut .. 1 open."""
    oy = int(round(sway))
    # background: deep red with radial streaks
    ang = np.degrees(np.arctan2(YS - 70, XS - FACE_C))
    rad = np.hypot(XS - FACE_C, YS - 70)
    t = np.clip(rad / 190, 0, 1)
    bg = np.stack([np.interp(t, [0, 0.35, 0.7, 1], [C[n][k] for n in ("dm3", "dm2", "dm1", "dm0")])
                   for k in range(3)], -1)
    sc.rgb[:] = bg
    sc.owner[:] = L["Sky"]
    sc.dith[:] = 30
    rays = (np.mod(ang + f * 1.5, 12.0) < 1.3) & (rad > 70)
    sc.put(rays, "dm3", "Sky")
    sc.put(rays & (rad > 120), "dm2", "Sky")

    def P(x, y):
        return (x, y + oy)

    # neck + collar
    neck = polygon([P(140, 100), P(180, 100), P(186, 140), P(134, 140)])
    sc.put(neck, "sk1", "Sukuna")
    sc.put(neck & (XS > 166), "sk0", "Sukuna")
    collar_l = polygon([P(96, 180), P(118, 126), P(138, 120), P(160, 162), P(160, 180)])
    collar_r = polygon([P(224, 180), P(202, 126), P(182, 120), P(160, 162), P(160, 180)])
    for cm, tone in ((collar_r, "ki2"), (collar_l, "ki3")):
        sc.put(outline_of(cm), "ki0", "Sukuna")
        sc.put(cm, tone, "Sukuna")
    sc.put(polygon([P(138, 120), P(160, 162), P(182, 120), P(160, 140)]), "ob1", "Sukuna")
    # ears
    for ex in (106, 214):
        em = ellipse(P(ex + (2 if ex < 160 else -2), 64), 4, 8)
        sc.put(outline_of(em), "sk0", "Sukuna")
        sc.put(em, "sk1", "Sukuna")
        sc.put(ellipse(P(ex + (3 if ex < 160 else -3), 64), 1.5, 4.5), "sk0", "Sukuna")
    # face shape
    half = [P(160, 14), P(132, 18), P(116, 30), P(111, 52), P(112, 72), P(118, 90), P(128, 104), P(142, 116),
            P(154, 121), P(160, 122)]
    face = polygon(half + _mirror_pts(half[1:-1]))
    sc.put(outline_of(face), "sk0", "Sukuna")
    sc.put(face, "sk2", "Sukuna")
    sc.put(face & (XS > 172), "sk1", "Sukuna")
    sc.put(face & (XS > 164) & (XS <= 172) & ((XS + YS) % 2 == 0), "sk1", "Sukuna")
    sc.put(face & (XS < 140) & (YS < 60 + oy) & (YS > 24 + oy), "sk3", "Sukuna")
    # jaw / cheek shading
    sc.put(face & ~nb(face, 0, 3) & (XS > 140), "sk1", "Sukuna")
    sc.put(face & polygon([P(176, 78), P(204, 66), P(200, 92), P(180, 104)]), "sk1", "Sukuna")
    # red rim light on the shadow side
    sc.put(face & ~nb(face, 2, 0) & (XS > 170), "re1", "Sukuna")
    # hair: swept-up crown of spikes
    rng = np.random.default_rng(3)
    spikes = []
    for k in range(17):
        a = -168 + k * (156 / 16)
        base = v2((FACE_C, 46 + oy)) + unit(a) * v2((54, 34))
        ln = rng.uniform(26, 44) * (1.0 if abs(a + 90) < 60 else 0.7)
        tip = base + unit(a + rng.uniform(-14, 14) - 18 * math.copysign(1, a + 90) * 0) * ln
        spikes.append((base, tip, rng.uniform(9, 14)))
    hair = np.zeros_like(face)
    shades = []
    for base, tip, wd in spikes:
        d = (tip - base) / max(np.hypot(*(tip - base)), 1e-6)
        pp = v2((-d[1], d[0]))
        sp = polygon([base + pp * wd, tip, base - pp * wd])
        hair |= sp
        shades.append(sp)
    cap = ellipse(P(FACE_C, 34), 56, 26) & (YS < 40 + oy)
    side_l = polygon([P(108, 30), P(122, 24), P(120, 58), P(112, 62)])
    side_r = polygon(_mirror_pts([P(108, 30), P(122, 24), P(120, 58), P(112, 62)]))
    hair |= cap | side_l | side_r
    sc.put(outline_of(hair), "ph0", "Sukuna")
    sc.put(hair, "ph2", "Sukuna")
    for i, sp in enumerate(shades):
        sc.put(sp & ~nb(sp, -2, 0), "ph3" if i % 2 else "ph1", "Sukuna")
        sc.put(sp & ~nb(sp, 2, 0), "ph1", "Sukuna")
    sc.put(hair & ~nb(hair, 0, 2) & (YS > 20 + oy), "ph1", "Sukuna")           # shadow under the fringe
    sc.put(hair & polygon([P(120, 0), P(150, -5), P(140, 30), P(126, 34)]), "ph3", "Sukuna")
    sc.put(hair & polygon([P(128, -2), P(140, -6), P(134, 18)]), "ph4", "Sukuna")
    # forehead marks
    for sx in (-1, 1):
        x0 = FACE_C + sx * 14
        xs, ys = line_pts(P(x0, 40), P(x0 + sx * 10, 44))
        sc.put_px(xs, ys, "ol", "Sukuna")
        sc.put_px(xs, ys + 1, "ol", "Sukuna")
    # brows: thick, angled down to the centre
    for sx in (-1, 1):
        br = polygon([P(FACE_C + sx * 6, 52), P(FACE_C + sx * 32, 44), P(FACE_C + sx * 36, 47),
                      P(FACE_C + sx * 8, 57)])
        sc.put(br, "ph0", "Sukuna")
        sc.put(br & ~nb(br, 0, -1), "ph1", "Sukuna")
    # main eyes
    for sx in (-1, 1):
        ex = FACE_C + sx * 21
        eye = polygon([P(ex - 13 * sx, 60), P(ex - 4 * sx, 55), P(ex + 9 * sx, 56), P(ex + 14 * sx, 60),
                       P(ex + 6 * sx, 64), P(ex - 6 * sx, 64)])
        sc.put(eye, "ki4", "Sukuna")
        iris = disc(P(ex - 2 * sx, 60), 4.6) & eye
        sc.put(iris, "re1", "Sukuna")
        sc.put(iris & disc(P(ex - 3 * sx, 59), 3.0), "re2", "Sukuna")
        sc.put(iris & (np.abs(XS - (ex - 2 * sx)) < 1.0), "ob0", "Sukuna")     # slit pupil
        sc.put_px([ex - 4 * sx], [57 + oy], "white", "Sukuna")
        lid = eye & ~nb(eye, 0, -2)
        sc.put(lid, "ol", "Sukuna")
        sc.put(outline_of(eye) & (YS < 61 + oy), "ol", "Sukuna")
        sc.put(outline_of(eye) & (YS >= 61 + oy), "sk0", "Sukuna")
        # lash flick at the outer corner
        xs, ys = line_pts(P(ex + 14 * sx, 60), P(ex + 18 * sx, 56))
        sc.put_px(xs, ys, "ol", "Sukuna")
        if glow > 0:
            sc.light(np.exp(-((XS - ex) ** 2 + (YS - 60 - oy) ** 2) / 60.0), C["re2"], 0.35 * glow)
    # lower pair of eyes on the cheeks
    for sx in (-1, 1):
        ex = FACE_C + sx * 23
        h = 0.6 + 2.6 * lower_eyes
        e2 = ellipse(P(ex, 73), 8, h)
        sc.put(outline_of(e2), "ol", "Sukuna")
        if lower_eyes > 0.15:
            sc.put(e2, "re1", "Sukuna")
            sc.put(e2 & disc(P(ex - sx, 73), 2.5), "re2", "Sukuna")
            sc.put(e2 & (np.abs(XS - ex) < 0.8), "ob0", "Sukuna")
        else:
            sc.put(e2, "ol", "Sukuna")
    # tattoo stripes under the lower eyes
    for sx in (-1, 1):
        for k, yy in enumerate((80, 85)):
            x_in, x_out = FACE_C + sx * 15, FACE_C + sx * (37 - k * 4)
            xs, ys = line_pts(P(x_in, yy), P(x_out, yy + 1))
            sc.put_px(xs, ys, "ol", "Sukuna")
            sc.put_px(xs, ys + 1, "ol", "Sukuna")
    # nose
    xs, ys = line_pts(P(163, 64), P(165, 86))
    sc.put_px(xs, ys, "sk1", "Sukuna")
    sc.put_px([155, 156, 164, 165], [89 + oy] * 4, "sk0", "Sukuna")
    sc.put_px([157, 158, 159, 160, 161, 162, 163], [90 + oy] * 7, "sk1", "Sukuna")
    # mouth: a wide grin; talk opens it
    open_h = [1.0, 4.0, 7.0][int(talk)] + 2.0 * grin
    top = [P(132, 96 - 4 * grin), P(146, 100), P(160, 101), P(174, 100), P(188, 96 - 4 * grin)]
    bot = [P(188, 96 - 4 * grin), P(174, 100 + open_h), P(160, 101 + open_h * 1.2), P(146, 100 + open_h),
           P(132, 96 - 4 * grin)]
    mouth = polygon(top + bot[1:-1])
    sc.put(mouth, "re0", "Sukuna")
    teeth_top = mouth & (YS < 103 + oy - (0 if talk else 1))
    sc.put(teeth_top, "ki4", "Sukuna")
    sc.put(teeth_top & (np.mod(XS - 132, 5) == 0), "ki1", "Sukuna")
    if open_h > 4:
        low = mouth & (YS > 100 + oy + open_h - 2.5)
        sc.put(low, "ki3", "Sukuna")
        sc.put(low & (np.mod(XS - 134, 5) == 0), "ki1", "Sukuna")
    for sx in (-1, 1):     # fangs
        fx_ = FACE_C + sx * 12
        sc.put(polygon([P(fx_ - 2, 100), P(fx_ + 2, 100), P(fx_, 106)]) & mouth, "white", "Sukuna")
    sc.put(outline_of(mouth), "ol", "Sukuna")
    # smile creases
    for sx in (-1, 1):
        xs, ys = line_pts(P(FACE_C + sx * 30, 94 - 4 * grin), P(FACE_C + sx * 34, 102))
        sc.put_px(xs, ys, "sk0", "Sukuna")


def dialogue(sc, text, n_chars, f, name="SUKUNA"):
    """Two-line dialogue box at the bottom with a typewriter reveal."""
    x0, y0, x1, y1 = 14, 118, 306, 165
    box = (XS >= x0) & (XS <= x1) & (YS >= y0) & (YS <= y1)
    sc.put(box, "ob0", "Text")
    edge = box & ~(nb(box, 1, 0) & nb(box, -1, 0) & nb(box, 0, 1) & nb(box, 0, -1))
    sc.put(edge, "ki4", "Text")
    sc.put(dilate4(edge) & box & ~edge, "dm4", "Text")
    tag = (XS >= x0 + 6) & (XS <= x0 + 6 + text_width(name) + 7) & (YS >= y0 - 6) & (YS <= y0 + 5)
    sc.put(tag, "dm4", "Text")
    sc.put(tag & ~(nb(tag, 1, 0) & nb(tag, -1, 0) & nb(tag, 0, 1) & nb(tag, 0, -1)), "ki4", "Text")
    draw_text(sc, name, x0 + 10, y0 - 3, 1, fill="ki4", outline=False)
    lines = text.split("\n")
    shown = n_chars
    y = y0 + 12
    for ln in lines:
        part = ln[:max(0, shown)]
        shown -= len(ln)
        if part:
            x = (x0 + x1) // 2 - text_width(ln, 2) // 2
            draw_text(sc, part, x, y, 2, fill="ki4", edge="ob0", shade="ki2", outline=False)
        y += 17
    if n_chars >= sum(len(l) for l in lines) and (f // 4) % 2 == 0:
        sc.put((XS >= x1 - 14) & (XS <= x1 - 9) & (YS >= y1 - 8) & (YS <= y1 - 5) &
               (np.abs(XS - (x1 - 11.5)) <= (y1 - 5 - YS) * 0.9), "re2", "Text")


def letterbox(sc, k=1.0, h=14):
    hh = int(round(h * k))
    if hh <= 0:
        return
    bars = (YS < hh) | (YS >= H - hh)
    sc.put(bars, "ob0", "Text")
    sc.put((YS == hh) | (YS == H - hh - 1), "dm4", "Text")


# ---------------------------------------------------------------- Malevolent Shrine
SKULL = [".###.", "#####", "#o#o#", "#####", ".#.#."]


def draw_shrine(sc, bx, gy, rise=1.0, f=0, layer="Domain", clip_y=None):
    """Shrine standing on a mound of skulls; base centre (bx, gy). rise 0..1 lifts it out of the pool."""
    oy = (1 - smooth01(rise)) * 150
    clip_y = gy if clip_y is None else clip_y
    above = YS < clip_y

    def p(x, y):
        return v2((bx + x, gy + y + oy))

    def put(mask, col):
        sc.put(mask & above, col, layer)

    def box(x0, y0, x1, y1):
        a, b = p(x0, y0), p(x1, y1)
        return (XS >= a[0]) & (XS <= b[0]) & (YS >= a[1]) & (YS <= b[1])

    # horns
    for sx in (-1, 1):
        pts = [p(sx * 18, -110), p(sx * 26, -122), p(sx * 30, -134), p(sx * 28, -146)]
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            hm = capsule(a, b, 2.6 - 0.6 * i, 2.0 - 0.6 * i)
            put(outline_of(hm), "bn0")
            put(hm, "bn2")
            put(hm & ~nb(hm, sx, 0), "bn3")
    # upper roof
    up = polygon([p(-52, -102), p(-44, -100), p(-30, -114), p(30, -114), p(44, -100), p(52, -102), p(46, -96),
                  p(-46, -96)])
    put(outline_of(up), "dm0")
    put(up, "ob1")
    put(up & (np.mod(XS - bx, 4) == 0), "ob2")
    put(up & ~nb(up, 0, 1), "dm5")
    # upper block with an eye emblem
    blk = box(-28, -96, 28, -84)
    put(blk, "dm3")
    put(blk & ~nb(blk, 0, -1), "dm4")
    emb = ellipse(p(0, -90), 7, 3)
    put(emb, "bn3")
    put(disc(p(0, -90), 2.6) & emb, "re2" if (f // 3) % 4 else "re1")
    put(disc(p(0, -90), 1.0), "ob0")
    # lower roof
    lo = polygon([p(-70, -76), p(-60, -74), p(-44, -90), p(44, -90), p(60, -74), p(70, -76), p(62, -68),
                  p(-62, -68)])
    put(outline_of(lo), "dm0")
    put(lo, "ob1")
    put(lo & (np.mod(XS - bx, 4) == 0), "ob2")
    put(lo & ~nb(lo, 0, 1), "dm5")
    put(lo & ~nb(lo, 0, -1), "dm3")
    # lintel
    lin = box(-46, -68, 46, -62)
    put(lin, "dm3")
    put(lin & ~nb(lin, 0, -1), "dm5")
    # dark interior + the great mouth
    inside = box(-38, -62, 38, -24)
    put(inside, "dm0")
    mouth = ellipse(p(0, -43), 27, 15)
    put(mouth, "dm2")
    put(mouth & ellipse(p(0, -41), 20, 9), "re1" if (f // 2) % 3 else "dm4")
    for k in range(-24, 25, 6):
        tt = polygon([p(k - 3, -57), p(k + 3, -57), p(k, -50 + abs(k) * 0.12)])
        tb = polygon([p(k, -36 - abs(k) * 0.12), p(k - 3, -29), p(k + 3, -29)])
        for tm in (tt, tb):
            put(tm & mouth, "bn3")
            put(tm & mouth & ~nb(tm, 1, 0), "bn1")
    # pillars
    for x in (-40, -18, 18, 40):
        pl = box(x - 3, -62, x + 3, -22)
        put(outline_of(pl), "dm0")
        put(pl, "dm4")
        put(pl & ~nb(pl, -1, 0), "dm5")
        put(pl & ~nb(pl, 1, 0), "dm2")
    # platform
    plat = box(-50, -24, 50, -18)
    put(plat, "dm3")
    put(plat & ~nb(plat, 0, -1), "dm5")
    put(outline_of(plat), "dm0")
    # mound of skulls
    mound = polygon([p(-64, 2), p(-54, -10), p(-40, -18), p(40, -18), p(54, -10), p(64, 2)])
    put(mound, "bn0")
    rows = 0
    for yy in range(-17, 3, 5):
        off = 3 if rows % 2 else 0
        for xx in range(-64 + off, 64, 6):
            for r, row in enumerate(SKULL):
                for c, v in enumerate(row):
                    if v == ".":
                        continue
                    q = p(xx + c, yy + r)
                    qi, qj = int(round(q[1])) + M, int(round(q[0])) + M
                    if not (0 <= qi < CH and 0 <= qj < CW) or not mound[qi, qj] or not above[qi, qj]:
                        continue
                    col = "ob0" if v == "o" else ("bn3" if r == 0 else ("bn1" if c == 4 or r == 4 else "bn2"))
                    sc.put_px([q[0]], [q[1]], col, layer)
        rows += 1


def domain_world(sc, f, horizon=126, shrine_x=262, shrine_rise=1.0, moon=True):
    """Blood-red sky, black horizon and a pool that mirrors the shrine."""
    t = np.clip((YS + 10) / (horizon + 10), 0, 1)
    stops = [(0, "dm0"), (0.45, "dm1"), (0.8, "dm2"), (1.0, "dm3")]
    sky = np.stack([np.interp(t, [s[0] for s in stops], [C[s[1]][k] for s in stops]) for k in range(3)], -1)
    sc.rgb[:] = sky
    sc.owner[:] = L["Domain"]
    sc.dith[:] = 28
    if moon:
        mc = v2((shrine_x, 58))
        mm = disc(mc, 40)
        halo = np.clip(1 - (np.hypot(XS - mc[0], YS - mc[1]) - 40) / 40, 0, 1) ** 2
        sc.light(halo, C["dm4"], 0.6)
        sc.put(mm, "dm4", "Domain")
        sc.put(mm & ~disc(mc + v2((-3, -3)), 39), "dm3", "Domain")
        sc.put(mm & disc(mc + v2((-12, -10)), 14), "dm5", "Domain")
        sc.put(mm & disc(mc + v2((10, 12)), 8), "dm3", "Domain")
    # far black ridge
    ridge = YS > horizon - 3 - 2 * np.sin(XS * 0.07) - 1.5 * np.sin(XS * 0.23 + 1)
    sc.put(ridge & (YS < horizon), "dm0", "Domain")
    # the pool
    pool = YS >= horizon
    tp = np.clip((YS - horizon) / (H - horizon), 0, 1)
    pc = np.stack([np.interp(tp, [0, 0.5, 1], [C[n][k] for n in ("dm1", "dm2", "dm3")]) for k in range(3)], -1)
    sc.put_rgb(pool, pc, "Domain", dither=24)
    rip = pool & (np.mod(XS * 0.5 + np.sin(YS * 1.7) * 6 + f * 1.3 + YS * 3.1, 23) < 3 + tp * 4) & \
        (np.mod(YS, 3) == 0)
    sc.put(rip, "dm4", "Domain")
    if shrine_rise > 0:
        tmp = Scene()
        draw_shrine(tmp, shrine_x, 150, shrine_rise, f)
        m = tmp.owner >= 0
        sc.put_rgb(m, tmp.rgb, "Domain")
        # wobbly reflection under the waterline
        yy = np.arange(CH)
        src_y = np.clip(2 * (150 + M) - yy, 0, CH - 1)
        wob = np.round(np.sin(yy * 0.9 + f * 0.6) * 1.2).astype(int)
        src_x = np.clip(np.arange(CW)[None, :] + wob[:, None], 0, CW - 1)
        rm = m[src_y[:, None], src_x] & (yy[:, None] > 150 + M)
        rgb = tmp.rgb[src_y[:, None], src_x] * 0.45 + C["dm2"] * 0.25
        sc.put_rgb(rm & ((XS + YS) % 2 == 0) | (rm & (YS < 162)), rgb, "Domain", dither=20)


# ---------------------------------------------------------------- slashes
def _slash_set(f, n, seed=0):
    r = np.random.default_rng(f * 101 + seed)
    out = []
    for _ in range(n):
        c = v2((r.uniform(-10, W + 10), r.uniform(10, H - 6)))
        a = r.uniform(0, 180)
        ln = r.uniform(30, 140)
        out.append((c, a, ln))
    return out


def slashes(sc, f, rate, start, layer="FX", seed=0, x_min=-99):
    """Dismantle: invisible blades -- white cuts that flash, split open in red and fade.
    Nothing is drawn left of x_min (behind Rudeus' wall)."""
    for age in range(3):
        fb = f - age
        if fb < start:
            continue
        for c, a, ln in _slash_set(fb, int(rate(fb)), seed):
            d = unit(a)
            p0, p1 = c - d * ln / 2, c + d * ln / 2
            xs, ys = line_pts(p0, p1)
            keep = xs >= x_min + 2
            xs, ys = xs[keep], ys[keep]
            pp = v2((-d[1], d[0]))
            n = len(xs)
            if n < 3:
                continue
            if age == 0:
                # thin at the ends, bright core
                k = np.arange(n)
                core = (k > n * 0.08) & (k < n * 0.92)
                sc.put_px(xs + round(pp[0]), ys + round(pp[1]), "dm6", layer)
                sc.put_px(xs, ys, "white", layer)
                sc.put_px(xs[core] - round(pp[0]), ys[core] - round(pp[1]), "white", layer)
            elif age == 1:
                k = np.arange(n)
                mid = (k > n * 0.15) & (k < n * 0.85)
                sc.put_px(xs + round(pp[0] * 2), ys + round(pp[1] * 2), "dm5", layer)
                sc.put_px(xs - round(pp[0]), ys - round(pp[1]), "dm5", layer)
                sc.put_px(xs[mid], ys[mid], "ob0", layer)
                sc.put_px(xs[mid] + round(pp[0]), ys[mid] + round(pp[1]), "ob0", layer)
            else:
                keep = (np.arange(n) % 3) == 0
                sc.put_px(xs[keep], ys[keep], "dm4", layer)


def cleave(sc, c, r, a0, a1, t, layer="FX", x_min=-99):
    """Big crescent slash; t = frames since it was struck."""
    if t < 0 or t > 3:
        return
    sc_protect = sc.protect
    sc.protect = (XS < x_min) if sc_protect is None else (sc_protect | (XS < x_min))
    rr = np.hypot(XS - c[0], YS - c[1])
    a = np.degrees(np.arctan2(YS - c[1], XS - c[0]))
    rel = np.mod(a - a0, 360) / max(1e-6, np.mod(a1 - a0, 360))
    band = (rel <= 1) & (np.abs(rr - r) < 1 + 5 * np.sin(np.clip(rel, 0, 1) * math.pi) * (1 - t / 4))
    core = band & (np.abs(rr - r) < 1 + 2.4 * np.sin(np.clip(rel, 0, 1) * math.pi) * (1 - t / 4))
    if t == 0:
        sc.put(band, "dm6", layer)
        sc.put(core, "white", layer)
    elif t == 1:
        sc.put(band, "dm5", layer)
        sc.put(core, "ki4", layer)
    else:
        sc.put(band & ((XS + YS) % 2 == 0), "dm4", layer)
    sc.protect = sc_protect


def red_aura(sc, mask, f, power, layer="Back FX"):
    if power <= 0:
        return
    r1 = dilate4(mask) & ~mask
    rr = np.random.default_rng(f + 77)
    noise = rr.uniform(0, 1, CW)
    top = mask & ~nb(mask, 0, -1)
    flame = np.zeros_like(mask)
    hmax = int(3 + 7 * power)
    for k in range(1, hmax + 1):
        flame |= nb(top, 0, k) & (noise[None, :] > 0.35 + 0.6 * k / hmax)
    flame &= ~mask & ~r1
    free = ~mask
    sc.put(flame & free & ((XS + YS + f) % 2 == 0), "dm5", layer)
    sc.put(flame & free & ((XS + YS + f) % 2 == 1) & (noise[None, :] > 0.7), "ob0", layer)
    sc.put(r1 & free, "re2" if power > 0.6 else "dm5", layer)
