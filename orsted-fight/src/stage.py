"""Twilight mountain pass, drawn once (with the camera-shake margin) into the four static layers."""
import math
import numpy as np
from px import Canvas, hexc, nb, dilate4, ring4, v2
from world import W, H, M, GROUND, LI

SKY = [hexc(c) for c in ("110f22", "171530", "1f1b3e", "2a234e", "382c5e", "4a386c", "5e4476", "77527e")]
MTN_FAR = (hexc("2c2650"), hexc("3a3264"), hexc("524880"), hexc("8a80b4"))   # base, rim, lit, snow
MTN_NEAR = (hexc("1c1834"), hexc("262044"), hexc("342c58"))
CLIFF = (hexc("0e0c16"), hexc("1a1626"), hexc("262036"), hexc("342c48"), hexc("463c5e"))
GROUND_C = (hexc("221c2a"), hexc("2c2434"), hexc("3a2f40"), hexc("4a3d4c"), hexc("5c4c58"), hexc("6e5c66"))
GRASS = (hexc("16242a"), hexc("223a3c"), hexc("30524e"), hexc("44706a"))
MOON = (hexc("fff6e4"), hexc("efe2c8"), hexc("cfc0a8"), hexc("a89c8c"))
STAR = (hexc("ffffff"), hexc("c8c4f0"), hexc("8c86c0"))


def _profile(xs, seed, base, amp, octaves):
    rng = np.random.default_rng(seed)
    y = np.full(xs.shape, float(base))
    for i, (wl, a) in enumerate(octaves):
        ph = rng.uniform(0, 2 * math.pi)
        y += amp * a * np.sin(xs / wl * 2 * math.pi + ph)
    return y


def build():
    cv = Canvas(W + 2 * M, H + 2 * M, -M, -M)
    X, Y = cv.XS, cv.YS
    own = np.full((cv.h, cv.w), LI["Sky"], np.int16)
    rgba = np.zeros((cv.h, cv.w, 4), np.uint8)

    # ---------------- sky: dithered bands
    top, bottom = -M, 215
    t = np.clip((Y - top) / (bottom - top), 0, 1) ** 1.15 * (len(SKY) - 1)
    lo = np.floor(t).astype(int)
    frac = t - lo
    B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float) / 16.0
    thr = B[(Y.astype(int) % 4), (X.astype(int) % 4)]
    idx = np.clip(lo + (frac > thr), 0, len(SKY) - 1)
    pal = np.array(SKY, np.uint8)
    rgba[:] = pal[idx]

    # stars
    rng = np.random.default_rng(7)
    for _ in range(140):
        sx, sy = rng.integers(-M, W + M), int(rng.integers(-M, 150) ** 1.0)
        if sy > 140 * rng.random() + 10:
            continue
        b = rng.random()
        c = STAR[0] if b > 0.92 else STAR[1] if b > 0.6 else STAR[2]
        rgba[cv.point((sx, sy))] = c
        if b > 0.97:
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rgba[cv.point((sx + d[0], sy + d[1]))] = STAR[2]

    # moon (upper left: the light source for all the shading)
    mc, mr = v2((84, 52)), 15
    glow2 = cv.disc(mc, mr + 9) & cv.bayer(0.25)
    glow1 = cv.disc(mc, mr + 4) & cv.bayer(0.5)
    rgba[glow2] = SKY[4]
    rgba[glow1] = SKY[5]
    moon = cv.disc(mc, mr)
    rgba[moon] = MOON[1]
    rgba[moon & cv.disc(mc + v2((-3, -3)), mr - 3)] = MOON[0]
    dark = moon & ~cv.disc(mc + v2((-5, -4)), mr + 1)
    rgba[dark] = MOON[2]
    for c, r in (((90, 50), 2.5), ((78, 58), 2.0), ((88, 61), 1.5), ((80, 46), 1.4), ((94, 57), 1.2)):
        cr = cv.disc(c, r) & moon
        rgba[cr] = MOON[2]
        rgba[cr & ~nb(cr, 1, 1)] = MOON[3]
    # thin cloud streaks crossing the moon
    for (x0, y0, ln) in ((40, 60, 90), (60, 66, 60), (170, 40, 70), (300, 30, 110), (360, 72, 80)):
        streak = (np.abs(Y - y0) < 1.0) & (X > x0) & (X < x0 + ln)
        streak |= (np.abs(Y - (y0 + 2)) < 1.0) & (X > x0 + 12) & (X < x0 + ln - 18)
        rgba[streak & cv.bayer(0.7)] = SKY[3] if y0 < 50 else SKY[4]

    # ---------------- far mountains
    xs = np.arange(-M, W + M, dtype=float)
    far = _profile(xs, 3, 168, 18, [(140, 1.0), (63, 0.55), (27, 0.25), (11, 0.12)])
    far -= 30 * np.exp(-((xs - 330) / 40) ** 2) + 22 * np.exp(-((xs - 180) / 30) ** 2)
    fm = Y >= far[None, :][:, (X[0].astype(int) + M)]
    farline = far[(X.astype(int) + M)]
    fm = Y >= farline
    rgba[fm] = MTN_FAR[0]
    own[fm] = LI["Mountains"]
    slope = np.gradient(far)
    lit = fm & (Y < farline + 3) & (slope[(X.astype(int) + M)] < -0.15)
    rgba[lit] = MTN_FAR[2]
    rgba[fm & (Y < farline + 1.2)] = MTN_FAR[1]
    snow = fm & (Y < farline + 6 - (farline - 135) * 0.15) & (farline < 150)
    rgba[snow & cv.bayer(0.8)] = MTN_FAR[3]
    rgba[snow & (slope[(X.astype(int) + M)] > 0.25)] = MTN_FAR[2]
    # a few gullies running down from the peaks
    rng0 = np.random.default_rng(9)
    for gx in rng0.uniform(-M, W + M, 26):
        y0 = far[int(gx) + M]
        g = cv.line((gx, y0 + 2), (gx + rng0.uniform(-6, 6), y0 + rng0.uniform(10, 22)))
        rgba[g & fm] = MTN_FAR[1]

    # ---------------- near mountains
    near = _profile(xs, 11, 196, 10, [(110, 1.0), (47, 0.6), (19, 0.3), (7, 0.15)])
    nearline = near[(X.astype(int) + M)]
    nm = Y >= nearline
    rgba[nm] = MTN_NEAR[0]
    own[nm] = LI["Mountains"]
    nslope = np.gradient(near)
    nlit = nm & (Y < nearline + 2) & (nslope[(X.astype(int) + M)] < -0.1)
    rgba[nlit] = MTN_NEAR[2]
    rgba[nm & (Y < nearline + 1) & ~nlit] = MTN_NEAR[1]
    # pines on the near ridge
    rng = np.random.default_rng(5)
    for px_ in rng.uniform(-M, W + M, 70):
        base_y = near[int(px_) + M] + 2
        hgt = rng.uniform(5, 11)
        tri = cv.polygon([(px_, base_y - hgt), (px_ - hgt * 0.32, base_y + 1), (px_ + hgt * 0.32, base_y + 1)])
        rgba[tri] = MTN_NEAR[0]
        own[tri] = LI["Mountains"]
        rgba[tri & ~nb(tri, -1, 0) & nb(tri, 1, 0)] = MTN_NEAR[1]

    # ---------------- ground
    gy0 = 214
    gm = Y >= gy0 + np.sin(X * 0.05) * 1.5
    tg = np.clip((Y - gy0) / (H + M - gy0), 0, 1)
    gidx = np.clip((tg ** 0.8) * 4.2 + thr * 0.9, 0, 5).astype(int)
    gp = np.array(GROUND_C, np.uint8)
    rgba[gm] = gp[gidx][gm]
    own[gm] = LI["Ground"]
    # the road: a lighter worn band around the fighting line
    road = gm & (np.abs(Y - (GROUND + 2)) < 9 + np.sin(X * 0.03) * 1.5)
    rgba[road & (gidx < 5)] = GROUND_C[4]
    rgba[road & (np.abs(Y - (GROUND + 2)) < 5) & cv.bayer(0.35)] = GROUND_C[5]
    rgba[gm & (np.abs(Y - (GROUND + 2) + 9.5 + np.sin(X * 0.03) * 1.5) < 0.7) & cv.checker()] = GROUND_C[3]
    # horizon line where ground meets the near mountains
    rgba[gm & ~nb(gm, 0, -1)] = GROUND_C[1]
    # pebbles and rocks
    rng = np.random.default_rng(21)
    for _ in range(170):
        x = rng.uniform(-M, W + M)
        y = rng.uniform(gy0 + 3, H + M)
        r = rng.uniform(0.6, 1.6) * (0.6 + (y - gy0) / 60)
        if r < 0.8:
            rgba[cv.point((x, y))] = GROUND_C[1] if rng.random() < 0.5 else GROUND_C[5]
            continue
        st = cv.ellipse((x, y), r * 1.4, r)
        rgba[ring4(st) & nb(st, 0, -1)] = GROUND_C[0]
        rgba[st] = GROUND_C[3]
        rgba[st & ~nb(st, 0, -1)] = GROUND_C[5]
    # grass tufts, thicker towards the edges of the road
    for _ in range(140):
        x = rng.uniform(-M, W + M)
        y = rng.uniform(gy0 + 2, H + M)
        if abs(y - (GROUND + 2)) < 8 and rng.random() < 0.85:
            continue
        s = 0.6 + (y - gy0) / 50
        for b in range(int(3 + s * 2)):
            bx = x + rng.uniform(-2.5, 2.5) * s
            hgt = rng.uniform(2, 5) * s
            lean = rng.uniform(-1.5, 1.5)
            blade = cv.line((bx, y), (bx + lean, y - hgt))
            rgba[blade] = GRASS[rng.integers(1, 4)]
            own[blade] = LI["Ground"]
            rgba[cv.point((bx, y))] = GRASS[0]

    # ---------------- cliffs framing the pass (the "jaw")
    def cliff(pts, seed):
        m = cv.polygon(pts)
        rr = np.random.default_rng(seed)
        rgba[m] = CLIFF[1]
        own[m] = LI["Cliffs"]
        # strata
        for _ in range(26):
            y = rr.uniform(0, H)
            sl = rr.uniform(-0.25, 0.25)
            band = m & (np.abs(Y - (y + (X - pts[0][0]) * sl)) < rr.uniform(0.5, 1.6))
            rgba[band] = CLIFF[2] if rr.random() < 0.6 else CLIFF[0]
        # angular rock facets: lit on the upper-left, cracked on the lower-right
        x_lo, x_hi = min(p[0] for p in pts), max(p[0] for p in pts)
        for _ in range(30):
            c = v2((rr.uniform(x_lo, x_hi), rr.uniform(-M, H)))
            s_ = rr.uniform(4, 11)
            quad = [c + v2((rr.uniform(-1, 0.2) * s_, rr.uniform(-1, -0.4) * s_ * 0.7)),
                    c + v2((rr.uniform(0.4, 1) * s_, rr.uniform(-0.8, -0.1) * s_ * 0.7)),
                    c + v2((rr.uniform(0.3, 1) * s_, rr.uniform(0.3, 1) * s_ * 0.6)),
                    c + v2((rr.uniform(-1, -0.3) * s_, rr.uniform(0.2, 0.8) * s_ * 0.6))]
            blk = cv.polygon(quad) & m
            rgba[blk] = CLIFF[2] if rr.random() < 0.7 else CLIFF[1]
            rgba[blk & ~nb(blk, -1, 0)] = CLIFF[3]
            rgba[blk & ~nb(blk, 0, -1)] = CLIFF[3]
            rgba[ring4(blk) & m & ~nb(blk, -1, -1) & nb(blk, -1, 0)] = CLIFF[0]
            rgba[(ring4(blk) & m) & (nb(blk, -1, 0) | nb(blk, 0, -1))] = CLIFF[0]
        edge = m & ~(nb(m, -1, 0) & nb(m, 0, -1) & nb(m, 1, 0))
        rgba[edge] = CLIFF[3]
        lit = m & ~nb(m, -1, 0) | (m & ~nb(m, 0, -1))
        rgba[lit] = CLIFF[4]
        rgba[ring4(m) & ~m & (own != LI["Ground"])] = CLIFF[0]
        own[ring4(m) & ~m & (own != LI["Ground"])] = LI["Cliffs"]
        return m

    cliff([(-M, -M), (12, -M), (22, 30), (16, 62), (30, 92), (26, 130), (40, 160), (36, 186), (52, 212),
           (66, 236), (58, 252), (-M, 252)], 2)
    cliff([(W + M, -M), (462, -M), (452, 24), (458, 56), (444, 88), (450, 120), (436, 150), (442, 176),
           (426, 204), (414, 232), (424, 252), (W + M, 252)], 4)
    # foreground boulder bottom-right and bottom-left
    for c, rx, ry in (((448, 268), 34, 16), ((14, 270), 30, 14)):
        b = cv.ellipse(c, rx, ry)
        rgba[b] = CLIFF[1]
        rgba[b & ~nb(b, 0, -2)] = CLIFF[3]
        rgba[b & ~nb(b, 0, -1)] = CLIFF[4]
        rgba[ring4(b)] = CLIFF[0]
        own[b | ring4(b)] = LI["Cliffs"]

    return rgba, own


if __name__ == "__main__":
    import sys
    from PIL import Image
    rgba, own = build()
    img = rgba[M:M + H, M:M + W]
    Image.fromarray(img).resize((W * 2, H * 2), Image.NEAREST).save(sys.argv[1])
