"""Night-plain backdrop: banded sky, moon, stars, two mountain ridges and a ground plane.

`dim` darkens the whole scene (the sky closing in while the spell charges), `lit` is the light of
the beam spilling over sky, ridges and ground, and `flash` swaps everything for a two-tone
impact frame.
"""
import numpy as np
from canvas import C, XS, YS, XI, YI, BAYER, HORIZON, W, H, SHAPE, hexc, disc, pixels, nb

SKY = [hexc(c) for c in ("0a0920", "11102e", "191640", "231d56", "30266e", "443088", "5e3c98",
                         "7c58b8", "a486e0", "d2c4ff")]
SKY_EDGES = [10, 24, 38, 52, 64, 74]
RIDGE_FAR = [(-10, 74), (8, 67), (20, 71), (38, 62), (56, 72), (72, 67), (90, 73), (108, 64), (128, 71),
             (148, 63), (168, 70), (190, 66)]
RIDGE_NEAR = [(-10, 80), (14, 75), (30, 79), (52, 73), (70, 79), (96, 76), (120, 80), (140, 74), (164, 78),
              (190, 75)]
FAR_COL, NEAR_COL = hexc("1c1850"), hexc("130f36")
FAR_LIT, NEAR_LIT = hexc("3a4aa8"), hexc("242a78")
GROUND_RAMP = [hexc(c) for c in ("16132e", "1f1b40", "2a2556", "3a3378")]
GROUND_LIT = [hexc(c) for c in ("2c3478", "3a58aa", "5a92da", "a8e4ff")]
MOON = (27, 19)
TARGET_X = 126

_rng = np.random.RandomState(11)
STARS = [(int(_rng.randint(2, W - 2)), int(_rng.randint(2, 58)), int(_rng.randint(0, 3))) for _ in range(54)]
_PATCH = [(int(_rng.randint(-6, W + 6)), int(_rng.randint(HORIZON + 2, H + 4)), int(_rng.randint(0, 2)))
          for _ in range(70)]
_TUFT = [int(_rng.randint(0, W)) for _ in range(22)]


def _ridge(points):
    xs, ys = zip(*points)
    return np.interp(XS, xs, ys)


def _scale(col, k):
    return (int(round(col[0] * k)), int(round(col[1] * k)), int(round(col[2] * (k * 0.9 + 0.1))), 255)


def draw(sc, frame=0, dim=0.0, lit=0.0, flash=False):
    L = "Background"
    k = 1.0 - 0.55 * dim
    far = YS >= np.round(_ridge(RIDGE_FAR) + ((XI * 7) % 5 == 0))
    near = YS >= np.round(_ridge(RIDGE_NEAR) + ((XI * 5) % 3 == 0))
    ground = YS >= HORIZON
    if flash:
        ang = np.degrees(np.arctan2(YS - 99, XS - TARGET_X))
        ray = (np.floor(ang / 9.0).astype(int) * 7 % 5 < 2)
        sc.put(np.ones(SHAPE, bool), "fx_w", L)
        sc.put(ray & (np.hypot(XS - TARGET_X, YS - 99) > 26), "cy_1", L)
        sc.put(far, "cy_2", L)
        sc.put(near, "cy_3", L)
        sc.put(ground, "cy_2", L)
        sc.put(ground & (YS == HORIZON), "fx_w", L)
        sc.put(ground & (np.hypot((XS - TARGET_X) / 3.2, YS - 99) > 20) & ((YI - HORIZON) % 5 == 2), "cy_3", L)
        return

    # --- sky bands, dithered where two bands meet; beam light pushes bands brighter
    v = np.zeros(SHAPE)
    for e in SKY_EDGES:
        v += np.clip((YS - e) / 5.0 + 0.5, 0, 1)
    boost = lit * np.clip(1 - np.abs(XS - TARGET_X) / 62.0, 0, 1) * 3.2
    idx = np.clip(np.floor(v + boost + (BAYER - 0.5) * 0.9 + 0.5).astype(int), 0, len(SKY) - 1)
    for i, col in enumerate(SKY):
        sc.put(idx == i, _scale(col, k), L)

    # --- moon and stars
    moon = disc(MOON, 7.2) & ~disc((MOON[0] + 3.2, MOON[1] - 1.6), 6.4)
    sc.put(moon, _scale(hexc("e6ebff"), 0.75 + 0.25 * k), L)
    sc.put(moon & ~nb(moon, -1, 0) & ~nb(moon, 0, 1), _scale(hexc("a6b0ea"), 0.75 + 0.25 * k), L)
    for i, (x, y, kind) in enumerate(STARS):
        on = (frame + i * 3) % 7 < 5
        if kind == 2 and on:
            sc.put(pixels([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]), hexc("8e98dc"), L)
        sc.put(pixels([(x, y)]), hexc("f2f5ff") if (on and kind) else hexc("8e98dc"), L)

    # --- ridges
    sc.put(far, _scale(FAR_COL, k), L)
    sc.put(near, _scale(NEAR_COL, k), L)
    if lit > 0:
        reach = np.clip(1 - np.abs(XS - TARGET_X) / 80.0, 0, 1) * lit
        sc.put(far & ~nb(far, 0, -1) & (reach > 0.2), FAR_LIT, L)
        sc.put(far & ~nb(far, 0, -2) & (BAYER < reach * 0.9) & (reach > 0.35), FAR_LIT, L)
        sc.put(near & ~nb(near, 0, -1) & (reach > 0.2), NEAR_LIT, L)
        sc.put(near & ~nb(near, 0, -2) & (BAYER < reach * 0.7) & (reach > 0.4), NEAR_LIT, L)

    # --- ground plane
    sc.put(ground, _scale(GROUND_RAMP[1], k), L)
    sc.put(ground & (YS == HORIZON), _scale(GROUND_RAMP[3], k), L)
    sc.put(ground & (YS == HORIZON + 1) & (XI % 3 != 0), _scale(GROUND_RAMP[2], k), L)
    for x, y, kind in _PATCH:
        ln = 2 + (y - HORIZON) // 3
        m = ground & (YS == y) & (XS >= x) & (XS < x + ln)
        sc.put(m, _scale(GROUND_RAMP[0] if kind == 0 else GROUND_RAMP[2], k), L)
    for i, x in enumerate(_TUFT):
        sc.put(pixels([(x, HORIZON - 1)] + ([(x + 1, HORIZON - 2)] if i % 3 == 0 else [])), _scale(NEAR_COL, k), L)
    if lit > 0:
        d = np.hypot((XS - TARGET_X) / 78.0, (YS - 99) / 20.0)
        lv = np.floor(lit * (1 - d) * 5.2 + BAYER * 0.95).astype(int)
        for i, col in enumerate(GROUND_LIT):
            sc.put(ground & (lv == i + 1), col, L)
        sc.put(ground & (lv > len(GROUND_LIT)), GROUND_LIT[-1], L)
