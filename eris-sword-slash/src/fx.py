"""Effects layer: sword smears, sparks, dust, speed lines."""
import numpy as np
from rig import C, XS, YS, H, W, v2, line_px, disc


def smear(sc, pivot, r_in, r_out, a_from, a_to, warm=False, fade=1.0):
    """Crescent smear swept from a_from to a_to (degrees, screen space).
    Thick and bright at the leading edge (the blade), thinning to a dithered wisp."""
    px, py = pivot
    r = np.hypot(XS - px, YS - py)
    a = np.degrees(np.arctan2(YS - py, XS - px))
    span = a_to - a_from
    rel = np.mod((a - a_from) * np.sign(span), 360)
    t = rel / abs(span)                      # 0 at the trailing end, 1 at the leading blade
    inside = t <= 1.0
    max_thick = (r_out - r_in) * 0.62 * fade
    thick = max_thick * np.clip(t, 0, 1) ** 1.8
    band = inside & (r <= r_out + 0.5) & (r >= r_out - thick - 0.6)
    core = band & (t > 0.8) & (r > r_out - thick * 0.75)
    mid = band & (t > 0.45) & ~core
    tail = band & (t <= 0.45) & (t > 0.04)
    tail &= ((XS + YS) % 2 == 0) | (r > r_out - 1.0)
    edge = band & (r > r_out - 1.0) & (t > 0.15)
    sc.put(tail, C["fx_3"] if warm else C["fx_2"], "FX", solid=False)
    sc.put(mid, C["fx_2"] if fade < 0.6 else C["fx_1"], "FX", solid=False)
    sc.put(core, C["fx_w"], "FX", solid=False)
    if warm:
        sc.put(edge & (t <= 0.8), C["fx_4"], "FX", solid=False)
        sc.put(edge & (t <= 0.45), C["fx_3"], "FX", solid=False)
    else:
        sc.put(edge & ~core & (t > 0.35), C["fx_w"], "FX", solid=False)


def spark(sc, c, size, color="fx_w"):
    x, y = int(round(c[0])), int(round(c[1]))
    m = np.zeros((H, W), bool)
    for i in range(-size, size + 1):
        for (xx, yy) in ((x + i, y), (x, y + i)):
            if 0 <= xx < W and 0 <= yy < H:
                m[yy, xx] = True
    sc.put(m, C[color], "FX", solid=False)
    if size >= 2:
        core = np.zeros((H, W), bool)
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            if 0 <= x + dx < W and 0 <= y + dy < H:
                core[y + dy, x + dx] = True
        sc.put(core, C["fx_2"], "FX", solid=False)


def dust(sc, c, r, phase=0):
    m = disc(c, r) & ~disc(v2(c) + v2((0.8, 1.2)), r * 0.7)
    full = disc(c, r)
    sc.put(full & ((XS + YS + phase) % 2 == 0) & ~m, C["dust_d"], "FX", solid=False)
    sc.put(m, C["dust_l"], "FX", solid=False)


def speed_line(sc, p0, p1, color="fx_1"):
    sc.put(line_px(p0, p1), C[color], "FX", solid=False)


def draw(sc, f):
    kind = f[0]
    if kind == "smear":
        smear(sc, *f[1:])
    elif kind == "spark":
        spark(sc, *f[1:])
    elif kind == "dust":
        dust(sc, *f[1:])
    elif kind == "line":
        speed_line(sc, *f[1:])
