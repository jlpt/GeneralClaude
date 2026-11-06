"""Tiny 5x7 bitmap font + outlined text rendering into the Text layer."""
import numpy as np
from px import v2, dilate8, dilate4, nb
from palette import FX

GLYPHS = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#",
    "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.",
    "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####",
    "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.###.",
    "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": "#####|..#..|..#..|..#..|..#..|..#..|#####",
    "J": "..###|...#.|...#.|...#.|#..#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#",
    "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#",
    "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
    "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.",
    "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.",
    "V": "#...#|#...#|#...#|#...#|.#.#.|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.",
    "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####",
    "3": "####.|....#|....#|.###.|....#|....#|####.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.",
    "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": ".###.|#....|#....|####.|#...#|#...#|.###.",
    "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.",
    "9": ".###.|#...#|#...#|.####|....#|....#|.###.",
    "!": "..#..|..#..|..#..|..#..|..#..|.....|..#..",
    ".": ".....|.....|.....|.....|.....|.....|..#..",
    "-": ".....|.....|.....|.###.|.....|.....|.....",
    "*": ".....|..#..|#.#.#|.###.|#.#.#|..#..|.....",
    " ": ".....|.....|.....|.....|.....|.....|.....",
}
_G = {k: np.array([[c == "#" for c in row] for row in v.split("|")], bool) for k, v in GLYPHS.items()}


def text_mask(cv, s, at, scale=1, spacing=1, align="center", reveal=None):
    """Mask of the string's pixels. `at` is the top-centre (or top-left with align='left')."""
    n = len(s) if reveal is None else max(0, min(len(s), int(reveal)))
    adv = (5 + spacing) * scale
    width = len(s) * adv - spacing * scale
    x0 = at[0] - width / 2 if align == "center" else at[0]
    y0 = at[1]
    m = cv.empty()
    for i, ch in enumerate(s[:n]):
        g = _G.get(ch.upper(), _G[" "])
        gx = int(round(x0 + i * adv)) - cv.x0
        gy = int(round(y0)) - cv.y0
        for yy in range(7):
            for xx in range(5):
                if g[yy, xx]:
                    for sy in range(scale):
                        for sx in range(scale):
                            j, k = gx + xx * scale + sx, gy + yy * scale + sy
                            if 0 <= j < cv.w and 0 <= k < cv.h:
                                m[k, j] = True
    return m


def draw_text(wd, s, at, scale=1, fill="g2", top="g1", shade="g4", outline="ink", reveal=None, spacing=1,
              layer="Text", drop=True):
    cv = wd.cv
    m = text_mask(cv, s, at, scale, spacing, reveal=reveal)
    if not m.any():
        return
    o = dilate8(m)
    if drop:
        o |= nb(o, -1, -1)
    wd.put(o, FX[outline], layer)
    wd.put(m, FX[fill], layer)
    # light top half, dark bottom rows of every glyph
    ys = np.nonzero(m)[0]
    y_top = ys.min()
    gh = 7 * scale
    rel = (cv.YS - cv.y0 - y_top) % gh
    wd.put(m & (rel < gh * 0.35), FX[top], layer)
    wd.put(m & (rel >= gh * 0.8), FX[shade], layer)


def text_event(wd, t, ctx, s, at, scale=1, fill="g2", top="g1", shade="g4", cps=None, life=None, blink_out=4,
               spacing=1):
    """Typewriter-revealed text that blinks out over its last few frames."""
    if life is not None and t >= life - blink_out and (t % 2 == 0):
        return
    reveal = None if cps is None else t * cps + 1
    draw_text(wd, s, at, scale, fill, top, shade, reveal=reveal, spacing=spacing)
