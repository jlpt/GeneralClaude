#!/usr/bin/env python3
"""
recolor.py - repaint Banjo-Kazooie models into dream palettes.

Works on the files bk_asset_tool extracted (assets/model/XXXX.model.bin).
For each model in a theme it rewrites, in place and without changing sizes:
  - CI4/CI8 palettes and RGBA16/RGBA32 texels in the model's texture list
  - vertex colours (skipping vertices that are loaded while G_LIGHTING is on,
    because those four bytes are normals, not colours)

usage: recolor.py <pristine assets dir> <output assets dir>
"""
import colorsys, os, struct, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bkmodel import Model

G_VTX, G_CLEARGEOM, G_SETGEOM, G_ENDDL = 0x04, 0xB6, 0xB7, 0xB8
G_LIGHTING = 0x00020000


def make_xf(hue=0.0, sat=1.0, val=1.0, invert=False, gamma=1.0, tint=None, duo=None):
    """Colour transform. hue in degrees. duo=(h1,h2): remap by luminance between two hues."""
    def xf(r, g, b, jitter=0.0):
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        if invert:
            v = 1.0 - v
        if duo is not None:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            t = lum / 255.0
            h = (duo[0] + (duo[1] - duo[0]) * t) / 360.0 + jitter / 360.0
            s = min(1.0, 0.35 + s * 0.65)
        else:
            h = (h + (hue + jitter) / 360.0) % 1.0
        s = min(1.0, s * sat)
        v = min(1.0, (v ** gamma) * val)
        r2, g2, b2 = colorsys.hsv_to_rgb(h % 1.0, s, v)
        if tint is not None:
            r2, g2, b2 = r2 * tint[0], g2 * tint[1], b2 * tint[2]
        return (min(255, int(r2 * 255 + 0.5)), min(255, int(g2 * 255 + 0.5)), min(255, int(b2 * 255 + 0.5)))
    return xf


def xf5551(c, xf, jitter):
    r, g, b, a = (c >> 11) & 31, (c >> 6) & 31, (c >> 1) & 31, c & 1
    r, g, b = xf(r * 255 // 31, g * 255 // 31, b * 255 // 31, jitter)
    return ((r * 31 + 127) // 255 << 11) | ((g * 31 + 127) // 255 << 6) | ((b * 31 + 127) // 255 << 1) | a


def lit_vertices(m):
    """Indices of vertices that are loaded while G_LIGHTING is set."""
    lit = set()
    if not m.gfx_off:
        return lit
    d = m.d
    count = struct.unpack_from(">I", d, m.gfx_off)[0]
    lighting = False
    for i in range(count):
        w0, w1 = struct.unpack_from(">II", d, m.gfx_off + 8 + 8 * i)
        op = w0 >> 24
        if op == G_SETGEOM and (w1 & G_LIGHTING):
            lighting = True
        elif op == G_CLEARGEOM and (w1 & G_LIGHTING):
            lighting = False
        elif op == G_ENDDL:
            lighting = False
        elif op == G_VTX and lighting:
            n = (w0 >> 10) & 0x3F
            first = (w1 & 0xFFFFFF) // 16
            lit.update(range(first, first + n))
    return lit


def recolor_model(data, xf, uid, tex_jitter=0.0, do_vertices=True, do_textures=True):
    m = Model(data)
    d = m.d
    if do_textures and m.textures:
        offs = [t["offset"] for t in m.textures] + [m.tex_end]
        for i, t in enumerate(m.textures):
            start, end = offs[i], offs[i + 1]
            # stable per-texture hue offset so every surface drifts a different way
            seed = zlib.crc32(struct.pack(">HH", uid, i))
            jit = ((seed % 2001) / 1000.0 - 1.0) * tex_jitter
            typ = t["type"]
            if typ == 1:      # CI4: 16 colour RGBA5551 palette
                rng = range(start, min(end, start + 0x20), 2)
            elif typ == 2:    # CI8: 256 colour palette
                rng = range(start, min(end, start + 0x200), 2)
            elif typ == 4:    # RGBA16 texels (+ mip levels)
                rng = range(start, end - 1, 2)
            elif typ == 8:    # RGBA32
                for o in range(start, end - 3, 4):
                    d[o], d[o + 1], d[o + 2] = xf(d[o], d[o + 1], d[o + 2], jit)
                continue
            else:             # intensity formats have no hue
                continue
            for o in rng:
                c = (d[o] << 8) | d[o + 1]
                c = xf5551(c, xf, jit)
                d[o], d[o + 1] = c >> 8, c & 0xFF
    if do_vertices and m.vtx_base:
        lit = lit_vertices(m)
        for i in range(m.vtx_count2):
            if i in lit:
                continue
            o = m.vtx_base + 16 * i + 12
            d[o], d[o + 1], d[o + 2] = xf(d[o], d[o + 1], d[o + 2], 0.0)
    return bytes(d)


# ---------------------------------------------------------------- themes
# (asset ids, transform kwargs, per-texture hue jitter in degrees)
def ids(*ranges):
    out = []
    for r in ranges:
        if isinstance(r, tuple):
            out.extend(range(r[0], r[1] + 1))
        else:
            out.append(r)
    return out

THEMES = []

def theme(name, assets, jitter=0.0, **kw):
    THEMES.append((name, assets, make_xf(**kw), jitter))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    import recolor as registry   # themes.py registers into the imported module, not __main__
    import themes  # noqa: F401
    done = {}
    for name, assets, xf, jitter in registry.THEMES:
        for uid in assets:
            path = os.path.join(src, "model", "%04X.model.bin" % uid)
            if not os.path.exists(path) or uid in done:
                continue
            out = recolor_model(open(path, "rb").read(), xf, uid, jitter)
            open(os.path.join(dst, "model", "%04X.model.bin" % uid), "wb").write(out)
            done[uid] = name
    print("recoloured %d models across %d themes" % (len(done), len(registry.THEMES)))


if __name__ == "__main__":
    main()
