"""Render every frame and write the .aseprite file plus sprite-sheet and GIF exports."""
import json
import os
import sys
import numpy as np
from PIL import Image

from canvas import W, H, LAYERS, C
from anim import FRAMES, N_IDLE, render
from ase_writer import write_aseprite

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_dir = args[0] if args else "out"
os.makedirs(out_dir, exist_ok=True)
NAME = "starfall_mage"
SHEET_COLS = 8
GIF_SCALE = 4

durs = [d for _, _, d in FRAMES]
layers, flat = [], []
for fi in range(len(FRAMES)):
    sc, shake = render(fi)
    layers.append(sc.layers(shake))
    flat.append(sc.flat(shake))

# palette = every colour actually used: the named ramps first, then the backdrop's shades
used = set()
for img in flat:
    used |= {tuple(int(v) for v in px) for px in np.unique(img.reshape(-1, 4), axis=0) if px[3] > 0}
named = {}
for k, v in C.items():
    named.setdefault(v, k)
palette = [v for v in named if v in used]
palette += sorted(used - set(palette), key=lambda c: (c[0] + c[1] + c[2], c))
palette_names = [named.get(c, "") for c in palette]
assert len(palette) <= 256, len(palette)

TAGS = [
    dict(name="Idle", **{"from": 0, "to": N_IDLE - 1}, direction=0, color=(98, 200, 120)),
    dict(name="Gather", **{"from": N_IDLE, "to": N_IDLE + 5}, direction=0, repeat=1, color=(80, 160, 255)),
    dict(name="Charge", **{"from": N_IDLE + 6, "to": N_IDLE + 20}, direction=0, repeat=1, color=(150, 110, 255)),
    dict(name="Starfall", **{"from": N_IDLE + 21, "to": N_IDLE + 39}, direction=0, repeat=1, color=(255, 200, 80)),
    dict(name="Settle", **{"from": N_IDLE + 40, "to": len(FRAMES) - 1}, direction=0, repeat=1, color=(255, 110, 70)),
]

write_aseprite(
    os.path.join(out_dir, NAME + ".aseprite"), W, H, LAYERS,
    [dict(duration=d, layers=ls) for d, ls in zip(durs, layers)],
    tags=TAGS, palette=palette, palette_names=palette_names,
)

# sprite sheet (1x, 8 columns) + Aseprite-style JSON
rows = (len(flat) + SHEET_COLS - 1) // SHEET_COLS
sheet = Image.new("RGBA", (W * SHEET_COLS, H * rows), (0, 0, 0, 0))
meta = {"frames": [], "meta": {"app": "generated", "image": NAME + "_sheet.png", "format": "RGBA8888",
                                "size": {"w": sheet.width, "h": sheet.height}, "scale": "1",
                                "frameTags": [{"name": t["name"], "from": t["from"], "to": t["to"],
                                               "direction": "forward"} for t in TAGS]}}
for i, (img, d) in enumerate(zip(flat, durs)):
    x, y = (i % SHEET_COLS) * W, (i // SHEET_COLS) * H
    sheet.paste(Image.fromarray(img, "RGBA"), (x, y))
    meta["frames"].append({"filename": f"{NAME} {i}.aseprite", "frame": {"x": x, "y": y, "w": W, "h": H},
                           "rotated": False, "trimmed": False,
                           "spriteSourceSize": {"x": 0, "y": 0, "w": W, "h": H},
                           "sourceSize": {"w": W, "h": H}, "duration": d})
sheet.save(os.path.join(out_dir, NAME + "_sheet.png"))
with open(os.path.join(out_dir, NAME + "_sheet.json"), "w") as f:
    json.dump(meta, f, indent=1)

# preview GIF: the idle loop plays three times before the cast; one shared palette, no dithering
order = list(range(N_IDLE)) * 3 + list(range(N_IDLE, len(flat)))
index = {c[:3]: i for i, c in enumerate(palette)}
pal_bytes = bytes(v for c in palette for v in c[:3]) + bytes(3 * (256 - len(palette)))
gif = []
for i in order:
    rgb = flat[i][:, :, :3]
    packed = (rgb[:, :, 0].astype(np.int32) << 16) | (rgb[:, :, 1].astype(np.int32) << 8) | rgb[:, :, 2]
    lut = {(r << 16) | (g << 8) | b: n for (r, g, b), n in index.items()}
    idx = np.vectorize(lut.__getitem__, otypes=[np.uint8])(packed)
    im = Image.fromarray(idx, "P")
    im.putpalette(pal_bytes)
    gif.append(im.resize((W * GIF_SCALE, H * GIF_SCALE), Image.NEAREST))
gif[0].save(os.path.join(out_dir, NAME + "_preview.gif"), save_all=True, append_images=gif[1:],
            duration=[durs[i] for i in order], loop=0, optimize=False, disposal=1)
if "--frames" in sys.argv:
    fdir = os.path.join(out_dir, "_frames")
    os.makedirs(fdir, exist_ok=True)
    for i, img in enumerate(flat):
        Image.fromarray(img, "RGBA").resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(fdir, f"{i:02d}.png"))
print(f"{len(flat)} frames, {len(palette)} colours, total {sum(durs)} ms")
