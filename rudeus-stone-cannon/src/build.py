"""Render every frame and write the .aseprite file plus GIF / MP4 / sprite-sheet exports."""
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image

from canvas import W, H, LAYERS, P, hexc
from anim import N, DURATIONS, render
from ase_writer import write_aseprite

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_dir = args[0] if args else "out"
os.makedirs(out_dir, exist_ok=True)
NAME = "rudeus_stone_cannon"

frames, owners = [], []
for f in range(N):
    img, own = render(f)
    frames.append(img)
    owners.append(own)
    print(f"\rframe {f + 1}/{N}", end="", flush=True)
print()

# ---------------------------------------------------------------- palette (named ramps first)
used = set()
for img in frames:
    used |= {tuple(int(c) for c in px) for px in np.unique(img.reshape(-1, 3), axis=0)}
named = {hexc(v): k for k, v in P.items()}
palette = [hexc(v) for v in P.values() if hexc(v) in used]
palette += sorted(used - set(palette))
assert len(palette) <= 256, len(palette)
palette_rgba = [(r, g, b, 255) for r, g, b in palette]
palette_names = [named.get(c, "") for c in palette]

# ---------------------------------------------------------------- .aseprite
TAGS = [
    dict(name="Idle", **{"from": 0, "to": 11}, color=(98, 200, 120)),
    dict(name="Raise Arms", **{"from": 12, "to": 19}, color=(120, 170, 255)),
    dict(name="Gather", **{"from": 20, "to": 49}, color=(255, 170, 60)),
    dict(name="Form Bullet", **{"from": 50, "to": 67}, color=(255, 120, 40)),
    dict(name="Impact Frame", **{"from": 68, "to": 69}, color=(240, 240, 240)),
    dict(name="Fire", **{"from": 70, "to": 72}, color=(255, 230, 120)),
    dict(name="Impact", **{"from": 73, "to": 99}, color=(220, 80, 60)),
    dict(name="Recover", **{"from": 100, "to": N - 1}, color=(140, 140, 200)),
]


def split_layers(img, own):
    out = []
    for i in range(len(LAYERS)):
        lay = np.zeros((H, W, 4), np.uint8)
        m = own == i
        lay[m, :3] = img[m]
        lay[m, 3] = 255
        out.append(lay)
    return out


write_aseprite(
    os.path.join(out_dir, NAME + ".aseprite"), W, H, LAYERS,
    [dict(duration=d, layers=split_layers(img, own)) for d, img, own in zip(DURATIONS, frames, owners)],
    tags=TAGS, palette=palette_rgba, palette_names=palette_names,
)

# ---------------------------------------------------------------- sprite sheet (grid) + JSON
COLS = 12
rows = (N + COLS - 1) // COLS
sheet = Image.new("RGB", (W * COLS, H * rows), (0, 0, 0))
meta = {"frames": [], "meta": {"app": "generated", "image": NAME + "_sheet.png", "format": "RGB888",
                                "size": {"w": sheet.width, "h": sheet.height}, "scale": "1",
                                "frameTags": [{"name": t["name"], "from": t["from"], "to": t["to"],
                                               "direction": "forward"} for t in TAGS]}}
for i, (img, d) in enumerate(zip(frames, DURATIONS)):
    x, y = (i % COLS) * W, (i // COLS) * H
    sheet.paste(Image.fromarray(img), (x, y))
    meta["frames"].append({"filename": f"{NAME} {i}.aseprite", "frame": {"x": x, "y": y, "w": W, "h": H},
                           "rotated": False, "trimmed": False,
                           "spriteSourceSize": {"x": 0, "y": 0, "w": W, "h": H},
                           "sourceSize": {"w": W, "h": H}, "duration": d})
sheet.save(os.path.join(out_dir, NAME + "_sheet.png"), optimize=True)
with open(os.path.join(out_dir, NAME + "_sheet.json"), "w") as fh:
    json.dump(meta, fh, indent=1)

# ---------------------------------------------------------------- preview GIF (shared palette, 2x)
pal_img = Image.new("P", (1, 1))
flat_pal = [c for rgb in palette for c in rgb]
pal_img.putpalette(flat_pal + [0] * (768 - len(flat_pal)))
SCALE = 2
gif = []
for img in frames:
    im = Image.fromarray(img).resize((W * SCALE, H * SCALE), Image.NEAREST)
    gif.append(im.quantize(palette=pal_img, dither=Image.Dither.NONE))
gif[0].save(os.path.join(out_dir, NAME + "_preview.gif"), save_all=True, append_images=gif[1:],
            duration=DURATIONS, loop=0, optimize=False, disposal=1)

# ---------------------------------------------------------------- MP4 (4x, exact frame timing via concat list)
if "--mp4" in sys.argv:
    tmp = os.path.join(out_dir, "_mp4")
    os.makedirs(tmp, exist_ok=True)
    lst = []
    for i, (img, d) in enumerate(zip(frames, DURATIONS)):
        p = os.path.join(tmp, f"{i:04d}.png")
        Image.fromarray(img).resize((W * 4, H * 4), Image.NEAREST).save(p)
        lst.append(f"file '{os.path.abspath(p)}'\nduration {d / 1000:.3f}")
    lst.append(f"file '{os.path.abspath(os.path.join(tmp, f'{N - 1:04d}.png'))}'")
    with open(os.path.join(tmp, "list.txt"), "w") as fh:
        fh.write("\n".join(lst) + "\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", os.path.join(tmp, "list.txt"), "-vf", "fps=50", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    os.path.join(out_dir, NAME + ".mp4")], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)

print(f"{N} frames, {len(palette)} colours, total {sum(DURATIONS)} ms")
