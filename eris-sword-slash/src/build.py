"""Render every frame and write the .aseprite file plus GIF / sprite-sheet exports."""
import json
import os
import sys
import numpy as np
from PIL import Image

from rig import W, H, LAYERS, C
from anim import FRAMES, IDLE_FRAMES, render_pose
from ase_writer import write_aseprite

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_dir = args[0] if args else "out"
os.makedirs(out_dir, exist_ok=True)
NAME = "eris_sword_slash"

scenes = [render_pose(f) for f in FRAMES]
durs = [f["dur"] for f in FRAMES]
flat = [s.flat() for s in scenes]

# palette = every colour actually used, ordered like the named ramps
used = set()
for img in flat:
    a = img.reshape(-1, 4)
    used |= {tuple(int(c) for c in px) for px in a[a[:, 3] > 0]}
named = {v: k for k, v in C.items()}
palette = [v for v in C.values() if v in used]
palette += sorted(used - set(palette))
palette_names = [named.get(c, "") for c in palette]
assert len(palette) <= 256, len(palette)

TAGS = [
    dict(name="Idle", **{"from": 0, "to": IDLE_FRAMES - 1}, direction=0, color=(98, 200, 120)),
    dict(name="Draw Slash", **{"from": IDLE_FRAMES, "to": 5}, direction=0, repeat=1, color=(80, 160, 255)),
    dict(name="Heavy Slash", **{"from": 6, "to": 9}, direction=0, repeat=1, color=(255, 110, 70)),
    dict(name="Shoulder Pose + Sheathe", **{"from": 10, "to": len(FRAMES) - 1}, direction=0, repeat=1,
         color=(240, 200, 80)),
]

write_aseprite(
    os.path.join(out_dir, NAME + ".aseprite"), W, H, LAYERS,
    [dict(duration=d, layers=s.layers()) for d, s in zip(durs, scenes)],
    tags=TAGS, palette=palette, palette_names=palette_names,
)

# sprite sheet (1x, one row) + Aseprite-style JSON
sheet = Image.new("RGBA", (W * len(flat), H), (0, 0, 0, 0))
meta = {"frames": [], "meta": {"app": "generated", "image": NAME + "_sheet.png", "format": "RGBA8888",
                                "size": {"w": sheet.width, "h": H}, "scale": "1",
                                "frameTags": [{"name": t["name"], "from": t["from"], "to": t["to"],
                                               "direction": "forward"} for t in TAGS]}}
for i, (img, d) in enumerate(zip(flat, durs)):
    sheet.paste(Image.fromarray(img, "RGBA"), (i * W, 0))
    meta["frames"].append({"filename": f"{NAME} {i}.aseprite", "frame": {"x": i * W, "y": 0, "w": W, "h": H},
                           "rotated": False, "trimmed": False,
                           "spriteSourceSize": {"x": 0, "y": 0, "w": W, "h": H},
                           "sourceSize": {"w": W, "h": H}, "duration": d})
sheet.save(os.path.join(out_dir, NAME + "_sheet.png"))
with open(os.path.join(out_dir, NAME + "_sheet.json"), "w") as f:
    json.dump(meta, f, indent=1)

# preview GIF: idle plays twice before the attack, upscaled 4x on a dark backdrop
order = list(range(IDLE_FRAMES)) * 2 + list(range(IDLE_FRAMES, len(flat)))
bg = (38, 40, 52, 255)
gif = []
for i in order:
    base = Image.new("RGBA", (W, H), bg)
    base.alpha_composite(Image.fromarray(flat[i], "RGBA"))
    gif.append(base.convert("RGB").resize((W * 4, H * 4), Image.NEAREST))
gif[0].save(os.path.join(out_dir, NAME + "_preview.gif"), save_all=True, append_images=gif[1:],
            duration=[durs[i] for i in order], loop=0, optimize=False)
if "--dump" in sys.argv:
    np.save(os.path.join(out_dir, "_flat_frames.npy"), np.stack(flat))
print(f"{len(flat)} frames, {len(palette)} colours, total {sum(durs)} ms")
