"""Why held-out views matter: a flat 'billboard' beats an honest guess if you only score the input view.

Target: a red sphere. The policy sees ONE photo, straight from the front.
"""
import os

import numpy as np

from ig.episode import Episode, random_heldout_cams
from ig.render import Camera, render, to_png

RED = [0.85, 0.25, 0.2]
target = [{"type": "sphere", "center": [0, 0.4, 0], "radius": 0.4, "color": RED}]

candidates = {
    "exact answer (sphere r=0.40)":
        target,
    "honest but rough (sphere r=0.33)":
        [{"type": "sphere", "center": [0, 0.4, 0], "radius": 0.33, "color": RED}],
    "honest but rough (cube 0.8)":
        [{"type": "box", "center": [0, 0.4, 0], "size": [0.8, 0.8, 0.8], "color": RED}],
    "CHEAT: flat disc facing camera":
        [{"type": "cylinder", "center": [0, 0.4, 0.3], "radius": 0.37, "height": 0.02,
          "axis": "z", "color": RED}],
}

ep = Episode(target=target, input_cams=(Camera(0, 0),),
             heldout_cams=random_heldout_cams(np.random.default_rng(0), 8))

print(f"{'candidate':36s} {'score on input view':>20s} {'score on held-out views':>24s}")
for name, scene in candidates.items():
    s = ep.score(scene)
    print(f"{name:36s} {s['input']:20.3f} {s['heldout']:24.3f}")

os.makedirs("out", exist_ok=True)
cams = [Camera(0, 0), Camera(90, 10), Camera(45, 60)]
rows = [np.concatenate([render(sc, c, 96)[0] for c in cams], axis=1)
        for sc in [target, *candidates.values()][1:]]
to_png(np.concatenate(rows, axis=0), "out/cheat_demo.png", scale=2)
print("\nWrote out/cheat_demo.png: rows = candidates (same order), "
      "columns = input view, side view, high view")
