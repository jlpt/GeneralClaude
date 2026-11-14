"""An episode = a hidden target, a few input photos, and held-out test cameras."""
from dataclasses import dataclass, field

import numpy as np

from .render import Camera, render
from .reward import scene_score
from .scenes import random_scene, validate

INPUT_CAMERAS = (Camera(30, 25), Camera(150, 25), Camera(270, 25))


@dataclass
class Episode:
    target: list
    input_cams: tuple
    heldout_cams: tuple
    res: int = 64
    _cache: dict = field(default_factory=dict, repr=False)

    def _renders(self, which):
        if which not in self._cache:
            cams = self.input_cams if which == "input" else self.heldout_cams
            self._cache[which] = [render(self.target, c, self.res) for c in cams]
        return self._cache[which]

    def input_images(self):
        """What the policy is allowed to see."""
        return [r[0] for r in self._renders("input")]

    def score(self, scene):
        """Returns {'input': ..., 'heldout': ...}. Training reward should use 'heldout'."""
        try:
            scene = validate(scene)
        except (ValueError, KeyError, TypeError):
            return {"input": 0.0, "heldout": 0.0, "valid": False}
        return {
            "input": scene_score(scene, self._renders("input"), self.input_cams, self.res),
            "heldout": scene_score(scene, self._renders("heldout"), self.heldout_cams, self.res),
            "valid": True,
        }


def random_heldout_cams(rng, n):
    """Fresh angles each episode, including high and low ones the inputs never cover."""
    return tuple(Camera(float(rng.uniform(0, 360)), float(rng.uniform(5, 75))) for _ in range(n))


def make_episode(seed, level=1, n_input=3, n_heldout=6, res=64):
    rng = np.random.default_rng(seed)
    return Episode(target=random_scene(rng, level),
                   input_cams=INPUT_CAMERAS[:n_input],
                   heldout_cams=random_heldout_cams(rng, n_heldout),
                   res=res)
