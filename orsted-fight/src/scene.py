"""Frame assembly: tracks -> poses -> character canvases -> world -> (flat, layers)."""
import math
import numpy as np
from px import Canvas, v2
from timeline import finalize
from world import World, GROUND, W, H, LI
from palette import FX
import stage
from common import aura

AURA = {"dragon": (FX["p3"], FX["g2"]), "rage": (FX["o3"], FX["o2"]), "mana": (FX["c3"], FX["c1"])}


class Actor:
    def __init__(self, module, track, layer=None):
        self.m = module
        self.track = track
        self.layer = layer or module.NAME
        w, h, x0, y0 = module.CANVAS
        self.cv = Canvas(w, h, x0, y0)

    def pose(self, f):
        return finalize(self.track.at(f), f)

    def root(self, p):
        return v2((p.get("x", 0.0), GROUND - p.get("lift", 0.0)))


class Scene:
    def __init__(self, actors, events=(), cam=None, post=None):
        self.actors = actors
        self.events = list(events)
        self.cam = cam or (lambda f: (0, 0))
        self.post = post or (lambda f: {})
        bg, own = stage.build()
        self.world = World(bg, own)

    def render(self, f):
        wd = self.world
        wd.begin()
        ctx = {"f": f, "att": {}, "pose": {}, "root": {}, "scene": self}
        poses = []
        for a in self.actors:
            p = a.pose(f)
            if p.get("hide"):
                continue
            pp = a.pose(f - 1) if f > 0 else None
            poses.append((p.get("z", 0), a, p, pp))
        # ground shadows first
        for _, a, p, _ in poses:
            r = a.root(p)
            lift = p.get("lift", 0.0)
            s = max(0.35, 1.0 - lift / 90.0)
            sw = p.get("shadow_w", 11) * s
            sh = wd.cv.ellipse((r[0] + p.get("shadow_dx", 0) * p.get("face", 1), GROUND - 0.5), sw, 2.2 * s)
            wd.put(sh & (wd.cv.checker() | wd.cv.ellipse((r[0], GROUND - 0.5), sw * 0.7, 1.6 * s)), FX["shadow"],
                   "Shadows")
        for ev in self.events:
            if ev.order == "back" and ev.active(f):
                ev.draw(wd, f, ctx)
        for _, a, p, pp in sorted(poses, key=lambda t: t[0]):
            a.cv.clear()
            att = a.m.render(a.cv, p, pp)
            if p.get("aura", 0) > 0:
                cols = AURA[p.get("aura_col", "dragon")]
                aura(a.cv, cols, strength=min(p["aura"], 1.0), phase=f)
            r = a.root(p)
            fc = p.get("face", 1)
            wd.blit(a.cv, r, fc, a.layer, flash=FX["w"] if p.get("flash") else None)
            ctx["att"][a.layer] = {k: wd.to_world(v, r, fc) for k, v in att.items()}
            ctx["pose"][a.layer] = p
            ctx["root"][a.layer] = r
        for ev in self.events:
            if ev.order == "front" and ev.active(f):
                ev.draw(wd, f, ctx)
        post = self.post(f) or {}
        overlay = None
        if post.get("impact"):
            overlay = impact_overlay(wd, post["impact"])
        return wd.finish(shake=self.cam(f), fade=post.get("fade", 0.0), overlay=overlay)


def impact_overlay(wd, kind):
    """Classic impact frame: the world goes white (or black) and every character becomes a solid silhouette."""
    from world import M
    sl = (slice(M, M + H), slice(M, M + W))
    own = wd.cv.owner[sl]
    ov = np.zeros((H, W, 4), np.uint8)
    chars = (own >= LI["Rudeus"]) & (own <= LI["Orsted"])
    fx = own == LI["FX"]
    if kind == "white":
        ov[:] = FX["w"]
        ov[chars] = FX["ink"]
        ov[fx] = FX["c2"]
    else:
        ov[:] = FX["ink"]
        ov[chars] = FX["w"]
        ov[fx] = FX["g2"]
    return ov


class Event:
    """Scheduled effect: draw(world, local_t, ctx) for frames [f0, f1)."""

    def __init__(self, f0, f1, fn, order="front", **kw):
        self.f0, self.f1, self.fn, self.order, self.kw = f0, f1, fn, order, kw

    def active(self, f):
        return self.f0 <= f < self.f1

    def draw(self, wd, f, ctx):
        try:
            self.fn(wd, f - self.f0, ctx, **self.kw)
        except KeyError as e:          # the effect follows an actor that is hidden this frame
            if e.args[0] not in ctx["att"] and e.args[0] in ("Eris", "Rudeus", "Ruijerd", "Orsted"):
                return
            raise
