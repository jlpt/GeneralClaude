"""World canvas: background cache, character compositing, layers, camera shake and post effects."""
import numpy as np
from px import Canvas, v2
from common import FXP
from palette import FX

W, H = 480, 270
M = 10                     # margin around the screen so camera shake never reveals an edge
GROUND = 244               # y of the ground under the characters' feet

LAYERS = ["Sky", "Mountains", "Cliffs", "Ground", "Shadows", "FX (back)", "Rudeus", "Ruijerd", "Eris",
          "Orsted", "FX", "Text", "Overlay"]
LI = {n: i for i, n in enumerate(LAYERS)}
N_STATIC = 4               # Sky..Ground are written whole (they sit under everything)


class World:
    def __init__(self, bg_rgba, bg_owner):
        self.cv = Canvas(W + 2 * M, H + 2 * M, -M, -M)
        self.bg_rgba = bg_rgba
        self.bg_owner = bg_owner

    def begin(self):
        self.cv.rgba[:] = self.bg_rgba
        self.cv.owner[:] = self.bg_owner

    def to_world(self, local_pt, root, facing):
        return v2((round(root[0]) + local_pt[0] * facing, round(root[1]) + local_pt[1]))

    def blit(self, local, root, facing, layer, flash=None):
        img, own = local.rgba, local.owner
        rx, ry = int(round(root[0])), int(round(root[1]))
        if facing < 0:
            img = img[:, ::-1]
            own = own[:, ::-1]
            left = rx - (local.x0 + local.w - 1)
        else:
            left = rx + local.x0
        top = ry + local.y0
        cv = self.cv
        j0, i0 = left - cv.x0, top - cv.y0
        a0, b0 = max(0, -i0), max(0, -j0)
        a1, b1 = min(local.h, cv.h - i0), min(local.w, cv.w - j0)
        if a0 >= a1 or b0 >= b1:
            return
        src = img[a0:a1, b0:b1]
        so = own[a0:a1, b0:b1]
        m = src[:, :, 3] > 0
        dst = cv.rgba[i0 + a0:i0 + a1, j0 + b0:j0 + b1]
        do = cv.owner[i0 + a0:i0 + a1, j0 + b0:j0 + b1]
        src = src.copy()
        if flash is not None:
            src[m & (so != FXP)] = flash
        dst[m] = src[m]
        lay = np.where(so == FXP, LI["FX"], LI[layer])
        do[m] = lay[m]

    def put(self, mask, color, layer):
        self.cv.put(mask, color, LI[layer])

    def finish(self, shake=(0, 0), fade=0.0, flash=None, overlay=None):
        """Crop to the screen. Returns (flat RGBA, list of per-layer RGBA images)."""
        dx, dy = int(shake[0]), int(shake[1])
        sl = (slice(M + dy, M + dy + H), slice(M + dx, M + dx + W))
        rgba = self.cv.rgba[sl].copy()
        owner = self.cv.owner[sl].copy()
        bg = self.bg_rgba[sl]
        bgo = self.bg_owner[sl]
        layers = []
        for i in range(len(LAYERS)):
            img = np.zeros((H, W, 4), np.uint8)
            if i < N_STATIC:
                mm = bgo == i
                img[mm] = bg[mm]
            elif i != LI["Overlay"]:
                mm = owner == i
                img[mm] = rgba[mm]
            layers.append(img)
        ov = np.zeros((H, W, 4), np.uint8)
        if overlay is not None:
            mm = overlay[:, :, 3] > 0
            ov[mm] = overlay[mm]
        if fade > 0:
            scr = Canvas(W, H)
            fm = scr.bayer(fade) & (owner != LI["Text"])     # titles stay readable over the fade
            ov[fm] = FX["black"]
        layers[LI["Overlay"]] = ov
        flat = rgba
        mm = ov[:, :, 3] > 0
        flat[mm] = ov[mm]
        return flat, layers
