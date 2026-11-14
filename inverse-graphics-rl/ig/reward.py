"""Render-and-compare reward."""
import numpy as np

from .render import render


def view_score(pred, target, depth_scale=0.15):
    """Score in [0, 1]. Silhouette IoU gates everything; depth and color refine it.

    pred/target are (rgb, mask, depth) tuples from render().
    """
    (prgb, pm, pd), (trgb, tm, td) = pred, target
    union = (pm | tm).sum()
    if union == 0:
        return 1.0, {"iou": 1.0, "depth": 1.0, "color": 1.0}
    inter = pm & tm
    iou = inter.sum() / union
    if inter.sum() == 0:
        return 0.0, {"iou": 0.0, "depth": 0.0, "color": 0.0}
    depth_acc = float(np.exp(-np.abs(pd[inter] - td[inter]).mean() / depth_scale))
    color_acc = float(np.clip(1 - np.abs(prgb[inter] - trgb[inter]).mean() * 2, 0, 1))
    score = float(iou * (0.5 + 0.3 * depth_acc + 0.2 * color_acc))
    return score, {"iou": float(iou), "depth": depth_acc, "color": color_acc}


def scene_score(pred_scene, target_renders, cameras, res):
    """Mean view_score of pred_scene over cameras, given precomputed target renders."""
    scores = [view_score(render(pred_scene, cam, res), tgt)[0]
              for cam, tgt in zip(cameras, target_renders)]
    return float(np.mean(scores))
