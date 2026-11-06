"""Render the whole fight and write the MP4, preview GIF, .aseprite source and storyboard sheet.

    cd src && python3 build.py ..        # needs numpy, Pillow and ffmpeg on PATH
"""
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image

import story
from world import W, H, LAYERS
from ase_writer import write_aseprite
from font import text_mask
from px import Canvas

NAME = "orsted_fight"
FPS = story.FPS


def frame_ms(i):
    return round(i * 1000 / FPS)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "out"
    os.makedirs(out, exist_ok=True)
    scene = story.build()
    n = story.LAST + 1

    t0 = time.time()
    flats, layer_sets = [], []
    for f in range(n):
        flat, layers = scene.render(f)
        flats.append(flat)
        layer_sets.append(layers)
    print(f"rendered {n} frames in {time.time() - t0:.1f}s")

    # ---- merge runs of identical frames (holds) into single longer frames
    uniq, starts = [], []
    for i in range(n):
        if uniq and np.array_equal(flats[i], flats[uniq[-1]]):
            continue
        uniq.append(i)
        starts.append(i)
    ends = starts[1:] + [n]
    durs = [frame_ms(e) - frame_ms(s) for s, e in zip(starts, ends)]
    print(f"{len(uniq)} unique frames, {sum(durs) / 1000:.2f}s")

    # ---- MP4: 4x nearest-neighbour, every source frame (constant 15 fps), duplicated to 30 fps
    mp4 = os.path.join(out, NAME + ".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-vf", "scale=iw*4:ih*4:flags=neighbor,fps=30", "-c:v", "libx264",
           "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in flats:
        p.stdin.write(np.ascontiguousarray(f[:, :, :3]).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"

    # ---- palette shared by the GIF and the .aseprite file
    allpx = np.concatenate([flats[i].reshape(-1, 4) for i in uniq])
    cols = np.unique(allpx[allpx[:, 3] > 0][:, :3], axis=0)
    print(f"{len(cols)} colours")

    # ---- preview GIF (2x): global palette (the piece uses ~200 colours) and transparent-diff frames
    gif = os.path.join(out, NAME + "_preview.gif")
    vf = ("scale=iw*2:ih*2:flags=neighbor,split[a][b];[a]palettegen=max_colors=256:stats_mode=full:"
          "reserve_transparent=1[p];[b][p]paletteuse=dither=none:diff_mode=rectangle")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-vf", vf, "-loop", "0", gif]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in flats:
        p.stdin.write(np.ascontiguousarray(f[:, :, :3]).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg (gif) failed"

    # ---- .aseprite source: unique frames, one layer per character, a tag per beat
    tag_cols = [(98, 200, 120), (255, 110, 70), (80, 160, 255), (120, 200, 255), (200, 150, 60), (180, 120, 255),
                (255, 80, 80), (200, 200, 210)]
    tags = []
    for (name, a, b), c in zip(story.BEATS, tag_cols):
        ua = next(j for j, s in enumerate(starts) if ends[j] > a)
        ub = max(j for j, s in enumerate(starts) if s <= b)
        tags.append(dict(name=name, **{"from": ua, "to": ub}, direction=0, color=c))
    palette = [(int(r), int(g), int(b), 255) for r, g, b in cols]
    write_aseprite(os.path.join(out, NAME + ".aseprite"), W, H, LAYERS,
                   [dict(duration=d, layers=layer_sets[i]) for i, d in zip(uniq, durs)],
                   tags=tags, palette=palette if len(palette) <= 256 else palette[:256])

    # ---- storyboard: one key frame per beat, labelled
    keys = [34, 59, 108, 160, 205, 250, 297, 344]
    cw, ch = W, H
    sheet = np.zeros((ch * 2 + 4, cw * 4 + 12, 3), np.uint8)
    sheet[:] = (12, 10, 18)
    for k, (f, (name, a, b)) in enumerate(zip(keys, story.BEATS)):
        im = flats[f][:, :, :3].copy()
        cv = Canvas(W, H)
        label = f"{k + 1}. {name.upper()}"
        m = text_mask(cv, label, (6, 6), 1, align="left")
        from px import dilate8
        im[dilate8(m)] = (12, 10, 18)
        im[m] = (255, 212, 90)
        x, y = (k % 4) * (cw + 4), (k // 4) * (ch + 4)
        sheet[y:y + ch, x:x + cw] = im
    Image.fromarray(sheet).resize((sheet.shape[1], sheet.shape[0]), Image.NEAREST).save(
        os.path.join(out, "storyboard.png"))

    if "--frames" in sys.argv:
        fd = os.path.join(out, "frames")
        os.makedirs(fd, exist_ok=True)
        for i in uniq:
            Image.fromarray(flats[i]).save(os.path.join(fd, f"{i:04d}.png"))
    if "--dump" in sys.argv:
        np.save(os.path.join(out, "_flat_frames.npy"), np.stack([flats[i] for i in uniq]))
    print("done")


if __name__ == "__main__":
    main()
