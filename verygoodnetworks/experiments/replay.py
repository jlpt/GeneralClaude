#!/usr/bin/env python3
"""Record one episode of a trained brain and write it as a self-contained web page.

  replay.py --lib LIB --policy DIR/policy.pt --out replay.html [--steps 3000] [--seed 7]

The page animates the world: the creature (white, ring shows health), purple
(safe) and red (poisonous) food (faded when hidden), springs (blue), predators
(dark red) and big food (gold). It works offline in any browser.
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rl import Bridge, Policy, with_prev  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nworks replay</title>
<style>
:root { --bg:#0f1115; --panel:#181b22; --text:#e6e6e6; --muted:#9aa0aa; --accent:#7aa2f7; }
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) { --bg:#f4f5f7; --panel:#ffffff; --text:#1d1f23; --muted:#5d6470; --accent:#3557c4; } }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
main { max-width:1000px; margin:0 auto; padding:16px; }
h1 { font-size:18px; margin:0 0 4px; } p { margin:4px 0 12px; color:var(--muted); }
canvas { width:100%; height:auto; background:#0b0d10; border-radius:8px; display:block; }
.bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:10px 0; background:var(--panel); padding:8px 10px; border-radius:8px; }
button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:6px 12px; font:inherit; cursor:pointer; }
input[type=range] { flex:1; min-width:120px; } .stat { color:var(--muted); font-variant-numeric:tabular-nums; }
.key span { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 10px; vertical-align:middle; }
</style></head><body><main>
<h1>Nworks replay</h1>
<p>__TITLE__</p>
<canvas id="c" width="1300" height="768"></canvas>
<div class="bar"><button id="play">Pause</button>
<label>Speed <select id="speed"><option>1</option><option selected>4</option><option>10</option><option>25</option></select>x</label>
<input id="seek" type="range" min="0" value="0"><span class="stat" id="info"></span></div>
<div class="key stat"><span style="background:#fff"></span>creature<span style="background:#b48cff"></span>safe food<span style="background:#ff5a5a"></span>poison
<span style="background:#4aa3ff"></span>spring<span style="background:#8b1e2b"></span>predator<span style="background:#f5c542"></span>big food. Faded food is hidden from the creature.</div>
</main><script>
const D = __DATA__;
const cv = document.getElementById('c'), g = cv.getContext('2d');
const seek = document.getElementById('seek'), info = document.getElementById('info');
seek.max = D.frames.length - 1;
let f = 0, playing = true, acc = 0;
const colors = {1:'#b48cff', 2:'#ff5a5a', 3:'#4aa3ff', 4:'#8b1e2b', 5:'#f5c542'};
function draw() {
  const fr = D.frames[f];
  g.fillStyle = '#0b0d10'; g.fillRect(0, 0, cv.width, cv.height);
  for (let i = 0; i < fr.t.length; i++) {
    const k = fr.t[i], x = fr.x[i], y = fr.y[i], e = fr.e[i];
    g.globalAlpha = 1;
    if (k === 1 || k === 2) { g.globalAlpha = e ? 1 : 0.18; g.fillStyle = colors[k]; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); }
    else if (k === 3) { g.fillStyle = colors[3]; g.globalAlpha = 0.3 + 0.7 * e; g.beginPath(); g.arc(x, y, 18, 0, 7); g.fill(); }
    else if (k === 4) { g.fillStyle = colors[4]; g.globalAlpha = e ? 0.5 : 1; g.beginPath(); g.arc(x, y, 14, 0, 7); g.fill(); }
    else if (k === 5) { g.fillStyle = colors[5]; g.beginPath(); g.arc(x, y, 16, 0, 7); g.fill(); }
    else if (k === 0) { g.fillStyle = '#ffffff'; g.beginPath(); g.arc(x, y, 12, 0, 7); g.fill();
      g.strokeStyle = '#3ddc84'; g.lineWidth = 3; g.beginPath(); g.arc(x, y, 16, -1.57, -1.57 + 6.283 * e); g.stroke(); }
  }
  g.globalAlpha = 1;
  info.textContent = `step ${fr.step} · good bites ${fr.good} · poison bites ${fr.poison}` + (fr.dead ? ' · died' : '');
  seek.value = f;
}
function tick(ts) {
  if (playing) { acc += +document.getElementById('speed').value / 2; while (acc >= 1) { acc -= 1; if (f < D.frames.length - 1) f++; else playing = false; } draw(); }
  requestAnimationFrame(tick);
}
document.getElementById('play').onclick = () => { playing = !playing; document.getElementById('play').textContent = playing ? 'Pause' : 'Play'; if (f >= D.frames.length - 1) f = 0; };
seek.oninput = () => { f = +seek.value; draw(); };
draw(); requestAnimationFrame(tick);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--settings", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    bridge = Bridge(args.lib)
    lib = bridge.lib
    lib.nwb_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    ck = torch.load(args.policy)
    policy = Policy(ck["obs_dim"], ck["n_actions"], ck["hidden"], prev=ck.get("prev", False))
    policy.load_state_dict(ck["state"])
    settings = args.settings if args.settings is not None else ck["settings"]
    b = lib.nwb_create(1, args.seed, settings.encode(), args.steps + 1)
    dim = lib.nwb_obs_dim(b)
    obs = np.zeros((1, dim), np.float32)
    lib.nwb_observe(b, Bridge.ptr(obs, ctypes.c_float))
    rew, done, stats = np.zeros(1, np.float32), np.zeros(1, np.uint8), np.zeros(3, np.int32)
    cap = 2000
    buf = np.zeros(cap * 4, np.float32)
    h = torch.zeros(1, 1, ck["hidden"])
    prev_a, prev_r = torch.zeros(1, dtype=torch.long), torch.zeros(1)
    frames, good, poison = [], 0, 0

    def snapshot(step, dead):
        n = lib.nwb_state(b, 0, Bridge.ptr(buf, ctypes.c_float), cap)
        a = buf[:4 * n].reshape(-1, 4)
        frames.append({"step": step, "good": good, "poison": poison, "dead": dead,
                       "t": a[:, 0].astype(int).tolist(), "x": np.round(a[:, 1]).astype(int).tolist(),
                       "y": np.round(a[:, 2]).astype(int).tolist(), "e": np.round(a[:, 3], 2).tolist()})

    snapshot(0, False)
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            logits, _, h = policy.step(with_prev(policy, torch.from_numpy(obs), prev_a, prev_r), h)
            a_t = torch.distributions.Categorical(logits=logits).sample()
            a = a_t.numpy().astype(np.int32) % 18
            prev_a = a_t
        lib.nwb_step(b, Bridge.ptr(a, ctypes.c_int32), Bridge.ptr(obs, ctypes.c_float), Bridge.ptr(rew, ctypes.c_float),
                     Bridge.ptr(done, ctypes.c_uint8), Bridge.ptr(stats, ctypes.c_int32))
        prev_r = torch.from_numpy(rew.copy())
        good += int(stats[0])
        poison += int(stats[1])
        if done[0]:  # the episode restarted inside the batch: stop at the end of this one
            frames[-1]["dead"] = bool(stats[2])
            break
        if step % 2 == 0:
            snapshot(step, False)
    lib.nwb_destroy(b)
    title = args.title or f"A trained brain ({ck['steps']:,} training steps) in one episode. Settings: {settings}."
    page = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps({"frames": frames}, separators=(",", ":")))
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}: {len(frames)} frames, {good} good bites, {poison} poison bites, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
