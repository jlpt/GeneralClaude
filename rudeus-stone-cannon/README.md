# Rudeus: Stone Cannon (pixel art)

![preview](rudeus_stone_cannon_preview.gif)

Rudeus stands in a moonlit field, raises both hands above his head and builds a Stone Cannon. Rocks tear out of
the ground and spiral into a gravel ball, which packs into a spinning, rifled bullet. He fires it through a row of
barrel circles, and it blows a hole through a rock spire.

- **`rudeus_stone_cannon.aseprite`**: open it directly in Aseprite. The canvas is 320×180 RGB(A) with 124 frames and per-frame timings (7.46 s).
  - Layers, bottom to top: `Sky`, `Land`, `Crag`, `Back FX`, `Back arm`, `Rudeus`, `Front arm`, `Magic`, `FX`
  - Tags: `Idle` (0–11), `Raise Arms` (12–19), `Gather` (20–49), `Form Bullet` (50–67), `Impact Frame` (68–69),
    `Fire` (70–72), `Impact` (73–99), `Recover` (100–123)
  - The palette panel holds all 93 colours, named by ramp (`mg0`–`mg7` earth magic, `st1`–`st5` bullet, `rk0`–`rk6` rock, …).
- `rudeus_stone_cannon_preview.gif`: a 2× looping preview.
- `rudeus_stone_cannon.mp4`: a 4× (1280×720) video with the same frame timings.
- `rudeus_stone_cannon_sheet.png` + `.json`: a 1× sprite sheet (12 columns) with Aseprite-style JSON (frame rects, durations, tags).

## What happens

| Frames | Beat | Effects |
|---|---|---|
| 0–11 | Idle | Twinkling stars, drifting clouds, swaying grass, fade in |
| 12–19 | Raise arms | Arms swing up from the sides; the ground circle draws itself in |
| 20–49 | Gather | Overhead circle, light motes spiralling in, 20 rocks lifting off and orbiting into a gravel ball, glowing ground cracks, rising light, collapsing energy rings, flame aura, the scene darkening, a low rumble |
| 50–67 | Form bullet | The gravel packs into a pointed bullet that spins (moving rifling grooves and spin arcs) with orbiting sparks, crackling arcs and a heating tip; three barrel circles line up along the shot |
| 68–69 | Impact frame | Two stark silhouette frames with spiral speed rays (cream/black, then inverted) |
| 70–72 | Fire | Muzzle flash, an expanding sonic ring, a tapered energy beam, a helix trail, vapour cone rings, speed lines, barrel circles shattering, recoil (lean back, robe whipped back, dust kicked off his feet, grass flattened), heavy shake |
| 73–99 | Impact | White flash, double shockwave ring, 56 ballistic rock chunks (some red-hot with trails), exit spray out the far side, a billowing dust cloud with a fire core that dissolves away, sparks, radiating cracks, a molten hole cooling from white to dark, rising embers |
| 100–123 | Recover | He lowers his arms, the dust settles, fade out to loop |

Each pixel belongs to exactly one layer, so the layer stack composites to the final frame. Light from the spell is
baked into whichever layer it falls on. The two impact frames are recoloured versions of the normal layers.

## Regenerating

Everything is drawn by code. The head is hand-pixelled; the robe, legs and IK arms come from a small rig. The
backdrop, crag and every effect are procedural and snap to the fixed palette, with ordered dithering for light and
gradients. Each layer is quantised against its own sub-palette.

```
cd src && python3 build.py .. --mp4     # needs numpy + Pillow (and ffmpeg for --mp4)
```

- `canvas.py`: palette, scene buffer (with a margin for screen shake), primitives, quantiser
- `rudeus.py`: head sprites (`calm`, `focus`, `shout`), robe/legs, arms
- `world.py`: sky, moon, mountains, plain, grass, crag (Voronoi stone blocks, hole and cracks after the hit)
- `fx.py`: magic circles, motes, rocks, gravel ball, bullet, launch and impact effects
- `anim.py`: the timeline (poses, effect schedule, shake, darkness, frame durations)
- `build.py`: renders all frames and writes the exports (`ase_writer.py` is the native .aseprite writer)
