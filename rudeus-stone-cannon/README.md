# Rudeus: Stone Cannon (pixel art)

![preview](rudeus_stone_cannon_preview.gif)

Rudeus stands in a moonlit field, raises both hands above his head and builds a Stone Cannon. Rocks tear out of
the ground into a gravel ball, which packs into a rifled stone bullet. The bullet spins up until it screams like a
drill, gets fired through a row of barrel circles, bores into a rock spire and blows it apart. Sukuna was inside
the rock. He has a message from Orsted, and then he opens his domain.

- **`rudeus_stone_cannon.aseprite`**: open it directly in Aseprite. The canvas is 320×180 RGB(A) with 317 frames and per-frame timings (19.2 s).
  - Layers, bottom to top: `Sky`, `Land`, `Crag`, `Domain`, `Back FX`, `Back arm`, `Rudeus`, `Front arm`, `Sukuna`,
    `Magic`, `FX`, `Text`
  - There is one tag per beat (see the table below). The palette panel holds all 121 colours, named by ramp
    (`mg` earth magic, `st` bullet, `rk` rock, `ph`/`ki` Sukuna, `dm`/`bn` the domain, …).
- `rudeus_stone_cannon_preview.gif`: a 2× looping preview.
- `rudeus_stone_cannon.mp4`: a 4× (1280×720) video with the same frame timings.
- `rudeus_stone_cannon_sheet.png` + `.json`: a 1× sprite sheet (12 columns) with Aseprite-style JSON (frame rects, durations, tags).

## What happens

| Frames | Tag | What you see |
|---|---|---|
| 0–11 | Idle | Moonlit field, twinkling stars, swaying grass, fade in |
| 12–19 | Raise Arms | Arms swing up over his head; the ground circle draws itself in |
| 20–49 | Gather | Overhead circle, light motes spiralling in, 20 rocks lifting and orbiting into a gravel ball, glowing ground cracks, flame aura, the scene darkens and rumbles |
| 50–57 | Form Bullet | The gravel packs into a pointed bullet |
| 58–83 | Spin Up | It starts turning. Three helical flutes crawl backwards along the body like a drill bit, a rune band spins near the base, then spiral air streams wrap it, rings peel off, grit and sparks fly off sideways and the tip heats up |
| 84–97 | Close-up: Spin | Letterboxed close-up: the giant bullet screwing round inside a spinning accelerator circle over a swirling background, before blurring into pure speed |
| 98–107 | Lock On | Three barrel circles line up; max spin; the shake builds |
| 108–109 | Impact Frame | Two stark silhouette frames with spiral rays |
| 110–111 | Fire | Muzzle flash, sonic ring, energy beam, helix trail, barrel circles shattering, recoil |
| 112–126 | Tracking Shot / Arrive | Slow-motion follow shot: the spinning bullet with a long spiral vortex, vapour rings peeling away, the ground streaking past, until the rock face rushes in |
| 127–136 | Drill | It bores into the spire, flinging a spinning wheel of sparks and rock chips; the spire shudders and cracks |
| 137–152 | Breakthrough | Flash, shockwaves, debris, a dust cloud with a fire core, and the bullet bursting out the far side |
| 153–168 | Collapse | Cracks spread and the spire breaks into its stone blocks, which fall into a rubble heap |
| 169–190 | Reveal | Four red eyes burn through the dust, then Sukuna steps out of the silhouette, standing on the rubble |
| 191–230 | Close-up: Sukuna | Letterboxed face close-up. A typed dialogue box says "ORSTED SENDS HIS REGARDS." His grin widens and the lower eyes snap open |
| 231–244 | Hand Sign | He forms the seal, a red aura flares, red fissures run out under him, and the DOMAIN EXPANSION title card appears |
| 245–256 | Domain Expansion | The domain swallows the field in an expanding circle |
| 257–274 | Malevolent Shrine | A blood-red sky and moon over a mirror pool. The shrine (skull mound, fanged mouth, horned roofs) rises from the pool with its reflection. Rudeus shoves up an earth wall |
| 275–306 | Dismantle | Invisible slashes fill the screen and huge crescent cleaves land. Rudeus's wall stops them, but it gets diced into pieces that slide apart and fall |
| 307–316 | Finale | A giant X cleave, a red impact frame, then fade to black with only Sukuna's four eyes left |

Each pixel belongs to exactly one layer, so the layer stack composites to the final frame. Light is baked into
whichever layer it falls on.

## Regenerating

Everything is drawn by code: hand-pixelled heads on small rigs, with procedural backdrops and effects. Every pixel
snaps to the fixed palette, with ordered dithering for light and gradients.

```
cd src && python3 build.py .. --mp4     # needs numpy + Pillow (and ffmpeg for --mp4)
```

- `anim.py`: the timeline. `SEGMENTS` lists each beat with its frame count, so beats can be lengthened or reordered there.
- `spin.py`: the spinning bullet (fluted drill body, rune band, spin blur, vortex, flung sparks, drill spray)
- `shots.py`: the bullet close-up and the tracking shot
- `sukuna.py`: Sukuna's sprite and face close-up, the pixel font and dialogue box, the shrine, the domain, the slashes
- `rudeus.py`: Rudeus's head sprites (`calm`, `focus`, `shout`, `shock`), robe and legs, arms
- `world.py`: sky, moon, mountains, plain, grass, and the crag (stone blocks, hole, cracks, collapse, rubble)
- `fx.py`: magic circles, motes, rocks, the launch and impact effects, the earth wall
- `canvas.py`: palette, scene buffer (with a margin for screen shake), primitives, quantiser
- `build.py`: renders all frames and writes the exports (`ase_writer.py` is the native .aseprite writer)
