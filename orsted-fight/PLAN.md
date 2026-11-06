# Plan — Eris, Rudeus & Ruijerd vs Orsted (pixel-art fight animation)

## Deliverables
| File | What |
|---|---|
| `orsted_fight.mp4` | Full fight, 4× nearest-neighbour upscale (1920×1080), 30 fps container |
| `orsted_fight_preview.gif` | Same animation at 2× (960×540) |
| `orsted_fight.aseprite` | 480×270 source: one layer per character + background/FX/text layers, a tag per story beat |
| `storyboard.png` | Contact sheet of one key frame per beat |
| `src/` | Python generator (numpy + Pillow + ffmpeg). Everything is drawn by code, so it can be regenerated and tweaked |

## Technical spec
- **Canvas** 480×270 (16:9), ground line at y≈244. Characters are drawn at the same pixel scale as the
  existing `eris-sword-slash` piece (Eris ≈ 62 px tall), so her sprite carries over unchanged.
- **Timing** 15 fps base ("on twos"), ~24 s ≈ 360 frames. Identical consecutive frames are merged into one longer
  frame in the GIF/Aseprite exports.
- **Engine**
  - `px.py`: canvas with coordinate grids, plus bbox-clipped primitives (capsule, disc, ellipse, polygon, line),
    IK, outline/shading helpers. These are generalized from `eris-sword-slash/src/rig.py`.
  - Each character renders into its own local canvas (root = point on the ground under the pelvis, facing right).
    It is then flipped if it faces left and composited into the world with an owner map, so each pixel belongs to
    exactly one Aseprite layer.
  - `timeline.py`: per-character keyframe tracks. Keys are partial pose dicts that inherit from the previous key.
    Vectors and angles interpolate with easing (`linear/in/out/inout/step`), and discrete fields (expression,
    weapon state) snap. The procedural extras are breathing bob, auto hair/coat wind phase, and hit-flash.
  - Weapon smears are generated automatically from the weapon's angular velocity between frames. The other effects
    (spells, impacts, dust, shockwaves) are scheduled events that can follow a character's attachment points
    (sword tip, staff gem, palm).
  - Camera: screen shake from a margin-padded world canvas, plus fades and impact flashes as post-processing.

## Cast (all original pixel art, 3/4 view)
| Character | Height | Look | Weapon / kit |
|---|---|---|---|
| **Eris** | 62 px | Long crimson hair (procedural strands), red eyes. Black fur-collared jacket, crop top, shorts, as in the repo's existing Eris piece | Longsword: smears, overhead and full-power slashes |
| **Rudeus** | ~58 px | Messy brown hair, green eyes. Grey hooded mage robe over a cream shirt, brown trousers, boots | Staff *Aqua Heartia* (blue crystal head): magic circle, Stone Cannon, Quagmire |
| **Ruijerd** | ~72 px | Shaved head, pale skin, glowing red forehead jewel. Slate-teal tunic, sash, wrapped trousers | White trident: multi-thrust afterimages |
| **Orsted** | ~82 px | Tall, silver swept-back hair, glowing golden eyes, scowl. Long white coat flapping in the wind, black clothes | Bare-handed: Ran Ma (magic disruption), mana burst, blink-dash, palm strike |

## Stage
Twilight rocky pass ("Red Wyrm's Jaw" mood) in muted blue-violet so the red, gold and white of the cast stand out.
It has a dithered sky gradient, stars and a moon at the upper left (the light source matches the character
shading), two parallax mountain silhouettes, cliffs on both sides, and a dirt road with grass tufts and rocks.
Drifting wind motes keep it alive during holds. Ground shadows sit under each character.

## Storyboard (~24 s)
| # | Beat | ≈Time | Action |
|---|---|---|---|
| 1 | **Standoff** | 0.0–3.0 s | Fade in. The three heroes are in stances at the left, Orsted stands calm at the right with his coat whipping in the wind. Title text, then Orsted's eyes glint. |
| 2 | **Eris opens** | 3.0–6.0 s | Eris dashes (speed lines, dust) into a leaping overhead slash. Orsted flicks the blade away with the back of his hand (sparks, shake). Eris tumbles back and lands in a skid. |
| 3 | **Ruijerd** | 6.0–9.0 s | Ruijerd lunges into a rapid triple thrust (afterimages). Orsted slips every thrust, seizes the shaft and hurls him. Ruijerd lands on one knee. |
| 4 | **Stone Cannon** | 9.0–12.5 s | Rudeus raises his staff: magic circle, a rock forms and spins up, then fires. Orsted opens his palm and the *Ran Ma* ripple makes the stone crumble to dust in mid-air. |
| 5 | **Team combo** | 12.5–16.0 s | Rudeus slams his staff and Quagmire swallows Orsted's feet. Eris (low) and Ruijerd (leaping from above) strike together. Orsted releases a mana burst (shockwave ring, debris) and blows both away. |
| 6 | **Dragon God** | 16.0–19.0 s | Orsted vanishes (afterimages) and reappears in front of Rudeus. Palm strike: white flash frame, radial burst. Rudeus is launched and stays down. |
| 7 | **Last stand** | 19.0–22.0 s | Eris roars into a full-power slash (warm smear). Orsted catches the blade between two fingers, then a backhand drops her. Ruijerd's swipe is answered with an elbow and he falls too. |
| 8 | **Aftermath** | 22.0–24.5 s | Orsted stands alone in settling dust with his coat in the wind. His eyes glow, then fade to black with the end title. |

## Build order and QA
1. Engine (canvas, primitives, compositing, timeline), then the stage background. Check with a still render.
2. Port Eris, then build Rudeus, Ruijerd and Orsted (hand-pixelled heads + rigged bodies + weapons). Check with a
   turnaround sheet of key poses for each character, viewed zoomed in.
3. FX library (smears, sparks, dust, speed lines, magic circle, Stone Cannon, Ran Ma, Quagmire, shockwave,
   afterimages, flash), then pixel font for the titles.
4. Choreograph beat by beat, rendering key frames after each beat to check readability, contact and timing.
5. Export MP4, GIF, Aseprite and storyboard. Verify the Aseprite file parses and the layers composite back to the
   flat frames. Write the README.
6. Commit and push to the working branch.
