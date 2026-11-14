"""Ask Claude to reconstruct a hidden target from a few photos, then score it on held-out angles.

This produces exactly the data an RL / rejection-sampling pipeline needs:
one JSONL row per attempt with the prompt, the model's scene program, and its rewards.

    python llm_attempt.py --episodes 5 --samples 4 --refine 2 --level 2

  --samples N  : N independent attempts per target (a GRPO "group" / best-of-N pool)
  --refine K   : after each attempt, show the model renders of ITS scene from the INPUT cameras
                 and let it revise K times. Held-out views are never shown, so they stay a real test.
  --dry-run    : no API call; uses a crude stand-in guess so you can test the pipeline offline.

Needs ANTHROPIC_API_KEY (or another credential the anthropic SDK picks up) unless --dry-run.
"""
import argparse
import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

from ig.episode import make_episode
from ig.render import render

MODEL = "claude-opus-5-5"
IMG_RES = 256

SYSTEM = """You reconstruct 3D scenes from photos by writing a scene program.

Scene program format: a JSON list of primitives. Units are meters, y is up, the ground is y=0
and objects rest on or above it. Available primitives:
  {"type": "sphere",   "center": [x,y,z], "radius": r, "color": [r,g,b]}
  {"type": "box",      "center": [x,y,z], "size": [sx,sy,sz], "rot_y": degrees, "color": [r,g,b]}
  {"type": "cylinder", "center": [x,y,z], "radius": r, "height": h, "axis": "x"|"y"|"z", "color": [r,g,b]}
Colors are 0..1 floats (the stored base color; rendered pixels are darker on the shadow side).
rot_y rotates a box counterclockwise about the vertical axis when viewed from above.
Cameras are pinhole, square images, black background, lit from above.
Azimuth 0 means the camera sits on the +z axis looking toward -z; azimuth 90 sits on +x.

Your scene will be scored by rendering it from NEW camera angles you have not seen and comparing
with the real object, so model the full 3D shape, not just what each photo shows.

Reason about positions and sizes, then end your reply with the final program inside
<scene> ... </scene> tags."""


def png_b64(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def image_block(rgb):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(rgb)}}


def first_turn(ep):
    content = [{"type": "text", "text": f"Here are {len(ep.input_cams)} photos of the target."}]
    for i, cam in enumerate(ep.input_cams):
        content += [{"type": "text", "text": f"Photo {i + 1}: camera {cam.describe()}"},
                    image_block(render(ep.target, cam, IMG_RES)[0])]
    content.append({"type": "text", "text": "Write the scene program."})
    return {"role": "user", "content": content}


def feedback_turn(ep, scene, input_score):
    """Show the model its own scene from the INPUT cameras only, next to the targets."""
    content = [{"type": "text", "text":
                f"Your scene scores {input_score:.3f} (0-1) against these photos. "
                "Below, for each camera: the target photo, then a render of your scene. "
                "Fix the differences and write the complete revised program."}]
    for i, cam in enumerate(ep.input_cams):
        content += [{"type": "text", "text": f"Camera {i + 1} target:"},
                    image_block(render(ep.target, cam, IMG_RES)[0]),
                    {"type": "text", "text": f"Camera {i + 1} your scene:"},
                    image_block(render(scene, cam, IMG_RES)[0])]
    return {"role": "user", "content": content}


def parse_scene(text):
    m = re.findall(r"<scene>(.*?)</scene>", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m[-1].strip())
    except json.JSONDecodeError:
        return None


def call_claude(client, messages):
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=messages,
        output_config={"effort": "high"},
        # server-side fallback: if the model declines, the API retries on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return response, ""
    text = "".join(b.text for b in response.content if b.type == "text")
    return response, text


def dry_run_guess(ep):
    """Offline stand-in for the model: one gray box roughly covering the visible pixels."""
    text = ('Stand-in guess. <scene>[{"type": "box", "center": [0, 0.25, 0], '
            '"size": [0.8, 0.5, 0.8], "color": [0.6, 0.6, 0.6]}]</scene>')
    return None, text


def attempt(client, ep, refine, dry_run):
    messages = [first_turn(ep)]
    turns = []
    for k in range(refine + 1):
        response, text = dry_run_guess(ep) if dry_run else call_claude(client, messages)
        scene = parse_scene(text)
        score = ep.score(scene) if scene is not None else {"input": 0.0, "heldout": 0.0, "valid": False}
        turns.append({"turn": k, "scene": scene, "reward_input_views": score["input"],
                      "reward_heldout_views": score["heldout"], "valid": score["valid"],
                      "completion": text})
        print(f"    turn {k}: input-view score {score['input']:.3f} | held-out score {score['heldout']:.3f}"
              f"{'' if score['valid'] else '  (invalid program)'}")
        if k == refine or not score["valid"]:
            break
        # keep the full assistant content (including thinking blocks) when continuing the conversation
        messages.append({"role": "assistant", "content": response.content if response else text})
        messages.append(feedback_turn(ep, scene, score["input"]))
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/llm_attempts.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for e in range(args.episodes):
            seed = args.seed + e
            ep = make_episode(seed, level=args.level)
            print(f"episode seed={seed}: target has {len(ep.target)} primitive(s)")
            group = []
            for s in range(args.samples):
                print(f"  sample {s}")
                turns = attempt(client, ep, args.refine, args.dry_run)
                group.append(turns[-1]["reward_heldout_views"])
                f.write(json.dumps({"episode_seed": seed, "level": args.level, "sample": s,
                                    "target": ep.target, "turns": turns}) + "\n")
            g = np.array(group)
            adv = (g - g.mean()) / (g.std() + 1e-6) if g.std() > 1e-6 else np.zeros_like(g)
            print(f"  group rewards {np.round(g, 3).tolist()} -> GRPO advantages {np.round(adv, 2).tolist()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
