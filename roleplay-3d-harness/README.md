# Roleplay 3D Harness

A harness for LLM-driven characters who roleplay inside a shared 3D world. Each character is its own Claude conversation with its own persona, goals and secrets. Characters perceive the world through text observations and act through tools (`say`, `walk_to`, `emote`, `act`, `pick_up`, `give`, …). The server simulates the world and checks every action, so a character can't hear speech from across the map or take an item it isn't standing next to. The browser shows it all live in three.js.

You can watch, play one of the characters yourself, or act as the director: narrate events, spawn props, and send private stage directions to any character.

```
┌─────────── browser (three.js) ───────────┐        ┌──────────────── server (Node/TS) ────────────────┐
│ 3D scene · speech bubbles · animations   │◀─SSE── │ World: positions, perception ranges, items       │
│ Log · character inspector · director     │──POST─▶│ Simulation: schedules turns, runs tool calls     │
│ Player controls (click to walk, say, act)│        │ Brains: Claude (one conversation per character)  │
└──────────────────────────────────────────┘        │         Mock (offline, scripted)                 │
                                                    └──────────────────────────────────────────────────┘
```

## Quick start

```bash
cd roleplay-3d-harness
npm install
export ANTHROPIC_API_KEY=sk-ant-...   # or `ant auth login`; without credentials it runs the offline mock brain
npm start                             # http://localhost:3000
```

Press **▶ Play** to run in real time, or **⏭ Step** to give every AI character exactly one turn and then pause.

- **Play a character.** The *Traveler* (village) and the *Inspector* (station) are yours. Click the ground to walk (shift = run). Click people or things to target them. Type to speak. To take over any other character, tick *I control this character* in its inspector.
- **Direct.** In the *Director* tab, narrate events everyone perceives (for example "The sun dips toward the hills — dusk is close.") or spawn items. In a character's inspector, send a stage direction that reaches only that character, as an out-of-character system message.
- **Inspect.** Click a character to see its persona, its secrets, the exact observation it receives, its reasoning summary, its inner monologue, and every tool call with the world's response.

## How a turn works

1. The scheduler wakes a character when something happens near it, when someone speaks to it, or after it has been idle for a while. Characters that are busy walking or talking aren't woken.
2. The character receives an **observation**: where it is, what it can see (with distances and compass bearings), and everything it heard or saw since its last turn.
3. The brain returns tool calls. The world checks each one (range, reach, possession) and applies it. A rejected call returns an error the model sees on its next turn, together with the next observation.
4. The other characters perceive the results only if they are within sight or earshot (whisper ≈2.5m, normal ≈12m, shout ≈35m).

By default the world keeps running while characters think, which feels live. **Freeze world time while characters think** makes it turn-based and reproducible.

## Claude configuration

| env | default | |
|---|---|---|
| `RP_BRAIN` | `claude` if credentials are found, else `mock` | `claude` or `mock` |
| `RP_MODEL` | `claude-opus-5-5` | any Claude model id |
| `RP_EFFORT` | `low` | `low` … `max`. Low keeps turns fast; raise it for richer scheming |
| `RP_MAX_TURNS` | `150` | turn budget; the sim pauses when it's reached (adjustable in the UI) |
| `RP_MAX_CONCURRENT` | `3` | simultaneous model calls |
| `RP_LOCKSTEP` | off | `1` = start in turn-based mode |
| `RP_SCENARIO` | `village-festival` | scenario loaded at startup |
| `RP_RECORD` | on | `0` disables JSONL transcripts in `runs/` |

What the Claude brain does (`src/brains/claude.ts`):

- **One append-only conversation per character.** The system prompt and tools never change during a run, so the prompt cache stays warm; the UI shows the cache hit rate and an estimated cost. Thinking blocks are passed back unchanged.
- **Adaptive thinking with summarized display.** The inspector shows a summary of each character's reasoning.
- **Strict tool schemas.** Every input is also re-checked with zod on the server, so other brains go through the same checks.
- **Server-side compaction** (`compact-2026-01-12`) keeps long scenes within the context window.
- **Refusal fallback** (`fallbacks: "default"`) is on for models that support it. A turn that still ends in a refusal, or that hits `max_tokens`, is rolled back and nothing from it is executed.
- **Director notes** go in as mid-conversation `system` messages on models that support them, and as text in the user turn on other models.

## Headless runs

You can run scenes without the browser, for example to compare models or prompts or to regression-test a scenario:

```bash
npm run headless -- --scenario village-festival --brain claude --seconds 90 \
  --narrate "30:The sun dips toward the hills — dusk is close." --show-thoughts
```

Headless runs are turn-based by default (`--realtime` turns that off). Every run, headless or not, writes a JSONL transcript to `runs/`. It records each turn's observation, director notes, tool calls, results, monologue, token usage, and the world log.

## Writing scenarios

A scenario is a JSON file in `scenarios/`. The schema is in `src/scenario.ts`.

- `setting`: scene-setting text that every character receives.
- `landmarks`: named places, drawn as boxes, cylinders, cones or spheres with a size and colour.
- `items`: things characters can pick up, give and drop.
- `characters`: each one has a `persona`, `goals`, `secrets` (only that character sees them), an `appearance` (what others see), colours and a start position. Set `control: "player"` for a human-controlled character. `mockLines` are lines the offline brain says.
- `directorNotes`: notes for you only. They appear in the Director tab and are never sent to a character.

Coordinates are `[x, z]` on the ground plane, with north at −z. `worldSize` is the width of the square map.

## Extending

- **New actions.** Add an entry to `TOOLS` in `src/tools.ts` with a zod schema and a `run` function, and implement it on `World`. To make it visible, add an animation case in `animateCharacter` in `public/app.js`.
- **Other LLMs.** Implement the `Brain` interface in `src/brains/types.ts`: take `{observation, results, directions}` and return tool calls. The scheduler, world rules, UI and transcripts work unchanged.
- **Real models and art.** The renderer uses primitive shapes. To use your own art, swap `makeCharacter` and `makeItem` for GLTF loaders and drive their animation clips from the emote names.

## Tests

```bash
npm test        # world rules, tool schemas, and a Claude-brain contract test against a local fake API
npm run typecheck
```
