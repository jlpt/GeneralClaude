import { z } from "zod";
import { EMOTES, type ActionResult, type World } from "./world.js";

/**
 * The action vocabulary a character has in the world. Each tool has a JSON
 * schema (sent to the model) and a zod schema (re-validated on our side, so
 * any brain — not just Claude — goes through the same checks).
 */

interface ToolSpec<S extends z.ZodType> {
  name: string;
  description: string;
  schema: S;
  run: (world: World, actorId: string, input: z.infer<S>) => ActionResult;
}

const tool = <S extends z.ZodType>(t: ToolSpec<S>) => t;

export const TOOLS = [
  tool({
    name: "say",
    description:
      "Speak out loud, in character. Only people within earshot hear you (whisper ≈2.5m, normal ≈12m, shout ≈35m). " +
      "Use `to` to address one person by name, or \"everyone\".",
    schema: z.object({
      text: z.string().min(1).max(600).describe("Exactly what your character says. Dialogue only, no stage directions."),
      to: z.string().describe('Name of the person you are addressing, or "everyone".'),
      volume: z.enum(["whisper", "normal", "shout"]),
    }),
    run: (w, id, i) => w.say(id, i.text, i.volume, i.to),
  }),
  tool({
    name: "walk_to",
    description:
      "Start moving toward a place, person, or item you can see (by name), or a coordinate like \"3, -4\". " +
      "Movement takes time; you'll be told when you arrive. Walking toward a person follows them.",
    schema: z.object({
      destination: z.string().min(1),
      pace: z.enum(["walk", "run"]),
    }),
    run: (w, id, i) => w.walkTo(id, i.destination, i.pace),
  }),
  tool({
    name: "stop",
    description: "Stop moving.",
    schema: z.object({}),
    run: (w, id) => w.stop(id),
  }),
  tool({
    name: "emote",
    description: "Perform a visible body-language animation.",
    schema: z.object({ animation: z.enum(EMOTES) }),
    run: (w, id, i) => w.emote(id, i.animation),
  }),
  tool({
    name: "act",
    description:
      "Perform a physical action others can see, written in third person present tense without your name " +
      "(e.g. \"lights the lantern\", \"kneels to inspect the tracks\"). Optionally name a nearby target you must be within reach of.",
    schema: z.object({
      description: z.string().min(1).max(300),
      target: z.string().describe('Name of what you act on, or "none".'),
    }),
    run: (w, id, i) => w.act(id, i.description, i.target),
  }),
  tool({
    name: "look_at",
    description: "Turn to look at something or someone closely and get more detail about it.",
    schema: z.object({ target: z.string().min(1) }),
    run: (w, id, i) => w.lookAt(id, i.target),
  }),
  tool({
    name: "pick_up",
    description: "Pick up an item within reach (≈2m). You can hold two items.",
    schema: z.object({ item: z.string().min(1) }),
    run: (w, id, i) => w.pickUp(id, i.item),
  }),
  tool({
    name: "drop",
    description: "Put down an item you are holding.",
    schema: z.object({ item: z.string().min(1) }),
    run: (w, id, i) => w.drop(id, i.item),
  }),
  tool({
    name: "give",
    description: "Hand an item you are holding to someone within reach.",
    schema: z.object({ item: z.string().min(1), to: z.string().min(1) }),
    run: (w, id, i) => w.give(id, i.item, i.to),
  }),
  tool({
    name: "wait",
    description: "Do nothing for a while (1–30s) — listen, watch, let others act. You'll be woken early if someone addresses you.",
    schema: z.object({ seconds: z.number() }),
    run: (w, id, i) => w.wait(id, i.seconds),
  }),
] as const;

export type ToolName = (typeof TOOLS)[number]["name"];

/** JSON-schema tool definitions in the shape the Messages API expects. */
export function toolDefinitions() {
  return TOOLS.map((t) => {
    const schema = stripConstraints(z.toJSONSchema(t.schema)) as Record<string, unknown>;
    delete schema.$schema;
    return {
      name: t.name,
      description: t.description,
      strict: true,
      input_schema: { ...schema, type: "object" as const, additionalProperties: false },
    };
  });
}

/** Strict tool schemas don't take length/range constraints; zod still enforces them on our side. */
function stripConstraints(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(stripConstraints);
  if (!node || typeof node !== "object") return node;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node)) {
    if (["minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"].includes(k)) continue;
    out[k] = stripConstraints(v);
  }
  return out;
}

export function runTool(world: World, actorId: string, name: string, input: unknown): ActionResult {
  const t = TOOLS.find((x) => x.name === name);
  if (!t) return { ok: false, message: `Unknown action "${name}".` };
  const parsed = t.schema.safeParse(input);
  if (!parsed.success) return { ok: false, message: `Invalid input for ${name}: ${parsed.error.issues.map((i) => i.message).join("; ")}` };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (t.run as any)(world, actorId, parsed.data);
}
