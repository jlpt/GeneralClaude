import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const here = path.dirname(fileURLToPath(import.meta.url));
export const SCENARIO_DIR = path.resolve(here, "..", "scenarios");

const Vec2 = z.tuple([z.number(), z.number()]); // [x, z] on the ground plane

const Shape = z.object({
  kind: z.enum(["box", "cylinder", "cone", "sphere"]),
  size: z.tuple([z.number(), z.number(), z.number()]).default([2, 2, 2]),
  color: z.string().default("#888888"),
});

export const LandmarkSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().default(""),
  position: Vec2,
  radius: z.number().default(2),
  shape: Shape.optional(),
});

export const ItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().default(""),
  position: Vec2,
  color: z.string().default("#ffcc00"),
  shape: z.enum(["box", "sphere", "cylinder", "cone"]).default("box"),
});

export const CharacterSchema = z.object({
  id: z.string(),
  name: z.string(),
  /** Who this character is: personality, backstory, voice. Shown to the LLM. */
  persona: z.string(),
  /** What they want during this scene. Shown to the LLM. */
  goals: z.array(z.string()).default([]),
  /** Private knowledge only this character has. Shown to the LLM, never to others. */
  secrets: z.array(z.string()).default([]),
  /** One-line description others see when they look at this character. */
  appearance: z.string().default(""),
  position: Vec2,
  facing: z.number().default(0),
  color: z.string().default("#4a90d9"),
  headColor: z.string().default("#f1c27d"),
  height: z.number().default(1.0),
  /** "llm" = driven by the configured brain, "player" = driven by you in the browser. */
  control: z.enum(["llm", "player"]).default("llm"),
  /** Optional canned lines for the offline mock brain. */
  mockLines: z.array(z.string()).default([]),
});

export const ScenarioSchema = z.object({
  id: z.string(),
  title: z.string(),
  /** Scene-setting text every character receives. */
  setting: z.string(),
  /** Hidden notes for the director (you). Never sent to characters. */
  directorNotes: z.string().default(""),
  worldSize: z.number().default(40),
  groundColor: z.string().default("#5a7d4a"),
  skyColor: z.string().default("#9cc9ff"),
  landmarks: z.array(LandmarkSchema).default([]),
  items: z.array(ItemSchema).default([]),
  characters: z.array(CharacterSchema).min(1),
});

export type Scenario = z.infer<typeof ScenarioSchema>;
export type CharacterDef = z.infer<typeof CharacterSchema>;
export type LandmarkDef = z.infer<typeof LandmarkSchema>;
export type ItemDef = z.infer<typeof ItemSchema>;

export function listScenarios(): { id: string; title: string }[] {
  return fs
    .readdirSync(SCENARIO_DIR)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      const s = loadScenario(f.replace(/\.json$/, ""));
      return { id: s.id, title: s.title };
    });
}

export function loadScenario(id: string): Scenario {
  if (!/^[a-z0-9-_]+$/i.test(id)) throw new Error(`bad scenario id: ${id}`);
  const file = path.join(SCENARIO_DIR, `${id}.json`);
  const raw = JSON.parse(fs.readFileSync(file, "utf8"));
  return ScenarioSchema.parse(raw);
}
