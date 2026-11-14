/**
 * Run a scene without the browser and print what happens. Useful for
 * batch runs, comparing models/prompts, and regression-testing scenarios.
 *
 *   npm run headless -- --scenario village-festival --brain mock --seconds 90
 *   npm run headless -- --brain claude --seconds 60 --narrate "20:A bell rings out from the tower."
 */
import { parseArgs } from "node:util";
import { claudeBrain, claudeDefaults } from "./brains/claude.js";
import { mockBrain } from "./brains/mock.js";
import { Simulation, defaultSimOptions } from "./simulation.js";

const { values } = parseArgs({
  options: {
    scenario: { type: "string", default: "village-festival" },
    brain: { type: "string", default: process.env.RP_BRAIN || "mock" },
    seconds: { type: "string", default: "60" },
    "max-turns": { type: "string" },
    realtime: { type: "boolean", default: false },
    narrate: { type: "string", multiple: true, default: [] },
    "show-thoughts": { type: "boolean", default: false },
  },
});

const opts = { ...defaultSimOptions(), lockstep: !values.realtime };
if (values["max-turns"]) opts.maxTurns = Number(values["max-turns"]);
const [factory, label] = values.brain === "claude"
  ? [claudeBrain(), `${claudeDefaults().model} · effort ${claudeDefaults().effort}`]
  : [mockBrain(), "mock (offline)"];

const sim = Simulation.fromScenarioId(values.scenario!, factory, label, opts);

// "--narrate 20:text" schedules narration at t=20s
const cues = (values.narrate as string[]).map((n) => {
  const m = n.match(/^(\d+(?:\.\d+)?):(.*)$/);
  return m ? { t: Number(m[1]), text: m[2] } : { t: 0, text: n };
}).sort((a, b) => a.t - b.t);

sim.world.onLog((e) => {
  if (e.kind === "thought" && !values["show-thoughts"]) return;
  if (e.kind === "move" && e.text.includes(" toward ")) return; // keep output readable
  const tag = { say: "💬", act: "✋", emote: "🙂", move: "🚶", item: "📦", narrate: "📜", system: "⚙️ ", thought: "💭", error: "⚠️ " }[e.kind];
  console.log(`${e.t.toFixed(1).padStart(6)}s ${tag} ${e.text}`);
});

const limit = Number(values.seconds);
sim.play();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
while (sim.world.time < limit) {
  while (cues.length && cues[0].t <= sim.world.time) sim.narrate(cues.shift()!.text);
  sim.tick(0.1);
  if (!sim.running && !sim.inFlight) break; // budget reached or auth failure
  await sleep(sim.inFlight || values.realtime ? 20 : 0);
}
while (sim.inFlight) await sleep(50);

const t = sim.totals;
console.log(`\n— ${sim.world.time.toFixed(1)}s simulated, ${t.turns} turns, ${t.input + t.cacheRead + t.cacheWrite} input tokens (${t.cacheRead} cached), ${t.output} output tokens` + (t.costUsd ? `, ≈$${t.costUsd.toFixed(3)}` : ""));
sim.dispose();
