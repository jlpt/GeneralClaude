import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import type { Brain, BrainFactory, ToolOutcome, TurnOutput } from "./brains/types.js";
import { buildSystemPrompt } from "./prompt.js";
import { loadScenario, type Scenario } from "./scenario.js";
import { runTool } from "./tools.js";
import { World, type Character } from "./world.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const RUNS_DIR = path.resolve(here, "..", "runs");

export interface SimOptions {
  /** Pause world time while any character is thinking (turn-based feel, deterministic runs). */
  lockstep: boolean;
  /** Max simultaneous brain calls. */
  maxConcurrent: number;
  /** Minimum sim-seconds between two turns of the same character. */
  minGap: number;
  /** An idle character with nothing new to react to takes a turn after this many sim-seconds. */
  idleWake: number;
  /** Hard cap on brain turns per run; the sim pauses when reached. */
  maxTurns: number;
  /** World speed multiplier. */
  timeScale: number;
  /** Write a JSONL transcript under runs/. */
  record: boolean;
}

export const defaultSimOptions = (): SimOptions => ({
  lockstep: process.env.RP_LOCKSTEP === "1",
  maxConcurrent: Number(process.env.RP_MAX_CONCURRENT || 3),
  minGap: 1.5,
  idleWake: 10,
  maxTurns: Number(process.env.RP_MAX_TURNS || 150),
  timeScale: 1,
  record: process.env.RP_RECORD !== "0",
});

export interface AgentState {
  id: string;
  brain: Brain;
  systemPrompt: string;
  pending: ToolOutcome[];
  directions: string[];
  lastTurnAt: number;
  turns: number;
  backoffUntil: number;
  inFlight: boolean;
  history: { t: number; monologue?: string; thinking?: string; calls: { name: string; input: unknown; ok: boolean; message: string }[]; error?: string }[];
}

export interface Totals { turns: number; input: number; output: number; cacheRead: number; cacheWrite: number; costUsd: number }

export class Simulation {
  world: World;
  agents = new Map<string, AgentState>();
  running = false;
  totals: Totals = { turns: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0, costUsd: 0 };
  private stepPending: Set<string> | null = null;
  private stepDeadline = 0;
  private transcript: fs.WriteStream | null = null;
  private listeners: ((ev: string, data: unknown) => void)[] = [];

  constructor(public scenario: Scenario, public brainFactory: BrainFactory, public brainName: string, public opts: SimOptions = defaultSimOptions()) {
    this.world = new World(scenario);
    for (const c of this.world.characters.values()) {
      const systemPrompt = buildSystemPrompt(this.world, c);
      this.agents.set(c.id, {
        id: c.id, systemPrompt, brain: brainFactory({ world: this.world, character: c, systemPrompt }),
        pending: [], directions: [], lastTurnAt: -999, turns: 0, backoffUntil: 0, inFlight: false, history: [],
      });
    }
    this.world.onLog((e) => {
      this.notify("log", e);
      this.record({ type: "log", ...e });
    });
    if (opts.record) {
      fs.mkdirSync(RUNS_DIR, { recursive: true });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      this.transcript = fs.createWriteStream(path.join(RUNS_DIR, `${stamp}-${scenario.id}.jsonl`));
      this.record({ type: "start", scenario: scenario.id, brain: brainName, opts });
    }
    this.world.emit("system", `Scene loaded: ${scenario.title}. Brain: ${brainName}. Press Play to start.`);
  }

  static fromScenarioId(id: string, factory: BrainFactory, brainName: string, opts?: SimOptions) {
    return new Simulation(loadScenario(id), factory, brainName, opts);
  }

  on(fn: (ev: string, data: unknown) => void) { this.listeners.push(fn); }
  private notify(ev: string, data: unknown) { for (const fn of this.listeners) fn(ev, data); }
  private record(obj: unknown) { this.transcript?.write(JSON.stringify(obj) + "\n"); }

  dispose() { this.running = false; this.transcript?.end(); this.transcript = null; }

  get inFlight() { return [...this.agents.values()].filter((a) => a.inFlight).length; }

  /** Advance the simulation by dt real seconds. Call this from a fixed-rate loop. */
  tick(dt: number) {
    if (!this.running) return;
    if (!(this.opts.lockstep && this.inFlight > 0)) this.world.step(dt * this.opts.timeScale);
    this.schedule();
    if (this.stepPending && this.stepPending.size === 0) {
      const settled = [...this.world.characters.values()].every((c) => !c.target && !(c.speech && c.speech.until > this.world.time));
      if (settled || this.world.time > this.stepDeadline) { this.stepPending = null; this.pause(); }
    }
  }

  play() { this.running = true; this.notify("status", this.status()); }
  pause() { this.running = false; this.notify("status", this.status()); }

  /** Give every AI character exactly one turn, let the results play out, then pause. */
  step() {
    const ids = [...this.agents.values()].filter((a) => this.isAI(a.id)).map((a) => a.id);
    this.stepPending = new Set(ids);
    this.stepDeadline = this.world.time + 15;
    for (const id of ids) void this.takeTurn(id, true);
    this.running = true;
    this.notify("status", this.status());
  }

  isAI(id: string) { return this.world.characters.get(id)!.def.control === "llm"; }

  setControl(id: string, control: "llm" | "player") {
    const c = this.world.characters.get(id);
    if (!c) throw new Error("no such character");
    c.def.control = control;
    c.target = null;
    this.world.emit("system", control === "player" ? `You are now playing ${c.name}.` : `${c.name} is now controlled by ${this.brainName}.`);
    this.notify("scene", this.world.sceneInfo());
  }

  private ready(c: Character, a: AgentState): boolean {
    const t = this.world.time;
    if (a.inFlight || !this.isAI(c.id) || t < a.backoffUntil) return false;
    if (t - a.lastTurnAt < this.opts.minGap) return false;
    if (c.addressed || a.directions.length) return true; // wake early when spoken to
    if (this.world.isBusy(c)) return false;
    return c.inbox.length > 0 || a.pending.length > 0 || t - a.lastTurnAt >= this.opts.idleWake;
  }

  private schedule() {
    if (this.stepPending) return; // step mode: only the explicit turns run
    if (this.totals.turns + this.inFlight >= this.opts.maxTurns) {
      if (this.inFlight) return;
      this.world.emit("system", `Turn budget reached (${this.opts.maxTurns}). Paused — raise RP_MAX_TURNS or reset.`);
      this.pause();
      return;
    }
    // Longest-waiting characters go first.
    const queue = [...this.world.characters.values()]
      .map((c) => ({ c, a: this.agents.get(c.id)! }))
      .filter(({ c, a }) => this.ready(c, a))
      .sort((x, y) => x.a.lastTurnAt - y.a.lastTurnAt);
    for (const { c } of queue) {
      if (this.inFlight >= this.opts.maxConcurrent) break;
      void this.takeTurn(c.id);
    }
  }

  async takeTurn(id: string, fromStep = false) {
    const a = this.agents.get(id)!;
    const c = this.world.characters.get(id)!;
    if (a.inFlight) return;
    a.inFlight = true;
    c.thinking = true;
    a.lastTurnAt = this.world.time;
    const observation = this.world.observe(id);
    const takenInbox = c.inbox;
    c.inbox = [];
    c.addressed = false;
    const directions = a.directions;
    a.directions = [];
    const startedAt = this.world.time;

    let out: TurnOutput | null = null;
    try {
      out = await a.brain.turn({ observation, results: a.pending, directions });
    } catch (err) {
      c.inbox = [...takenInbox, ...c.inbox];
      a.directions = [...directions, ...a.directions];
      const msg = describeError(err);
      a.backoffUntil = this.world.time + (err instanceof Anthropic.RateLimitError ? 15 : 5);
      a.history.push({ t: startedAt, calls: [], error: msg });
      this.world.emit("error", `${c.name}: ${msg}`, id);
      this.record({ type: "error", agent: id, t: startedAt, error: msg });
      if (err instanceof Anthropic.AuthenticationError || err instanceof Anthropic.PermissionDeniedError) this.pause();
    } finally {
      a.inFlight = false;
      c.thinking = false;
      if (fromStep) this.stepPending?.delete(id);
    }
    if (!out) return;
    if (out.rolledBack) {
      c.inbox = [...takenInbox, ...c.inbox];
    } else {
      a.pending = [];
    }

    this.totals.turns++;
    a.turns++;
    if (out.usage) {
      this.totals.input += out.usage.input;
      this.totals.output += out.usage.output;
      this.totals.cacheRead += out.usage.cacheRead;
      this.totals.cacheWrite += out.usage.cacheWrite;
      this.totals.costUsd += out.usage.costUsd ?? 0;
    }

    const calls: AgentState["history"][number]["calls"] = [];
    // Let the director's possession take effect mid-thought: a player-controlled character doesn't act on stale AI turns.
    for (const call of out.calls) {
      const res = this.isAI(id)
        ? runTool(this.world, id, call.name, call.input)
        : { ok: false, message: "A human took over your character before you could act; this action did not happen." };
      a.pending.push({ id: call.id, name: call.name, ok: res.ok, message: res.message });
      calls.push({ name: call.name, input: call.input, ok: res.ok, message: res.message });
      if (!res.ok) this.world.emit("error", `${c.name} tried ${call.name}: ${res.message}`, id);
    }
    if (out.monologue) this.world.emit("thought", `${c.name} (thinks): ${out.monologue}`, id);
    const entry = { t: startedAt, monologue: out.monologue, thinking: out.thinking, calls };
    a.history.push(entry);
    if (a.history.length > 60) a.history.shift();
    this.record({ type: "turn", agent: id, model: out.model, observation, directions, ...entry, usage: out.usage, stopReason: out.stopReason });
    this.notify("agent", this.agentInfo(id));
    this.notify("status", this.status());
  }

  // ------------------------------------------------------------ director & player

  narrate(text: string) { this.world.narrate(text); }

  direct(id: string, text: string) {
    const a = this.agents.get(id);
    if (!a) throw new Error("no such character");
    a.directions.push(text);
    this.world.emit("system", `Direction to ${this.world.characters.get(id)!.name}: ${text}`);
  }

  playerAction(id: string, name: string, input: unknown) {
    if (this.isAI(id)) throw new Error("that character is AI-controlled; take control first");
    const res = runTool(this.world, id, name, input);
    if (!res.ok) this.world.emit("error", res.message, id);
    return res;
  }

  // ------------------------------------------------------------ views

  status() {
    return { running: this.running, lockstep: this.opts.lockstep, timeScale: this.opts.timeScale, brain: this.brainName, totals: this.totals, maxTurns: this.opts.maxTurns, inFlight: this.inFlight };
  }

  agentInfo(id: string) {
    const a = this.agents.get(id)!;
    const c = this.world.characters.get(id)!;
    return {
      id, name: c.name, control: c.def.control, persona: c.def.persona, goals: c.def.goals, secrets: c.def.secrets,
      turns: a.turns, systemPrompt: a.systemPrompt, observation: this.world.observe(id), history: a.history.slice(-25),
    };
  }
}

function describeError(err: unknown): string {
  if (err instanceof Anthropic.AuthenticationError) return "Authentication failed — set ANTHROPIC_API_KEY (or run `ant auth login`), or start with RP_BRAIN=mock.";
  if (err instanceof Anthropic.RateLimitError) return "Rate limited — backing off.";
  if (err instanceof Anthropic.BadRequestError) return `Bad request: ${err.message}`;
  if (err instanceof Anthropic.APIError) return `API error ${err.status ?? ""}: ${err.message}`;
  return err instanceof Error ? err.message : String(err);
}
