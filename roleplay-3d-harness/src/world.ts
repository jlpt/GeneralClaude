import type { CharacterDef, ItemDef, LandmarkDef, Scenario } from "./scenario.js";

/**
 * The authoritative world simulation. Coordinates are on the ground plane:
 * x grows to the east, z grows to the south (so north is -z), y is up.
 * Everything an LLM does goes through the action methods here, which
 * validate it against physical constraints (distance, possession, etc.).
 */

export const RANGES = { whisper: 2.5, normal: 12, shout: 35, sight: 30, reach: 2.2 } as const;
export const SPEEDS = { walk: 1.6, run: 3.6 } as const;

export const EMOTES = [
  "wave", "nod", "shake_head", "bow", "jump", "dance", "sit", "stand",
  "point", "laugh", "cry", "shrug", "cheer", "think",
] as const;
export type Emote = (typeof EMOTES)[number];
export type Volume = "whisper" | "normal" | "shout";
export type Pace = keyof typeof SPEEDS;

export interface Vec { x: number; z: number }

export interface Character {
  id: string;
  name: string;
  def: CharacterDef;
  pos: Vec;
  facing: number; // radians; 0 faces north (-z), PI/2 faces east (+x)
  target: { pos: Vec; label: string; followId?: string; stopDist: number; pace: Pace } | null;
  holding: string[]; // item ids
  emote: { name: Emote; until: number } | null;
  posture: "standing" | "sitting";
  speech: { text: string; volume: Volume; to: string | null; until: number } | null;
  action: { text: string; until: number } | null;
  waitUntil: number;
  /** Things this character perceived since its last decision. */
  inbox: string[];
  /** Set when someone addressed this character directly; wakes it early. */
  addressed: boolean;
  thinking: boolean;
}

export interface Item { id: string; name: string; def: ItemDef; pos: Vec; holder: string | null }
export interface Landmark { id: string; name: string; def: LandmarkDef; pos: Vec }

export interface LogEntry {
  t: number;
  kind: "say" | "act" | "emote" | "move" | "item" | "narrate" | "system" | "thought" | "error";
  actor?: string; // character id
  text: string;
}

export interface ActionResult { ok: boolean; message: string }

const fmt = (n: number) => (Math.round(n * 10) / 10).toFixed(1);
const dist = (a: Vec, b: Vec) => Math.hypot(a.x - b.x, a.z - b.z);
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];

function bearing(from: Vec, to: Vec): string {
  const ang = Math.atan2(to.x - from.x, -(to.z - from.z)); // 0 = north, clockwise
  const idx = Math.round(((ang + 2 * Math.PI) % (2 * Math.PI)) / (Math.PI / 4)) % 8;
  return COMPASS[idx];
}

function facingName(f: number): string {
  const idx = Math.round((((f % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (Math.PI / 4)) % 8;
  return COMPASS[idx];
}

export class World {
  time = 0;
  characters = new Map<string, Character>();
  items = new Map<string, Item>();
  landmarks = new Map<string, Landmark>();
  log: LogEntry[] = [];
  private listeners: ((e: LogEntry) => void)[] = [];
  private spawnCounter = 0;

  constructor(public scenario: Scenario) {
    for (const l of scenario.landmarks) {
      this.landmarks.set(l.id, { id: l.id, name: l.name, def: l, pos: { x: l.position[0], z: l.position[1] } });
    }
    for (const i of scenario.items) {
      this.items.set(i.id, { id: i.id, name: i.name, def: i, pos: { x: i.position[0], z: i.position[1] }, holder: null });
    }
    for (const c of scenario.characters) {
      this.characters.set(c.id, {
        id: c.id, name: c.name, def: c,
        pos: { x: c.position[0], z: c.position[1] },
        facing: c.facing, target: null, holding: [], emote: null, posture: "standing",
        speech: null, action: null, waitUntil: 0, inbox: [], addressed: false, thinking: false,
      });
    }
  }

  onLog(fn: (e: LogEntry) => void) { this.listeners.push(fn); }

  emit(kind: LogEntry["kind"], text: string, actor?: string) {
    const e: LogEntry = { t: Math.round(this.time * 10) / 10, kind, text, actor };
    this.log.push(e);
    if (this.log.length > 2000) this.log.shift();
    for (const fn of this.listeners) fn(e);
  }

  // ---------------------------------------------------------------- lookup

  /** Resolve a name the LLM typed into an entity. Accepts ids, names, partial names, or "x,z". */
  resolve(query: string, self?: Character):
    | { kind: "character"; ent: Character }
    | { kind: "item"; ent: Item }
    | { kind: "landmark"; ent: Landmark }
    | { kind: "point"; pos: Vec }
    | null {
    const q = query.trim().toLowerCase().replace(/^the /, "");
    const coord = q.match(/^\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?$/);
    if (coord) return { kind: "point", pos: this.clamp({ x: parseFloat(coord[1]), z: parseFloat(coord[2]) }) };

    type Cand = { kind: "character" | "item" | "landmark"; ent: Character | Item | Landmark; score: number };
    const cands: Cand[] = [];
    const consider = (kind: Cand["kind"], ent: Character | Item | Landmark) => {
      if (self && ent === self) return;
      const name = ent.name.toLowerCase();
      const id = ent.id.toLowerCase();
      let score = 0;
      if (q === id || q === name) score = 3;
      else if (name.startsWith(q) || name.split(/\s+/).includes(q)) score = 2;
      else if (name.includes(q) || q.includes(name)) score = 1;
      if (score) cands.push({ kind, ent, score });
    };
    this.characters.forEach((c) => consider("character", c));
    this.items.forEach((i) => consider("item", i));
    this.landmarks.forEach((l) => consider("landmark", l));
    if (!cands.length) return null;
    // Prefer best score, then nearest.
    const origin = self?.pos ?? { x: 0, z: 0 };
    cands.sort((a, b) => b.score - a.score || dist(origin, this.posOf(a.ent)) - dist(origin, this.posOf(b.ent)));
    const best = cands[0];
    return { kind: best.kind, ent: best.ent } as never;
  }

  posOf(ent: Character | Item | Landmark): Vec {
    if ("holder" in ent && ent.holder) return this.characters.get(ent.holder)!.pos;
    return ent.pos;
  }

  clamp(p: Vec): Vec {
    const h = this.scenario.worldSize / 2 - 0.5;
    return { x: Math.max(-h, Math.min(h, p.x)), z: Math.max(-h, Math.min(h, p.z)) };
  }

  nearestLandmark(p: Vec): Landmark | null {
    let best: Landmark | null = null;
    let bd = Infinity;
    this.landmarks.forEach((l) => {
      const d = dist(p, l.pos) - l.def.radius;
      if (d < bd) { bd = d; best = l; }
    });
    return best && bd < 4 ? best : null;
  }

  // ---------------------------------------------------------------- perception

  /** Deliver a perception to every character within range of `origin` (except `except`). */
  private broadcast(origin: Vec, range: number, text: (listener: Character) => string | null, except?: string) {
    this.characters.forEach((c) => {
      if (c.id === except) return;
      if (dist(c.pos, origin) > range) return;
      const t = text(c);
      if (t) c.inbox.push(`[${fmt(this.time)}s] ${t}`);
    });
  }

  /** Build the text observation a character receives at the start of its turn. */
  observe(id: string): string {
    const me = this.characters.get(id)!;
    const lines: string[] = [];
    const near = this.nearestLandmark(me.pos);
    lines.push(`Time: ${fmt(this.time)}s into the scene.`);
    lines.push(
      `You are at (${fmt(me.pos.x)}, ${fmt(me.pos.z)})${near ? `, at the ${near.name}` : ""}, facing ${facingName(me.facing)}, ${me.posture}.`,
    );
    if (me.target) lines.push(`You are currently ${me.target.pace === "run" ? "running" : "walking"} toward ${me.target.label}.`);
    const held = me.holding.map((i) => this.items.get(i)!.name);
    lines.push(`You are holding: ${held.length ? held.join(", ") : "nothing"}.`);

    const seen: string[] = [];
    this.characters.forEach((c) => {
      if (c === me) return;
      const d = dist(me.pos, c.pos);
      if (d > RANGES.sight) return;
      const bits: string[] = [];
      if (c.def.appearance) bits.push(c.def.appearance);
      if (c.target) bits.push(`${c.target.pace === "run" ? "running" : "walking"} toward ${c.target.label}`);
      else if (c.posture === "sitting") bits.push("sitting");
      if (c.emote && c.emote.until > this.time) bits.push(`currently: ${c.emote.name.replace("_", " ")}`);
      if (c.action && c.action.until > this.time) bits.push(`doing: ${c.action.text}`);
      if (c.holding.length) bits.push(`holding ${c.holding.map((i) => this.items.get(i)!.name).join(", ")}`);
      seen.push(`- ${c.name} (person), ${fmt(d)}m ${bearing(me.pos, c.pos)}${bits.length ? " — " + bits.join("; ") : ""}`);
    });
    this.items.forEach((it) => {
      if (it.holder) return;
      const d = dist(me.pos, it.pos);
      if (d > RANGES.sight) return;
      seen.push(`- ${it.name} (item on the ground), ${fmt(d)}m ${bearing(me.pos, it.pos)}${it.def.description ? " — " + it.def.description : ""}`);
    });
    this.landmarks.forEach((l) => {
      const d = Math.max(0, dist(me.pos, l.pos) - l.def.radius);
      seen.push(`- ${l.name} (place), ${d < 0.5 ? "right here" : `${fmt(d)}m ${bearing(me.pos, l.pos)}`}${l.def.description ? " — " + l.def.description : ""}`);
    });
    lines.push("", "What you can see:", ...(seen.length ? seen : ["- nothing of note"]));

    lines.push("", "What happened since your last turn:");
    lines.push(...(me.inbox.length ? me.inbox : ["- nothing new"]));
    return lines.join("\n");
  }

  // ---------------------------------------------------------------- simulation

  step(dt: number) {
    this.time += dt;
    this.characters.forEach((c) => {
      if (c.speech && c.speech.until <= this.time) c.speech = null;
      if (c.emote && c.emote.until <= this.time) c.emote = null;
      if (c.action && c.action.until <= this.time) c.action = null;
      if (!c.target) return;
      if (c.target.followId) {
        const f = this.characters.get(c.target.followId);
        if (f) c.target.pos = { ...f.pos };
      }
      const dx = c.target.pos.x - c.pos.x;
      const dz = c.target.pos.z - c.pos.z;
      const d = Math.hypot(dx, dz);
      const stop = c.target.stopDist;
      if (d <= stop + 0.05) {
        c.inbox.push(`[${fmt(this.time)}s] You arrived at ${c.target.label}.`);
        this.emit("move", `${c.name} arrives at ${c.target.label}.`, c.id);
        c.target = null;
        return;
      }
      const stepLen = Math.min(SPEEDS[c.target.pace] * dt, d - stop);
      c.pos.x += (dx / d) * stepLen;
      c.pos.z += (dz / d) * stepLen;
      c.facing = Math.atan2(dx, -dz);
      c.pos = this.clamp(c.pos);
    });
  }

  /** True while a character is still carrying out something it started. */
  isBusy(c: Character): boolean {
    return !!c.target || (!!c.speech && c.speech.until > this.time) || c.waitUntil > this.time;
  }

  // ---------------------------------------------------------------- actions

  walkTo(id: string, destination: string, pace: Pace): ActionResult {
    const me = this.characters.get(id)!;
    const r = this.resolve(destination, me);
    if (!r) return { ok: false, message: `There is no "${destination}" here. Use a name from what you can see, or coordinates like "3, -4".` };
    if (me.posture === "sitting") me.posture = "standing";
    let pos: Vec, label: string, stopDist = 0.3, followId: string | undefined;
    switch (r.kind) {
      case "point": pos = r.pos; label = `(${fmt(pos.x)}, ${fmt(pos.z)})`; break;
      case "landmark": pos = { ...r.ent.pos }; label = `the ${r.ent.name}`; stopDist = Math.max(0.3, r.ent.def.radius * 0.8); break;
      case "item": pos = { ...this.posOf(r.ent) }; label = `the ${r.ent.name}`; stopDist = 0.8; break;
      case "character": pos = { ...r.ent.pos }; label = r.ent.name; stopDist = 1.3; followId = r.ent.id; break;
    }
    const d = dist(me.pos, pos);
    if (d <= stopDist + 0.05) return { ok: true, message: `You are already at ${label}.` };
    me.target = { pos, label, stopDist, pace, followId };
    const eta = (d - stopDist) / SPEEDS[pace];
    this.emit("move", `${me.name} ${pace === "run" ? "runs" : "walks"} toward ${label}.`, id);
    return { ok: true, message: `You start ${pace === "run" ? "running" : "walking"} toward ${label} (${fmt(d)}m, about ${fmt(eta)}s). You'll be told when you arrive.` };
  }

  stop(id: string): ActionResult {
    const me = this.characters.get(id)!;
    me.target = null;
    return { ok: true, message: "You stop where you are." };
  }

  say(id: string, text: string, volume: Volume, to: string): ActionResult {
    const me = this.characters.get(id)!;
    let target: Character | null = null;
    const toQ = to.trim().toLowerCase();
    if (toQ && !["everyone", "all", "", "nobody", "self", "myself"].includes(toQ)) {
      const r = this.resolve(to, me);
      if (!r || r.kind !== "character") return { ok: false, message: `"${to}" is not a person you can talk to here. Use "everyone" to speak to the room.` };
      target = r.ent;
      me.facing = Math.atan2(target.pos.x - me.pos.x, -(target.pos.z - me.pos.z));
    }
    const range = RANGES[volume];
    const verb = volume === "whisper" ? "whispers" : volume === "shout" ? "shouts" : "says";
    me.speech = { text, volume, to: target?.id ?? null, until: this.time + 2.5 + text.length * 0.06 };
    this.broadcast(me.pos, Math.max(range, volume === "whisper" ? 6 : range), (l) => {
      const d = dist(l.pos, me.pos);
      if (volume === "whisper" && l !== target) return d <= 6 ? `${me.name} whispers something to ${target?.name ?? "themselves"}.` : null;
      if (d > range) return null;
      if (l === target) { l.addressed = true; return `${me.name} ${verb} to you: "${text}"`; }
      if (target) return `${me.name} ${verb} to ${target.name}: "${text}"`;
      if (text.toLowerCase().includes(l.name.toLowerCase().split(" ")[0])) l.addressed = true;
      return `${me.name} ${verb}: "${text}"`;
    }, id);
    this.emit("say", `${me.name} ${verb}${target ? ` to ${target.name}` : ""}: "${text}"`, id);
    const heardBy = [...this.characters.values()].filter((c) => c !== me && dist(c.pos, me.pos) <= range && (volume !== "whisper" || c === target));
    return { ok: true, message: heardBy.length ? `Heard by: ${heardBy.map((c) => c.name).join(", ")}.` : "Nobody was close enough to hear you." };
  }

  emote(id: string, name: Emote): ActionResult {
    const me = this.characters.get(id)!;
    if (name === "sit") { me.posture = "sitting"; me.target = null; }
    if (name === "stand") me.posture = "standing";
    me.emote = { name, until: this.time + (name === "sit" || name === "stand" ? 0.8 : 2.5) };
    const label = name.replace("_", " ");
    this.broadcast(me.pos, RANGES.sight, () => `${me.name} ${name === "think" ? "looks thoughtful" : `does: ${label}`}.`, id);
    this.emit("emote", `${me.name} *${label}*`, id);
    return { ok: true, message: `You ${label}.` };
  }

  act(id: string, description: string, target: string): ActionResult {
    const me = this.characters.get(id)!;
    let tgt = "";
    if (target.trim() && !["none", "nothing", ""].includes(target.trim().toLowerCase())) {
      const r = this.resolve(target, me);
      if (!r || r.kind === "point") return { ok: false, message: `You can't find "${target}" nearby.` };
      const d = dist(me.pos, this.posOf(r.ent)) - (r.kind === "landmark" ? r.ent.def.radius : 0);
      if (d > RANGES.reach + 1) return { ok: false, message: `The ${r.ent.name} is ${fmt(d)}m away — walk closer first.` };
      const p = this.posOf(r.ent);
      if (r.ent !== me) me.facing = Math.atan2(p.x - me.pos.x, -(p.z - me.pos.z));
      tgt = r.ent.name;
    }
    me.action = { text: description, until: this.time + 4 };
    this.broadcast(me.pos, RANGES.sight, () => `${me.name} ${description}${tgt && !description.toLowerCase().includes(tgt.toLowerCase()) ? ` (${tgt})` : ""}.`, id);
    this.emit("act", `${me.name} ${description}`, id);
    return { ok: true, message: `Done. Others nearby saw you ${description}. (The world doesn't simulate the effects; the narrator or others may react.)` };
  }

  lookAt(id: string, target: string): ActionResult {
    const me = this.characters.get(id)!;
    const r = this.resolve(target, me);
    if (!r) return { ok: false, message: `You don't see "${target}".` };
    const p = r.kind === "point" ? r.pos : this.posOf(r.ent);
    me.facing = Math.atan2(p.x - me.pos.x, -(p.z - me.pos.z));
    if (r.kind === "point") return { ok: true, message: `You look toward (${fmt(p.x)}, ${fmt(p.z)}).` };
    const d = dist(me.pos, p);
    let detail = r.ent.def && "description" in r.ent.def ? (r.ent.def as { description?: string }).description ?? "" : "";
    if (r.kind === "character") {
      const c = r.ent;
      detail = [c.def.appearance, c.holding.length ? `holding ${c.holding.map((i) => this.items.get(i)!.name).join(", ")}` : "", c.posture].filter(Boolean).join("; ");
      if (d <= RANGES.normal) c.inbox.push(`[${fmt(this.time)}s] ${me.name} looks at you.`);
    }
    return { ok: true, message: `You look at ${r.ent.name} (${fmt(d)}m ${bearing(me.pos, p)}). ${detail}`.trim() };
  }

  pickUp(id: string, itemName: string): ActionResult {
    const me = this.characters.get(id)!;
    const r = this.resolve(itemName, me);
    if (!r || r.kind !== "item") return { ok: false, message: `There is no item called "${itemName}".` };
    const it = r.ent;
    if (it.holder === id) return { ok: false, message: `You are already holding the ${it.name}.` };
    if (it.holder) return { ok: false, message: `${this.characters.get(it.holder)!.name} is holding the ${it.name}. You could ask them for it.` };
    const d = dist(me.pos, it.pos);
    if (d > RANGES.reach) return { ok: false, message: `The ${it.name} is ${fmt(d)}m away — walk to it first.` };
    if (me.holding.length >= 2) return { ok: false, message: "Your hands are full (2 items max). Drop or give something first." };
    it.holder = id;
    me.holding.push(it.id);
    this.broadcast(me.pos, RANGES.sight, () => `${me.name} picks up the ${it.name}.`, id);
    this.emit("item", `${me.name} picks up the ${it.name}.`, id);
    return { ok: true, message: `You pick up the ${it.name}.` };
  }

  drop(id: string, itemName: string): ActionResult {
    const me = this.characters.get(id)!;
    const it = me.holding.map((i) => this.items.get(i)!).find((i) => i.name.toLowerCase().includes(itemName.toLowerCase()) || i.id === itemName);
    if (!it) return { ok: false, message: `You aren't holding "${itemName}".` };
    me.holding = me.holding.filter((i) => i !== it.id);
    it.holder = null;
    it.pos = { x: me.pos.x + Math.sin(me.facing) * 0.6, z: me.pos.z - Math.cos(me.facing) * 0.6 };
    this.broadcast(me.pos, RANGES.sight, () => `${me.name} puts down the ${it.name}.`, id);
    this.emit("item", `${me.name} puts down the ${it.name}.`, id);
    return { ok: true, message: `You put down the ${it.name}.` };
  }

  give(id: string, itemName: string, to: string): ActionResult {
    const me = this.characters.get(id)!;
    const it = me.holding.map((i) => this.items.get(i)!).find((i) => i.name.toLowerCase().includes(itemName.toLowerCase()) || i.id === itemName);
    if (!it) return { ok: false, message: `You aren't holding "${itemName}".` };
    const r = this.resolve(to, me);
    if (!r || r.kind !== "character") return { ok: false, message: `"${to}" isn't a person here.` };
    const other = r.ent;
    const d = dist(me.pos, other.pos);
    if (d > RANGES.reach + 0.5) return { ok: false, message: `${other.name} is ${fmt(d)}m away — walk to them first.` };
    if (other.holding.length >= 2) return { ok: false, message: `${other.name}'s hands are full.` };
    me.holding = me.holding.filter((i) => i !== it.id);
    other.holding.push(it.id);
    it.holder = other.id;
    other.addressed = true;
    this.broadcast(me.pos, RANGES.sight, (l) => (l === other ? `${me.name} hands you the ${it.name}.` : `${me.name} hands the ${it.name} to ${other.name}.`), id);
    this.emit("item", `${me.name} gives the ${it.name} to ${other.name}.`, id);
    return { ok: true, message: `You give the ${it.name} to ${other.name}.` };
  }

  wait(id: string, seconds: number): ActionResult {
    const me = this.characters.get(id)!;
    const s = Math.max(1, Math.min(30, seconds));
    me.waitUntil = this.time + s;
    return { ok: true, message: `You wait up to ${fmt(s)}s. You'll be woken early if someone speaks to you.` };
  }

  // ---------------------------------------------------------------- director

  narrate(text: string) {
    this.characters.forEach((c) => { c.inbox.push(`[${fmt(this.time)}s] (Narrator) ${text}`); c.addressed = true; });
    this.emit("narrate", text);
  }

  spawnItem(name: string, pos: Vec, color = "#ffcc00", description = ""): Item {
    const id = `spawn-${++this.spawnCounter}`;
    const def: ItemDef = { id, name, description, position: [pos.x, pos.z], color, shape: "box" };
    const it: Item = { id, name, def, pos: this.clamp(pos), holder: null };
    this.items.set(id, it);
    this.broadcast(it.pos, RANGES.sight, () => `A ${name} appears ${description ? `(${description}) ` : ""}nearby.`);
    this.emit("item", `A ${name} appears at (${fmt(pos.x)}, ${fmt(pos.z)}).`);
    return it;
  }

  // ---------------------------------------------------------------- client snapshot

  snapshot() {
    return {
      time: this.time,
      characters: [...this.characters.values()].map((c) => ({
        id: c.id, name: c.name, x: c.pos.x, z: c.pos.z, facing: c.facing,
        moving: !!c.target, pace: c.target?.pace ?? null,
        emote: c.emote && c.emote.until > this.time ? c.emote.name : null,
        posture: c.posture,
        speech: c.speech ? { text: c.speech.text, volume: c.speech.volume } : null,
        action: c.action ? c.action.text : null,
        holding: c.holding, thinking: c.thinking, control: c.def.control,
      })),
      items: [...this.items.values()].map((i) => ({ id: i.id, x: i.pos.x, z: i.pos.z, holder: i.holder })),
    };
  }

  /** Static scene description the client builds meshes from once. */
  sceneInfo() {
    const s = this.scenario;
    return {
      id: s.id, title: s.title, setting: s.setting, directorNotes: s.directorNotes, worldSize: s.worldSize,
      groundColor: s.groundColor, skyColor: s.skyColor,
      landmarks: s.landmarks,
      items: [...this.items.values()].map((i) => ({ id: i.id, name: i.name, color: i.def.color, shape: i.def.shape })),
      characters: s.characters.map((c) => ({ id: c.id, name: c.name, color: c.color, headColor: c.headColor, height: c.height, control: c.control })),
    };
  }
}
