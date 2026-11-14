import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { claudeBrain, claudeDefaults } from "./brains/claude.js";
import { mockBrain } from "./brains/mock.js";
import type { BrainFactory } from "./brains/types.js";
import { listScenarios } from "./scenario.js";
import { Simulation, defaultSimOptions } from "./simulation.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "..");
const PUBLIC = path.join(ROOT, "public");
const THREE = path.join(ROOT, "node_modules", "three");
const PORT = Number(process.env.PORT || 3000);

type BrainKind = "claude" | "mock";

function defaultBrain(): BrainKind {
  const env = process.env.RP_BRAIN as BrainKind | undefined;
  if (env === "claude" || env === "mock") return env;
  const hasCreds = !!(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN) ||
    fs.existsSync(path.join(os.homedir(), ".config", "anthropic"));
  return hasCreds ? "claude" : "mock";
}

function makeBrain(kind: BrainKind): [BrainFactory, string] {
  if (kind === "mock") return [mockBrain(), "mock (offline)"];
  const o = claudeDefaults();
  return [claudeBrain(), `${o.model} · effort ${o.effort}`];
}

let brainKind = defaultBrain();
let sim = createSim(process.env.RP_SCENARIO || "village-festival");

function createSim(scenarioId: string) {
  const [factory, label] = makeBrain(brainKind);
  const s = Simulation.fromScenarioId(scenarioId, factory, label, defaultSimOptions());
  s.on((ev, data) => broadcast(ev, data));
  return s;
}

// ---------------------------------------------------------------- SSE

const clients = new Set<http.ServerResponse>();
function send(res: http.ServerResponse, ev: string, data: unknown) {
  res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`);
}
function broadcast(ev: string, data: unknown) { for (const c of clients) send(c, ev, data); }
function hello(res: http.ServerResponse) {
  send(res, "scene", sim.world.sceneInfo());
  send(res, "status", sim.status());
  send(res, "backlog", sim.world.log.slice(-300));
}

// ---------------------------------------------------------------- loop

const DT = 0.1;
setInterval(() => {
  sim.tick(DT);
  if (clients.size) broadcast("state", sim.world.snapshot());
}, DT * 1000);

// ---------------------------------------------------------------- http

const MIME: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml" };

function serveFile(res: http.ServerResponse, base: string, rel: string) {
  const file = path.resolve(base, "." + path.posix.normalize("/" + rel));
  if (!file.startsWith(base + path.sep) && file !== base) return notFound(res);
  fs.readFile(file, (err, buf) => {
    if (err) return notFound(res);
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(buf);
  });
}
const notFound = (res: http.ServerResponse) => { res.writeHead(404); res.end("not found"); };
const json = (res: http.ServerResponse, code: number, body: unknown) => {
  res.writeHead(code, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};

async function readBody(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  let raw = "";
  for await (const chunk of req) { raw += chunk; if (raw.length > 1e5) throw new Error("body too large"); }
  return raw ? JSON.parse(raw) : {};
}

const str = (v: unknown, name: string) => {
  if (typeof v !== "string" || !v.trim()) throw new Error(`${name} is required`);
  return v.trim();
};

const routes: Record<string, (b: Record<string, unknown>) => unknown> = {
  "/api/play": () => { sim.play(); return sim.status(); },
  "/api/pause": () => { sim.pause(); return sim.status(); },
  "/api/step": () => { sim.step(); return sim.status(); },
  "/api/settings": (b) => {
    if (typeof b.lockstep === "boolean") sim.opts.lockstep = b.lockstep;
    if (typeof b.timeScale === "number") sim.opts.timeScale = Math.max(0.25, Math.min(4, b.timeScale));
    if (typeof b.maxTurns === "number") sim.opts.maxTurns = Math.max(0, b.maxTurns);
    broadcast("status", sim.status());
    return sim.status();
  },
  "/api/reset": (b) => {
    if (b.brain === "claude" || b.brain === "mock") brainKind = b.brain;
    const id = typeof b.scenario === "string" ? b.scenario : sim.scenario.id;
    const next = createSim(id); // throws on a bad scenario, keeping the old sim
    sim.dispose();
    sim = next;
    for (const c of clients) hello(c);
    return sim.status();
  },
  "/api/narrate": (b) => { sim.narrate(str(b.text, "text")); return { ok: true }; },
  "/api/direct": (b) => { sim.direct(str(b.id, "id"), str(b.text, "text")); return { ok: true }; },
  "/api/spawn": (b) => {
    const it = sim.world.spawnItem(str(b.name, "name"), { x: Number(b.x) || 0, z: Number(b.z) || 0 }, typeof b.color === "string" ? b.color : undefined, typeof b.description === "string" ? b.description : "");
    broadcast("scene", sim.world.sceneInfo());
    return { ok: true, id: it.id };
  },
  "/api/control": (b) => { sim.setControl(str(b.id, "id"), b.control === "player" ? "player" : "llm"); return { ok: true }; },
  "/api/act": (b) => sim.playerAction(str(b.id, "id"), str(b.tool, "tool"), b.input ?? {}),
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  const p = url.pathname;
  try {
    if (req.method === "GET" && p === "/api/events") {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
      clients.add(res);
      hello(res);
      req.on("close", () => clients.delete(res));
      return;
    }
    if (req.method === "GET" && p === "/api/scenarios") return json(res, 200, { scenarios: listScenarios(), current: sim.scenario.id, brain: brainKind });
    if (req.method === "GET" && p.startsWith("/api/agent/")) {
      const id = decodeURIComponent(p.slice("/api/agent/".length));
      if (!sim.agents.has(id)) return json(res, 404, { error: "no such character" });
      return json(res, 200, sim.agentInfo(id));
    }
    if (req.method === "POST" && routes[p]) return json(res, 200, routes[p](await readBody(req)) ?? { ok: true });
    if (req.method === "GET" && p.startsWith("/vendor/three/")) return serveFile(res, THREE, p.slice("/vendor/three/".length));
    if (req.method === "GET") return serveFile(res, PUBLIC, p === "/" ? "index.html" : p.slice(1));
    notFound(res);
  } catch (err) {
    json(res, 400, { error: err instanceof Error ? err.message : String(err) });
  }
});

server.listen(PORT, () => {
  console.log(`Roleplay 3D harness on http://localhost:${PORT}`);
  console.log(`Scenario: ${sim.scenario.title} · Brain: ${sim.brainName}`);
  if (brainKind === "mock") console.log("No Anthropic credentials found — using the offline mock brain. Set ANTHROPIC_API_KEY (or RP_BRAIN=claude) to use Claude.");
});
