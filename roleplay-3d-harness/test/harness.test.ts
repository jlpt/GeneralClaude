import assert from "node:assert/strict";
import http from "node:http";
import { after, before, test } from "node:test";
import { loadScenario } from "../src/scenario.js";
import { World } from "../src/world.js";
import { runTool, toolDefinitions } from "../src/tools.js";
import { buildSystemPrompt } from "../src/prompt.js";

// ---------------------------------------------------------------- world rules

test("speech is only heard within range", () => {
  const w = new World(loadScenario("village-festival"));
  const res = w.say("mara", "Has anyone seen the key?", "normal", "everyone");
  assert.ok(res.ok);
  const heard = (id: string) => w.characters.get(id)!.inbox.some((l) => l.includes("seen the key"));
  assert.ok(heard("traveler")); // ~5m away
  assert.ok(heard("selene"));   // ~11m, inside normal range (12m)
  w.characters.get("bram")!.pos = { x: -20, z: -20 };
  w.say("mara", "Anyone?", "normal", "everyone");
  assert.ok(!w.characters.get("bram")!.inbox.some((l) => l.includes("Anyone?"))); // ~29m away
  const whisper = w.say("mara", "psst", "whisper", "Pip");
  assert.match(whisper.message, /Nobody/);      // Pip is far away
});

test("items need to be within reach and can be handed over", () => {
  const w = new World(loadScenario("village-festival"));
  assert.equal(runTool(w, "selene", "pick_up", { item: "paper lantern" }).ok, false); // 3m away
  runTool(w, "selene", "walk_to", { destination: "paper lantern", pace: "walk" });
  for (let i = 0; i < 100; i++) w.step(0.1);
  assert.ok(runTool(w, "selene", "pick_up", { item: "lantern" }).ok);
  assert.equal(runTool(w, "selene", "give", { item: "lantern", to: "Pip" }).ok, false); // too far
  assert.equal(runTool(w, "selene", "teleport", {}).ok, false);
  assert.equal(runTool(w, "selene", "say", { text: "" , to: "everyone", volume: "normal" }).ok, false);
});

test("tool schemas are strict-compatible", () => {
  for (const t of toolDefinitions()) {
    assert.equal(t.input_schema.type, "object");
    assert.equal(t.input_schema.additionalProperties, false);
    assert.ok(!JSON.stringify(t.input_schema).includes("Length"));
  }
});

// ---------------------------------------------------------------- Claude brain contract (fake API)

const requests: any[] = [];
const replies: any[] = [];
let server: http.Server;

before(async () => {
  server = http.createServer(async (req, res) => {
    let body = "";
    for await (const c of req) body += c;
    requests.push({ url: req.url, headers: req.headers, body: JSON.parse(body) });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(replies.shift()));
  });
  await new Promise<void>((r) => server.listen(0, r));
  const port = (server.address() as any).port;
  process.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${port}`;
  process.env.ANTHROPIC_API_KEY = "test-key";
  delete process.env.ANTHROPIC_AUTH_TOKEN;
});
after(() => server.close());

const msg = (content: any[], stop_reason = "tool_use") => ({
  id: "msg_" + Math.random(), type: "message", role: "assistant", model: "claude-opus-5-5", content, stop_reason,
  stop_sequence: null, usage: { input_tokens: 100, output_tokens: 20, cache_read_input_tokens: 900, cache_creation_input_tokens: 0 },
});

test("ClaudeBrain builds an append-only conversation", async () => {
  const { ClaudeBrain } = await import("../src/brains/claude.js");
  const w = new World(loadScenario("village-festival"));
  const c = w.characters.get("mara")!;
  const brain = new ClaudeBrain({ world: w, character: c, systemPrompt: buildSystemPrompt(w, c) },
    { model: "claude-opus-5-5", effort: "low", maxTokens: 16000 });

  replies.push(msg([
    { type: "thinking", thinking: "Bram looks shifty.", signature: "sig" },
    { type: "text", text: "I should ask around." },
    { type: "tool_use", id: "tu_1", name: "say", input: { text: "Has anyone seen the key?", to: "everyone", volume: "shout" } },
    { type: "tool_use", id: "tu_2", name: "walk_to", input: { destination: "Old Bram", pace: "walk" } },
  ]));
  const out1 = await brain.turn({ observation: "OBS1", results: [], directions: [] });
  assert.deepEqual(out1.calls.map((x) => x.name), ["say", "walk_to"]);
  assert.equal(out1.monologue, "I should ask around.");
  assert.equal(out1.thinking, "Bram looks shifty.");
  assert.ok(out1.usage!.costUsd! > 0);

  const r1 = requests[0];
  assert.equal(r1.body.model, "claude-opus-5-5");
  assert.equal(r1.body.fallbacks, "default");
  assert.deepEqual(r1.body.thinking, { type: "adaptive", display: "summarized" });
  assert.equal(r1.body.output_config.effort, "low");
  assert.deepEqual(r1.body.tool_choice, { type: "auto" });
  assert.match(r1.headers["anthropic-beta"], /compact-2026-01-12/);
  assert.match(r1.headers["anthropic-beta"], /server-side-fallback-2026-07-01/);
  assert.ok(r1.body.tools.every((t: any) => t.strict === true));

  replies.push(msg([{ type: "text", text: "Waiting." }], "end_turn"));
  await brain.turn({
    observation: "OBS2",
    results: [{ id: "tu_1", name: "say", ok: true, message: "Heard by: Pip" }, { id: "tu_2", name: "walk_to", ok: false, message: "nope" }],
    directions: ["Be more suspicious."],
  });
  const m = requests[1].body.messages;
  assert.deepEqual(m.map((x: any) => x.role), ["user", "assistant", "user", "system"]);
  assert.equal(m[1].content[0].type, "thinking");             // thinking replayed unchanged
  assert.equal(m[1].content[0].signature, "sig");
  assert.deepEqual(m[2].content.map((b: any) => b.type), ["tool_result", "tool_result", "text"]);
  assert.equal(m[2].content[1].is_error, true);
  assert.equal(m[3].content, "Director: Be more suspicious.");
  assert.equal(requests[1].body.system, requests[0].body.system); // stable, cacheable prefix

  // A refusal rolls the turn back so the next request doesn't carry it.
  replies.push({ ...msg([], "refusal"), stop_details: { type: "refusal", category: null, explanation: "" } });
  const out3 = await brain.turn({ observation: "OBS3", results: [], directions: [] });
  assert.equal(out3.rolledBack, true);
  replies.push(msg([{ type: "text", text: "ok" }], "end_turn"));
  await brain.turn({ observation: "OBS4", results: [], directions: [] });
  const roles = requests[3].body.messages.map((x: any) => x.role);
  assert.deepEqual(roles, ["user", "assistant", "user", "system", "assistant", "user"]);
  assert.equal(requests[3].body.messages.at(-1).content.at(-1).text, "OBS4");
});
