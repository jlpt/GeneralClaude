import type { Brain, BrainContext, TurnInput, TurnOutput } from "./types.js";

/**
 * A scripted stand-in brain so the harness runs without an API key: it
 * replies when spoken to, wanders between landmarks, and uses canned lines.
 * Handy for testing scenarios, the renderer, and your own tool changes.
 */
export class MockBrain implements Brain {
  readonly label = "mock (offline)";
  private n = 0;
  private lines: string[];

  constructor(private ctx: BrainContext) {
    this.lines = ctx.character.def.mockLines.length
      ? ctx.character.def.mockLines
      : ["What a day.", "Has anyone seen anything strange?", "I could use a rest.", "Lovely weather, isn't it?"];
  }

  private id() { return `mock_${this.ctx.character.id}_${++this.n}`; }
  private pick<T>(a: T[]): T { return a[Math.floor(Math.random() * a.length)]; }

  async turn(input: TurnInput): Promise<TurnOutput> {
    await new Promise((r) => setTimeout(r, 150 + Math.random() * 350)); // pretend latency
    const { world, character: me } = this.ctx;
    const addressed = input.observation.match(/\] (.+?) (?:says|shouts|whispers) to you: "(.*)"/);
    if (addressed && Math.random() < 0.6) {
      const [, who, what] = addressed;
      const reply = what.endsWith("?")
        ? this.pick(["Hmm, good question.", "I'm not sure, honestly.", "Ask me again later.", "Maybe!"])
        : this.pick(this.lines);
      return {
        monologue: `${who} spoke to me. I'll answer.`,
        calls: [
          { id: this.id(), name: "look_at", input: { target: who } },
          { id: this.id(), name: "say", input: { text: reply, to: who, volume: "normal" } },
        ],
      };
    }
    const r = Math.random();
    const landmarks = [...world.landmarks.values()];
    const others = [...world.characters.values()].filter((c) => c !== me);
    if (r < 0.35 && landmarks.length) {
      return { calls: [{ id: this.id(), name: "walk_to", input: { destination: this.pick(landmarks).name, pace: Math.random() < 0.2 ? "run" : "walk" } }] };
    }
    if (r < 0.6 && others.length) {
      const o = this.pick(others);
      return { calls: [{ id: this.id(), name: "say", input: { text: this.pick(this.lines), to: Math.random() < 0.5 ? o.name : "everyone", volume: "normal" } }] };
    }
    if (r < 0.75) {
      return { calls: [{ id: this.id(), name: "emote", input: { animation: this.pick(["wave", "nod", "laugh", "shrug", "think", "dance", "cheer"]) } }] };
    }
    if (r < 0.85 && others.length) {
      return { calls: [{ id: this.id(), name: "walk_to", input: { destination: this.pick(others).name, pace: "walk" } }] };
    }
    return { calls: [{ id: this.id(), name: "wait", input: { seconds: 3 + Math.random() * 5 } }] };
  }
}

export const mockBrain = () => (ctx: BrainContext) => new MockBrain(ctx);
