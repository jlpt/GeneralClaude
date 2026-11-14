import Anthropic from "@anthropic-ai/sdk";
import { toolDefinitions } from "../tools.js";
import type { Brain, BrainContext, TurnInput, TurnOutput, Usage } from "./types.js";

type Effort = "low" | "medium" | "high" | "xhigh" | "max";

export interface ClaudeBrainOptions {
  model: string;
  effort: Effort;
  maxTokens: number;
}

export const claudeDefaults = (): ClaudeBrainOptions => ({
  model: process.env.RP_MODEL || "claude-opus-5-5",
  effort: (process.env.RP_EFFORT as Effort) || "low",
  maxTokens: Number(process.env.RP_MAX_TOKENS || 16000),
});

// $ per million tokens: [input, output, cache read]; cache writes bill at 1.25x input.
const PRICES: Record<string, [number, number, number]> = {
  "claude-opus-5-5": [4, 20, 0.2],
  "claude-sonnet-5-5": [2, 10, 0.1],
  "claude-haiku-5-5": [0.1, 0.5, 0.01],
  "claude-fable-5-2": [8, 40, 0.2],
};

// Models that accept `role: "system"` messages mid-conversation. Elsewhere we
// fall back to putting director notes in the user turn.
const SUPPORTS_MID_SYSTEM = /opus-5|opus-4-8|fable|mythos|sonnet-5-5|haiku-5-5/;
// Models that take the server-side refusal fallback parameter.
const SUPPORTS_FALLBACKS = /^claude-(opus-5-5|opus-5|fable-5-[12]|sonnet-5-5)$/;

let sharedClient: Anthropic | null = null;
const client = () => (sharedClient ??= new Anthropic());

/**
 * One Claude conversation per character. The history is append-only: each
 * turn adds a user message (tool results for last turn + the new observation),
 * an optional director system message, and the assistant reply. That keeps
 * the prompt cache warm and replays thinking blocks unchanged. Server-side
 * compaction summarizes the history if a long scene approaches the limit.
 */
export class ClaudeBrain implements Brain {
  readonly label: string;
  private messages: Anthropic.Beta.BetaMessageParam[] = [];
  private readonly tools = toolDefinitions();
  private readonly betas: Anthropic.Beta.AnthropicBeta[];

  constructor(private ctx: BrainContext, private opts: ClaudeBrainOptions = claudeDefaults()) {
    this.label = `${opts.model} (effort ${opts.effort})`;
    this.betas = ["compact-2026-01-12"];
    if (SUPPORTS_FALLBACKS.test(opts.model)) this.betas.push("server-side-fallback-2026-07-01");
  }

  async turn(input: TurnInput): Promise<TurnOutput> {
    const content: Anthropic.Beta.BetaContentBlockParam[] = input.results.map((r) => ({
      type: "tool_result" as const,
      tool_use_id: r.id,
      content: r.message,
      ...(r.ok ? {} : { is_error: true }),
    }));
    const midSystem = SUPPORTS_MID_SYSTEM.test(this.opts.model);
    const directionText = input.directions.map((d) => `Director: ${d}`).join("\n");
    content.push({
      type: "text",
      text: input.observation + (directionText && !midSystem ? `\n\n(Out-of-character stage direction)\n${directionText}` : ""),
    });

    const added = 1 + (directionText && midSystem ? 1 : 0);
    this.messages.push({ role: "user", content });
    if (directionText && midSystem) this.messages.push({ role: "system", content: directionText });

    let response: Anthropic.Beta.BetaMessage;
    try {
      response = await client().beta.messages.create({
        model: this.opts.model,
        max_tokens: this.opts.maxTokens,
        betas: this.betas,
        ...(SUPPORTS_FALLBACKS.test(this.opts.model) ? { fallbacks: "default" as const } : {}),
        system: this.ctx.systemPrompt,
        tools: this.tools,
        tool_choice: { type: "auto" },
        messages: this.messages,
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort: this.opts.effort },
        cache_control: { type: "ephemeral" },
        context_management: { edits: [{ type: "compact_20260112" }] },
      });
    } catch (err) {
      // Nothing was appended on the model's side; drop our unsent tail so the
      // harness can retry this turn with the same results.
      this.messages.splice(this.messages.length - added, added);
      throw err;
    }

    const usage = this.usage(response);
    if (response.stop_reason === "refusal" || response.stop_reason === "max_tokens") {
      // Don't act on a cut-off turn. Roll back so the next turn starts clean.
      this.messages.splice(this.messages.length - added, added);
      return { calls: [], rolledBack: true, stopReason: response.stop_reason, usage, model: response.model,
        monologue: response.stop_reason === "refusal" ? `(declined: ${response.stop_details?.category ?? "unspecified"})` : "(turn cut off at max_tokens)" };
    }

    // Append the full content (thinking, compaction, tool_use blocks) unchanged.
    this.messages.push({ role: "assistant", content: response.content });

    const out: TurnOutput = { calls: [], usage, stopReason: response.stop_reason ?? undefined, model: response.model };
    const mono: string[] = [];
    const thoughts: string[] = [];
    for (const block of response.content) {
      if (block.type === "tool_use") out.calls.push({ id: block.id, name: block.name, input: block.input });
      else if (block.type === "text" && block.text.trim()) mono.push(block.text.trim());
      else if (block.type === "thinking" && block.thinking.trim()) thoughts.push(block.thinking.trim());
    }
    if (mono.length) out.monologue = mono.join("\n");
    if (thoughts.length) out.thinking = thoughts.join("\n");
    return out;
  }

  private usage(r: Anthropic.Beta.BetaMessage): Usage {
    const u = r.usage;
    const input = u.input_tokens ?? 0;
    const output = u.output_tokens ?? 0;
    const cacheRead = u.cache_read_input_tokens ?? 0;
    const cacheWrite = u.cache_creation_input_tokens ?? 0;
    const p = PRICES[r.model] ?? PRICES[this.opts.model];
    const costUsd = p ? (input * p[0] + cacheWrite * p[0] * 1.25 + cacheRead * p[2] + output * p[1]) / 1e6 : null;
    return { input, output, cacheRead, cacheWrite, costUsd };
  }
}

export const claudeBrain = (opts?: Partial<ClaudeBrainOptions>) => (ctx: BrainContext) =>
  new ClaudeBrain(ctx, { ...claudeDefaults(), ...opts });
