import type { Character, World } from "../world.js";

/** A tool call a brain wants to make this turn. */
export interface ToolCall { id: string; name: string; input: unknown }
/** What happened when the harness ran a tool call. */
export interface ToolOutcome { id: string; name: string; ok: boolean; message: string }

export interface TurnInput {
  /** Fresh text observation of the world from this character's point of view. */
  observation: string;
  /** Results of the calls the brain made on its previous turn. */
  results: ToolOutcome[];
  /** Out-of-character stage directions from the director, if any. */
  directions: string[];
}

export interface Usage {
  input: number; output: number; cacheRead: number; cacheWrite: number; costUsd: number | null;
}

export interface TurnOutput {
  calls: ToolCall[];
  /** Plain text the model wrote outside tools: private inner monologue, never shown to others. */
  monologue?: string;
  /** Summarized reasoning, when the brain exposes it. */
  thinking?: string;
  usage?: Usage;
  stopReason?: string;
  model?: string;
  /** True when the brain discarded this turn; the harness re-sends the same results next time. */
  rolledBack?: boolean;
}

export interface Brain {
  readonly label: string;
  turn(input: TurnInput): Promise<TurnOutput>;
}

export interface BrainContext { world: World; character: Character; systemPrompt: string }
export type BrainFactory = (ctx: BrainContext) => Brain;
