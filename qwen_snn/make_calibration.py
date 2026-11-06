"""Generate calibration chats with the original model (writes data/calibration_chats.json).

    python make_calibration.py --model /path/to/Qwen3.5-0.8B
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What were the main causes of the French Revolution?",
    "Write a Python function that checks whether a string is a palindrome, with a short explanation.",
    "Give me a recipe for a simple vegetable soup.",
    "What is the difference between weather and climate?",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed?",
    "Describe the water cycle to a ten-year-old.",
    "List five tips for getting better sleep.",
    "用中文介绍一下长城的历史。",
    "Write a haiku about autumn rain, then explain the imagery.",
    "How do vaccines train the immune system?",
    "Compare TCP and UDP in a short table and a paragraph.",
    "What should I consider when adopting a dog?",
    "Translate into French and Spanish: 'The library opens at nine every morning.'",
    "Write a formal email asking a professor for a recommendation letter.",
    "Why is the sky blue?",
    "Explain what a neural network is without using math.",
    "Summarize the plot of a typical detective novel in your own words.",
    "Write a SQL query that finds the top three customers by total order value, and explain it.",
    "What are black holes and how do they form?",
    "Give arguments for and against working from home.",
    "Explain recursion with an example in JavaScript.",
    "Write a limerick about a cat who learned to code.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration_chats.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    chats = []
    for i in range(0, len(PROMPTS), 8):
        batch = [[{"role": "user", "content": p}] for p in PROMPTS[i:i + 8]]
        enc = tok.apply_chat_template(batch, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True, padding=True)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9)
        for msgs, row in zip(batch, out):
            reply = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            chats.append(msgs + [{"role": "assistant", "content": reply}])
            print("---", msgs[0]["content"][:60], "->", reply[:80].replace("\n", " "), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(chats, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
