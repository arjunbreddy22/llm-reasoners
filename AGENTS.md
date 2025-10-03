# AGENTS.md

This repository is a local working copy of the llm-reasoners project. It provides a common framework to run and compare step‑by‑step reasoning algorithms with LLMs, such as Tree‑of‑Thought (ToT), Monte‑Carlo Tree Search (MCTS), Reasoning‑via‑Planning (RAP), Chain‑of‑Thought (CoT), beam/greedy search variants, and a visualization stack for inspecting search traces. Backends include SGLang, HuggingFace Transformers, Llama (2/3), llama.cpp, OpenAI‑style APIs, etc.

## Repo Structure (high‑level)
- `llm-reasoners/`
  - `reasoners/`: Core abstractions and algorithms (WorldModel, SearchConfig, Reasoner; DFS/Beam/MCTS/etc.; visualization utilities).
  - `examples/`: Reproducible examples for multiple tasks and algorithms (e.g., ToT, RAP, CoT, SGL, ReasonerAgent). Our current focus lives here.
  - `assets/`, packaging files, and project READMEs.

## Current Focus: Game24 with MCTS over ToT (Qwen)
Path: `llm-reasoners/examples/ToT/game24_mcts_qwen3`

Goal: make MCTS‑over‑ToT for the Game24 puzzle robust when using Qwen models via SGLang, while keeping validations strict and honest.

### Problems Observed
- Qwen3 tends to output analytical “thinking” before (or instead of) structured actions, which previously led to:
  - Parsing failures (no action lines extracted),
  - Illegal actions (operations referencing numbers not in the current state),
  - State corruption (malformed or incorrect `(left: ...)`),
  - Reward text that didn’t end with strict labels.
- CoT baseline returned `<think>` or long prose without a final, valid “(EXPRESSION) = 24” line; naive parsing marked everything incorrect.

### What We Implemented
Action extraction and MCTS stability
- Strip `<think>` blocks before parsing candidate lines.
- Non‑anchored equation matching to catch numbered/bulleted lines.
- Minimal validator for each proposed action:
  - Both operands must be present in the current multiset (duplicate‑aware),
  - Compute `C = A op B` and rebuild `(left: ...)` deterministically,
  - Enforce left size = k−1 for any k.
- Dynamic proposal constraints: require “left: exactly k−1 numbers …” with a tiny, state‑consistent example.
- Keep logits‑style reward for SGLang via `get_loglikelihood(..., [sure, likely, impossible])` (no sampling fallback), and cache values.

CoT baseline robustness
- Removed newline‑based early stopping and added tolerant post‑processing:
  - Strip `<think>`, search globally for any “expression = 24” (incl. `= 24.0`/`= 24.` and cases like `expr1 = expr2 = 24`),
  - Extract the math subexpression and balance unmatched parentheses before evaluation,
  - Evaluate numerically (via `sympy.simplify`) and require exact multiset match with the four inputs (decimal‑aware, duplicates respected),
  - If nothing matches, optionally run a minimal second pass requesting a single, final line; still validated strictly.

General guardrails
- No stop triggers on “= 24” to avoid prematurely cutting mid‑analysis mentions.
- Validations are strict: a candidate must (1) evaluate to 24 (within tolerance), and (2) use each input number exactly once.

### How To Run (quick)
- CoT baseline (SGLang, Qwen2.5):
  ```bash
  python llm-reasoners/examples/ToT/game24_mcts_qwen3/cot.py \
    --base_lm sglang --sglang_model Qwen/Qwen2.5-7B-Instruct \
    --sglang_url http://127.0.0.1:30001
  ```

- MCTS over ToT (SGLang, Qwen2.5/3):
  ```bash
  python llm-reasoners/examples/ToT/game24_mcts_qwen3/inference.py \
    --base_lm sglang --sglang_model Qwen/Qwen2.5-7B-Instruct \
    --sglang_url http://127.0.0.1:30001
  ```
  Notes:
  - Terminal correctness is logged and can be post‑hoc verified with the same robust validator logic.
  - For Qwen3 (which prefers analytic prose), the dynamic left‑size constraint and strict proposal validation help avoid malformed moves.

## Status and Next Steps
What’s working now
- CoT can correctly recognize valid final expressions embedded in prose (incl. `expr1 = expr2 = 24` style), and MCTS proposal extraction is robust against malformed actions and state corruption.
- Reward evaluation avoids brittle free‑form parsing using SGLang’s choice‑scoring.

Remaining gaps / optional improvements
- Qwen3 sometimes under‑produces actions at k=2; a bounded retry with slightly larger token budget (only when actions < N) can help without changing core behavior.
- Consider trimming long static header examples in proposal prompts to reduce echoing and keep the model focused on the current state.
- If desired, add a “Final:” sentinel line in CoT prompting to make the final answer easier to extract without stop triggers.

## Principles (for future changes)
- Keep validations strict and honest; never accept partial solutions (e.g., `6*4=24`) as correct.
- Prefer post‑hoc correctness checks over intrusive decoding constraints.
- Contain prompt engineering changes to example code; don’t weaken algorithmic guarantees in the core library.

