# Implementation Details for Reproducibility

This note documents the standalone DRG -> SC-2 runner used for the current reviewer-control experiments. The active implementation is the top-level runner at `standalone/deployable_drg_sc2/run_pipeline.py`; the nested `standalone/deployable_drg_sc2/deployable_drg_sc2/` copy is retained for portability but may lag behind the active file.

## Prompts

All math-style runs use Hugging Face `tokenizer.apply_chat_template(...)` with `add_generation_prompt=True`. For Qwen3 thinking runs, pass `--enable_thinking true`; otherwise the tokenizer default is left unchanged with `--enable_thinking auto`.

### Baseline B0 Prompt

Logical messages:

```python
[
  {
    "role": "system",
    "content": "You are a helpful math assistant. Think step by step and give the final answer in \\boxed{}."
  },
  {
    "role": "user",
    "content": problem_text
  }
]
```

Generation config:

```text
do_sample = False
temperature = None
top_p = None
max_new_tokens = dataset budget
```

### Retry Prompt

The default current headline setting is previous-attempt-only retry:

```text
retry_prompt_mode = user
retry_instruction_mode = none
retry_previous_attempt_chars = 1200
retry_previous_attempt_variant = real
retry_temperature = 0.7
retry_top_p = 0.95
```

The retry user message is exactly:

```text
{problem_text}

Previous attempt (may be flawed):
{last_1200_characters_of_baseline_text}
```

No additional instruction string is inserted in this default condition. If `--retry_instruction_mode restart` is used, this exact string is inserted between the problem and previous-attempt block:

```text
Restart from scratch. Try again and give your final answer.
```

For the clean T=0.7 control, set `--retry_previous_attempt_chars 0` and `--retry_instruction_mode none`; then the retry user message is only `{problem_text}`.

### SC-2 Prompt

SC-2 uses the same logical prompt as B0, with two independent sampled generations:

```text
do_sample = True
temperature = 0.7
top_p = 0.95
k = 2
```

SC-2 answer selection is:

```text
if both answers extractable and equivalent: return sample 1 answer
if both answers extractable and disagree: return sample 1 answer
if only sample 1 extractable: return sample 1 answer
if only sample 2 extractable: return sample 2 answer
if neither extractable: return no-answer
```

SC-8 uses the same prompt and sampling settings with `k=8`, then clusters answers by mathematical equivalence and returns the largest cluster, tie-broken by earliest sample.

## Deploy Routing Logic

For each problem:

```text
1. Generate greedy baseline B0.
2. Compute trigger signals on B0:
   repetition hit OR length hit OR stall hit.
3. If not triggered:
   if baseline answer extractable: path_A, return baseline
   else: path_C_nontriggered, use SC-2 fallback
4. If triggered:
   generate sampled retry at T=0.7 using the retry prompt above.
5. Gate:
   if retry answer agrees with baseline answer:
      path_B_agree, return baseline
   else if high pathology:
      if retry answer extractable: path_B_accept, return retry
      else: path_C_triggered, use SC-2 fallback
   else:
      if baseline answer extractable: path_B_keep, return baseline
      else: path_C_triggered, use SC-2 fallback
6. SC-2 fallback:
   use the SC-2 answer-selection rule above.
```

The current default gate is `--gate_policy pathology`, where:

```text
high_pathology = trigger_signal_count >= 2
```

The trigger itself fires when any one signal is active:

```text
triggered = repetition_hit OR length_hit OR stall_hit
```

## Trigger Definitions

### Unigram Repetition

Unigram repetition is whitespace based, not tokenizer based:

```python
toks = [tok for tok in re.split(r"\s+", text.strip()) if tok]
repetition_unigram = 1.0 - len(set(toks)) / len(toks)
```

Default threshold:

```text
repetition_hit = repetition_unigram >= 0.7
```

### Length Trigger

For each dataset/run, compute the baseline token-count distribution and use the 85th percentile:

```text
length_hit = baseline_tokens >= percentile(baseline_token_counts, 85)
```

For fixed-budget stress runs, this often equals the generation cap.

### Stall Detector

Default `k=4`. Pseudocode:

```python
lines = [line.strip() for line in text.splitlines() if line.strip()]
if len(lines) < k + 1:
    return False

seen = set()
new_flags = []
for line in lines:
    terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", line))
    new_flags.append(bool(terms - seen))
    seen |= terms

stall_hit = not any(new_flags[-k:])
```

Here a "term" is any alphabetic identifier matching `[A-Za-z][A-Za-z0-9_]*`. This is intentionally simple: it does not parse LaTeX or distinguish math variables from English words.

### Math-Symbol Removal Control

The `retry_previous_attempt_variant=no_math_symbols` mechanism control removes symbol-heavy content with:

```text
$...$ spans
LaTeX commands matching \\[A-Za-z]+(?:\{[^{}]*\})*
characters in [0-9=+\-*/^_<>|()[\]{}]
```

Whitespace is then collapsed. This control is separate from the stall detector.

## Answer Extraction and Scoring

Answer extraction first discards model-internal thinking when present:

```python
if "</think>" in text:
    visible_text = text.rsplit("</think>", 1)[-1]
else:
    visible_text = text
```

For math-style datasets, the extractor then scans the visible text for the last balanced `\boxed{...}` expression and returns its contents. If no boxed expression is found, the prediction is `None`.

For GSM8K, the extractor first uses the last `\boxed{...}` if present, otherwise it accepts `#### number`.

For GPQA-style multiple choice, the extractor accepts `\boxed{A}` through `\boxed{D}` or the last textual pattern like `answer: A`, `final answer: A`, or `choice A`.

For math equivalence:

1. Normalize both prediction and gold with `normalize_math_answer`.
2. If normalized strings match exactly, score correct.
3. Otherwise call symbolic equivalence with a 5-second alarm timeout.

Import order for symbolic equivalence:

```text
1. skythought.evals.util.math_parsing_util.math_equal
2. standalone/deployable_drg_sc2/symbolic_math.py
3. standalone package fallback .symbolic_math
4. experiments/convergence/WordSaladChopper/wscgen/utils.math_equal
```

The local fallback uses `sympy`, `latex2sympy2`, `sympy.parsing.latex.parse_latex`, and `sympy.parsing.sympy_parser.parse_expr`. In the current pod:

```text
torch = 2.5.1+cu124
transformers = 5.2.0
datasets = 3.6.0
accelerate = 1.13.0
sympy = 1.13.1
latex2sympy2 = installed, package __version__ unavailable
skythought = installed, package __version__ unavailable
```

## Models and Checkpoints

Use the paper-facing model label in tables, and keep the exact checkpoint/local path in provenance. For Ministral, the official Mistral model-card/product name is **Ministral 3 14B** with served model id `ministral-14b-2512`; our experiments used the official Hugging Face reasoning checkpoint under that family, `mistralai/Ministral-3-14B-Reasoning-2512`.

Current local model directories:

| Paper label | Official / checkpoint id | Local path | Revision |
| --- | --- | --- | --- |
| Qwen3-8B | `Qwen/Qwen3-8B` | `base_llms/Qwen3-8B` | local snapshot; no explicit revision pinned in current download script |
| Qwen3-14B | `Qwen/Qwen3-14B` | `base_llms/Qwen3-14B` | local snapshot; no explicit revision pinned in current download script |
| DS-R1-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `base_llms/DeepSeek-R1-Distill-Qwen-7B` | local snapshot; no explicit revision pinned in current download script |
| DS-R1-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `base_llms/DeepSeek-R1-Distill-Llama-8B` | local snapshot; no explicit revision pinned in current download script |
| Ministral 3 14B | official/API id `ministral-14b-2512`; HF checkpoint `mistralai/Ministral-3-14B-Reasoning-2512` | `base_llms/Ministral-3-14B-Reasoning-2512` | local snapshot; no explicit revision pinned in current download script |

## Datasets

| Paper label | Local path | Problem count in current artifacts | Primary budget(s) used |
| --- | --- | ---: | --- |
| MATH-500 | `datasets/math500` | 500 | 4096 |
| AIME 2024 | `datasets/aime_2024` | 30 | 4096, 32768 |
| AMC | `datasets/amc` | 40 | 4096, 32768 |
| GSM8K | `datasets/gsm8k` | 8792 | 4096-style guard run |
| GPQA Diamond | `datasets/gpqa_diamond` | 198 | 32768 |
| OlympiadBench Physics OE-to-EN | `datasets/olympiadbench_physics_oe_to_en` | 236 | 4096 |

For final release, include either the exact Hugging Face commit hashes or checksums for each local model snapshot. The current `download_model.py` downloads from the active Hugging Face revision unless the local directory is already populated.

## Random Seeds

The CLI seed is `--seed`. Main single-seed runs used `--seed 0`. The seed-variance batch in `outputs_standalone/deployable_drg_sc2/seed_results/` uses retry seeds `1, 2, 3, 4, 5`.

Per-generation seeds are deterministic functions of base seed, problem id, sample index, and method tag:

```python
tag_hash = sum(ord(ch) for ch in tag)
generation_seed = base_seed + problem_id * 10007 + sample_idx * 997 + tag_hash * 131
```

For DRG retry:

```text
tag = DRG_RETRY
sample_idx = 0
```

For SC-2:

```text
tag = SC2
sample_idx = 0, 1
```

The random-trigger control uses:

```python
rng = random.Random(trigger_random_seed + problem_id * 104729)
triggered = rng.random() < trigger_random_rate
```

Default `trigger_random_seed = 1729`.

## Hardware and Token Cost

Example development hardware (adjust for your environment):

```text
Multi-GPU node with float16-capable GPUs (e.g., V100-class)
CUDA runtime visible to PyTorch: cu124 (or your local build)
```

The reports use generated-token accounting as the hardware-independent cost metric. Representative Qwen3 4k runs:

| Run | B0 avg tokens | DRG avg tokens | SC-2 avg tokens | DRG / SC-2 token ratio |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B MATH-500 4k | 3068.1 | 5011.6 | 6177.7 | 0.811 |
| Qwen3-14B MATH-500 4k | 2908.0 | 4612.6 | 5836.8 | 0.790 |
| Qwen3-8B AIME-2024 4k | 4096.0 | 7962.4 | 8171.2 | 0.974 |
| Qwen3-8B AMC 4k | 3597.2 | 6570.2 | 7362.7 | 0.892 |

GPU-hours are environment-dependent and should be computed from launcher logs:

```text
gpu_hours = wall_clock_hours * number_of_visible_gpus_for_that_run
tokens_per_gpu_hour = total_generated_tokens / gpu_hours
```

The multi-GPU launcher writes timestamps to `launcher.log` and the seed batch writes timestamps to `outputs_standalone/deployable_drg_sc2/seed_results/_logs/batch.log`.

## Code and Data Release Statement

Release bundle should include:

```text
standalone/deployable_drg_sc2/
standalone/deployable_drg_sc2/requirements.txt
standalone/deployable_drg_sc2/fetch_dataset.py
standalone/deployable_drg_sc2/download_model.py
standalone/deployable_drg_sc2/run_pipeline.py
standalone/deployable_drg_sc2/launch_multi_gpu.py
standalone/deployable_drg_sc2/aggregate_seed_runs.py
standalone/deployable_drg_sc2/answer_normalization.py
standalone/deployable_drg_sc2/symbolic_math.py
```

Generated outputs are written under:

```text
outputs_standalone/deployable_drg_sc2/
```

Each run contains `summary.json`, `REPORT.md`, and JSONL records in `records/`. The JSONL records are sufficient to recompute trigger overlap, gate decisions, correctness transitions, deploy fallback behavior, token accounting, McNemar tests, and seed-variance summaries without rerunning model inference.

Datasets are fetched into local `datasets/` directories with `fetch_dataset.py`. Model weights are not redistributed by this repo; instead, provide download commands plus exact Hugging Face checkpoint revisions or local snapshot checksums.
