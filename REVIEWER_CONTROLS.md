# Reviewer-Control Experiment Plan

This file maps the highest-priority reviewer doubts to standalone-runner support.

## Coverage Matrix

| Doubt | Captured? | How |
| --- | --- | --- |
| Real trigger vs random trigger at same rate | Yes | `--trigger_source heuristic` vs `--trigger_source random --trigger_random_rate <rate>` |
| Universal vs selective triggering | Yes | `--trigger_source all` vs `--trigger_source heuristic` |
| Correctness preservation / C->W counts | Yes | `REPORT.md` now includes DRG C->W/W->C transitions; `build_component_baselines.py` also reports C->W |
| Operator-null comparison | Partial | Clean/instruction/previous/full scaffold are supported; legacy 8-operator suite remains outside standalone |
| Main-result seed variance | Yes | Use `--seed <s>` and aggregate with `aggregate_seed_runs.py` |
| Triggered retry without previous context | Yes | `--retry_previous_attempt_chars 0` |
| Previous context but no pathology gate | Yes | `--gate_policy accept_all_disagreements` |
| Clean restart + same gate | Yes | `--retry_previous_attempt_chars 0 --gate_policy pathology` |
| Selective SC-2 only on triggered examples | Yes, post-hoc | `build_component_baselines.py` |
| Budget-matched SC | Partial | SC-2/SC-8 are supported; arbitrary SC-K is not yet generalized |
| Agreement-gated two-sample retry without detector | Yes | `--trigger_source all --gate_policy accept_all_disagreements` |
| Real vs unrelated/shuffled/boxed/no-symbol previous trace | Yes | `--retry_previous_attempt_variant real|other_problem|shuffled_lines|boxed_region|no_math_symbols` |
| Summary of failed attempt | Not yet | Needs an explicit summarizer/model step; intentionally not silently faked |

## Qwen3 Control Completion

Use the existing Qwen3 previous-only run to get the real trigger rate. For Qwen3-8B MATH-500, the previous-only run had 358/500 triggered, so random rate is `0.716`.

Random trigger, same approximate rate:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/Qwen3-8B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --trigger_source random \
  --trigger_random_rate 0.716 \
  --trigger_random_seed 1729 \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode user \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 1200 \
  --retry_previous_attempt_variant real \
  --run_name math500_qwen3_8b_random_trigger_rate0716_previous_only
```

Universal trigger:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/Qwen3-8B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --trigger_source all \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode user \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 1200 \
  --retry_previous_attempt_variant real \
  --run_name math500_qwen3_8b_universal_trigger_previous_only
```

Component post-hoc baselines for any completed run:

```bash
python3 standalone/deployable_drg_sc2/build_component_baselines.py \
  --run_dir outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_bf16_t07_previous_only_thinking_4gpu \
  --out outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/QWEN3_8B_COMPONENT_BASELINES.md
```

## Main-Result Seed Variance

Rerun the same method with seeds `0..4`, ideally reusing the same greedy baseline for speed/provenance.

Example seed-1 command:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/Qwen3-8B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --seed 1 \
  --reuse_b0_jsonl outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_bf16_t07_previous_only_thinking_4gpu/records/b0_records.jsonl \
  --trigger_source heuristic \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode user \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 1200 \
  --retry_previous_attempt_variant real \
  --run_name math500_qwen3_8b_previous_only_seed1
```

Aggregate after all seeds complete:

```bash
python3 standalone/deployable_drg_sc2/aggregate_seed_runs.py \
  --runs \
  outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_previous_only_seed0 \
  outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_previous_only_seed1 \
  outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_previous_only_seed2 \
  outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_previous_only_seed3 \
  outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/math500_qwen3_8b_previous_only_seed4 \
  --out outputs_standalone/deployable_drg_sc2/math500/Qwen3-8B/QWEN3_8B_SEED_VARIANCE.md
```

## Component-Matched Baselines

Triggered clean retry with same gate:

```bash
--trigger_source heuristic --retry_previous_attempt_chars 0 --gate_policy pathology
```

Previous-context retry with no pathology gate:

```bash
--trigger_source heuristic --retry_previous_attempt_chars 1200 --gate_policy accept_all_disagreements
```

Agreement-gated retry without detector:

```bash
--trigger_source all --retry_previous_attempt_chars 1200 --gate_policy accept_all_disagreements
```

Preservation-only no-accept control:

```bash
--trigger_source heuristic --retry_previous_attempt_chars 1200 --gate_policy never_accept
```

## Previous-Attempt Mechanism Controls

All use the same outer method and differ only in the previous-attempt block:

```bash
--retry_previous_attempt_variant real
--retry_previous_attempt_variant other_problem
--retry_previous_attempt_variant shuffled_lines
--retry_previous_attempt_variant boxed_region
--retry_previous_attempt_variant no_math_symbols
```

Recommended run names:

- `math500_qwen3_8b_prev_real`
- `math500_qwen3_8b_prev_other_problem`
- `math500_qwen3_8b_prev_shuffled_lines`
- `math500_qwen3_8b_prev_boxed_region`
- `math500_qwen3_8b_prev_no_math_symbols`
