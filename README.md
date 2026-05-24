# Standalone DRG -> SC-2

This folder is a portable implementation of the method spec:

- `B0`: single greedy baseline
- `DRG`: greedy baseline + sampled retry + fixed trigger/gate
- `SC-2`: self-consistency with `K=2`, `T=0.7`
- `SC-8`: self-consistency with `K=8`, `T=0.7`
- `DRG -> SC-2 (oracle)`: route all DRG errors to SC-2
- `DRG -> SC-2 (deploy)`: route only no-extractable-answer cases to SC-2

It is intentionally separate from `experiments/convergence/` so it is easier to move or reuse on new datasets.

For migration to another machine, see:

- [PORTING_GUIDE.md](PORTING_GUIDE.md)
- [requirements.txt](requirements.txt)
- [IMPLEMENTATION_DETAILS_FOR_REPRODUCIBILITY.md](IMPLEMENTATION_DETAILS_FOR_REPRODUCIBILITY.md) for exact prompts, trigger/gate logic, scoring, seeds, models, and hardware notes.
- [EXPERIMENTS_E1_E2.md](EXPERIMENTS_E1_E2.md) for the continuation-vs-reflection and WSC/Xie trigger-substitution paper experiments.
- [REVIEWER_CONTROLS.md](REVIEWER_CONTROLS.md) for Qwen3 controls, seed variance, component baselines, and previous-trace mechanism controls.

## Method Defaults

- Trigger: `rep >= 0.7` OR `len >= p85` OR `stall(k=4)`
- Gate: accept disagreeing retry only when `signal_count >= 2`
- Retry: sampled regeneration with `T=0.7`, `top_p=0.95`; the clean control uses the original problem only, while the current previous-only condition appends the last `1200` characters of the baseline attempt.
- Fallback: SC-2 only when the chosen DRG path has no extractable answer

Important:

- There is no intervention scaffold and no extra instruction string in the clean or previous-only retry conditions.
- The clean retry condition uses only the original problem. The previous-only condition uses the original problem plus `Previous attempt (may be flawed):` followed by the baseline tail.
- Both conditions induce trajectory divergence with standard `T=0.7` sampling rather than relying on greedy nondeterminism.
- The original Qwen operator-equivalence artifacts are still informative, but they should now be interpreted as an appendix-style result about operator wording *within a shared scaffold*. They are not the definition of this standalone method.
- If you want the old strict greedy ablation, set `--retry_temperature 0`. On some deterministic setups, strict greedy rerun can reproduce the baseline text exactly and therefore yield no DRG benefit.

## Output Layout

Each run writes to its own dedicated folder under:

`outputs_standalone/deployable_drg_sc2/<dataset>/<model>/<run_name_or_timestamp>/`

Main outputs:

- `config.json`
- `REPORT.md`
- `summary.json`
- `records/b0_records.jsonl`
- `records/drg_records.jsonl`
- `records/sc2_records.jsonl`
- `records/sc8_records.jsonl` if enabled
- `records/oracle_records.jsonl`
- `records/deploy_records.jsonl`
- `fetch_dataset.py`
- `download_model.py`
- `symbolic_math.py`

## Multi-GPU

Yes. There are two ways to use multiple GPUs now:

- `run_pipeline.py` supports manual data sharding with `--num_shards` and `--shard_id`.
- `run_pipeline.py` also supports single-worker model sharding with `--device_map auto` when one model/context does not fit on one GPU.
- `launch_multi_gpu.py` is the higher-level launcher that spreads shards across a GPU list and then merges everything back into one final report.

The launcher preserves the method spec correctly:

- It runs baseline shards first.
- It merges baseline lengths and computes one global `p85` cutoff.
- It then launches the DRG shards with that shared cutoff.
- It merges shard outputs into one final `REPORT.md` and `summary.json`.

Use the two modes differently:

- If one run fits on one GPU and you want more throughput, use dataset sharding with `launch_multi_gpu.py`.
- If one long-context run does not fit on one GPU, use one process with multiple visible GPUs and `--device_map auto`.

## Example Commands

Bootstrap models and datasets first if the new machine does not already have them:

```bash
python3 standalone/deployable_drg_sc2/download_model.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model mistralai/Ministral-3-14B-Reasoning-2512 \
  --base_dir models
```

For paper-facing tables, label the latter as `Ministral 3 14B` and cite the official served model id `ministral-14b-2512`; the command above downloads the Hugging Face reasoning checkpoint used in the reported runs.

```bash
python3 standalone/deployable_drg_sc2/fetch_dataset.py \
  --output-dir datasets \
  --tasks math500 aime_2024 amc
```

Full MATH-500 run:

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --run_name math500_qwen7b_full
```

Strict greedy-retry ablation:

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --retry_temperature 0 \
  --run_name math500_qwen7b_greedy_retry_ablation
```

Reuse existing baseline and SC artifacts while generating only the missing DRG stage:

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --reuse_b0_jsonl outputs/math500_qwen7b_guard_v1/day1_generations.jsonl \
  --reuse_sc2_jsonl outputs/math500_qwen7b_guard_v1/intervention_baseline_sc/apples_to_apples_records.jsonl \
  --reuse_sc8_jsonl outputs/math500_qwen7b_guard_v1/intervention_baseline_sc/apples_to_apples_records.jsonl \
  --run_name math500_qwen7b_reuse_existing
```

AIME 2024:

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/aime_2024 \
  --split all \
  --max_new_tokens 32768 \
  --run_name aime2024_qwen7b
```

AMC:

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/amc \
  --split all \
  --max_new_tokens 32768 \
  --run_name amc_qwen7b
```

Multi-GPU MATH-500 run on GPUs `0,1,2,3`:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --run_name math500_qwen7b_4gpu
```

Multi-GPU AIME 2024:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/aime_2024 \
  --split all \
  --max_new_tokens 32768 \
  --gpu_list 0,1,2,3 \
  --run_name aime2024_qwen7b_4gpu
```

Single large-model run sharded across all visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path base_llms/Ministral-3-14B-Reasoning-2512 \
  --dataset_path datasets/aime_2024 \
  --split all \
  --max_new_tokens 32768 \
  --dtype float16 \
  --device_map auto \
  --run_name aime2024_ministral14b_model_sharded
```

This local path corresponds to the `Ministral 3 14B` family; keep the exact `Reasoning-2512` checkpoint in reproducibility notes, but use `Ministral 3 14B` as the table/model-card name.

## Notes

- The script supports `math500`, `aime_2024`, and `amc` out of the box because they all use the same math-style prompt and answer extraction.
- The standalone folder now also includes local bootstrap helpers:
  - `download_model.py` for pulling HuggingFace models to a local directory
  - `fetch_dataset.py` for dataset download/prep workflows
- Corrected math scoring is now bundled locally:
  - the scorer first tries SkyThought if available
  - otherwise it falls back to the local `symbolic_math.py`
- Existing SC jsonl files can be reused from either this standalone folder or the older apples-to-apples combined files.
- `SC-8` can be skipped with `--skip_sc8` when you want a cheaper first pass.
- On V100s, prefer `--dtype float16` rather than `bfloat16`.
- The default retry is sampled. If you want strict greedy rerun for diagnosis, use `--retry_temperature 0`.
