# Porting Guide

This folder is intended to be portable, but it is not a full experiment bundle by itself. On a new machine, you need three pieces:

1. This folder:
   - `standalone/deployable_drg_sc2/`
2. A local model directory:
   - passed as `--model_path`
3. A local HuggingFace dataset directory saved with `datasets.load_from_disk(...)`:
   - passed as `--dataset_path`

## What Is Self-Contained

The standalone folder contains:

- generation and routing logic
- answer extraction and normalization
- local symbolic equivalence fallback for corrected math scoring
- report generation
- multi-GPU sharded launcher
- model download helper
- dataset fetch/prep helper

The standalone folder does not contain:

- model weights
- tokenizer files
- already-prepared dataset folders
- the original paper artifact runners

## Required Python Environment

Create a fresh environment and install:

```bash
pip install -r standalone/deployable_drg_sc2/requirements.txt
```

Optional but recommended for Math-style symbolic equivalence:

- `skythought` with `skythought.evals.util.math_parsing_util.math_equal`

If SkyThought is unavailable, the pipeline still runs and falls back to the local symbolic matcher in `symbolic_math.py`.

## Required Inputs

### Models

`--model_path` must point to a local HuggingFace model directory that works with:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`

Examples:

- `base_llms/DeepSeek-R1-Distill-Qwen-7B`
- `base_llms/Ministral-3-14B-Reasoning-2512`

The second path is the Hugging Face reasoning checkpoint used locally. In paper-facing tables, label this model as `Ministral 3 14B` and cite the official served model id `ministral-14b-2512`; keep `mistralai/Ministral-3-14B-Reasoning-2512` only as checkpoint provenance.

You can create those folders with:

```bash
python3 standalone/deployable_drg_sc2/download_model.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model mistralai/Ministral-3-14B-Reasoning-2512 \
  --base_dir models
```

### Datasets

`--dataset_path` must point to a dataset previously saved with:

```python
dataset.save_to_disk("/path/to/dataset_dir")
```

The runner loads data with `datasets.load_from_disk(...)`.

You can fetch common benchmark datasets with:

```bash
python3 standalone/deployable_drg_sc2/fetch_dataset.py \
  --output-dir datasets \
  --tasks math500 aime_2024 amc
```

Supported dataset shapes:

- `math500`-style:
  - fields like `problem`, `answer`, and optionally `unique_id` or `id`
- `gsm8k`-style:
  - fields like `question`, `answer`
- `gpqa_diamond`-style:
  - fields like `question`, `answer`, optionally `unique_id`

If your new dataset uses different field names, update:

- `get_example_fields(...)` in [run_pipeline.py](run_pipeline.py)
- possibly `extract_predicted_answer(...)` if the answer format differs

## Directory Contract On The New Machine

A simple portable layout is:

```text
project_root/
  standalone/
    deployable_drg_sc2/
  models/
    DeepSeek-R1-Distill-Qwen-7B/
    Ministral-3-14B-Reasoning-2512/
  datasets/
    math500/
    aime_2024/
    amc/
```

Then run from `project_root/`.

## Example Commands

### Single-GPU or single-process run

```bash
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path models/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --run_name math500_qwen7b_ported
```

### Sharded multi-GPU run

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path models/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --run_name math500_qwen7b_4gpu
```

### Large model across multiple visible GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 standalone/deployable_drg_sc2/run_pipeline.py \
  --model_path models/Ministral-3-14B-Reasoning-2512 \
  --dataset_path datasets/aime_2024 \
  --split all \
  --max_new_tokens 32768 \
  --dtype float16 \
  --device_map auto \
  --run_name aime2024_ministral14b
```

## What To Copy From This Machine

At minimum, copy:

- `standalone/deployable_drg_sc2/`

Then either:

- copy local model folders and local dataset folders too

or:

- recreate those folders on the target machine with the bundled helpers:
  - `standalone/deployable_drg_sc2/download_model.py`
  - `standalone/deployable_drg_sc2/fetch_dataset.py`

## What You Still Need To Prepare Yourself

This standalone folder now includes model and dataset bootstrap helpers, but you may still want to adjust them for your own registry names, dataset subsets, or local storage layout.

## Reproducibility Notes

- Default retry mode is `T=0.7`, `top_p=0.95`, with the original problem only.
- Set `--retry_temperature 0` for the strict greedy-retry ablation.
- Outputs are always written under:
  - `outputs_standalone/deployable_drg_sc2/<dataset>/<model>/<run_name>/`

## Quick Porting Checklist

- Copy `standalone/deployable_drg_sc2/`
- Install `requirements.txt`
- Confirm `python3 -m py_compile standalone/deployable_drg_sc2/run_pipeline.py`
- Place model folder locally
- Place dataset folder locally
- Run one 2-example smoke test with `--limit 2`
- Then launch the full run
