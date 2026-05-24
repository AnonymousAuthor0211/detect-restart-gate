# E1 / E2 Experiment Wiring

This note documents the two paper-positioning experiments now wired into the standalone runner.

## E1: Continuation vs Reflection

Goal: test whether the previous trace helps because it is a verbatim continuation context, not because the model is asked to reflect on a labeled previous attempt.

Two matched retry conditions:

- `truncate-to-problem`: retry from the original problem only.
- `assistant-continuation`: retry prompt is the original chat template plus the full baseline assistant trace verbatim, then generation continues. No instruction, no previous-attempt label, no truncation.

Run truncate-to-problem:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode user \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 0 \
  --run_name math500_qwen7b_e1_truncate_to_problem
```

Run verbatim assistant continuation. To keep the baseline and SC-2 samples identical, reuse the records from the truncate-to-problem run:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --reuse_b0_jsonl outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_e1_truncate_to_problem/records/b0_records.jsonl \
  --reuse_sc2_jsonl outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_e1_truncate_to_problem/records/sc2_records.jsonl \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode assistant_continuation \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 0 \
  --run_name math500_qwen7b_e1_assistant_continuation
```

Compare the completed runs:

```bash
python3 standalone/deployable_drg_sc2/compare_drg_runs.py \
  --a outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_e1_truncate_to_problem \
  --b outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_e1_assistant_continuation \
  --label_a truncate_to_problem \
  --label_b assistant_continuation \
  --out outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/E1_CONTINUATION_VS_TRUNCATE.md
```

## E2: WSC / Xie Trigger Substitution

Goal: substitute the WordSaladChopper/Xie learned repetition classifier as the trigger, while keeping the DRG gate and previous-attempt retry context fixed.

Download a model-specific WSC probe. The public WSC release provides probes for DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-1.5B, and DeepSeek-R1-Distill-Llama-8B.

```bash
mkdir -p probes/wsc/DeepSeek-R1-Distill-Qwen-7B_s1
wget https://huggingface.co/xiewenya/WordSaladChopper_Classifier/resolve/main/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl \
  -O probes/wsc/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl
```

Run with the learned trigger and the same previous-only retry context:

```bash
python3 standalone/deployable_drg_sc2/launch_multi_gpu.py \
  --model_path base_llms/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/math500 \
  --split all \
  --max_new_tokens 4096 \
  --gpu_list 0,1,2,3 \
  --dtype float16 \
  --enable_thinking true \
  --skip_sc8 \
  --trigger_source wsc \
  --wsc_probe_path probes/wsc/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl \
  --wsc_threshold 0.5 \
  --wsc_streak_len 2 \
  --wsc_short_streak_len 5 \
  --wsc_len_threshold 10 \
  --retry_temperature 0.7 \
  --retry_top_p 0.95 \
  --retry_prompt_mode user \
  --retry_instruction_mode none \
  --retry_previous_attempt_chars 1200 \
  --run_name math500_qwen7b_e2_wsc_trigger_previous_only
```

Compare against your heuristic-trigger previous-only run:

```bash
python3 standalone/deployable_drg_sc2/compare_drg_runs.py \
  --a outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_helpful_bf16_t07_previous_only_4gpu \
  --b outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/math500_qwen7b_e2_wsc_trigger_previous_only \
  --label_a heuristic_trigger \
  --label_b wsc_trigger \
  --out outputs_standalone/deployable_drg_sc2/math500/DeepSeek-R1-Distill-Qwen-7B/E2_WSC_TRIGGER_COMPARISON.md
```

## External Trigger Mode

If you compute Xie/WSC trigger decisions outside this runner, use:

```jsonl
{"problem_id": 0, "triggered": true, "score": 0.83}
{"problem_id": 1, "triggered": false, "score": 0.12}
```

Then pass:

```bash
--trigger_source external --trigger_external_jsonl path/to/triggers.jsonl
```

The gate remains unchanged: retry is accepted only when the retry answer disagrees and the original `rep/length/stall` pathology count satisfies `signal_count >= 2`.
