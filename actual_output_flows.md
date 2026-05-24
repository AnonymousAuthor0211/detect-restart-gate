# Actual Output Flows: Before and After DRG

This file shows concrete JSONL artifacts from saved runs. Each example follows the same pattern:

```text
problem -> B0 greedy output -> trigger/gate -> DRG retry output -> final selected answer
```

The output snippets are tail excerpts from `baseline_text` and `retry_text`, lightly trimmed for length. The extracted answers and decisions are the exact record fields from `records/drg_records.jsonl` and `records/deploy_records.jsonl`.

## Source Runs

```text
AMC:
outputs_standalone/deployable_drg_sc2/amc/Qwen3-8B/amc_qwen3_8b_fp16_4k_t07_previous_only_thinking_4gpu/

AIME 2024:
outputs_standalone/deployable_drg_sc2/aime_2024/Qwen3-8B/aime2024_qwen3_8b_fp16_4k_t07_previous_only_thinking_4gpu/

OlympiadBench Physics:
outputs_standalone/deployable_drg_sc2/olympiadbench_physics_oe_to_en/Qwen3-8B/olympiadbench_physics_oe_to_qwen3_8b_bf16_4k_t07_previous_only_thinking_12gpu_v2/
```

Shared settings:

```text
model_path = base_llms/Qwen3-8B
max_new_tokens = 4096
retry_temperature = 0.7
retry_top_p = 0.95
retry_previous_attempt_chars = 1200
enable_thinking = true
```

## Flow 1: AMC Coinland

Question summary:

```text
Coins have values 6, 10, and 15 cents. Let x be the most expensive item value
that cannot be purchased exactly. Return the sum of the digits of x.
```

Record:

```text
run = amc_qwen3_8b_fp16_4k_t07_previous_only_thinking_4gpu
problem_id = 1
gold_answer = 11
```

### Before DRG: B0 Greedy

Record fields:

| Field | Value |
| --- | --- |
| `baseline_predicted_answer` | `None` |
| `baseline_correct` | `0` |
| `baseline_tokens` | `4096` |

B0 tail excerpt:

```text
... maybe there's a larger number? Let me check 23, which we saw can't be
formed. But 23 is smaller than 29. So 29 is larger. So maybe 29 is the answer?

But let me check another way. Let me think about the problem again. Maybe
there's a mistake in my reasoning. Let me check numbers from 1 to 30 again,
to make sure I didn't miss any numbers that can't be formed.

Starting from 1:

1-5: No
6: Yes
7: No
8: No
9: No
10: Yes
11: No
12: Yes (6+6)
13: No
14: No
15: Yes
16: Yes (10+6)
17: No
18:
```

The baseline is reasoning in the right neighborhood, but it hits the token budget before a final boxed answer. The extractor therefore returns `None`.

### DRG Trigger and Gate

| Field | Value |
| --- | --- |
| `trigger_rep_unigram` | `0.7684` |
| `trigger_rep_hit` | `1` |
| `trigger_length_hit` | `1` |
| `trigger_stall_hit` | `1` |
| `trigger_signal_count` | `3` |
| `high_pathology` | `1` |

Path:

```text
B0 no-answer + high pathology -> generate A1 retry
```

### After DRG: Retry A1

Record fields:

| Field | Value |
| --- | --- |
| `retry_predicted_answer` | `11` |
| `retry_correct` | `1` |
| `agree_with_baseline_answer` | `0` |
| `accepted_retry` | `1` |
| `decision_reason` | `disagree_high_pathology_accept_retry` |
| `drg_final_source` | `retry` |
| `drg_final_predicted_answer` | `11` |
| `drg_final_correct` | `1` |
| `deploy path` | `path_B_accept` |
| `fallback_used` | `0` |

Retry tail excerpt:

```text
### Step 3: Determine the Largest Unformable Value

From the above, the largest number that cannot be formed is 29. All numbers
greater than 29 can be formed using combinations of 6, 10, and 15.

### Step 4: Compute the Sum of Digits of 29

2 + 9 = 11

### Final Answer

\boxed{11}
```

Artifact-level flow:

```text
B0: no extractable answer, wrong
Trigger: repetition + length + stall
Retry: extracts 11, correct
Gate: high pathology and disagreement, accept retry
Final: 11, correct, no SC-2 fallback
```

## Flow 2: AIME 2024 Aya's Walk

Question summary:

```text
Aya walks 9 km and stops at a coffee shop. At speed s the whole trip takes
4 hours including t minutes at the shop. At speed s+2 it takes 2 hours
24 minutes including the same t. Find the total minutes at speed s+1/2.
```

Record:

```text
run = aime2024_qwen3_8b_fp16_4k_t07_previous_only_thinking_4gpu
problem_id = 0
gold_answer = 204
```

### Before DRG: B0 Greedy

Record fields:

| Field | Value |
| --- | --- |
| `baseline_predicted_answer` | `None` |
| `baseline_correct` | `0` |
| `baseline_tokens` | `4096` |

B0 tail excerpt:

```text
First case: walking at 2.5 km/h. Time to walk 9 km is 9 / 2.5 = 3.6 hours.
Then, adding 24 minutes (which is 0.4 hours) gives 3.6 + 0.4 = 4 hours.
Correct.

Second case: walking at 4.5 km/h. Time to walk 9 km is 9 / 4.5 = 2 hours.
Adding 24 minutes (0.4 hours) gives 2.4 hours. Correct.

Third case: walking at 3 km/h. Time to walk 9 km is 3 hours. Adding 24
minutes gives 3 + 0.4 = 3.4 hours, which is 3 hours and 24 minutes, which is 20
```

Here B0 has essentially solved the problem, but the generation cuts off mid-final-answer and never emits a balanced `\boxed{...}`. The record is therefore a no-answer baseline.

### DRG Trigger and Gate

| Field | Value |
| --- | --- |
| `trigger_rep_unigram` | `0.8170` |
| `trigger_rep_hit` | `1` |
| `trigger_length_hit` | `1` |
| `trigger_stall_hit` | `0` |
| `trigger_signal_count` | `2` |
| `high_pathology` | `1` |

Path:

```text
B0 no-answer + high pathology -> generate A1 retry
```

### After DRG: Retry A1

Record fields:

| Field | Value |
| --- | --- |
| `retry_predicted_answer` | `204` |
| `retry_correct` | `1` |
| `agree_with_baseline_answer` | `0` |
| `accepted_retry` | `1` |
| `decision_reason` | `disagree_high_pathology_accept_retry` |
| `drg_final_source` | `retry` |
| `drg_final_predicted_answer` | `204` |
| `drg_final_correct` | `1` |
| `deploy path` | `path_B_accept` |
| `fallback_used` | `0` |

Retry tail excerpt:

```text
Substitute s = 2.5 into equation (1):

9 / 2.5 + t / 60 = 4
3.6 + t / 60 = 4
t = 24 minutes

At speed s + 1/2 = 3 km/h:

Walking time = 9 / 3 = 3 hours
Coffee shop time = 24 minutes
Total time = 180 + 24 = 204 minutes

### Final Answer

\boxed{204}
```

Artifact-level flow:

```text
B0: no extractable answer, wrong
Trigger: repetition + length
Retry: extracts 204, correct
Gate: high pathology and disagreement, accept retry
Final: 204, correct, no SC-2 fallback
```

## Flow 3: OlympiadBench Physics String Scaling

Question summary:

```text
Physics proportional-reasoning problem about scaling a violin/string by sqrt(2).
The target answer is the frequency ratio sqrt(2)/2.
```

Record:

```text
run = olympiadbench_physics_oe_to_qwen3_8b_bf16_4k_t07_previous_only_thinking_12gpu_v2
problem_id = 2
gold_answer = \frac{\sqrt{2}}{2}
```

### Before DRG: B0 Greedy

Record fields:

| Field | Value |
| --- | --- |
| `baseline_predicted_answer` | `None` |
| `baseline_correct` | `0` |
| `baseline_tokens` | `4096` |

B0 tail excerpt:

```text
T is proportional to A and mu is proportional to A. Therefore, T/mu is
sigma / density, which is constant. Therefore, frequency is proportional
to 1/L. Therefore, if L increases by sqrt(2), frequency decreases by
1/sqrt(2). Therefore, f2 / f1 = 1 / sqrt(2). Therefore, the answer is
1 over square root of 2, which can be written as sqrt(2)/2.

But let
```

This is another important artifact pattern: B0 reaches the correct natural-language answer, but does not emit the paper's required boxed answer before the cutoff.

### DRG Trigger and Gate

| Field | Value |
| --- | --- |
| `trigger_rep_unigram` | `0.8194` |
| `trigger_rep_hit` | `1` |
| `trigger_length_hit` | `1` |
| `trigger_stall_hit` | `0` |
| `trigger_signal_count` | `2` |
| `high_pathology` | `1` |

Path:

```text
B0 no-answer + high pathology -> generate A1 retry
```

### After DRG: Retry A1

Record fields:

| Field | Value |
| --- | --- |
| `retry_predicted_answer` | `\frac{\sqrt{2}}{2}` |
| `retry_correct` | `1` |
| `agree_with_baseline_answer` | `0` |
| `accepted_retry` | `1` |
| `decision_reason` | `disagree_high_pathology_accept_retry` |
| `drg_final_source` | `retry` |
| `drg_final_predicted_answer` | `\frac{\sqrt{2}}{2}` |
| `drg_final_correct` | `1` |
| `deploy path` | `path_B_accept` |
| `fallback_used` | `0` |

Retry tail excerpt:

```text
Since frequency is inversely proportional to length:

f2 / f1 = L1 / L2 = L1 / (sqrt(2) L1) = 1 / sqrt(2)

This can also be expressed as:

f2 / f1 = sqrt(2) / 2

### Final Answer

\boxed{\dfrac{\sqrt{2}}{2}}
```

Artifact-level flow:

```text
B0: correct reasoning but no extractable boxed answer
Trigger: repetition + length
Retry: extracts \frac{\sqrt{2}}{2}, correct
Gate: high pathology and disagreement, accept retry
Final: \frac{\sqrt{2}}{2}, correct, no SC-2 fallback
```

## Paper Takeaway

These examples show the most common qualitative benefit in the 4k Qwen3-8B previous-attempt setting: DRG often does not need to invent a new solution strategy. It turns a long or repetitive trace that failed the answer-extraction contract into a shorter retry that emits a clean boxed final answer.

