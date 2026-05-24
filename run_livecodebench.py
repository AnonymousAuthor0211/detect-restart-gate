#!/usr/bin/env python3
"""
Lightweight LiveCodeBench adapter for the standalone DRG experiments.

This intentionally stays separate from run_pipeline.py because code tasks need
test execution rather than boxed-answer extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    import run_pipeline as rp
else:
    from . import run_pipeline as rp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--dataset_name", type=str, default="livecodebench/code_generation_lite")
    p.add_argument("--version_tag", type=str, default="release_v2")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--enable_thinking", type=str, default="true", choices=["auto", "true", "false"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--output_root", type=Path, default=Path("outputs_standalone/deployable_drg_sc2"))
    p.add_argument("--methods", type=str, default="B0,DRG,SC2")
    p.add_argument("--sc_temperature", type=float, default=0.7)
    p.add_argument("--sc_top_p", type=float, default=0.95)
    p.add_argument("--retry_temperature", type=float, default=0.7)
    p.add_argument("--retry_top_p", type=float, default=0.95)
    p.add_argument("--retry_previous_attempt_chars", type=int, default=1200)
    p.add_argument("--trigger_rep_threshold", type=float, default=0.7)
    p.add_argument("--trigger_length_percentile", type=float, default=85.0)
    p.add_argument("--trigger_stall_lines", type=int, default=4)
    p.add_argument("--gate_pathology_threshold", type=int, default=2)
    p.add_argument("--eval_timeout", type=float, default=6.0)
    p.add_argument("--progress_every", type=int, default=1)
    p.add_argument("--save_sample_texts", action="store_true", default=False)
    return p.parse_args()


def sanitize_slug(text: str) -> str:
    return rp.sanitize_slug(text)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    model_tag = sanitize_slug(args.model_path.name or args.model_path.as_posix().split("/")[-1])
    run_tag = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / "livecodebench" / model_tag / sanitize_slug(run_tag)


def import_lcb_utils():
    repo_root = Path(__file__).resolve().parents[2]
    wsc_root = repo_root / "experiments/convergence/WordSaladChopper/reproduced"
    if str(wsc_root) not in sys.path:
        sys.path.insert(0, str(wsc_root))
    from pipeline.datasets.livecodebench.livecodebench_util import (  # type: ignore
        has_test_type,
        post_process_code,
        translate_private_test_cases,
        unsafe_lcb_runTests,
    )

    try:
        from skythought.evals.util.common import has_code  # type: ignore
    except Exception:
        has_code = None
    return has_test_type, post_process_code, translate_private_test_cases, unsafe_lcb_runTests, has_code


def load_lcb_examples(args: argparse.Namespace) -> List[dict]:
    from datasets import load_dataset

    has_test_type, _, translate_private_test_cases, _, _ = import_lcb_utils()
    ds = load_dataset(
        args.dataset_name,
        version_tag=args.version_tag,
        split=args.split,
        trust_remote_code=True,
    )
    end = None if args.limit is None else args.start_idx + args.limit
    rows = ds.select(range(args.start_idx, min(end or len(ds), len(ds))))
    examples: List[dict] = []
    for local_idx, row in enumerate(rows):
        global_idx = args.start_idx + local_idx
        tests = translate_private_test_cases(row["private_test_cases"])
        is_stdin = bool(has_test_type(row["public_test_cases"], "stdin"))
        prompt = build_lcb_prompt(dict(row), is_stdin)
        examples.append(
            {
                "problem_id": global_idx,
                "unique_id": str(row.get("question_id", global_idx)),
                "question_title": row.get("question_title", ""),
                "prompt": prompt,
                "test": tests,
                "is_stdin": is_stdin,
                "public_test_cases": row.get("public_test_cases", ""),
                "starter_code": row.get("starter_code", ""),
            }
        )
    return examples


def build_lcb_prompt(row: dict, is_stdin: bool) -> str:
    question = str(row.get("question_content", "")).strip()
    starter = str(row.get("starter_code", "") or "").strip()
    public_tests = str(row.get("public_test_cases", "") or "").strip()
    title = str(row.get("question_title", "") or "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    parts.append("Solve the following programming problem in Python 3.")
    if is_stdin:
        parts.append("Your program should read from standard input and write to standard output.")
    else:
        parts.append("Implement the required function exactly matching the provided starter code/signature.")
    parts.append("Return only the final solution code in one Python fenced code block.")
    parts.append(f"Problem:\n{question}")
    if starter:
        parts.append(f"Starter code:\n```python\n{starter}\n```")
    if public_tests:
        parts.append(f"Public tests/examples:\n{public_tests}")
    return "\n\n".join(parts)


def coding_system_prompt() -> str:
    return (
        "You are a helpful competitive-programming assistant. Think carefully, "
        "then provide the final Python 3 solution code in one fenced code block."
    )


def build_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": coding_system_prompt()},
        {"role": "user", "content": prompt},
    ]


def generate_code_response(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    seed: Optional[int],
    enable_thinking: str,
) -> Tuple[str, int]:
    import torch

    input_device = rp.resolve_generation_device(model, device)
    chat_kwargs = {}
    if enable_thinking != "auto":
        chat_kwargs["enable_thinking"] = enable_thinking == "true"
    prompt_obj = tokenizer.apply_chat_template(
        build_messages(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **chat_kwargs,
    )
    input_ids, attention_mask = rp._extract_prompt_tensors(prompt_obj, torch, input_device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
        attention_mask=attention_mask,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        if do_sample and seed is not None:
            fork_devices = []
            if input_device.startswith("cuda"):
                try:
                    fork_devices = [int(input_device.split(":")[1])]
                except Exception:
                    fork_devices = []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed)
                if input_device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                out = model.generate(input_ids, **gen_kwargs)
        else:
            out = model.generate(input_ids, **gen_kwargs)
    gen_ids = out[0, input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=False), int(gen_ids.shape[0])


def visible_after_think(text: str) -> str:
    marker = "</think>"
    return text.rsplit(marker, 1)[-1] if marker in text else text


def fenced_code_blocks(text: str) -> List[str]:
    visible = visible_after_think(text)
    return re.findall(r"```(?:python|py)?\s*(.*?)```", visible, flags=re.DOTALL | re.IGNORECASE)


def extract_code(text: str, has_code_func=None, post_process_code_func=None) -> Optional[str]:
    visible = visible_after_think(text)
    candidates: List[str] = []
    if has_code_func is not None:
        try:
            candidates = list(has_code_func(visible))
        except Exception:
            candidates = []
    if not candidates:
        candidates = fenced_code_blocks(text)
    if not candidates and re.search(r"\b(def|class|import|from)\b", visible):
        candidates = [visible]
    if not candidates:
        return None
    code = candidates[-1]
    if post_process_code_func is not None:
        code = post_process_code_func(code)
    return code.strip() or None


def evaluate_code(problem: dict, response: str, timeout: float) -> dict:
    _, post_process_code, _, unsafe_lcb_runTests, has_code = import_lcb_utils()
    has_fenced_code = bool(fenced_code_blocks(response))
    code = extract_code(response, has_code, post_process_code)
    if not code:
        return {
            "correct": False,
            "has_code": False,
            "has_fenced_code": False,
            "code": None,
            "reason": "no_code",
        }
    try:
        result_list = unsafe_lcb_runTests(
            problem,
            code,
            timeout=timeout,
            runtime_debug=False,
            is_extracted=not problem["is_stdin"],
        )
        details = [bool(row[0]) for row in result_list]
        correct = bool(details) and all(details)
        return {
            "correct": correct,
            "has_code": True,
            "has_fenced_code": has_fenced_code,
            "code": code,
            "passed_tests": int(sum(details)),
            "total_tests": int(len(details)),
            "reason": "passed" if correct else "failed_tests",
        }
    except Exception as exc:
        return {
            "correct": False,
            "has_code": True,
            "has_fenced_code": has_fenced_code,
            "code": code,
            "reason": f"eval_error:{type(exc).__name__}:{exc}",
        }


def signal_count(signals: dict) -> int:
    return int(bool(signals["rep_trigger"])) + int(bool(signals["len_trigger"])) + int(bool(signals["stall_trigger"]))


def compute_signals(text: str, tokens: int, length_cutoff: float, args: argparse.Namespace) -> dict:
    rep = rp.repetition_unigram(text)
    stall = rp.stalled_no_new_symbols(text, args.trigger_stall_lines)
    signals = {
        "rep": rep,
        "tokens": int(tokens),
        "length_cutoff": float(length_cutoff),
        "stall": bool(stall),
        "rep_trigger": bool(rep >= args.trigger_rep_threshold),
        "len_trigger": bool(tokens >= length_cutoff),
        "stall_trigger": bool(stall),
    }
    signals["signal_count"] = signal_count(signals)
    signals["triggered"] = bool(
        signals["rep_trigger"] or signals["len_trigger"] or signals["stall_trigger"]
    )
    return signals


def retry_prompt(problem_prompt: str, baseline_text: str, args: argparse.Namespace) -> str:
    tail = visible_after_think(baseline_text)[-max(0, int(args.retry_previous_attempt_chars)) :].strip()
    if not tail:
        return problem_prompt
    return problem_prompt.rstrip() + "\n\nPrevious attempt (may be flawed):\n" + tail


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_bool(rows: Sequence[dict], key: str) -> Tuple[int, float]:
    n = len(rows)
    c = sum(1 for row in rows if row.get(key))
    return c, c / n if n else float("nan")


def run(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = resolve_output_dir(args)
    rec_dir = out_dir / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    examples = load_lcb_examples(args)
    if not examples:
        raise RuntimeError("No LiveCodeBench examples loaded.")

    dtype = rp.torch_dtype_from_str(args.dtype, torch)
    tokenizer = rp.load_tokenizer_for_model(args.model_path, AutoTokenizer)
    model = rp.load_text_generation_model(args.model_path, dtype, args.device, args.device_map, AutoModelForCausalLM)

    methods = {m.strip().upper() for m in args.methods.split(",") if m.strip()}
    t0 = time.time()

    b0_rows: List[dict] = []
    for i, ex in enumerate(examples, 1):
        text, toks = generate_code_response(
            model,
            tokenizer,
            ex["prompt"],
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            do_sample=False,
            temperature=None,
            top_p=None,
            seed=None,
            enable_thinking=args.enable_thinking,
        )
        ev = evaluate_code(ex, text, args.eval_timeout)
        row = {
            "problem_id": ex["problem_id"],
            "unique_id": ex["unique_id"],
            "question_title": ex["question_title"],
            "is_stdin": ex["is_stdin"],
            "tokens": toks,
            "correct": ev["correct"],
            "has_code": ev["has_code"],
            "has_fenced_code": ev["has_fenced_code"],
            "eval_reason": ev["reason"],
            "code": ev["code"] if args.save_sample_texts else None,
            "text": text if args.save_sample_texts else None,
        }
        b0_rows.append(row)
        if i % args.progress_every == 0 or i == len(examples):
            print(f"{time.strftime('%F %T')} [B0] {i}/{len(examples)} correct={sum(r['correct'] for r in b0_rows)}", flush=True)
    write_jsonl(rec_dir / "b0_records.jsonl", b0_rows)

    length_cutoff = rp.percentile([row["tokens"] for row in b0_rows], args.trigger_length_percentile)

    drg_rows: List[dict] = []
    if "DRG" in methods:
        b0_by_id = {row["problem_id"]: row for row in b0_rows}
        for i, ex in enumerate(examples, 1):
            b0 = b0_by_id[ex["problem_id"]]
            baseline_text = b0["text"]
            if baseline_text is None:
                # Regenerate text only if the user asked not to save texts. This keeps default
                # records compact while preserving DRG functionality.
                baseline_text, _ = generate_code_response(
                    model,
                    tokenizer,
                    ex["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    seed=None,
                    enable_thinking=args.enable_thinking,
                )
            signals = compute_signals(baseline_text, int(b0["tokens"]), length_cutoff, args)
            selected = "baseline"
            path = "path_A"
            retry_row = None
            if signals["triggered"]:
                r_prompt = retry_prompt(ex["prompt"], baseline_text, args)
                r_seed = rp.make_generation_seed(args.seed, int(ex["problem_id"]), 0, "lcb_retry")
                r_text, r_toks = generate_code_response(
                    model,
                    tokenizer,
                    r_prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    do_sample=args.retry_temperature > 0,
                    temperature=args.retry_temperature if args.retry_temperature > 0 else None,
                    top_p=args.retry_top_p if args.retry_temperature > 0 else None,
                    seed=r_seed if args.retry_temperature > 0 else None,
                    enable_thinking=args.enable_thinking,
                )
                r_ev = evaluate_code(ex, r_text, args.eval_timeout)
                retry_code = r_ev["code"] or ""
                baseline_code = b0.get("code") or extract_code(baseline_text)
                agree = bool(baseline_code and retry_code and baseline_code.strip() == retry_code.strip())
                high_pathology = signals["signal_count"] >= args.gate_pathology_threshold
                if agree:
                    path = "path_B_agree"
                elif high_pathology and r_ev["has_code"]:
                    selected = "retry"
                    path = "path_B_accept"
                else:
                    path = "path_B_keep"
                retry_row = {
                    "tokens": r_toks,
                    "correct": r_ev["correct"],
                    "has_code": r_ev["has_code"],
                    "has_fenced_code": r_ev["has_fenced_code"],
                    "eval_reason": r_ev["reason"],
                    "code": r_ev["code"] if args.save_sample_texts else None,
                    "text": r_text if args.save_sample_texts else None,
                    "code_agree": agree,
                }
            final_correct = bool(retry_row["correct"]) if selected == "retry" and retry_row else bool(b0["correct"])
            final_has_code = bool(retry_row["has_code"]) if selected == "retry" and retry_row else bool(b0["has_code"])
            final_has_fenced_code = (
                bool(retry_row["has_fenced_code"])
                if selected == "retry" and retry_row
                else bool(b0.get("has_fenced_code"))
            )
            drg_pass_at_2 = bool(b0["correct"]) or bool(retry_row["correct"] if retry_row else False)
            drg_rows.append(
                {
                    "problem_id": ex["problem_id"],
                    "unique_id": ex["unique_id"],
                    "path": path,
                    "selected": selected,
                    "baseline_correct": bool(b0["correct"]),
                    "baseline_has_code": bool(b0["has_code"]),
                    "baseline_has_fenced_code": bool(b0.get("has_fenced_code")),
                    "retry_correct": retry_row["correct"] if retry_row else None,
                    "retry_has_code": retry_row["has_code"] if retry_row else None,
                    "retry_has_fenced_code": retry_row["has_fenced_code"] if retry_row else None,
                    "drg_pass_at_2": drg_pass_at_2,
                    "final_correct": final_correct,
                    "final_has_code": final_has_code,
                    "final_has_fenced_code": final_has_fenced_code,
                    "baseline_tokens": int(b0["tokens"]),
                    "retry_tokens": int(retry_row["tokens"]) if retry_row else 0,
                    "total_tokens": int(b0["tokens"]) + (int(retry_row["tokens"]) if retry_row else 0),
                    "trigger_signals": signals,
                }
            )
            if i % args.progress_every == 0 or i == len(examples):
                print(f"{time.strftime('%F %T')} [DRG] {i}/{len(examples)} correct={sum(r['final_correct'] for r in drg_rows)}", flush=True)
        write_jsonl(rec_dir / "drg_records.jsonl", drg_rows)

    sc2_rows: List[dict] = []
    if "SC2" in methods:
        for i, ex in enumerate(examples, 1):
            samples = []
            for sample_idx in range(2):
                seed = rp.make_generation_seed(args.seed, int(ex["problem_id"]), sample_idx, "lcb_sc2")
                text, toks = generate_code_response(
                    model,
                    tokenizer,
                    ex["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    do_sample=True,
                    temperature=args.sc_temperature,
                    top_p=args.sc_top_p,
                    seed=seed,
                    enable_thinking=args.enable_thinking,
                )
                ev = evaluate_code(ex, text, args.eval_timeout)
                samples.append(
                    {
                        "sample_idx": sample_idx,
                        "tokens": toks,
                        "correct": ev["correct"],
                        "has_code": ev["has_code"],
                        "has_fenced_code": ev["has_fenced_code"],
                        "eval_reason": ev["reason"],
                        "code": ev["code"] if args.save_sample_texts else None,
                        "text": text if args.save_sample_texts else None,
                    }
                )
            sc2_rows.append(
                {
                    "problem_id": ex["problem_id"],
                    "unique_id": ex["unique_id"],
                    "sample1_correct": bool(samples[0]["correct"]),
                    "sample2_correct": bool(samples[1]["correct"]),
                    "pass_at_2": any(bool(s["correct"]) for s in samples),
                    "has_code_any": any(bool(s["has_code"]) for s in samples),
                    "has_fenced_code_any": any(bool(s["has_fenced_code"]) for s in samples),
                    "tokens": sum(int(s["tokens"]) for s in samples),
                    "samples": samples,
                }
            )
            if i % args.progress_every == 0 or i == len(examples):
                print(f"{time.strftime('%F %T')} [SC2] {i}/{len(examples)} pass@2={sum(r['pass_at_2'] for r in sc2_rows)}", flush=True)
        write_jsonl(rec_dir / "sc2_records.jsonl", sc2_rows)

    write_report(out_dir, args, examples, b0_rows, drg_rows, sc2_rows, length_cutoff, elapsed=time.time() - t0)
    return out_dir


def fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    examples: Sequence[dict],
    b0_rows: Sequence[dict],
    drg_rows: Sequence[dict],
    sc2_rows: Sequence[dict],
    length_cutoff: float,
    *,
    elapsed: float,
) -> None:
    n = len(examples)
    b0_correct, b0_acc = summarize_bool(b0_rows, "correct")
    b0_no_answer = 1.0 - (sum(1 for r in b0_rows if r.get("has_fenced_code")) / n if n else 0.0)
    b0_tokens = mean([int(r["tokens"]) for r in b0_rows]) if b0_rows else float("nan")
    sc2_tokens = mean([int(r["tokens"]) for r in sc2_rows]) if sc2_rows else None
    sc2_sample1_tokens = sc2_tokens / 2.0 if sc2_tokens is not None else None
    sc2_pass2_tokens = sc2_tokens

    rows = [("B0", b0_acc, b0_correct, b0_tokens, b0_no_answer)]
    if drg_rows:
        drg_correct = sum(1 for r in drg_rows if r.get("final_correct"))
        drg_acc = drg_correct / n
        drg_tokens = mean([int(r["total_tokens"]) for r in drg_rows])
        drg_no_answer = 1.0 - sum(1 for r in drg_rows if r.get("final_has_fenced_code", r.get("final_has_code"))) / n
        rows.append(("DRG selected pass@1", drg_acc, drg_correct, drg_tokens, drg_no_answer))
        drg_pass2 = sum(
            1
            for r in drg_rows
            if bool(r.get("drg_pass_at_2", bool(r.get("baseline_correct")) or bool(r.get("retry_correct"))))
        )
        retry_or_baseline_has_answer = sum(
            1
            for r in drg_rows
            if bool(r.get("baseline_has_fenced_code", True))
            or bool(r.get("retry_has_fenced_code", r.get("retry_has_code")))
        )
        rows.append(
            (
                "DRG oracle pass@2",
                drg_pass2 / n,
                drg_pass2,
                drg_tokens,
                1.0 - retry_or_baseline_has_answer / n,
            )
        )
    if sc2_rows:
        s1_correct = sum(1 for r in sc2_rows if r.get("sample1_correct"))
        s2_pass = sum(1 for r in sc2_rows if r.get("pass_at_2"))
        s1_no_answer = 1.0 - sum(
            1
            for r in sc2_rows
            if (r.get("samples") or [{}])[0].get("has_fenced_code", (r.get("samples") or [{}])[0].get("has_code"))
        ) / n
        sc2_no_answer = 1.0 - sum(1 for r in sc2_rows if r.get("has_fenced_code_any", r.get("has_code_any"))) / n
        rows.append(("SC-2 sample-1 pass@1", s1_correct / n, s1_correct, sc2_sample1_tokens or float("nan"), s1_no_answer))
        rows.append(("SC-2 pass@2", s2_pass / n, s2_pass, sc2_pass2_tokens or float("nan"), sc2_no_answer))

    lines = [
        "# LiveCodeBench DRG Report",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model_path}`",
        f"- Dataset: `{args.dataset_name}`",
        f"- Version tag: `{args.version_tag}`",
        f"- Split/start/limit: `{args.split}` / `{args.start_idx}` / `{args.limit}`",
        f"- Problems: `{n}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        f"- Output dir: `{out_dir}`",
        f"- Thinking: `{args.enable_thinking}`",
        f"- Retry: T=`{args.retry_temperature}`, top_p=`{args.retry_top_p}`, previous tail chars=`{args.retry_previous_attempt_chars}`",
        f"- Trigger: `rep >= {args.trigger_rep_threshold}` OR `len >= p{args.trigger_length_percentile}` OR `stall(k={args.trigger_stall_lines})`",
        f"- Gate: `signal_count >= {args.gate_pathology_threshold}`; agreement uses exact extracted-code equality",
        f"- Length cutoff: `{length_cutoff:.2f}` tokens",
        f"- Eval timeout: `{args.eval_timeout}` seconds/test",
        f"- Elapsed: `{elapsed / 60.0:.1f}` minutes",
        "",
        "## Results",
        "",
        "| Method | Accuracy | Correct | Avg tokens | Ratio vs SC-2 sample-1 | Ratio vs SC-2 pass@2 | No-answer rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, acc, correct, tokens, no_answer in rows:
        ratio_s1 = tokens / sc2_sample1_tokens if sc2_sample1_tokens else None
        ratio_s2 = tokens / sc2_pass2_tokens if sc2_pass2_tokens else None
        lines.append(
            f"| {name} | {fmt(acc)} | {correct}/{n} | {tokens:.1f} | "
            f"{fmt(ratio_s1)} | {fmt(ratio_s2)} | {fmt(no_answer)} |"
        )

    if drg_rows:
        path_counts = Counter(str(r["path"]) for r in drg_rows)
        triggered = sum(1 for r in drg_rows if r["trigger_signals"]["triggered"])
        accepts = path_counts.get("path_B_accept", 0)
        c_to_w = sum(1 for r in drg_rows if r["baseline_correct"] and not r["final_correct"])
        w_to_c = sum(1 for r in drg_rows if (not r["baseline_correct"]) and r["final_correct"])
        lines += [
            "",
            "## DRG Breakdown",
            "",
            f"- Triggered: `{triggered}/{n}`",
            f"- Accepted retries: `{accepts}`",
            f"- W->C: `{w_to_c}`",
            f"- C->W: `{c_to_w}`",
            f"- Paths: `{dict(path_counts)}`",
        ]
    if drg_rows and sc2_rows:
        drg_by_id = {r["problem_id"]: r for r in drg_rows}
        sc_by_id = {r["problem_id"]: r for r in sc2_rows}
        def overlap(drg_key: str, sc_key: str) -> dict:
            both = drg_only = sc_only = neither = 0
            for ex in examples:
                pid = ex["problem_id"]
                d = bool(drg_by_id[pid].get(drg_key))
                s = bool(sc_by_id[pid].get(sc_key))
                if d and s:
                    both += 1
                elif d:
                    drg_only += 1
                elif s:
                    sc_only += 1
                else:
                    neither += 1
            return {
                "n": n,
                "both": both,
                "drg_only": drg_only,
                "sc_only": sc_only,
                "neither": neither,
                "union_accuracy": (both + drg_only + sc_only) / n if n else float("nan"),
            }
        lines += [
            "",
            "## Paired Comparisons",
            "",
            f"- Apples-to-apples pass@1, `DRG selected` vs `SC-2 sample-1`: `{overlap('final_correct', 'sample1_correct')}`",
            f"- Apples-to-apples two-shot oracle, `DRG baseline OR retry` vs `SC-2 pass@2`: `{overlap('drg_pass_at_2', 'pass_at_2')}`",
            f"- Hard deploy-style comparison, `DRG selected` vs `SC-2 pass@2`: `{overlap('final_correct', 'pass_at_2')}`",
        ]

    lines += [
        "",
        "## Caveat",
        "",
        "- `No-answer` means no fenced Python code block was parsed from the visible response after any `</think>` block. Code that parses but fails tests is counted as a wrong answer, not a missing answer.",
        "- LiveCodeBench scoring executes generated Python against tests; this adapter uses the existing local LCB utility with a reliability guard and timeout, not a full container sandbox.",
        "- `SC-2 pass@2` is a code-benchmark coverage metric, not a deployable majority-vote answer selector. The report therefore separates pass@1, two-shot oracle, and hard deploy-style comparisons.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = run(args)
    print(f"Final report: {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
