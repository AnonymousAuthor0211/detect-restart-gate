#!/usr/bin/env python3
"""
Standalone implementation of:
  DRG with clean same-problem retry
  SC-2 / SC-8
  Deployable DRG -> SC-2
  Oracle DRG -> SC-2

The folder is intentionally self-contained so it can be moved without the
rest of the convergence experiment tree.
"""

from __future__ import annotations

import argparse
import importlib
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
    from answer_normalization import math_answers_equal, normalize_math_answer
else:
    from .answer_normalization import math_answers_equal, normalize_math_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", "math500", "gsm8k", "gpqa_diamond"],
    )
    p.add_argument("--split", type=str, default="all")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="HF device_map for model sharding, e.g. 'auto'. Use this for one large model across multiple visible GPUs.",
    )
    p.add_argument(
        "--enable_thinking",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Optional chat-template thinking control. Use 'false' for Qwen3 no-thinking "
            "mode; 'auto' leaves the tokenizer default unchanged."
        ),
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=Path, default=None, help="Explicit run/shard output directory.")
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs_standalone/deployable_drg_sc2"),
        help="Dedicated output root for this standalone pipeline.",
    )
    p.add_argument(
        "--reuse_b0_jsonl",
        type=Path,
        default=None,
        help="Optional existing greedy baseline records/day1 jsonl to reuse.",
    )
    p.add_argument(
        "--reuse_sc2_jsonl",
        type=Path,
        default=None,
        help="Optional existing SC-2 records. Supports standalone or apples-to-apples jsonl.",
    )
    p.add_argument(
        "--reuse_sc8_jsonl",
        type=Path,
        default=None,
        help="Optional existing SC-8 records. Supports standalone or apples-to-apples jsonl.",
    )
    p.add_argument("--skip_sc8", action="store_true", default=False)
    p.add_argument("--sc_temperature", type=float, default=0.7)
    p.add_argument("--sc_top_p", type=float, default=0.95)
    p.add_argument(
        "--retry_temperature",
        type=float,
        default=0.7,
        help="Retry temperature for DRG. Set to 0.0 for strict greedy rerun.",
    )
    p.add_argument(
        "--retry_top_p",
        type=float,
        default=0.95,
        help="Retry top-p for DRG when retry_temperature > 0.",
    )
    p.add_argument(
        "--retry_instruction_mode",
        type=str,
        default="none",
        choices=["none", "restart"],
        help="Retry prompt instruction ablation. 'restart' appends the canonical restart instruction.",
    )
    p.add_argument(
        "--retry_prompt_mode",
        type=str,
        default="user",
        choices=["user", "assistant_continuation"],
        help=(
            "Retry prompt construction. 'user' uses the original problem plus optional "
            "instruction/previous-attempt text in the user message. "
            "'assistant_continuation' re-encodes the original problem and the full baseline "
            "assistant trace verbatim, then continues generation with no extra instruction."
        ),
    )
    p.add_argument(
        "--retry_previous_attempt_chars",
        type=int,
        default=0,
        help="Append this many trailing characters from the baseline attempt to the retry prompt. 0 disables it.",
    )
    p.add_argument(
        "--trigger_source",
        type=str,
        default="heuristic",
        choices=["heuristic", "wsc", "external", "random", "all", "none"],
        help=(
            "Trigger source for DRG. 'heuristic' is rep/length/stall. 'wsc' uses a "
            "WordSaladChopper/Xie-style learned repetition probe. 'external' reads "
            "precomputed trigger decisions from --trigger_external_jsonl. 'random' "
            "triggers a deterministic random subset; 'all' triggers every problem."
        ),
    )
    p.add_argument(
        "--trigger_external_jsonl",
        type=Path,
        default=None,
        help="JSONL with problem_id or unique_id and triggered/wsc_triggered for --trigger_source external.",
    )
    p.add_argument("--wsc_probe_path", type=Path, default=None, help="Path to WSC/Xie probe.pkl for --trigger_source wsc.")
    p.add_argument(
        "--wsc_repo_path",
        type=Path,
        default=Path("experiments/convergence/WordSaladChopper"),
        help="Local WordSaladChopper checkout containing wscgen.prober.",
    )
    p.add_argument("--wsc_prober_kind", type=str, default="logistic", choices=["logistic", "mlp"])
    p.add_argument("--wsc_threshold", type=float, default=0.5)
    p.add_argument("--wsc_streak_len", type=int, default=2)
    p.add_argument("--wsc_short_streak_len", type=int, default=5)
    p.add_argument("--wsc_len_threshold", type=int, default=10)
    p.add_argument(
        "--wsc_split_mode",
        type=str,
        default="paragraph",
        choices=["paragraph", "line"],
        help="Post-hoc chunks used for WSC trigger scoring over the baseline trace.",
    )
    p.add_argument("--trigger_random_rate", type=float, default=0.5)
    p.add_argument("--trigger_random_seed", type=int, default=1729)
    p.add_argument(
        "--gate_policy",
        type=str,
        default="pathology",
        choices=["pathology", "accept_all_disagreements", "never_accept"],
        help=(
            "DRG accept policy after a triggered retry. 'pathology' is the default gate; "
            "'accept_all_disagreements' removes the pathology gate; 'never_accept' is a preservation control."
        ),
    )
    p.add_argument(
        "--retry_previous_attempt_variant",
        type=str,
        default="real",
        choices=["real", "other_problem", "shuffled_lines", "boxed_region", "no_math_symbols"],
        help="Mechanism-control transform for previous-attempt text when retry_previous_attempt_chars > 0.",
    )
    p.add_argument(
        "--retry_previous_other_shift",
        type=int,
        default=1,
        help="Problem-id shift used by retry_previous_attempt_variant=other_problem within the available baseline rows.",
    )
    p.add_argument("--trigger_rep_threshold", type=float, default=0.7)
    p.add_argument("--trigger_length_percentile", type=float, default=85.0)
    p.add_argument("--trigger_stall_lines", type=int, default=4)
    p.add_argument("--gate_pathology_threshold", type=int, default=2)
    p.add_argument(
        "--methods",
        type=str,
        default="B0,DRG,SC2,SC8,ORACLE,DEPLOY",
        help="Comma-separated subset of methods to materialize for this run.",
    )
    p.add_argument("--length_cutoff_override", type=float, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--skip_report", action="store_true", default=False)
    p.add_argument("--progress_every", type=int, default=5, help="Emit a progress update every N completed problems.")
    p.add_argument("--save_sample_texts", action="store_true", default=False)
    return p.parse_args()


def require_packages() -> None:
    missing = []
    for pkg in ("torch", "transformers"):
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    try:
        import_hf_datasets()
    except Exception:
        missing.append("datasets")
    if missing:
        raise RuntimeError("Missing required packages: " + ", ".join(missing))


def import_hf_datasets():
    repo_root = Path(__file__).resolve().parents[2]
    saved = list(sys.path)
    try:
        cleaned = []
        for entry in saved:
            try:
                resolved = Path(entry or ".").resolve()
            except Exception:
                cleaned.append(entry)
                continue
            if resolved == repo_root:
                continue
            cleaned.append(entry)
        sys.path = cleaned
        ds_mod = importlib.import_module("datasets")
        return ds_mod.DatasetDict, ds_mod.concatenate_datasets, ds_mod.load_from_disk
    finally:
        sys.path = saved


def torch_dtype_from_str(name: str, torch) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_tokenizer_for_model(model_path: Path, AutoTokenizer) -> Any:
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tok = getattr(proc, "tokenizer", None)
        if tok is None:
            raise
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_text_generation_model(
    model_path: Path,
    dtype: Any,
    device: str,
    device_map: Optional[str],
    AutoModelForCausalLM,
) -> Any:
    hf_device_map = device_map if device_map else None
    base_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=hf_device_map,
    )
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                **base_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_path, **base_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "Mistral3Config" not in msg and "Ministral3Config" not in msg:
            raise
        from transformers import Mistral3ForConditionalGeneration

        try:
            wrapped = Mistral3ForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="eager",
                **base_kwargs,
            )
        except TypeError:
            wrapped = Mistral3ForConditionalGeneration.from_pretrained(model_path, **base_kwargs)
        model = getattr(wrapped, "language_model", wrapped)

    if not hf_device_map:
        model = model.to(device)
    model.eval()
    return model


def resolve_generation_device(model, fallback_device: str) -> str:
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        preferred_keys = (
            "model.embed_tokens",
            "language_model.model.embed_tokens",
            "transformer.wte",
            "embed_tokens",
        )
        for key in preferred_keys:
            if key in hf_map:
                target = hf_map[key]
                if isinstance(target, int):
                    return f"cuda:{target}"
                if isinstance(target, str) and target not in {"cpu", "disk"}:
                    return target
        for target in hf_map.values():
            if isinstance(target, int):
                return f"cuda:{target}"
            if isinstance(target, str) and target not in {"cpu", "disk"}:
                return target
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except Exception:
        return fallback_device


def infer_dataset_type(dataset_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    low = str(dataset_path).lower()
    if "gpqa" in low:
        return "gpqa_diamond"
    if "gsm8k" in low:
        return "gsm8k"
    return "math500"


def sanitize_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._") or "run"


def parse_methods(methods: str) -> List[str]:
    wanted = [item.strip().upper() for item in methods.split(",") if item.strip()]
    allowed = {"B0", "DRG", "SC2", "SC8", "ORACLE", "DEPLOY"}
    unknown = [item for item in wanted if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return wanted


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    dataset_tag = sanitize_slug(args.dataset_path.name or "dataset")
    model_tag = sanitize_slug(args.model_path.name or args.model_path.as_posix().split("/")[-1])
    run_tag = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / dataset_tag / model_tag / sanitize_slug(run_tag)


def load_dataset(dataset_path: Path, split: str):
    DatasetDict, concatenate_datasets, load_from_disk = import_hf_datasets()
    ds: DatasetDict = load_from_disk(str(dataset_path))
    split = split.strip()
    if split.lower() == "all":
        keys = list(ds.keys())
        parts = [ds[k] for k in keys]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if "," in split:
        keys = [item.strip() for item in split.split(",") if item.strip()]
        parts = [ds[k] for k in keys]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return ds[split]


def get_example_fields(ex: dict, dataset_type: str, idx: int) -> Tuple[str, str, str]:
    if dataset_type == "gpqa_diamond":
        q = ex.get("question", "")
        a = ex.get("answer", "")
        uid = ex.get("unique_id")
        if not uid:
            md = ex.get("metadata") or {}
            uid = f"gpqa_diamond:{md.get('problem_idx', idx)}"
        return q, a, uid
    if dataset_type == "gsm8k":
        q = ex.get("question", "")
        raw = ex.get("answer", "")
        match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", raw)
        a = match.group(1) if match else raw
        uid = ex.get("unique_id", f"gsm8k:{idx}")
        return q, a, uid
    problem = ex.get("problem") or ex.get("question", "")
    answer = ex.get("answer", "")
    unique_id = ex.get("unique_id")
    if not unique_id:
        raw_id = ex.get("id")
        if raw_id is not None and str(raw_id).strip():
            unique_id = f"math500:{raw_id}"
        else:
            unique_id = f"math500:{idx}"
    return problem, answer, unique_id


def system_prompt(dataset_type: str) -> str:
    if dataset_type == "gpqa_diamond":
        return (
            "You are a helpful assistant. Think step by step and provide your final answer "
            "as a single letter (A, B, C, or D) in \\boxed{}."
        )
    if dataset_type == "gsm8k":
        return "You are a helpful math assistant. Think step by step and give the final answer in \\boxed{}."
    return "You are a helpful math assistant. Think step by step and give the final answer in \\boxed{}."


def build_messages(problem_text: str, dataset_type: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt(dataset_type)},
        {"role": "user", "content": problem_text},
    ]


RESTART_INSTRUCTION = "Restart from scratch. Try again and give your final answer."


def boxed_region(text: str, window: int = 300) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return text[-window:].strip()
    return text[max(0, idx - window // 2) : min(len(text), idx + window)].strip()


def strip_math_symbols(text: str) -> str:
    # Keep natural-language scaffolding while removing equations/symbol-heavy content.
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})*", " ", text)
    text = re.sub(r"[0-9=+\-*/^_<>|()[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transform_previous_attempt_text(text: str, args: argparse.Namespace, *, problem_id: int) -> str:
    variant = getattr(args, "retry_previous_attempt_variant", "real")
    if variant == "real" or not text:
        return text
    if variant == "shuffled_lines":
        import random

        lines = [line for line in text.splitlines() if line.strip()]
        rng = random.Random(int(args.seed) + int(problem_id) * 1009 + 17)
        rng.shuffle(lines)
        return "\n".join(lines)
    if variant == "boxed_region":
        return boxed_region(text)
    if variant == "no_math_symbols":
        return strip_math_symbols(text)
    if variant == "other_problem":
        return text
    raise ValueError(f"Unknown retry_previous_attempt_variant: {variant}")


def build_retry_problem_text(problem_text: str, baseline_text: str, args: argparse.Namespace, *, problem_id: int = 0) -> str:
    parts = [problem_text.strip()]
    if args.retry_instruction_mode == "restart":
        parts.append(RESTART_INSTRUCTION)
    previous_chars = max(0, int(args.retry_previous_attempt_chars))
    if previous_chars > 0:
        previous_text = transform_previous_attempt_text(baseline_text, args, problem_id=problem_id)
        tail = previous_text[-previous_chars:].strip()
        if tail:
            parts.append(f"Previous attempt (may be flawed):\n{tail}")
    return "\n\n".join(part for part in parts if part)


def split_wsc_chunks(text: str, mode: str) -> List[str]:
    """Split while preserving separators so hidden-state probes see the original trace text."""
    if not text:
        return []
    if mode == "line":
        return [chunk for chunk in text.splitlines(keepends=True) if chunk]

    pieces = re.split(r"(\n\s*\n)", text)
    chunks: List[str] = []
    idx = 0
    while idx < len(pieces):
        chunk = pieces[idx]
        if idx + 1 < len(pieces):
            chunk += pieces[idx + 1]
            idx += 2
        else:
            idx += 1
        if chunk:
            chunks.append(chunk)
    if len(chunks) <= 1:
        return [chunk for chunk in text.splitlines(keepends=True) if chunk] or [text]
    return chunks


def load_external_trigger_map(path: Optional[Path]) -> Dict[Any, dict]:
    if path is None:
        raise ValueError("--trigger_source external requires --trigger_external_jsonl")
    rows = read_jsonl(path)
    out: Dict[Any, dict] = {}
    for row in rows:
        keys = []
        if row.get("problem_id") is not None:
            keys.append(int(row["problem_id"]))
        if row.get("unique_id") is not None:
            keys.append(str(row["unique_id"]))
        if not keys:
            raise ValueError(f"External trigger row lacks problem_id/unique_id: {row}")
        for key in keys:
            out[key] = row
    return out


def external_trigger_decision(trigger_map: Dict[Any, dict], ex: dict) -> dict:
    row = trigger_map.get(int(ex["problem_id"]), trigger_map.get(str(ex["unique_id"])))
    if row is None:
        raise KeyError(f"No external trigger decision for problem_id={ex['problem_id']} unique_id={ex['unique_id']}")
    raw = row.get("triggered", row.get("wsc_triggered", row.get("classifier_triggered")))
    if raw is None:
        raise KeyError(f"External trigger row has no triggered/wsc_triggered/classifier_triggered field: {row}")
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "y"}:
            triggered = 1
        elif low in {"0", "false", "no", "n"}:
            triggered = 0
        else:
            raise ValueError(f"Cannot parse external trigger boolean value: {raw!r}")
    else:
        triggered = int(bool(raw))
    return {
        "triggered": triggered,
        "score": row.get("score", row.get("wsc_score", row.get("classifier_score"))),
        "metadata": row,
    }


def random_trigger_decision(problem_id: int, args: argparse.Namespace) -> dict:
    import random

    rate = max(0.0, min(1.0, float(args.trigger_random_rate)))
    rng = random.Random(int(args.trigger_random_seed) + int(problem_id) * 104729)
    score = rng.random()
    return {"triggered": int(score < rate), "score": score, "rate": rate}


def load_wsc_prober(args: argparse.Namespace):
    if args.wsc_probe_path is None:
        raise ValueError("--trigger_source wsc requires --wsc_probe_path")
    repo_path = Path(args.wsc_repo_path).resolve()
    probe_path = Path(args.wsc_probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"WSC probe not found: {probe_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"WordSaladChopper repo not found: {repo_path}")
    sys.path.insert(0, str(repo_path))
    try:
        from wscgen.prober import build_prober
    except Exception as exc:
        raise RuntimeError(f"Could not import wscgen.prober from {repo_path}") from exc
    return build_prober(args.wsc_prober_kind).load(probe_path)


def wsc_trigger_decision(model, tokenizer, problem_text: str, baseline_text: str, dataset_type: str, prober, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    chunks = split_wsc_chunks(baseline_text, args.wsc_split_mode)
    if not chunks:
        return {"triggered": 0, "scores": [], "max_score": 0.0, "chunks": 0}

    input_device = resolve_generation_device(model, args.device)
    messages = build_messages(problem_text, dataset_type)
    chat_kwargs = {}
    if args.enable_thinking != "auto":
        chat_kwargs["enable_thinking"] = args.enable_thinking == "true"
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    all_ids = list(prompt_ids)
    end_positions: List[int] = []
    chunk_token_lengths: List[int] = []
    for chunk in chunks:
        chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
        if not chunk_ids:
            continue
        all_ids.extend(chunk_ids)
        end_positions.append(len(all_ids) - 1)
        chunk_token_lengths.append(len(chunk_ids))
    if not end_positions:
        return {"triggered": 0, "scores": [], "max_score": 0.0, "chunks": 0}

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=input_device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_hidden_states=True, use_cache=False)
    hidden_layer = out.hidden_states[-2][0, end_positions, :].detach().float().cpu().numpy()
    scores = [float(x) for x in np.asarray(prober.predict_proba(hidden_layer)).reshape(-1)]

    long_streak = 0
    short_streak = 0
    triggered = 0
    for score, token_len in zip(scores, chunk_token_lengths):
        if score > args.wsc_threshold:
            if token_len < args.wsc_len_threshold:
                long_streak = 0
                short_streak += 1
                if short_streak >= args.wsc_short_streak_len:
                    triggered = 1
                    break
            else:
                short_streak = 0
                long_streak += 1
                if long_streak >= args.wsc_streak_len:
                    triggered = 1
                    break
        else:
            long_streak = 0
            short_streak = 0

    return {
        "triggered": int(triggered),
        "scores": scores,
        "max_score": max(scores) if scores else 0.0,
        "chunks": len(scores),
        "threshold": float(args.wsc_threshold),
        "streak_len": int(args.wsc_streak_len),
        "short_streak_len": int(args.wsc_short_streak_len),
    }


def extract_last_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    starts = [m.start() for m in re.finditer(re.escape(marker), text)]
    if not starts:
        return None
    for start in reversed(starts):
        idx = start + len(marker)
        depth = 1
        out: List[str] = []
        while idx < len(text):
            ch = text[idx]
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out).strip()
                out.append(ch)
            else:
                out.append(ch)
            idx += 1
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    return None


def extract_gpqa_choice(text: str) -> Optional[str]:
    patterns = [
        r"\\boxed\{([ABCD])\}",
        r"\b(?:answer|final answer|choice)\s*[:=]?\s*([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def answer_surface_text(text: str) -> str:
    """Score only the visible answer after model-internal thinking, when present."""
    marker = "</think>"
    if marker in text:
        return text.rsplit(marker, 1)[-1]
    return text


def extract_predicted_answer(text: str, dataset_type: str) -> Optional[str]:
    text = answer_surface_text(text)
    if dataset_type == "gpqa_diamond":
        return extract_gpqa_choice(text)
    if dataset_type == "gsm8k":
        return extract_numeric_answer(text)
    return extract_last_boxed(text)


def normalize_answer(ans: Optional[str], dataset_type: str) -> Optional[str]:
    if ans is None:
        return None
    text = str(ans).strip()
    if not text:
        return None
    if dataset_type == "gpqa_diamond":
        text = text.upper()
        return text if text in {"A", "B", "C", "D"} else None
    if dataset_type == "gsm8k":
        text = text.replace(",", "")
        return text
    return normalize_math_answer(text)


def answers_equal(pred: Optional[str], gold: Optional[str], dataset_type: str) -> bool:
    if dataset_type == "math500":
        return math_answers_equal(pred, gold)
    return normalize_answer(pred, dataset_type) == normalize_answer(gold, dataset_type)


def repetition_unigram(text: str) -> float:
    toks = [tok for tok in re.split(r"\s+", text.strip()) if tok]
    if not toks:
        return 0.0
    return float(1.0 - len(set(toks)) / len(toks))


def stalled_no_new_symbols(text: str, n_lines: int) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < n_lines + 1:
        return False
    seen = set()
    new_flags = []
    symbol_re = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    for line in lines:
        syms = set(symbol_re.findall(line))
        new_flags.append(bool(syms - seen))
        seen |= syms
    return not any(new_flags[-n_lines:])


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def make_generation_seed(base_seed: int, problem_id: int, sample_idx: int, tag: str) -> int:
    tag_hash = sum(ord(ch) for ch in tag)
    return int(base_seed + problem_id * 10007 + sample_idx * 997 + tag_hash * 131)


def _extract_prompt_tensors(prompt_obj, torch, device: str):
    if isinstance(prompt_obj, torch.Tensor):
        input_ids = prompt_obj.long().to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        return input_ids, attention_mask
    if hasattr(prompt_obj, "get"):
        input_ids = prompt_obj.get("input_ids")
        attention_mask = prompt_obj.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.long().to(device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            else:
                attention_mask = attention_mask.to(device)
            return input_ids, attention_mask
    if isinstance(prompt_obj, dict):
        input_ids = prompt_obj.get("input_ids")
        attention_mask = prompt_obj.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.long().to(device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            else:
                attention_mask = attention_mask.to(device)
            return input_ids, attention_mask
    raise TypeError(f"Unsupported chat template return type: {type(prompt_obj)}")


def generate_one(
    model,
    tokenizer,
    problem_text: str,
    dataset_type: str,
    *,
    max_new_tokens: int,
    device: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    seed: Optional[int],
    enable_thinking: str = "auto",
    assistant_prefix_text: Optional[str] = None,
) -> Tuple[str, int, Optional[str]]:
    import torch

    input_device = resolve_generation_device(model, device)
    messages = build_messages(problem_text, dataset_type)
    chat_kwargs = {}
    if enable_thinking != "auto":
        chat_kwargs["enable_thinking"] = enable_thinking == "true"
    prompt_obj = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **chat_kwargs,
    )
    input_ids, attention_mask = _extract_prompt_tensors(prompt_obj, torch, input_device)
    if assistant_prefix_text:
        prefix_obj = tokenizer(
            assistant_prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prefix_ids = prefix_obj["input_ids"].long().to(input_device)
        prefix_mask = torch.ones_like(prefix_ids, device=input_device)
        input_ids = torch.cat([input_ids, prefix_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, prefix_mask], dim=-1)

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
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if assistant_prefix_text:
        text = assistant_prefix_text + text
    pred = normalize_answer(extract_predicted_answer(text, dataset_type), dataset_type)
    return text, int(gen_ids.shape[0]), pred


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_progress(output_dir: Path, payload: dict) -> None:
    path = output_dir / "progress.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def progress_prefix(args: argparse.Namespace) -> str:
    if args.num_shards > 1:
        return f"[shard {args.shard_id + 1}/{args.num_shards}]"
    return "[run]"


def log_progress(args: argparse.Namespace, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} {progress_prefix(args)} {message}", flush=True)


def update_method_progress(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    method: str,
    completed: int,
    total: int,
    status: str,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "method": method,
        "completed": int(completed),
        "total": int(total),
        "status": status,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    write_progress(output_dir, payload)


def cluster_vote(answers: Sequence[Optional[str]], dataset_type: str) -> Optional[str]:
    clusters: List[dict] = []
    for idx, answer in enumerate(answers):
        if answer is None:
            continue
        matched = False
        for cluster in clusters:
            if answers_equal(answer, cluster["rep"], dataset_type):
                cluster["count"] += 1
                cluster["first_idx"] = min(cluster["first_idx"], idx)
                matched = True
                break
        if not matched:
            clusters.append({"rep": answer, "count": 1, "first_idx": idx})
    if not clusters:
        return None
    clusters.sort(key=lambda item: (-item["count"], item["first_idx"]))
    return clusters[0]["rep"]


def sc2_fallback_choice(sample_answers: Sequence[Optional[str]], dataset_type: str) -> Tuple[Optional[str], str]:
    a1 = sample_answers[0] if len(sample_answers) >= 1 else None
    a2 = sample_answers[1] if len(sample_answers) >= 2 else None
    if a1 is not None and a2 is not None and answers_equal(a1, a2, dataset_type):
        return a1, "both_extractable_agree"
    if a1 is not None and a2 is not None:
        return a1, "both_extractable_disagree_keep_s1"
    if a1 is not None:
        return a1, "only_s1_extractable"
    if a2 is not None:
        return a2, "only_s2_extractable"
    return None, "neither_extractable"


def load_or_build_examples(args: argparse.Namespace) -> List[dict]:
    dataset_type = infer_dataset_type(args.dataset_path, args.dataset_type)
    ds = load_dataset(args.dataset_path, args.split)
    end = len(ds) if args.limit is None else min(len(ds), args.start_idx + args.limit)
    examples = []
    for idx in range(args.start_idx, end):
        if args.num_shards > 1 and (idx % args.num_shards) != args.shard_id:
            continue
        ex = ds[idx]
        problem, gold, unique_id = get_example_fields(ex, dataset_type, idx)
        examples.append(
            {
                "problem_id": idx,
                "unique_id": unique_id,
                "problem": problem,
                "gold_answer": normalize_answer(gold, dataset_type) if dataset_type != "math500" else gold,
                "raw_gold_answer": gold,
                "dataset_type": dataset_type,
            }
        )
    return examples


def load_existing_b0(path: Path, examples_by_pid: Dict[int, dict], dataset_type: str) -> Dict[int, dict]:
    loaded = {}
    for row in read_jsonl(path):
        pid = int(row["problem_id"])
        if pid not in examples_by_pid:
            continue
        text = row.get("generated_text") or row.get("baseline_text") or ""
        pred = row.get("predicted_answer") or row.get("baseline_predicted_answer")
        pred = normalize_answer(pred or extract_predicted_answer(text, dataset_type), dataset_type)
        gold = examples_by_pid[pid]["raw_gold_answer"]
        loaded[pid] = {
            "problem_id": pid,
            "unique_id": examples_by_pid[pid]["unique_id"],
            "dataset_type": dataset_type,
            "ground_truth": gold,
            "generated_text": text,
            "predicted_answer": pred,
            "correct": int(answers_equal(pred, gold, dataset_type)),
            "total_tokens": int(row.get("total_tokens", row.get("baseline_tokens", 0)) or 0),
        }
    return loaded


def load_existing_sc(path: Path, wanted_method: str, examples_by_pid: Dict[int, dict], dataset_type: str) -> Dict[int, dict]:
    loaded = {}
    for row in read_jsonl(path):
        row_method = row.get("method")
        if row_method is not None and row_method != wanted_method:
            continue
        pid = int(row["problem_id"])
        if pid not in examples_by_pid:
            continue
        sample_answers = row.get("sample_answers") or row.get("sc_sample_answers") or []
        sample_answers = [normalize_answer(ans, dataset_type) for ans in sample_answers]
        sample_tokens = row.get("sample_tokens") or row.get("sc_sample_tokens") or []
        sample_texts = row.get("sample_texts") or row.get("generated_texts") or []
        selected = row.get("selected_answer") or row.get("predicted_answer")
        selected = normalize_answer(selected, dataset_type)
        if selected is None and sample_answers:
            if wanted_method == "SC2":
                selected, _ = sc2_fallback_choice(sample_answers, dataset_type)
            else:
                selected = cluster_vote(sample_answers, dataset_type)
        gold = examples_by_pid[pid]["raw_gold_answer"]
        loaded[pid] = {
            "problem_id": pid,
            "unique_id": examples_by_pid[pid]["unique_id"],
            "dataset_type": dataset_type,
            "ground_truth": gold,
            "method": wanted_method,
            "k": int(row.get("k", 2 if wanted_method == "SC2" else 8)),
            "sample_answers": sample_answers,
            "sample_tokens": [int(x) for x in sample_tokens],
            "sample_texts": sample_texts,
            "selected_answer": selected,
            "correct": int(answers_equal(selected, gold, dataset_type)),
            "total_tokens": int(row.get("total_tokens", sum(int(x) for x in sample_tokens)) or 0),
        }
    return loaded


def run_baseline(
    *,
    examples: Sequence[dict],
    examples_by_pid: Dict[int, dict],
    existing: Dict[int, dict],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> Dict[int, dict]:
    rows = dict(existing)
    missing = [ex for ex in examples if ex["problem_id"] not in rows]
    output_dir = Path(args.output_dir)
    method_name = "B0"
    total = len(missing)
    if not missing:
        update_method_progress(args, output_dir=output_dir, method=method_name, completed=0, total=0, status="reused")
        return rows
    log_progress(args, f"{method_name} starting on {total} problems")
    for idx, ex in enumerate(missing, start=1):
        update_method_progress(
            args,
            output_dir=output_dir,
            method=method_name,
            completed=idx - 1,
            total=total,
            status="running",
            extra={"current_problem_id": int(ex["problem_id"])},
        )
        text, tokens, pred = generate_one(
            model,
            tokenizer,
            ex["problem"],
            ex["dataset_type"],
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            do_sample=False,
            temperature=None,
            top_p=None,
            seed=None,
            enable_thinking=args.enable_thinking,
        )
        pid = ex["problem_id"]
        rows[pid] = {
            "problem_id": pid,
            "unique_id": ex["unique_id"],
            "dataset_type": ex["dataset_type"],
            "ground_truth": ex["raw_gold_answer"],
            "generated_text": text,
            "predicted_answer": pred,
            "correct": int(answers_equal(pred, ex["raw_gold_answer"], ex["dataset_type"])),
            "total_tokens": tokens,
        }
        if idx == 1 or idx % max(1, args.progress_every) == 0 or idx == total:
            log_progress(args, f"{method_name} progress {idx}/{total} (pid={pid})")
        update_method_progress(
            args,
            output_dir=output_dir,
            method=method_name,
            completed=idx,
            total=total,
            status="running" if idx < total else "completed",
            extra={"current_problem_id": int(ex["problem_id"])},
        )
    return rows


def run_sc_method(
    *,
    method_name: str,
    k: int,
    examples: Sequence[dict],
    existing: Dict[int, dict],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> Dict[int, dict]:
    rows = dict(existing)
    missing = [ex for ex in examples if ex["problem_id"] not in rows]
    output_dir = Path(args.output_dir)
    total = len(missing)
    if not missing:
        update_method_progress(args, output_dir=output_dir, method=method_name, completed=0, total=0, status="reused")
        return rows
    log_progress(args, f"{method_name} starting on {total} problems with k={k}")
    for idx, ex in enumerate(missing, start=1):
        sample_answers: List[Optional[str]] = []
        sample_tokens: List[int] = []
        sample_texts: List[str] = []
        for sample_idx in range(k):
            update_method_progress(
                args,
                output_dir=output_dir,
                method=method_name,
                completed=idx - 1,
                total=total,
                status="running",
                extra={
                    "current_problem_id": int(ex["problem_id"]),
                    "current_sample": int(sample_idx + 1),
                    "samples_per_problem": int(k),
                },
            )
            text, tokens, pred = generate_one(
                model,
                tokenizer,
                ex["problem"],
                ex["dataset_type"],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                do_sample=True,
                temperature=args.sc_temperature,
                top_p=args.sc_top_p,
                seed=make_generation_seed(args.seed, ex["problem_id"], sample_idx, method_name),
                enable_thinking=args.enable_thinking,
            )
            sample_answers.append(pred)
            sample_tokens.append(tokens)
            if args.save_sample_texts:
                sample_texts.append(text)

        if method_name == "SC2":
            selected_answer, decision = sc2_fallback_choice(sample_answers, ex["dataset_type"])
        else:
            selected_answer = cluster_vote(sample_answers, ex["dataset_type"])
            decision = "majority_vote"
        rows[ex["problem_id"]] = {
            "problem_id": ex["problem_id"],
            "unique_id": ex["unique_id"],
            "dataset_type": ex["dataset_type"],
            "ground_truth": ex["raw_gold_answer"],
            "method": method_name,
            "k": k,
            "sample_answers": sample_answers,
            "sample_tokens": sample_tokens,
            "sample_texts": sample_texts,
            "selected_answer": selected_answer,
            "selection_reason": decision,
            "correct": int(answers_equal(selected_answer, ex["raw_gold_answer"], ex["dataset_type"])),
            "total_tokens": int(sum(sample_tokens)),
        }
        if idx == 1 or idx % max(1, args.progress_every) == 0 or idx == total:
            log_progress(args, f"{method_name} progress {idx}/{total} (pid={ex['problem_id']})")
        update_method_progress(
            args,
            output_dir=output_dir,
            method=method_name,
            completed=idx,
            total=total,
            status="running" if idx < total else "completed",
            extra={
                "current_problem_id": int(ex["problem_id"]),
                "current_sample": int(k),
                "samples_per_problem": int(k),
            },
        )
    return rows


def run_greedy_drg(
    *,
    examples: Sequence[dict],
    baseline_rows: Dict[int, dict],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> Tuple[List[dict], float]:
    baseline_lengths = [int(baseline_rows[ex["problem_id"]]["total_tokens"]) for ex in examples]
    output_dir = Path(args.output_dir)
    if args.length_cutoff_override is not None:
        length_cutoff = float(args.length_cutoff_override)
    else:
        length_cutoff = percentile(baseline_lengths, args.trigger_length_percentile)

    rows = []
    total = len(examples)
    retry_mode = "greedy" if args.retry_temperature <= 0 else f"sampled(T={args.retry_temperature}, top_p={args.retry_top_p})"
    retry_prompt = (
        f"mode={args.retry_prompt_mode}, "
        f"instruction={args.retry_instruction_mode}, "
        f"previous_attempt_chars={args.retry_previous_attempt_chars}, "
        f"previous_attempt_variant={args.retry_previous_attempt_variant}"
    )
    external_triggers = load_external_trigger_map(args.trigger_external_jsonl) if args.trigger_source == "external" else None
    wsc_prober = load_wsc_prober(args) if args.trigger_source == "wsc" else None
    log_progress(
        args,
        f"DRG starting on {total} problems with length cutoff {length_cutoff:.2f}, "
        f"trigger_source={args.trigger_source}, retry={retry_mode}, retry_prompt=({retry_prompt})",
    )
    for idx, ex in enumerate(examples, start=1):
        pid = ex["problem_id"]
        update_method_progress(
            args,
            output_dir=output_dir,
            method="DRG",
            completed=idx - 1,
            total=total,
            status="running",
            extra={"current_problem_id": int(pid), "length_cutoff": float(length_cutoff)},
        )
        base = baseline_rows[pid]
        base_text = base["generated_text"]
        base_pred = base["predicted_answer"]
        base_tokens = int(base["total_tokens"])

        rep = repetition_unigram(base_text)
        rep_hit = int(rep >= args.trigger_rep_threshold)
        length_hit = int(base_tokens >= length_cutoff)
        stall_hit = int(stalled_no_new_symbols(base_text, args.trigger_stall_lines))
        signal_count = rep_hit + length_hit + stall_hit
        heuristic_triggered = int(signal_count >= 1)
        external_score = None
        random_trigger_score = None
        wsc_triggered = None
        wsc_max_score = None
        wsc_chunk_count = None
        wsc_scores = None
        if args.trigger_source == "heuristic":
            triggered = heuristic_triggered
        elif args.trigger_source == "external":
            ext = external_trigger_decision(external_triggers or {}, ex)
            triggered = int(ext["triggered"])
            external_score = ext.get("score")
        elif args.trigger_source == "random":
            rand = random_trigger_decision(pid, args)
            triggered = int(rand["triggered"])
            random_trigger_score = float(rand["score"])
        elif args.trigger_source == "all":
            triggered = 1
        elif args.trigger_source == "none":
            triggered = 0
        elif args.trigger_source == "wsc":
            wsc = wsc_trigger_decision(model, tokenizer, ex["problem"], base_text, ex["dataset_type"], wsc_prober, args)
            wsc_triggered = int(wsc["triggered"])
            wsc_max_score = float(wsc.get("max_score", 0.0))
            wsc_chunk_count = int(wsc.get("chunks", 0))
            wsc_scores = wsc.get("scores")
            triggered = wsc_triggered
        else:
            raise ValueError(f"Unknown trigger_source: {args.trigger_source}")

        retry_text = None
        retry_pred = None
        retry_tokens = 0
        retry_correct = 0
        agree = 0
        accept_retry = 0
        decision_reason = "not_triggered"
        final_source = "baseline"
        final_text = base_text
        final_pred = base_pred
        final_correct = int(base["correct"])

        if triggered:
            retry_do_sample = args.retry_temperature > 0
            context_text = base_text
            if args.retry_previous_attempt_variant == "other_problem":
                available_pids = sorted(int(key) for key in baseline_rows)
                pos = available_pids.index(int(pid))
                other_pid = available_pids[(pos + int(args.retry_previous_other_shift)) % len(available_pids)]
                if other_pid == int(pid) and len(available_pids) > 1:
                    other_pid = available_pids[(pos + 1) % len(available_pids)]
                context_text = baseline_rows[other_pid]["generated_text"]
            retry_problem_text = build_retry_problem_text(ex["problem"], context_text, args, problem_id=pid)
            assistant_prefix_text = None
            if args.retry_prompt_mode == "assistant_continuation":
                retry_problem_text = ex["problem"]
                assistant_prefix_text = base_text
            retry_text, retry_tokens, retry_pred = generate_one(
                model,
                tokenizer,
                retry_problem_text,
                ex["dataset_type"],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                do_sample=retry_do_sample,
                temperature=args.retry_temperature if retry_do_sample else None,
                top_p=args.retry_top_p if retry_do_sample else None,
                seed=make_generation_seed(args.seed, ex["problem_id"], 0, "DRG_RETRY") if retry_do_sample else None,
                enable_thinking=args.enable_thinking,
                assistant_prefix_text=assistant_prefix_text,
            )
            retry_correct = int(answers_equal(retry_pred, ex["raw_gold_answer"], ex["dataset_type"]))
            agree = int(answers_equal(base_pred, retry_pred, ex["dataset_type"]))
            high_pathology = int(signal_count >= args.gate_pathology_threshold)
            accept_disagreement = (
                args.gate_policy == "accept_all_disagreements"
                or (args.gate_policy == "pathology" and high_pathology)
            )
            if agree:
                decision_reason = "agree_same_answer"
                accept_retry = 0
                final_source = "baseline"
            elif accept_disagreement:
                decision_reason = (
                    "disagree_accept_all_no_pathology_gate"
                    if args.gate_policy == "accept_all_disagreements"
                    else "disagree_high_pathology_accept_retry"
                )
                accept_retry = 1
                final_source = "retry"
                final_text = retry_text
                final_pred = retry_pred
                final_correct = retry_correct
            else:
                decision_reason = (
                    "disagree_never_accept_keep_baseline"
                    if args.gate_policy == "never_accept"
                    else "disagree_low_pathology_keep_baseline"
                )
                accept_retry = 0
                final_source = "baseline"
        else:
            high_pathology = 0

        rows.append(
            {
                "problem_id": pid,
                "unique_id": ex["unique_id"],
                "dataset_type": ex["dataset_type"],
                "ground_truth": ex["raw_gold_answer"],
                "baseline_text": base_text,
                "baseline_predicted_answer": base_pred,
                "baseline_correct": int(base["correct"]),
                "baseline_tokens": base_tokens,
                "trigger_rep_unigram": rep,
                "trigger_rep_hit": rep_hit,
                "trigger_length_hit": length_hit,
                "trigger_stall_hit": stall_hit,
                "trigger_signal_count": signal_count,
                "heuristic_triggered": heuristic_triggered,
                "trigger_source": args.trigger_source,
                "external_trigger_score": external_score,
                "random_trigger_score": random_trigger_score,
                "trigger_random_rate": float(args.trigger_random_rate),
                "wsc_triggered": wsc_triggered,
                "wsc_max_score": wsc_max_score,
                "wsc_chunk_count": wsc_chunk_count,
                "wsc_scores": wsc_scores,
                "triggered": triggered,
                "retry_text": retry_text,
                "retry_predicted_answer": retry_pred,
                "retry_correct": retry_correct,
                "retry_tokens": retry_tokens,
                "retry_prompt_mode": args.retry_prompt_mode,
                "retry_instruction_mode": args.retry_instruction_mode,
                "retry_previous_attempt_chars": int(args.retry_previous_attempt_chars),
                "retry_previous_attempt_variant": args.retry_previous_attempt_variant,
                "gate_policy": args.gate_policy,
                "agree_with_baseline_answer": agree,
                "high_pathology": high_pathology,
                "accepted_retry": accept_retry,
                "decision_reason": decision_reason,
                "drg_final_source": final_source,
                "drg_final_text": final_text,
                "drg_final_predicted_answer": final_pred,
                "drg_final_correct": final_correct,
                "total_tokens": base_tokens + retry_tokens,
            }
        )
        if idx == 1 or idx % max(1, args.progress_every) == 0 or idx == total:
            log_progress(args, f"DRG progress {idx}/{total} (pid={pid}, triggered={triggered})")
        update_method_progress(
            args,
            output_dir=output_dir,
            method="DRG",
            completed=idx,
            total=total,
            status="running" if idx < total else "completed",
            extra={"current_problem_id": int(pid), "length_cutoff": float(length_cutoff)},
        )
    return rows, length_cutoff


def build_deploy_records(
    *,
    examples: Sequence[dict],
    drg_rows: Sequence[dict],
    sc2_rows: Dict[int, dict],
) -> List[dict]:
    drg_by_pid = {int(row["problem_id"]): row for row in drg_rows}
    out = []
    for ex in examples:
        pid = ex["problem_id"]
        drg = drg_by_pid[pid]
        sc2 = sc2_rows[pid]
        baseline_pred = drg["baseline_predicted_answer"]
        retry_pred = drg["retry_predicted_answer"]
        path = None
        final_source = None
        final_pred = None
        final_correct = 0
        sc_tokens = 0
        fallback_used = 0
        fallback_reason = None
        fallback_choice = None

        if int(drg["triggered"]) == 0:
            if baseline_pred is not None:
                path = "path_A"
                final_source = "baseline"
                final_pred = baseline_pred
                final_correct = int(drg["baseline_correct"])
            else:
                path = "path_C_nontriggered"
                final_source = "sc2"
                fallback_used = 1
        else:
            if int(drg["agree_with_baseline_answer"]) == 1:
                path = "path_B_agree"
                final_source = "baseline"
                final_pred = baseline_pred
                final_correct = int(drg["baseline_correct"])
            elif int(drg["high_pathology"]) == 1:
                if retry_pred is not None:
                    path = "path_B_accept"
                    final_source = "retry"
                    final_pred = retry_pred
                    final_correct = int(drg["retry_correct"])
                else:
                    path = "path_C_triggered"
                    final_source = "sc2"
                    fallback_used = 1
            else:
                if baseline_pred is not None:
                    path = "path_B_keep"
                    final_source = "baseline"
                    final_pred = baseline_pred
                    final_correct = int(drg["baseline_correct"])
                else:
                    path = "path_C_triggered"
                    final_source = "sc2"
                    fallback_used = 1

        if fallback_used:
            fallback_choice, fallback_reason = sc2_fallback_choice(sc2.get("sample_answers", []), ex["dataset_type"])
            final_pred = fallback_choice
            final_correct = int(answers_equal(final_pred, ex["raw_gold_answer"], ex["dataset_type"]))
            sc_tokens = int(sc2["total_tokens"])

        total_tokens = int(drg["total_tokens"]) + sc_tokens
        drg_pre_fallback_correct = 0
        if path == "path_C_triggered":
            drg_pre_fallback_correct = int(answers_equal(retry_pred, ex["raw_gold_answer"], ex["dataset_type"])) if int(drg["high_pathology"]) == 1 else int(drg["baseline_correct"])
        elif path == "path_C_nontriggered":
            drg_pre_fallback_correct = int(drg["baseline_correct"])

        out.append(
            {
                "problem_id": pid,
                "unique_id": ex["unique_id"],
                "dataset_type": ex["dataset_type"],
                "ground_truth": ex["raw_gold_answer"],
                "path": path,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "fallback_choice": fallback_choice,
                "fallback_from_triggered": int(path == "path_C_triggered"),
                "fallback_from_nontriggered": int(path == "path_C_nontriggered"),
                "baseline_correct": int(drg["baseline_correct"]),
                "drg_final_correct": int(drg["drg_final_correct"]),
                "drg_pre_fallback_correct": drg_pre_fallback_correct,
                "final_correct": final_correct,
                "baseline_predicted_answer": baseline_pred,
                "retry_predicted_answer": retry_pred,
                "drg_final_predicted_answer": drg["drg_final_predicted_answer"],
                "final_predicted_answer": final_pred,
                "baseline_tokens": int(drg["baseline_tokens"]),
                "retry_tokens": int(drg["retry_tokens"]),
                "sc_tokens": sc_tokens,
                "total_tokens": total_tokens,
                "trigger_signals": {
                    "rep_hit": int(drg["trigger_rep_hit"]),
                    "length_hit": int(drg["trigger_length_hit"]),
                    "stall_hit": int(drg["trigger_stall_hit"]),
                    "signal_count": int(drg["trigger_signal_count"]),
                },
            }
        )
    return out


def build_oracle_records(
    *,
    examples: Sequence[dict],
    drg_rows: Sequence[dict],
    sc2_rows: Dict[int, dict],
) -> List[dict]:
    drg_by_pid = {int(row["problem_id"]): row for row in drg_rows}
    out = []
    for ex in examples:
        pid = ex["problem_id"]
        drg = drg_by_pid[pid]
        sc2 = sc2_rows[pid]
        if int(drg["drg_final_correct"]) == 1:
            final_source = "drg"
            final_pred = drg["drg_final_predicted_answer"]
            final_correct = 1
            routed = 0
            total_tokens = int(drg["total_tokens"])
        else:
            final_source = "sc2"
            final_pred = sc2["selected_answer"]
            final_correct = int(sc2["correct"])
            routed = 1
            total_tokens = int(drg["total_tokens"]) + int(sc2["total_tokens"])
        out.append(
            {
                "problem_id": pid,
                "unique_id": ex["unique_id"],
                "dataset_type": ex["dataset_type"],
                "ground_truth": ex["raw_gold_answer"],
                "oracle_routed_to_sc2": routed,
                "drg_final_correct": int(drg["drg_final_correct"]),
                "sc2_correct": int(sc2["correct"]),
                "final_source": final_source,
                "final_predicted_answer": final_pred,
                "final_correct": final_correct,
                "total_tokens": total_tokens,
            }
        )
    return out


def summarize_method(
    records: Sequence[dict],
    *,
    correct_key: str = "correct",
    answer_key: Optional[str] = None,
) -> dict:
    total = len(records)
    correct = int(sum(int(row.get(correct_key, 0)) for row in records))
    avg_tokens = mean([int(row.get("total_tokens", 0)) for row in records]) if records else 0.0
    if answer_key is None:
        final_answers = [
            row.get("final_predicted_answer", row.get("selected_answer", row.get("predicted_answer")))
            for row in records
        ]
    else:
        final_answers = [row.get(answer_key) for row in records]
    no_answer = int(sum(1 for ans in final_answers if ans is None))
    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "avg_tokens": avg_tokens,
        "no_answer_rate": (no_answer / total) if total else 0.0,
    }


def overlap_stats(a_records: Sequence[dict], b_records: Sequence[dict], a_key: str, b_key: str) -> dict:
    both = a_only = b_only = neither = 0
    for a_row, b_row in zip(a_records, b_records):
        a = int(a_row[a_key]) == 1
        b = int(b_row[b_key]) == 1
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    n = len(a_records)
    return {
        "n": n,
        "both": both,
        "drg_only": a_only,
        "sc_only": b_only,
        "neither": neither,
        "union_accuracy": ((both + a_only + b_only) / n) if n else 0.0,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    examples: Sequence[dict],
    b0_records: Sequence[dict],
    drg_records: Sequence[dict],
    sc2_records: Sequence[dict],
    sc8_records: Optional[Sequence[dict]],
    oracle_records: Sequence[dict],
    deploy_records: Sequence[dict],
    length_cutoff: float,
    output_dir: Path,
) -> dict:
    results = {
        "B0": summarize_method(b0_records, answer_key="predicted_answer"),
        "DRG": summarize_method(drg_records, correct_key="drg_final_correct", answer_key="drg_final_predicted_answer"),
        "SC-2": summarize_method(sc2_records, answer_key="selected_answer"),
        "DRG->SC-2 (oracle)": summarize_method(oracle_records, correct_key="final_correct", answer_key="final_predicted_answer"),
        "DRG->SC-2 (deploy)": summarize_method(deploy_records, correct_key="final_correct", answer_key="final_predicted_answer"),
    }
    if sc8_records is not None:
        results["SC-8"] = summarize_method(sc8_records, answer_key="selected_answer")

    sc2_avg = results["SC-2"]["avg_tokens"] or 1.0
    sc8_avg = results.get("SC-8", {}).get("avg_tokens")
    for metrics in results.values():
        metrics["token_ratio_vs_sc2"] = metrics["avg_tokens"] / sc2_avg if sc2_avg else None
        metrics["token_ratio_vs_sc8"] = (metrics["avg_tokens"] / sc8_avg) if sc8_avg else None

    overlap = {
        "DRG_vs_SC2": overlap_stats(drg_records, sc2_records, "drg_final_correct", "correct"),
    }
    if sc8_records is not None:
        overlap["DRG_vs_SC8"] = overlap_stats(drg_records, sc8_records, "drg_final_correct", "correct")

    path_counts = Counter(row["path"] for row in deploy_records)
    fallback_records = [row for row in deploy_records if int(row["fallback_used"]) == 1]
    fallback = {
        "fallback_rate": (len(fallback_records) / len(deploy_records)) if deploy_records else 0.0,
        "fallback_from_triggered": int(sum(int(row["fallback_from_triggered"]) for row in deploy_records)),
        "fallback_from_nontriggered": int(sum(int(row["fallback_from_nontriggered"]) for row in deploy_records)),
        "fallback_rescues": int(
            sum(1 for row in fallback_records if int(row["drg_pre_fallback_correct"]) == 0 and int(row["final_correct"]) == 1)
        ),
        "fallback_worsens": int(
            sum(1 for row in fallback_records if int(row["drg_pre_fallback_correct"]) == 1 and int(row["final_correct"]) == 0)
        ),
    }
    drg_transitions = {
        "C_to_C": int(sum(1 for row in drg_records if int(row.get("baseline_correct", 0)) == 1 and int(row.get("drg_final_correct", 0)) == 1)),
        "C_to_W": int(sum(1 for row in drg_records if int(row.get("baseline_correct", 0)) == 1 and int(row.get("drg_final_correct", 0)) == 0)),
        "W_to_C": int(sum(1 for row in drg_records if int(row.get("baseline_correct", 0)) == 0 and int(row.get("drg_final_correct", 0)) == 1)),
        "W_to_W": int(sum(1 for row in drg_records if int(row.get("baseline_correct", 0)) == 0 and int(row.get("drg_final_correct", 0)) == 0)),
    }

    return {
        "config": {
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "dataset_type": examples[0]["dataset_type"] if examples else infer_dataset_type(args.dataset_path, args.dataset_type),
            "split": args.split,
            "start_idx": args.start_idx,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "system_prompt": system_prompt(examples[0]["dataset_type"] if examples else infer_dataset_type(args.dataset_path, args.dataset_type)),
            "seed": args.seed,
            "reuse_b0_jsonl": str(args.reuse_b0_jsonl) if args.reuse_b0_jsonl else None,
            "reuse_sc2_jsonl": str(args.reuse_sc2_jsonl) if args.reuse_sc2_jsonl else None,
            "reuse_sc8_jsonl": str(args.reuse_sc8_jsonl) if args.reuse_sc8_jsonl else None,
            "skip_sc8": bool(args.skip_sc8),
            "sc_temperature": args.sc_temperature,
            "sc_top_p": args.sc_top_p,
            "retry_temperature": args.retry_temperature,
            "retry_top_p": args.retry_top_p,
            "retry_prompt_mode": args.retry_prompt_mode,
            "retry_instruction_mode": args.retry_instruction_mode,
            "retry_previous_attempt_chars": args.retry_previous_attempt_chars,
            "retry_previous_attempt_variant": args.retry_previous_attempt_variant,
            "retry_previous_other_shift": args.retry_previous_other_shift,
            "trigger_source": args.trigger_source,
            "trigger_external_jsonl": str(args.trigger_external_jsonl) if args.trigger_external_jsonl else None,
            "trigger_random_rate": args.trigger_random_rate,
            "trigger_random_seed": args.trigger_random_seed,
            "gate_policy": args.gate_policy,
            "wsc_probe_path": str(args.wsc_probe_path) if args.wsc_probe_path else None,
            "wsc_threshold": args.wsc_threshold,
            "wsc_streak_len": args.wsc_streak_len,
            "wsc_short_streak_len": args.wsc_short_streak_len,
            "wsc_len_threshold": args.wsc_len_threshold,
            "wsc_split_mode": args.wsc_split_mode,
            "trigger_rep_threshold": args.trigger_rep_threshold,
            "trigger_length_percentile": args.trigger_length_percentile,
            "trigger_stall_lines": args.trigger_stall_lines,
            "gate_pathology_threshold": args.gate_pathology_threshold,
        },
        "output_dir": str(output_dir),
        "problem_count": len(examples),
        "length_cutoff": length_cutoff,
        "results": results,
        "overlap": overlap,
        "fallback": fallback,
        "drg_transitions": drg_transitions,
        "deploy_path_counts": dict(sorted(path_counts.items())),
    }


def render_report(summary: dict) -> str:
    lines = []
    cfg = summary["config"]
    lines.append("# Standalone DRG -> SC-2 Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: `{cfg['model_path']}`")
    lines.append(f"- Dataset: `{cfg['dataset_path']}`")
    lines.append(f"- Dataset type: `{cfg['dataset_type']}`")
    lines.append(f"- Problems: `{summary['problem_count']}`")
    lines.append(f"- Output dir: `{summary['output_dir']}`")
    if cfg.get("system_prompt"):
        lines.append(f"- System prompt: `{cfg['system_prompt']}`")
    if cfg.get("trigger_source", "heuristic") == "heuristic":
        lines.append(f"- Trigger: `rep >= {cfg['trigger_rep_threshold']}` OR `len >= p{cfg['trigger_length_percentile']}` OR `stall(k={cfg['trigger_stall_lines']})`")
    elif cfg.get("trigger_source") == "wsc":
        lines.append(
            f"- Trigger: WSC/Xie learned classifier from `{cfg.get('wsc_probe_path')}` "
            f"with threshold `{cfg.get('wsc_threshold')}`, streak `{cfg.get('wsc_streak_len')}`"
        )
        lines.append(f"- Gate pathology signals: `rep/length/stall`, threshold `{cfg['gate_pathology_threshold']}`")
    else:
        lines.append(f"- Trigger: external decisions from `{cfg.get('trigger_external_jsonl')}`")
        lines.append(f"- Gate pathology signals: `rep/length/stall`, threshold `{cfg['gate_pathology_threshold']}`")
    if cfg.get("trigger_source") == "random":
        lines[-2:] = []
        lines.append(
            f"- Trigger: random deterministic subset with rate `{cfg.get('trigger_random_rate')}` "
            f"and seed `{cfg.get('trigger_random_seed')}`"
        )
        lines.append(f"- Gate pathology signals: `rep/length/stall`, threshold `{cfg['gate_pathology_threshold']}`")
    if cfg.get("trigger_source") == "all":
        lines[-2:] = []
        lines.append("- Trigger: universal, every problem is retried")
        lines.append(f"- Gate pathology signals: `rep/length/stall`, threshold `{cfg['gate_pathology_threshold']}`")
    if cfg.get("trigger_source") == "none":
        lines[-2:] = []
        lines.append("- Trigger: none, no problem is retried")
        lines.append(f"- Gate pathology signals: `rep/length/stall`, threshold `{cfg['gate_pathology_threshold']}`")
    lines.append(f"- Gate: `{cfg.get('gate_policy', 'pathology')}`")
    if float(cfg.get("retry_temperature", 0.0)) > 0:
        lines.append(f"- Retry: sampled regeneration with `T={cfg['retry_temperature']}` and `top_p={cfg['retry_top_p']}`")
    else:
        lines.append("- Retry: strict greedy rerun")
    lines.append(
        f"- Retry prompt ablation: mode `{cfg.get('retry_prompt_mode', 'user')}`, "
        f"instruction `{cfg.get('retry_instruction_mode', 'none')}`, "
        f"previous-attempt tail `{cfg.get('retry_previous_attempt_chars', 0)}` chars, "
        f"variant `{cfg.get('retry_previous_attempt_variant', 'real')}`"
    )
    lines.append(f"- Length cutoff: `{summary['length_cutoff']:.2f}` tokens")
    lines.append("")
    lines.append("## Unified Results")
    lines.append("")
    lines.append("| Method | Accuracy | Correct | Avg tokens | Token ratio vs SC-2 | Token ratio vs SC-8 | No-answer rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    ordered_methods = ["B0", "DRG", "SC-2", "SC-8", "DRG->SC-2 (oracle)", "DRG->SC-2 (deploy)"]
    for method in ordered_methods:
        metrics = summary["results"].get(method)
        if metrics is None:
            continue
        sc8_ratio = metrics["token_ratio_vs_sc8"]
        sc8_ratio_text = f"{sc8_ratio:.3f}" if sc8_ratio is not None else "n/a"
        lines.append(
            f"| {method} | {metrics['accuracy']:.3f} | {metrics['correct']} | {metrics['avg_tokens']:.1f} | "
            f"{metrics['token_ratio_vs_sc2']:.3f} | {sc8_ratio_text} | {metrics['no_answer_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Overlap")
    lines.append("")
    for label, stats in summary["overlap"].items():
        lines.append(f"- {label}: `{stats}`")
    lines.append("")
    lines.append("## DRG Correctness Transitions")
    lines.append("")
    transitions = summary.get("drg_transitions", {})
    lines.append(f"- C->C: `{transitions.get('C_to_C', 0)}`")
    lines.append(f"- C->W: `{transitions.get('C_to_W', 0)}`")
    lines.append(f"- W->C: `{transitions.get('W_to_C', 0)}`")
    lines.append(f"- W->W: `{transitions.get('W_to_W', 0)}`")
    lines.append("")
    lines.append("## Deployable Fallback")
    lines.append("")
    fallback = summary["fallback"]
    lines.append(f"- Fallback rate: `{fallback['fallback_rate']:.3f}`")
    lines.append(f"- Fallback from triggered: `{fallback['fallback_from_triggered']}`")
    lines.append(f"- Fallback from non-triggered: `{fallback['fallback_from_nontriggered']}`")
    lines.append(f"- Fallback rescues: `{fallback['fallback_rescues']}`")
    lines.append(f"- Fallback worsens: `{fallback['fallback_worsens']}`")
    lines.append("")
    lines.append("## Deploy Paths")
    lines.append("")
    for path_name, count in summary["deploy_path_counts"].items():
        lines.append(f"- {path_name}: `{count}`")
    return "\n".join(lines) + "\n"


def save_config(args: argparse.Namespace, output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                key: str(val) if isinstance(val, Path) else val
                for key, val in vars(args).items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    require_packages()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    output_dir = resolve_output_dir(args)
    args.output_dir = output_dir
    records_dir = output_dir / "records"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, output_dir)
    write_progress(output_dir, {"status": "starting", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log_progress(args, f"output dir: {output_dir}")

    examples = load_or_build_examples(args)
    if not examples:
        raise RuntimeError("No examples found for the requested dataset slice.")
    examples_by_pid = {int(ex["problem_id"]): ex for ex in examples}
    dataset_type = examples[0]["dataset_type"]
    log_progress(args, f"loaded {len(examples)} examples for dataset_type={dataset_type}")

    need_b0 = any(method in methods for method in ("B0", "DRG", "ORACLE", "DEPLOY"))
    need_drg = any(method in methods for method in ("DRG", "ORACLE", "DEPLOY"))
    need_sc2 = any(method in methods for method in ("SC2", "ORACLE", "DEPLOY"))
    need_sc8 = ("SC8" in methods) and (not args.skip_sc8)
    need_oracle = "ORACLE" in methods
    need_deploy = "DEPLOY" in methods

    existing_b0 = load_existing_b0(args.reuse_b0_jsonl, examples_by_pid, dataset_type) if (args.reuse_b0_jsonl and need_b0) else {}
    existing_sc2 = load_existing_sc(args.reuse_sc2_jsonl, "SC2", examples_by_pid, dataset_type) if (args.reuse_sc2_jsonl and need_sc2) else {}
    existing_sc8 = load_existing_sc(args.reuse_sc8_jsonl, "SC8", examples_by_pid, dataset_type) if (args.reuse_sc8_jsonl and need_sc8) else {}

    need_model = False
    if need_b0 and len(existing_b0) < len(examples):
        need_model = True
    if need_sc2 and len(existing_sc2) < len(examples):
        need_model = True
    if need_sc8 and len(existing_sc8) < len(examples):
        need_model = True
    if need_drg:
        need_model = True

    model = None
    tokenizer = None
    if need_model:
        log_progress(args, f"loading model from {args.model_path} on device={args.device} device_map={args.device_map}")
        dtype = torch_dtype_from_str(args.dtype, torch)
        tokenizer = load_tokenizer_for_model(args.model_path, AutoTokenizer)
        model = load_text_generation_model(args.model_path, dtype, args.device, args.device_map, AutoModelForCausalLM)
        log_progress(args, "model loaded")

    b0_by_pid = None
    ordered_b0 = None
    if need_b0:
        if model is None and len(existing_b0) < len(examples):
            raise RuntimeError("Baseline generation requested but model is unavailable.")
        b0_by_pid = run_baseline(
            examples=examples,
            examples_by_pid=examples_by_pid,
            existing=existing_b0,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        ordered_b0 = [b0_by_pid[ex["problem_id"]] for ex in examples]
        write_jsonl(records_dir / "b0_records.jsonl", ordered_b0)

    sc2_by_pid = None
    ordered_sc2 = None
    if need_sc2:
        if model is None and len(existing_sc2) < len(examples):
            raise RuntimeError("SC-2 generation requested but model is unavailable.")
        sc2_by_pid = run_sc_method(
            method_name="SC2",
            k=2,
            examples=examples,
            existing=existing_sc2,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        ordered_sc2 = [sc2_by_pid[ex["problem_id"]] for ex in examples]
        write_jsonl(records_dir / "sc2_records.jsonl", ordered_sc2)

    ordered_sc8 = None
    if need_sc8:
        if model is None and len(existing_sc8) < len(examples):
            raise RuntimeError("SC-8 generation requested but model is unavailable.")
        sc8_by_pid = run_sc_method(
            method_name="SC8",
            k=8,
            examples=examples,
            existing=existing_sc8,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        ordered_sc8 = [sc8_by_pid[ex["problem_id"]] for ex in examples]
        write_jsonl(records_dir / "sc8_records.jsonl", ordered_sc8)

    drg_records = None
    if need_drg:
        if b0_by_pid is None:
            raise RuntimeError("DRG requires baseline records.")
        if model is None:
            raise RuntimeError("DRG generation requested but model is unavailable.")
        drg_records, length_cutoff = run_greedy_drg(
            examples=examples,
            baseline_rows=b0_by_pid,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        write_jsonl(records_dir / "drg_records.jsonl", drg_records)
    elif b0_by_pid is not None:
        length_cutoff = (
            float(args.length_cutoff_override)
            if args.length_cutoff_override is not None
            else percentile([int(b0_by_pid[ex["problem_id"]]["total_tokens"]) for ex in examples], args.trigger_length_percentile)
        )
    else:
        length_cutoff = float(args.length_cutoff_override or 0.0)

    deploy_records = None
    if need_deploy:
        if drg_records is None or sc2_by_pid is None:
            raise RuntimeError("DEPLOY requires DRG and SC-2 records.")
        deploy_records = build_deploy_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
        write_jsonl(records_dir / "deploy_records.jsonl", deploy_records)

    oracle_records = None
    if need_oracle:
        if drg_records is None or sc2_by_pid is None:
            raise RuntimeError("ORACLE requires DRG and SC-2 records.")
        oracle_records = build_oracle_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
        write_jsonl(records_dir / "oracle_records.jsonl", oracle_records)

    if not args.skip_report:
        if ordered_b0 is None or drg_records is None or ordered_sc2 is None:
            raise RuntimeError("Report generation requires B0, DRG, and SC-2 records.")
        if deploy_records is None:
            deploy_records = build_deploy_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
            write_jsonl(records_dir / "deploy_records.jsonl", deploy_records)
        if oracle_records is None:
            oracle_records = build_oracle_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
            write_jsonl(records_dir / "oracle_records.jsonl", oracle_records)

        summary = build_summary(
            args=args,
            examples=examples,
            b0_records=ordered_b0,
            drg_records=drg_records,
            sc2_records=ordered_sc2,
            sc8_records=ordered_sc8,
            oracle_records=oracle_records,
            deploy_records=deploy_records,
            length_cutoff=length_cutoff,
            output_dir=output_dir,
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
        log_progress(args, f"report written to {output_dir / 'REPORT.md'}")
    write_progress(
        output_dir,
        {
            "status": "finished",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem_count": len(examples),
        },
    )
    log_progress(args, "run complete")


if __name__ == "__main__":
    main()
