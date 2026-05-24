#!/usr/bin/env python3
"""
Multi-GPU launcher for the standalone DRG -> SC-2 pipeline.

Uses data sharding across GPUs while preserving the global baseline length
cutoff required by the method spec.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    import run_pipeline as rp
else:
    from . import run_pipeline as rp


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
        help="Optional HF device_map to pass through to each worker. Leave unset for data-parallel sharding.",
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
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs_standalone/deployable_drg_sc2"),
    )
    p.add_argument(
        "--gpu_list",
        type=str,
        required=True,
        help=(
            "Comma-separated physical GPU ids for one-GPU workers, e.g. 0,1,2,3. "
            "For model-sharded workers, use semicolon-separated groups, e.g. "
            "'0,1,2,3;4,5,6,7'."
        ),
    )
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--reuse_b0_jsonl", type=Path, default=None)
    p.add_argument("--reuse_sc2_jsonl", type=Path, default=None)
    p.add_argument("--reuse_sc8_jsonl", type=Path, default=None)
    p.add_argument("--skip_sc8", action="store_true", default=False)
    p.add_argument("--sc_temperature", type=float, default=0.7)
    p.add_argument("--sc_top_p", type=float, default=0.95)
    p.add_argument("--retry_temperature", type=float, default=0.7)
    p.add_argument("--retry_top_p", type=float, default=0.95)
    p.add_argument(
        "--retry_instruction_mode",
        type=str,
        default="none",
        choices=["none", "restart"],
        help="Retry prompt instruction ablation.",
    )
    p.add_argument(
        "--retry_prompt_mode",
        type=str,
        default="user",
        choices=["user", "assistant_continuation"],
        help="Retry prompt mode. Use assistant_continuation for E1 verbatim continuation.",
    )
    p.add_argument(
        "--retry_previous_attempt_chars",
        type=int,
        default=0,
        help="Append this many trailing baseline chars to retry prompt.",
    )
    p.add_argument("--trigger_source", type=str, default="heuristic", choices=["heuristic", "wsc", "external", "random", "all", "none"])
    p.add_argument("--trigger_external_jsonl", type=Path, default=None)
    p.add_argument("--wsc_probe_path", type=Path, default=None)
    p.add_argument("--wsc_repo_path", type=Path, default=Path("experiments/convergence/WordSaladChopper"))
    p.add_argument("--wsc_prober_kind", type=str, default="logistic", choices=["logistic", "mlp"])
    p.add_argument("--wsc_threshold", type=float, default=0.5)
    p.add_argument("--wsc_streak_len", type=int, default=2)
    p.add_argument("--wsc_short_streak_len", type=int, default=5)
    p.add_argument("--wsc_len_threshold", type=int, default=10)
    p.add_argument("--wsc_split_mode", type=str, default="paragraph", choices=["paragraph", "line"])
    p.add_argument("--trigger_random_rate", type=float, default=0.5)
    p.add_argument("--trigger_random_seed", type=int, default=1729)
    p.add_argument("--gate_policy", type=str, default="pathology", choices=["pathology", "accept_all_disagreements", "never_accept"])
    p.add_argument(
        "--retry_previous_attempt_variant",
        type=str,
        default="real",
        choices=["real", "other_problem", "shuffled_lines", "boxed_region", "no_math_symbols"],
    )
    p.add_argument("--retry_previous_other_shift", type=int, default=1)
    p.add_argument("--trigger_rep_threshold", type=float, default=0.7)
    p.add_argument("--trigger_length_percentile", type=float, default=85.0)
    p.add_argument("--trigger_stall_lines", type=int, default=4)
    p.add_argument("--gate_pathology_threshold", type=int, default=2)
    p.add_argument("--progress_poll_seconds", type=int, default=20)
    p.add_argument("--save_sample_texts", action="store_true", default=False)
    return p.parse_args()


def save_launcher_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload = {
        key: str(val) if isinstance(val, Path) else val
        for key, val in vars(args).items()
    }
    (output_dir / "launcher_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_ids(args: argparse.Namespace) -> List[str]:
    separator = ";" if ";" in args.gpu_list else ","
    ids = [item.strip() for item in args.gpu_list.split(separator) if item.strip()]
    if not ids:
        raise ValueError("--gpu_list must not be empty")
    return ids


def base_runner_args(args: argparse.Namespace) -> List[str]:
    runner_args = [
        "--model_path",
        str(args.model_path),
        "--dataset_path",
        str(args.dataset_path),
        "--dataset_type",
        args.dataset_type,
        "--split",
        args.split,
        "--start_idx",
        str(args.start_idx),
        "--device",
        args.device,
        "--enable_thinking",
        args.enable_thinking,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--trigger_rep_threshold",
        str(args.trigger_rep_threshold),
        "--trigger_length_percentile",
        str(args.trigger_length_percentile),
        "--trigger_stall_lines",
        str(args.trigger_stall_lines),
        "--gate_pathology_threshold",
        str(args.gate_pathology_threshold),
        "--sc_temperature",
        str(args.sc_temperature),
        "--sc_top_p",
        str(args.sc_top_p),
        "--retry_temperature",
        str(args.retry_temperature),
        "--retry_top_p",
        str(args.retry_top_p),
        "--retry_prompt_mode",
        args.retry_prompt_mode,
        "--retry_instruction_mode",
        args.retry_instruction_mode,
        "--retry_previous_attempt_chars",
        str(args.retry_previous_attempt_chars),
        "--trigger_source",
        args.trigger_source,
        "--wsc_repo_path",
        str(args.wsc_repo_path),
        "--wsc_prober_kind",
        args.wsc_prober_kind,
        "--wsc_threshold",
        str(args.wsc_threshold),
        "--wsc_streak_len",
        str(args.wsc_streak_len),
        "--wsc_short_streak_len",
        str(args.wsc_short_streak_len),
        "--wsc_len_threshold",
        str(args.wsc_len_threshold),
        "--wsc_split_mode",
        args.wsc_split_mode,
        "--trigger_random_rate",
        str(args.trigger_random_rate),
        "--trigger_random_seed",
        str(args.trigger_random_seed),
        "--gate_policy",
        args.gate_policy,
        "--retry_previous_attempt_variant",
        args.retry_previous_attempt_variant,
        "--retry_previous_other_shift",
        str(args.retry_previous_other_shift),
        "--skip_report",
    ]
    if args.trigger_external_jsonl is not None:
        runner_args.extend(["--trigger_external_jsonl", str(args.trigger_external_jsonl)])
    if args.wsc_probe_path is not None:
        runner_args.extend(["--wsc_probe_path", str(args.wsc_probe_path)])
    if args.limit is not None:
        runner_args.extend(["--limit", str(args.limit)])
    if args.device_map is not None:
        runner_args.extend(["--device_map", args.device_map])
    if args.save_sample_texts:
        runner_args.append("--save_sample_texts")
    return runner_args


def launcher_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [launcher] {message}", flush=True)


def read_progress(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_stage_snapshot(stage_name: str, shard_dirs: Sequence[Path]) -> None:
    parts = []
    for shard_dir in shard_dirs:
        shard_name = shard_dir.name
        progress = read_progress(shard_dir / "progress.json")
        if progress is None:
            parts.append(f"{shard_name}:starting")
            continue
        method = progress.get("method", progress.get("status", "unknown"))
        completed = progress.get("completed")
        total = progress.get("total")
        status = progress.get("status", "unknown")
        current_problem_id = progress.get("current_problem_id")
        current_sample = progress.get("current_sample")
        samples_per_problem = progress.get("samples_per_problem")
        if completed is not None and total is not None:
            core = f"{shard_name}:{method}:{completed}/{total}:{status}"
        else:
            core = f"{shard_name}:{method}:{status}"
        if current_problem_id is not None:
            core += f":pid={current_problem_id}"
        if current_sample is not None and samples_per_problem is not None:
            core += f":sample={current_sample}/{samples_per_problem}"
        parts.append(core)
    launcher_log(f"stage={stage_name} " + " | ".join(parts))


def launch_stage(
    *,
    stage_name: str,
    methods: str,
    args: argparse.Namespace,
    output_dir: Path,
    gpus: List[str],
    extra_args: Optional[List[str]] = None,
) -> None:
    stage_root = output_dir / "shards" / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "run_pipeline.py"
    procs = []
    shard_dirs: List[Path] = []
    launcher_log(f"starting stage={stage_name} on {len(gpus)} GPUs")
    for shard_id, gpu in enumerate(gpus):
        shard_dir = stage_root / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_dirs.append(shard_dir)
        cmd = [
            args.python_bin,
            str(script_path),
            *base_runner_args(args),
            "--methods",
            methods,
            "--num_shards",
            str(len(gpus)),
            "--shard_id",
            str(shard_id),
            "--output_dir",
            str(shard_dir),
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = shard_dir / "launcher.log"
        log_handle = log_path.open("w", encoding="utf-8")
        launcher_log(f"launching stage={stage_name} shard={shard_id} gpu={gpu} log={log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_handle, log_path, shard_dir))

    failures = []
    completed = set()
    last_snapshot = 0.0
    while len(completed) < len(procs):
        now = time.time()
        if now - last_snapshot >= max(1, int(args.progress_poll_seconds)):
            print_stage_snapshot(stage_name, shard_dirs)
            last_snapshot = now
        for idx, (proc, handle, log_path, shard_dir) in enumerate(procs):
            if idx in completed:
                continue
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            completed.add(idx)
            progress = read_progress(shard_dir / "progress.json")
            status = progress.get("status") if progress else "unknown"
            launcher_log(f"finished stage={stage_name} shard={idx} exit={code} status={status} log={log_path}")
            if code != 0:
                failures.append((code, log_path))
        if len(completed) < len(procs):
            time.sleep(1)

    if failures:
        details = ", ".join(f"exit={code} log={path}" for code, path in failures)
        raise RuntimeError(f"Stage {stage_name} failed: {details}")
    launcher_log(f"stage={stage_name} complete")


def merge_stage_records(stage_root: Path, filename: str, output_path: Path) -> List[dict]:
    rows = []
    seen = set()
    for shard_dir in sorted(stage_root.glob("shard_*")):
        path = shard_dir / "records" / filename
        if not path.exists():
            continue
        for row in rp.read_jsonl(path):
            pid = int(row["problem_id"])
            if pid in seen:
                raise RuntimeError(f"Duplicate problem_id {pid} while merging {filename}")
            seen.add(pid)
            rows.append(row)
    rows.sort(key=lambda item: int(item["problem_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rp.write_jsonl(output_path, rows)
    return rows


def validate_problem_ids(rows: List[dict], expected_ids: List[int], label: str) -> None:
    actual = [int(row["problem_id"]) for row in rows]
    if actual != expected_ids:
        raise RuntimeError(f"{label} problem ids do not match the requested dataset slice.")


def build_runner_namespace(args: argparse.Namespace, output_dir: Path, length_cutoff: float, skip_sc8: bool) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["output_dir"] = output_dir
    payload["length_cutoff_override"] = length_cutoff
    payload["num_shards"] = 1
    payload["shard_id"] = 0
    payload["methods"] = "B0,DRG,SC2" + ("" if skip_sc8 else ",SC8") + ",ORACLE,DEPLOY"
    payload["skip_report"] = False
    return argparse.Namespace(**payload)


def main() -> None:
    args = parse_args()
    gpus = gpu_ids(args)
    launcher_log(f"output_root={args.output_root} run_name={args.run_name} gpus={','.join(gpus)}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = rp.resolve_output_dir(args)
    records_dir = output_dir / "records"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    save_launcher_config(args, output_dir)

    full_args = argparse.Namespace(**vars(args))
    full_args.output_dir = output_dir
    full_args.num_shards = 1
    full_args.shard_id = 0
    full_examples = rp.load_or_build_examples(full_args)
    expected_ids = [int(ex["problem_id"]) for ex in full_examples]
    dataset_type = full_examples[0]["dataset_type"]

    if args.reuse_b0_jsonl:
        launcher_log(f"reusing baseline from {args.reuse_b0_jsonl}")
        b0_rows = rp.load_existing_b0(args.reuse_b0_jsonl, {ex["problem_id"]: ex for ex in full_examples}, dataset_type)
        merged_b0 = [b0_rows[pid] for pid in expected_ids]
        rp.write_jsonl(records_dir / "b0_records.jsonl", merged_b0)
    else:
        launch_stage(stage_name="b0", methods="B0", args=args, output_dir=output_dir, gpus=gpus)
        merged_b0 = merge_stage_records(output_dir / "shards" / "b0", "b0_records.jsonl", records_dir / "b0_records.jsonl")
        validate_problem_ids(merged_b0, expected_ids, "B0")

    length_cutoff = rp.percentile([int(row["total_tokens"]) for row in merged_b0], args.trigger_length_percentile)
    launcher_log(f"global baseline length cutoff p{args.trigger_length_percentile}={length_cutoff:.2f}")

    if args.reuse_sc2_jsonl:
        launcher_log(f"reusing SC2 from {args.reuse_sc2_jsonl}")
        sc2_rows_map = rp.load_existing_sc(args.reuse_sc2_jsonl, "SC2", {ex["problem_id"]: ex for ex in full_examples}, dataset_type)
        merged_sc2 = [sc2_rows_map[pid] for pid in expected_ids]
        rp.write_jsonl(records_dir / "sc2_records.jsonl", merged_sc2)
    else:
        launch_stage(stage_name="sc2", methods="SC2", args=args, output_dir=output_dir, gpus=gpus)
        merged_sc2 = merge_stage_records(output_dir / "shards" / "sc2", "sc2_records.jsonl", records_dir / "sc2_records.jsonl")
        validate_problem_ids(merged_sc2, expected_ids, "SC2")

    merged_sc8 = None
    if not args.skip_sc8:
        if args.reuse_sc8_jsonl:
            launcher_log(f"reusing SC8 from {args.reuse_sc8_jsonl}")
            sc8_rows_map = rp.load_existing_sc(args.reuse_sc8_jsonl, "SC8", {ex["problem_id"]: ex for ex in full_examples}, dataset_type)
            merged_sc8 = [sc8_rows_map[pid] for pid in expected_ids]
            rp.write_jsonl(records_dir / "sc8_records.jsonl", merged_sc8)
        else:
            launch_stage(stage_name="sc8", methods="SC8", args=args, output_dir=output_dir, gpus=gpus)
            merged_sc8 = merge_stage_records(output_dir / "shards" / "sc8", "sc8_records.jsonl", records_dir / "sc8_records.jsonl")
            validate_problem_ids(merged_sc8, expected_ids, "SC8")

    launch_stage(
        stage_name="drg",
        methods="DRG",
        args=args,
        output_dir=output_dir,
        gpus=gpus,
        extra_args=[
            "--reuse_b0_jsonl",
            str(records_dir / "b0_records.jsonl"),
            "--length_cutoff_override",
            str(length_cutoff),
        ],
    )
    merged_drg = merge_stage_records(output_dir / "shards" / "drg", "drg_records.jsonl", records_dir / "drg_records.jsonl")
    validate_problem_ids(merged_drg, expected_ids, "DRG")

    sc2_by_pid = {int(row["problem_id"]): row for row in merged_sc2}
    deploy_records = rp.build_deploy_records(examples=full_examples, drg_rows=merged_drg, sc2_rows=sc2_by_pid)
    oracle_records = rp.build_oracle_records(examples=full_examples, drg_rows=merged_drg, sc2_rows=sc2_by_pid)
    rp.write_jsonl(records_dir / "deploy_records.jsonl", deploy_records)
    rp.write_jsonl(records_dir / "oracle_records.jsonl", oracle_records)

    summary_args = build_runner_namespace(args, output_dir, length_cutoff, args.skip_sc8)
    summary = rp.build_summary(
        args=summary_args,
        examples=full_examples,
        b0_records=merged_b0,
        drg_records=merged_drg,
        sc2_records=merged_sc2,
        sc8_records=merged_sc8,
        oracle_records=oracle_records,
        deploy_records=deploy_records,
        length_cutoff=length_cutoff,
        output_dir=output_dir,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(rp.render_report(summary), encoding="utf-8")
    launcher_log(f"final report written to {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
