#!/usr/bin/env python3
"""Rebuild standalone DRG reports from completed JSONL records.

This is intentionally generation-free. Use it when a multi-GPU launcher run
finished its shard work but failed while assembling the final summary/report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import run_pipeline as rp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument("--dataset_type", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--reuse_b0_jsonl", type=Path, default=None)
    parser.add_argument("--reuse_sc2_jsonl", type=Path, default=None)
    parser.add_argument("--reuse_sc8_jsonl", type=Path, default=None)
    parser.add_argument("--skip_sc8", action="store_true", default=None)
    parser.add_argument("--sc_temperature", type=float, default=None)
    parser.add_argument("--sc_top_p", type=float, default=None)
    parser.add_argument("--retry_temperature", type=float, default=None)
    parser.add_argument("--retry_top_p", type=float, default=None)
    parser.add_argument("--retry_prompt_mode", type=str, default=None, choices=["user", "assistant_continuation"])
    parser.add_argument("--retry_instruction_mode", type=str, default=None, choices=["none", "restart"])
    parser.add_argument("--retry_previous_attempt_chars", type=int, default=None)
    parser.add_argument(
        "--retry_previous_attempt_variant",
        type=str,
        default=None,
        choices=["real", "other_problem", "shuffled_lines", "boxed_region", "no_math_symbols"],
    )
    parser.add_argument("--retry_previous_other_shift", type=int, default=None)
    parser.add_argument("--trigger_source", type=str, default=None, choices=["heuristic", "wsc", "external", "random", "all", "none"])
    parser.add_argument("--trigger_external_jsonl", type=Path, default=None)
    parser.add_argument("--trigger_random_rate", type=float, default=None)
    parser.add_argument("--trigger_random_seed", type=int, default=None)
    parser.add_argument("--gate_policy", type=str, default=None, choices=["pathology", "accept_all_disagreements", "never_accept"])
    parser.add_argument("--wsc_probe_path", type=Path, default=None)
    parser.add_argument("--wsc_threshold", type=float, default=None)
    parser.add_argument("--wsc_streak_len", type=int, default=None)
    parser.add_argument("--wsc_short_streak_len", type=int, default=None)
    parser.add_argument("--wsc_len_threshold", type=int, default=None)
    parser.add_argument("--wsc_split_mode", type=str, default=None, choices=["paragraph", "line"])
    parser.add_argument("--trigger_rep_threshold", type=float, default=None)
    parser.add_argument("--trigger_length_percentile", type=float, default=None)
    parser.add_argument("--trigger_stall_lines", type=int, default=None)
    parser.add_argument("--gate_pathology_threshold", type=int, default=None)
    return parser.parse_args()


def load_config(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "launcher_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coalesce(cli_value: Any, config: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config and config[key] is not None:
        return config[key]
    return default


def as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def build_namespace(cli: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    output_dir = cli.output_dir
    skip_sc8 = cli.skip_sc8
    if skip_sc8 is None:
        skip_sc8 = bool(config.get("skip_sc8", False))
    payload = {
        "model_path": as_path(coalesce(cli.model_path, config, "model_path", None)),
        "dataset_path": as_path(coalesce(cli.dataset_path, config, "dataset_path", None)),
        "dataset_type": coalesce(cli.dataset_type, config, "dataset_type", "auto"),
        "split": coalesce(cli.split, config, "split", "all"),
        "start_idx": int(coalesce(cli.start_idx, config, "start_idx", 0)),
        "limit": coalesce(cli.limit, config, "limit", None),
        "max_new_tokens": int(coalesce(cli.max_new_tokens, config, "max_new_tokens", 4096)),
        "device": coalesce(cli.device, config, "device", "cuda"),
        "device_map": coalesce(cli.device_map, config, "device_map", None),
        "dtype": coalesce(cli.dtype, config, "dtype", "bfloat16"),
        "seed": int(coalesce(cli.seed, config, "seed", 0)),
        "run_name": config.get("run_name"),
        "output_dir": output_dir,
        "output_root": as_path(coalesce(cli.output_root, config, "output_root", Path("outputs_standalone/deployable_drg_sc2"))),
        "reuse_b0_jsonl": as_path(coalesce(cli.reuse_b0_jsonl, config, "reuse_b0_jsonl", None)),
        "reuse_sc2_jsonl": as_path(coalesce(cli.reuse_sc2_jsonl, config, "reuse_sc2_jsonl", None)),
        "reuse_sc8_jsonl": as_path(coalesce(cli.reuse_sc8_jsonl, config, "reuse_sc8_jsonl", None)),
        "skip_sc8": skip_sc8,
        "sc_temperature": float(coalesce(cli.sc_temperature, config, "sc_temperature", 0.7)),
        "sc_top_p": float(coalesce(cli.sc_top_p, config, "sc_top_p", 0.95)),
        "retry_temperature": float(coalesce(cli.retry_temperature, config, "retry_temperature", 0.7)),
        "retry_top_p": float(coalesce(cli.retry_top_p, config, "retry_top_p", 0.95)),
        "retry_prompt_mode": coalesce(cli.retry_prompt_mode, config, "retry_prompt_mode", "user"),
        "retry_instruction_mode": coalesce(cli.retry_instruction_mode, config, "retry_instruction_mode", "none"),
        "retry_previous_attempt_chars": int(coalesce(cli.retry_previous_attempt_chars, config, "retry_previous_attempt_chars", 0)),
        "retry_previous_attempt_variant": coalesce(cli.retry_previous_attempt_variant, config, "retry_previous_attempt_variant", "real"),
        "retry_previous_other_shift": int(coalesce(cli.retry_previous_other_shift, config, "retry_previous_other_shift", 1)),
        "trigger_source": coalesce(cli.trigger_source, config, "trigger_source", "heuristic"),
        "trigger_external_jsonl": as_path(coalesce(cli.trigger_external_jsonl, config, "trigger_external_jsonl", None)),
        "trigger_random_rate": float(coalesce(cli.trigger_random_rate, config, "trigger_random_rate", 0.5)),
        "trigger_random_seed": int(coalesce(cli.trigger_random_seed, config, "trigger_random_seed", 1729)),
        "gate_policy": coalesce(cli.gate_policy, config, "gate_policy", "pathology"),
        "wsc_probe_path": as_path(coalesce(cli.wsc_probe_path, config, "wsc_probe_path", None)),
        "wsc_threshold": float(coalesce(cli.wsc_threshold, config, "wsc_threshold", 0.5)),
        "wsc_streak_len": int(coalesce(cli.wsc_streak_len, config, "wsc_streak_len", 2)),
        "wsc_short_streak_len": int(coalesce(cli.wsc_short_streak_len, config, "wsc_short_streak_len", 5)),
        "wsc_len_threshold": int(coalesce(cli.wsc_len_threshold, config, "wsc_len_threshold", 10)),
        "wsc_split_mode": coalesce(cli.wsc_split_mode, config, "wsc_split_mode", "paragraph"),
        "trigger_rep_threshold": float(coalesce(cli.trigger_rep_threshold, config, "trigger_rep_threshold", 0.7)),
        "trigger_length_percentile": float(coalesce(cli.trigger_length_percentile, config, "trigger_length_percentile", 85.0)),
        "trigger_stall_lines": int(coalesce(cli.trigger_stall_lines, config, "trigger_stall_lines", 4)),
        "gate_pathology_threshold": int(coalesce(cli.gate_pathology_threshold, config, "gate_pathology_threshold", 2)),
        "length_cutoff_override": None,
        "num_shards": 1,
        "shard_id": 0,
        "methods": "B0,DRG,SC2,SC8,ORACLE,DEPLOY",
        "skip_report": False,
        "progress_every": 5,
        "save_sample_texts": False,
    }
    missing = [key for key in ("model_path", "dataset_path") if payload[key] is None]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")
    return argparse.Namespace(**payload)


def summarize_validation(drg_records: list[dict], deploy_records: list[dict]) -> dict:
    triggered = [row for row in drg_records if int(row.get("triggered", 0)) == 1]
    triggered_with_text = [
        row for row in triggered if row.get("baseline_text") is not None and row.get("retry_text") is not None
    ]
    same_text = sum(1 for row in triggered_with_text if row.get("baseline_text") == row.get("retry_text"))
    different_text = len(triggered_with_text) - same_text
    answer_disagree = sum(
        1
        for row in triggered
        if row.get("baseline_predicted_answer") != row.get("retry_predicted_answer")
    )
    fixed = sum(
        1
        for row in drg_records
        if int(row.get("baseline_correct", 0)) == 0 and int(row.get("drg_final_correct", 0)) == 1
    )
    worsened = sum(
        1
        for row in drg_records
        if int(row.get("baseline_correct", 0)) == 1 and int(row.get("drg_final_correct", 0)) == 0
    )
    return {
        "n": len(drg_records),
        "baseline_correct": int(sum(int(row.get("baseline_correct", 0)) for row in drg_records)),
        "drg_correct": int(sum(int(row.get("drg_final_correct", 0)) for row in drg_records)),
        "triggered": len(triggered),
        "triggered_with_text": len(triggered_with_text),
        "triggered_same_text": int(same_text),
        "triggered_different_text": int(different_text),
        "triggered_answer_disagreement": int(answer_disagree),
        "accepted_retry": int(sum(int(row.get("accepted_retry", 0)) for row in drg_records)),
        "drg_fixes": int(fixed),
        "drg_worsens": int(worsened),
        "drg_final_source_counts": dict(Counter(str(row.get("drg_final_source")) for row in drg_records)),
        "decision_reason_counts": dict(Counter(str(row.get("decision_reason")) for row in drg_records)),
        "deploy_path_counts": dict(Counter(str(row.get("path")) for row in deploy_records)),
    }


def render_validation(metrics: dict, summary: dict) -> str:
    n = metrics["n"] or 1
    lines = [
        "# Clean T=0.7 Retry Validation",
        "",
        "## Core Checks",
        "",
        f"- B0 accuracy: `{metrics['baseline_correct'] / n:.3f}` ({metrics['baseline_correct']}/{metrics['n']})",
        f"- DRG accuracy: `{metrics['drg_correct'] / n:.3f}` ({metrics['drg_correct']}/{metrics['n']})",
        f"- DRG fixes / worsens: `{metrics['drg_fixes']}` / `{metrics['drg_worsens']}`",
        f"- Triggered rows: `{metrics['triggered']}`",
        f"- Triggered retry text divergence: `{metrics['triggered_different_text']}/{metrics['triggered_with_text']}`",
        f"- Triggered answer disagreements: `{metrics['triggered_answer_disagreement']}`",
        f"- Accepted retries: `{metrics['accepted_retry']}`",
        "",
        "## Pathology / Routing Counts",
        "",
        f"- DRG final source counts: `{metrics['drg_final_source_counts']}`",
        f"- DRG decision reason counts: `{metrics['decision_reason_counts']}`",
        f"- Deploy path counts: `{metrics['deploy_path_counts']}`",
        "",
        "## Interpretation",
        "",
    ]
    b0_acc = metrics["baseline_correct"] / n
    if abs(b0_acc - 0.718) <= 0.01:
        lines.append("- The regenerated B0 is close to the canonical `71.8%` baseline, so this run is suitable for direct paper-aligned comparison.")
    else:
        lines.append("- The regenerated B0 is still not close to the canonical `71.8%` baseline, so treat direct comparison to the original paper artifacts with caution.")
    lines.append("- The retry is original-problem-only with `T=0.7`; there is no scaffold, previous attempt, or intervention instruction.")
    lines.append(f"- Report summary is in `{summary['output_dir']}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    cli = parse_args()
    output_dir = cli.output_dir
    records_dir = output_dir / "records"
    config = load_config(output_dir)
    args = build_namespace(cli, config)

    examples = rp.load_or_build_examples(args)
    b0_records = rp.read_jsonl(records_dir / "b0_records.jsonl")
    drg_records = rp.read_jsonl(records_dir / "drg_records.jsonl")
    sc2_records = rp.read_jsonl(records_dir / "sc2_records.jsonl")
    sc8_path = records_dir / "sc8_records.jsonl"
    sc8_records = None if args.skip_sc8 or not sc8_path.exists() else rp.read_jsonl(sc8_path)
    sc2_by_pid = {int(row["problem_id"]): row for row in sc2_records}

    deploy_path = records_dir / "deploy_records.jsonl"
    if deploy_path.exists():
        deploy_records = rp.read_jsonl(deploy_path)
    else:
        deploy_records = rp.build_deploy_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
        rp.write_jsonl(deploy_path, deploy_records)

    oracle_path = records_dir / "oracle_records.jsonl"
    if oracle_path.exists():
        oracle_records = rp.read_jsonl(oracle_path)
    else:
        oracle_records = rp.build_oracle_records(examples=examples, drg_rows=drg_records, sc2_rows=sc2_by_pid)
        rp.write_jsonl(oracle_path, oracle_records)

    length_cutoff = rp.percentile([int(row["total_tokens"]) for row in b0_records], args.trigger_length_percentile)
    summary = rp.build_summary(
        args=args,
        examples=examples,
        b0_records=b0_records,
        drg_records=drg_records,
        sc2_records=sc2_records,
        sc8_records=sc8_records,
        oracle_records=oracle_records,
        deploy_records=deploy_records,
        length_cutoff=length_cutoff,
        output_dir=output_dir,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(rp.render_report(summary), encoding="utf-8")

    validation = summarize_validation(drg_records, deploy_records)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "VALIDATION.md").write_text(render_validation(validation, summary), encoding="utf-8")
    print(f"rebuilt report: {output_dir / 'REPORT.md'}")
    print(f"validation: {output_dir / 'VALIDATION.md'}")


if __name__ == "__main__":
    main()
