#!/usr/bin/env python3
"""Aggregate multiple standalone runs for seed-variance reporting."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


METHODS = ["B0", "DRG", "SC-2", "DRG->SC-2 (oracle)", "DRG->SC-2 (deploy)", "SC-8"]


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def fmt_pm(values: Sequence[float], scale: float = 1.0, digits: int = 3) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0] * scale:.{digits}f}"
    return f"{mean(values) * scale:.{digits}f} ± {stdev(values) * scale:.{digits}f}"


def paired_bootstrap_ci(diffs: Sequence[float], *, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    vals = []
    for _ in range(n_boot):
        vals.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def paired_stats(run_dir: Path) -> dict:
    drg_path = run_dir / "records" / "drg_records.jsonl"
    sc2_path = run_dir / "records" / "sc2_records.jsonl"
    if not drg_path.exists() or not sc2_path.exists():
        return {}
    drg = {int(row["problem_id"]): row for row in read_jsonl(drg_path)}
    sc2 = {int(row["problem_id"]): row for row in read_jsonl(sc2_path)}
    common = sorted(set(drg) & set(sc2))
    drg_only = 0
    sc2_only = 0
    diffs = []
    for pid in common:
        dc = int(drg[pid].get("drg_final_correct", 0))
        sc = int(sc2[pid].get("correct", 0))
        diffs.append(dc - sc)
        if dc and not sc:
            drg_only += 1
        elif sc and not dc:
            sc2_only += 1
    lo, hi = paired_bootstrap_ci(diffs, seed=13)
    return {
        "n": len(common),
        "drg_only": drg_only,
        "sc2_only": sc2_only,
        "mcnemar_p": exact_mcnemar_p(drg_only, sc2_only),
        "diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "bootstrap_ci": [lo, hi],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Completed standalone run directories.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    paired = []
    for run_dir in args.runs:
        summary = load_summary(run_dir)
        summary["_run_dir"] = str(run_dir)
        summaries.append(summary)
        ps = paired_stats(run_dir)
        if ps:
            ps["_run_dir"] = str(run_dir)
            paired.append(ps)

    lines = ["# Seed Variance Aggregate", "", "## Runs", ""]
    for item in summaries:
        cfg = item.get("config", {})
        lines.append(f"- `{item['_run_dir']}` seed `{cfg.get('seed')}`")
    lines.extend(["", "## Mean ± Std", "", "| Method | Accuracy | Avg tokens | Correct |", "| --- | ---: | ---: | ---: |"])

    for method in METHODS:
        accs = []
        toks = []
        corrects = []
        for summary in summaries:
            metrics = summary.get("results", {}).get(method)
            if not metrics:
                continue
            accs.append(float(metrics.get("accuracy", 0.0)))
            toks.append(float(metrics.get("avg_tokens", 0.0)))
            corrects.append(float(metrics.get("correct", 0.0)))
        if accs:
            lines.append(f"| {method} | {fmt_pm(accs)} | {fmt_pm(toks, digits=1)} | {fmt_pm(corrects, digits=1)} |")

    if paired:
        lines.extend(
            [
                "",
                "## Paired DRG vs SC-2",
                "",
                "| Run | DRG-only | SC-2-only | Acc diff | Bootstrap 95% CI | McNemar p |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in paired:
            lo, hi = item["bootstrap_ci"]
            lines.append(
                f"| `{item['_run_dir']}` | {item['drg_only']} | {item['sc2_only']} | "
                f"{item['diff']:.3f} | [{lo:.3f}, {hi:.3f}] | {item['mcnemar_p']:.3g} |"
            )

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
