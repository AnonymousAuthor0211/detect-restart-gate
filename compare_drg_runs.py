#!/usr/bin/env python3
"""Compare two completed standalone DRG runs with paired statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def drg_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "records" / "drg_records.jsonl",
        run_dir / "drg_records.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find drg_records.jsonl under {run_dir}")


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def accuracy(rows: Iterable[dict], key: str = "drg_final_correct") -> tuple[int, int, float]:
    rows = list(rows)
    correct = sum(int(row.get(key, 0)) for row in rows)
    total = len(rows)
    return correct, total, correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="First run dir.")
    parser.add_argument("--b", type=Path, required=True, help="Second run dir.")
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    a_rows = {int(row["problem_id"]): row for row in read_jsonl(drg_path(args.a))}
    b_rows = {int(row["problem_id"]): row for row in read_jsonl(drg_path(args.b))}
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        raise RuntimeError("No overlapping problem_id values.")

    a_common = [a_rows[pid] for pid in common]
    b_common = [b_rows[pid] for pid in common]
    a_correct, total, a_acc = accuracy(a_common)
    b_correct, _, b_acc = accuracy(b_common)

    a_only = 0
    b_only = 0
    both = 0
    neither = 0
    trigger_both = 0
    trigger_a_only = 0
    trigger_b_only = 0
    trigger_neither = 0
    for pid in common:
        ac = int(a_rows[pid].get("drg_final_correct", 0))
        bc = int(b_rows[pid].get("drg_final_correct", 0))
        if ac and bc:
            both += 1
        elif ac and not bc:
            a_only += 1
        elif bc and not ac:
            b_only += 1
        else:
            neither += 1

        at = int(a_rows[pid].get("triggered", 0))
        bt = int(b_rows[pid].get("triggered", 0))
        if at and bt:
            trigger_both += 1
        elif at and not bt:
            trigger_a_only += 1
        elif bt and not at:
            trigger_b_only += 1
        else:
            trigger_neither += 1

    def count(rows: List[dict], pred) -> int:
        return sum(1 for row in rows if pred(row))

    lines = [
        f"# DRG Run Comparison: {args.label_a} vs {args.label_b}",
        "",
        "## Setup",
        "",
        f"- {args.label_a}: `{args.a}`",
        f"- {args.label_b}: `{args.b}`",
        f"- Common problems: `{len(common)}`",
        "",
        "## Accuracy",
        "",
        "| Run | Accuracy | Correct | Triggered | Accepted Retry | Fixes | Worsens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows, correct, acc in [
        (args.label_a, a_common, a_correct, a_acc),
        (args.label_b, b_common, b_correct, b_acc),
    ]:
        triggered = count(rows, lambda row: int(row.get("triggered", 0)) == 1)
        accepted = count(rows, lambda row: int(row.get("accepted_retry", 0)) == 1)
        fixes = count(rows, lambda row: int(row.get("baseline_correct", 0)) == 0 and int(row.get("drg_final_correct", 0)) == 1)
        worsens = count(rows, lambda row: int(row.get("baseline_correct", 0)) == 1 and int(row.get("drg_final_correct", 0)) == 0)
        lines.append(f"| {label} | {acc:.3f} | {correct}/{total} | {triggered} | {accepted} | {fixes} | {worsens} |")

    p_value = exact_mcnemar_p(a_only, b_only)
    lines.extend(
        [
            "",
            "## Paired Correctness",
            "",
            f"- Both correct: `{both}`",
            f"- {args.label_a}-only correct: `{a_only}`",
            f"- {args.label_b}-only correct: `{b_only}`",
            f"- Neither correct: `{neither}`",
            f"- Exact two-sided McNemar p-value: `{p_value}`",
            "",
            "## Trigger Overlap",
            "",
            f"- Both triggered: `{trigger_both}`",
            f"- {args.label_a}-only triggered: `{trigger_a_only}`",
            f"- {args.label_b}-only triggered: `{trigger_b_only}`",
            f"- Neither triggered: `{trigger_neither}`",
        ]
    )

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
