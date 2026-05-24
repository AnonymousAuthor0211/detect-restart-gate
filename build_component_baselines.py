#!/usr/bin/env python3
"""Post-hoc component baselines from one completed standalone run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(name: str, correct: List[int], tokens: List[int], baseline_correct: List[int]) -> dict:
    n = len(correct)
    return {
        "name": name,
        "accuracy": sum(correct) / n if n else 0.0,
        "correct": sum(correct),
        "avg_tokens": sum(tokens) / n if n else 0.0,
        "c_to_w": sum(1 for b, f in zip(baseline_correct, correct) if b == 1 and f == 0),
        "w_to_c": sum(1 for b, f in zip(baseline_correct, correct) if b == 0 and f == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = args.run_dir / "records"
    b0 = {int(row["problem_id"]): row for row in read_jsonl(records / "b0_records.jsonl")}
    drg = {int(row["problem_id"]): row for row in read_jsonl(records / "drg_records.jsonl")}
    sc2 = {int(row["problem_id"]): row for row in read_jsonl(records / "sc2_records.jsonl")}
    pids = sorted(set(b0) & set(drg) & set(sc2))
    baseline_correct = [int(b0[pid]["correct"]) for pid in pids]

    rows = []
    rows.append(
        summarize(
            "B0",
            baseline_correct,
            [int(b0[pid]["total_tokens"]) for pid in pids],
            baseline_correct,
        )
    )
    rows.append(
        summarize(
            "DRG",
            [int(drg[pid]["drg_final_correct"]) for pid in pids],
            [int(drg[pid]["total_tokens"]) for pid in pids],
            baseline_correct,
        )
    )
    rows.append(
        summarize(
            "SC-2",
            [int(sc2[pid]["correct"]) for pid in pids],
            [int(sc2[pid]["total_tokens"]) for pid in pids],
            baseline_correct,
        )
    )

    selective_correct = []
    selective_tokens = []
    high_path_correct = []
    high_path_tokens = []
    for pid in pids:
        if int(drg[pid].get("triggered", 0)) == 1:
            selective_correct.append(int(sc2[pid]["correct"]))
            selective_tokens.append(int(b0[pid]["total_tokens"]) + int(sc2[pid]["total_tokens"]))
        else:
            selective_correct.append(int(b0[pid]["correct"]))
            selective_tokens.append(int(b0[pid]["total_tokens"]))

        if int(drg[pid].get("high_pathology", 0)) == 1:
            high_path_correct.append(int(sc2[pid]["correct"]))
            high_path_tokens.append(int(b0[pid]["total_tokens"]) + int(sc2[pid]["total_tokens"]))
        else:
            high_path_correct.append(int(b0[pid]["correct"]))
            high_path_tokens.append(int(b0[pid]["total_tokens"]))

    rows.append(summarize("selective_SC2_on_triggered", selective_correct, selective_tokens, baseline_correct))
    rows.append(summarize("selective_SC2_on_high_pathology", high_path_correct, high_path_tokens, baseline_correct))

    lines = [
        "# Component Baselines",
        "",
        f"- Run dir: `{args.run_dir}`",
        f"- Problems: `{len(pids)}`",
        "",
        "| Method | Accuracy | Correct | Avg tokens | C->W | W->C |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['accuracy']:.3f} | {row['correct']} | "
            f"{row['avg_tokens']:.1f} | {row['c_to_w']} | {row['w_to_c']} |"
        )
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
