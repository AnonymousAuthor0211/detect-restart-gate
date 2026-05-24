#!/usr/bin/env python3
"""
Portable math equivalence helpers.

This is a lightweight local fallback for symbolic equivalence when the
SkyThought evaluator is unavailable. It is adapted from the local
WordSaladChopper math utilities so the standalone pipeline can travel
without depending on the full experiments tree.
"""

from __future__ import annotations

import re
from math import isclose
from typing import Optional

from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def choice_answer_clean(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    choices = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if choices:
        pred = choices[-1]
    else:
        pred = pred.strip().strip(".")
    return pred.rstrip(".").rstrip("/")


def parse_digits(num: object) -> Optional[float]:
    text = re.sub(r",", "", str(num))
    try:
        return float(text)
    except Exception:
        if text.endswith("%"):
            text = text[:-1]
            if text.endswith("\\"):
                text = text[:-1]
            try:
                return float(text) / 100.0
            except Exception:
                return None
    return None


def is_digit(num: object) -> bool:
    return parse_digits(num) is not None


def str_to_pmatrix(input_str: str) -> str:
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for match in matrix_str:
        stripped = match.strip("{}")
        pmatrix = r"\begin{pmatrix}" + stripped.replace(",", "\\\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a: str, b: str) -> bool:
    def _parse(text: str):
        for parser in (parse_latex, parse_expr, latex2sympy):
            try:
                return parser(text.replace("\\\\", "\\"))
            except Exception:
                try:
                    return parser(text)
                except Exception:
                    pass
        return text

    lhs = _parse(a)
    rhs = _parse(b)

    try:
        if str(lhs) == str(rhs) or lhs == rhs:
            return True
    except Exception:
        pass

    try:
        if lhs.equals(rhs) or simplify(lhs - rhs) == 0:
            return True
    except Exception:
        pass

    try:
        if abs(lhs.lhs - lhs.rhs).equals(abs(rhs.lhs - rhs.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(lhs)), float(N(rhs))):
            return True
    except Exception:
        pass

    try:
        if lhs.shape == rhs.shape:
            lhs_r = lhs.applyfunc(lambda x: round(x, 3))
            rhs_r = rhs.applyfunc(lambda x: round(x, 3))
            if lhs_r.equals(rhs_r):
                return True
    except Exception:
        pass

    return False


def math_equal(
    prediction: Optional[str],
    reference: Optional[str],
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if reference in {"A", "B", "C", "D", "E"} and choice_answer_clean(str(prediction)) == reference:
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)
            if pred_num is not None and ref_num is not None:
                candidates = [ref_num / 100.0, ref_num, ref_num * 100.0] if include_percentage else [ref_num]
                for item in candidates:
                    try:
                        if is_close and numeric_equal(pred_num, item):
                            return True
                        if (not is_close) and item == pred_num:
                            return True
                    except Exception:
                        continue
                return False
    except Exception:
        pass

    pred = str(prediction).strip()
    ref = str(reference).strip()
    if not pred:
        return False

    if "pmatrix" in pred and "pmatrix" not in ref:
        ref = str_to_pmatrix(ref)

    pred_str, ref_str = pred, ref
    if (
        pred.startswith("[")
        and pred.endswith("]")
        and not ref.startswith("(")
    ) or (
        pred.startswith("(")
        and pred.endswith(")")
        and not ref.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for token in ["{", "}", "(", ")"]:
        pred_str = pred_str.replace(token, "")
        ref_str = ref_str.replace(token, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if re.match(r"(\(|\[).+(\)|\])", pred) and re.match(r"(\(|\[).+(\)|\])", ref):
        pred_parts = pred[1:-1].split(",")
        ref_parts = ref[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))):
                return True

    if (
        (pred.startswith("\\begin{pmatrix}") or pred.startswith("\\begin{bmatrix}"))
        and (pred.endswith("\\end{pmatrix}") or pred.endswith("\\end{bmatrix}"))
        and (ref.startswith("\\begin{pmatrix}") or ref.startswith("\\begin{bmatrix}"))
        and (ref.endswith("\\end{pmatrix}") or ref.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [line.strip() for line in pred[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in ref[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        if len(pred_lines) == len(ref_lines):
            matched = True
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) != len(ref_parts):
                    matched = False
                    break
                if not all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))):
                    matched = False
                    break
            if matched:
                return True

    if pred.count("=") == 1 and ref.count("=") == 1:
        pred_eq = pred.split("=")
        pred_expr = f"{pred_eq[0].strip()} - ({pred_eq[1].strip()})"
        ref_eq = ref.split("=")
        ref_expr = f"{ref_eq[0].strip()} - ({ref_eq[1].strip()})"
        if symbolic_equal(pred_expr, ref_expr) or symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif pred.count("=") == 1 and len(pred.split("=")[0].strip()) <= 2 and "=" not in ref:
        if math_equal(pred.split("=")[1], ref, include_percentage, is_close):
            return True
    elif ref.count("=") == 1 and len(ref.split("=")[0].strip()) <= 2 and "=" not in pred:
        if math_equal(pred, ref.split("=")[1], include_percentage, is_close):
            return True

    return symbolic_equal(pred, ref)
