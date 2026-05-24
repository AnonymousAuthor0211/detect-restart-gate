#!/usr/bin/env python3
"""
Portable answer normalization helpers for math-style evaluation.
"""

from __future__ import annotations

import signal
import re
from typing import Optional

_MATH_EQUAL_FN = None
_MATH_EQUAL_IMPORT_TRIED = False
_MATH_EQUAL_TIMEOUT_SECONDS = 5


class _MathEqualTimeout(Exception):
    pass


def _raise_math_equal_timeout(signum, frame):
    raise _MathEqualTimeout()


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    if len(parts) <= 1:
        return s
    for part in parts[1:]:
        out += "\\frac"
        if part.startswith("{"):
            out += part
            continue
        if len(part) < 2:
            return s
        a = part[0]
        b = part[1]
        tail = part[2:] if len(part) > 2 else ""
        if b != "{":
            out += "{" + a + "}{" + b + "}" + tail
        else:
            out += "{" + a + "}" + b + tail
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", s)


def _fix_simple_slash_fraction(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        if s == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        pass
    return s


def _fix_mixed_number(s: str) -> str:
    m = re.match(r"^(-?\d+)\\frac\{(\d+)\}\{(\d+)\}$", s)
    if m:
        whole = int(m.group(1))
        numer = int(m.group(2))
        denom = int(m.group(3))
        sign = -1 if whole < 0 else 1
        new_numer = abs(whole) * denom + numer
        return f"\\frac{{{sign * new_numer}}}{{{denom}}}"
    return s


def _strip_function_prefix(s: str) -> str:
    return re.sub(r"^[a-zA-Z]\([a-zA-Z]\)=", "", s)


def _unwrap_latex_text(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    return s


def _strip_trailing_units(s: str) -> str:
    unit_tokens = [
        "gallons?",
        "miles?",
        "meters?",
        "meter",
        "feet",
        "foot",
        "inches?",
        "inch",
        "seconds?",
        "second",
        "hours?",
        "hour",
        "cm",
        "mm",
        "km",
        "m",
        "grade",
        "units?",
        "per",
    ]
    pat = re.compile(r"(?i)(?:" + "|".join(unit_tokens) + r")(?:\^\{?\d+\}?|\d+)?$")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\+$", "", s)
        s = pat.sub("", s)
    return s


def normalize_math_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None

    raw = str(ans).strip()
    if not raw:
        return None

    had_decorators = bool(re.search(r"\\(?:text|mathrm)\{|\\\$|\$|\\circ", raw))

    s = raw
    s = s.replace("\n", "")
    s = re.sub(r"\\+text\{", r"\\text{", s)
    s = re.sub(r"\\+mathrm\{", r"\\mathrm{", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = _unwrap_latex_text(s)

    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\ ", "")
    s = s.replace(" ", "")

    s = re.sub(r"(?i)\^\{?(st|nd|rd|th)\}?", "", s)
    s = re.sub(r"(?i)(st|nd|rd|th)grade$", "", s)

    s = _strip_function_prefix(s)
    s = _fix_sqrt(s)
    s = _fix_fracs(s)
    s = _fix_mixed_number(s)
    s = _fix_simple_slash_fraction(s)

    s = re.sub(r"\\(?=\-?\d+(\\|\)|,|\]|$))", "", s)

    if had_decorators:
        s = _strip_trailing_units(s)

    s = re.sub(r"\s+", "", s)
    return s if s else None


def _get_skythought_math_equal():
    global _MATH_EQUAL_FN, _MATH_EQUAL_IMPORT_TRIED
    if _MATH_EQUAL_IMPORT_TRIED:
        return _MATH_EQUAL_FN
    _MATH_EQUAL_IMPORT_TRIED = True
    try:
        from skythought.evals.util.math_parsing_util import math_equal as _math_equal  # type: ignore

        _MATH_EQUAL_FN = _math_equal
    except Exception:
        try:
            from symbolic_math import math_equal as _math_equal  # type: ignore

            _MATH_EQUAL_FN = _math_equal
        except Exception:
            try:
                from .symbolic_math import math_equal as _math_equal  # type: ignore

                _MATH_EQUAL_FN = _math_equal
            except Exception:
                try:
                    from experiments.convergence.WordSaladChopper.wscgen.utils import math_equal as _math_equal  # type: ignore

                    _MATH_EQUAL_FN = _math_equal
                except Exception:
                    _MATH_EQUAL_FN = None
    return _MATH_EQUAL_FN


def math_answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True

    math_equal_fn = _get_skythought_math_equal()
    if math_equal_fn is None:
        return False
    old_handler = None
    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_math_equal_timeout)
        signal.alarm(_MATH_EQUAL_TIMEOUT_SECONDS)
        return bool(math_equal_fn(p, g))
    except Exception:
        return False
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
