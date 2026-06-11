#!/usr/bin/env python3
"""Evaluate FinanceBench result JSONL files with deterministic metrics.

The script is intentionally dependency-free so it can run in a fresh checkout.
It supports the result files already checked into this repository and future
prediction files that contain a gold answer plus a model answer.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UNIT_PATTERN = (
    r"(?:usd(?:\s+(?:millions?|billions?))?|dollars?|millions?|billions?|"
    r"shares?|bps|basis points)"
)
MONEY_SCALE_RE = re.compile(
    r"\b(?:usd\s+)?(?P<scale>millions?|billions?)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    rf"(?<![A-Za-z_])(?P<number>\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?)(?![A-Za-z_])"
    rf"(?:\s*(?P<unit>{UNIT_PATTERN}))?",
    re.IGNORECASE,
)
NUMERIC_ANSWER_RE = re.compile(
    r"^\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?"
    rf"(?:\s*{UNIT_PATTERN})?$",
    re.IGNORECASE,
)
FILING_BLOCK_RE = re.compile(
    r"\[START OF FILING\].*?\[END OF FILING\]",
    re.IGNORECASE | re.DOTALL,
)
FILING_START_RE = re.compile(r"\[START OF FILING\].*$", re.IGNORECASE | re.DOTALL)
FILING_END_RE = re.compile(r"^.*?\[END OF FILING\]", re.IGNORECASE | re.DOTALL)
CONTEXT_YEAR_RE = re.compile(r"\b(?:FY|fiscal year)\s*\d{2,4}\b", re.IGNORECASE)
BARE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_NAME_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)
CONTEXT_DATE_RE = re.compile(
    rf"\b{MONTH_NAME_PATTERN}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+"
    r"(?:19|20)\d{2}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_NAME_PATTERN}\.?,?\s+"
    r"(?:19|20)\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b",
    re.IGNORECASE,
)
ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n|\b)(?:final\s+(?:answer|result)|answer|result)\s*"
    r"(?:is|was|were|would\s+be|:|=)\s*",
    re.IGNORECASE | re.DOTALL,
)
CALCULATION_RESULT_RE = re.compile(
    r"[^=\n]*(?:[/+*]|\s-\s)[^=\n]*=\s*"
    r"(?P<value>\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?)"
    rf"(?:\s*(?P<unit>{UNIT_PATTERN}))?"
    r"(?=\s*(?:[.,;:]|\n|$))",
    re.IGNORECASE,
)
REFUSAL_PHRASES = (
    "as an ai",
    "cannot provide",
    "can't provide",
    "do not have access",
    "don't have access",
    "doesn't provide",
    "does not provide",
    "i don't have",
    "i do not have",
    "no information",
    "no explicit mention",
    "no explicit information",
    "not enough information",
    "not provided",
    "does not explicitly mention",
    "doesn't explicitly mention",
    "does not explicitly outline",
    "doesn't explicitly outline",
    "does not explicitly state",
    "doesn't explicitly state",
    "recommend checking",
    "unable to",
)


@dataclass
class FileSummary:
    path: Path
    rows: int
    deterministic_matches: int
    deterministic_scored: int
    labels: Counter[str]


@dataclass(frozen=True)
class NumberCandidate:
    value: float
    is_percent: bool = False
    unit: str | None = None


def normalize_text(value: Any) -> str:
    """Return a compact, case-insensitive representation for exact matching."""

    return " ".join(str(value).strip().lower().split())


def infer_answer_money_scale(question: Any) -> str:
    """Infer whether numeric gold answers are expressed in millions or billions."""

    text = normalize_text(question)
    matches = list(MONEY_SCALE_RE.finditer(text))
    if not matches:
        return "millions"

    scale = matches[-1].group("scale")
    if scale.lower().startswith("billion"):
        return "billions"
    return "millions"


def expects_money_answer(question: Any, gold_answer: Any) -> bool:
    text = normalize_text(question)
    if MONEY_SCALE_RE.search(text) or "usd" in text or "dollar" in text:
        return True

    gold_text = normalize_text(gold_answer)
    return "$" in str(gold_answer) or bool(MONEY_SCALE_RE.search(gold_text))


def magnitude_multiplier(
    unit: str | None,
    *,
    answer_money_scale: str = "millions",
) -> float:
    """Return a multiplier that normalizes common money magnitudes to the answer scale."""

    if not unit:
        return 1.0

    normalized_unit = normalize_text(unit)
    if "billion" in normalized_unit:
        return 1.0 if answer_money_scale == "billions" else 1000.0
    if "million" in normalized_unit:
        return 0.001 if answer_money_scale == "billions" else 1.0
    return 1.0


def parse_number_candidate(
    token: str,
    unit: str | None = None,
    *,
    answer_money_scale: str = "millions",
) -> NumberCandidate | None:
    """Parse a currency/percent/parenthesized number token with unit metadata."""

    text = token.strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    is_percent = "%" in text
    text = text.replace("$", "").replace(",", "").replace("%", "")

    try:
        value = float(text)
    except ValueError:
        return None

    if is_percent:
        value /= 100.0

    value *= magnitude_multiplier(unit, answer_money_scale=answer_money_scale)

    if negative:
        value = -value

    return NumberCandidate(
        value=value,
        is_percent=is_percent,
        unit=normalize_text(unit) if unit else None,
    )


def parse_number_token(
    token: str,
    unit: str | None = None,
    *,
    answer_money_scale: str = "millions",
) -> float | None:
    """Parse a currency/percent/parenthesized number token into a float."""

    candidate = parse_number_candidate(
        token,
        unit,
        answer_money_scale=answer_money_scale,
    )
    return candidate.value if candidate else None


def extract_number_candidates(
    value: Any,
    *,
    answer_money_scale: str = "millions",
) -> list[NumberCandidate]:
    """Extract numeric tokens with unit metadata."""

    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return [NumberCandidate(float(value))]
        return []

    candidates: list[NumberCandidate] = []
    for match in NUMBER_RE.finditer(str(value)):
        parsed = parse_number_candidate(
            match.group("number"),
            match.group("unit"),
            answer_money_scale=answer_money_scale,
        )
        if parsed is not None:
            candidates.append(parsed)
    return candidates


def extract_numbers(value: Any, *, answer_money_scale: str = "millions") -> list[float]:
    """Extract numeric tokens from strings or return numeric values directly."""

    return [
        candidate.value
        for candidate in extract_number_candidates(
            value,
            answer_money_scale=answer_money_scale,
        )
    ]


def strip_filing_context(value: str) -> str:
    text = FILING_BLOCK_RE.sub(" ", value)
    text = FILING_START_RE.sub(" ", text)
    text = FILING_END_RE.sub(" ", text)
    return text.strip()


def strip_temporal_context(
    value: str,
    *,
    answer_money_scale: str = "millions",
) -> str:
    """Remove date/year context when another numeric answer candidate remains."""

    text = value
    without_full_dates = CONTEXT_DATE_RE.sub(" ", text)
    if extract_numbers(without_full_dates, answer_money_scale=answer_money_scale):
        text = without_full_dates

    text = CONTEXT_YEAR_RE.sub(" ", text)
    numbers = extract_numbers(text, answer_money_scale=answer_money_scale)
    if len(numbers) <= 1:
        return text

    without_bare_years = BARE_YEAR_RE.sub(" ", text)
    if extract_numbers(without_bare_years, answer_money_scale=answer_money_scale):
        return without_bare_years

    return text


def extract_answer_number_candidates(
    value: Any,
    *,
    answer_money_scale: str = "millions",
) -> list[NumberCandidate]:
    """Extract answer-like numeric candidates, not copied context."""

    if not isinstance(value, str):
        return extract_number_candidates(value, answer_money_scale=answer_money_scale)

    text = strip_filing_context(value)
    if not text:
        return []

    marker_matches = list(ANSWER_MARKER_RE.finditer(text))
    if marker_matches:
        return extract_number_candidates(
            strip_temporal_context(
                text[marker_matches[-1].end() :],
                answer_money_scale=answer_money_scale,
            ),
            answer_money_scale=answer_money_scale,
        )[:1]

    calculation_matches = list(CALCULATION_RESULT_RE.finditer(text))
    if calculation_matches:
        parsed = parse_number_candidate(
            calculation_matches[-1].group("value"),
            calculation_matches[-1].group("unit"),
            answer_money_scale=answer_money_scale,
        )
        if parsed is not None:
            return [parsed]

    candidates = extract_number_candidates(
        strip_temporal_context(text, answer_money_scale=answer_money_scale),
        answer_money_scale=answer_money_scale,
    )
    return candidates if len(candidates) <= 1 else []


def extract_answer_numbers(
    value: Any,
    *,
    answer_money_scale: str = "millions",
) -> list[float]:
    """Extract numbers that are likely to be the answer, not copied context."""

    return [
        candidate.value
        for candidate in extract_answer_number_candidates(
            value,
            answer_money_scale=answer_money_scale,
        )
    ]


def is_numeric_answer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return bool(NUMERIC_ANSWER_RE.fullmatch(str(value).strip()))


def looks_like_refusal(value: Any) -> bool:
    text = normalize_text(value)
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def numbers_match(
    expected: float,
    observed: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return math.isclose(
        observed,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def is_money_unit(unit: str | None) -> bool:
    return bool(unit) and any(
        marker in unit
        for marker in ("usd", "dollar", "million", "billion")
    )


def has_compatible_unit(
    expected: NumberCandidate,
    observed: NumberCandidate,
    *,
    answer_expects_money: bool,
) -> bool:
    if not answer_expects_money:
        return True

    if observed.is_percent:
        return False
    if observed.unit and not is_money_unit(observed.unit):
        return False

    return True


def rounded_billion_tolerance(
    gold_answer: Any,
    answer_money_scale: str,
) -> float | None:
    if answer_money_scale != "billions":
        return None

    text = str(gold_answer).strip()
    match = re.search(r"[-+]?\d+\.(?P<decimals>\d+)", text)
    if not match:
        return None

    return 0.5 * 10 ** (-len(match.group("decimals")))


def deterministic_match(
    gold_answer: Any,
    model_answer: Any,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    question: Any = None,
) -> bool:
    """Score a response without an LLM judge.

    Numeric gold answers are matched against the response's answer span.
    Non-numeric gold answers fall back to exact normalized string equality.
    This deliberately avoids claiming semantic equivalence for free-form text.
    """

    if not is_numeric_answer(gold_answer):
        if looks_like_refusal(model_answer):
            return False
        return normalize_text(gold_answer) == normalize_text(model_answer)

    answer_money_scale = infer_answer_money_scale(question)
    answer_expects_money = expects_money_answer(question, gold_answer)
    effective_absolute_tolerance = max(
        absolute_tolerance,
        rounded_billion_tolerance(gold_answer, answer_money_scale) or 0.0,
    )
    expected_numbers = extract_number_candidates(
        gold_answer,
        answer_money_scale=answer_money_scale,
    )
    observed_numbers = extract_answer_number_candidates(
        model_answer,
        answer_money_scale=answer_money_scale,
    )

    if expected_numbers:
        matched_pairs = [
            (expected, observed)
            for expected in expected_numbers
            for observed in observed_numbers
            if numbers_match(
                expected.value,
                observed.value,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=effective_absolute_tolerance,
            )
            and has_compatible_unit(
                expected,
                observed,
                answer_expects_money=answer_expects_money,
            )
        ]
        if not matched_pairs:
            return False

        if looks_like_refusal(model_answer):
            return any(
                numbers_match(
                    0.0,
                    expected.value,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=effective_absolute_tolerance,
                )
                and numbers_match(
                    0.0,
                    observed.value,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=effective_absolute_tolerance,
                )
                for expected, observed in matched_pairs
            )

        return True

    return False


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def get_gold_answer(row: dict[str, Any]) -> Any:
    if "gold_answer" in row:
        return row["gold_answer"]
    if "answer" in row:
        return row["answer"]
    raise KeyError("missing gold_answer or answer")


def get_model_answer(row: dict[str, Any]) -> Any:
    for field in (
        "final_answer",
        "model_answer",
        "prediction",
        "predicted_answer",
        "response",
    ):
        if field in row:
            return row[field]
    raise KeyError(
        "missing final_answer, model_answer, prediction, predicted_answer, or response"
    )


def summarize_file(
    path: Path,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> FileSummary:
    labels: Counter[str] = Counter()
    rows = 0
    deterministic_scored = 0
    deterministic_matches = 0

    for row in read_jsonl(path):
        rows += 1
        label = row.get("label")
        if label is not None:
            labels[str(label)] += 1

        try:
            gold_answer = get_gold_answer(row)
            model_answer = get_model_answer(row)
        except KeyError:
            continue

        deterministic_scored += 1
        if deterministic_match(
            gold_answer,
            model_answer,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            question=row.get("question"),
        ):
            deterministic_matches += 1

    return FileSummary(
        path=path,
        rows=rows,
        deterministic_matches=deterministic_matches,
        deterministic_scored=deterministic_scored,
        labels=labels,
    )


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Input file(s) not found: {missing_list}")

    return paths


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"


def render_table(summaries: list[FileSummary]) -> str:
    headers = [
        "file",
        "rows",
        "human_correct",
        "human_incorrect",
        "human_refusal",
        "deterministic_match",
    ]
    rows = []
    totals: defaultdict[str, int] = defaultdict(int)

    for summary in summaries:
        human_correct = summary.labels.get("Correct Answer", 0)
        human_incorrect = summary.labels.get("Incorrect Answer", 0)
        human_refusal = summary.labels.get("Refusal", 0)
        rows.append(
            [
                str(summary.path),
                str(summary.rows),
                percent(human_correct, summary.rows),
                percent(human_incorrect, summary.rows),
                percent(human_refusal, summary.rows),
                percent(summary.deterministic_matches, summary.deterministic_scored),
            ]
        )
        totals["rows"] += summary.rows
        totals["human_correct"] += human_correct
        totals["human_incorrect"] += human_incorrect
        totals["human_refusal"] += human_refusal
        totals["deterministic_matches"] += summary.deterministic_matches
        totals["deterministic_scored"] += summary.deterministic_scored

    if len(summaries) > 1:
        rows.append(
            [
                "TOTAL",
                str(totals["rows"]),
                percent(totals["human_correct"], totals["rows"]),
                percent(totals["human_incorrect"], totals["rows"]),
                percent(totals["human_refusal"], totals["rows"]),
                percent(
                    totals["deterministic_matches"],
                    totals["deterministic_scored"],
                ),
            ]
        )

    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def render_json(summaries: list[FileSummary]) -> str:
    payload = []
    for summary in summaries:
        payload.append(
            {
                "file": str(summary.path),
                "rows": summary.rows,
                "labels": dict(summary.labels),
                "deterministic_matches": summary.deterministic_matches,
                "deterministic_scored": summary.deterministic_scored,
                "deterministic_match_rate": (
                    summary.deterministic_matches / summary.deterministic_scored
                    if summary.deterministic_scored
                    else None
                ),
            }
        )
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinanceBench result or prediction JSONL files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL file paths or glob patterns, for example results/*.jsonl.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.001,
        help="Relative tolerance for numeric deterministic matching.",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numeric deterministic matching.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paths = expand_paths(args.inputs)
    summaries = [
        summarize_file(
            path,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        for path in paths
    ]

    if args.format == "json":
        print(render_json(summaries))
    else:
        print(render_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
