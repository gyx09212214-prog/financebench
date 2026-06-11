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


NUMBER_RE = re.compile(r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?")
NUMERIC_ANSWER_RE = re.compile(
    r"^\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?"
    r"(?:\s*(?:usd|dollars?|millions?|billions?|shares?|bps|basis points))?$",
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
    "not enough information",
    "not provided",
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


def normalize_text(value: Any) -> str:
    """Return a compact, case-insensitive representation for exact matching."""

    return " ".join(str(value).strip().lower().split())


def parse_number_token(token: str) -> float | None:
    """Parse a currency/percent/parenthesized number token into a float."""

    text = token.strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("$", "").replace(",", "").replace("%", "")

    try:
        value = float(text)
    except ValueError:
        return None

    return -value if negative else value


def extract_numbers(value: Any) -> list[float]:
    """Extract numeric tokens from strings or return numeric values directly."""

    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return [float(value)]
        return []

    numbers: list[float] = []
    for token in NUMBER_RE.findall(str(value)):
        parsed = parse_number_token(token)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


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


def deterministic_match(
    gold_answer: Any,
    model_answer: Any,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    """Score a response without an LLM judge.

    Numeric gold answers are matched against any number in the model answer.
    Non-numeric gold answers fall back to exact normalized string equality.
    This deliberately avoids claiming semantic equivalence for free-form text.
    """

    if looks_like_refusal(model_answer):
        return False

    if not is_numeric_answer(gold_answer):
        return normalize_text(gold_answer) == normalize_text(model_answer)

    expected_numbers = extract_numbers(gold_answer)
    observed_numbers = extract_numbers(model_answer)

    if expected_numbers:
        return any(
            numbers_match(
                expected,
                observed,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            for expected in expected_numbers
            for observed in observed_numbers
        )

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
    for field in ("model_answer", "prediction", "predicted_answer", "response"):
        if field in row:
            return row[field]
    raise KeyError("missing model_answer, prediction, predicted_answer, or response")


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
        default=0.01,
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
