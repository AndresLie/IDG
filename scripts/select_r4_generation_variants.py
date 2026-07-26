from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one generated variant per matched R4 sample.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in NAME=METADATA_JSONL form; provide at least two.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [_parse_candidate(value) for value in args.candidate]
    if len(candidates) < 2:
        raise ValueError("At least two --candidate values are required")

    indexed = {name: {_key(row): row for row in _read_rows(path)} for name, path in candidates}
    key_sets = [set(rows) for rows in indexed.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Candidate metadata cohorts do not match")

    selected_rows: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(key_sets[0]):
        options = [(name, indexed[name][key]) for name, _path in candidates]
        name, selected = max(options, key=lambda item: _rank(_selected_attempt(item[1])))
        record = json.loads(json.dumps(selected))
        record["latency_sec"] = sum(float(option_row.get("latency_sec", 0.0)) for _option_name, option_row in options)
        record.setdefault("settings", {})["generation_variant_arbitration"] = {
            "selected_candidate": name,
            "policy": "accepted_then_critic_score_then_visibility_target_distance",
            "candidates": {
                option_name: {
                    "variant": option_row["variant"],
                    "output_path": option_row["output_path"],
                    "latency_sec": float(option_row.get("latency_sec", 0.0)),
                    "attempt_count": len(option_row["critic_guided_generation"]["attempts"]),
                    "attempt": _selected_attempt(option_row),
                }
                for option_name, option_row in options
            },
        }
        selected_rows.append(record)
        selection_counts[name] += 1
        category_counts[str(record["category"])][name] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    report_path = args.output.with_suffix(".md")
    report_path.write_text(
        _report(selection_counts, category_counts, candidates, len(selected_rows)),
        encoding="utf-8",
    )
    print(args.output)


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"Invalid --candidate {value!r}; expected NAME=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row["category"]),
        str(row["defect_type"]),
        str(row["background_path"]),
        str(row["quality_profile"]),
        int(row["generation_seed"]),
    )


def _selected_attempt(row: dict[str, Any]) -> dict[str, Any]:
    critic = row["critic_guided_generation"]
    selected = int(critic["selected_attempt_index"])
    return next(attempt for attempt in critic["attempts"] if int(attempt["attempt_index"]) == selected)


def _rank(attempt: dict[str, Any]) -> tuple[int, float, float, float]:
    accepted = 1 if bool(attempt.get("accepted", False)) else 0
    score = float(attempt.get("critic_score", 0.0))
    visibility = float(attempt.get("critic_defect_visibility_score", 0.0))
    target = float(attempt.get("critic_visibility_target", 0.0))
    visibility_rank = -abs(visibility - target) if target > 0.0 else visibility
    coverage = float(attempt.get("adaptive_mask_coverage_score", 0.0))
    return accepted, score, visibility_rank, coverage


def _report(
    selection_counts: Counter[str],
    category_counts: dict[str, Counter[str]],
    candidates: list[tuple[str, Path]],
    sample_count: int,
) -> str:
    lines = [
        "# R4 Generation Variant Arbitration",
        "",
        "This is a category-agnostic diagnostic selection over already generated "
        "samples. It uses no official masks, but it is not independent evidence "
        "because the same critic defines both selection and automated evaluation.",
        "",
        f"Samples: `{sample_count}`",
        "",
        "| Candidate | Selected |",
        "| --- | ---: |",
    ]
    for name, path in candidates:
        lines.append(f"| `{name}` | {selection_counts[name]} |")
        lines.append(f"<!-- {name}: {path} -->")
    lines.extend(["", "| Category | " + " | ".join(f"`{name}`" for name, _ in candidates) + " |", "| --- | " + " | ".join("---:" for _ in candidates) + " |"])
    for category, counts in sorted(category_counts.items()):
        lines.append(f"| `{category}` | " + " | ".join(str(counts[name]) for name, _ in candidates) + " |")
    lines.extend(
        [
            "",
            "Promotion requires blind human agreement and downstream utility; "
            "this arbitration report alone cannot validate the critic.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
