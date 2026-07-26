from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ATTEMPT_METRICS = (
    "critic_defect_visibility_score",
    "adaptive_mask_coverage_score",
    "texture_preservation_score",
    "leakage_score",
    "outside_change_fraction",
    "critic_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review the matched R4 visibility-controller re-audit.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="controller")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    baseline = _read_rows(args.baseline)
    controller = _read_rows(args.controller)
    pairs = _match_rows(baseline, controller)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _pair_metric_rows(pairs)
    _write_csv(args.output_dir / "paired_metrics.csv", metric_rows)
    intervals = {
        metric: _stratified_bootstrap_interval(
            metric_rows,
            metric,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, metric in enumerate((*ATTEMPT_METRICS, "latency_sec"))
    }
    summary = _summarize(pairs, metric_rows, intervals, candidate_name=args.candidate_name)
    (args.output_dir / "r4_visibility_reaudit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "r4_visibility_reaudit.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    _write_comparison_sheet(
        pairs,
        args.output_dir / "r4_visibility_comparison.png",
        candidate_name=args.candidate_name,
    )
    _write_blind_sheet(
        pairs,
        args.output_dir / "r4_visibility_blind_audit.png",
        args.output_dir / "r4_visibility_blind_key.json",
        seed=args.seed,
        candidate_name=args.candidate_name,
    )
    print(args.output_dir / "r4_visibility_reaudit.md")


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


def _match_rows(
    baseline: list[dict[str, Any]],
    controller: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left = {_key(row): row for row in baseline}
    right = {_key(row): row for row in controller}
    if left.keys() != right.keys():
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(f"R4 arms do not match: missing baseline={missing_left}, missing controller={missing_right}")
    return [(left[key], right[key]) for key in sorted(left)]


def _selected_attempt(row: dict[str, Any]) -> dict[str, Any]:
    critic = row["critic_guided_generation"]
    selected = int(critic["selected_attempt_index"])
    return next(attempt for attempt in critic["attempts"] if int(attempt["attempt_index"]) == selected)


def _pair_metric_rows(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, controller in pairs:
        before = _selected_attempt(baseline)
        after = _selected_attempt(controller)
        row: dict[str, Any] = {
            "category": baseline["category"],
            "defect_type": baseline["defect_type"],
            "morphology": baseline["settings"].get("quality_morphology", "unknown"),
            "background_path": baseline["background_path"],
            "generation_seed": baseline["generation_seed"],
            "baseline_accepted": bool(before["accepted"]),
            "controller_accepted": bool(after["accepted"]),
            "baseline_attempt_count": len(baseline["critic_guided_generation"]["attempts"]),
            "controller_attempt_count": _generated_attempt_count(controller),
            "baseline_selected_attempt": before["attempt_index"],
            "controller_selected_attempt": after["attempt_index"],
            "baseline_latency_sec": float(baseline["latency_sec"]),
            "controller_latency_sec": float(controller["latency_sec"]),
            "latency_sec_delta": float(controller["latency_sec"]) - float(baseline["latency_sec"]),
        }
        for metric in ATTEMPT_METRICS:
            row[f"baseline_{metric}"] = float(before[metric])
            row[f"controller_{metric}"] = float(after[metric])
            row[f"{metric}_delta"] = float(after[metric]) - float(before[metric])
        rows.append(row)
    return rows


def _stratified_bootstrap_interval(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["morphology"])].append(float(row[f"{metric}_delta"]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    strata = [np.asarray(values, dtype=np.float64) for values in grouped.values()]
    for index in range(samples):
        sampled = [rng.choice(values, size=len(values), replace=True) for values in strata]
        draws[index] = float(np.concatenate(sampled).mean())
    values = [float(row[f"{metric}_delta"]) for row in rows]
    return {
        "mean_delta": float(mean(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _summarize(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric_rows: list[dict[str, Any]],
    intervals: dict[str, dict[str, float]],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in metric_rows}):
        subset = [row for row in metric_rows if row["category"] == category]
        by_category[category] = {
            "samples": len(subset),
            "visibility_delta": mean(float(row["critic_defect_visibility_score_delta"]) for row in subset),
            "coverage_delta": mean(float(row["adaptive_mask_coverage_score_delta"]) for row in subset),
            "texture_delta": mean(float(row["texture_preservation_score_delta"]) for row in subset),
            "leakage_delta": mean(float(row["leakage_score_delta"]) for row in subset),
            "baseline_acceptance": mean(float(row["baseline_accepted"]) for row in subset),
            "controller_acceptance": mean(float(row["controller_accepted"]) for row in subset),
        }
    baseline_attempts = [_selected_attempt(pair[0]) for pair in pairs]
    controller_attempts = [_selected_attempt(pair[1]) for pair in pairs]
    visibility = intervals["critic_defect_visibility_score"]
    texture = intervals["texture_preservation_score"]
    leakage = intervals["leakage_score"]
    automated_keep = (
        visibility["mean_delta"] >= 0.020
        and visibility["ci95_low"] > 0.0
        and texture["mean_delta"] >= -0.010
        and leakage["mean_delta"] >= -0.010
    )
    return {
        "sample_count": len(pairs),
        "candidate_name": candidate_name,
        "matched_inputs": True,
        "categories": Counter(str(pair[0]["category"]) for pair in pairs),
        "baseline": {
            "accepted": sum(bool(attempt["accepted"]) for attempt in baseline_attempts),
            "attempts_generated": sum(len(pair[0]["critic_guided_generation"]["attempts"]) for pair in pairs),
            "reject_reasons": Counter(reason for attempt in baseline_attempts for reason in attempt["reject_reasons"]),
        },
        "controller": {
            "accepted": sum(bool(attempt["accepted"]) for attempt in controller_attempts),
            "attempts_generated": sum(_generated_attempt_count(pair[1]) for pair in pairs),
            "reject_reasons": Counter(reason for attempt in controller_attempts for reason in attempt["reject_reasons"]),
        },
        "paired_intervals": intervals,
        "by_category": by_category,
        "automated_keep_gate": {
            "visibility_mean_delta_at_least_0.020": visibility["mean_delta"] >= 0.020,
            "visibility_ci_excludes_zero": visibility["ci95_low"] > 0.0,
            "texture_regression_at_most_0.010": texture["mean_delta"] >= -0.010,
            "leakage_regression_at_most_0.010": leakage["mean_delta"] >= -0.010,
            "passed": automated_keep,
            "human_review_status": "pending_two_blind_reviewers",
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    intervals = summary["paired_intervals"]
    gate = summary["automated_keep_gate"]
    candidate_name = str(summary.get("candidate_name", "controller"))
    candidate_title = candidate_name.replace("_", " ").replace("-", " ").title()
    if gate["passed"]:
        decision = (
            f"**{candidate_name} passes the automated gate, but is not yet promotable.** "
            "The same critic participates in selection and evaluation, and the blind human audit is pending."
        )
    else:
        decision = (
            f"**Do not promote {candidate_name} from this pilot.** "
            "The automated joint keep gate failed; the blind human audit remains pending."
        )
    lines = [
        f"# R4 {candidate_title} Re-Audit",
        "",
        "## Decision",
        "",
        decision,
        "",
        "The two arms use identical backgrounds, masks, prompts, quality profiles, and base seeds. "
        f"The candidate arm is `{candidate_name}`.",
        "",
        "## Paired Results",
        "",
        "| Metric | Mean delta | Stratified 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for metric in (*ATTEMPT_METRICS, "latency_sec"):
        result = intervals[metric]
        lines.append(
            f"| `{metric}` | `{result['mean_delta']:+.4f}` | "
            f"`[{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}]` |"
        )
    if candidate_name == "mask-local":
        interpretation = (
            "Mask-local rendering removes outside-mask change, but the soft-envelope restore attenuates "
            "an already weak inpainted defect. Visibility, coverage, and acceptance all regress, including "
            "on wood. Keep this path default-off; do not tune crop size against this pilot."
        )
    elif candidate_name == "fixed-adapter":
        interpretation = (
            "The fixed adapter produces a statistically detectable but practically negligible change. "
            "It improves acceptance only on metal nut and leaves wood at zero accepted samples. The "
            "adapter gate and spatial-conditioning path require redesign before this variant can support "
            "a visible-generation claim."
        )
    elif candidate_name == "clone-harmonized":
        interpretation = (
            "Clone harmonization is not a universal replacement: it strongly regresses metal nut and tile. "
            "However, it materially improves the previously failed wood cases. This interaction motivates "
            "per-sample quality arbitration rather than category rules or a single global generator."
        )
    elif candidate_name == "critic-arbitrated":
        interpretation = (
            "Category-agnostic critic arbitration is the first R4 candidate to pass all automated gates. "
            "It preserves text-only generation where clone transfer is weak and selects clone transfer for "
            "the under-edited wood cases. This is promising routing evidence, not independent validation: "
            "the same critic selected and scored the outputs, so blind review and downstream utility remain "
            "mandatory before promotion."
        )
    else:
        interpretation = (
            "The controller increases compute and recovers a small number of rejected samples, but the "
            "gain is not broad enough to justify deployment. Wood remains the dominant under-editing "
            "failure. Additional global strength escalation is therefore the wrong next move."
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- Baseline: `{summary['baseline']['accepted']}/{summary['sample_count']}` accepted from "
            f"`{summary['baseline']['attempts_generated']}` attempts.",
            f"- {candidate_title}: `{summary['controller']['accepted']}/{summary['sample_count']}` accepted from "
            f"`{summary['controller']['attempts_generated']}` attempts.",
            f"- Baseline reject reasons: `{dict(summary['baseline']['reject_reasons'])}`.",
            f"- {candidate_title} reject reasons: `{dict(summary['controller']['reject_reasons'])}`.",
            "",
            "## Category Diagnosis",
            "",
            "| Category | N | Visibility Δ | Coverage Δ | Texture Δ | Leakage Δ | Acceptance |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, row in summary["by_category"].items():
        lines.append(
            f"| `{category}` | {row['samples']} | {row['visibility_delta']:+.4f} | "
            f"{row['coverage_delta']:+.4f} | {row['texture_delta']:+.4f} | "
            f"{row['leakage_delta']:+.4f} | "
            f"{row['baseline_acceptance']:.0%} → {row['controller_acceptance']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Keep Gate",
            "",
            f"- Visibility gain `>= 0.020`: **{_pass(gate['visibility_mean_delta_at_least_0.020'])}**",
            f"- Visibility interval excludes zero: **{_pass(gate['visibility_ci_excludes_zero'])}**",
            f"- Texture regression `<= 0.010`: **{_pass(gate['texture_regression_at_most_0.010'])}**",
            f"- Leakage regression `<= 0.010`: **{_pass(gate['leakage_regression_at_most_0.010'])}**",
            f"- Joint automated gate: **{_pass(gate['passed'])}**",
            "- Human gate: **PENDING**. Use `r4_visibility_blind_audit.png` with two independent reviewers; "
            "do not open the key until both judgments are recorded.",
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
        ]
    )
    return "\n".join(lines)


def _pass(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _generated_attempt_count(row: dict[str, Any]) -> int:
    arbitration = row.get("settings", {}).get("generation_variant_arbitration", {})
    candidates = arbitration.get("candidates", {}) if isinstance(arbitration, dict) else {}
    if isinstance(candidates, dict) and candidates:
        return sum(int(candidate.get("attempt_count", 1)) for candidate in candidates.values())
    return len(row["critic_guided_generation"]["attempts"])


def _write_comparison_sheet(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    path: Path,
    *,
    candidate_name: str,
) -> None:
    candidate_title = candidate_name.replace("_", " ").replace("-", " ").title()
    headers = ("Background", "Mask", "Baseline", candidate_title, "Baseline change", f"{candidate_title} change")
    tile = 176
    label_height = 36
    sheet = Image.new("RGB", (tile * len(headers), label_height + tile * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, header in enumerate(headers):
        draw.text((column * tile + 6, 10), header, fill="black", font=font)
    for row_index, (baseline, controller) in enumerate(pairs):
        background = Image.open(baseline["background_path"]).convert("RGB")
        mask = Image.open(baseline["inpaint_mask_path"]).convert("L")
        before = Image.open(baseline["output_path"]).convert("RGB")
        after = Image.open(controller["output_path"]).convert("RGB")
        crop = _mask_crop(mask)
        images = (
            background.crop(crop),
            _mask_visual(background, mask).crop(crop),
            before.crop(crop),
            after.crop(crop),
            _change_visual(background, before).crop(crop),
            _change_visual(background, after).crop(crop),
        )
        y = label_height + row_index * tile
        for column, image in enumerate(images):
            sheet.paste(_fit(image, tile), (column * tile, y))
        selected = _selected_attempt(controller)
        draw.rectangle((0, y, tile * len(headers) - 1, y + tile - 1), outline=(210, 210, 210))
        draw.text(
            (4, y + 4),
            f"{baseline['category']} s{baseline['settings']['seed_index']} "
            f"V {selected['critic_defect_visibility_score']:.2f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=font,
        )
    sheet.save(path)


def _write_blind_sheet(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    image_path: Path,
    key_path: Path,
    *,
    seed: int,
    candidate_name: str,
) -> None:
    rng = np.random.default_rng(seed)
    tile = 220
    label_height = 38
    sheet = Image.new("RGB", (tile * 3, label_height + tile * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, header in enumerate(("Background", "A", "B")):
        draw.text((column * tile + 8, 12), header, fill="black", font=font)
    key: list[dict[str, Any]] = []
    for index, (baseline, controller) in enumerate(pairs):
        background = Image.open(baseline["background_path"]).convert("RGB")
        mask = Image.open(baseline["inpaint_mask_path"]).convert("L")
        crop = _mask_crop(mask)
        arms = [
            ("baseline", Image.open(baseline["output_path"]).convert("RGB")),
            (candidate_name, Image.open(controller["output_path"]).convert("RGB")),
        ]
        if bool(rng.integers(0, 2)):
            arms.reverse()
        y = label_height + index * tile
        for column, image in enumerate((background, arms[0][1], arms[1][1])):
            sheet.paste(_fit(image.crop(crop), tile), (column * tile, y))
        draw.text(
            (4, y + 4),
            f"case {index + 1:02d}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=font,
        )
        key.append(
            {
                "case": index + 1,
                "category": baseline["category"],
                "generation_seed": baseline["generation_seed"],
                "A": arms[0][0],
                "B": arms[1][0],
            }
        )
    sheet.save(image_path)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")


def _mask_crop(mask: Image.Image) -> tuple[int, int, int, int]:
    active = np.asarray(mask.convert("L"), dtype=np.uint8) > 8
    ys, xs = np.where(active)
    width, height = mask.size
    if not len(xs):
        return 0, 0, width, height
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = max(x1 - x0, y1 - y0, min(width, height) // 4)
    span = min(max(span * 2, 96), min(width, height))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    left = min(max(0, cx - span // 2), width - span)
    top = min(max(0, cy - span // 2), height - span)
    return left, top, left + span, top + span


def _fit(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _mask_visual(background: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(background.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L").resize(background.size), dtype=np.float32) / 255.0
    overlay = rgb.copy()
    overlay[..., 0] = 255.0
    out = rgb * (1.0 - 0.55 * alpha[..., None]) + overlay * (0.55 * alpha[..., None])
    return Image.fromarray(np.uint8(np.clip(out, 0.0, 255.0)), "RGB")


def _change_visual(background: Image.Image, generated: Image.Image) -> Image.Image:
    bg = np.asarray(background.convert("RGB"), dtype=np.float32) / 255.0
    gen = np.asarray(generated.convert("RGB").resize(background.size), dtype=np.float32) / 255.0
    diff = np.abs(gen - bg).mean(axis=2)
    scale = np.clip(diff / 0.20, 0.0, 1.0)
    heat = np.zeros((*scale.shape, 3), dtype=np.float32)
    heat[..., 0] = scale * 255.0
    heat[..., 1] = np.clip((scale - 0.25) * 255.0, 0.0, 255.0)
    heat[..., 2] = np.clip((1.0 - scale) * 80.0, 0.0, 80.0)
    return Image.fromarray(np.uint8(heat), "RGB")


if __name__ == "__main__":
    main()
