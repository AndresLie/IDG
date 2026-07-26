from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from iadgen_v2.config import AppConfig
from iadgen_v2.records import write_json


R6_SCHEMA_VERSION = 1
REQUIRED_INPUTS = {
    "selector_evidence",
    "locked_candidate_analysis",
    "locked_metrics",
    "locked_seal",
    "visibility_reaudit",
    "synthetic_utility",
}
GENERATED_FILES = (
    "r6_evidence_package.json",
    "evidence_index.json",
    "claim_ledger.csv",
    "claim_ledger.md",
    "headline_results.csv",
    "headline_results.md",
    "locked_failure_sheet.csv",
    "locked_failure_sheet.md",
    "risk_coverage.csv",
    "risk_coverage.md",
    "external_baseline_status.json",
    "external_baseline_status.md",
    "paper_package.md",
)


def build_r6_evidence_package(config: AppConfig) -> Path:
    settings = config.data.get("r6_evidence", {})
    if not isinstance(settings, dict):
        raise ValueError("r6_evidence must be a mapping")
    output_dir = config.resolve_path(str(settings.get("output_dir", config.report_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_index = _verify_inputs(config, settings)
    artifacts = {row["label"]: Path(row["path"]) for row in evidence_index["artifacts"]}
    selector = _read_json(artifacts["selector_evidence"])
    locked = _read_json(artifacts["locked_candidate_analysis"])
    visibility = _read_json(artifacts["visibility_reaudit"])
    utility = _read_json(artifacts["synthetic_utility"])

    locked_rows = _read_csv(artifacts["locked_metrics"])
    locked_summary = _summarize_locked_metrics(locked_rows, locked)
    headline = _headline_rows(selector, locked, locked_summary, visibility, utility)
    external = _external_baseline_status(config, settings)
    claims = _claim_rows(
        selector,
        locked,
        locked_summary,
        visibility,
        utility,
        external,
    )
    risk_coverage = _risk_coverage_rows(selector, locked)

    write_json(output_dir / "evidence_index.json", evidence_index)
    _write_csv(output_dir / "claim_ledger.csv", claims)
    (output_dir / "claim_ledger.md").write_text(
        _claim_ledger_markdown(claims),
        encoding="utf-8",
    )
    _write_csv(output_dir / "headline_results.csv", headline)
    (output_dir / "headline_results.md").write_text(
        _headline_markdown(headline),
        encoding="utf-8",
    )
    _write_csv(output_dir / "locked_failure_sheet.csv", locked_summary["per_category"])
    (output_dir / "locked_failure_sheet.md").write_text(
        _locked_failure_markdown(locked_summary),
        encoding="utf-8",
    )
    _write_csv(output_dir / "risk_coverage.csv", risk_coverage)
    (output_dir / "risk_coverage.md").write_text(
        _risk_coverage_markdown(risk_coverage),
        encoding="utf-8",
    )
    write_json(output_dir / "external_baseline_status.json", external)
    (output_dir / "external_baseline_status.md").write_text(
        _external_status_markdown(external),
        encoding="utf-8",
    )

    status_counts = {
        status: sum(row["status"] == status for row in claims)
        for status in ("supported", "provisional", "rejected", "not_tested")
    }
    package = {
        "schema_version": R6_SCHEMA_VERSION,
        "decision": "freeze_mask_architecture_and_report_bounded_claims",
        "status_counts": status_counts,
        "primary_supported_claim": (
            "Development-real candidate calibration improves candidate selection "
            "over synthetic-corruption-only calibration under category shift."
        ),
        "primary_failed_claim": (
            "The evaluated synthetic corpus improves independent downstream anomaly segmentation."
        ),
        "release_status": "not_release_ready",
        "external_reproduction_status": {
            row["id"]: row["status"] for row in external["entries"]
        },
        "locked_summary": locked_summary["category_macro"],
        "locked_image_weighted_summary": locked_summary["overall"],
        "source_artifact_hashes": {
            row["label"]: row["sha256"] for row in evidence_index["artifacts"]
        },
    }
    write_json(output_dir / "r6_evidence_package.json", package)
    (output_dir / "paper_package.md").write_text(
        _paper_package_markdown(package, headline, claims, locked_summary, external),
        encoding="utf-8",
    )
    _write_output_hashes(output_dir)
    return output_dir / "r6_evidence_package.json"


def _verify_inputs(config: AppConfig, settings: dict[str, Any]) -> dict[str, Any]:
    configured = settings.get("inputs", {})
    if not isinstance(configured, dict):
        raise ValueError("r6_evidence.inputs must be a mapping")
    missing = REQUIRED_INPUTS - set(configured)
    if missing:
        raise ValueError(f"r6_evidence.inputs is missing required artifacts: {sorted(missing)}")

    rows = []
    for label in sorted(REQUIRED_INPUTS):
        item = configured[label]
        if not isinstance(item, dict):
            raise ValueError(f"r6_evidence.inputs.{label} must be a mapping")
        path = config.resolve_path(str(item.get("path", "")))
        expected = str(item.get("sha256", "")).lower()
        if len(expected) != 64:
            raise ValueError(f"r6_evidence.inputs.{label}.sha256 must be a full SHA-256")
        if not path.is_file():
            raise FileNotFoundError(f"R6 input artifact does not exist: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"R6 input hash mismatch for {label}: expected {expected}, got {actual}"
            )
        rows.append(
            {
                "label": label,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": actual,
                "evidence_tier": str(item.get("evidence_tier", "unspecified")),
                "role": str(item.get("role", "")),
                "verified": True,
            }
        )
    return {
        "schema_version": R6_SCHEMA_VERSION,
        "integrity": "all_inputs_hash_verified",
        "artifacts": rows,
    }


def _headline_rows(
    selector: dict[str, Any],
    locked: dict[str, Any],
    locked_summary: dict[str, Any],
    visibility: dict[str, Any],
    utility: dict[str, Any],
) -> list[dict[str, Any]]:
    widened = _selector_variant(selector, "widened")
    nonwidened = _selector_variant(selector, "non-widened")
    primary = locked["primary_paired_bootstrap"]
    locked_real = locked["aggregates"]["real_candidate_widened"]["category_macro"]
    locked_syn = locked["aggregates"]["synthetic_baseline"]["category_macro"]
    visibility_interval = visibility["paired_intervals"]["critic_defect_visibility_score"]
    utility_interval = utility["intervals"]["pixel_ap"]
    frozen = locked_summary["category_macro"]

    return [
        _headline(
            "R1",
            "development selector calibration, widened pool",
            "Dice regret reduction",
            widened["selection"]["synthetic"]["mean_regret"],
            widened["selection"]["real_lco"]["mean_regret"],
            widened["regret_delta_syn_minus_real"]["mean"],
            widened["regret_delta_syn_minus_real"]["ci95"],
            "development category-held-out",
            "supported",
        ),
        _headline(
            "R1",
            "development selector calibration, non-widened pool",
            "Dice regret reduction",
            nonwidened["selection"]["synthetic"]["mean_regret"],
            nonwidened["selection"]["real_lco"]["mean_regret"],
            nonwidened["regret_delta_syn_minus_real"]["mean"],
            nonwidened["regret_delta_syn_minus_real"]["ci95"],
            "development category-held-out",
            "supported",
        ),
        _headline(
            "R3",
            "locked selector calibration",
            "category-macro Dice",
            locked_syn["selected_dice"],
            locked_real["selected_dice"],
            primary["mean"],
            primary["ci95"],
            "sealed locked-category",
            "supported",
        ),
        _headline(
            "R3",
            "frozen generic auto-mask runtime",
            "category-macro Dice",
            None,
            frozen["dice"],
            None,
            None,
            "sealed locked-category",
            "release gate failed",
        ),
        _headline(
            "R3",
            "candidate ceiling versus deployed selector",
            "category-macro Dice",
            locked_real["selected_dice"],
            locked_real["oracle_dice"],
            locked_real["oracle_dice"] - locked_real["selected_dice"],
            None,
            "sealed locked-category diagnostic",
            "selection gap remains",
        ),
        _headline(
            "R4",
            "critic arbitration versus baseline generation",
            "critic visibility",
            None,
            None,
            visibility_interval["mean_delta"],
            [visibility_interval["ci95_low"], visibility_interval["ci95_high"]],
            "development automated critic",
            "provisional; human review pending",
        ),
        _headline(
            "R5",
            "normal plus synthetic versus normal only",
            "pixel AP",
            None,
            None,
            utility_interval["mean_delta"],
            [utility_interval["ci95_low"], utility_interval["ci95_high"]],
            "development deterministic downstream",
            "promotion gate failed",
        ),
    ]


def _headline(
    sprint: str,
    comparison: str,
    metric: str,
    baseline: float | None,
    candidate: float | None,
    delta: float | None,
    interval: list[float] | None,
    evidence_tier: str,
    decision: str,
) -> dict[str, Any]:
    return {
        "sprint": sprint,
        "comparison": comparison,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "ci95_low": interval[0] if interval else None,
        "ci95_high": interval[1] if interval else None,
        "evidence_tier": evidence_tier,
        "decision": decision,
    }


def _claim_rows(
    selector: dict[str, Any],
    locked: dict[str, Any],
    locked_summary: dict[str, Any],
    visibility: dict[str, Any],
    utility: dict[str, Any],
    external: dict[str, Any],
) -> list[dict[str, Any]]:
    widened = _selector_variant(selector, "widened")
    primary = locked["primary_paired_bootstrap"]
    locked_real = locked["aggregates"]["real_candidate_widened"]["category_macro"]
    visibility_interval = visibility["paired_intervals"]["critic_defect_visibility_score"]
    utility_interval = utility["intervals"]["pixel_ap"]
    return [
        {
            "claim_id": "C1",
            "claim": "Real-candidate calibration improves selector regret on development categories.",
            "status": "supported",
            "evidence_tier": "development category-held-out",
            "estimate": widened["regret_delta_syn_minus_real"]["mean"],
            "ci95": _format_ci(widened["regret_delta_syn_minus_real"]["ci95"]),
            "limitation": "Official development masks calibrate the selector after proposal generation.",
        },
        {
            "claim_id": "C2",
            "claim": "The selector-calibration gain transfers to ten locked MVTec categories.",
            "status": "supported",
            "evidence_tier": "sealed locked-category",
            "estimate": primary["mean"],
            "ci95": _format_ci(primary["ci95"]),
            "limitation": "This confirms a selector contribution, not release-level mask quality.",
        },
        {
            "claim_id": "C3",
            "claim": "Candidate generation contains substantially better masks than the deployed selector publishes.",
            "status": "supported",
            "evidence_tier": "sealed locked-category diagnostic",
            "estimate": locked_real["oracle_dice"] - locked_real["selected_dice"],
            "ci95": "",
            "limitation": "Oracle candidate choice is diagnostic and unavailable at runtime.",
        },
        {
            "claim_id": "C4",
            "claim": "The generic auto-mask architecture is ready for release.",
            "status": "rejected",
            "evidence_tier": "sealed locked-category",
            "estimate": locked_summary["category_macro"]["dice"],
            "ci95": "",
            "limitation": "Macro Dice and worst-category quality miss the preregistered engineering targets.",
        },
        {
            "claim_id": "C5",
            "claim": "Critic arbitration improves perceptual defect generation.",
            "status": "provisional",
            "evidence_tier": "development automated critic",
            "estimate": visibility_interval["mean_delta"],
            "ci95": _format_ci(
                [visibility_interval["ci95_low"], visibility_interval["ci95_high"]]
            ),
            "limitation": "The automated gate passed, but two-reviewer blind validation is incomplete.",
        },
        {
            "claim_id": "C6",
            "claim": "The evaluated synthetic corpus improves independent downstream segmentation.",
            "status": "rejected",
            "evidence_tier": "development deterministic downstream",
            "estimate": utility_interval["mean_delta"],
            "ci95": _format_ci([utility_interval["ci95_low"], utility_interval["ci95_high"]]),
            "limitation": "The primary pixel-AP interval includes zero and the promotion gate failed.",
        },
        {
            "claim_id": "C7",
            "claim": "The method generalizes to VisA or MVTec AD 2.",
            "status": "not_tested",
            "evidence_tier": "external",
            "estimate": "",
            "ci95": "",
            "limitation": _external_generalization_limitation(external),
        },
        {
            "claim_id": "C8",
            "claim": "The method is state of the art against official recent baselines.",
            "status": "not_tested",
            "evidence_tier": "external baseline",
            "estimate": "",
            "ci95": "",
            "limitation": "Official SubspaceAD reproduction is blocked; literature numbers are not local measurements.",
        },
    ]


def _external_generalization_limitation(external: dict[str, Any]) -> str:
    statuses = {
        str(row["id"]): str(row["status"])
        for row in external.get("entries", [])
        if row.get("kind") == "external_dataset"
    }
    visa_ready = statuses.get("visa") == "ready"
    mvtec_ad_2_ready = statuses.get("mvtec_ad_2") == "ready"
    if visa_ready and mvtec_ad_2_ready:
        return (
            "External dataset prerequisites are ready, but frozen quantitative "
            "inference and locked evaluation have not run."
        )
    if visa_ready:
        return (
            "VisA runtime data is ready, but frozen quantitative inference and "
            "locked evaluation have not run; MVTec AD 2 remains blocked."
        )
    return "Neither external dataset is present in the local reproducible environment."


def _summarize_locked_metrics(
    rows: list[dict[str, str]],
    locked: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("locked_metrics is empty")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    candidate_categories = locked["aggregates"]["real_candidate_widened"]["per_category"]
    result_rows = []
    numeric = (
        "dice",
        "iou",
        "precision",
        "recall",
        "search_region_recall",
        "pixel_auroc",
        "pixel_ap",
        "aupro",
    )
    for category in sorted(grouped):
        category_rows = grouped[category]
        summary = {
            key: fmean(float(row[key]) for row in category_rows)
            for key in numeric
        }
        candidate = candidate_categories[category]
        diagnoses = []
        if summary["precision"] < 0.20 and summary["recall"] > 0.70:
            diagnoses.append("oversegmentation")
        if summary["search_region_recall"] < 0.60:
            diagnoses.append("localization")
        if float(candidate["dice_regret"]) > 0.30:
            diagnoses.append("selection_regret")
        if summary["dice"] < 0.30 and not diagnoses:
            diagnoses.append("low_mask_quality")
        result_rows.append(
            {
                "category": category,
                "images": len(category_rows),
                **summary,
                "oracle_dice": float(candidate["oracle_dice"]),
                "dice_regret": float(candidate["dice_regret"]),
                "accepted_coverage": fmean(
                    str(row["disposition"]) != "needs_review" for row in category_rows
                ),
                "candidate_selector_accepted_coverage": float(candidate["accepted_coverage"]),
                "zero_dice_images": sum(float(row["dice"]) == 0.0 for row in category_rows),
                "diagnosis": ";".join(diagnoses) or "no_dominant_failure",
            }
        )
    category_macro = {
        key: fmean(float(row[key]) for row in result_rows)
        for key in numeric
    }
    category_macro.update(
        {
            "oracle_dice": fmean(float(row["oracle_dice"]) for row in result_rows),
            "dice_regret": fmean(float(row["dice_regret"]) for row in result_rows),
            "accepted_coverage": fmean(float(row["accepted_coverage"]) for row in result_rows),
            "candidate_selector_accepted_coverage": fmean(
                float(row["candidate_selector_accepted_coverage"]) for row in result_rows
            ),
            "images": len(rows),
            "categories": len(result_rows),
            "zero_dice_images": sum(int(row["zero_dice_images"]) for row in result_rows),
        }
    )
    overall = {
        key: fmean(float(row[key]) for row in rows)
        for key in numeric
    }
    overall.update(
        {
            "accepted_coverage": fmean(
                str(row["disposition"]) != "needs_review" for row in rows
            ),
            "images": len(rows),
            "categories": len(result_rows),
            "zero_dice_images": sum(float(row["dice"]) == 0.0 for row in rows),
        }
    )
    return {"category_macro": category_macro, "overall": overall, "per_category": result_rows}


def _risk_coverage_rows(
    selector: dict[str, Any],
    locked: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for name, variant in selector["variants"].items():
        pool = "widened" if "widened" in name.lower() and "non-widened" not in name.lower() else "non-widened"
        for point in variant.get("risk_coverage_real_lco", []):
            rows.append(
                {
                    "cohort": "development",
                    "selector": "real_candidate_lco",
                    "pool": pool,
                    "coverage": float(point["coverage"]),
                    "mean_regret": float(point["mean_regret"]),
                    "mean_selected_dice": "",
                    "note": "category-held-out calibration",
                }
            )

    samples = []
    for sample in locked["per_sample"]:
        result = sample["selectors"]["real_candidate_widened"]
        samples.append(
            {
                "confidence": float(result["expected_iou"]),
                "regret": float(result["dice_regret"]),
                "dice": float(result["selected_dice"]),
            }
        )
    samples.sort(key=lambda row: (-row["confidence"], row["regret"], -row["dice"]))
    for coverage in (0.25, 0.50, 0.75, 1.00):
        count = max(1, round(len(samples) * coverage))
        retained = samples[:count]
        rows.append(
            {
                "cohort": "locked",
                "selector": "real_candidate_widened",
                "pool": "widened",
                "coverage": coverage,
                "mean_regret": fmean(row["regret"] for row in retained),
                "mean_selected_dice": fmean(row["dice"] for row in retained),
                "note": "post-lock image-weighted diagnostic; no threshold tuning",
            }
        )
    return sorted(
        rows,
        key=lambda row: (str(row["cohort"]), str(row["pool"]), float(row["coverage"])),
    )


def _external_baseline_status(
    config: AppConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    configured = settings.get("external_baselines", {})
    if not isinstance(configured, dict):
        raise ValueError("r6_evidence.external_baselines must be a mapping")
    entries = []
    for baseline_id in sorted(configured):
        item = configured[baseline_id]
        if not isinstance(item, dict):
            raise ValueError(f"r6_evidence.external_baselines.{baseline_id} must be a mapping")
        module_checks = []
        for module in sorted(str(value) for value in item.get("required_python_modules", [])):
            module_checks.append({"module": module, "available": importlib.util.find_spec(module) is not None})
        path_checks = []
        for value in sorted(str(value) for value in item.get("required_paths", [])):
            path = config.resolve_path(value)
            path_checks.append({"path": str(path), "available": path.exists()})
        blockers = [
            f"missing Python module: {row['module']}" for row in module_checks if not row["available"]
        ]
        blockers.extend(f"missing local path: {row['path']}" for row in path_checks if not row["available"])
        entries.append(
            {
                "id": baseline_id,
                "name": str(item.get("name", baseline_id)),
                "kind": str(item.get("kind", "baseline")),
                "status": "ready" if not blockers else "blocked",
                "official_url": str(item.get("official_url", "")),
                "official_repository": str(item.get("official_repository", "")),
                "official_commit": str(item.get("official_commit", "")),
                "protocol": str(item.get("protocol", "")),
                "module_checks": module_checks,
                "path_checks": path_checks,
                "blockers": blockers,
                "measurement_policy": "literature results are not reported as local measurements",
            }
        )
    return {
        "schema_version": R6_SCHEMA_VERSION,
        "decision": "external comparisons remain pending until every local prerequisite is present",
        "entries": entries,
    }


def _selector_variant(selector: dict[str, Any], pool: str) -> dict[str, Any]:
    for name, value in selector["variants"].items():
        lowered = name.lower()
        if pool == "widened" and "widened" in lowered and "non-widened" not in lowered:
            return value
        if pool == "non-widened" and "non-widened" in lowered:
            return value
    raise ValueError(f"Selector evidence does not contain the {pool} pool")


def _claim_ledger_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# R6 Claim Ledger",
        "",
        "Claims are bounded by the strongest completed evidence tier. Literature values are not local results.",
        "",
        "| ID | Status | Evidence tier | Claim | Estimate | 95% CI | Limitation |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['claim_id']} | {row['status']} | {row['evidence_tier']} | "
            f"{row['claim']} | {_fmt(row['estimate'])} | {row['ci95']} | {row['limitation']} |"
        )
    return "\n".join(lines) + "\n"


def _headline_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# R6 Headline Results",
        "",
        "| Sprint | Comparison | Metric | Baseline | Candidate | Delta | 95% CI | Decision |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        interval = _format_ci([row["ci95_low"], row["ci95_high"]]) if row["ci95_low"] is not None else ""
        lines.append(
            f"| {row['sprint']} | {row['comparison']} | {row['metric']} | "
            f"{_fmt(row['baseline'])} | {_fmt(row['candidate'])} | {_fmt(row['delta'])} | "
            f"{interval} | {row['decision']} |"
        )
    return "\n".join(lines) + "\n"


def _locked_failure_markdown(summary: dict[str, Any]) -> str:
    macro = summary["category_macro"]
    lines = [
        "# Locked-Category Failure Sheet",
        "",
        "This is a sealed post-lock diagnostic. It must not be used to tune the exposed categories.",
        "",
        f"- Category-macro Dice: `{macro['dice']:.4f}`",
        f"- Category-macro precision / recall: `{macro['precision']:.4f}` / `{macro['recall']:.4f}`",
        f"- Category-macro search recall: `{macro['search_region_recall']:.4f}`",
        f"- Oracle Dice / selection regret: `{macro['oracle_dice']:.4f}` / `{macro['dice_regret']:.4f}`",
        f"- Accepted coverage (image-weighted / category-macro): "
        f"`{summary['overall']['accepted_coverage']:.4f}` / `{macro['accepted_coverage']:.4f}`",
        f"- Zero-Dice images: `{macro['zero_dice_images']} / {macro['images']}`",
        "",
        "| Category | N | Dice | Precision | Recall | Search recall | Oracle Dice | Regret | Zero Dice | Diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["per_category"]:
        lines.append(
            f"| {row['category']} | {row['images']} | {row['dice']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['search_region_recall']:.4f} | "
            f"{row['oracle_dice']:.4f} | {row['dice_regret']:.4f} | "
            f"{row['zero_dice_images']} | {row['diagnosis']} |"
        )
    return "\n".join(lines) + "\n"


def _risk_coverage_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Selector Risk-Coverage Summary",
        "",
        "| Cohort | Pool | Coverage | Mean regret | Mean selected Dice | Note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['cohort']} | {row['pool']} | {float(row['coverage']):.2f} | "
            f"{float(row['mean_regret']):.4f} | {_fmt(row['mean_selected_dice'])} | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def _external_status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# External Baseline Readiness",
        "",
        "A blocked row is not replaced by a paper-reported score or a locally inspired approximation.",
        "",
        "| Entry | Kind | Status | Blockers | Official source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in status["entries"]:
        source = row["official_repository"] or row["official_url"]
        blockers = "<br>".join(row["blockers"]) or "none"
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['status']} | {blockers} | {source} |"
        )
    return "\n".join(lines) + "\n"


def _paper_package_markdown(
    package: dict[str, Any],
    headline: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    locked_summary: dict[str, Any],
    external: dict[str, Any],
) -> str:
    macro = locked_summary["category_macro"]
    supported = [row for row in claims if row["status"] == "supported"]
    rejected = [row for row in claims if row["status"] == "rejected"]
    lines = [
        "# R6 Paper Evidence Package",
        "",
        "## Research Position",
        "",
        "The defensible contribution is calibrated candidate selection for automatic industrial-defect pseudo-labels. "
        "It is not a release-ready universal segmenter, and the evaluated synthetic corpus did not improve the independent downstream endpoint.",
        "",
        "## Strongest Result",
        "",
        "Development-real candidate calibration beat synthetic-corruption-only calibration on the sealed locked pool: "
        f"`+{headline[2]['delta']:.4f}` category-macro Dice, "
        f"95% CI `{_format_ci([headline[2]['ci95_low'], headline[2]['ci95_high']])}`.",
        "",
        "## Locked Performance",
        "",
        f"- Category-macro Dice: `{macro['dice']:.4f}`",
        f"- Category-macro precision / recall: `{macro['precision']:.4f}` / `{macro['recall']:.4f}`",
        f"- Category-macro search recall: `{macro['search_region_recall']:.4f}`",
        f"- Oracle Dice: `{macro['oracle_dice']:.4f}`",
        f"- Selection regret: `{macro['dice_regret']:.4f}`",
        "",
        "The ranking evidence is stronger than the published binary masks. High recall with low precision, plus the oracle gap, identifies over-segmentation and candidate ranking as the dominant weaknesses.",
        "",
        "## Supported Claims",
        "",
    ]
    lines.extend(f"- **{row['claim_id']}**: {row['claim']}" for row in supported)
    lines.extend(["", "## Rejected Claims", ""])
    lines.extend(f"- **{row['claim_id']}**: {row['claim']} {row['limitation']}" for row in rejected)
    lines.extend(["", "## External Evidence Status", ""])
    for row in external["entries"]:
        if row["status"] == "ready":
            lines.append(
                f"- `{row['name']}`: local prerequisites ready; quantitative result pending."
            )
        else:
            lines.append(f"- `{row['name']}`: blocked ({'; '.join(row['blockers'])}).")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{package['decision']}`",
            "",
            "Freeze R1-R5 results. Acquire the external data/model dependencies, run official baselines without altering the frozen method, and report failures rather than reopening development tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty R6 table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_output_hashes(output_dir: Path) -> None:
    lines = []
    for name in GENERATED_FILES:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected R6 output was not generated: {path}")
        lines.append(f"{_sha256_file(path)}  {name}")
    (output_dir / "artifact_hashes.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_ci(values: Iterable[float]) -> str:
    low, high = list(values)
    return f"[{low:+.4f}, {high:+.4f}]"


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
