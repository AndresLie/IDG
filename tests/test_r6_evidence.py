from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from iadgen_v2.config import AppConfig
from iadgen_v2.r6_evidence import GENERATED_FILES, build_r6_evidence_package


def test_r6_evidence_package_is_hash_verified_and_deterministic(tmp_path: Path) -> None:
    config = _r6_fixture(tmp_path)

    first = build_r6_evidence_package(config)
    first_hashes = {
        name: hashlib.sha256((first.parent / name).read_bytes()).hexdigest()
        for name in (*GENERATED_FILES, "artifact_hashes.sha256")
    }
    second = build_r6_evidence_package(config)
    second_hashes = {
        name: hashlib.sha256((second.parent / name).read_bytes()).hexdigest()
        for name in (*GENERATED_FILES, "artifact_hashes.sha256")
    }

    assert first_hashes == second_hashes
    package = json.loads(first.read_text(encoding="utf-8"))
    claims = (first.parent / "claim_ledger.md").read_text(encoding="utf-8")
    external = json.loads((first.parent / "external_baseline_status.json").read_text(encoding="utf-8"))
    assert package["status_counts"] == {
        "not_tested": 2,
        "provisional": 1,
        "rejected": 2,
        "supported": 3,
    }
    assert "C2 | supported | sealed locked-category" in claims
    assert "VisA runtime data is ready" in claims
    assert external["entries"][0]["status"] == "blocked"
    assert external["entries"][1]["status"] == "ready"


def test_r6_evidence_package_rejects_changed_source_artifact(tmp_path: Path) -> None:
    config = _r6_fixture(tmp_path)
    selector_path = Path(
        config.data["r6_evidence"]["inputs"]["selector_evidence"]["path"]
    )
    selector_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch for selector_evidence"):
        build_r6_evidence_package(config)


def _r6_fixture(tmp_path: Path) -> AppConfig:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    selector = {
        "variants": {
            "A-S1b+A-S2 (widened, deployed pool)": {
                "selection": {
                    "synthetic": {"mean_regret": 0.30, "mean_selected_dice": 0.25},
                    "real_lco": {"mean_regret": 0.15, "mean_selected_dice": 0.45},
                },
                "regret_delta_syn_minus_real": {
                    "mean": 0.15,
                    "ci95": [0.05, 0.25],
                },
                "risk_coverage_real_lco": [
                    {"coverage": 0.5, "mean_regret": 0.10},
                    {"coverage": 1.0, "mean_regret": 0.15},
                ],
            },
            "A-S1b (non-widened pool)": {
                "selection": {
                    "synthetic": {"mean_regret": 0.20, "mean_selected_dice": 0.32},
                    "real_lco": {"mean_regret": 0.10, "mean_selected_dice": 0.43},
                },
                "regret_delta_syn_minus_real": {
                    "mean": 0.10,
                    "ci95": [0.02, 0.18],
                },
                "risk_coverage_real_lco": [
                    {"coverage": 0.5, "mean_regret": 0.07},
                    {"coverage": 1.0, "mean_regret": 0.10},
                ],
            },
        }
    }
    category = {
        "images": 1,
        "selected_dice": 0.30,
        "selected_iou": 0.20,
        "precision": 0.20,
        "recall": 0.80,
        "oracle_dice": 0.60,
        "dice_regret": 0.30,
        "accepted_coverage": 1.0,
    }
    locked = {
        "aggregates": {
            "synthetic_baseline": {
                "category_macro": {
                    "selected_dice": 0.20,
                    "oracle_dice": 0.60,
                }
            },
            "real_candidate_widened": {
                "category_macro": {
                    "selected_dice": 0.30,
                    "oracle_dice": 0.60,
                },
                "per_category": {
                    "part_a": category,
                    "part_b": {**category, "selected_dice": 0.40, "dice_regret": 0.20},
                },
            },
        },
        "primary_paired_bootstrap": {
            "mean": 0.10,
            "ci95": [0.03, 0.17],
        },
        "per_sample": [
            {
                "selectors": {
                    "real_candidate_widened": {
                        "expected_iou": 0.80,
                        "dice_regret": 0.10,
                        "selected_dice": 0.50,
                    }
                }
            },
            {
                "selectors": {
                    "real_candidate_widened": {
                        "expected_iou": 0.30,
                        "dice_regret": 0.40,
                        "selected_dice": 0.10,
                    }
                }
            },
        ],
    }
    visibility = {
        "paired_intervals": {
            "critic_defect_visibility_score": {
                "mean_delta": 0.03,
                "ci95_low": 0.01,
                "ci95_high": 0.05,
            }
        }
    }
    utility = {
        "intervals": {
            "pixel_ap": {
                "mean_delta": -0.001,
                "ci95_low": -0.04,
                "ci95_high": 0.04,
            }
        }
    }
    locked_csv = "\n".join(
        [
            "sample_id,category,defect_type,dice,iou,precision,recall,predicted_positive_rate,search_region_recall,pixel_auroc,pixel_ap,aupro,disposition,confidence",
            "part_a/x/0,part_a,x,0.20,0.11,0.12,0.70,0.1,0.40,0.90,0.30,0.80,soft_mask_only,0.7",
            "part_b/x/0,part_b,x,0.40,0.25,0.25,0.80,0.1,0.80,0.95,0.50,0.90,soft_mask_only,0.8",
        ]
    ) + "\n"
    values = {
        "selector_evidence": (json.dumps(selector, sort_keys=True) + "\n").encode(),
        "locked_candidate_analysis": (json.dumps(locked, sort_keys=True) + "\n").encode(),
        "locked_metrics": locked_csv.encode(),
        "locked_seal": b'{"sealed": true}\n',
        "visibility_reaudit": (json.dumps(visibility, sort_keys=True) + "\n").encode(),
        "synthetic_utility": (json.dumps(utility, sort_keys=True) + "\n").encode(),
    }
    configured_inputs = {}
    for label, content in values.items():
        suffix = ".csv" if label == "locked_metrics" else ".json"
        path = inputs_dir / f"{label}{suffix}"
        path.write_bytes(content)
        configured_inputs[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "evidence_tier": "test",
            "role": label,
        }

    config_path = tmp_path / "configs" / "r6.yaml"
    config_path.parent.mkdir()
    config_path.write_text("# test config\n", encoding="utf-8")
    return AppConfig(
        path=config_path,
        data={
            "project": {
                "output_dir": str(tmp_path / "outputs"),
                "report_dir": str(tmp_path / "reports"),
            },
            "dataset": {
                "root": str(tmp_path / "data"),
                "targets": {"part_a": ["x"]},
            },
            "generation": {},
            "models": {},
            "evaluation": {},
            "r6_evidence": {
                "output_dir": str(tmp_path / "reports"),
                "inputs": configured_inputs,
                "external_baselines": {
                    "missing_baseline": {
                        "name": "Missing baseline",
                        "required_python_modules": ["module_that_does_not_exist_for_r6_test"],
                        "required_paths": [str(tmp_path / "missing-model")],
                    },
                    "visa": {
                        "name": "VisA",
                        "kind": "external_dataset",
                        "required_paths": [str(inputs_dir)],
                    },
                },
            },
        },
    )
