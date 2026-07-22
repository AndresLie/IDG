from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SurfaceTarget:
    category: str
    defect_type: str


@dataclass(frozen=True)
class AppConfig:
    path: Path
    data: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent.parent if self.path.parent.name == "configs" else self.path.parent

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def dataset_root(self) -> Path:
        return self.resolve_path(self.data["dataset"]["root"])

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.data["project"]["output_dir"])

    @property
    def report_dir(self) -> Path:
        return self.resolve_path(self.data["project"]["report_dir"])

    @property
    def targets(self) -> list[SurfaceTarget]:
        configured = self.data["dataset"]["targets"]
        return [
            SurfaceTarget(category=str(category), defect_type=str(defect_type))
            for category, defect_types in configured.items()
            for defect_type in defect_types
        ]

    def split_spec(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "seed": int(self.data["dataset"].get("seed", 1337)),
            "adaptation_per_defect": int(self.data["dataset"].get("adaptation_per_defect", 5)),
            "targets": self.data["dataset"]["targets"],
        }

    def split_spec_fingerprint(self) -> str:
        return fingerprint(self.split_spec())

    def generation_fingerprint(self, model: str) -> str:
        return fingerprint(
            {
                "split_spec_fingerprint": self.split_spec_fingerprint(),
                "generation": self.data["generation"],
                "model": model,
                "model_settings": self.data["models"].get(model, {}),
            }
        )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for section in ("project", "dataset", "generation", "models", "evaluation"):
        if section not in data:
            raise ValueError(f"Missing config section: {section}")
    targets = data["dataset"].get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("dataset.targets must map MVTec categories to non-empty defect-type lists")
    for category, defect_types in targets.items():
        if not isinstance(category, str) or not isinstance(defect_types, list) or not defect_types:
            raise ValueError("dataset.targets must map MVTec categories to non-empty defect-type lists")
    config = AppConfig(config_path, data)
    # Local import avoids making the core config dataclass depend on governance.
    from iadgen_v2.governance import validate_governance_config

    validate_governance_config(config)
    return config


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
