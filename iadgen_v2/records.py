from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnomalySample:
    category: str
    defect_type: str
    image_path: str
    mask_path: str
    training_mask_path: str | None = None
    eval_mask_path: str | None = None
    uncertainty_mask_path: str | None = None
    positive_core_path: str | None = None
    possible_region_path: str | None = None
    label_policy: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationRecord:
    category: str
    defect_type: str
    background_path: str
    source_image_path: str
    source_mask_path: str
    insertion_mask_path: str
    output_path: str
    model: str
    seed: int
    latency_sec: float | None
    settings: dict[str, Any]
    error: str | None = None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_record(path: Path, record: GenerationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_records(path: Path) -> list[GenerationRecord]:
    if not path.exists():
        return []
    return [
        GenerationRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
