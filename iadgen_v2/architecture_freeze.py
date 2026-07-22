from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from iadgen_v2.config import AppConfig
from iadgen_v2.governance import architecture_core_fingerprint, governance_settings
from iadgen_v2.records import write_json


def freeze_generic_architecture(config: AppConfig) -> Path:
    settings = governance_settings(config)
    gate_value = settings.get("generic_mask_gate_path")
    if not gate_value:
        raise ValueError("Architecture freeze requires research_governance.generic_mask_gate_path")
    gate_path = config.resolve_path(str(gate_value))
    if not gate_path.exists():
        raise FileNotFoundError(f"Development gate not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("go", False)):
        raise ValueError("Architecture cannot be frozen because the development gate did not pass")
    architecture_tag = str(settings.get("release_architecture_tag", settings.get("architecture_tag", "v3-generic-evidence-rc1")))
    output_path = config.report_dir / "architecture_freeze" / f"{architecture_tag}.json"
    write_json(
        output_path,
        {
            "go": True,
            "architecture_tag": architecture_tag,
            "architecture_core_fingerprint": architecture_core_fingerprint(config),
            "development_gate_path": str(gate_path),
            "development_gate": gate,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_freeze_category_tuning_forbidden": True,
        },
    )
    return output_path
