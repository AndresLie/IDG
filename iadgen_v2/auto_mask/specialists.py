from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SpecialistDescriptor:
    mode: str
    structure_profiles: frozenset[str]
    scales: frozenset[str] = frozenset()
    role: str = "proposal"

    def applies(self, structure_profile: str, attributes: Mapping[str, str] | None = None) -> bool:
        if structure_profile not in self.structure_profiles:
            return False
        if not self.scales:
            return True
        scale = str((attributes or {}).get("scale", "unknown"))
        return scale in self.scales


SPECIALISTS = {
    "repeated_chain_refiner": SpecialistDescriptor(
        mode="repeated_chain_refiner",
        structure_profiles=frozenset({"repeated_chain"}),
    ),
    "polar_rim_residual": SpecialistDescriptor(
        mode="polar_rim_residual",
        structure_profiles=frozenset({"ring_sector"}),
        scales=frozenset({"micro", "small"}),
    ),
    "edge_border_layout": SpecialistDescriptor(
        mode="edge_border_layout",
        structure_profiles=frozenset({"edge_border"}),
    ),
    # Compatibility alias used only by the frozen specialist baseline.
    "zipper_fabric_border_layout": SpecialistDescriptor(
        mode="zipper_fabric_border_layout",
        structure_profiles=frozenset({"edge_border"}),
    ),
}


def specialist_applicability(
    mode: str,
    *,
    structure_profile: str,
    attributes: Mapping[str, str] | None = None,
) -> bool | None:
    descriptor = SPECIALISTS.get(mode)
    if descriptor is None:
        return None
    return descriptor.applies(structure_profile, attributes)
