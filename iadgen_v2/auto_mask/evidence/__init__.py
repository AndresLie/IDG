from iadgen_v2.auto_mask.evidence.base import (
    FunctionalEvidenceProvider,
    apply_soft_spatial_prior,
    fuse_evidence_maps,
    robust_probability,
)
from iadgen_v2.auto_mask.evidence.foundation import MultiScaleDinoProvider, RegisteredDinoResidualProvider

__all__ = [
    "FunctionalEvidenceProvider",
    "MultiScaleDinoProvider",
    "RegisteredDinoResidualProvider",
    "apply_soft_spatial_prior",
    "fuse_evidence_maps",
    "robust_probability",
]
