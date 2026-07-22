"""Modular contracts for the generalization-first automatic mask pipeline."""

from iadgen_v2.auto_mask.contracts import (
    DEFAULT_AUTO_CANDIDATE_MODES,
    AutoMaskContext,
    AutoMaskRecord,
    CandidateProposal,
    EvidenceMap,
    EvidenceProvider,
    PatchFeatureCache,
    RegistrationResult,
    SelectionDecision,
)

__all__ = [
    "DEFAULT_AUTO_CANDIDATE_MODES",
    "AutoMaskContext",
    "AutoMaskRecord",
    "CandidateProposal",
    "EvidenceMap",
    "EvidenceProvider",
    "PatchFeatureCache",
    "RegistrationResult",
    "SelectionDecision",
]
