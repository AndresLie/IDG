from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iadgen_v2.auto_mask.contracts import CandidateProposal, SelectionDecision
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES


@dataclass(frozen=True)
class CandidatePrediction:
    mode: str
    expected_iou: float
    expected_precision: float
    expected_recall: float
    conformal_iou_lower_bound: float
    source_disagreement: float
    direct_expected_iou: float | None = None
    precision_recall_iou: float | None = None
    prediction_inconsistency: float = 0.0


class GenericCandidateSelector:
    def __init__(self, model_path: Path | None = None, *, edge_swap_margin: float = 0.02) -> None:
        self.model_path = model_path
        self.edge_swap_margin = float(edge_swap_margin)
        self.bundle: dict[str, Any] | None = None
        if model_path is not None and model_path.exists():
            import joblib

            loaded = joblib.load(model_path)
            if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
                raise ValueError(f"Unsupported selector bundle: {model_path}")
            self.bundle = loaded

    def select(self, proposals: list[CandidateProposal]) -> tuple[SelectionDecision, list[CandidatePrediction]]:
        if not proposals:
            return SelectionDecision(None, None, None, None, 0.0, "needs_review", ("no_valid_proposals",)), []
        predictions = [self._predict(proposal) for proposal in proposals]
        rank_key = lambda item: (item.conformal_iou_lower_bound, item.expected_iou)
        best = max(predictions, key=rank_key)
        # Margin gate: an edge-refined candidate only displaces the best
        # non-edge candidate when its uncertainty-discounted IoU clearly wins.
        # Edge candidates are newer to the reliability model, so a near-tie
        # should fall back to the in-distribution candidate rather than risk a
        # confident mis-rank (observed as a zipper regression in the A/B run).
        if best.mode.startswith("edge_"):
            non_edge = [item for item in predictions if not item.mode.startswith("edge_")]
            if non_edge:
                best_non_edge = max(non_edge, key=rank_key)
                if best.conformal_iou_lower_bound < best_non_edge.conformal_iou_lower_bound + self.edge_swap_margin:
                    best = best_non_edge
        if best.conformal_iou_lower_bound >= 0.45 and best.source_disagreement <= 0.20:
            disposition = "hard_mask_ok"
        elif best.expected_iou >= 0.25:
            disposition = "soft_mask_only"
        else:
            disposition = "needs_review"
        confidence = float(np.clip(best.conformal_iou_lower_bound / max(1e-6, best.expected_iou), 0.0, 1.0))
        reasons = (
            f"selector={'calibrated_model' if self.bundle is not None else 'normal_only_heuristic'}",
            f"iou_lower_bound={best.conformal_iou_lower_bound:.4f}",
            f"source_disagreement={best.source_disagreement:.4f}",
        )
        return SelectionDecision(
            selected_mode=best.mode,
            expected_iou=best.expected_iou,
            expected_precision=best.expected_precision,
            expected_recall=best.expected_recall,
            confidence=confidence,
            disposition=disposition,
            reasons=reasons,
            conformal_iou_lower_bound=best.conformal_iou_lower_bound,
            source_disagreement=best.source_disagreement,
        ), predictions

    def _predict(self, proposal: CandidateProposal) -> CandidatePrediction:
        features = np.asarray([[float(proposal.measurements.get(name, 0.0)) for name in MEASUREMENT_NAMES]], dtype=np.float32)
        if self.bundle is not None:
            direct_expected_iou = float(np.clip(self.bundle["iou_model"].predict(features)[0], 0.0, 1.0))
            expected_precision = float(np.clip(self.bundle["precision_model"].predict(features)[0], 0.0, 1.0))
            expected_recall = float(np.clip(self.bundle["recall_model"].predict(features)[0], 0.0, 1.0))
            calibrator = self.bundle.get("iou_calibrator")
            if calibrator is not None:
                direct_expected_iou = float(np.clip(calibrator.predict([direct_expected_iou])[0], 0.0, 1.0))
            residual = float(self.bundle.get("conformal_residual_q90", 0.15))
        else:
            coverage = float(proposal.measurements.get("evidence_coverage", 0.0))
            agreement = float(proposal.measurements.get("source_agreement", 0.0))
            disagreement = float(proposal.measurements.get("source_disagreement", 1.0))
            area = float(proposal.measurements.get("area_fraction", 0.0))
            direct_expected_iou = float(np.clip(0.10 + 0.48 * coverage + 0.30 * agreement - 0.32 * disagreement - 0.25 * max(0.0, area - 0.20), 0.0, 1.0))
            expected_precision = float(np.clip(0.15 + 0.55 * coverage + 0.20 * agreement - 0.25 * disagreement, 0.0, 1.0))
            expected_recall = float(np.clip(0.10 + 0.40 * agreement + 0.30 * min(1.0, area / 0.08) - 0.20 * disagreement, 0.0, 1.0))
            residual = 0.18
        precision_recall_iou = _iou_from_precision_recall(expected_precision, expected_recall)
        # The three regressors are trained independently. On out-of-distribution
        # defects they can disagree sharply, so rank by the conservative value
        # that is consistent with the predicted precision and recall.
        expected_iou = min(direct_expected_iou, precision_recall_iou)
        return CandidatePrediction(
            mode=proposal.mode,
            expected_iou=expected_iou,
            expected_precision=expected_precision,
            expected_recall=expected_recall,
            conformal_iou_lower_bound=max(0.0, expected_iou - residual),
            source_disagreement=float(proposal.measurements.get("source_disagreement", 1.0)),
            direct_expected_iou=direct_expected_iou,
            precision_recall_iou=precision_recall_iou,
            prediction_inconsistency=abs(direct_expected_iou - precision_recall_iou),
        )


def _iou_from_precision_recall(precision: float, recall: float) -> float:
    denominator = precision + recall - precision * recall
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip((precision * recall) / denominator, 0.0, 1.0))


def fit_selector_bundle(rows: list[dict[str, Any]], output_path: Path) -> Path:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict

    if len(rows) < 50:
        raise ValueError("Selector training requires at least 50 synthetic candidate rows")
    features = np.asarray([[float(row.get(name, 0.0)) for name in MEASUREMENT_NAMES] for row in rows], dtype=np.float32)
    groups = np.asarray([str(row["category"]) for row in rows])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Selector training requires at least three development categories")
    targets = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float32) for name in ("iou", "precision", "recall")}
    splitter = GroupKFold(n_splits=len(unique_groups))
    models = {}
    out_of_fold = {}
    for name, target in targets.items():
        estimator = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=17)
        out_of_fold[name] = cross_val_predict(estimator, features, target, groups=groups, cv=splitter)
        estimator.fit(features, target)
        models[f"{name}_model"] = estimator
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(out_of_fold["iou"], targets["iou"])
    calibrated = calibrator.predict(out_of_fold["iou"])
    residual_q90 = float(np.quantile(np.maximum(0.0, calibrated - targets["iou"]), 0.90))
    bundle = {
        "schema_version": 1,
        "measurement_names": list(MEASUREMENT_NAMES),
        **models,
        "iou_calibrator": calibrator,
        "conformal_residual_q90": residual_q90,
        "development_categories": unique_groups.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path
