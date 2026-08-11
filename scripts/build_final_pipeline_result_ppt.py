from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps
from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import build_auto_mask_progress_ppt as base


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "presentations" / "final_pipeline_result"
ASSET_DIR = OUT_DIR / "assets"
PPTX_PATH = OUT_DIR / "current_pipeline_final_result_report.pptx"
OUTLINE_PATH = OUT_DIR / "current_pipeline_final_result_report_outline.md"
DATA_PATH = OUT_DIR / "current_pipeline_final_result_data.json"

V42_DIRS = {
    "V4.2d": ROOT / "reports" / "v4_2d_candidate_decision" / "candidate_ranking",
    "V4.2e": ROOT / "reports" / "v4_2e_area_calibrated_candidates" / "candidate_ranking",
    "V4.2f": ROOT / "reports" / "v4_2f_uncertainty_contours" / "candidate_ranking",
}
RERUN_MANIFEST_DIR = (
    ROOT
    / "outputs"
    / "v4_2f_uncertainty_contours"
    / "experiment_manifests"
    / "auto-mask-train-candidate-calibrator"
)
SELECTOR_EVIDENCE = ROOT / "reports" / "selector_evidence_pack" / "selector_evidence_pack.json"
VISA_FAILURE = (
    ROOT
    / "reports"
    / "v3_generic_evidence_visa"
    / "locked_evaluation"
    / "v3-generic-evidence-rc2-operational-20260726"
    / "locked_failure_analysis.json"
)
R4_RESULT = (
    ROOT
    / "reports"
    / "r4_visibility_reaudit"
    / "arbitrated_comparison"
    / "r4_visibility_reaudit.json"
)
R5_RESULT = (
    ROOT
    / "reports"
    / "r5_independent_synthetic_utility_deterministic"
    / "utility_review"
    / "r5_synthetic_utility_review.json"
)

SOURCE_LABELS = {
    "btad_exposed": "BTAD",
    "kodytek_exposed": "Kodytek",
    "ksdd2_exposed": "KSDD2",
    "mvtec_development": "MVTec dev",
    "visa_exposed": "VisA",
}

NAVY = base.NAVY
NAVY_2 = base.NAVY_2
SLATE = base.SLATE
MUTED = base.MUTED
LIGHT = base.LIGHT
PALE = base.PALE
WHITE = base.WHITE
TEAL = base.TEAL
TEAL_DARK = base.TEAL_DARK
CYAN = base.CYAN
ORANGE = base.ORANGE
RED = base.RED
GREEN = base.GREEN
PURPLE = base.PURPLE


@dataclass(frozen=True)
class CandidateRun:
    name: str
    selected_dice: float
    oracle_dice: float
    baseline_dice: float
    expected_iou_mae: float
    selection_regret: float
    candidate_count: int
    folds: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProjectEvidence:
    candidate_runs: tuple[CandidateRun, ...]
    rerun_status: str
    rerun_elapsed_seconds: float
    rerun_cache_hit: bool
    rerun_model_sha256: str
    rerun_metrics_sha256: str
    selector_regret_gain: float
    selector_regret_ci: tuple[float, float]
    visa_macro_dice: float
    visa_macro_ci: tuple[float, float]
    visa_search_recall: float
    visa_accepted_coverage: float
    r4_visibility_gain: float
    r4_visibility_ci: tuple[float, float]
    r4_acceptance_before: int
    r4_acceptance_after: int
    r5_pixel_ap_gain: float
    r5_pixel_ap_ci: tuple[float, float]
    r5_passed: bool


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    assets = build_assets()

    prs = Presentation()
    prs.slide_width = base.SLIDE_W
    prs.slide_height = base.SLIDE_H

    add_title_slide(prs, assets)
    add_executive_slide(prs, evidence)
    add_motivation_slide(prs)
    add_research_questions_slide(prs)
    add_pipeline_slide(prs)
    add_auto_mask_architecture_slide(prs)
    add_mask_roles_slide(prs)
    add_governance_slide(prs)
    add_dataset_slide(prs)
    add_selector_contribution_slide(prs, evidence)
    add_v42f_headline_slide(prs, evidence)
    add_source_results_slide(prs, evidence)
    add_candidate_evolution_slide(prs, evidence)
    add_gate_audit_slide(prs, evidence)
    add_mvtec_visual_slide(prs, assets)
    add_visa_visual_slide(prs, evidence, assets)
    add_generation_slide(prs, evidence, assets)
    add_downstream_slide(prs, evidence, assets)
    add_contribution_limitations_slide(prs)
    add_paper_positioning_slide(prs)
    add_reproducibility_slide(prs, evidence)

    set_core_properties(prs)
    prs.save(PPTX_PATH)
    write_outline(evidence)
    DATA_PATH.write_text(json.dumps(asdict(evidence), indent=2) + "\n", encoding="utf-8")
    validate_presentation(PPTX_PATH, expected_slides=21)
    print(PPTX_PATH)
    print(OUTLINE_PATH)
    print(DATA_PATH)


def load_evidence() -> ProjectEvidence:
    candidate_runs = []
    for name, report_dir in V42_DIRS.items():
        metrics = load_json(report_dir / "leave_dataset_out_candidate_metrics.json")
        overall = metrics["overall"]
        candidate_runs.append(
            CandidateRun(
                name=name,
                selected_dice=float(overall["dataset_macro_selected_dice"]),
                oracle_dice=float(overall["dataset_macro_oracle_dice"]),
                baseline_dice=float(overall["dataset_macro_baseline_dice"]),
                expected_iou_mae=float(overall["dataset_macro_expected_iou_mae"]),
                selection_regret=float(overall["dataset_macro_selection_regret"]),
                candidate_count=int(overall["held_out_candidate_count"]),
                folds=tuple(metrics["folds"]),
            )
        )

    rerun = load_latest_succeeded_manifest(RERUN_MANIFEST_DIR)
    calibration_manifest = load_json(V42_DIRS["V4.2f"] / "candidate_calibration_manifest.json")
    selector = load_json(SELECTOR_EVIDENCE)["variants"]["A-S1b+A-S2 (widened, deployed pool)"]
    visa = load_json(VISA_FAILURE)
    r4 = load_json(R4_RESULT)
    r5 = load_json(R5_RESULT)
    selector_delta = selector["regret_delta_syn_minus_real"]
    visibility = r4["paired_intervals"]["critic_defect_visibility_score"]
    pixel_ap = r5["intervals"]["pixel_ap"]
    return ProjectEvidence(
        candidate_runs=tuple(candidate_runs),
        rerun_status=str(rerun["status"]),
        rerun_elapsed_seconds=float(rerun["elapsed_seconds"]),
        rerun_cache_hit=bool(calibration_manifest["candidate_row_cache"]["cache_hit"]),
        rerun_model_sha256=str(calibration_manifest["model_sha256"]),
        rerun_metrics_sha256=str(calibration_manifest["metrics_sha256"]),
        selector_regret_gain=float(selector_delta["mean"]),
        selector_regret_ci=tuple(float(value) for value in selector_delta["ci95"]),
        visa_macro_dice=float(visa["macro"]["dice"]),
        visa_macro_ci=tuple(float(value) for value in visa["macro_dice_category_bootstrap_95ci"]),
        visa_search_recall=float(visa["macro"]["search_region_recall"]),
        visa_accepted_coverage=float(visa["macro"]["accepted_coverage"]),
        r4_visibility_gain=float(visibility["mean_delta"]),
        r4_visibility_ci=(float(visibility["ci95_low"]), float(visibility["ci95_high"])),
        r4_acceptance_before=int(r4["baseline"]["accepted"]),
        r4_acceptance_after=int(r4["controller"]["accepted"]),
        r5_pixel_ap_gain=float(pixel_ap["mean_delta"]),
        r5_pixel_ap_ci=(float(pixel_ap["ci95_low"]), float(pixel_ap["ci95_high"])),
        r5_passed=bool(r5["gate"]["passed"]),
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required presentation evidence is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def load_latest_succeeded_manifest(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Experiment manifest directory is missing: {directory}")
    candidates = []
    for path in directory.glob("*.json"):
        if path.name == "latest.json":
            continue
        data = load_json(path)
        if data.get("status") == "succeeded" and "elapsed_seconds" in data:
            candidates.append((str(data.get("finished_at_utc", "")), path.name, data))
    if not candidates:
        raise RuntimeError(f"No succeeded experiment manifest found in {directory}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def build_assets() -> dict[str, Path]:
    bottle_zipper = ROOT / "reports" / "mvtec_bottle_zipper_auto_mask" / "official_mask_evaluation" / "bottle_zipper_auto_mask_visual_review.png"
    wood = ROOT / "reports" / "phase9_reliability_targeted" / "wood_visual_result" / "wood_end_to_end_visual_result.png"
    visa = ROOT / "reports" / "v3_generic_evidence_visa" / "locked_evaluation" / "v3-generic-evidence-rc2-operational-20260726" / "locked_best_worst_contact_sheet.png"
    generation = ROOT / "reports" / "r4_visibility_reaudit" / "arbitrated_comparison" / "r4_visibility_comparison.png"
    prediction = ROOT / "reports" / "r5_independent_synthetic_utility_deterministic" / "phase5" / "qwen" / "prediction_examples" / "supervised_resnet18_unet" / "critic_arbitrated" / "all" / "r0.25_s1337" / "prediction_contact_sheet.png"
    for path in (bottle_zipper, wood, visa, generation, prediction):
        if not path.is_file():
            raise FileNotFoundError(f"Required visual artifact is missing: {path}")

    assets = {
        "bottle": crop_fraction(bottle_zipper, ASSET_DIR / "bottle_masks.png", 0.00, 0.49),
        "zipper": crop_fraction(bottle_zipper, ASSET_DIR / "zipper_masks.png", 0.49, 1.00),
        "wood": crop_fraction(wood, ASSET_DIR / "wood_pipeline.png", 0.00, 0.62),
        "visa": crop_fraction(visa, ASSET_DIR / "visa_best_worst.png", 0.00, 0.39),
        "generation": crop_fraction(generation, ASSET_DIR / "generation_wood.png", 0.62, 1.00),
        "prediction": resize_image(prediction, ASSET_DIR / "r5_prediction.png", 760),
    }
    assets["title"] = make_mosaic(
        [assets["bottle"], assets["zipper"], assets["wood"], assets["visa"]],
        ASSET_DIR / "title_mosaic.png",
    )
    return assets


def crop_fraction(source: Path, target: Path, top_fraction: float, bottom_fraction: float) -> Path:
    image = Image.open(source).convert("RGB")
    top = max(0, min(image.height - 1, int(round(image.height * top_fraction))))
    bottom = max(top + 1, min(image.height, int(round(image.height * bottom_fraction))))
    crop = image.crop((0, top, image.width, bottom))
    if crop.width > 1300:
        height = int(round(crop.height * 1300 / crop.width))
        crop = crop.resize((1300, height), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    crop.save(target, optimize=True)
    return target


def resize_image(source: Path, target: Path, max_width: int) -> Path:
    image = Image.open(source).convert("RGB")
    if image.width > max_width:
        height = int(round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    return target


def make_mosaic(paths: list[Path], target: Path) -> Path:
    canvas = Image.new("RGB", (1200, 800), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    boxes = [(0, 0, 600, 400), (600, 0, 1200, 400), (0, 400, 600, 800), (600, 400, 1200, 800)]
    for path, box in zip(paths, boxes):
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(
            image,
            (box[2] - box[0], box[3] - box[1]),
            method=Image.Resampling.LANCZOS,
        )
        canvas.paste(image, (box[0], box[1]))
        draw.rectangle(box, outline=(71, 85, 105), width=3)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)
    return target


def set_core_properties(prs: Presentation) -> None:
    props = prs.core_properties
    props.title = "ImgGen v2 Final Pipeline Result Report"
    props.subject = "Automatic pseudo-mask generation, synthetic generation, and downstream evaluation"
    props.author = "ImgGen v2 Research Project"
    props.keywords = "industrial anomaly, pseudo-mask, Qwen, DINOv2, SAM2, candidate selection, synthetic defects"
    props.comments = "Generated from the frozen V4.2f rerun and canonical project evidence."


def blank_slide(prs: Presentation, title: str, section: str) -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = LIGHT
    base.add_rect(slide, 0, 0, 13.333, 0.12, NAVY, line=None)
    base.add_text(slide, 0.55, 0.27, 10.8, 0.48, title, 25, NAVY, bold=True)
    base.add_text(slide, 10.75, 0.32, 1.95, 0.25, section.upper(), 8.5, TEAL_DARK, bold=True, align=PP_ALIGN.RIGHT)
    add_footer(slide, len(prs.slides))
    return slide


def add_footer(slide: object, number: int) -> None:
    base.add_line(slide, 0.55, 7.08, 12.75, 7.08, RGBColor(203, 213, 225), 0.7)
    base.add_text(slide, 0.58, 7.12, 5.8, 0.18, "ImgGen v2 | Final Pipeline Result", 8, MUTED)
    base.add_text(slide, 11.8, 7.12, 0.9, 0.18, f"{number:02d}", 8, MUTED, align=PP_ALIGN.RIGHT)


def add_title_slide(prs: Presentation, assets: dict[str, Path]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    base.add_rect(slide, 0, 0, 0.18, 7.5, TEAL, line=None)
    base.add_text(slide, 0.72, 0.64, 7.0, 0.35, "FINAL RESEARCH CHECKPOINT", 12, RGBColor(94, 234, 212), bold=True)
    base.add_text(slide, 0.72, 1.18, 6.65, 1.55, "Automatic industrial defect\npseudo-mask pipeline", 31, WHITE, bold=True)
    base.add_text(slide, 0.75, 3.02, 6.35, 0.78, "Architecture, reproduced results, generalization evidence, and honest limitations", 17, RGBColor(203, 213, 225))
    base.add_pill(slide, 0.75, 4.15, 1.62, 0.38, "1,476 images", TEAL)
    base.add_pill(slide, 2.51, 4.15, 1.82, 0.38, "5 data sources", PURPLE)
    base.add_pill(slide, 4.47, 4.15, 1.72, 0.38, "292 tests", GREEN)
    base.add_text(slide, 0.75, 6.58, 6.4, 0.32, "Frozen V4.2f rerun | August 2026", 11, RGBColor(148, 163, 184))
    base.add_picture_cover(slide, assets["title"], 7.55, 0.62, 5.25, 6.2, radius=True)


def add_executive_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "Executive conclusion", "Summary")
    base.add_text(slide, 0.75, 0.95, 11.8, 0.65, "The research implementation is complete. Further tuning on the same exposed data is unlikely to strengthen the paper.", 20, NAVY, bold=True, align=PP_ALIGN.CENTER)
    run = run_by_name(evidence, "V4.2f")
    cards = [
        (f"{run.oracle_dice:.3f}", "macro oracle Dice", "+0.013 vs V4.2d", GREEN),
        (f"{run.selected_dice:.3f}", "macro selected Dice", "-0.006 vs V4.2d", RED),
        (f"{evidence.visa_macro_dice:.3f}", "locked VisA Dice", "generalization gate failed", RED),
        (f"{evidence.r5_pixel_ap_gain:+.3f}", "synthetic pixel-AP delta", "95% CI crosses zero", ORANGE),
    ]
    for index, card in enumerate(cards):
        base.add_metric_card(slide, 0.62 + index * 3.08, 1.95, 2.78, 1.46, *card)
    base.add_callout(slide, 0.78, 4.05, 3.65, 1.45, "Confirmed contribution", "Real-candidate, leave-category-out selector calibration reduces regret with a confidence interval excluding zero.", TEAL)
    base.add_callout(slide, 4.84, 4.05, 3.65, 1.45, "Negative result", "The generic auto-mask architecture does not transfer at release quality to locked VisA categories.", RED)
    base.add_callout(slide, 8.90, 4.05, 3.65, 1.45, "Paper-ready position", "Present the method, validated selector result, null synthesis utility, and measured failure modes without overclaiming.", PURPLE)
    base.add_text(slide, 1.0, 6.25, 11.3, 0.38, "Decision: freeze the architecture and move from model development to paper packaging.", 15, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_motivation_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Why automatic pseudo-masks matter", "Motivation")
    base.add_text(slide, 0.75, 0.92, 11.8, 0.55, "Many controllable industrial defect-generation workflows begin with a manually supplied bounding box or pixel mask.", 20, NAVY, bold=True, align=PP_ALIGN.CENTER)
    stages = [
        ("Inspect", "Find the defect", ORANGE),
        ("Annotate", "Draw box or mask", ORANGE),
        ("Generate", "Diffusion / inpaint", PURPLE),
        ("Train", "Use image-mask pairs", TEAL),
    ]
    for index, (title, body, color) in enumerate(stages):
        x = 0.78 + index * 3.08
        base.draw_pipeline_box(slide, x, 2.05, 2.45, 1.22, title, body, color, str(index + 1))
        if index < len(stages) - 1:
            base.add_arrow(slide, x + 2.48, 2.67, x + 2.90, 2.67, SLATE)
    base.add_callout(slide, 0.88, 4.02, 3.45, 1.42, "Cost", "Pixel masks require skilled labor and are difficult to scale across parts and factories.", ORANGE)
    base.add_callout(slide, 4.92, 4.02, 3.45, 1.42, "Consistency", "Boxes and masks vary by annotator, especially for fuzzy, broad, or low-contrast defects.", RED)
    base.add_callout(slide, 8.96, 4.02, 3.45, 1.42, "Project objective", "Infer uncertainty-aware, purpose-specific pseudo-masks from normal images and a few unlabeled defects.", TEAL)
    base.add_text(slide, 0.8, 6.28, 11.75, 0.42, "The project removes manual spatial annotation from the user workflow, but does not claim automatic masks are human ground truth.", 13, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_research_questions_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Research questions and claim boundaries", "Framing")
    questions = [
        ("RQ1", "Can normal-reference and multimodal evidence generate useful automatic pseudo-masks?", TEAL),
        ("RQ2", "Can a category-agnostic selector choose the right candidate under domain shift?", PURPLE),
        ("RQ3", "Do selected masks produce visible, low-leakage synthetic defects?", ORANGE),
        ("RQ4", "Does the synthetic corpus improve an independent downstream student?", RED),
    ]
    for index, (tag, text, color) in enumerate(questions):
        y = 1.0 + index * 1.28
        base.add_rect(slide, 0.78, y, 11.78, 0.96, WHITE, line=RGBColor(203, 213, 225), radius=True)
        base.add_pill(slide, 1.02, y + 0.22, 0.72, 0.42, tag, color, font_size=11)
        base.add_text(slide, 2.02, y + 0.20, 9.95, 0.48, text, 15, NAVY, bold=True, valign=MSO_ANCHOR.MIDDLE)
    base.add_text(slide, 0.9, 6.35, 11.5, 0.38, "Confirmed: selector calibration. Mixed: mask architecture and visual synthesis. Null: downstream synthetic utility.", 14, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_pipeline_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "End-to-end research pipeline", "Architecture")
    labels = [
        ("Inputs", "normal + few defects", GREEN),
        ("Localize", "Qwen soft regions", CYAN),
        ("Evidence", "DINO / residual / texture", PURPLE),
        ("Propose", "binary + soft candidates", ORANGE),
        ("Select", "calibrated rank + abstain", TEAL),
        ("Generate", "SD1.5 + critic", RED),
    ]
    for index, (title, body, color) in enumerate(labels):
        x = 0.45 + index * 2.12
        base.draw_stage_card(slide, x, 1.35, 1.8, 1.34, title, body, color, index + 1)
        if index < len(labels) - 1:
            base.add_arrow(slide, x + 1.82, 2.02, x + 2.05, 2.02, SLATE)
    base.add_rect(slide, 0.78, 3.35, 11.78, 2.25, PALE, line=RGBColor(203, 213, 225), radius=True)
    base.add_text(slide, 1.0, 3.58, 3.4, 0.3, "Outputs are role-specific", 16, NAVY, bold=True)
    base.add_bullet_list(slide, 1.0, 4.04, 3.5, 1.2, ["eval_tight", "training_soft / medium", "uncertainty_map", "inpaint_soft"], 11, NAVY)
    base.add_text(slide, 4.80, 3.58, 3.4, 0.3, "Evidence is auditable", 16, NAVY, bold=True)
    base.add_bullet_list(slide, 4.80, 4.04, 3.5, 1.2, ["candidate metadata", "source reliability", "selection confidence", "visual contact sheets"], 11, NAVY)
    base.add_text(slide, 8.60, 3.58, 3.4, 0.3, "Claims are gated", 16, NAVY, bold=True)
    base.add_bullet_list(slide, 8.60, 4.04, 3.5, 1.2, ["official-mask firewall", "leave-source-out folds", "preregistered thresholds", "locked evaluation"], 11, NAVY)
    base.add_text(slide, 0.9, 6.28, 11.6, 0.4, "The architecture is complete; the measured evidence determines which components remain experimental.", 14, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_auto_mask_architecture_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Automatic mask generation core", "Architecture")
    columns = [
        ("Context", ["defect image", "up to 16 normals", "Qwen soft prior", "foreground estimate"], CYAN),
        ("Evidence", ["DINO memory", "registered residual", "MuSc rarity", "FFT / texture", "PatchCore guided"], PURPLE),
        ("Candidates", ["fused quantiles", "components", "SAM2 refinement", "posterior compact", "uncertainty contours"], ORANGE),
        ("Decision", ["candidate features", "list ranking", "risk calibration", "hard / soft / review"], TEAL),
    ]
    for index, (title, items, color) in enumerate(columns):
        x = 0.55 + index * 3.15
        base.add_rect(slide, x, 1.02, 2.8, 4.85, WHITE, line=RGBColor(203, 213, 225), radius=True)
        base.add_rect(slide, x, 1.02, 2.8, 0.68, color, line=None, radius=True)
        base.add_text(slide, x + 0.12, 1.19, 2.56, 0.3, title, 16, WHITE, bold=True, align=PP_ALIGN.CENTER)
        for item_index, item in enumerate(items):
            y = 1.98 + item_index * 0.72
            base.add_rect(slide, x + 0.24, y, 2.32, 0.48, PALE, line=RGBColor(226, 232, 240), radius=True)
            base.add_text(slide, x + 0.34, y + 0.10, 2.12, 0.25, item, 10.5, NAVY, bold=True, align=PP_ALIGN.CENTER)
        if index < len(columns) - 1:
            base.add_arrow(slide, x + 2.82, 3.42, x + 3.08, 3.42, SLATE)
    base.add_text(slide, 0.88, 6.25, 11.6, 0.45, "Zero-shot rule: category and defect names do not route the generic headline path; official masks never create candidates.", 13, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_mask_roles_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "One defect, multiple mask roles", "Pseudo-labels")
    roles = [
        ("positive_core", "high-confidence pixels", GREEN),
        ("eval_tight", "conservative binary evaluation", TEAL),
        ("training_medium", "hard line-like training", CYAN),
        ("training_soft", "fuzzy uncertainty-aware target", PURPLE),
        ("possible_region", "supported outer extent", ORANGE),
        ("uncertainty_map", "source and boundary disagreement", RED),
        ("inpaint_soft", "soft generation envelope", SLATE),
    ]
    for index, (name, body, color) in enumerate(roles):
        row, col = divmod(index, 4)
        x = 0.58 + col * 3.12
        y = 1.05 + row * 2.12
        base.add_callout(slide, x, y, 2.83, 1.55, name, body, color)
    base.add_rect(slide, 0.95, 5.58, 11.42, 0.72, PALE, line=RGBColor(203, 213, 225), radius=True)
    base.add_text(slide, 1.15, 5.78, 11.0, 0.3, "Ambiguous pixels are down-weighted in Phase 3 and Phase 5 instead of being treated as certain foreground.", 13, NAVY, bold=True, align=PP_ALIGN.CENTER)
    base.add_text(slide, 0.9, 6.50, 11.5, 0.3, "This is an engineering strength; downstream gains still require independent validation.", 11.5, MUTED, align=PP_ALIGN.CENTER)


def add_governance_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Evaluation firewall and scientific integrity", "Protocol")
    base.add_text(slide, 0.82, 0.96, 11.7, 0.46, "Runtime and evaluation data are separated before execution.", 19, NAVY, bold=True, align=PP_ALIGN.CENTER)
    base.draw_pipeline_box(slide, 0.85, 1.82, 3.15, 1.35, "Runtime tree", "normal + unlabeled defect images", TEAL, "A")
    base.draw_pipeline_box(slide, 5.08, 1.82, 3.15, 1.35, "Sealed outputs", "masks + hashes + manifests", PURPLE, "B")
    base.draw_pipeline_box(slide, 9.30, 1.82, 3.15, 1.35, "Locked evaluate", "official masks opened afterward", ORANGE, "C")
    base.add_arrow(slide, 4.05, 2.50, 4.91, 2.50, SLATE)
    base.add_arrow(slide, 8.28, 2.50, 9.13, 2.50, SLATE)
    checks = [
        ("Content hashes", "datasets, configs, models, reports", GREEN),
        ("Leave-source-out", "five exposed sources, no in-fold labels", TEAL),
        ("Preregistration", "thresholds fixed before result access", PURPLE),
        ("Negative results", "failed gates retained and reported", RED),
    ]
    for index, item in enumerate(checks):
        base.add_factor_card(slide, 0.65 + index * 3.12, 4.10, 2.82, 1.27, *item)
    base.add_text(slide, 0.9, 6.20, 11.5, 0.42, "This firewall is part of the contribution: it prevents benchmark masks from silently becoming training signals.", 13.5, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_dataset_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Evidence base", "Protocol")
    rows = [
        ["MVTec development", "144", "5", "exposed development"],
        ["VisA", "1,200", "12", "locked once, then exposed for V4"],
        ["BTAD", "60", "3", "external development"],
        ["KSDD2", "60", "1", "external development"],
        ["Kodytek wood", "12", "1", "external development"],
        ["V4.2f total", "1,476", "10 source categories", "leave-dataset-out"],
    ]
    base.add_table(slide, 0.72, 1.02, 7.15, 4.95, ["Dataset", "Images", "Categories", "Role"], rows, [2.2, 0.9, 1.2, 2.6])
    base.add_callout(slide, 8.30, 1.08, 4.28, 1.35, "Development evidence", "V4.2f compares proposal and selection behavior across five complete source-held-out folds.", TEAL)
    base.add_callout(slide, 8.30, 2.72, 4.28, 1.35, "Locked evidence", "MVTec locked categories confirm selector benefit but reject release quality. VisA exposes severe transfer failure.", RED)
    base.add_callout(slide, 8.30, 4.36, 4.28, 1.35, "Interpretation rule", "VisA can no longer support another locked claim. A future confirmatory run needs a new untouched benchmark.", ORANGE)
    base.add_text(slide, 0.85, 6.35, 11.8, 0.3, "More tuning on these exposed sets would increase researcher degrees of freedom without creating stronger evidence.", 12.5, MUTED, italic=True, align=PP_ALIGN.CENTER)


def add_selector_contribution_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "Confirmed contribution: real-candidate selector calibration", "Results")
    lo, hi = evidence.selector_regret_ci
    base.add_metric_card(slide, 0.72, 1.04, 3.48, 1.55, f"{evidence.selector_regret_gain:+.3f}", "regret reduction vs synthetic selector", f"95% CI [{lo:+.3f}, {hi:+.3f}]", TEAL)
    base.add_callout(slide, 4.62, 1.04, 3.75, 1.55, "Matched candidate pool", "The comparison changes the selector, not proposal availability. Candidate-pool hashes are identical within each arm.", PURPLE)
    base.add_callout(slide, 8.78, 1.04, 3.75, 1.55, "Leave-category-out", "Each development category is predicted by calibration trained without that category.", GREEN)
    base.add_text(slide, 0.82, 3.10, 5.55, 0.34, "Mean selection regret", 16, NAVY, bold=True)
    add_simple_bars(slide, 0.82, 3.55, 5.55, 2.00, ["Synthetic calibration", "Real LCO calibration"], [0.3515, 0.1571], [RED, TEAL], maximum=0.40)
    base.add_text(slide, 6.82, 3.10, 5.55, 0.34, "Mean selected Dice", 16, NAVY, bold=True)
    add_simple_bars(slide, 6.82, 3.55, 5.55, 2.00, ["Synthetic calibration", "Real LCO calibration"], [0.2248, 0.4692], [RED, TEAL], maximum=0.55)
    base.add_text(slide, 0.9, 6.24, 11.5, 0.42, "This is the paper's strongest positive quantitative claim. It should lead the contribution list.", 14, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def add_v42f_headline_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "V4.2f reproduced headline result", "Results")
    d = run_by_name(evidence, "V4.2d")
    f = run_by_name(evidence, "V4.2f")
    cards = [
        (f"{d.oracle_dice:.3f} -> {f.oracle_dice:.3f}", "macro oracle Dice", f"gain {f.oracle_dice - d.oracle_dice:+.4f}", GREEN),
        (f"{d.selected_dice:.3f} -> {f.selected_dice:.3f}", "macro selected Dice", f"change {f.selected_dice - d.selected_dice:+.4f}", RED),
        (f"{f.candidate_count:,}", "held-out candidates", f"{(f.candidate_count / d.candidate_count - 1) * 100:.2f}% growth", PURPLE),
    ]
    for index, card in enumerate(cards):
        base.add_metric_card(slide, 0.72 + index * 4.08, 1.08, 3.68, 1.55, *card)
    base.add_text(slide, 0.82, 3.10, 11.6, 0.34, "Bounded disagreement-aware contours improve the candidate ceiling, but the selector cannot convert that ceiling into reliable utility.", 16, NAVY, bold=True, align=PP_ALIGN.CENTER)
    base.add_callout(slide, 0.82, 3.76, 3.56, 1.58, "Proposal result", "All five source oracles are non-regressive. New contours become strict oracle winners on 125 images.", GREEN)
    base.add_callout(slide, 4.88, 3.76, 3.56, 1.58, "Selection result", "Only 12 new contours are selected. Kodytek falls back to its weaker baseline after refitting.", RED)
    base.add_callout(slide, 8.94, 3.76, 3.56, 1.58, "Decision", "Keep the mechanism as research infrastructure, leave it default-off, and preserve V4.2d fallback.", ORANGE)
    base.add_text(slide, 0.9, 6.30, 11.5, 0.4, f"Rerun status: {evidence.rerun_status}; cache hit: {evidence.rerun_cache_hit}; elapsed: {evidence.rerun_elapsed_seconds:.1f} s", 12, MUTED, align=PP_ALIGN.CENTER)


def add_source_results_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "V4.2f result by source", "Results")
    run = run_by_name(evidence, "V4.2f")
    labels = [SOURCE_LABELS[fold["dataset_id"]] for fold in run.folds]
    baseline = [float(fold["baseline_dice"]) for fold in run.folds]
    selected = [float(fold["selected_dice"]) for fold in run.folds]
    oracle = [float(fold["oracle_dice"]) for fold in run.folds]
    add_clustered_chart(slide, 0.72, 1.04, 8.0, 5.45, labels, [("Baseline", baseline, SLATE), ("Selected", selected, TEAL), ("Oracle", oracle, ORANGE)], maximum=0.75)
    base.add_callout(slide, 9.04, 1.20, 3.55, 1.35, "Best behavior", "BTAD and MVTec obtain small selected gains; KSDD2 safely abstains at baseline.", GREEN)
    base.add_callout(slide, 9.04, 2.92, 3.55, 1.35, "Main regression", "Kodytek selected Dice drops 0.3625 -> 0.3164 because the refitted fold no longer overrides.", RED)
    base.add_callout(slide, 9.04, 4.64, 3.55, 1.35, "Binding failure", "VisA oracle barely moves and selected output remains at the weak baseline.", ORANGE)


def add_candidate_evolution_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "Candidate-quality sprint evolution", "Ablation")
    labels = [run.name for run in evidence.candidate_runs]
    selected = [run.selected_dice for run in evidence.candidate_runs]
    oracle = [run.oracle_dice for run in evidence.candidate_runs]
    add_clustered_chart(slide, 0.68, 1.05, 7.25, 4.85, labels, [("Selected Dice", selected, TEAL), ("Oracle Dice", oracle, ORANGE)], maximum=0.58)
    rows = []
    for run in evidence.candidate_runs:
        rows.append([run.name, f"{run.candidate_count:,}", f"{run.expected_iou_mae:.4f}", f"{run.selection_regret:.4f}"])
    base.add_table(slide, 8.25, 1.20, 4.35, 3.45, ["Sprint", "Candidates", "MAE", "Regret"], rows, [1.0, 1.4, 1.0, 1.0])
    base.add_callout(slide, 8.25, 4.95, 4.35, 1.25, "Interpretation", "V4.2f recovers almost all V4.2e oracle gain with 22.6% fewer candidates, but neither proposal family passes deployment.", PURPLE)
    base.add_text(slide, 0.88, 6.38, 6.9, 0.3, "The widening oracle-selection gap is the reason to stop fixed candidate sweeps.", 12.5, MUTED, italic=True, align=PP_ALIGN.CENTER)


def add_gate_audit_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "V4.2f preregistered gate audit", "Decision")
    d = run_by_name(evidence, "V4.2d")
    f = run_by_name(evidence, "V4.2f")
    visa_d = fold_by_dataset(d, "visa_exposed")
    visa_f = fold_by_dataset(f, "visa_exposed")
    rows = [
        ["VisA oracle gain", ">= +0.030", f"{visa_f['oracle_dice'] - visa_d['oracle_dice']:+.4f}", "FAIL"],
        ["Macro oracle gain", ">= +0.010", f"{f.oracle_dice - d.oracle_dice:+.4f}", "PASS"],
        ["Per-source oracle", "no regression", "all non-regressive", "PASS"],
        ["Candidate growth", "<= 20%", f"{(f.candidate_count / d.candidate_count - 1) * 100:.2f}%", "PASS"],
        ["Selected regression", "<= 0.005", f"{d.selected_dice - f.selected_dice:.5f}", "FAIL"],
        ["Expected-IoU MAE", "<= 0.12/source", "Kodytek 0.1264", "FAIL"],
    ]
    add_gate_table(slide, 0.75, 1.10, 7.75, 5.25, rows)
    base.add_callout(slide, 8.85, 1.28, 3.65, 1.48, "Proposal mechanism", "The bounded contour family is valid and efficient. Macro ceiling improvement is real.", GREEN)
    base.add_callout(slide, 8.85, 3.12, 3.65, 1.48, "Primary endpoint", "The intended cross-domain VisA boundary problem remains essentially unchanged.", RED)
    base.add_callout(slide, 8.85, 4.96, 3.65, 1.48, "Promotion decision", "No deployment. Do not round the near-miss selected guard into a pass.", ORANGE)


def add_mvtec_visual_slide(prs: Presentation, assets: dict[str, Path]) -> None:
    slide = blank_slide(prs, "Visual evidence: MVTec and wood", "Qualitative")
    panels = [
        ("Bottle", assets["bottle"], "Curved rims and contamination show both tight successes and over-segmented masks."),
        ("Zipper", assets["zipper"], "Repeated tooth geometry is localized well in some cases; border and tiny defects remain difficult."),
        ("Wood", assets["wood"], "Morphology routing separates hard scratches from fuzzy soft-label scuffs."),
    ]
    for index, (title, path, caption) in enumerate(panels):
        x = 0.48 + index * 4.28
        base.add_text(slide, x, 0.92, 3.95, 0.3, title, 16, NAVY, bold=True, align=PP_ALIGN.CENTER)
        base.add_rect(slide, x, 1.32, 3.95, 4.40, WHITE, line=RGBColor(203, 213, 225), radius=True)
        base.add_picture_contain(slide, path, x + 0.10, 1.42, 3.75, 4.18)
        base.add_text(slide, x + 0.05, 5.88, 3.85, 0.70, caption, 9.5, MUTED, align=PP_ALIGN.CENTER)
    base.add_text(slide, 0.85, 6.64, 11.65, 0.26, "Visual masks are pseudo-labels. Official masks appear only in evaluation overlays.", 10.5, RED, bold=True, align=PP_ALIGN.CENTER)


def add_visa_visual_slide(prs: Presentation, evidence: ProjectEvidence, assets: dict[str, Path]) -> None:
    slide = blank_slide(prs, "Locked VisA transfer: the architecture-level failure", "Generalization")
    base.add_rect(slide, 0.55, 1.00, 7.55, 5.60, WHITE, line=RGBColor(203, 213, 225), radius=True)
    base.add_picture_contain(slide, assets["visa"], 0.68, 1.12, 7.30, 5.35)
    lo, hi = evidence.visa_macro_ci
    base.add_metric_card(slide, 8.45, 1.10, 4.02, 1.55, f"{evidence.visa_macro_dice:.3f}", "macro Dice", f"95% CI [{lo:.3f}, {hi:.3f}]", RED)
    base.add_metric_card(slide, 8.45, 2.98, 4.02, 1.55, f"{evidence.visa_search_recall:.3f}", "search-region recall", "localization also transfers weakly", ORANGE)
    base.add_metric_card(slide, 8.45, 4.86, 4.02, 1.55, f"{evidence.visa_accepted_coverage:.3f}", "accepted coverage", "quality confidence is miscalibrated", PURPLE)
    base.add_text(slide, 0.88, 6.66, 11.65, 0.24, "High recall often comes with broad masks; only chewing-gum exceeds 0.30 category Dice.", 10.5, RED, bold=True, align=PP_ALIGN.CENTER)


def add_generation_slide(prs: Presentation, evidence: ProjectEvidence, assets: dict[str, Path]) -> None:
    slide = blank_slide(prs, "Synthetic generation: visible improvement, incomplete validation", "Generation")
    base.add_rect(slide, 0.55, 1.00, 7.45, 5.62, WHITE, line=RGBColor(203, 213, 225), radius=True)
    base.add_picture_contain(slide, assets["generation"], 0.68, 1.14, 7.18, 5.32)
    lo, hi = evidence.r4_visibility_ci
    base.add_metric_card(slide, 8.35, 1.08, 4.10, 1.52, f"{evidence.r4_visibility_gain:+.3f}", "visibility-score gain", f"95% CI [{lo:+.3f}, {hi:+.3f}]", GREEN)
    base.add_metric_card(slide, 8.35, 2.92, 4.10, 1.52, f"{evidence.r4_acceptance_before} -> {evidence.r4_acceptance_after}", "critic acceptance", "18 matched inputs", TEAL)
    base.add_callout(slide, 8.35, 4.76, 4.10, 1.42, "Important caveat", "The critic is not an independent human judge. Two-reviewer blind validation remains incomplete.", ORANGE)
    base.add_text(slide, 0.86, 6.67, 11.65, 0.22, "Generation capability is demonstrated; a general perceptual-quality claim is not.", 10.5, MUTED, italic=True, align=PP_ALIGN.CENTER)


def add_downstream_slide(prs: Presentation, evidence: ProjectEvidence, assets: dict[str, Path]) -> None:
    slide = blank_slide(prs, "Independent downstream utility: null result", "Evaluation")
    base.add_rect(slide, 0.72, 1.05, 4.30, 5.55, WHITE, line=RGBColor(203, 213, 225), radius=True)
    base.add_picture_contain(slide, assets["prediction"], 0.92, 1.25, 3.90, 5.10)
    lo, hi = evidence.r5_pixel_ap_ci
    base.add_metric_card(slide, 5.45, 1.15, 3.25, 1.55, f"{evidence.r5_pixel_ap_gain:+.4f}", "pixel-AP delta", f"95% CI [{lo:+.4f}, {hi:+.4f}]", RED)
    base.add_metric_card(slide, 9.10, 1.15, 3.25, 1.55, "5", "paired training seeds", "strict deterministic execution", TEAL)
    base.add_callout(slide, 5.45, 3.15, 3.25, 1.55, "Fair comparison", "Same ResNet18 U-Net, matched optimizer steps, paired seeds, and no PatchCore fusion.", GREEN)
    base.add_callout(slide, 9.10, 3.15, 3.25, 1.55, "Promotion gate", "Failed: the mean gain is below +0.02 and the bootstrap interval includes zero.", RED)
    base.add_rect(slide, 5.45, 5.18, 6.90, 0.90, PALE, line=RGBColor(203, 213, 225), radius=True)
    base.add_text(slide, 5.72, 5.42, 6.35, 0.38, "Conclusion: synthetic generation is optional augmentation, not a proven detector improvement.", 13, NAVY, bold=True, align=PP_ALIGN.CENTER)
    base.add_text(slide, 5.45, 6.38, 6.9, 0.28, f"Gate passed: {evidence.r5_passed}", 10.5, MUTED, align=PP_ALIGN.CENTER)


def add_contribution_limitations_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "What the project contributes, and what it does not", "Discussion")
    base.add_text(slide, 0.78, 0.92, 5.65, 0.35, "Defensible contributions", 18, TEAL_DARK, bold=True)
    base.add_text(slide, 6.92, 0.92, 5.65, 0.35, "Measured limitations", 18, RED, bold=True)
    positives = [
        "End-to-end annotation-free pseudo-mask workflow",
        "Frozen multimodal and normal-reference evidence field",
        "Purpose-specific hard, soft, and uncertainty masks",
        "Real-candidate selector calibration with positive CI",
        "Official-mask firewall and reproducible manifests",
        "Honest negative-result and abstention reporting",
    ]
    negatives = [
        "Generic masks do not transfer at release quality",
        "Candidate oracle remains low on VisA",
        "Selection destabilizes when candidate pools expand",
        "Qwen localization can still fail under domain shift",
        "Generation critic lacks completed human validation",
        "Synthetic utility confidence interval includes zero",
    ]
    base.add_rect(slide, 0.68, 1.42, 5.92, 4.78, WHITE, line=RGBColor(203, 213, 225), radius=True)
    base.add_rect(slide, 6.74, 1.42, 5.92, 4.78, WHITE, line=RGBColor(203, 213, 225), radius=True)
    base.add_bullet_list(slide, 1.0, 1.75, 5.30, 4.12, positives, 12, NAVY, bullet_color=TEAL, spacing=10)
    base.add_bullet_list(slide, 7.06, 1.75, 5.30, 4.12, negatives, 12, NAVY, bullet_color=RED, spacing=10)
    base.add_text(slide, 0.85, 6.45, 11.65, 0.30, "The limitations do not invalidate the work; they define the correct scope of the paper.", 13, PURPLE, bold=True, align=PP_ALIGN.CENTER)


def add_paper_positioning_slide(prs: Presentation) -> None:
    slide = blank_slide(prs, "Recommended paper positioning", "Conclusion")
    base.add_rect(slide, 0.75, 1.02, 11.82, 1.15, NAVY, line=None, radius=True)
    base.add_text(slide, 1.02, 1.27, 11.28, 0.58, "A rigorous study of automatic pseudo-label generation for scarce-data industrial defects, showing where uncertainty-aware selection helps and where current multimodal evidence fails to generalize.", 16.5, WHITE, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    sections = [
        ("Lead claim", "Real-candidate calibration improves candidate choice under matched pools.", TEAL),
        ("System claim", "The full automatic pipeline runs reproducibly from inputs through visual evidence.", PURPLE),
        ("Negative finding", "Generic transfer and synthetic downstream utility do not meet promotion gates.", RED),
        ("Scientific value", "The project localizes failure to proposal quality, calibration transfer, and domain shift.", ORANGE),
    ]
    for index, section in enumerate(sections):
        row, col = divmod(index, 2)
        base.add_callout(slide, 0.88 + col * 6.04, 2.72 + row * 1.72, 5.55, 1.35, *section)
    base.add_text(slide, 0.9, 6.34, 11.55, 0.42, "Do not market V4.2f as a deployed improvement. Use it as a preregistered boundary on diminishing returns.", 13, RED, bold=True, align=PP_ALIGN.CENTER)


def add_reproducibility_slide(prs: Presentation, evidence: ProjectEvidence) -> None:
    slide = blank_slide(prs, "Final checkpoint and reproducibility", "Conclusion")
    cards = [
        ("292", "automated tests", "all passing", GREEN),
        ("1,476", "V4.2f images", "five source-held-out folds", TEAL),
        ("32,244", "candidate records", "bounded +15.92% pool", PURPLE),
        (f"{evidence.rerun_elapsed_seconds:.0f}s", "cache-hit rerun", evidence.rerun_status, ORANGE),
    ]
    for index, card in enumerate(cards):
        base.add_metric_card(slide, 0.58 + index * 3.12, 1.02, 2.84, 1.48, *card)
    base.add_rect(slide, 0.78, 3.03, 11.78, 2.55, PALE, line=RGBColor(203, 213, 225), radius=True)
    base.add_text(slide, 1.04, 3.28, 2.45, 0.28, "Frozen checkpoint", 15, NAVY, bold=True)
    base.add_text(slide, 3.72, 3.28, 8.3, 0.28, "19bf4bf  Complete V4.2f uncertainty-contour study", 12, NAVY, bold=True, font=base.FONT_MONO)
    base.add_text(slide, 1.04, 3.94, 2.45, 0.28, "Model SHA-256", 12, MUTED, bold=True)
    base.add_text(slide, 3.72, 3.94, 8.3, 0.34, evidence.rerun_model_sha256, 9.5, NAVY, font=base.FONT_MONO)
    base.add_text(slide, 1.04, 4.60, 2.45, 0.28, "Metrics SHA-256", 12, MUTED, bold=True)
    base.add_text(slide, 3.72, 4.60, 8.3, 0.34, evidence.rerun_metrics_sha256, 9.5, NAVY, font=base.FONT_MONO)
    base.add_text(slide, 0.9, 5.98, 11.55, 0.36, "Final action: freeze architecture, preserve failed gates, and package the paper figures, tables, limitations, and reproducibility commands.", 14, TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
    base.add_text(slide, 0.9, 6.48, 11.55, 0.28, "No additional same-data contour or selector tuning is recommended.", 11.5, RED, bold=True, align=PP_ALIGN.CENTER)


def add_simple_bars(
    slide: object,
    x: float,
    y: float,
    w: float,
    h: float,
    labels: list[str],
    values: list[float],
    colors: list[RGBColor],
    *,
    maximum: float,
) -> None:
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        yy = y + index * 0.86
        base.add_text(slide, x, yy, 1.65, 0.28, label, 10, NAVY, bold=True)
        base.add_rect(slide, x + 1.72, yy, w - 2.35, 0.32, RGBColor(226, 232, 240), line=None, radius=True)
        bar_width = max(0.02, (w - 2.35) * max(0.0, min(value / maximum, 1.0)))
        base.add_rect(slide, x + 1.72, yy, bar_width, 0.32, color, line=None, radius=True)
        base.add_text(slide, x + w - 0.58, yy - 0.01, 0.55, 0.28, f"{value:.3f}", 10, color, bold=True, align=PP_ALIGN.RIGHT)


def add_clustered_chart(
    slide: object,
    x: float,
    y: float,
    w: float,
    h: float,
    categories: list[str],
    series: list[tuple[str, list[float], RGBColor]],
    *,
    maximum: float,
) -> None:
    data = ChartData()
    data.categories = categories
    for name, values, _ in series:
        data.add_series(name, values)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.minimum_scale = 0.0
    chart.value_axis.maximum_scale = maximum
    chart.value_axis.major_unit = 0.1
    chart.value_axis.has_major_gridlines = True
    chart.category_axis.tick_labels.font.size = Pt(9)
    chart.value_axis.tick_labels.font.size = Pt(8)
    for item, (_, _, color) in zip(chart.series, series):
        item.format.fill.solid()
        item.format.fill.fore_color.rgb = color
        item.format.line.fill.background()


def add_gate_table(slide: object, x: float, y: float, w: float, h: float, rows: list[list[str]]) -> None:
    table_shape = slide.shapes.add_table(len(rows) + 1, 4, Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    widths = [2.5, 1.6, 1.7, 0.8]
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(w * width / sum(widths))
    for col, label in enumerate(("Gate", "Target", "Observed", "Result")):
        cell = table.cell(0, col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = label
        base.format_cell(cell, 10, WHITE, bold=True, align=PP_ALIGN.CENTER)
    for row_index, row in enumerate(rows, start=1):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_index % 2 else PALE
            cell.text = value
            color = GREEN if value == "PASS" else RED if value == "FAIL" else NAVY
            base.format_cell(cell, 9.5, color, bold=value in {"PASS", "FAIL"}, align=PP_ALIGN.CENTER)


def run_by_name(evidence: ProjectEvidence, name: str) -> CandidateRun:
    return next(run for run in evidence.candidate_runs if run.name == name)


def fold_by_dataset(run: CandidateRun, dataset_id: str) -> dict[str, Any]:
    return next(fold for fold in run.folds if fold["dataset_id"] == dataset_id)


def write_outline(evidence: ProjectEvidence) -> None:
    titles = [
        "Automatic industrial defect pseudo-mask pipeline",
        "Executive conclusion",
        "Why automatic pseudo-masks matter",
        "Research questions and claim boundaries",
        "End-to-end research pipeline",
        "Automatic mask generation core",
        "One defect, multiple mask roles",
        "Evaluation firewall and scientific integrity",
        "Evidence base",
        "Confirmed selector contribution",
        "V4.2f reproduced headline result",
        "V4.2f result by source",
        "Candidate-quality sprint evolution",
        "V4.2f preregistered gate audit",
        "Visual evidence: MVTec and wood",
        "Locked VisA transfer failure",
        "Synthetic generation result",
        "Independent downstream utility",
        "Contributions and limitations",
        "Recommended paper positioning",
        "Final checkpoint and reproducibility",
    ]
    run = run_by_name(evidence, "V4.2f")
    lines = [
        "# ImgGen v2 Final Pipeline Result Presentation",
        "",
        f"PPTX: `{PPTX_PATH}`",
        f"Evidence snapshot: `{DATA_PATH}`",
        "",
        "## Slide Outline",
        "",
        *(f"{index}. {title}" for index, title in enumerate(titles, start=1)),
        "",
        "## Reproduced V4.2f Result",
        "",
        f"- Rerun status: `{evidence.rerun_status}`",
        f"- Cache hit: `{evidence.rerun_cache_hit}`",
        f"- Elapsed seconds: `{evidence.rerun_elapsed_seconds:.3f}`",
        f"- Macro selected Dice: `{run.selected_dice:.6f}`",
        f"- Macro oracle Dice: `{run.oracle_dice:.6f}`",
        f"- Candidate count: `{run.candidate_count}`",
        "",
        "## Final Decision",
        "",
        "The architecture is frozen. The deck separates the confirmed selector contribution from failed generalization and downstream synthetic-utility gates.",
    ]
    OUTLINE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_presentation(path: Path, *, expected_slides: int) -> None:
    prs = Presentation(path)
    if len(prs.slides) != expected_slides:
        raise RuntimeError(f"Expected {expected_slides} slides, found {len(prs.slides)}")
    for slide_index, slide in enumerate(prs.slides, start=1):
        if not slide.shapes:
            raise RuntimeError(f"Slide {slide_index} is empty")
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0:
                raise RuntimeError(f"Slide {slide_index} has a shape outside the top/left boundary")
            if shape.left + shape.width > prs.slide_width + Inches(0.05):
                raise RuntimeError(f"Slide {slide_index} has a shape beyond the right boundary")
            if shape.top + shape.height > prs.slide_height + Inches(0.05):
                raise RuntimeError(f"Slide {slide_index} has a shape beyond the bottom boundary")


if __name__ == "__main__":
    main()
