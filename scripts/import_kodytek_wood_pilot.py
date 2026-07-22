from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


DATASET = "iluvvatar/wood_surface_defects"
CONFIG = "default"
SPLIT = "train"
REVISION = "4b12277dc906869ec6e0552ac9d3975df9a7aa12"
LICENSE = "CC BY 4.0"
SOURCE_DOI = "10.5281/zenodo.4694695"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TARGET_LABELS = ("Crack", "resin", "knot_with_crack")
TARGET_NAMES = {
    "Crack": "crack",
    "resin": "resin",
    "knot_with_crack": "knot_with_crack",
}
SCAN_OFFSETS = (
    1600,
    2700,
    4200,
    5200,
    7800,
    8500,
    8900,
    9200,
    9300,
    9800,
    10000,
    10200,
    10300,
    10800,
    12900,
    13600,
    14600,
    14900,
    15800,
    16200,
    16800,
    17100,
    17500,
    17800,
    18100,
    18700,
    19100,
    19800,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact real-world Kodytek wood-defect pilot.")
    parser.add_argument("--output", type=Path, default=Path("data/kodytek_wood_pilot"))
    parser.add_argument("--clean-count", type=int, default=18)
    parser.add_argument("--per-defect", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already exists and is not empty; pass --overwrite")

    rows = discover_rows(args.clean_count, args.per_defect, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    records = materialize(rows, output, args.crop_size, args.image_size, args.seed)
    write_provenance(output, records, args)
    write_contact_sheet(output, records)
    print(output)
    print(f"records={len(records)} clean={sum(r['split_role'] == 'clean' for r in records)}")


def discover_rows(clean_count: int, per_defect: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    clean: list[dict[str, Any]] = []
    defects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    offsets = list(SCAN_OFFSETS)
    rng.shuffle(offsets)

    with requests.Session() as session:
        for offset in offsets:
            response = session.get(
                ROWS_API,
                params={
                    "dataset": DATASET,
                    "config": CONFIG,
                    "split": SPLIT,
                    "offset": offset,
                    "length": 100,
                },
                timeout=90,
            )
            response.raise_for_status()
            candidates = response.json()["rows"]
            rng.shuffle(candidates)
            for item in candidates:
                row = item["row"]
                objects = row.get("objects") or []
                if not objects and len(clean) < clean_count:
                    clean.append(item)
                    continue
                if len(objects) != 1:
                    continue
                label = str(objects[0].get("label", ""))
                if label not in TARGET_LABELS or len(defects[label]) >= per_defect:
                    continue
                if not useful_box(label, objects[0].get("bb", [])):
                    continue
                defects[label].append(item)

            if len(clean) >= clean_count and all(len(defects[label]) >= per_defect for label in TARGET_LABELS):
                break

    missing = {
        "clean": max(0, clean_count - len(clean)),
        **{label: max(0, per_defect - len(defects[label])) for label in TARGET_LABELS},
    }
    if any(missing.values()):
        raise RuntimeError(f"Could not assemble requested pilot: {missing}")

    selected = clean[:clean_count]
    for label in TARGET_LABELS:
        selected.extend(defects[label][:per_defect])
    return selected


def useful_box(label: str, box: list[float]) -> bool:
    if len(box) != 4:
        return False
    _, _, width, height = (float(value) for value in box)
    area = width * height
    if label == "Crack":
        return area >= 0.003 and max(width, height) >= 0.18
    if label == "resin":
        return area >= 0.0015 and max(width, height) >= 0.06
    return area >= 0.006 and max(width, height) >= 0.10


def materialize(
    selected: list[dict[str, Any]],
    output: Path,
    crop_size: int,
    image_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)
    with requests.Session() as session:
        for item in selected:
            row = item["row"]
            image_meta = row["image"]
            response = session.get(image_meta["src"], timeout=120)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            objects = row.get("objects") or []
            if objects:
                label = str(objects[0]["label"])
                role = TARGET_NAMES[label]
                box = [float(value) for value in objects[0]["bb"]]
                crop_box = defect_crop(image.size, box, crop_size)
            else:
                label = "good"
                role = "good"
                box = None
                crop_box = clean_crop(image.size, crop_size, int(row["id"]), seed)

            crop = image.crop(crop_box).resize((image_size, image_size), Image.Resampling.LANCZOS)
            index = counters[role]
            counters[role] += 1
            stem = f"{index:03d}"
            if role == "good":
                image_path = output / "kodytek_wood" / "train" / "good" / f"{stem}.png"
                split_role = "clean"
                bbox_path = None
                crop_bbox = None
            else:
                image_path = output / "kodytek_wood" / "test" / role / f"{stem}.png"
                bbox_path = output / "kodytek_wood" / "reference_bbox_masks" / role / f"{stem}_bbox.png"
                crop_bbox = transformed_bbox(image.size, box, crop_box, image_size)
                save_bbox_mask(bbox_path, crop_bbox, image_size)
                split_role = "defect"

            image_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(image_path)
            records.append(
                {
                    "row_idx": int(item["row_idx"]),
                    "source_id": int(row["id"]),
                    "source_image_width": int(image_meta["width"]),
                    "source_image_height": int(image_meta["height"]),
                    "source_sha256": hashlib.sha256(response.content).hexdigest(),
                    "source_objects": objects,
                    "source_label": label,
                    "split_role": split_role,
                    "pipeline_defect_type": None if role == "good" else role,
                    "crop_box_xyxy": list(crop_box),
                    "crop_bbox_xyxy": crop_bbox,
                    "image_path": str(image_path),
                    "reference_bbox_mask_path": str(bbox_path) if bbox_path else None,
                    "annotation_use": (
                        "audit_reference_only_not_auto_mask_input" if bbox_path else "clean_sample_selection"
                    ),
                }
            )
            print(f"downloaded row={item['row_idx']} role={role} -> {image_path}", flush=True)
    return records


def defect_crop(image_size: tuple[int, int], box: list[float], crop_size: int) -> tuple[int, int, int, int]:
    width, height = image_size
    center_x, center_y, _, _ = box
    cx = int(round(center_x * width))
    cy = int(round(center_y * height))
    size = min(crop_size, width, height)
    left = max(0, min(width - size, cx - size // 2))
    top = max(0, min(height - size, cy - size // 2))
    return left, top, left + size, top + size


def clean_crop(image_size: tuple[int, int], crop_size: int, source_id: int, seed: int) -> tuple[int, int, int, int]:
    width, height = image_size
    size = min(crop_size, width, height)
    rng = random.Random(seed + source_id)
    left = rng.randint(0, max(0, width - size))
    top = rng.randint(0, max(0, height - size))
    return left, top, left + size, top + size


def transformed_bbox(
    image_size: tuple[int, int],
    box: list[float],
    crop_box: tuple[int, int, int, int],
    output_size: int,
) -> list[int]:
    width, height = image_size
    cx, cy, bw, bh = box
    x0 = (cx - bw / 2) * width
    y0 = (cy - bh / 2) * height
    x1 = (cx + bw / 2) * width
    y1 = (cy + bh / 2) * height
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    scale_x = output_size / (crop_right - crop_left)
    scale_y = output_size / (crop_bottom - crop_top)
    transformed = [
        round((x0 - crop_left) * scale_x),
        round((y0 - crop_top) * scale_y),
        round((x1 - crop_left) * scale_x),
        round((y1 - crop_top) * scale_y),
    ]
    transformed[0] = max(0, min(output_size - 1, transformed[0]))
    transformed[1] = max(0, min(output_size - 1, transformed[1]))
    transformed[2] = max(transformed[0] + 1, min(output_size, transformed[2]))
    transformed[3] = max(transformed[1] + 1, min(output_size, transformed[3]))
    return transformed


def save_bbox_mask(path: Path, bbox: list[int], image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (image_size, image_size), 0)
    ImageDraw.Draw(mask).rectangle(tuple(bbox), fill=255)
    mask.save(path)


def write_provenance(output: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    provenance = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "huggingface_dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "original_zenodo_url": "https://zenodo.org/records/4694695",
        "original_doi": SOURCE_DOI,
        "license": LICENSE,
        "citation": (
            "Kodytek, P., Bodzas, A., & Bilik, P. A large-scale image dataset of wood surface "
            "defects for automated vision-based quality control processes. F1000Research 10:581."
        ),
        "selection": {
            "seed": args.seed,
            "clean_count": args.clean_count,
            "per_defect": args.per_defect,
            "target_labels": list(TARGET_LABELS),
            "scan_offsets": list(SCAN_OFFSETS),
            "crop_size": args.crop_size,
            "output_image_size": args.image_size,
        },
        "annotation_policy": (
            "Published boxes were used to select and center reproducible pilot crops. They are saved "
            "under reference_bbox_masks for audit only and are not placed in ground_truth or supplied "
            "to the automatic pseudo-mask generator."
        ),
        "records": records,
    }
    (output / "source_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "LICENSE_SOURCE.txt").write_text(
        "Source dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "Original record: https://zenodo.org/records/4694695\n"
        "Hugging Face conversion: https://huggingface.co/datasets/iluvvatar/wood_surface_defects\n",
        encoding="utf-8",
    )


def write_contact_sheet(output: Path, records: list[dict[str, Any]]) -> None:
    defects = [record for record in records if record["split_role"] == "defect"]
    clean = [record for record in records if record["split_role"] == "clean"][:4]
    shown = clean + defects
    thumb = 220
    label_height = 44
    columns = 4
    rows = (len(shown) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(shown):
        x = (index % columns) * thumb
        y = (index // columns) * (thumb + label_height)
        image = Image.open(record["image_path"]).convert("RGB").resize((thumb, thumb))
        sheet.paste(image, (x, y))
        if record["crop_bbox_xyxy"]:
            box = [round(value * thumb / 512) for value in record["crop_bbox_xyxy"]]
            draw.rectangle((x + box[0], y + box[1], x + box[2], y + box[3]), outline=(255, 190, 0), width=2)
        label = f"{record['pipeline_defect_type'] or 'good'} | row {record['row_idx']}"
        draw.text((x + 5, y + thumb + 8), label, fill=(20, 30, 45), font=font)
    sheet.save(output / "import_contact_sheet.png")


if __name__ == "__main__":
    main()
