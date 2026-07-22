from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


DEFECTS = {
    "bottle": ("broken_large", "broken_small", "contamination"),
    "zipper": ("broken_teeth", "fabric_border", "split_teeth"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated MVTec bottle/zipper auto-mask pilot.")
    parser.add_argument("--source", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output", type=Path, default=Path("data/mvtec_bottle_zipper_auto_mask"))
    parser.add_argument("--samples-per-defect", type=int, default=4)
    parser.add_argument("--normal-count", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; pass --overwrite")

    rng = random.Random(args.seed)
    records = []
    for category, defect_types in DEFECTS.items():
        normals = sorted((source / category / "train" / "good").glob("*.png"))
        if len(normals) < args.normal_count:
            raise FileNotFoundError(f"Need {args.normal_count} normals for {category}, found {len(normals)}")
        selected_normals = rng.sample(normals, args.normal_count)
        for index, path in enumerate(selected_normals):
            target = output / category / "train" / "good" / f"{index:03d}.png"
            resize_image(path, target, args.image_size, mask=False)
            records.append({"category": category, "role": "normal", "source": str(path), "target": str(target)})

        for defect_type in defect_types:
            images = sorted((source / category / "test" / defect_type).glob("*.png"))
            selected = evenly_spaced(images, args.samples_per_defect)
            for index, image_path in enumerate(selected):
                official_mask = source / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                if not official_mask.exists():
                    raise FileNotFoundError(official_mask)
                target_image = output / category / "test" / defect_type / f"{index:03d}.png"
                reference_mask = output / "official_reference_masks" / category / defect_type / f"{index:03d}_mask.png"
                resize_image(image_path, target_image, args.image_size, mask=False)
                resize_image(official_mask, reference_mask, args.image_size, mask=True)
                records.append(
                    {
                        "category": category,
                        "defect_type": defect_type,
                        "role": "defect",
                        "source_image": str(image_path),
                        "source_mask": str(official_mask),
                        "target_image": str(target_image),
                        "official_reference_mask": str(reference_mask),
                    }
                )

    manifest = {
        "source": str(source),
        "output": str(output),
        "dataset": "MVTec AD",
        "license": "CC BY-NC-SA 4.0",
        "official_page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "seed": args.seed,
        "samples_per_defect": args.samples_per_defect,
        "normal_count": args.normal_count,
        "image_size": args.image_size,
        "annotation_policy": (
            "Official pixel masks are copied to official_reference_masks for evaluation only. "
            "The auto-mask command writes pseudo-labels to category/ground_truth and cannot read "
            "the isolated official reference directory."
        ),
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"Need {count} images, found {len(paths)}")
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)]


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def resize_image(source: Path, target: Path, image_size: int, *, mask: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("L" if mask else "RGB")
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    image.resize((image_size, image_size), resampling).save(target)


if __name__ == "__main__":
    main()
