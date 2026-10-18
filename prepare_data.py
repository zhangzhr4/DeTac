from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_data(
    data_root: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Generate splits for the released marker/nomarker/mask image layout."""
    data_root = Path(data_root)
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1.")
    manifest_dir = data_root / "manifests"
    if manifest_dir.exists() and any(manifest_dir.iterdir()):
        raise ValueError("manifests must be absent or empty; existing splits will not be overwritten.")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images: dict[str, dict[str, Path]] = {}
    for name in ("marker", "nomarker", "mask"):
        folder = data_root / name
        if not folder.is_dir():
            raise ValueError(f"Missing image directory: {folder}")
        images[name] = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                if path.stem in images[name]:
                    raise ValueError(f"Duplicate sample ID in {folder}: {path.stem}")
                images[name][path.stem] = path
    sample_ids = set(images["marker"])
    if not sample_ids or any(set(paths) != sample_ids for paths in images.values()):
        raise ValueError("marker/, nomarker/, and mask/ must contain matching, nonempty sample IDs.")

    groups: dict[str, list[dict]] = {}
    for sample_id in sorted(sample_ids):
        parts = sample_id.rsplit("__", 1)
        if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
            raise ValueError(f"Expected filenames like Obj_01__0000.jpg, got sample ID: {sample_id}")
        object_id = parts[0]
        row = {"sample_id": sample_id, "object_id": object_id}
        row.update({
            f"{name}_path": str(images[name][sample_id].relative_to(data_root))
            for name in ("marker", "nomarker", "mask")
        })
        groups.setdefault(object_id, []).append(row)
    objects = sorted(groups)
    if len(objects) < 2:
        raise ValueError("At least two objects are required for training and validation splits.")

    random_generator = random.Random(seed)
    shuffled_objects = objects[:]
    random_generator.shuffle(shuffled_objects)
    val_count = min(len(objects) - 1, max(1, int(round(len(objects) * val_ratio))))
    val_objects = set(shuffled_objects[:val_count])
    splits: dict[str, list[dict]] = {name: [] for name in ("all", "train", "val", "test")}
    for object_id in objects:
        rows = sorted(groups[object_id], key=lambda row: int(row["sample_id"].rsplit("__", 1)[1]))
        test_id = random_generator.choice(rows)["sample_id"]
        for row in rows:
            split = "test" if row["sample_id"] == test_id else "val" if object_id in val_objects else "train"
            splits[split].append(row)
            splits["all"].append(row)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training and validation must both contain samples; provide multiple paired frames per object.")
    for name, rows in splits.items():
        _write_jsonl(manifest_dir / f"{name}.jsonl", rows)
    return {
        "objects": len(objects), "samples": len(splits["all"]),
        **{f"{name}_samples": len(splits[name]) for name in ("train", "val", "test")},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DeTac train, validation, and test manifests.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/detac",
        help="Extracted dataset containing marker/, nomarker/, and mask/; generate manifests in place.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio by object after reserving one test sample per object.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    stats = prepare_data(args.data_root, val_ratio=args.val_ratio, seed=args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
