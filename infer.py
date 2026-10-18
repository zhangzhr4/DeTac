"""Predict shading and marker images using the DeTac DDPM model."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm.auto import tqdm

from detac.inference import infer_detac, load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InputSample:
    sample_id: str
    marker_path: Path
    nomarker_path: Path | None = None
    mask_path: Path | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Mixed tactile image or directory (searched recursively).")
    source.add_argument("--split", choices=("train", "val", "test", "all"), help="Prepared dataset split.")
    source.add_argument("--manifest", type=Path, help="Custom JSONL manifest of mixed images.")
    parser.add_argument("--data-root", type=Path, default=Path("data/detac"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=1000, help="DDPM sampling steps (paper: T=1000).")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="0 processes all inputs.")
    return parser


def resolve_inputs(args: argparse.Namespace) -> list[InputSample]:
    if args.input is not None:
        if args.input.is_file():
            return [InputSample(args.input.stem, args.input)]
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        paths = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        return [InputSample("__".join(p.relative_to(args.input).with_suffix("").parts), p) for p in paths]
    manifest = args.manifest or args.data_root / "manifests" / f"{args.split}.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [InputSample(
        row["sample_id"], args.data_root / row["marker_path"],
        args.data_root / row["nomarker_path"] if row.get("nomarker_path") else None,
        args.data_root / row["mask_path"] if row.get("mask_path") else None,
    ) for row in rows]


def compute_gray_psnr(pred_path: Path, gt_path: Path) -> float:
    """Match calculate_metrics.py: saved uint8 images, PIL grayscale, range 255."""
    with Image.open(pred_path) as pred_image, Image.open(gt_path) as gt_image:
        pred = np.asarray(pred_image.convert("L"), dtype=np.float64)
        gt = np.asarray(gt_image.convert("L"), dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Image size mismatch: {pred_path} and {gt_path}")
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def compute_mask_iou(pred_path: Path, gt_path: Path) -> float:
    """Compare the saved binary prediction with the thresholded GT mask."""
    with Image.open(pred_path) as pred_image, Image.open(gt_path) as gt_image:
        pred = np.asarray(pred_image.convert("L")) >= 128
        gt = np.asarray(gt_image.convert("L")) >= 128
    if pred.shape != gt.shape:
        raise ValueError(f"Mask size mismatch: {pred_path} and {gt_path}")
    union = int(np.count_nonzero(pred | gt))
    return float(np.count_nonzero(pred & gt) / union) if union else 1.0


def main() -> None:
    args = build_argparser().parse_args()
    entries = resolve_inputs(args)
    if args.max_images > 0:
        entries = entries[:args.max_images]
    if not entries:
        raise ValueError("No input images found.")
    ids = [entry.sample_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Input names collide; use unique stems or unique manifest sample_id values.")
    if any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in ids):
        raise ValueError("sample_id must be a nonempty filename without directory separators.")
    for entry in entries:
        for path in (entry.marker_path, entry.nomarker_path, entry.mask_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    expected_size = (int(config.get("image_width", 640)), int(config.get("image_height", 480)))
    for entry in entries:
        for path in (entry.marker_path, entry.nomarker_path, entry.mask_path):
            if path is not None:
                with Image.open(path) as source:
                    if source.size != expected_size:
                        raise ValueError(f"{path}: expected width/height {expected_size}, got {source.size}.")
    scheduler = DDPMScheduler(num_train_timesteps=int(config.get("num_train_timesteps", 1000)))
    folders = {name: args.output_dir / name for name in ("shading", "markers", "binary_masks", "panels")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    records = []
    metric_values: dict[str, list[float]] = {"gray_psnr": [], "mask_iou": []}
    for entry in tqdm(entries, desc="Predicting", unit="image"):
        sample_id, path = entry.sample_id, entry.marker_path
        with Image.open(path) as source:
            mixed = source.convert("RGB")
        result = infer_detac(model, scheduler, mixed, device, args.num_inference_steps, args.mask_threshold)
        outputs = {
            "shading": result["markerless_image"],
            "markers": result["mask_probability"],
            "binary_masks": result["mask_binary"],
        }
        outputs["panels"] = torch.cat([
            TF.to_tensor(mixed), outputs["shading"], outputs["markers"].repeat(3, 1, 1),
        ], dim=2)
        record = {"sample_id": sample_id, "input": str(path)}
        for name, tensor in outputs.items():
            target = folders[name] / f"{sample_id}.png"
            save_image(tensor, target)
            record[name] = str(target)
        metrics = {}
        if entry.nomarker_path is not None:
            record["ground_truth_shading"] = str(entry.nomarker_path)
            metrics["gray_psnr"] = compute_gray_psnr(Path(record["shading"]), entry.nomarker_path)
        if entry.mask_path is not None:
            record["ground_truth_mask"] = str(entry.mask_path)
            metrics["mask_iou"] = compute_mask_iou(Path(record["binary_masks"]), entry.mask_path)
        if metrics:
            record["metrics"] = metrics
            for name, value in metrics.items():
                metric_values[name].append(value)
        records.append(record)
    summary = {
        "checkpoint": str(args.checkpoint), "num_images": len(records),
        "num_inference_steps": args.num_inference_steps, "seed": args.seed,
        "mask_threshold": args.mask_threshold, "records": records,
        "metrics": {name: float(np.mean(values)) for name, values in metric_values.items() if values},
        "metric_counts": {name: len(values) for name, values in metric_values.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if metric_values["gray_psnr"]:
        print(f"Grayscale PSNR: {summary['metrics']['gray_psnr']:.4f} dB ({len(metric_values['gray_psnr'])} images)")
    if metric_values["mask_iou"]:
        print(f"Mask IoU: {summary['metrics']['mask_iou']:.4f} ({len(metric_values['mask_iou'])} images)")
    if not summary["metrics"]:
        print("Predictions saved. No ground truth provided; metrics were skipped.")


if __name__ == "__main__":
    main()
