from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from detac.data import AVAILABLE_AUGMENTATION_PROFILES, DeTacDataset
from detac.model import AVAILABLE_ARCHITECTURES, BaseDeTacModel, build_detac_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeTac multi-task diffusion model.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/detac",
        help="Prepared DeTac dataset root.",
    )
    parser.add_argument("--train-manifest", type=str, default="", help="Optional override for train manifest.")
    parser.add_argument("--val-manifest", type=str, default="", help="Optional override for validation manifest.")
    parser.add_argument("--image-height", type=int, default=480, help="Full input image height.")
    parser.add_argument("--image-width", type=int, default=640, help="Full input image width.")
    parser.add_argument(
        "--use-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable online synchronized augmentation for the training split only (disabled by default).",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="moderate",
        choices=[profile for profile in AVAILABLE_AUGMENTATION_PROFILES if profile != "none"],
        help="Augmentation strength profile used when --use-augmentation is enabled.",
    )
    parser.add_argument(
        "--augmentation-apply-probability",
        type=float,
        default=0.5,
        help="Probability of applying the augmentation pipeline to a training sample when enabled.",
    )
    parser.add_argument(
        "--marker-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include marker-aware branches when --use-augmentation is enabled.",
    )
    parser.add_argument(
        "--marker-aug-size-only",
        action="store_true",
        help="When --marker-aug is enabled, keep only marker-size augmentation for augmented samples and disable color-related augmentation branches.",
    )
    parser.add_argument(
        "--marker-aug-expand-all-circles",
        action="store_true",
        help="When marker augmentation is enabled, expand every marker component instead of leaving some components unchanged.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="pixel_concat_unet",
        choices=AVAILABLE_ARCHITECTURES,
        help="Paper architecture identifier, also stored in legacy checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Default learning rate.")
    parser.add_argument(
        "--diffusion-lr",
        type=float,
        default=1e-5,
        help="Shading restoration branch learning rate (paper: 1e-5).",
    )
    parser.add_argument(
        "--mask-lr",
        type=float,
        default=1e-4,
        help="Marker extraction branch learning rate (paper: 1e-4).",
    )
    parser.add_argument("--lambda-mask", type=float, default=0.5, help="Marker extraction loss weight (lambda_mar).")
    parser.add_argument(
        "--use-marker-ring-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add marker-neighborhood weighted reconstruction loss on a ring region around the marker mask.",
    )
    parser.add_argument(
        "--lambda-marker-ring",
        type=float,
        default=0.05,
        help="Weight for marker ring reconstruction loss when enabled.",
    )
    parser.add_argument(
        "--marker-ring-radius",
        type=int,
        default=2,
        help="Dilate radius in pixels used to build the marker ring region.",
    )
    parser.add_argument(
        "--marker-ring-start-epoch",
        type=int,
        default=101,
        help="Epoch number from which marker ring loss becomes active.",
    )
    parser.add_argument(
        "--use-change-mask-consistency-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use predicted diffusion change map to supervise the mask branch after marker-ring-start-epoch.",
    )
    parser.add_argument(
        "--lambda-change-mask-consistency",
        type=float,
        default=0.05,
        help="Weight for diffusion-to-mask change-map consistency loss.",
    )
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="DDPM training timesteps.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save a numbered checkpoint every N epochs.")
    parser.add_argument(
        "--amp",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16"],
        help="Mixed precision mode for training.",
    )
    parser.add_argument("--vis-samples", type=int, default=2, help="How many train/val samples to visualize per epoch.")
    parser.add_argument(
        "--vis-inference-steps",
        type=int,
        default=20,
        help="Number of reverse diffusion steps used for epoch visualizations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume automatically from the latest checkpoint under --output-dir.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint file or checkpoint directory to resume training from.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/detac", help="Checkpoint directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (not a guarantee of bitwise reproducibility).")
    return parser


def _resolve_manifests(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_root = Path(args.data_root)
    train_manifest = Path(args.train_manifest) if args.train_manifest else data_root / "manifests" / "train.jsonl"
    val_manifest = Path(args.val_manifest) if args.val_manifest else data_root / "manifests" / "val.jsonl"
    return data_root, train_manifest, val_manifest


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    data_root, train_manifest, val_manifest = _resolve_manifests(args)
    image_size = (args.image_height, args.image_width)

    train_augmentation_profile = args.augmentation_profile if args.use_augmentation else "none"

    train_dataset = DeTacDataset(
        manifest_path=train_manifest,
        data_root=data_root,
        expected_image_size=image_size,
        augmentation_profile=train_augmentation_profile,
        augmentation_apply_probability=args.augmentation_apply_probability if args.use_augmentation else 1.0,
        use_marker_augmentation=args.marker_aug,
        marker_aug_size_only=args.marker_aug_size_only,
        marker_aug_expand_all_circles=args.marker_aug_expand_all_circles,
    )
    val_dataset = DeTacDataset(
        manifest_path=val_manifest,
        data_root=data_root,
        expected_image_size=image_size,
        augmentation_profile="none",
        augmentation_apply_probability=1.0,
        use_marker_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def compute_losses(
    model: BaseDeTacModel,
    noise_scheduler: DDPMScheduler,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    current_epoch: int,
    mask_loss_fn: nn.Module,
    lambda_mask: float,
    use_marker_ring_loss: bool,
    lambda_marker_ring: float,
    marker_ring_radius: int,
    marker_ring_start_epoch: int,
    use_change_mask_consistency_loss: bool,
    lambda_change_mask_consistency: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    marker_img = batch["marker_img"].to(device, non_blocking=True)
    nomarker_img = batch["nomarker_img"].to(device, non_blocking=True)
    mask_gt = batch["mask_gt"].to(device, non_blocking=True)

    prepared_targets = model.prepare_training_targets(nomarker_img)
    diffusion_target = prepared_targets["diffusion_target"]

    true_noise = torch.randn_like(diffusion_target)
    timesteps = torch.randint(
        low=0,
        high=noise_scheduler.config.num_train_timesteps,
        size=(diffusion_target.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_img = noise_scheduler.add_noise(diffusion_target, true_noise, timesteps)

    outputs = model(noisy_img=noisy_img, timesteps=timesteps, marker_img=marker_img)
    loss_diff = F.mse_loss(outputs["noise_pred"], true_noise)
    loss_mask = mask_loss_fn(outputs["mask_logits"], mask_gt)

    marker_ring_active = use_marker_ring_loss and current_epoch >= marker_ring_start_epoch
    change_mask_consistency_active = use_change_mask_consistency_loss and current_epoch >= marker_ring_start_epoch
    need_image_space_aux = marker_ring_active or change_mask_consistency_active
    if need_image_space_aux:
        alpha_t = noise_scheduler.alphas_cumprod[timesteps].to(device=device, dtype=diffusion_target.dtype)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        pred_x0 = (noisy_img - torch.sqrt(1.0 - alpha_t) * outputs["noise_pred"]) / torch.sqrt(alpha_t)
        pred_decoded = model.decode_denoised_sample(pred_x0)
    else:
        pred_decoded = None

    if marker_ring_active:
        ring_radius = max(marker_ring_radius, 1)
        ring_kernel_size = 2 * ring_radius + 1
        pred_mask_for_ring = (torch.sigmoid(outputs["mask_logits"]).detach() > 0.5).to(mask_gt.dtype)
        dilated_ring_mask = F.max_pool2d(
            pred_mask_for_ring,
            kernel_size=ring_kernel_size,
            stride=1,
            padding=ring_radius,
        )
        ring_mask = (dilated_ring_mask - pred_mask_for_ring).clamp(min=0.0, max=1.0)
        ring_denominator = ring_mask.sum() + 1e-6
        loss_marker_ring = (
            torch.abs(pred_decoded - nomarker_img) * ring_mask
        ).sum() / ring_denominator
    else:
        loss_marker_ring = torch.zeros((), device=device)

    if change_mask_consistency_active:
        pred_change_map = torch.mean(torch.abs(pred_decoded.detach() - marker_img), dim=1, keepdim=True)
        # Eq. (9): sharpen the channel-mean change with gamma = 2.
        pred_change_map = pred_change_map.square()
        change_max = pred_change_map.amax(dim=(2, 3), keepdim=True)
        normalized_change_map = pred_change_map / (change_max + 1e-6)
        pred_mask_probability = torch.sigmoid(outputs["mask_logits"])
        loss_change_mask_consistency = F.l1_loss(pred_mask_probability, normalized_change_map)
    else:
        loss_change_mask_consistency = torch.zeros((), device=device)

    total_loss = (
        loss_diff
        + lambda_mask * loss_mask
        + lambda_marker_ring * loss_marker_ring
        + lambda_change_mask_consistency * loss_change_mask_consistency
    )

    stats = {
        "loss": float(total_loss.detach().item()),
        "loss_diff": float(loss_diff.detach().item()),
        "loss_mask": float(loss_mask.detach().item()),
        "loss_marker_ring": float(loss_marker_ring.detach().item()),
        "loss_change_mask_consistency": float(loss_change_mask_consistency.detach().item()),
    }
    return total_loss, stats


def _get_autocast_dtype(amp_mode: str) -> torch.dtype | None:
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        return torch.bfloat16
    return None


def build_optimizer(model: BaseDeTacModel, args: argparse.Namespace) -> AdamW:
    diffusion_lr = args.diffusion_lr if args.diffusion_lr is not None else args.lr
    mask_lr = args.mask_lr if args.mask_lr is not None else args.lr

    mask_params = []
    diffusion_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("mask_head."):
            mask_params.append(param)
        else:
            diffusion_params.append(param)

    param_groups = []
    if diffusion_params:
        param_groups.append({"params": diffusion_params, "lr": diffusion_lr})
    if mask_params:
        param_groups.append({"params": mask_params, "lr": mask_lr})
    return AdamW(param_groups, lr=args.lr)


def _find_latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists():
        return latest_path

    epoch_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not epoch_checkpoints:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")

    def _extract_epoch(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(epoch_checkpoints, key=_extract_epoch)


def _load_existing_history(history_path: Path) -> list[dict[str, object]]:
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as fp:
        history = json.load(fp)

    if isinstance(history, list):
        return history
    return []


def _save_history(history_path: Path, history: list[dict[str, object]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = history_path.with_suffix(history_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2, ensure_ascii=True)
    temp_path.replace(history_path)


def _get_entry_val_loss(history_entry: dict[str, object]) -> float:
    val_metrics = history_entry.get("val", {})
    if not isinstance(val_metrics, dict):
        return float("inf")

    loss = val_metrics.get("loss", float("inf"))
    try:
        return float(loss)
    except (TypeError, ValueError):
        return float("inf")


def _initialize_best_tracking(
    history: list[dict[str, object]],
    marker_ring_start_epoch: int,
    track_post_aux_best: bool,
) -> dict[str, float | int | None]:
    best_post_aux = {"loss": float("inf"), "epoch": None}

    for history_entry in history:
        try:
            epoch = int(history_entry.get("epoch", -1))
        except (TypeError, ValueError):
            continue

        loss = _get_entry_val_loss(history_entry)
        if track_post_aux_best and epoch >= marker_ring_start_epoch and loss < best_post_aux["loss"]:
            best_post_aux = {"loss": loss, "epoch": epoch}

    return best_post_aux


def maybe_resume_training(
    model: BaseDeTacModel,
    optimizer: AdamW,
    output_dir: Path,
    resume: bool,
    resume_from: str,
    device: torch.device,
) -> tuple[int, list[dict[str, object]]]:
    if not resume and not resume_from:
        return 1, []

    if resume_from:
        resume_path = Path(resume_from)
        checkpoint_path = _find_latest_checkpoint_path(resume_path) if resume_path.is_dir() else resume_path
    else:
        checkpoint_path = _find_latest_checkpoint_path(output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    history = _load_existing_history(checkpoint_path.parent / "history.json")
    print(
        json.dumps(
            {
                "resume_from": str(checkpoint_path),
                "resumed_epoch": int(checkpoint.get("epoch", 0)),
                "next_epoch": start_epoch,
            },
            ensure_ascii=True,
        )
    )
    return start_epoch, history


def run_epoch(
    model: BaseDeTacModel,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    amp_mode: str,
    current_epoch: int,
    mask_loss_fn: nn.Module,
    lambda_mask: float,
    use_marker_ring_loss: bool,
    lambda_marker_ring: float,
    marker_ring_radius: int,
    marker_ring_start_epoch: int,
    use_change_mask_consistency_loss: bool,
    lambda_change_mask_consistency: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "loss_diff": 0.0,
        "loss_mask": 0.0,
        "loss_marker_ring": 0.0,
        "loss_change_mask_consistency": 0.0,
    }
    steps = 0
    autocast_dtype = _get_autocast_dtype(amp_mode)

    context_manager = torch.enable_grad if is_train else torch.no_grad
    with context_manager():
        for batch in loader:
            autocast_enabled = device.type == "cuda" and autocast_dtype is not None
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                loss, stats = compute_losses(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    batch=batch,
                    device=device,
                    current_epoch=current_epoch,
                    mask_loss_fn=mask_loss_fn,
                    lambda_mask=lambda_mask,
                    use_marker_ring_loss=use_marker_ring_loss,
                    lambda_marker_ring=lambda_marker_ring,
                    marker_ring_radius=marker_ring_radius,
                    marker_ring_start_epoch=marker_ring_start_epoch,
                    use_change_mask_consistency_loss=use_change_mask_consistency_loss,
                    lambda_change_mask_consistency=lambda_change_mask_consistency,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            for key in totals:
                totals[key] += stats[key]
            steps += 1

    return {key: value / max(steps, 1) for key, value in totals.items()}


def _build_checkpoint_payload(
    epoch: int,
    model: BaseDeTacModel,
    optimizer: AdamW,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: BaseDeTacModel,
    optimizer: AdamW,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"

    checkpoint = _build_checkpoint_payload(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        args=args,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    torch.save(checkpoint, latest_path)
    if epoch % args.checkpoint_interval == 0:
        torch.save(checkpoint, checkpoint_path)


def save_named_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: BaseDeTacModel,
    optimizer: AdamW,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    checkpoint = _build_checkpoint_payload(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        args=args,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    torch.save(checkpoint, checkpoint_path)


def _to_display_rgb(image: torch.Tensor) -> torch.Tensor:
    return image.detach().cpu().add(1.0).div(2.0).clamp(0.0, 1.0)


def _to_display_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().cpu().clamp(0.0, 1.0)
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    return mask


@torch.no_grad()
def generate_visual_predictions(
    model: BaseDeTacModel,
    noise_scheduler: DDPMScheduler,
    marker_img: torch.Tensor,
    device: torch.device,
    num_inference_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    marker_batch = marker_img.to(device, non_blocking=True)
    mask_probability = torch.sigmoid(model.predict_mask(marker_batch))

    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    generated_sample = model.make_initial_diffusion_sample(marker_batch)

    for timestep in noise_scheduler.timesteps:
        timestep_batch = torch.full(
            (generated_sample.shape[0],),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        noise_pred = model.predict_noise(generated_sample, timestep_batch, marker_batch)
        denoise_step = noise_scheduler.step(noise_pred, timestep, generated_sample)
        generated_sample = denoise_step.prev_sample

    reconstructed_image = model.decode_denoised_sample(generated_sample)
    return reconstructed_image.detach().cpu(), mask_probability.detach().cpu()


def save_epoch_visualizations(
    model: BaseDeTacModel,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    split: str,
    num_samples: int,
    num_inference_steps: int,
) -> None:
    if num_samples <= 0:
        return

    batch = next(iter(loader), None)
    if batch is None:
        return

    marker_img = batch["marker_img"][:num_samples]
    nomarker_img = batch["nomarker_img"][:num_samples]
    mask_gt = batch["mask_gt"][:num_samples]
    sample_ids = batch["sample_id"][:num_samples]

    was_training = model.training
    pred_markerless, pred_mask = generate_visual_predictions(
        model=model,
        noise_scheduler=noise_scheduler,
        marker_img=marker_img,
        device=device,
        num_inference_steps=num_inference_steps,
    )
    model.train(was_training)

    vis_dir = output_dir / "visualizations" / f"epoch_{epoch:03d}" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    for index, sample_id in enumerate(sample_ids):
        panel = torch.cat(
            [
                _to_display_rgb(marker_img[index]),
                _to_display_rgb(nomarker_img[index]),
                _to_display_rgb(pred_markerless[index]),
                _to_display_mask(mask_gt[index]),
                _to_display_mask(pred_mask[index]),
            ],
            dim=2,
        )
        save_image(panel, vis_dir / f"{sample_id}.png")


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    image_size = (args.image_height, args.image_width)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader = create_dataloaders(args)

    model = build_detac_model(
        architecture=args.architecture,
        image_size=image_size,
    ).to(device)
    optimizer = build_optimizer(model, args)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    mask_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, history = maybe_resume_training(
        model=model,
        optimizer=optimizer,
        output_dir=output_dir,
        resume=args.resume,
        resume_from=args.resume_from,
        device=device,
    )

    track_post_aux_best = args.use_marker_ring_loss or args.use_change_mask_consistency_loss
    best_post_aux_state = _initialize_best_tracking(
        history=history,
        marker_ring_start_epoch=args.marker_ring_start_epoch,
        track_post_aux_best=track_post_aux_best,
    )

    epoch_iterator = tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_iterator:
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_mode=args.amp,
            current_epoch=epoch,
            mask_loss_fn=mask_loss_fn,
            lambda_mask=args.lambda_mask,
            use_marker_ring_loss=args.use_marker_ring_loss,
            lambda_marker_ring=args.lambda_marker_ring,
            marker_ring_radius=args.marker_ring_radius,
            marker_ring_start_epoch=args.marker_ring_start_epoch,
            use_change_mask_consistency_loss=args.use_change_mask_consistency_loss,
            lambda_change_mask_consistency=args.lambda_change_mask_consistency,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            optimizer=None,
            scaler=None,
            amp_mode=args.amp,
            current_epoch=epoch,
            mask_loss_fn=mask_loss_fn,
            lambda_mask=args.lambda_mask,
            use_marker_ring_loss=args.use_marker_ring_loss,
            lambda_marker_ring=args.lambda_marker_ring,
            marker_ring_radius=args.marker_ring_radius,
            marker_ring_start_epoch=args.marker_ring_start_epoch,
            use_change_mask_consistency_loss=args.use_change_mask_consistency_loss,
            lambda_change_mask_consistency=args.lambda_change_mask_consistency,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        _save_history(output_dir / "history.json", history)

        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        current_val_loss = float(val_metrics["loss"])
        if track_post_aux_best and epoch >= args.marker_ring_start_epoch and current_val_loss < float(best_post_aux_state["loss"]):
            best_post_aux_state = {"loss": current_val_loss, "epoch": epoch}
            save_named_checkpoint(
                checkpoint_path=output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        epoch_iterator.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                },
                ensure_ascii=True,
            )
        )

        save_epoch_visualizations(
            model=model,
            loader=train_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            output_dir=output_dir,
            epoch=epoch,
            split="train",
            num_samples=args.vis_samples,
            num_inference_steps=args.vis_inference_steps,
        )
        save_epoch_visualizations(
            model=model,
            loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            output_dir=output_dir,
            epoch=epoch,
            split="val",
            num_samples=args.vis_samples,
            num_inference_steps=args.vis_inference_steps,
        )

    _save_history(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
