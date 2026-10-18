from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DDPMScheduler
from PIL import Image
from torchvision.transforms import functional as TF

from .model import BaseDeTacModel, build_detac_model


def _normalize_rgb_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image.convert("RGB"))
    return tensor.mul(2.0).sub(1.0)


@torch.no_grad()
def infer_detac(
    model: BaseDeTacModel,
    noise_scheduler: DDPMScheduler,
    marker_image: str | Path | Image.Image | torch.Tensor,
    device: str | torch.device,
    num_inference_steps: int = 50,
    mask_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Full-image inference.

    The input marker image is directly fed to the model at its original size and
    the outputs keep exactly the same spatial resolution.
    Tensor inputs must already be normalized to [-1, 1]; PIL images and paths
    are normalized here. No range guessing is performed on tensor inputs.
    """

    if isinstance(marker_image, (str, Path)):
        marker_tensor = _normalize_rgb_tensor(Image.open(marker_image))
    elif isinstance(marker_image, Image.Image):
        marker_tensor = _normalize_rgb_tensor(marker_image)
    elif isinstance(marker_image, torch.Tensor):
        marker_tensor = marker_image.detach().cpu()
    else:
        raise TypeError("marker_image must be a path, PIL image, or torch.Tensor")

    if marker_tensor.ndim != 3 or marker_tensor.shape[0] != 3:
        raise ValueError("marker_image must have shape [3, H, W]")

    if not 1 <= num_inference_steps <= noise_scheduler.config.num_train_timesteps:
        raise ValueError("num_inference_steps must be between 1 and the training timestep count.")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be within [0, 1].")

    model = model.to(device)
    model.eval()

    marker_batch = marker_tensor.unsqueeze(0).to(device)
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
    full_markerless = reconstructed_image.squeeze(0).cpu().add(1.0).div(2.0).clamp(0.0, 1.0)
    full_mask = mask_probability.squeeze(0).cpu().clamp(0.0, 1.0)

    return {
        "markerless_image": full_markerless,
        "mask_probability": full_mask,
        "mask_binary": (full_mask >= mask_threshold).float(),
    }


def load_checkpoint(path: str | Path, device: str | torch.device) -> tuple[BaseDeTacModel, dict]:
    """Load state-dict checkpoints produced by DeTac or the original DualTac trainer."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("args", {})
    if not isinstance(config, dict):
        raise TypeError("Checkpoint args must be a dictionary.")
    model = build_detac_model(
        architecture=config.get("architecture", "pixel_concat_unet"),
        image_size=(int(config.get("image_height", 480)), int(config.get("image_width", 640))),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint must contain model_state_dict or state_dict.")
    model.load_state_dict(state)
    model.eval()
    return model, config
