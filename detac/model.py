from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from diffusers import UNet2DModel


AVAILABLE_ARCHITECTURES = ("pixel_concat_unet",)


def _resolve_image_size(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)

    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a sequence of two integers.")
    return (int(image_size[0]), int(image_size[1]))


class MaskSegmentationHead(nn.Module):
    """Lightweight segmentation head for marker mask prediction."""

    def __init__(self, in_channels: int = 3, hidden_channels: Sequence[int] = (32, 64)) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("hidden_channels must contain exactly two values.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[1], 1, kernel_size=3, padding=1),
        )

    def forward(self, marker_img: torch.Tensor) -> torch.Tensor:
        return self.net(marker_img)


class BaseDeTacModel(nn.Module):
    """Shared interface for DeTac training and sampling."""

    architecture_name = "base"

    def __init__(self, mask_hidden_channels: Sequence[int] = (32, 64)) -> None:
        super().__init__()
        self.mask_head = MaskSegmentationHead(
            in_channels=3,
            hidden_channels=mask_hidden_channels,
        )

    def prepare_training_targets(self, nomarker_img: torch.Tensor) -> dict[str, Any]:
        return {
            "diffusion_target": nomarker_img,
            "aux_losses": {},
        }

    def make_initial_diffusion_sample(self, marker_img: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(marker_img)

    def decode_denoised_sample(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

    def predict_mask(self, marker_img: torch.Tensor) -> torch.Tensor:
        return self.mask_head(marker_img)

    def predict_noise(
        self,
        noisy_img: torch.Tensor,
        timesteps: torch.Tensor,
        marker_img: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        noisy_img: torch.Tensor,
        timesteps: torch.Tensor,
        marker_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "noise_pred": self.predict_noise(noisy_img, timesteps, marker_img),
            "mask_logits": self.predict_mask(marker_img),
        }


class DeTacModel(BaseDeTacModel):
    """Pixel-space shading restoration and parallel marker extraction (Appendix A.1)."""

    architecture_name = "pixel_concat_unet"

    def __init__(
        self,
        image_size: int | Sequence[int] = (480, 640),
        unet_channels: Sequence[int] = (64, 128, 256, 256),
        layers_per_block: int = 2,
        mask_hidden_channels: Sequence[int] = (32, 64),
    ) -> None:
        super().__init__(mask_hidden_channels=mask_hidden_channels)

        sample_size = _resolve_image_size(image_size)
        down_block_types = tuple("DownBlock2D" for _ in unet_channels)
        up_block_types = tuple("UpBlock2D" for _ in unet_channels)

        self.diffusion_unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=6,
            out_channels=3,
            layers_per_block=layers_per_block,
            block_out_channels=tuple(unet_channels),
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            norm_num_groups=8,
        )

    def predict_noise(
        self,
        noisy_img: torch.Tensor,
        timesteps: torch.Tensor,
        marker_img: torch.Tensor,
    ) -> torch.Tensor:
        diffusion_input = torch.cat([noisy_img, marker_img], dim=1)
        return self.diffusion_unet(diffusion_input, timesteps).sample


def build_detac_model(
    architecture: str = "pixel_concat_unet",
    image_size: int | Sequence[int] = (480, 640),
    **kwargs: Any,
) -> BaseDeTacModel:
    """Build the paper architecture; retain its identifier for legacy checkpoints."""
    if architecture != "pixel_concat_unet":
        raise ValueError(f"Unsupported architecture {architecture!r}; DeTac uses pixel_concat_unet.")
    return DeTacModel(image_size=image_size, **kwargs)
