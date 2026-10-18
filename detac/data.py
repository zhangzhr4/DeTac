from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


AVAILABLE_AUGMENTATION_PROFILES = ("none", "moderate", "strong")


def _sample_uniform(low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand(()).item())


def _sample_bool(probability: float) -> bool:
    return bool(torch.rand(()).item() < probability)


def _build_shading_map(image: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    _, height, width = image.shape
    coarse_height = max(height // 32, 4)
    coarse_width = max(width // 32, 4)
    shading = torch.rand((3, coarse_height, coarse_width), dtype=image.dtype)
    shading = TF.resize(
        shading,
        size=[height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return shading.mul(max_value - min_value).add(min_value).clamp(min_value, max_value)


def _odd_kernel_from_sigma(sigma: float) -> int:
    kernel = max(3, int(round(sigma * 4.0)))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return (mask > 0.5).float()
    kernel_size = 2 * radius + 1
    return (
        torch.nn.functional.max_pool2d(
            mask.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        .squeeze(0)
        .clamp(0.0, 1.0)
    )


def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return (mask > 0.5).float()
    kernel_size = 2 * radius + 1
    inverted = 1.0 - mask
    eroded_inverted = torch.nn.functional.max_pool2d(
        inverted.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ).squeeze(0)
    return (1.0 - eroded_inverted).clamp(0.0, 1.0)


def _soften_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _odd_kernel_from_sigma(sigma)
    return TF.gaussian_blur(mask.clamp(0.0, 1.0), kernel_size=[kernel, kernel], sigma=sigma).clamp(0.0, 1.0)


def _extract_marker_components(mask: torch.Tensor) -> list[tuple[float, float, int]]:
    binary = (mask[0] > 0.5).to(torch.uint8).cpu()
    height, width = binary.shape
    visited = torch.zeros((height, width), dtype=torch.bool)
    components: list[tuple[float, float, int]] = []

    for y in range(height):
        for x in range(width):
            if binary[y, x].item() == 0 or visited[y, x].item():
                continue

            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            points_y: list[int] = []
            points_x: list[int] = []

            while queue:
                cy, cx = queue.popleft()
                points_y.append(cy)
                points_x.append(cx)

                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx].item() or binary[ny, nx].item() == 0:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            area = len(points_y)
            center_y = float(sum(points_y)) / max(area, 1)
            center_x = float(sum(points_x)) / max(area, 1)
            components.append((center_y, center_x, area))

    return components


def _draw_disk(
    canvas: torch.Tensor,
    center_y: float,
    center_x: float,
    radius: int,
) -> None:
    _, height, width = canvas.shape
    y_min = max(0, int(center_y) - radius)
    y_max = min(height, int(center_y) + radius + 1)
    x_min = max(0, int(center_x) - radius)
    x_max = min(width, int(center_x) + radius + 1)

    if y_min >= y_max or x_min >= x_max:
        return

    ys = torch.arange(y_min, y_max, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(x_min, x_max, dtype=torch.float32).view(1, -1)
    disk = (((ys - center_y) ** 2 + (xs - center_x) ** 2) <= float(radius * radius)).to(canvas.dtype)
    canvas[0, y_min:y_max, x_min:x_max] = torch.maximum(canvas[0, y_min:y_max, x_min:x_max], disk)


def _apply_sparse_circular_marker_expansion(
    mask: torch.Tensor,
    *,
    stride_choices: tuple[int, ...] = (3, 4),
    radius_min: int = 3,
    radius_max: int = 6,
    expand_all_circles: bool = False,
) -> torch.Tensor:
    binary_mask = (mask > 0.5).float()
    components = _extract_marker_components(binary_mask)
    if not components:
        return binary_mask

    components = sorted(components, key=lambda item: (item[0], item[1]))
    stride = int(stride_choices[torch.randint(0, len(stride_choices), ()).item()])
    offset = int(torch.randint(0, stride, ()).item())
    expanded_mask = binary_mask.clone()

    for index, (center_y, center_x, area) in enumerate(components):
        if (not expand_all_circles) and index % stride == offset:
            continue

        base_radius = max(1.0, (float(area) / 3.141592653589793) ** 0.5)
        extra_radius = int(torch.randint(radius_min, radius_max + 1, ()).item())
        disk_radius = max(int(round(base_radius)) + extra_radius, extra_radius)
        _draw_disk(expanded_mask, center_y=center_y, center_x=center_x, radius=disk_radius)

    return expanded_mask.clamp(0.0, 1.0)


class DeTacAugmentor:
    def __init__(
        self,
        profile: str,
        apply_probability: float = 1.0,
        use_marker_augmentation: bool = False,
        marker_aug_size_only: bool = False,
        marker_aug_expand_all_circles: bool = False,
    ) -> None:
        if profile not in AVAILABLE_AUGMENTATION_PROFILES:
            raise ValueError(
                f"Unsupported augmentation profile '{profile}'. "
                f"Available choices: {', '.join(AVAILABLE_AUGMENTATION_PROFILES)}"
            )
        if not 0.0 <= apply_probability <= 1.0:
            raise ValueError("apply_probability must be within [0, 1].")
        self.profile = profile
        self.apply_probability = apply_probability
        self.use_marker_augmentation = use_marker_augmentation
        self.marker_aug_size_only = marker_aug_size_only
        self.marker_aug_expand_all_circles = marker_aug_expand_all_circles

    def _get_profile_config(self) -> dict[str, float]:
        if self.profile == "moderate":
            return {
                "geom_prob": 0.5,
                "affine_prob": 0.35,
                "rotation_deg": 8.0,
                "translate_frac": 0.05,
                "scale_min": 0.95,
                "scale_max": 1.05,
                "color_prob": 0.7,
                "channel_gain_min": 0.9,
                "channel_gain_max": 1.1,
                "gamma_min": 0.9,
                "gamma_max": 1.1,
                "saturation_min": 0.9,
                "saturation_max": 1.15,
                "hue_delta": 0.02,
                "shading_prob": 0.3,
                "shading_min": 0.8,
                "shading_max": 1.2,
                "blur_prob": 0.2,
                "blur_sigma_min": 0.3,
                "blur_sigma_max": 1.0,
                "noise_prob": 0.2,
                "noise_sigma_min": 0.002,
                "noise_sigma_max": 0.01,
                "marker_dilate_prob": 0.8,
                "marker_dilate_radius_min": 2,
                "marker_dilate_radius_max": 4,
                "marker_erode_radius": 1,
                "marker_soft_sigma_min": 1.4,
                "marker_soft_sigma_max": 2.4,
                "marker_darkness_min": 0.4,
                "marker_darkness_max": 0.72,
                "marker_tint_min": 0.96,
                "marker_tint_max": 1.04,
                "marker_ring_radius_min": 1,
                "marker_ring_radius_max": 3,
                "marker_ring_soft_sigma_min": 0.9,
                "marker_ring_soft_sigma_max": 1.8,
                "marker_ring_strength_min": 0.04,
                "marker_ring_strength_max": 0.08,
                "marker_residual_prob": 0.5,
                "marker_residual_sigma_min": 1.0,
                "marker_residual_sigma_max": 2.0,
                "marker_residual_strength_min": 0.02,
                "marker_residual_strength_max": 0.05,
                "marker_edge_sigma_min": 1.6,
                "marker_edge_sigma_max": 2.6,
                "marker_edge_threshold_min": 0.42,
                "marker_edge_threshold_max": 0.56,
                "marker_final_blur_prob": 0.8,
                "marker_final_blur_sigma_min": 0.9,
                "marker_final_blur_sigma_max": 1.8,
            }
        if self.profile == "strong":
            return {
                "geom_prob": 0.65,
                "affine_prob": 0.45,
                "rotation_deg": 10.0,
                "translate_frac": 0.06,
                "scale_min": 0.93,
                "scale_max": 1.07,
                "color_prob": 0.85,
                "channel_gain_min": 0.8,
                "channel_gain_max": 1.2,
                "gamma_min": 0.82,
                "gamma_max": 1.2,
                "saturation_min": 0.8,
                "saturation_max": 1.3,
                "hue_delta": 0.04,
                "shading_prob": 0.45,
                "shading_min": 0.7,
                "shading_max": 1.3,
                "blur_prob": 0.3,
                "blur_sigma_min": 0.4,
                "blur_sigma_max": 1.2,
                "noise_prob": 0.3,
                "noise_sigma_min": 0.003,
                "noise_sigma_max": 0.012,
                "marker_dilate_prob": 0.85,
                "marker_dilate_radius_min": 4,
                "marker_dilate_radius_max": 6,
                "marker_erode_radius": 1,
                "marker_soft_sigma_min": 1.8,
                "marker_soft_sigma_max": 3.0,
                "marker_darkness_min": 0.35,
                "marker_darkness_max": 0.65,
                "marker_tint_min": 0.95,
                "marker_tint_max": 1.05,
                "marker_ring_radius_min": 2,
                "marker_ring_radius_max": 4,
                "marker_ring_soft_sigma_min": 1.0,
                "marker_ring_soft_sigma_max": 2.0,
                "marker_ring_strength_min": 0.05,
                "marker_ring_strength_max": 0.10,
                "marker_residual_prob": 0.5,
                "marker_residual_sigma_min": 1.2,
                "marker_residual_sigma_max": 2.4,
                "marker_residual_strength_min": 0.02,
                "marker_residual_strength_max": 0.05,
                "marker_edge_sigma_min": 2.0,
                "marker_edge_sigma_max": 3.2,
                "marker_edge_threshold_min": 0.40,
                "marker_edge_threshold_max": 0.55,
                "marker_final_blur_prob": 0.85,
                "marker_final_blur_sigma_min": 1.2,
                "marker_final_blur_sigma_max": 2.2,
            }
        return {}

    @staticmethod
    def _apply_affine(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        mask_gt: torch.Tensor,
        rotation_deg: float,
        translate_frac: float,
        scale_min: float,
        scale_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = marker_img.shape
        angle = _sample_uniform(-rotation_deg, rotation_deg)
        translate_x = int(round(_sample_uniform(-translate_frac, translate_frac) * width))
        translate_y = int(round(_sample_uniform(-translate_frac, translate_frac) * height))
        scale = _sample_uniform(scale_min, scale_max)
        shear = [0.0, 0.0]

        marker_img = TF.affine(
            marker_img,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        nomarker_img = TF.affine(
            nomarker_img,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask_gt = TF.affine(
            mask_gt,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        return marker_img, nomarker_img, mask_gt

    @staticmethod
    def _apply_color_jitter(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        channel_gain_min: float,
        channel_gain_max: float,
        gamma_min: float,
        gamma_max: float,
        saturation_min: float,
        saturation_max: float,
        hue_delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gains = torch.tensor(
            [
                _sample_uniform(channel_gain_min, channel_gain_max),
                _sample_uniform(channel_gain_min, channel_gain_max),
                _sample_uniform(channel_gain_min, channel_gain_max),
            ],
            dtype=marker_img.dtype,
        ).view(3, 1, 1)
        gamma = _sample_uniform(gamma_min, gamma_max)
        saturation = _sample_uniform(saturation_min, saturation_max)
        hue = _sample_uniform(-hue_delta, hue_delta)

        marker_img = (marker_img * gains).clamp(0.0, 1.0)
        nomarker_img = (nomarker_img * gains).clamp(0.0, 1.0)
        marker_img = TF.adjust_gamma(marker_img, gamma=gamma)
        nomarker_img = TF.adjust_gamma(nomarker_img, gamma=gamma)
        marker_img = TF.adjust_saturation(marker_img, saturation)
        nomarker_img = TF.adjust_saturation(nomarker_img, saturation)
        marker_img = TF.adjust_hue(marker_img, hue)
        nomarker_img = TF.adjust_hue(nomarker_img, hue)
        return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0)

    @staticmethod
    def _apply_shading(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        shading_min: float,
        shading_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shading = _build_shading_map(
            image=marker_img,
            min_value=shading_min,
            max_value=shading_max,
        )
        marker_img = (marker_img * shading).clamp(0.0, 1.0)
        nomarker_img = (nomarker_img * shading).clamp(0.0, 1.0)
        return marker_img, nomarker_img

    @staticmethod
    def _apply_crop_zoom(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        mask_gt: torch.Tensor,
        *,
        crop_ratio_min: float = 0.35,
        crop_ratio_max: float = 0.65,
        center_jitter_fraction: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = marker_img.shape
        crop_ratio = _sample_uniform(crop_ratio_min, crop_ratio_max)
        crop_height = max(32, int(round(height * crop_ratio)))
        crop_width = max(32, int(round(width * crop_ratio)))

        top_max = max(0, height - crop_height)
        left_max = max(0, width - crop_width)
        center_y = height / 2.0
        center_x = width / 2.0
        top_center = int(round(center_y - crop_height / 2.0))
        left_center = int(round(center_x - crop_width / 2.0))
        top_jitter = int(round(top_max * center_jitter_fraction))
        left_jitter = int(round(left_max * center_jitter_fraction))
        top = top_center + (int(torch.randint(-top_jitter, top_jitter + 1, ()).item()) if top_jitter > 0 else 0)
        left = left_center + (int(torch.randint(-left_jitter, left_jitter + 1, ()).item()) if left_jitter > 0 else 0)
        top = min(max(top, 0), top_max)
        left = min(max(left, 0), left_max)

        marker_img = TF.resized_crop(
            marker_img,
            top=top,
            left=left,
            height=crop_height,
            width=crop_width,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        nomarker_img = TF.resized_crop(
            nomarker_img,
            top=top,
            left=left,
            height=crop_height,
            width=crop_width,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_gt = TF.resized_crop(
            mask_gt,
            top=top,
            left=left,
            height=crop_height,
            width=crop_width,
            size=[height, width],
            interpolation=InterpolationMode.NEAREST,
        )
        return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0), (mask_gt > 0.5).float()

    @staticmethod
    def _apply_standard_augmentation(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        mask_gt: torch.Tensor,
        config: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _sample_bool(config["geom_prob"]):
            if _sample_bool(0.5):
                marker_img = TF.hflip(marker_img)
                nomarker_img = TF.hflip(nomarker_img)
                mask_gt = TF.hflip(mask_gt)
            if _sample_bool(0.5):
                marker_img = TF.vflip(marker_img)
                nomarker_img = TF.vflip(nomarker_img)
                mask_gt = TF.vflip(mask_gt)
            if _sample_bool(0.5):
                angle = 90 * int(torch.randint(1, 4, ()).item())
                marker_img = TF.rotate(marker_img, angle, interpolation=InterpolationMode.BILINEAR)
                nomarker_img = TF.rotate(nomarker_img, angle, interpolation=InterpolationMode.BILINEAR)
                mask_gt = TF.rotate(mask_gt, angle, interpolation=InterpolationMode.NEAREST)

        if _sample_bool(config["affine_prob"]):
            marker_img, nomarker_img, mask_gt = DeTacAugmentor._apply_affine(
                marker_img=marker_img,
                nomarker_img=nomarker_img,
                mask_gt=mask_gt,
                rotation_deg=config["rotation_deg"],
                translate_frac=config["translate_frac"],
                scale_min=config["scale_min"],
                scale_max=config["scale_max"],
            )

        if _sample_bool(config["color_prob"]):
            marker_img, nomarker_img = DeTacAugmentor._apply_color_jitter(
                marker_img=marker_img,
                nomarker_img=nomarker_img,
                channel_gain_min=config["channel_gain_min"],
                channel_gain_max=config["channel_gain_max"],
                gamma_min=config["gamma_min"],
                gamma_max=config["gamma_max"],
                saturation_min=config["saturation_min"],
                saturation_max=config["saturation_max"],
                hue_delta=config["hue_delta"],
            )

        if _sample_bool(config["shading_prob"]):
            marker_img, nomarker_img = DeTacAugmentor._apply_shading(
                marker_img=marker_img,
                nomarker_img=nomarker_img,
                shading_min=config["shading_min"],
                shading_max=config["shading_max"],
            )

        if _sample_bool(config["blur_prob"]):
            sigma = _sample_uniform(config["blur_sigma_min"], config["blur_sigma_max"])
            marker_img = TF.gaussian_blur(marker_img, kernel_size=[5, 5], sigma=sigma)
            nomarker_img = TF.gaussian_blur(nomarker_img, kernel_size=[5, 5], sigma=sigma)

        if _sample_bool(config["noise_prob"]):
            sigma = _sample_uniform(config["noise_sigma_min"], config["noise_sigma_max"])
            noise = torch.randn_like(marker_img) * sigma
            marker_img = (marker_img + noise).clamp(0.0, 1.0)
            nomarker_img = (nomarker_img + noise).clamp(0.0, 1.0)

        return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0), (mask_gt > 0.5).float()

    @staticmethod
    def _render_marker_from_mask(
        nomarker_img: torch.Tensor,
        marker_reference: torch.Tensor,
        mask_gt: torch.Tensor,
        config: dict[str, float],
    ) -> torch.Tensor:
        binary_mask = (mask_gt > 0.5).float()
        soft_sigma = _sample_uniform(config["marker_soft_sigma_min"], config["marker_soft_sigma_max"])
        soft_mask = _soften_mask(binary_mask, sigma=soft_sigma)

        marker_darkness = _sample_uniform(config["marker_darkness_min"], config["marker_darkness_max"])
        channel_tint = torch.tensor(
            [
                _sample_uniform(config["marker_tint_min"], config["marker_tint_max"]),
                _sample_uniform(config["marker_tint_min"], config["marker_tint_max"]),
                _sample_uniform(config["marker_tint_min"], config["marker_tint_max"]),
            ],
            dtype=nomarker_img.dtype,
        ).view(3, 1, 1)
        marker_fill = (nomarker_img * marker_darkness * channel_tint).clamp(0.0, 1.0)
        marker_img = nomarker_img * (1.0 - soft_mask) + marker_fill * soft_mask

        ring_radius = int(
            torch.randint(
                int(config["marker_ring_radius_min"]),
                int(config["marker_ring_radius_max"]) + 1,
                (),
            ).item()
        )
        ring_mask = (_dilate_mask(binary_mask, ring_radius) - binary_mask).clamp(0.0, 1.0)
        ring_soft = _soften_mask(
            ring_mask,
            sigma=_sample_uniform(config["marker_ring_soft_sigma_min"], config["marker_ring_soft_sigma_max"]),
        )
        ring_strength = _sample_uniform(
            config["marker_ring_strength_min"],
            config["marker_ring_strength_max"],
        )
        marker_img = (marker_img * (1.0 - ring_soft * ring_strength)).clamp(0.0, 1.0)

        if _sample_bool(config["marker_residual_prob"]):
            residual = (nomarker_img - marker_reference).abs().mean(dim=0, keepdim=True)
            residual = residual / residual.amax().clamp_min(1e-6)
            residual_soft = _soften_mask(
                residual,
                sigma=_sample_uniform(config["marker_residual_sigma_min"], config["marker_residual_sigma_max"]),
            )
            residual_strength = _sample_uniform(
                config["marker_residual_strength_min"],
                config["marker_residual_strength_max"],
            )
            marker_img = (marker_img * (1.0 - residual_soft * residual_strength)).clamp(0.0, 1.0)

        return marker_img.clamp(0.0, 1.0)

    @staticmethod
    def _apply_marker_size_blur_augmentation(
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        mask_gt: torch.Tensor,
        config: dict[str, float],
        *,
        expand_all_circles: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        marker_img, nomarker_img, binary_mask = DeTacAugmentor._apply_crop_zoom(
            marker_img=marker_img,
            nomarker_img=nomarker_img,
            mask_gt=mask_gt,
            crop_ratio_min=0.35,
            crop_ratio_max=0.65,
            center_jitter_fraction=0.2,
        )

        binary_mask = _apply_sparse_circular_marker_expansion(
            binary_mask,
            stride_choices=(3, 4),
            radius_min=3,
            radius_max=6,
            expand_all_circles=expand_all_circles,
        )

        edge_sigma = _sample_uniform(config["marker_edge_sigma_min"], config["marker_edge_sigma_max"])
        soft_mask = _soften_mask(binary_mask, sigma=edge_sigma)
        threshold = _sample_uniform(config["marker_edge_threshold_min"], config["marker_edge_threshold_max"])
        binary_mask = (soft_mask > threshold).float()

        marker_img = DeTacAugmentor._render_marker_from_mask(
            nomarker_img=nomarker_img,
            marker_reference=marker_img,
            mask_gt=binary_mask,
            config=config,
        )

        if _sample_bool(config["marker_final_blur_prob"]):
            sigma = _sample_uniform(config["marker_final_blur_sigma_min"], config["marker_final_blur_sigma_max"])
            kernel = _odd_kernel_from_sigma(sigma)
            marker_img = TF.gaussian_blur(marker_img, kernel_size=[kernel, kernel], sigma=sigma)

        return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0), (binary_mask > 0.5).float()

    def apply(
        self,
        marker_img: torch.Tensor,
        nomarker_img: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self._get_profile_config()
        if not config:
            return marker_img, nomarker_img, mask_gt
        if not _sample_bool(self.apply_probability):
            return marker_img, nomarker_img, mask_gt

        if not self.use_marker_augmentation:
            marker_img, nomarker_img, mask_gt = self._apply_standard_augmentation(
                marker_img=marker_img,
                nomarker_img=nomarker_img,
                mask_gt=mask_gt,
                config=config,
            )
            return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0), (mask_gt > 0.5).float()

        if self.marker_aug_size_only:
            marker_img, nomarker_img, mask_gt = self._apply_marker_size_blur_augmentation(
                marker_img=marker_img,
                nomarker_img=nomarker_img,
                mask_gt=mask_gt,
                config=config,
                expand_all_circles=self.marker_aug_expand_all_circles,
            )
        else:
            branch_selector = torch.rand(()).item()
            if branch_selector < 0.5:
                marker_img, nomarker_img, mask_gt = self._apply_standard_augmentation(
                    marker_img=marker_img,
                    nomarker_img=nomarker_img,
                    mask_gt=mask_gt,
                    config=config,
                )
            elif branch_selector < 0.75:
                marker_img, nomarker_img, mask_gt = self._apply_marker_size_blur_augmentation(
                    marker_img=marker_img,
                    nomarker_img=nomarker_img,
                    mask_gt=mask_gt,
                    config=config,
                    expand_all_circles=self.marker_aug_expand_all_circles,
                )
            else:
                marker_img, nomarker_img, mask_gt = self._apply_marker_size_blur_augmentation(
                    marker_img=marker_img,
                    nomarker_img=nomarker_img,
                    mask_gt=mask_gt,
                    config=config,
                    expand_all_circles=self.marker_aug_expand_all_circles,
                )
                marker_img, nomarker_img = self._apply_color_jitter(
                    marker_img=marker_img,
                    nomarker_img=nomarker_img,
                    channel_gain_min=config["channel_gain_min"],
                    channel_gain_max=config["channel_gain_max"],
                    gamma_min=config["gamma_min"],
                    gamma_max=config["gamma_max"],
                    saturation_min=config["saturation_min"],
                    saturation_max=config["saturation_max"],
                    hue_delta=config["hue_delta"],
                )
                marker_img, nomarker_img = self._apply_shading(
                    marker_img=marker_img,
                    nomarker_img=nomarker_img,
                    shading_min=config["shading_min"],
                    shading_max=config["shading_max"],
                )

        return marker_img.clamp(0.0, 1.0), nomarker_img.clamp(0.0, 1.0), (mask_gt > 0.5).float()


def _load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    samples: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return samples


class DeTacDataset(Dataset):
    """
    Dataset for aligned tactile triplets.

    All images are loaded at their original resolution and directly fed to the model.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        expected_image_size: tuple[int, int] | None = (480, 640),
        augmentation_profile: str = "none",
        augmentation_apply_probability: float = 1.0,
        use_marker_augmentation: bool = False,
        marker_aug_size_only: bool = False,
        marker_aug_expand_all_circles: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root is not None else self.manifest_path.parent.parent
        self.expected_image_size = expected_image_size
        self.samples = _load_manifest(self.manifest_path)
        self.augmentation_profile = augmentation_profile
        self.augmentation_apply_probability = augmentation_apply_probability
        self.use_marker_augmentation = use_marker_augmentation
        self.marker_aug_size_only = marker_aug_size_only
        self.augmentor = DeTacAugmentor(
            profile=augmentation_profile,
            apply_probability=augmentation_apply_probability,
            use_marker_augmentation=use_marker_augmentation,
            marker_aug_size_only=marker_aug_size_only,
            marker_aug_expand_all_circles=marker_aug_expand_all_circles,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.data_root / path

    def _validate_image_size(self, image_size: tuple[int, int], sample_id: str) -> None:
        if self.expected_image_size is None:
            return

        expected_height, expected_width = self.expected_image_size
        width, height = image_size
        if (height, width) != (expected_height, expected_width):
            raise ValueError(
                f"Unexpected image size for sample {sample_id}: "
                f"got {(height, width)}, expected {(expected_height, expected_width)}"
            )

    @staticmethod
    def _normalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mul(2.0).sub(1.0)

    @staticmethod
    def _load_mask_tensor(mask: Image.Image) -> torch.Tensor:
        mask_tensor = TF.to_tensor(mask.convert("L"))
        return (mask_tensor > 0.5).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        marker_img = Image.open(self._resolve_path(sample["marker_path"])).convert("RGB")
        nomarker_img = Image.open(self._resolve_path(sample["nomarker_path"])).convert("RGB")
        mask_gt = Image.open(self._resolve_path(sample["mask_path"])).convert("L")

        if marker_img.size != nomarker_img.size or marker_img.size != mask_gt.size:
            raise ValueError(f"Image size mismatch for sample {sample['sample_id']}")

        self._validate_image_size(marker_img.size, sample["sample_id"])

        marker_tensor = TF.to_tensor(marker_img)
        nomarker_tensor = TF.to_tensor(nomarker_img)
        mask_tensor = self._load_mask_tensor(mask_gt)

        marker_tensor, nomarker_tensor, mask_tensor = self.augmentor.apply(
            marker_img=marker_tensor,
            nomarker_img=nomarker_tensor,
            mask_gt=mask_tensor,
        )

        return {
            "sample_id": sample["sample_id"],
            "object_id": sample["object_id"],
            "marker_img": self._normalize_rgb(marker_tensor),
            "nomarker_img": self._normalize_rgb(nomarker_tensor),
            "mask_gt": mask_tensor,
        }
