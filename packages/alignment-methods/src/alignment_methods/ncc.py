"""Normalized cross-correlation implementations and scoring helpers."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

ScoreFunc = Union[
    Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    Literal["default", "size_preference", "size_penalty"],
]


def _select_score_func(
    score_func: ScoreFunc,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    if score_func == "default":
        return _default_score_func_agnostic
    if score_func == "size_preference":
        return ncc_with_size_preference
    if score_func == "size_penalty":
        return ncc_with_size_penalty
    if callable(score_func):
        return score_func
    raise ValueError(
        "score_func must be a callable or one of 'default', 'size_preference', 'size_penalty'"
    )


def ncc(
    img1: torch.Tensor,
    img2: torch.Tensor,
    shift_range: Union[int, Tuple[int, int]] = 0,
    stride: int = 1,
    score_func: Optional[ScoreFunc] = "default",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute Normalized Cross Correlation between two images with support for shifting.

    Args:
        img1: First input image tensor of shape (C, H, W) or (H, W)
        img2: Second input image tensor of shape (C, H, W) or (H, W), same dimensions as img1
        shift_range: Maximum shift distance. If int, applies to both x and y directions.
                     If tuple (max_shift_y, max_shift_x), applies separately to y and x.
        stride: Stride for shifting positions (default 1)
        score_func: How to calculate score from overlapping regions. Can be:
                    - A string literal: "default", "size_preference", "size_penalty"
                    - A custom function for calculating score from overlapping regions
                    Default is "default" (overlap size agnostic NCC).
        device: Device to perform computation on (default: same as input tensors)

    Returns:
        Correlation tensor of shape (2*max_shift_y//stride+1, 2*max_shift_x//stride+1) containing NCC values
    """
    if device is None:
        device = img1.device if hasattr(img1, "device") else torch.device("cpu")

    img1 = torch.as_tensor(img1, device=device).float()
    img2 = torch.as_tensor(img2, device=device).float()

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions, got {img1.shape} and {img2.shape}"
        )
    if img1.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensors, got {img1.dim()}D")

    C, H, W = img1.shape

    if isinstance(shift_range, int):
        max_shift_y = max_shift_x = shift_range
    else:
        max_shift_y, max_shift_x = shift_range

    if max_shift_y > H // 2 or max_shift_x > W // 2:
        raise ValueError("Shift range too large for image dimensions")

    selected_score_func = _select_score_func(score_func)

    output_shape = (2 * max_shift_y // stride + 1, 2 * max_shift_x // stride + 1)
    ncc_map = torch.zeros(output_shape, device=device)

    mean1 = torch.mean(img1)
    std1 = torch.std(img1)
    mean2 = torch.mean(img2)
    std2 = torch.std(img2)

    img1_norm = (img1 - mean1) / std1 if std1 != 0 else img1 - mean1
    img2_norm = (img2 - mean2) / std2 if std2 != 0 else img2 - mean2

    total_size = C * H * W

    shift_idx_y = 0
    for dy in range(-max_shift_y, max_shift_y + 1, stride):
        shift_idx_x = 0
        for dx in range(-max_shift_x, max_shift_x + 1, stride):
            start_y_1 = max(0, -dy)
            end_y_1 = min(H, H - dy)
            start_x_1 = max(0, -dx)
            end_x_1 = min(W, W - dx)

            start_y_2 = max(0, dy)
            end_y_2 = min(H, H + dy)
            start_x_2 = max(0, dx)
            end_x_2 = min(W, W + dx)

            if end_y_1 <= start_y_1 or end_x_1 <= start_x_1:
                ncc_map[shift_idx_y, shift_idx_x] = torch.tensor(
                    float("-inf"), device=device
                )
                shift_idx_x += 1
                continue

            overlap1 = img1_norm[:, start_y_1:end_y_1, start_x_1:end_x_1]
            overlap2 = img2_norm[:, start_y_2:end_y_2, start_x_2:end_x_2]

            if overlap1.shape != overlap2.shape:
                raise RuntimeError(
                    f"Overlap shapes don't match: {overlap1.shape} vs {overlap2.shape}"
                )

            score = selected_score_func(overlap1, overlap2, total_size)
            ncc_map[shift_idx_y, shift_idx_x] = score
            shift_idx_x += 1
        shift_idx_y += 1

    return ncc_map


def ncc_efficient(
    img1: torch.Tensor,
    img2: torch.Tensor,
    shift_range: Union[int, Tuple[int, int]] = 0,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute NCC using convolutional formulation for speed."""
    if device is None:
        device = img1.device if hasattr(img1, "device") else torch.device("cpu")

    img1 = torch.as_tensor(img1, device=device).float()
    img2 = torch.as_tensor(img2, device=device).float()

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions, got {img1.shape} and {img2.shape}"
        )
    if img1.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensors, got {img1.dim()}D")

    C, H, W = img1.shape

    if isinstance(shift_range, int):
        max_shift_y = max_shift_x = shift_range
    else:
        max_shift_y, max_shift_x = shift_range

    if max_shift_y > H // 2 or max_shift_x > W // 2:
        raise ValueError("Shift range too large for image dimensions")

    mean1 = torch.mean(img1)
    std1 = torch.std(img1)
    mean2 = torch.mean(img2)
    std2 = torch.std(img2)

    img1_norm = (img1 - mean1) / std1 if std1 != 0 else img1 - mean1
    img2_norm = (img2 - mean2) / std2 if std2 != 0 else img2 - mean2

    padded_img2 = F.pad(
        img2_norm,
        (max_shift_x, max_shift_x, max_shift_y, max_shift_y),
        mode="constant",
        value=0,
    )

    flipped_img1 = torch.flip(img1_norm, [-2, -1])
    kernel = flipped_img1.unsqueeze(0)

    correlations = F.conv2d(padded_img2.unsqueeze(0), kernel, stride=stride).squeeze(0)

    sum_sq1 = torch.sum(img1_norm**2)
    padded_img2_sq = F.pad(
        img2_norm**2,
        (max_shift_x, max_shift_x, max_shift_y, max_shift_y),
        mode="constant",
        value=0,
    )
    ones_kernel = torch.ones_like(img1_norm).unsqueeze(0)

    sum_sq2 = F.conv2d(padded_img2_sq.unsqueeze(0), ones_kernel, stride=stride).squeeze(
        0
    )

    norm_factor = torch.sqrt(sum_sq1 * sum_sq2 + 1e-8)
    ncc_map = correlations.squeeze(0) / norm_factor

    output_height = 2 * max_shift_y // stride + 1
    output_width = 2 * max_shift_x // stride + 1

    conv_h, conv_w = ncc_map.shape[-2], ncc_map.shape[-1]
    center_h, center_w = conv_h // 2, conv_w // 2

    start_h = center_h - (output_height // 2)
    end_h = start_h + output_height
    start_w = center_w - (output_width // 2)
    end_w = start_w + output_width

    result = ncc_map[:, start_h:end_h, start_w:end_w]

    if result.shape[0] == 1:
        result = result.squeeze(0)

    return result


def ncc_batched(
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    shift_range: Union[int, Tuple[int, int]] = 0,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Vectorized NCC for batches of image pairs."""
    if device is None:
        device = (
            img1_batch.device if hasattr(img1_batch, "device") else torch.device("cpu")
        )

    img1_batch = torch.as_tensor(img1_batch, device=device).float()
    img2_batch = torch.as_tensor(img2_batch, device=device).float()

    single_pair = False
    if img1_batch.dim() == 2:
        img1_batch = img1_batch.unsqueeze(0).unsqueeze(0)
        img2_batch = img2_batch.unsqueeze(0).unsqueeze(0)
        single_pair = True
    elif img1_batch.dim() == 3:
        if img2_batch.shape == img1_batch.shape:
            img1_batch = img1_batch.unsqueeze(0)
            img2_batch = img2_batch.unsqueeze(0)
            single_pair = True
        else:
            img1_batch = img1_batch.unsqueeze(1)
            img2_batch = img2_batch.unsqueeze(1)

    if img1_batch.shape != img2_batch.shape:
        if img1_batch.shape[0] == img2_batch.shape[0]:
            if img1_batch.shape[1] == 1 and img2_batch.shape[1] > 1:
                img1_batch = img1_batch.expand(-1, img2_batch.shape[1], -1, -1)
            elif img2_batch.shape[1] == 1 and img1_batch.shape[1] > 1:
                img2_batch = img2_batch.expand(-1, img1_batch.shape[1], -1, -1)
        if img1_batch.shape != img2_batch.shape:
            raise ValueError(
                f"Batched images must have compatible shapes, got {img1_batch.shape} and {img2_batch.shape}"
            )

    B, C, H, W = img1_batch.shape

    if isinstance(shift_range, int):
        max_shift_y = max_shift_x = shift_range
    else:
        max_shift_y, max_shift_x = shift_range

    if max_shift_y > H // 2 or max_shift_x > W // 2:
        raise ValueError("Shift range too large for image dimensions")

    mean1 = torch.mean(img1_batch.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
    std1 = (
        torch.std(img1_batch.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-8
    )
    mean2 = torch.mean(img2_batch.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
    std2 = (
        torch.std(img2_batch.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-8
    )

    img1_norm = (img1_batch - mean1) / std1
    img2_norm = (img2_batch - mean2) / std2

    padded_img2 = F.pad(
        img2_norm,
        (max_shift_x, max_shift_x, max_shift_y, max_shift_y),
        mode="constant",
        value=0,
    )

    flipped_img1 = torch.flip(img1_norm, [-2, -1])

    correlations = []
    for b in range(B):
        single_img1 = flipped_img1[b : b + 1]
        single_padded_img2 = padded_img2[b : b + 1]
        single_corr = F.conv2d(single_padded_img2, single_img1, stride=stride)
        correlations.append(single_corr.squeeze(0).squeeze(0))

    correlations = torch.stack(correlations, dim=0)

    padded_img2_sq = F.pad(
        img2_norm**2,
        (max_shift_x, max_shift_x, max_shift_y, max_shift_y),
        mode="constant",
        value=0,
    )
    ones_kernel = torch.ones(1, C, H, W, device=device)

    sum_sq2_all = []
    for b in range(B):
        single_padded_img2_sq = padded_img2_sq[b : b + 1]
        single_sum_sq2 = F.conv2d(
            single_padded_img2_sq, ones_kernel, stride=stride
        ).squeeze(0)
        sum_sq2_all.append(single_sum_sq2.squeeze(0))

    sum_sq2 = torch.stack(sum_sq2_all, dim=0)
    sum_sq1 = torch.sum(img1_norm**2, dim=[1, 2, 3], keepdim=False)

    norm_factor = torch.sqrt(sum_sq1.view(B, 1, 1) * sum_sq2 + 1e-8)
    ncc_map = correlations / norm_factor

    if single_pair:
        ncc_map = ncc_map.squeeze(0)

    return ncc_map


def _default_score_func_agnostic(
    overlap1: torch.Tensor, overlap2: torch.Tensor, total_size: int
) -> torch.Tensor:
    correlation = torch.sum(overlap1 * overlap2)
    normalized_correlation = correlation / torch.sqrt(
        torch.sum(overlap1**2) * torch.sum(overlap2**2) + 1e-8
    )
    return normalized_correlation


def _default_score_func(
    overlap1: torch.Tensor, overlap2: torch.Tensor, total_size: int
) -> torch.Tensor:
    correlation = torch.sum(overlap1 * overlap2)
    normalized_correlation = correlation / torch.sqrt(
        torch.sum(overlap1**2) * torch.sum(overlap2**2) + 1e-8
    )
    return normalized_correlation


def ncc_with_size_preference(
    overlap1: torch.Tensor, overlap2: torch.Tensor, total_size: int
) -> torch.Tensor:
    correlation = torch.sum(overlap1 * overlap2)
    norm_factor = torch.sqrt(torch.sum(overlap1**2) * torch.sum(overlap2**2) + 1e-8)
    normalized_correlation = correlation / norm_factor

    overlap_proportion = overlap1.numel() / total_size
    weighted_score = normalized_correlation * overlap_proportion
    return weighted_score


def ncc_with_size_penalty(
    overlap1: torch.Tensor, overlap2: torch.Tensor, total_size: int
) -> torch.Tensor:
    correlation = torch.sum(overlap1 * overlap2)
    norm_factor = torch.sqrt(torch.sum(overlap1**2) * torch.sum(overlap2**2) + 1e-8)
    normalized_correlation = correlation / norm_factor

    overlap_proportion = overlap1.numel() / total_size
    penalty_factor = min(1.0, 2 * overlap_proportion)
    penalized_score = normalized_correlation * penalty_factor
    return penalized_score


__all__ = [
    "ncc",
    "ncc_efficient",
    "ncc_batched",
    "ncc_with_size_preference",
    "ncc_with_size_penalty",
]
