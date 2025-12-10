"""Two-stage correlation combining phase correlation with NCC refinement."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from .ncc import ScoreFunc, _select_score_func
from .phase import phase_correlation, phase_correlation_peak


def two_stage_correlation(
    img1: torch.Tensor,
    img2: torch.Tensor,
    shift_range: int = 50,
    top_k: int = 5,
    score_func: Union[str, Callable] = "default",
    device: Optional[torch.device] = None,
) -> Tuple[Tuple[int, int], float]:
    """
    Phase correlation to get candidates, NCC refinement to pick best.
    """
    if device is None:
        device = img1.device if hasattr(img1, "device") else torch.device("cpu")

    img1 = torch.as_tensor(img1, device=device).float()
    img2 = torch.as_tensor(img2, device=device).float()

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions, got {img1.shape} and {img2.shape}"
        )
    if img1.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensors, got {img1.dim()}D")

    phase_corr = phase_correlation(img1, img2, device)

    flat_phase_corr = phase_corr.flatten()
    top_k_values, top_k_indices = torch.topk(
        flat_phase_corr, k=min(top_k, flat_phase_corr.numel())
    )

    H, W = phase_corr.shape
    top_k_y = top_k_indices // W
    top_k_x = top_k_indices % W

    candidate_shifts = []
    for y, x in zip(top_k_y, top_k_x):
        y_val = y.item() if hasattr(y, "item") else int(y)
        x_val = x.item() if hasattr(x, "item") else int(x)

        converted_y = y_val if y_val <= H // 2 else y_val - H
        converted_x = x_val if x_val <= W // 2 else x_val - W

        candidate_shifts.append((-converted_y, -converted_x))

    selected_score_func = _select_score_func(score_func)  # type: ignore[arg-type]

    best_shift: Optional[Tuple[int, int]] = None
    best_value = float("-inf")

    total_size = img1.numel()
    neighborhood_size = 2
    C, H, W = img1.shape

    for shift_y, shift_x in candidate_shifts:
        for dy in range(-neighborhood_size, neighborhood_size + 1):
            for dx in range(-neighborhood_size, neighborhood_size + 1):
                new_shift_y = shift_y + dy
                new_shift_x = shift_x + dx

                if abs(new_shift_y) > shift_range or abs(new_shift_x) > shift_range:
                    continue

                start_y_1 = max(0, -new_shift_y)
                end_y_1 = min(H, H - new_shift_y)
                start_x_1 = max(0, -new_shift_x)
                end_x_1 = min(W, W - new_shift_x)

                start_y_2 = max(0, new_shift_y)
                end_y_2 = min(H, H + new_shift_y)
                start_x_2 = max(0, new_shift_x)
                end_x_2 = min(W, W + new_shift_x)

                if end_y_1 <= start_y_1 or end_x_1 <= start_x_1:
                    continue

                overlap1 = img1[:, start_y_1:end_y_1, start_x_1:end_x_1]
                overlap2 = img2[:, start_y_2:end_y_2, start_x_2:end_x_2]

                current_value = selected_score_func(
                    overlap1, overlap2, total_size
                ).item()

                if current_value > best_value:
                    best_value = current_value
                    best_shift = (new_shift_y, new_shift_x)

    if best_shift is None:
        shift_y, shift_x, pc_val = phase_correlation_peak(img1, img2, device)
        return (shift_y, shift_x), pc_val

    return best_shift, best_value


__all__ = ["two_stage_correlation"]
