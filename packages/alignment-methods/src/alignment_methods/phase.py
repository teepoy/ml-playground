"""Phase correlation utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def phase_correlation(
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute phase correlation map between two images."""
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

    C, H, W = img1.shape
    combined_phase_corr = torch.zeros((H, W), device=device)

    for c in range(C):
        channel1 = img1[c]
        channel2 = img2[c]

        fft1 = torch.fft.fft2(channel1)
        fft2 = torch.fft.fft2(channel2)

        cross_power = fft1 * torch.conj(fft2)
        magnitude = torch.clamp(torch.abs(cross_power), min=1e-8)
        normalized_cross_power = cross_power / magnitude

        phase_corr = torch.real(torch.fft.ifft2(normalized_cross_power))
        combined_phase_corr += phase_corr

    return combined_phase_corr


def phase_correlation_peak(
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, float]:
    """Find peak shift from phase correlation map."""
    phase_corr = phase_correlation(img1, img2, device)

    max_idx = torch.argmax(phase_corr)
    H, W = phase_corr.shape
    peak_y = max_idx // W
    peak_x = max_idx % W

    converted_peak_y = peak_y if peak_y <= H // 2 else peak_y - H
    converted_peak_x = peak_x if peak_x <= W // 2 else peak_x - W

    shift_y = -converted_peak_y
    shift_x = -converted_peak_x

    peak_value = phase_corr.flatten()[max_idx].item()

    shift_y = int(shift_y) if not isinstance(shift_y, int) else shift_y
    shift_x = int(shift_x) if not isinstance(shift_x, int) else shift_x

    return shift_y, shift_x, peak_value


__all__ = ["phase_correlation", "phase_correlation_peak"]
