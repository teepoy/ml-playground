import pathlib

import pytest
import torch
import torchvision.transforms.functional as TF
from alignment_methods import (
    ncc_efficient,
    phase_correlation_peak,
    two_stage_correlation,
)
from PIL import Image


def _extract_shift(ncc_map: torch.Tensor) -> tuple[int, int]:
    h, w = ncc_map.shape
    max_idx = torch.argmax(ncc_map)
    return max_idx // w - (h // 2), max_idx % w - (w // 2)


def test_two_stage_matches_known_shift():
    import torch
    from alignment_methods import phase_correlation_peak, two_stage_correlation

    def test_two_stage_matches_true_shift():
        img = torch.zeros((3, 64, 64))
        img[0, 20:30, 20:30] = 1.0
        img[1, 10:18, 42:50] = 0.5
        img[2, 35:45, 8:18] = -0.7

        expected = (5, -7)
        shifted = torch.roll(img, shifts=expected, dims=(1, 2))

        detected, value = two_stage_correlation(img, shifted, shift_range=12, top_k=5)

        assert abs(detected[0] - expected[0]) <= 1
        assert abs(detected[1] - expected[1]) <= 1
        assert value > 0

    def test_two_stage_falls_back_to_phase_when_needed():
        img = torch.zeros((1, 32, 32))
        img[:, 4:8, 4:8] = 1.0
        shifted = torch.roll(img, shifts=(0, 0), dims=(1, 2))

        detected, value = two_stage_correlation(img, shifted, shift_range=0, top_k=1)
        pc_shift = phase_correlation_peak(img, shifted)

        assert detected == (pc_shift[0], pc_shift[1])
        assert value == pc_shift[2]
