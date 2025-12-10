import pytest
import torch
from alignment_methods import (
    ncc,
    ncc_batched,
    ncc_efficient,
    ncc_with_size_penalty,
    ncc_with_size_preference,
)


def _argmax_shift(ncc_map: torch.Tensor) -> tuple[int, int]:
    h, w = ncc_map.shape
    max_idx = torch.argmax(ncc_map)
    return max_idx // w - (h // 2), max_idx % w - (w // 2)


def test_ncc_finds_known_shift():
    img1 = torch.zeros((3, 20, 20))
    img1[0, 5:10, 5:10] = 1


import torch
from alignment_methods import (
    ncc,
    ncc_batched,
    ncc_efficient,
    ncc_with_size_penalty,
    ncc_with_size_preference,
)


def _peak_shift(corr_map: torch.Tensor) -> tuple[int, int]:
    h, w = corr_map.shape
    idx = torch.argmax(corr_map)
    return int(idx // w - (h // 2)), int(idx % w - (w // 2))


def test_ncc_peaks_at_zero_for_identical():
    torch.manual_seed(0)
    img = torch.zeros((3, 16, 16))
    img[0, 4:8, 4:8] = 1.0

    result = ncc(img, img, shift_range=3)

    assert result.shape == (7, 7)
    assert _peak_shift(result) == (0, 0)


def test_ncc_detects_known_shift():
    torch.manual_seed(1)
    img = torch.zeros((3, 20, 20))
    img[1, 5:10, 6:11] = 1.0
    expected_shift = (2, -3)
    shifted = torch.roll(img, shifts=expected_shift, dims=(1, 2))

    result = ncc(img, shifted, shift_range=5)

    assert _peak_shift(result) == expected_shift


def test_ncc_handles_2d_and_stride():
    img = torch.arange(0, 25, dtype=torch.float32).view(5, 5)
    result = ncc(img, img, shift_range=2, stride=2)

    assert result.shape == (3, 3)
    assert _peak_shift(result) == (0, 0)


def test_ncc_batched_matches_individual():
    torch.manual_seed(2)
    batch = torch.randn(2, 3, 12, 12)
    shifts = [(1, 0), (-2, 3)]
    shifted = torch.stack(
        [torch.roll(batch[i], shifts=shifts[i], dims=(1, 2)) for i in range(2)]
    )

    batched = ncc_batched(batch, shifted, shift_range=3)
    singles = torch.stack(
        [ncc_efficient(batch[i], shifted[i], shift_range=3) for i in range(2)]
    )

    assert torch.allclose(batched, singles, atol=1e-4, rtol=1e-4)


def test_size_preferences_return_finite_scores():
    base = torch.zeros((1, 8, 8))
    base[:, 2:6, 2:6] = 1.0
    shifted = torch.roll(base, shifts=(1, 1), dims=(1, 2))

    pref = ncc(base, shifted, shift_range=2, score_func="size_preference")
    pen = ncc(base, shifted, shift_range=2, score_func="size_penalty")

    assert torch.isfinite(pref).all()
    assert torch.isfinite(pen).all()
    assert not torch.allclose(pref, pen)
