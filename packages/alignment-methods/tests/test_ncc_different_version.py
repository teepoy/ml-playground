import torch
from alignment_methods import ncc, ncc_efficient, phase_correlation


def test_invalid_shift_range_raises_value_error():
    img = torch.ones((3, 10, 10))
    try:
        ncc(img, img, shift_range=10)
    except ValueError as exc:
        assert "Shift range too large" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized shift range")


def test_phase_correlation_supports_2d_inputs():
    img = torch.zeros((12, 12))
    img[3:7, 4:8] = 1.0
    shifted = torch.roll(img, shifts=(1, -1), dims=(0, 1))

    corr = phase_correlation(img, shifted)
    assert corr.shape == (12, 12)


def test_ncc_efficient_matches_shape():
    img = torch.zeros((1, 16, 16))
    img[:, 5:9, 6:10] = 1.0
    shifted = torch.roll(img, shifts=(2, 2), dims=(1, 2))

    standard = ncc(img, shifted, shift_range=3)
    efficient = ncc_efficient(img, shifted, shift_range=3)

    assert standard.shape == efficient.shape
