import torch
from alignment_methods import ncc_efficient, phase_correlation, phase_correlation_peak


def test_phase_correlation_detects_shift_multi_channel():
    img1 = torch.zeros((3, 24, 24))
    img1[0, 4:10, 4:10] = 1
    img1[1, 8:14, 8:14] = 0.6
    img1[2, 12:18, 10:16] = 0.3

    shift = (3, -2)
    img2 = torch.roll(img1, shifts=shift, dims=(1, 2))

    y, x, _ = phase_correlation_peak(img1, img2)
    assert abs(y - shift[0]) <= 1
    assert abs(x - shift[1]) <= 1


def test_phase_correlation_vs_ncc_align():
    torch.manual_seed(0)
    img1 = torch.randn((3, 32, 32))
    shift = (-2, 4)
    img2 = torch.roll(img1, shifts=shift, dims=(1, 2))

    y_pc, x_pc, _ = phase_correlation_peak(img1, img2)
    ncc_map = ncc_efficient(img1, img2, shift_range=6)
    h, w = ncc_map.shape
    max_idx = torch.argmax(ncc_map)
    y_ncc = max_idx // w - (h // 2)
    x_ncc = max_idx % w - (w // 2)

    assert abs(y_pc - shift[0]) <= 1
    assert abs(x_pc - shift[1]) <= 1
    assert abs(y_ncc - shift[0]) <= 1
    assert abs(x_ncc - shift[1]) <= 2


def test_phase_correlation_invalid_shapes_raises():
    img1 = torch.zeros((3, 10, 10))
    img2 = torch.zeros((3, 10, 11))
    try:
        phase_correlation(img1, img2)
    except ValueError as exc:
        assert "same dimensions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")
