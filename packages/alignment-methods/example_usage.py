"""
Example Usage: NCC and Phase Correlation Methods

This file demonstrates how to use the implemented correlation methods for different use cases.
"""

import sys
import time
from pathlib import Path

import torch

# Ensure local src is available when running the script directly
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment_methods import (
    ncc,
    ncc_batched,
    ncc_efficient,
    ncc_with_size_penalty,
    ncc_with_size_preference,
    phase_correlation,
    phase_correlation_peak,
)


def example_translation_detection():
    """Example: Detecting pure translation between images using both methods."""
    print("Example 1: Translation Detection")
    print("-" * 40)

    # Create a base image
    img1 = torch.zeros((3, 64, 64))
    img1[0, 10:20, 10:20] = 1.0  # Bright square in channel 0
    img1[1, 15:25, 15:25] = 0.7  # Different pattern in channel 1
    img1[2, 20:30, 20:30] = 0.5  # Different pattern in channel 2

    # Create a shifted version (true shift: +5 in y, +8 in x)
    img2 = torch.roll(img1, shifts=(5, 8), dims=(1, 2))

    print(f"Image size: {img1.shape}")
    print("True translation: (5, 8)")

    # Method 1: Phase Correlation (fast and accurate for translations)
    start_time = time.time()
    pc_y, pc_x, pc_val = phase_correlation_peak(img1, img2)
    pc_time = time.time() - start_time
    print(
        f"Phase Correlation: shift=({pc_y}, {pc_x}), value={pc_val:.3f}, time={pc_time:.4f}s"
    )

    # Method 2: NCC Efficient (good for limited search ranges)
    start_time = time.time()
    ncc_result = ncc_efficient(img1, img2, shift_range=10)  # Search within 10 pixels
    ncc_time = time.time() - start_time

    # Find peak in NCC result
    max_idx = torch.argmax(ncc_result)
    h, w = ncc_result.shape
    ncc_y = max_idx // w - (h // 2)  # Convert to actual shift
    ncc_x = max_idx % w - (w // 2)  # Convert to actual shift
    ncc_val = ncc_result.flatten()[max_idx].item()
    print(
        f"NCC Efficient: shift=({ncc_y}, {ncc_x}), value={ncc_val:.3f}, time={ncc_time:.4f}s"
    )

    print()


def example_batch_processing():
    """Example: Batch processing multiple image pairs."""
    print("Example 2: Batch Processing")
    print("-" * 40)

    # Create batch of image pairs with different shifts
    batch_size = 4
    imgs1 = torch.randn((batch_size, 3, 32, 32))

    # Apply different shifts to create second batch
    shifts = [(2, 3), (-1, 4), (3, -2), (0, 5)]
    imgs2 = torch.zeros_like(imgs1)
    for i, (dy, dx) in enumerate(shifts):
        imgs2[i] = torch.roll(imgs1[i], shifts=(dy, dx), dims=(1, 2))

    print(f"Processing batch of {batch_size} image pairs")
    print(f"True shifts: {shifts}")

    # Process with batched NCC
    start_time = time.time()
    batch_result = ncc_batched(imgs1, imgs2, shift_range=5)
    batch_time = time.time() - start_time
    print(f"Batched NCC time: {batch_time:.4f}s")
    print(f"Batch result shape: {batch_result.shape}")

    # Process with phase correlation (would need a loop for batch processing since
    # there's no batched version yet)
    start_time = time.time()
    pc_results = []
    for i in range(batch_size):
        shift_y, shift_x, val = phase_correlation_peak(imgs1[i], imgs2[i])
        pc_results.append((shift_y, shift_x, val))
    pc_time = time.time() - start_time
    print(f"Phase Correlation (loop) time: {pc_time:.4f}s")
    print(f"Phase Correlation results: {[(r[0], r[1]) for r in pc_results]}")

    print()


def example_custom_scoring():
    """Example: Using different scoring functions for NCC."""
    print("Example 3: Custom Scoring Functions")
    print("-" * 40)

    # Create images where overlap size might matter
    img1 = torch.zeros((1, 50, 50))
    img1[0, 10:30, 10:30] = 1.0  # Large pattern

    img2 = torch.zeros((1, 50, 50))
    img2[0, 15:25, 15:25] = 1.0  # Smaller overlapping pattern

    print("Comparing different scoring functions:")

    # Default (overlap-size agnostic)
    result_default = ncc(img1, img2, shift_range=5, score_func="default")
    max_idx = torch.argmax(result_default)
    h, w = result_default.shape
    y = max_idx // w - (h // 2)
    x = max_idx % w - (w // 2)
    val = result_default.flatten()[max_idx].item()
    print(f"Default: shift=({y}, {x}), value={val:.3f}")

    # Size preference
    result_pref = ncc(img1, img2, shift_range=5, score_func="size_preference")
    max_idx = torch.argmax(result_pref)
    y = max_idx // w - (h // 2)
    x = max_idx % w - (w // 2)
    val = result_pref.flatten()[max_idx].item()
    print(f"Size Preference: shift=({y}, {x}), value={val:.3f}")

    # Size penalty
    result_penalty = ncc(img1, img2, shift_range=5, score_func="size_penalty")
    max_idx = torch.argmax(result_penalty)
    y = max_idx // w - (h // 2)
    x = max_idx % w - (w // 2)
    val = result_penalty.flatten()[max_idx].item()
    print(f"Size Penalty: shift=({y}, {x}), value={val:.3f}")

    # Custom function
    result_custom = ncc(img1, img2, shift_range=5, score_func=ncc_with_size_preference)
    max_idx = torch.argmax(result_custom)
    y = max_idx // w - (h // 2)
    x = max_idx % w - (w // 2)
    val = result_custom.flatten()[max_idx].item()
    print(f"Custom Function: shift=({y}, {x}), value={val:.3f}")

    print()


def example_performance_comparison():
    """Example: Performance comparison on different image sizes."""
    print("Example 4: Performance Comparison")
    print("-" * 40)

    sizes = [(32, 32), (64, 64)]
    img1 = torch.randn((1, 64, 64))  # Create largest first, then crop
    img2 = torch.roll(img1, shifts=(3, 4), dims=(1, 2))  # Always same shift

    for h, w in sizes:
        # Crop to current size
        img1_crop = img1[:, :h, :w]
        img2_crop = img2[:, :h, :w]

        print(f"Image size: {h}x{w}")

        # Phase correlation
        start_time = time.time()
        phase_correlation_peak(img1_crop, img2_crop)
        pc_time = time.time() - start_time

        # NCC efficient with limited range
        start_time = time.time()
        ncc_efficient(img1_crop, img2_crop, shift_range=8)
        ncc_time = time.time() - start_time

        print(f"  Phase Correlation: {pc_time:.6f}s")
        print(f"  NCC Efficient:     {ncc_time:.6f}s")
        print(f"  Speedup:           {ncc_time/pc_time:.1f}x")
        print()


if __name__ == "__main__":
    print("NCC and Phase Correlation Usage Examples")
    print("=" * 50)

    example_translation_detection()
    example_batch_processing()
    example_custom_scoring()
    example_performance_comparison()

    print("For detailed comparison of methods, see METHOD_COMPARISON.md")
