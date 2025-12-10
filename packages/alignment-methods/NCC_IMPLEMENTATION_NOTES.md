# NCC Implementation: Normal vs Efficient Version Comparison

## Overview
The NCC (Normalized Cross Correlation) implementation provides two versions:
1. **Normal version** (`ncc`): Uses explicit loops to calculate correlation for each shift
2. **Efficient version** (`ncc_efficient`): Uses convolution operations for faster computation

## Do they produce the same results?

**Short answer**: No, the efficient version does NOT produce mathematically identical results to the normal version, though they are conceptually similar.

## How the algorithms differ:

### Normal Version (ncc):
- Explicitly iterates through each possible shift position
- For each shift, calculates:
  1. Overlapping regions between the two images
  2. Local NCC of those overlapping regions (subtracting local means of each overlapping region)
  3. Applies the score function (default NCC)
- More intuitive and readable
- Slower for large images or large shift ranges

### Efficient Version (ncc_efficient):
- Uses convolution operations to compute all cross-correlations at once
- Leverages PyTorch's optimized convolution functions
- More complex but significantly faster, especially for large images
- Computes normalized cross-correlation using pre-computed normalization factors
- **Important difference**: Does NOT compute local NCC (with local means subtracted for each shift)

## Result Equivalency:

- The efficient version computes a form of normalized cross-correlation, but not the true local NCC
- The normal version computes the true NCC for each overlapping region, where local means are subtracted
- Results will be related but not identical
- The correlation patterns will be similar, but the exact values may differ

## Mathematical Difference:

- **Normal NCC**: For each shift, computes `sum((A - mean(A))*(B - mean(B))) / sqrt(sum((A-mean(A))^2)*sum((B-mean(B))^2))` where means are computed locally for each overlapping region
- **Efficient version**: Computes `sum(A * B) / sqrt(sum(A^2) * sum(shifted_B^2))` where A is globally normalized

## Performance comparison:

The efficient version is significantly faster because:
- Convolution operations are highly optimized in PyTorch
- No Python loops - computations happen in C++/CUDA
- All shifts computed in parallel rather than sequentially

## When to use which:

- Use `ncc` when you need the mathematically correct NCC with local mean subtraction or custom score functions
- Use `ncc_efficient` for better performance when you need fast cross-correlation that's similar to NCC but don't require the exact local mean subtraction
- Use `ncc_batched` for processing multiple image pairs simultaneously for maximum throughput

## Additional Vectorized Implementation:

The library also includes `ncc_batched` which can process batches of image pairs:
- Accepts input tensors with batch dimension (B, C, H, W)
- Processes multiple image pairs in parallel
- Can also handle single image pairs with automatic dimension expansion
- Returns results for all pairs in a single tensor (B, output_height, output_width)

## Updated Score Function Interface:

The main `ncc` function now supports string literals for preset scoring functions:
- `score_func="default"` (overlap-size agnostic NCC - new default)
- `score_func="size_preference"` (weights correlation by overlapping region size)
- `score_func="size_penalty"` (penalizes small overlapping regions)
- `score_func=custom_function` (for backward compatibility with custom callables)

The new default ("default") uses a size-agnostic normalized cross correlation that does not weight results by overlap area.

## Additional Method: Phase Correlation

The library also includes Phase Correlation functions which are highly efficient for detecting pure translations:

- `phase_correlation(img1, img2)` - Computes phase correlation matrix
- `phase_correlation_peak(img1, img2)` - Returns the (y, x) shift and peak value

Phase Correlation is significantly faster than NCC (O(N log N) vs O(N²)) for translation detection and shows superior accuracy for pure translational shifts.

## New Two-Stage Method

The library includes a new two-stage correlation method that combines the speed of Phase Correlation with the accuracy of NCC:

- `two_stage_correlation(img1, img2, shift_range=50, top_k=5, ...)` - Performs Phase Correlation to find top-k candidate shifts, then evaluates NCC on those candidates

This approach provides a good balance between speed and accuracy.

## File Structure

The implementations are split across focused modules:

- `ncc.py` holds NCC variants and scoring helpers.
- `phase.py` contains phase correlation utilities.
- `two_stage.py` combines the two approaches for coarse-to-fine alignment.
