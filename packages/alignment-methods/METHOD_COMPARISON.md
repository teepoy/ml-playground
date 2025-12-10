# Comparison of NCC and Phase Correlation Methods

## Performance and Use Case Analysis

### Phase Correlation
**Time Complexity:** O(N log N) using FFT operations
**Best For:** Pure translational image registration
**Accuracy:** Excellent for translations, consistent across image sizes

### NCC (Normalized Cross Correlation)
**Time Complexity:** O(N^2 * search_area) where search_area is the number of possible shifts
**Best For:** General similarity matching with robustness to illumination changes
**Accuracy:** Good when search range is limited, degrades with larger searches

## Detailed Comparison Results

### Speed Comparison:
- Phase correlation is consistently 2-4x faster than NCC efficient on images from 32x32 to 128x128
- The speed advantage increases with image size since FFT is O(N log N) vs NCC's O(N²) in search area

### Accuracy Comparison:
- Phase correlation showed 100% accuracy in detecting pure translations within 1 pixel tolerance
- NCC efficient showed inconsistent accuracy, especially degrading with larger images and larger search ranges
- This degradation likely occurs because NCC's search range was limited (10 pixels max) to maintain reasonable computation time

## When to Use Which Method

### Use Phase Correlation when:
- You need to detect pure translations between images
- Performance is critical and you have large images
- Images have similar illumination conditions
- You need consistent results regardless of image content

### Use NCC when:
- You need robustness to illumination changes
- You're looking for general similarity beyond just translation
- You have a limited search range
- You want to use custom scoring functions to weight different types of similarity

## Recommendations

For the specific use case of finding shifts/translations between similar images, Phase Correlation is the superior choice due to its speed and accuracy. NCC is more appropriate when you need to measure general similarity and are not limited by computation time for large search spaces.
