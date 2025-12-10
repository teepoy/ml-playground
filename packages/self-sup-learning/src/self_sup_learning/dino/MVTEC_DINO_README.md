# MVTec-AD Anomaly Detection using DINO Features

This script performs anomaly detection on the MVTec-AD dataset using DINO (self-supervised learning) features and clustering techniques.

## Approach

The implementation follows these key principles:

1. **Dataset Structure Handling**: The script properly handles the MVTec-AD dataset structure where data is organized as `<background-class>/test/<defect-class>`. Each background class (e.g., "bottle", "cable") is processed separately.

2. **Within-Class Clustering**: For each background class, we perform clustering analysis only within that class, respecting the requirement to cluster results under the same background class separately.

3. **DINO Feature Extraction**: The script extracts features using a DINO model (or a fallback model if the actual DINO model is not available) to capture high-level visual representations.

4. **Clustering Methods**: Multiple clustering algorithms (K-Means, Spectral Clustering, HDBSCAN) are applied with different numbers of clusters to find the best separation between normal and defective samples.

5. **Anomaly Detection Evaluation**: The script evaluates how well normal ("good") samples cluster separately from defective samples, providing an "anomaly detection capability" score.

## Files Generated

- `mvtec_ad_dino_clustering_results.csv`: Overall clustering performance metrics
- `mvtec_ad_dino_detailed_results.csv`: Detailed clustering results with cluster distributions
- `mvtec_ad_anomaly_detection_analysis.csv`: Specialized analysis for anomaly detection performance

## Anomaly Detection Capability Score

This score indicates how well the clustering method separates normal from defective samples:
- **1.0**: Perfect separation (normal and defective samples in completely different clusters)
- **0.0**: Complete mixing (normal and defective samples in the same clusters)

Higher scores indicate better potential for anomaly detection.

## Usage

1. Download the MVTec-AD dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Place it in the expected directory structure
3. Run the script: `python mvtec_dino_classification.py`

## Key Features

- Handles the specific MVTec-AD directory structure
- Processes each background class independently
- Multiple clustering algorithms and cluster numbers
- Detailed evaluation of anomaly detection capabilities
- Comprehensive result reporting

## Limitations

- Requires DINO model setup (falls back to ResNet if unavailable)
- Performance may vary based on dataset size and defect types
- Some clustering methods may fail on high-dimensional features
