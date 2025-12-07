# Quick Start Guide

## Overview
The clustering analysis has been enhanced with:
1. **ARI and NMI metrics** displayed prominently in output
2. **Dual logging** to both console and log files
3. **Split configuration files** for easier experimentation

## Quick Test (Recommended First Step)
Test on 5,000 samples to verify everything works:
```bash
cd /home/jin/Desktop/mm
python playground/clustering/quick_clustering_test.py
```
Output: `clustering_test_results/` (~1-2 minutes)

## Run Single Experiment
Run one specific configuration:
```bash
cd /home/jin/Desktop/mm

# K-means with k=10
python playground/clustering/clustering_analysis.py --config-name=kmeans/kmeans_k10

# VBGMM with 100 components
python playground/clustering/clustering_analysis.py --config-name=vbgmm/vbgmm_n100

# HDBSCAN with specific parameters
python playground/clustering/clustering_analysis.py --config-name=hdbscan/hdbscan_mcs200_ms20
```

Or use the example script:
```bash
./playground/clustering/run_example.sh
```

## Run All Experiments
Run all 20 configurations sequentially:
```bash
cd /home/jin/Desktop/mm
./playground/clustering/run_all_experiments.sh
```
⚠️ **Warning**: This runs on 50,000 samples and may take several hours.

## Understanding the Output

### Console Output
```
================================================================================
MAE ViT Embedding Clustering Analysis
================================================================================
[Configuration details...]

Loading embeddings from LanceDB...
Loaded 50000 embeddings with dimension 768
Using true labels for evaluation (1000 classes)

Running k-means clustering...
  n_clusters=50
    Clustering completed in 12.34s
  Results:
    Internal Metrics:
      Silhouette Score: 0.0612        ← How well-separated clusters are
      Calinski-Harabasz Score: 356.42 ← Cluster density (higher = better)
      Davies-Bouldin Score: 2.54      ← Cluster separation (lower = better)
    External Metrics (vs Ground Truth):
      Adjusted Rand Index: 0.0134     ← Agreement with true labels (0-1)
      Normalized Mutual Info: 0.3521  ← Mutual information (0-1)
      V-Measure: 0.3498               ← Harmonic mean of homogeneity & completeness
```

### Files Generated
In the output directory (default: `clustering_results/`):
- `clustering_log_YYYYMMDD_HHMMSS.txt` - Complete log of the run
- `clustering_metrics_YYYYMMDD_HHMMSS.json` - All metrics in JSON format
- `clustering_metrics_YYYYMMDD_HHMMSS.csv` - Metrics in CSV format
- `predictions/` - Cluster predictions for each method
- `visualizations/` - UMAP/t-SNE plots (if enabled)

## Configuration Files

### Available Configs
```
configs/
├── base_config.yaml              # Shared settings
├── kmeans/
│   ├── kmeans_k10.yaml          # k=10
│   ├── kmeans_k50.yaml          # k=50
│   ├── kmeans_k100.yaml         # k=100
│   ├── kmeans_k200.yaml         # k=200
│   ├── kmeans_k500.yaml         # k=500
│   └── kmeans_k1000.yaml        # k=1000
├── vbgmm/
│   ├── vbgmm_n50.yaml           # n_components=50
│   ├── vbgmm_n100.yaml          # n_components=100
│   ├── vbgmm_n200.yaml          # n_components=200
│   ├── vbgmm_n500.yaml          # n_components=500
│   └── vbgmm_n1000.yaml         # n_components=1000
└── hdbscan/
    ├── hdbscan_mcs100_ms10.yaml  # min_cluster_size=100, min_samples=10
    ├── hdbscan_mcs100_ms20.yaml
    ├── hdbscan_mcs100_ms50.yaml
    ├── hdbscan_mcs200_ms10.yaml
    ├── hdbscan_mcs200_ms20.yaml
    ├── hdbscan_mcs200_ms50.yaml
    ├── hdbscan_mcs500_ms10.yaml
    ├── hdbscan_mcs500_ms20.yaml
    └── hdbscan_mcs500_ms50.yaml
```

### Customize Parameters
### Override Parameters
```bash
# Change output directory (use ++ to force override)
python playground/clustering/clustering_analysis.py \
    --config-name=kmeans/kmeans_k50 \
    ++output.results_dir=./my_custom_results

# Disable visualization to speed up
python playground/clustering/clustering_analysis.py \
    --config-name=vbgmm/vbgmm_n100 \
    ++visualization.enabled=false

# Change database path
python playground/clustering/clustering_analysis.py \
    --config-name=hdbscan/hdbscan_mcs200_ms20 \
    ++database.path=/path/to/other/lancedb
```

## Analyze Results
After running experiments, analyze the results:
```bash
python playground/clustering/analyze_clustering_results.py clustering_results/
```

This will:
- Load all metrics from JSON files
- Create comparison plots
- Generate a summary report
- Identify best-performing configurations

## Tips

### Performance Expectations
- **Quick test (5K samples)**: 1-2 minutes per method
- **Full run (50K samples)**: 10-30 minutes per configuration
- **All 20 configs**: 3-10 hours total

### Best Practices
1. **Start with quick test**: Verify setup before full run
2. **Run single config first**: Test one config file to check output
3. **Monitor first experiment**: Watch logs for any issues
4. **Check disk space**: Results with visualizations can be large (~1-2 GB)

### Common Issues
- **VBGMM collapses to 1 cluster**: Normal for high-dimensional data
- **HDBSCAN finds 0 clusters**: Try smaller min_cluster_size/min_samples
- **Low ARI/NMI scores**: Expected for unsupervised on complex data
- **Long runtime**: Consider using subset of data or disabling visualization

### Interpretation Guide
- **ARI < 0.1**: Poor agreement with ground truth
- **ARI 0.1-0.3**: Moderate agreement
- **ARI > 0.3**: Good agreement
- **NMI > 0.5**: Strong mutual information with true labels

## Documentation Files
- `UPDATE_SUMMARY.md` - What changed in this update
- `CONFIG_GUIDE.md` - Detailed configuration documentation
- `CLUSTERING_README.md` - Clustering methodology overview
- `QUICKSTART.md` - This file

## Troubleshooting
If you encounter issues:
1. Check the log file in the output directory
2. Verify LanceDB path: `/home/jin/Desktop/mm/lancedb`
3. Ensure embeddings exist: `lancedb/imagenet_mae_embeddings.lance`
4. Check Python environment has all dependencies installed
