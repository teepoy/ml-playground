# Clustering Analysis Update Summary

## Changes Made

### 1. Enhanced Metrics Display with ARI and NMI
- **Added to clustering_analysis.py**: The script now displays Adjusted Rand Index (ARI) and Normalized Mutual Info (NMI) metrics prominently in the console output
- **Metrics displayed for all methods**: k-means, VBGMM, and HDBSCAN
- **Two-tier display**:
  - **Internal Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin (don't require ground truth)
  - **External Metrics**: ARI, NMI, V-Measure (computed against ImageNet ground truth labels)

### 2. Dual Logging (File + Console)
- **TeeLogger class**: New custom logger that writes to both file and console simultaneously
- **Automatic log file creation**: Each run creates a timestamped log file: `clustering_log_{timestamp}.txt`
- **Location**: Logs saved in the output directory specified in config (default: `./clustering_results/`)
- **Real-time streaming**: Console shows output as it's generated, file captures complete session
- **Clean shutdown**: Logger properly closes at end of script

### 3. Split Configuration Files
- **Modular structure**: Each config file contains 1 method + 1 parameter set
- **Base config**: `configs/base_config.yaml` contains shared settings (database, output, evaluation, visualization)
- **Method directories**:
  - `configs/kmeans/` - 6 configs (k=10, 50, 100, 200, 500, 1000)
  - `configs/vbgmm/` - 5 configs (n_components=50, 100, 200, 500, 1000)
  - `configs/hdbscan/` - 9 configs (3 min_cluster_size × 3 min_samples combinations)
- **Total**: 20 individual config files for easy experimentation

## Usage Examples

### Run Individual Experiments
```bash
# K-means with k=50
python playground/clustering/clustering_analysis.py --config-name=kmeans/kmeans_k50

# VBGMM with 100 components
python playground/clustering/clustering_analysis.py --config-name=vbgmm/vbgmm_n100

# HDBSCAN with specific parameters
python playground/clustering/clustering_analysis.py --config-name=hdbscan/hdbscan_mcs200_ms20
```

### Run All Configs for One Method
```bash
# All k-means experiments
cd /home/jin/Desktop/mm
for k in 10 50 100 200 500 1000; do
    python playground/clustering/clustering_analysis.py --config-name=kmeans/kmeans_k${k}
done

# All VBGMM experiments
for n in 50 100 200 500 1000; do
    python playground/clustering/clustering_analysis.py --config-name=vbgmm/vbgmm_n${n}
done

# All HDBSCAN experiments
for mcs in 100 200 500; do
    for ms in 10 20 50; do
        python playground/clustering/clustering_analysis.py --config-name=hdbscan/hdbscan_mcs${mcs}_ms${ms}
    done
done
```

### Override Parameters
```bash
# Change output directory
python playground/clustering/clustering_analysis.py --config-name=kmeans/kmeans_k10 output.results_dir=./my_results

# Disable visualization to speed up
python playground/clustering/clustering_analysis.py --config-name=vbgmm/vbgmm_n50 visualization.enabled=false
```

## Output Example

When you run clustering now, you'll see:
```
================================================================================
MAE ViT Embedding Clustering Analysis
================================================================================
[Config YAML content...]

Loading embeddings from LanceDB...
  Database: /home/jin/Desktop/mm/lancedb
  Table: imagenet_mae_embeddings
Loaded 50000 embeddings with dimension 768
Using true labels for evaluation (1000 classes)

Running k-means clustering...
  n_clusters=50
    Clustering completed in 12.34s
  Results:
    Internal Metrics:
      Silhouette Score: 0.0612
      Calinski-Harabasz Score: 356.42
      Davies-Bouldin Score: 2.54
    External Metrics (vs Ground Truth):
      Adjusted Rand Index: 0.0134
      Normalized Mutual Info: 0.3521
      V-Measure: 0.3498

================================================================================
Clustering analysis complete!
Results saved to: ./clustering_results
Log saved to: ./clustering_results/clustering_log_20240115_143022.txt
================================================================================
```

## Files Modified/Created

### Modified:
- `/home/jin/Desktop/mm/playground/clustering/clustering_analysis.py`
  - Added TeeLogger class for dual logging
  - Enhanced metrics display with structured output
  - Added log file management

### Created:
- `/home/jin/Desktop/mm/playground/clustering/configs/base_config.yaml`
- `/home/jin/Desktop/mm/playground/clustering/configs/kmeans/kmeans_k*.yaml` (6 files)
- `/home/jin/Desktop/mm/playground/clustering/configs/vbgmm/vbgmm_n*.yaml` (5 files)
- `/home/jin/Desktop/mm/playground/clustering/configs/hdbscan/hdbscan_mcs*_ms*.yaml` (9 files)
- `/home/jin/Desktop/mm/playground/clustering/CONFIG_GUIDE.md` - Comprehensive usage guide

## Key Features

1. **Better Metrics Visibility**: ARI and NMI now prominently displayed in console output
2. **Persistent Logs**: Full run details saved to timestamped log files
3. **Real-time Progress**: Console shows output as it happens
4. **Modular Configuration**: Easy to run individual experiments
5. **Batch Processing**: Simple shell loops to run multiple configs
6. **Backward Compatible**: Original `clustering_config.yaml` still works

## Quick Reference

### Metrics Interpretation:
- **Silhouette Score**: Higher is better (range: -1 to 1)
- **Calinski-Harabasz**: Higher is better (no fixed range)
- **Davies-Bouldin**: Lower is better (≥0)
- **ARI**: Higher is better (range: -1 to 1, 0=random, 1=perfect)
- **NMI**: Higher is better (range: 0 to 1, 0=no agreement, 1=perfect)
- **V-Measure**: Higher is better (range: 0 to 1)

### Quick Test vs Full Run:
- **quick_clustering_test.py**: 5,000 samples, fast testing (~1-2 min)
- **clustering_analysis.py**: 50,000 samples, comprehensive results (~10-30 min per method)

## Documentation:
- See `CONFIG_GUIDE.md` for detailed configuration documentation
- See `CLUSTERING_README.md` for clustering methodology overview
