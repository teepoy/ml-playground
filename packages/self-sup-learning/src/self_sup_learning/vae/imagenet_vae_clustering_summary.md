# ImageNet Subset VAE Clustering Analysis

## Overview
This document summarizes the results of training a Variational Autoencoder (VAE) on the ImageNet subset dataset and performing clustering analysis on the resulting embeddings.

## Dataset Information
- **Dataset**: ImageNet subset with 10 classes
- **Total images**: 5,465 (3,818 train, 822 validation, 825 test)
- **Classes**:
  1. European fire salamander (n01629819)
  2. Wallaby, brush kangaroo (n01877812)
  3. Slug (n01945685)
  4. Dowitcher (n02033041)
  5. Komondor (n02105505)
  6. Siberian husky (n02110185)
  7. Tabby, tabby cat (n02123045)
  8. Minibus (n03769881)
  9. Radio, wireless (n04041544)
  10. Reflex camera (n04069434)

## VAE Architecture
- **Input**: 64x64x3 images
- **Latent dimension**: 128
- **Encoder**: CNN with 3 conv layers, outputting mu and logvar
- **Decoder**: Transposed CNN to reconstruct the input
- **Training**: 50 epochs with Adam optimizer

## Clustering Analysis

### Methods Tested
1. K-Means clustering with 1, 5, 10, 20, and 30 clusters
2. Spectral clustering with 1, 5, 10, 20, and 30 clusters
3. HDBSCAN (hierarchical density-based clustering)

### Performance Metrics
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Adjusted Rand Index (ARI)**: Measures similarity between clusterings
- **Normalized Mutual Information (NMI)**: Information-theoretic measure
- **Silhouette Score**: Measures how well-separated clusters are

### Results Summary

| Dataset | Method   | Best Configuration | V-Measure | ARI   | NMI   | Silhouette |
|---------|----------|-------------------|-----------|-------|-------|------------|
| Train   | K-Means  | 30 clusters       | 0.1572    | 0.0716| 0.1572| 0.0000     |
| Test    | K-Means  | 30 clusters       | 0.1595    | 0.0628| 0.1595| 0.0000     |
| Train   | Spectral | 30 clusters       | 0.0477    | 0.0021| 0.0477| 0.0000     |
| Test    | Spectral | 30 clusters       | 0.0669    | -0.0006|0.0669| 0.0000     |
| Train   | HDBSCAN  | Auto              | 0.0301    | -0.0009|0.0301| 0.1982     |
| Test    | HDBSCAN  | Auto              | 0.0000    | 0.0000| 0.0000| 0.0000     |

### Key Findings

1. **K-Means performs best**: Across both train and test sets, K-Means clustering achieved the highest V-Measure scores.

2. **Consistent performance**: The model shows similar performance on both training and test sets, indicating good generalization. The test set actually performed slightly better than the training set (V-Measure of 0.1595 vs 0.1572 for K-Means).

3. **Limited clustering quality**: The V-Measure scores are relatively low (around 0.16 at best), suggesting that the VAE embeddings don't strongly preserve the class structure for clustering purposes. This is expected for complex, high-dimensional image datasets.

4. **HDBSCAN performance**: HDBSCAN identified very few meaningful clusters on the test set (resulting in 0.0000 scores), while finding some structure on the training set.

5. **Optimal cluster count**: For K-Means, the best performance was achieved with 30 clusters (more than the 10 true classes), suggesting the VAE latent space creates sub-clusters within each class.

## Comparison with Iris Dataset Results

| Dataset    | Best Method | Best V-Measure | Best Silhouette | Notes |
|------------|-------------|----------------|-----------------|-------|
| Iris       | Spectral (6 clusters) | 0.6511 | 0.6395 | Simple, low-dimensional dataset |
| ImageNet (this) | K-Means (30 clusters) | 0.1595 | 0.0000 | Complex, high-dimensional images |

### Key Differences
1. **Performance Gap**: The VAE on ImageNet achieved significantly lower clustering performance compared to the Iris dataset, which is expected given the complexity difference.

2. **Method Preference**: On Iris data, Spectral clustering performed best; on ImageNet, K-Means performed best.

3. **Cluster Count**: The best performance on Iris used 6 (2x the true classes), while ImageNet needed 30 (3x the true classes) for best results.

## Conclusion

The VAE successfully learned embeddings for the ImageNet subset, but the clustering performance is limited due to the complexity of the image data compared to low-dimensional tabular data like Iris. The best clustering performance achieved was V-Measure of 0.1595 with K-Means using 30 clusters on the test set.

This demonstrates that while VAEs can learn compressed representations of complex image data, the latent space may not preserve class structure as effectively as needed for simple clustering algorithms. More sophisticated clustering techniques or different VAE architectures might be needed for better results on complex image datasets.
