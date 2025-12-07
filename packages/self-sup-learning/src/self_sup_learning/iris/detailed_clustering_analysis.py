#!/usr/bin/env python3
"""
Detailed analysis and visualization of clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


def load_and_analyze_results():
    """Load clustering results and create detailed analysis"""

    # Load the results
    results_df = pd.read_csv("clustering_results.csv")

    # Load the original data and embeddings
    iris = load_iris()
    embeddings = np.load("iris_embeddings.npy")

    # Parse the labels from string representation using ast to avoid eval
    import ast

    results_df["labels"] = results_df["labels"].apply(
        lambda x: np.array(ast.literal_eval(x.replace("\n", "").replace("  ", " ")))
    )

    print("Clustering Results Analysis")
    print("=" * 60)

    # Find best clustering result for each method
    methods = ["KMeans", "Spectral", "HDBSCAN"]

    for method in methods:
        method_results = results_df[
            results_df["method_n_clusters"].str.startswith(method)
        ]
        if len(method_results) > 0:
            best_v_measure_idx = method_results["v_measure"].idxmax()
            best_result = method_results.loc[best_v_measure_idx]
            print(f"\\nBest {method} result:")
            print(f"  Configuration: {best_result['method_n_clusters']}")
            print(f"  V-Measure: {best_result['v_measure']:.4f}")
            print(f"  Silhouette: {best_result['silhouette']:.4f}")
            print(f"  ARI: {best_result['ari']:.4f}")
            print(f"  NMI: {best_result['nmi']:.4f}")

    # Create detailed comparison plots
    create_comparison_plots(results_df, embeddings, iris)

    return results_df


def create_comparison_plots(results_df, embeddings, iris):
    """Create detailed comparison plots"""

    # Prepare data for plotting
    plot_data = []
    for idx, row in results_df.iterrows():
        method_n_clusters = row["method_n_clusters"]
        labels = np.array(row["labels"])
        silhouette = row["silhouette"]
        v_measure = row["v_measure"]
        ari = row["ari"]
        nmi = row["nmi"]

        for i, (emb, true_label) in enumerate(zip(embeddings, iris.target)):
            plot_data.append(
                {
                    "method_n_clusters": method_n_clusters,
                    "embedding_1": emb[0],
                    "embedding_2": emb[1],
                    "predicted_label": labels[i],
                    "true_label": iris.target_names[true_label],
                    "silhouette": silhouette,
                    "v_measure": v_measure,
                    "ari": ari,
                    "nmi": nmi,
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Silhouette scores by method
    sns.boxplot(data=plot_df, x="method_n_clusters", y="silhouette", ax=axes[0, 0])
    axes[0, 0].set_title("Silhouette Score by Method and Cluster Count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Plot 2: V-Measure by method
    sns.boxplot(data=plot_df, x="method_n_clusters", y="v_measure", ax=axes[0, 1])
    axes[0, 1].set_title("V-Measure by Method and Cluster Count")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: ARI by method
    sns.boxplot(data=plot_df, x="method_n_clusters", y="ari", ax=axes[1, 0])
    axes[1, 0].set_title("Adjusted Rand Index by Method and Cluster Count")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: NMI by method
    sns.boxplot(data=plot_df, x="method_n_clusters", y="nmi", ax=axes[1, 1])
    axes[1, 1].set_title("Normalized Mutual Information by Method and Cluster Count")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("clustering_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Create scatter plots for the best clustering results
    methods = ["KMeans", "Spectral", "HDBSCAN"]
    best_results = {}

    for method in methods:
        method_results = plot_df[plot_df["method_n_clusters"].str.startswith(method)]
        if len(method_results) > 0:
            best_v_measure_row = method_results.loc[
                method_results["v_measure"].idxmax()
            ]
            best_results[method] = best_v_measure_row["method_n_clusters"]

    # Create a subplot for each best method
    fig, axes = plt.subplots(1, len(best_results), figsize=(15, 5))
    if len(best_results) == 1:
        axes = [axes]

    for idx, (method, config) in enumerate(best_results.items()):
        method_data = plot_df[plot_df["method_n_clusters"] == config]

        axes[idx].scatter(
            method_data["embedding_1"],
            method_data["embedding_2"],
            c=method_data["predicted_label"],
            cmap="viridis",
            alpha=0.7,
        )
        axes[idx].set_title(
            f'{method} Best Result\\n(V-Measure: {method_data["v_measure"].iloc[0]:.3f})'
        )
        axes[idx].set_xlabel("Embedding Dimension 1")
        axes[idx].set_ylabel("Embedding Dimension 2")

    plt.tight_layout()
    plt.savefig("best_clustering_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def compare_with_original_features():
    """Compare clustering results using original features vs embeddings"""
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    from sklearn.metrics import v_measure_score
    from sklearn.preprocessing import StandardScaler

    print("\\n" + "=" * 60)
    print("Comparison: Original Features vs VAE Embeddings")
    print("=" * 60)

    # Load original iris data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize original features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load embeddings
    embeddings = np.load("iris_embeddings.npy")

    # Perform K-means with 3 clusters on both representations
    kmeans_orig = KMeans(n_clusters=3, random_state=42)
    kmeans_embed = KMeans(n_clusters=3, random_state=42)

    labels_orig = kmeans_orig.fit_predict(X_scaled)
    labels_embed = kmeans_embed.fit_predict(embeddings)

    # Calculate metrics
    v_measure_orig = v_measure_score(y, labels_orig)
    v_measure_embed = v_measure_score(y, labels_embed)

    print(f"V-Measure with original features: {v_measure_orig:.4f}")
    print(f"V-Measure with VAE embeddings:   {v_measure_embed:.4f}")

    if v_measure_embed > v_measure_orig:
        print("VAE embeddings performed better than original features for clustering!")
    elif v_measure_embed < v_measure_orig:
        print("Original features performed better than VAE embeddings for clustering.")
    else:
        print("VAE embeddings and original features performed equally for clustering.")


def main():
    print("Loading and analyzing clustering results...")
    results_df = load_and_analyze_results()

    print("\\nCreating detailed comparison analysis...")
    compare_with_original_features()

    print(
        "\\nAnalysis complete! Check the generated plots and CSV file for detailed results."
    )


if __name__ == "__main__":
    main()
