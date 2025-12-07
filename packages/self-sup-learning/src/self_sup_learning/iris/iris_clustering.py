#!/usr/bin/env python3
"""
Script to extract embeddings from the trained VAE model, perform clustering,
and evaluate clustering results against ground truth.
"""


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.datasets import load_iris
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


class IrisVAE(torch.nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(IrisVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
        )

        # Latent space
        self.fc_mu = torch.nn.Linear(8, latent_dim)
        self.fc_logvar = torch.nn.Linear(8, latent_dim)

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, input_dim),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def load_model_and_data():
    """Load the trained VAE model and iris dataset"""
    # Load the trained model
    model = IrisVAE(input_dim=4, latent_dim=2)
    model.load_state_dict(
        torch.load("iris_vae_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()

    # Load the scaler
    scaler = joblib.load("iris_scaler.pkl")

    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Normalize the features
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    return model, scaler, X_tensor, X, y, iris.target_names


def extract_embeddings(model, X_tensor):
    """Extract embeddings from the trained VAE model"""
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        # Use mean as the embedding
        embeddings = mu.numpy()
    return embeddings


def perform_clustering(embeddings, n_clusters_list, y_true, class_names):
    """Perform clustering and evaluate results"""
    results = {}

    clustering_methods = {
        "KMeans": lambda n: KMeans(n_clusters=n, random_state=42).fit(embeddings),
        "Spectral": lambda n: SpectralClustering(
            n_clusters=n, random_state=42, affinity="rbf"
        ).fit(embeddings),
    }

    # Note: HDBSCAN doesn't take n_clusters as a parameter, so handle it separately
    hdbscan_cluster = HDBSCAN(min_cluster_size=5).fit(embeddings)

    # For each clustering method and number of clusters
    for method_name, cluster_func in clustering_methods.items():
        for n_clusters in n_clusters_list:
            # Perform clustering
            labels = cluster_func(n_clusters).labels_

            # Evaluate clustering
            silhouette = (
                silhouette_score(embeddings, labels)
                if len(np.unique(labels)) > 1
                else 0
            )
            v_measure = v_measure_score(y_true, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)

            results[f"{method_name}_{n_clusters}"] = {
                "labels": labels,
                "silhouette": silhouette,
                "v_measure": v_measure,
                "ari": ari,
                "nmi": nmi,
            }

    # Special handling for HDBSCAN
    hdbscan_labels = hdbscan_cluster.labels_
    # Only evaluate if we have more than one cluster (excluding noise points)
    if len(np.unique(hdbscan_labels[hdbscan_labels != -1])) > 1:
        # For silhouette score, exclude noise points
        non_noise_mask = hdbscan_labels != -1
        if (
            np.sum(non_noise_mask) > 1
            and len(np.unique(hdbscan_labels[non_noise_mask])) > 1
        ):
            silhouette = silhouette_score(
                embeddings[non_noise_mask], hdbscan_labels[non_noise_mask]
            )
        else:
            silhouette = 0

        # For other metrics, we can include noise as a separate class
        v_measure = v_measure_score(y_true, hdbscan_labels)
        ari = adjusted_rand_score(y_true, hdbscan_labels)
        nmi = normalized_mutual_info_score(y_true, hdbscan_labels)

        results["HDBSCAN_any"] = {
            "labels": hdbscan_labels,
            "silhouette": silhouette,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
        }
    else:
        # If HDBSCAN didn't find any clusters, create a single cluster
        results["HDBSCAN_any"] = {
            "labels": np.zeros(len(embeddings), dtype=int),
            "silhouette": 0,
            "v_measure": v_measure_score(y_true, np.zeros(len(embeddings), dtype=int)),
            "ari": adjusted_rand_score(y_true, np.zeros(len(embeddings), dtype=int)),
            "nmi": normalized_mutual_info_score(
                y_true, np.zeros(len(embeddings), dtype=int)
            ),
        }

    return results


def plot_clustering_results(embeddings, results, y_true, class_names):
    """Plot clustering results"""
    n_clusters_list = [1, 2, 3, 6]  # 1x, 2x, 3x the real classes (3)

    # Create a subplot for each clustering method and class number
    methods = ["KMeans", "Spectral", "HDBSCAN"]
    fig, axes = plt.subplots(len(methods), len(n_clusters_list), figsize=(20, 12))

    for i, method in enumerate(methods):
        for j, n_clusters in enumerate(n_clusters_list):
            if method == "HDBSCAN":
                # HDBSCAN doesn't take n_clusters as parameter, so just use the same result
                key = f"{method}_any"
            else:
                key = f"{method}_{n_clusters}"

            if key in results:
                labels = results[key]["labels"]

                ax = axes[i, j]
                scatter = ax.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    c=labels,
                    cmap="viridis",
                    alpha=0.7,
                )
                ax.set_title(
                    f'{method} with {n_clusters if method != "HDBSCAN" else "auto"} clusters'
                )
                ax.set_xlabel("Embedding Dimension 1")
                ax.set_ylabel("Embedding Dimension 2")

    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Also plot ground truth
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=y_true, cmap="viridis", alpha=0.7
    )
    plt.title("Ground Truth Labels")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.colorbar(scatter)
    classes = class_names
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=plt.cm.viridis(i / 2),
            markersize=8,
        )
        for i in range(3)
    ]
    plt.legend(handles, classes)
    plt.savefig("ground_truth.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_results_table(results):
    """Print clustering results in a formatted table"""
    print("\\n" + "=" * 100)
    print("Clustering Results Summary")
    print("=" * 100)
    print(
        f"{'Method':<15} {'n_clusters':<10} {'Silhouette':<12} {'V-Measure':<12} {'ARI':<12} {'NMI':<12}"
    )
    print("-" * 100)

    for key, result in results.items():
        method_parts = key.split("_")
        method = method_parts[0]
        n_clusters = method_parts[1] if len(method_parts) > 1 else "auto"

        print(
            f"{method:<15} {n_clusters:<10} {result['silhouette']:<12.4f} {result['v_measure']:<12.4f} {result['ari']:<12.4f} {result['nmi']:<12.4f}"
        )

    print("=" * 100)


def main():
    print("Loading model and data...")
    model, scaler, X_tensor, X_orig, y_true, class_names = load_model_and_data()

    print("Extracting embeddings from VAE model...")
    embeddings = extract_embeddings(model, X_tensor)
    print(f"Extracted embeddings with shape: {embeddings.shape}")

    # Define class numbers: 1x (3), 2x (6), 3x (9) as well as 1 class for baseline
    n_clusters_list = [1, 3, 6, 9]

    print("Performing clustering...")
    results = perform_clustering(embeddings, n_clusters_list, y_true, class_names)

    print_results_table(results)

    print("\\nPlotting clustering results...")
    plot_clustering_results(embeddings, results, y_true, class_names)

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            "method_n_clusters": list(results.keys()),
            "silhouette": [results[k]["silhouette"] for k in results.keys()],
            "v_measure": [results[k]["v_measure"] for k in results.keys()],
            "ari": [results[k]["ari"] for k in results.keys()],
            "nmi": [results[k]["nmi"] for k in results.keys()],
            "labels": [results[k]["labels"] for k in results.keys()],
        }
    )

    # Convert numpy array labels to lists for proper CSV saving
    results_df_copy = results_df.copy()
    results_df_copy["labels"] = results_df_copy["labels"].apply(lambda x: x.tolist())
    results_df_copy.to_csv("clustering_results.csv", index=False)
    print("\\nResults saved to clustering_results.csv")

    # Save embeddings for future use
    np.save("iris_embeddings.npy", embeddings)
    print("Embeddings saved to iris_embeddings.npy")


if __name__ == "__main__":
    main()
