#!/usr/bin/env python3
"""
Script to extract embeddings from the trained ImageNet VAE model,
perform clustering on both training and test sets, and evaluate results
"""


import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class ImageNetVAE(torch.nn.Module):
    def __init__(self, input_channels=3, image_size=64, latent_dim=128):
        super(ImageNetVAE, self).__init__()

        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 4  # After 2 pooling operations

        # Encoder
        self.encoder = torch.nn.Sequential(
            # Input: 3 x 64 x 64
            torch.nn.Conv2d(
                input_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # 32 x 32 x 32
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 8 x 8
            torch.nn.ReLU(),
        )

        # Calculate the size after encoder
        self.encoded_features = (
            128 * (self.feature_size // 2) * (self.feature_size // 2)
        )

        # Latent space
        self.fc_mu = torch.nn.Linear(self.encoded_features, latent_dim)
        self.fc_logvar = torch.nn.Linear(self.encoded_features, latent_dim)

        # Decoder
        self.fc_decode = torch.nn.Linear(latent_dim, self.encoded_features)
        self.decoder = torch.nn.Sequential(
            # Input: 128 x 8 x 8 (after fc_decode reshape)
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 32 x 32 x 32
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32, input_channels, kernel_size=4, stride=2, padding=1
            ),  # 3 x 64 x 64
            torch.nn.Sigmoid(),  # Sigmoid to keep values between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.encoded_features)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, self.feature_size // 2, self.feature_size // 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


def extract_embeddings(model, data_loader, device="cpu"):
    """Extract embeddings from the trained VAE model"""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            # Use mean as the embedding
            batch_embeddings = mu.cpu().numpy()
            embeddings.extend(batch_embeddings)
            labels.extend(batch_labels.numpy())

    return np.array(embeddings), np.array(labels)


def perform_clustering(embeddings, n_clusters_list, y_true, class_names, dataset_name):
    """Perform clustering and evaluate results for a specific dataset"""
    results = {}

    clustering_methods = {
        "KMeans": lambda n: KMeans(n_clusters=n, random_state=42).fit(embeddings),
        "Spectral": lambda n: SpectralClustering(
            n_clusters=n, random_state=42, affinity="rbf", n_jobs=-1
        ).fit(embeddings),
    }

    # Handle HDBSCAN separately since it doesn't take n_clusters parameter
    try:
        hdbscan_cluster = HDBSCAN(min_cluster_size=5, min_samples=3).fit(embeddings)
        hdbscan_labels = hdbscan_cluster.labels_
    except:
        # If HDBSCAN fails, use a single cluster as fallback
        hdbscan_labels = np.zeros(len(embeddings), dtype=int)

    # For each clustering method and number of clusters
    for method_name, cluster_func in clustering_methods.items():
        for n_clusters in n_clusters_list:
            # Perform clustering
            labels = cluster_func(n_clusters).labels_

            # Evaluate clustering - only calculate silhouette score if we have more than 1 cluster and clusters have more than 1 point
            if len(np.unique(labels)) > 1 and all(
                np.sum(labels == lab) > 1 for lab in np.unique(labels)
            ):
                try:
                    silhouette = silhouette_score(embeddings, labels)
                except:
                    silhouette = 0  # Set to 0 if silhouette calculation fails
            else:
                silhouette = 0  # Set to 0 if all points in one cluster

            v_measure = v_measure_score(y_true, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)

            results[f"{dataset_name}_{method_name}_{n_clusters}"] = {
                "labels": labels,
                "silhouette": silhouette,
                "v_measure": v_measure,
                "ari": ari,
                "nmi": nmi,
                "dataset": dataset_name,
            }

    # Process HDBSCAN result
    if len(np.unique(hdbscan_labels[hdbscan_labels != -1])) > 1:  # More than just noise
        # Calculate silhouette for non-noise points only
        non_noise_mask = hdbscan_labels != -1
        if (
            np.sum(non_noise_mask) > 1
            and len(np.unique(hdbscan_labels[non_noise_mask])) > 1
        ):
            try:
                silhouette = silhouette_score(
                    embeddings[non_noise_mask], hdbscan_labels[non_noise_mask]
                )
            except:
                silhouette = 0
        else:
            silhouette = 0

        v_measure = v_measure_score(y_true, hdbscan_labels)
        ari = adjusted_rand_score(y_true, hdbscan_labels)
        nmi = normalized_mutual_info_score(y_true, hdbscan_labels)

        results[f"{dataset_name}_HDBSCAN_any"] = {
            "labels": hdbscan_labels,
            "silhouette": silhouette,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": dataset_name,
        }
    else:
        # If HDBSCAN didn't find clusters, use all as one cluster
        v_measure = v_measure_score(y_true, np.zeros(len(embeddings), dtype=int))
        ari = adjusted_rand_score(y_true, np.zeros(len(embeddings), dtype=int))
        nmi = normalized_mutual_info_score(y_true, np.zeros(len(embeddings), dtype=int))

        results[f"{dataset_name}_HDBSCAN_any"] = {
            "labels": np.zeros(len(embeddings), dtype=int),
            "silhouette": 0,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": dataset_name,
        }

    return results


def main():
    print("Loading trained VAE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model = ImageNetVAE(input_channels=3, image_size=64, latent_dim=128).to(device)
    model.load_state_dict(torch.load("imagenet_vae_model.pth", map_location=device))
    model.eval()

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # Load training and test datasets
    train_dataset = ImageNetSubsetDataset(
        root_dir="/home/jin/Desktop/mm/data/imagenet_subset/train", transform=transform
    )

    test_dataset = ImageNetSubsetDataset(
        root_dir="/home/jin/Desktop/mm/data/imagenet_subset/test", transform=transform
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Extract embeddings
    print("Extracting embeddings from training set...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    print(f"Training embeddings shape: {train_embeddings.shape}")

    print("Extracting embeddings from test set...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Define cluster numbers: 1x (10), 2x (20), 3x (30) the real classes (10)
    n_clusters_list = [1, 5, 10, 20, 30]

    print("Performing clustering on training set...")
    train_results = perform_clustering(
        train_embeddings,
        n_clusters_list,
        train_labels,
        train_dataset.dataset.classes,
        "train",
    )

    print("Performing clustering on test set...")
    test_results = perform_clustering(
        test_embeddings,
        n_clusters_list,
        test_labels,
        test_dataset.dataset.classes,
        "test",
    )

    # Combine results
    all_results = {**train_results, **test_results}

    # Print results table
    print("\n" + "=" * 120)
    print("Clustering Results Summary")
    print("=" * 120)
    print(
        f"{'Dataset':<8} {'Method':<12} {'n_clusters':<10} {'Silhouette':<12} {'V-Measure':<12} {'ARI':<12} {'NMI':<12}"
    )
    print("-" * 120)

    for key, result in all_results.items():
        parts = key.split("_")
        dataset_name = parts[0]
        method = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        n_clusters = parts[-1] if parts[-1] != "any" else "auto"

        print(
            f"{dataset_name:<8} {method:<12} {n_clusters:<10} {result['silhouette']:<12.4f} {result['v_measure']:<12.4f} {result['ari']:<12.4f} {result['nmi']:<12.4f}"
        )

    print("=" * 120)

    # Create summary CSV
    results_df = pd.DataFrame(
        {
            "method_dataset_n_clusters": list(all_results.keys()),
            "dataset": [all_results[k]["dataset"] for k in all_results.keys()],
            "method": [k.split("_")[1] for k in all_results.keys()],
            "n_clusters": [
                k.split("_")[-1] if k.split("_")[-1] != "any" else "auto"
                for k in all_results.keys()
            ],
            "silhouette": [all_results[k]["silhouette"] for k in all_results.keys()],
            "v_measure": [all_results[k]["v_measure"] for k in all_results.keys()],
            "ari": [all_results[k]["ari"] for k in all_results.keys()],
            "nmi": [all_results[k]["nmi"] for k in all_results.keys()],
        }
    )

    results_df.to_csv("imagenet_clustering_results.csv", index=False)
    print("\nResults saved to imagenet_clustering_results.csv")
    print(f"Total results: {len(results_df)}")

    # Save embeddings
    np.save("imagenet_train_embeddings.npy", train_embeddings)
    np.save("imagenet_test_embeddings.npy", test_embeddings)
    print(
        "Embeddings saved to imagenet_train_embeddings.npy and imagenet_test_embeddings.npy"
    )


if __name__ == "__main__":
    main()
