#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for the Iris dataset
This script creates, trains and saves a VAE model for the iris dataset
"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class IrisVAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(IrisVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),  # Sigmoid to keep values between 0 and 1
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


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def load_and_preprocess_iris():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled)

    # Since VAE is unsupervised, we only need the features
    dataset = TensorDataset(X_tensor)

    return dataset, X_scaled, y, scaler


def train_vae(model, dataset, epochs=100, batch_size=16, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

    return losses


def visualize_latent_space(model, X_scaled, y, title="Latent Space of Iris VAE"):
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = model.reparameterize(mu, logvar)

    z = z.numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    # Add legend
    classes = ["setosa", "versicolor", "virginica"]
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

    plt.tight_layout()
    plt.savefig("iris_vae_latent_space.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_reconstruction(model, X_scaled, scaler):
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        recon_x, mu, logvar = model(X_tensor)

    # Denormalize for visualization
    original = scaler.inverse_transform(X_tensor.numpy())
    reconstructed = scaler.inverse_transform(recon_x.numpy())

    # Plot original vs reconstructed
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

    for i in range(4):
        row = i // 2
        col = i % 2

        axes[row, col].scatter(original[:, i], reconstructed[:, i], alpha=0.7)
        axes[row, col].plot(
            [original[:, i].min(), original[:, i].max()],
            [original[:, i].min(), original[:, i].max()],
            "r--",
            lw=2,
        )
        axes[row, col].set_xlabel(f"Original {feature_names[i]}")
        axes[row, col].set_ylabel(f"Reconstructed {feature_names[i]}")
        axes[row, col].set_title(f"{feature_names[i]}: Original vs Reconstructed")

    plt.tight_layout()
    plt.savefig("iris_vae_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    print("Loading and preprocessing iris dataset...")
    dataset, X_scaled, y, scaler = load_and_preprocess_iris()

    print("Initializing VAE model...")
    model = IrisVAE(input_dim=4, latent_dim=2)

    print("Training VAE model...")
    losses = train_vae(model, dataset, epochs=200)

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("iris_vae_training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Visualizing latent space...")
    visualize_latent_space(model, X_scaled, y)

    print("Visualizing reconstruction...")
    visualize_reconstruction(model, X_scaled, scaler)

    # Save the trained model
    model_path = "iris_vae_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save the scaler for future use
    import joblib

    joblib.dump(scaler, "iris_scaler.pkl")
    print("Scaler saved to iris_scaler.pkl")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
