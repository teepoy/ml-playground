#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for the ImageNet subset dataset
This script creates, trains and saves a VAE model for the ImageNet subset
"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class ImageNetVAE(nn.Module):
    def __init__(self, input_channels=3, image_size=64, latent_dim=128):
        super(ImageNetVAE, self).__init__()

        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 4  # After 2 pooling operations

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(
                input_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 8 x 8
            nn.ReLU(),
        )

        # Calculate the size after encoder
        self.encoded_features = (
            128 * (self.feature_size // 2) * (self.feature_size // 2)
        )

        # Latent space
        self.fc_mu = nn.Linear(self.encoded_features, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_features, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoded_features)
        self.decoder = nn.Sequential(
            # Input: 128 x 8 x 8 (after fc_decode reshape)
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, input_channels, kernel_size=4, stride=2, padding=1
            ),  # 3 x 64 x 64
            nn.Sigmoid(),  # Sigmoid to keep values between 0 and 1
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


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


def train_vae(model, train_loader, epochs=50, learning_rate=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.4f}")

    return losses


def main():
    print("Setting up ImageNet subset VAE training...")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    train_dataset = ImageNetSubsetDataset(
        root_dir="/home/jin/Desktop/mm/data/imagenet_subset/train", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(train_loader)}")

    # Initialize model
    model = ImageNetVAE(input_channels=3, image_size=64, latent_dim=128).to(device)

    print("Model initialized. Starting training...")

    # Train the model
    losses = train_vae(model, train_loader, epochs=50, device=device)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("imagenet_vae_training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Save the trained model
    model_path = "imagenet_vae_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
