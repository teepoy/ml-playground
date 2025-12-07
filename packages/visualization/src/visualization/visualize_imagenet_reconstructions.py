#!/usr/bin/env python3
"""
Script to visualize and evaluate VAE reconstructions on ImageNet subset
"""

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
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


def visualize_reconstructions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageNetVAE(input_channels=3, image_size=64, latent_dim=128).to(device)
    model.load_state_dict(torch.load("imagenet_vae_model.pth", map_location=device))
    model.eval()

    # Transform for inference
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # Load a subset of test data for visualization
    dataset = ImageFolder(
        root="/home/jin/Desktop/mm/data/imagenet_subset/test", transform=transform
    )

    # Take the first 8 images for visualization
    indices = list(range(min(8, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            # Denormalize if needed (VAE outputs are already in [0,1] due to sigmoid)
            # Combine original and reconstructed images
            comparison = torch.cat([data[:8], recon_batch[:8]])

            # Save the comparison image
            vutils.save_image(
                comparison.cpu(),
                "imagenet_vae_reconstruction_comparison.png",
                nrow=8,
                normalize=False,
                padding=2,
            )

            # Calculate reconstruction error
            mse_loss = torch.nn.functional.mse_loss(recon_batch, data)
            print(f"Average MSE reconstruction error: {mse_loss.item():.4f}")

            # Show first 4 originals and their reconstructions separately
            originals = data[:4].cpu()
            reconstructions = recon_batch[:4].cpu()

            # Save originals and reconstructions
            vutils.save_image(
                originals, "imagenet_originals.png", nrow=4, normalize=False
            )
            vutils.save_image(
                reconstructions, "imagenet_reconstructions.png", nrow=4, normalize=False
            )

            break  # Only process the first batch

    print("Reconstruction visualization saved!")

    # Get some statistics about the latent space
    print("Latent space statistics:")
    print(f"  Mean of mu: {mu.mean().item():.4f}")
    print(f"  Std of mu: {mu.std().item():.4f}")
    print(f"  Mean of logvar: {logvar.mean().item():.4f}")
    print(f"  Std of logvar: {logvar.std().item():.4f}")


def main():
    print("Visualizing VAE reconstructions...")
    visualize_reconstructions()
    print("Reconstruction analysis completed!")


if __name__ == "__main__":
    main()
