#!/usr/bin/env python3
"""
Script to load the pretrained VAE model for the iris dataset and demonstrate its usage
"""

import joblib
import numpy as np
import torch
from sklearn.datasets import load_iris


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


def load_trained_vae(model_path="iris_vae_model.pth", scaler_path="iris_scaler.pkl"):
    """
    Load the trained VAE model and associated scaler
    """
    # Initialize the model
    model = IrisVAE(input_dim=4, latent_dim=2)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode

    # Load the scaler
    scaler = joblib.load(scaler_path)

    print(f"Loaded VAE model from {model_path}")
    print(f"Loaded scaler from {scaler_path}")

    return model, scaler


def encode_iris_sample(model, scaler, sample):
    """
    Encode a single iris sample to the latent space
    sample: array-like of shape (4,) representing [sepal_length, sepal_width, petal_length, petal_width]
    """
    # Convert to tensor and normalize
    sample_tensor = torch.FloatTensor(scaler.transform([sample]))

    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(sample_tensor)
        z = model.reparameterize(mu, logvar)

    return z.numpy()[0]


def decode_from_latent(model, scaler, latent_vector):
    """
    Decode a latent vector back to the original feature space
    latent_vector: array-like of shape (2,) representing the latent coordinates
    """
    # Convert to tensor
    z = torch.FloatTensor([latent_vector])

    # Decode to original space
    with torch.no_grad():
        recon_x = model.decode(z)

    # Denormalize
    recon_sample = scaler.inverse_transform(recon_x.numpy())[0]

    return recon_sample


def reconstruct_iris_sample(model, scaler, sample):
    """
    Reconstruct an iris sample by encoding then decoding
    """
    # Encode to latent space
    z = encode_iris_sample(model, scaler, sample)

    # Decode back to original space
    recon_sample = decode_from_latent(model, scaler, z)

    return recon_sample, z


def main():
    print("Loading trained VAE model...")
    model, scaler = load_trained_vae()

    print("\\nLoading iris dataset for demonstration...")
    iris = load_iris()

    # Get a sample from each class
    unique_targets = np.unique(iris.target)

    print("\\nDemonstrating VAE capabilities:")
    print("=" * 50)

    for target_idx, target_name in enumerate(iris.target_names):
        # Find first sample of this class
        sample_idx = np.where(iris.target == target_idx)[0][0]
        original_sample = iris.data[sample_idx]
        original_target = iris.target[sample_idx]

        print(f"\\n{target_name.upper()} Sample (Index {sample_idx}):")
        print(f"Original features: {original_sample}")
        print(f"Actual class: {iris.target_names[original_target]}")

        # Reconstruct the sample
        reconstructed_sample, latent_coords = reconstruct_iris_sample(
            model, scaler, original_sample
        )

        print(f"Reconstructed:   {reconstructed_sample}")
        print(f"Latent coords:   {latent_coords}")

        # Calculate reconstruction error
        mse = np.mean((original_sample - reconstructed_sample) ** 2)
        print(f"Reconstruction MSE: {mse:.6f}")

    print("\\n" + "=" * 50)
    print("Generating new samples by sampling from latent space...")

    # Generate some points in the latent space
    for i in range(3):
        # Random point in latent space
        latent_point = np.random.normal(0, 1, size=(2,))
        generated_sample = decode_from_latent(model, scaler, latent_point)

        print(
            f"Generated sample {i+1} from latent point {latent_point}: {generated_sample}"
        )


if __name__ == "__main__":
    main()
