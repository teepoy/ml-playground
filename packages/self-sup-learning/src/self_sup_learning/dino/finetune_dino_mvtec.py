#!/usr/bin/env python3
"""
Fine-tuning DINO v2 model on MVTec-AD 'pill' type dataset for improved anomaly detection
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_project_root():
    """Find the project root directory by looking for the main project markers."""
    current_path = Path(__file__).resolve()
    # Start from current directory and go up the hierarchy
    for parent in current_path.parents:
        # Check for the main project root which has both .git and packages directory
        if (parent / ".git").exists() and (parent / "packages").exists():
            return parent
    # If no main project markers found, return the directory containing the project structure we expect
    return (
        current_path.parent.parent.parent.parent.parent
    )  # fallback to previous method


def get_dino_paths():
    """Get the DINO config and checkpoint paths relative to project root."""
    project_root = get_project_root()
    config_path = (
        project_root
        / "packages"
        / "mmpretrain"
        / "configs"
        / "dinov2"
        / "vit-base-p14_dinov2-pre_headless.py"
    )
    checkpoint_path = (
        project_root
        / "packages"
        / "mmpretrain"
        / "pretrained"
        / "vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth"
    )
    return str(config_path), str(checkpoint_path)


class MVTecBackgroundDataset(Dataset):
    """
    Dataset class for MVTec-AD that handles the train/test structure for any background class.
    """

    def __init__(
        self,
        root_dir,
        pill_type="pill",
        split="train",
        transform=None,
        use_train_defects=True,
    ):
        """
        Args:
            root_dir (str): Root directory of the MVTec-AD dataset
            pill_type (str): Type of object (default "pill")
            split (str): Dataset split to load ("train" or "test")
            transform: Transformations to apply to the images
            use_train_defects (bool): Whether to include defect types in train split (for self-supervised learning)
        """
        self.root_dir = root_dir
        self.pill_type = pill_type
        self.split = split
        self.transform = transform

        # Define the path for the specific pill type and split
        split_path = os.path.join(root_dir, pill_type, split)

        if not os.path.exists(split_path):
            raise ValueError(f"Path does not exist: {split_path}")

        # Get all subdirectories (defect classes in test, or just "good" in train)
        self.image_paths = []
        self.image_labels = []

        for defect_class in os.listdir(split_path):
            defect_path = os.path.join(split_path, defect_class)
            if os.path.isdir(defect_path):
                # Only include "good" samples for training if not using defects
                if (
                    split == "train"
                    and not use_train_defects
                    and defect_class != "good"
                ):
                    continue

                for img_file in os.listdir(defect_path):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(defect_path, img_file)
                        self.image_paths.append(img_path)

                        # Label as 0 for good, 1 for defects (or just 0 for all if only interested in features)
                        label = 0 if defect_class == "good" else 1
                        self.image_labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {pill_type}/{split}")
        print(
            f"Image label distribution: {dict(zip(*np.unique(self.image_labels, return_counts=True)))}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_mvtec_datasets(data_path, pill_type="pill", use_train_defects=False):
    """Create train and test datasets for the specified background type"""

    # Ensure data_path is a string for compatibility
    if isinstance(data_path, Path):
        data_path = str(data_path)

    # Define transforms for DINO (similar to original values but with standard normalization)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test transform (deterministic)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = MVTecBackgroundDataset(
        root_dir=data_path,
        pill_type=pill_type,
        split="train",
        transform=train_transform,
        use_train_defects=use_train_defects,
    )

    test_dataset = MVTecBackgroundDataset(
        root_dir=data_path,
        pill_type=pill_type,
        split="test",
        transform=test_transform,
        use_train_defects=True,  # Include defect types in test for evaluation
    )

    return train_dataset, test_dataset


def load_dino_model():
    """Load the DINOv2 model for fine-tuning using relative paths from project root"""
    # Get the DINO paths
    config_path, checkpoint_path = get_dino_paths()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    try:
        # Import necessary modules
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        from mmpretrain.models import build_classifier

        # Load config and model
        cfg = Config.fromfile(config_path)
        model = build_classifier(cfg.model)
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model
    except ImportError as e:
        raise ImportError(f"Failed to import required modules: {e}")
    except Exception as e:
        raise Exception(f"Error loading DINO model: {e}")


def setup_model_for_finetuning(model, num_classes=2, freeze_backbone=False):
    """Prepare the model for fine-tuning"""
    if model is None:
        return None

    # For DINO models, we typically want to fine-tune the feature extractor
    # but we need to make sure it still outputs features appropriately

    # Check if the model has a head attribute before accessing it
    model_has_head = hasattr(model, "head") and model.head is not None

    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # If the model has a head (that is not None), unfreeze it to allow for fine-tuning
        if model_has_head:
            for param in model.head.parameters():
                param.requires_grad = True
        # If no head exists (head is None) or head attribute doesn't exist,
        # we don't unfreeze anything specific - the whole network remains frozen

    return model


def fine_tune_dino_model(
    model, train_dataset, test_dataset, device, epochs=10, lr=1e-5, batch_size=16
):
    """Fine-tune the DINO model using self-supervised approach on the pill dataset"""

    # For DINO fine-tuning, we'll use a self-supervised approach
    # Create augmented data loaders - each image will be augmented twice to create pairs
    from torchvision.transforms import InterpolationMode

    # Define strong augmentations for self-supervised learning
    base_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=21)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Custom dataset that returns two augmented versions of each image
    class AugmentedDataset(Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            # Get the original image and label
            img_path, label = (
                self.base_dataset.image_paths[idx],
                self.base_dataset.image_labels[idx],
            )
            img = Image.open(img_path).convert("RGB")

            # Apply the same transform twice to generate two augmented views
            # Use different random states for each view to create different augmentations
            import torchvision.transforms.functional as F

            # Apply random augmentations differently to each view
            # For this simpler version, let's use different transforms
            # But to make the implementation easier, we'll use the same transform but different random augmentations will happen each time

            view1 = self.transform(img)
            view2 = self.transform(img)

            return view1, view2, label

    # Create augmented datasets
    train_aug_dataset = AugmentedDataset(train_dataset, base_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_aug_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Move model to device
    model = model.to(device)

    # Define optimizer - we'll use a lower weight decay for self-supervised learning
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01
    )

    # Scheduler for learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.7)

    model.train()
    print(f"Starting self-supervised fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, (data1, data2, targets) in enumerate(train_loader):
            data1, data2 = data1.to(device), data2.to(device)

            optimizer.zero_grad()

            # Get features from both augmented views
            features1 = model(data1)
            features2 = model(data2)

            # Handle different output formats
            if isinstance(features1, (list, tuple)):
                features1 = features1[0]
            if isinstance(features2, (list, tuple)):
                features2 = features2[0]

            # Flatten features if needed
            if len(features1.shape) > 2:
                features1 = features1.view(features1.size(0), -1)
            if len(features2.shape) > 2:
                features2 = features2.view(features2.size(0), -1)

            # Normalize features to unit vectors (for cosine similarity)
            features1 = torch.nn.functional.normalize(features1, dim=1)
            features2 = torch.nn.functional.normalize(features2, dim=1)

            # Calculate cosine similarity between features of same image
            cosine_sim = torch.sum(
                features1 * features2, dim=1
            )  # This is cosine similarity
            # Maximize the similarity (negative because we want to minimize the loss)
            loss = -cosine_sim.mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:  # Print every 20 batches
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

        scheduler.step()

        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {epoch_loss:.4f}")

    print("Self-supervised fine-tuning completed!")
    return model


def extract_features_finetuned(model, data_loader, device):
    """Extract features from the fine-tuned model"""
    # If we get here, model should not be None (would have raised an exception in load_dino_model)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)

            # Get features from the model
            # For DINO, we want the intermediate representations
            batch_features = model(data)

            # Handle different output formats
            if isinstance(batch_features, (list, tuple)):
                # If it's a list/tuple of tensors, take the first element
                batch_features = batch_features[0]

            # Flatten the features if necessary (for ViT models, may need to flatten)
            batch_features = batch_features.view(batch_features.size(0), -1)

            features.extend(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(features), np.array(labels)


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy, all_preds, all_targets


def main():
    # Get the background type from command line arguments or default to 'pill'
    import sys

    background_type = sys.argv[1] if len(sys.argv) > 1 else "pill"

    print(f"Fine-tuning DINO v2 model on MVTec-AD '{background_type}' dataset...")

    # Get project root and set MVTec-AD dataset path relative to it
    project_root = get_project_root()

    # Try both possible naming conventions for MVTec-AD
    mvtec_path_options = [
        project_root / "data" / "mvtec_ad",  # underscore version
        project_root / "data" / "mvtec-ad",  # hyphen version
        project_root / "data" / "MVTec-AD",  # capitalized version
    ]

    mvtec_path = None
    bg_train_path = None

    for path in mvtec_path_options:
        potential_train_path = path / background_type / "train"
        if os.path.exists(str(potential_train_path)):
            mvtec_path = path
            bg_train_path = potential_train_path
            break

    if bg_train_path is None:
        # If none exist, default to the hyphen version (most common)
        mvtec_path = project_root / "data" / "mvtec-ad"
        bg_train_path = mvtec_path / background_type / "train"

    if not os.path.exists(str(bg_train_path)):
        print(
            f"MVTec-AD '{background_type}' train dataset not found at {bg_train_path}"
        )
        print(
            "Please ensure the MVTec-AD dataset is properly downloaded and structured."
        )
        print(
            f"Expected directory structure: data/mvtec-ad/{background_type}/train/good/"
        )
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print("Creating datasets...")
    train_dataset, test_dataset = create_mvtec_datasets(
        data_path=mvtec_path,
        pill_type=background_type,  # Use the specified background type
        use_train_defects=False,  # Only use 'good' images for training
    )

    if len(train_dataset) == 0:
        print("No training data found, exiting.")
        return

    # Load the pretrained DINO model
    print("Loading pretrained DINO model...")
    try:
        model = load_dino_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Required model files are missing. Please download and set up the DINO model properly."
        )
        return  # Exit the program
    except Exception as e:
        print(f"Error: {e}")
        print(
            "Failed to load the DINO model. Please check your installation and files."
        )
        return  # Exit the program

    # Prepare model for fine-tuning
    # For DINO fine-tuning, we typically don't freeze the backbone since we want to adapt the whole model
    model = setup_model_for_finetuning(model, freeze_backbone=False)

    # Fine-tune the model
    print("Starting fine-tuning process...")
    fine_tuned_model = fine_tune_dino_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        epochs=5,  # Reduced for testing
        lr=1e-5,
        batch_size=8,  # Reduced batch size if GPU memory is limited
    )

    # Save the fine-tuned model
    model_filename = f"fine_tuned_dino_{background_type}.pth"
    torch.save(fine_tuned_model.state_dict(), model_filename)
    print(f"Fine-tuned model saved to {model_filename}")

    # Extract features from fine-tuned model for clustering analysis
    print("Extracting features from fine-tuned model for clustering analysis...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    train_features, train_labels = extract_features_finetuned(
        fine_tuned_model, train_loader, device
    )
    test_features, test_labels = extract_features_finetuned(
        fine_tuned_model, test_loader, device
    )

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Perform clustering analysis similar to the original approach
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )

    # Use only test set with 'good' as 0 and defects as 1 for anomaly detection
    unique_labels = np.unique(test_labels)
    print(
        f"Test labels distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}"
    )

    # Only perform clustering if we have more than one class in test set
    if len(np.unique(test_labels)) > 1:
        # Cluster the test features with k=2 (normal vs anomaly)
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(test_features)

        # Evaluate clustering vs true labels
        silhouette = silhouette_score(test_features, cluster_labels)
        v_measure = v_measure_score(test_labels, cluster_labels)
        ari = adjusted_rand_score(test_labels, cluster_labels)
        nmi = normalized_mutual_info_score(test_labels, cluster_labels)

        print(f"\nClustering Results for Fine-tuned DINO:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"V-Measure: {v_measure:.4f}")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")

        # Save results
        results_filename = f"fine_tuned_dino_{background_type}_results.csv"
        results_df = pd.DataFrame(
            {
                "metric": ["silhouette", "v_measure", "ari", "nmi"],
                "value": [silhouette, v_measure, ari, nmi],
            }
        )
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved to {results_filename}")
    else:
        print("Only one class in test set, skipping clustering evaluation")
        print(
            "All test samples belong to the same class, so clustering is not meaningful"
        )

    # Save features
    np.save(f"fine_tuned_{background_type}_train_features.npy", train_features)
    np.save(f"fine_tuned_{background_type}_test_features.npy", test_features)
    print("Features saved for further analysis")

    print("\nFine-tuning and evaluation completed!")


if __name__ == "__main__":
    main()
