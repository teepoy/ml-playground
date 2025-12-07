"""
Basic usage example for COCO S3 Loader.

This example demonstrates how to use the CocoS3Dataset with PyTorch DataLoader
for training machine learning models.
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from coco_s3_loader import CocoS3Dataset


def main():
    """Demonstrate basic usage of CocoS3Dataset."""

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    # Note: Replace with your actual COCO JSON file path
    dataset = CocoS3Dataset(
        annotation_file="path/to/coco_annotations.json",
        transform=transform,
        handle_errors="skip",  # Skip images that fail to load
    )

    print(f"Dataset loaded with {len(dataset)} images")
    print(f"Number of categories: {len(dataset.get_categories())}")

    # Print categories
    print("\nCategories:")
    for cat in dataset.get_categories():
        print(f"  - {cat['name']} (id: {cat['id']})")

    # Create dataloader with batching
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: x,  # Custom collate for dict annotations
    )

    # Iterate through a few batches
    print("\nIterating through batches (batch_size=4):")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only show first 3 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Number of samples: {len(batch)}")

        for sample_idx, sample in enumerate(batch):
            if sample is None:  # Handle skipped images
                print(f"  Sample {sample_idx + 1}: Skipped (load error)")
                continue

            image, annotation = sample
            print(f"  Sample {sample_idx + 1}:")
            print(f"    Image shape: {image.shape}")
            print(f"    Image ID: {annotation['image_info']['id']}")
            print(f"    Number of annotations: {len(annotation['annotations'])}")

    # Access specific image info without loading
    print("\n\nAccessing image metadata without loading:")
    image_info = dataset.get_image_info(0)
    print("First image:")
    print(f"  ID: {image_info['id']}")
    print(f"  File: {image_info['file_name']}")
    print(f"  Size: {image_info['width']}x{image_info['height']}")


if __name__ == "__main__":
    # Note: This example requires a valid COCO JSON file with S3 URLs
    # For testing, you can create a sample file or use the test fixtures
    print("COCO S3 Loader - Basic Usage Example")
    print("=" * 50)
    print("\nThis is a template. Update the annotation_file path to run.")
    print("\nFor a working example, see the tests in tests/test_dataset.py")
