#!/usr/bin/env python3
"""
Script to create a subset of ImageNet with 10 classes and 100-1000 images per class
"""

import argparse
import os
import random
import shutil


def get_imagenet_classes(train_dir):
    """Get list of ImageNet classes from the train directory"""
    classes = [
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]
    return sorted(classes)


def select_random_classes(classes, num_classes=10, seed=42):
    """Select random classes from the list"""
    random.seed(seed)
    selected = random.sample(classes, num_classes)
    return selected


def copy_images_to_subset(
    source_dir, target_dir, class_dirs, min_images=100, max_images=1000, seed=42
):
    """Copy a random subset of images from selected classes to target directory"""
    random.seed(seed)

    os.makedirs(target_dir, exist_ok=True)

    class_mapping = {}  # To keep track of original class names

    for i, class_dir in enumerate(class_dirs):
        source_class_path = os.path.join(source_dir, class_dir)
        if not os.path.isdir(source_class_path):
            print(f"Warning: {source_class_path} is not a directory, skipping...")
            continue

        # Get all images in the class
        all_images = [
            img
            for img in os.listdir(source_class_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]

        # Determine how many images to copy (random number between min and max, or total if less)
        num_to_copy = min(len(all_images), max_images)
        if len(all_images) >= min_images:
            num_to_copy = random.randint(min_images, min(max_images, len(all_images)))
        else:
            print(
                f"Warning: Class {class_dir} has only {len(all_images)} images, less than minimum {min_images}. Using all available."
            )
            num_to_copy = len(all_images)

        # Randomly select images
        selected_images = random.sample(all_images, num_to_copy)

        # Create target class directory
        target_class_dir = (
            f"class_{i:03d}_{class_dir}"  # Include original class name for reference
        )
        target_class_path = os.path.join(target_dir, target_class_dir)
        os.makedirs(target_class_path, exist_ok=True)

        class_mapping[target_class_dir] = {
            "original_name": class_dir,
            "count": num_to_copy,
            "selected_images": selected_images,
        }

        # Copy selected images
        for img in selected_images:
            source_img_path = os.path.join(source_class_path, img)
            target_img_path = os.path.join(target_class_path, img)
            shutil.copy2(source_img_path, target_img_path)

        print(
            f"Copied {num_to_copy} images from class {class_dir} to {target_class_dir}"
        )

    return class_mapping


def main():
    parser = argparse.ArgumentParser(description="Create a subset of ImageNet")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/home/jin/Desktop/mm/data/imagenet/train",
        help="Source ImageNet train directory",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/home/jin/Desktop/mm/data/imagenet_subset",
        help="Target directory for the subset",
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of classes to include"
    )
    parser.add_argument(
        "--min-images", type=int, default=100, help="Minimum number of images per class"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images per class",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print(f"Creating ImageNet subset with {args.num_classes} classes")
    print(
        f"Each class will have between {args.min_images} and {args.max_images} images"
    )

    # Get all available classes
    all_classes = get_imagenet_classes(args.source_dir)
    print(f"Found {len(all_classes)} classes in ImageNet")

    # Select random classes
    selected_classes = select_random_classes(all_classes, args.num_classes, args.seed)
    print(f"Selected classes: {selected_classes}")

    # Create the subset
    class_mapping = copy_images_to_subset(
        args.source_dir,
        args.target_dir,
        selected_classes,
        args.min_images,
        args.max_images,
        args.seed,
    )

    # Save class mapping to a file for reference
    mapping_file = os.path.join(args.target_dir, "class_mapping.txt")
    with open(mapping_file, "w") as f:
        f.write("ImageNet Subset Class Mapping\n")
        f.write("=" * 50 + "\n")
        for target_class, info in class_mapping.items():
            f.write(
                f"{target_class} -> Original: {info['original_name']}, Count: {info['count']}\n"
            )

    print(f"\nSubset created successfully in {args.target_dir}")
    print(f"Class mapping saved to {mapping_file}")

    # Print summary
    total_images = sum(info["count"] for info in class_mapping.values())
    print("\nSummary:")
    print(f"  Total classes: {len(class_mapping)}")
    print(f"  Total images: {total_images}")
    print(f"  Average images per class: {total_images / len(class_mapping):.1f}")


if __name__ == "__main__":
    main()
