#!/usr/bin/env python3
"""
Script to split the ImageNet subset into train, validation, and test sets
"""

import os
import random
import shutil

from sklearn.model_selection import train_test_split


def split_dataset(
    source_dir,
    train_dir,
    val_dir,
    test_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """Split dataset into train, validation, and test sets"""
    random.seed(seed)

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all classes in the source directory
    classes = [
        d
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
        and not d.startswith(".")
        and d not in ["val", "test", "train", "__pycache__"]
    ]

    print(f"Found {len(classes)} classes to split")

    split_stats = {}

    for class_name in classes:
        class_source_path = os.path.join(source_dir, class_name)

        # Get all images in this class
        all_images = [
            img
            for img in os.listdir(class_source_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".JPEG"))
        ]

        # Need at least 3 images to split into 3 sets
        if len(all_images) < 3:
            print(
                f"Warning: Class {class_name} has only {len(all_images)} images, not enough to split. Assigning all to train."
            )
            # Put all images in train
            train_images = all_images
            val_images = []
            test_images = []
        else:
            # Ensure each split has at least 1 sample if possible
            min_test = max(1, int(len(all_images) * test_ratio))
            min_val = max(1, int(len(all_images) * val_ratio))

            # If we don't have enough images to satisfy minimums, adjust ratios
            if (
                len(all_images) < min_test + min_val + 1
            ):  # +1 for at least 1 training sample
                # Redistribute by reducing the required minima
                test_count = max(
                    1, len(all_images) // 4
                )  # At least 1, or 25% if possible
                val_count = max(
                    1, (len(all_images) - test_count) // 3
                )  # At least 1, or 1/3 of what's left
                train_count = len(all_images) - test_count - val_count
                if train_count <= 0:
                    # If we still don't have enough, just evenly distribute
                    train_count = len(all_images) // 3
                    val_count = (len(all_images) - train_count) // 2
                    test_count = len(all_images) - train_count - val_count

                # Split accordingly
                train_images = all_images[:train_count]
                val_images = all_images[train_count : train_count + val_count]
                test_images = all_images[train_count + val_count :]
            else:
                # Split the images normally
                train_val_images, test_images = train_test_split(
                    all_images, test_size=test_ratio, random_state=seed
                )

                # Then split the remainder into train and validation
                train_images, val_images = train_test_split(
                    train_val_images,
                    test_size=val_ratio / (train_ratio + val_ratio),
                    random_state=seed,
                )

        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

        split_stats[class_name] = {
            "total": len(all_images),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

        print(
            f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
        )

    return split_stats


def main():
    source_dir = "/home/jin/Desktop/mm/data/imagenet_subset"
    train_dir = "/home/jin/Desktop/mm/data/imagenet_subset/train"
    val_dir = "/home/jin/Desktop/mm/data/imagenet_subset/val"
    test_dir = "/home/jin/Desktop/mm/data/imagenet_subset/test"

    print("Splitting ImageNet subset into train, validation, and test sets...")
    print("Train ratio: 70%, Validation ratio: 15%, Test ratio: 15%")

    split_stats = split_dataset(source_dir, train_dir, val_dir, test_dir)

    # Print summary
    total_train = sum(stats["train"] for stats in split_stats.values())
    total_val = sum(stats["val"] for stats in split_stats.values())
    total_test = sum(stats["test"] for stats in split_stats.values())
    total_all = sum(stats["total"] for stats in split_stats.values())

    print("\nSplit Summary:")
    print(f"  Total images: {total_all}")
    print(f"  Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"  Test: {total_test} ({total_test/total_all*100:.1f}%)")
    print(f"  Classes: {len(split_stats)}")

    # Save detailed stats
    stats_file = os.path.join(os.path.dirname(source_dir), "split_stats.txt")
    with open(stats_file, "w") as f:
        f.write("ImageNet Subset Split Statistics\n")
        f.write("=" * 50 + "\n")
        for class_name, stats in split_stats.items():
            f.write(
                f"{class_name}: total={stats['total']}, train={stats['train']}, val={stats['val']}, test={stats['test']}\n"
            )

    print(f"\nSplit statistics saved to {stats_file}")


if __name__ == "__main__":
    main()
