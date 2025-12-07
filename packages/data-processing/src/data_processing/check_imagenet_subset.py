#!/usr/bin/env python3
"""
Script to verify the ImageNet subset dataset
"""

import os


def check_dataset_split(dataset_path):
    """Check the dataset split structure"""

    splits = ["train", "val", "test"]
    class_counts = {}
    split_info = {}

    for split in splits:
        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            print(f"Split {split} does not exist!")
            continue

        classes = [
            d
            for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ]

        class_counts[split] = len(classes)
        split_info[split] = {}

        total_images = 0
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            images = [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".JPEG")
                )
            ]
            count = len(images)
            split_info[split][class_name] = count
            total_images += count

        print(f"{split.upper()}: {total_images} images across {len(classes)} classes")

    # Print detailed breakdown
    print("\nDetailed breakdown:")
    for class_name in split_info["train"].keys():
        train_count = split_info["train"].get(class_name, 0)
        val_count = split_info["val"].get(class_name, 0) if "val" in split_info else 0
        test_count = (
            split_info["test"].get(class_name, 0) if "test" in split_info else 0
        )
        total = train_count + val_count + test_count
        print(
            f"  {class_name}: {train_count} train, {val_count} val, {test_count} test (total: {total})"
        )


def main():
    dataset_path = "/home/jin/Desktop/mm/data/imagenet_subset"

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return

    print(f"Checking ImageNet subset at {dataset_path}")
    print("=" * 50)

    check_dataset_split(dataset_path)

    print("\n" + "=" * 50)
    print("Dataset verification completed!")


if __name__ == "__main__":
    main()
