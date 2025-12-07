"""
Example demonstrating format conversion between COCO and Label Studio.

This example shows how to convert annotations between COCO format
and Label Studio JSON format.
"""

import json

from coco_s3_loader import (
    coco_file_to_label_studio_file,
    coco_to_label_studio,
    label_studio_file_to_coco_file,
    label_studio_to_coco,
)


def example_coco_to_label_studio():
    """Example: Convert COCO format to Label Studio format."""
    print("=" * 60)
    print("Example 1: COCO to Label Studio Conversion")
    print("=" * 60)

    # Sample COCO data
    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": "s3://my-bucket/images/dog.jpg",
                "width": 640,
                "height": 480,
            },
            {
                "id": 2,
                "file_name": "s3://my-bucket/images/cat.jpg",
                "width": 800,
                "height": 600,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],  # [x, y, width, height] in pixels
                "area": 30000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [50, 75, 300, 200],
                "area": 60000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "dog", "supercategory": "animal"},
            {"id": 2, "name": "cat", "supercategory": "animal"},
        ],
    }

    print("\nInput COCO format:")
    print(json.dumps(coco_data, indent=2))

    # Convert to Label Studio format
    label_studio_tasks = coco_to_label_studio(coco_data, project_id=1)

    print("\nOutput Label Studio format:")
    print(json.dumps(label_studio_tasks, indent=2))

    print("\nKey observations:")
    print("- Bounding boxes converted from pixels to percentages")
    print("- Category IDs converted to category names")
    print("- Structure changed to Label Studio task format")


def example_label_studio_to_coco():
    """Example: Convert Label Studio format to COCO format."""
    print("\n" + "=" * 60)
    print("Example 2: Label Studio to COCO Conversion")
    print("=" * 60)

    # Sample Label Studio tasks
    label_studio_tasks = [
        {
            "id": 1,
            "data": {"image": "s3://my-bucket/images/person.jpg"},
            "annotations": [
                {
                    "id": "annotation_1",
                    "result": [
                        {
                            "id": "result_1",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 1000,
                            "original_height": 800,
                            "value": {
                                "x": 10.0,  # 10% from left
                                "y": 12.5,  # 12.5% from top
                                "width": 20.0,  # 20% of image width
                                "height": 25.0,  # 25% of image height
                                "rotation": 0,
                                "rectanglelabels": ["person"],
                            },
                        }
                    ],
                }
            ],
        }
    ]

    print("\nInput Label Studio format:")
    print(json.dumps(label_studio_tasks, indent=2))

    # Convert to COCO format
    coco_data = label_studio_to_coco(label_studio_tasks)

    print("\nOutput COCO format:")
    print(json.dumps(coco_data, indent=2))

    print("\nKey observations:")
    print("- Bounding boxes converted from percentages to pixels")
    print("- Category names extracted and assigned IDs")
    print("- Structure changed to COCO format")
    print(
        f"- Calculated bbox: x={coco_data['annotations'][0]['bbox'][0]:.1f}, "
        f"y={coco_data['annotations'][0]['bbox'][1]:.1f}, "
        f"w={coco_data['annotations'][0]['bbox'][2]:.1f}, "
        f"h={coco_data['annotations'][0]['bbox'][3]:.1f}"
    )


def example_round_trip_conversion():
    """Example: Demonstrate round-trip conversion."""
    print("\n" + "=" * 60)
    print("Example 3: Round-trip Conversion (COCO → Label Studio → COCO)")
    print("=" * 60)

    # Original COCO data
    original_coco = {
        "images": [
            {
                "id": 1,
                "file_name": "s3://bucket/image.jpg",
                "width": 800,
                "height": 600,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 180],
                "area": 36000,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "vehicle", "supercategory": "object"}],
    }

    print("\nOriginal COCO:")
    print(f"  Bbox: {original_coco['annotations'][0]['bbox']}")

    # Convert to Label Studio
    tasks = coco_to_label_studio(original_coco)
    ls_bbox = tasks[0]["annotations"][0]["result"][0]["value"]
    print("\nLabel Studio (percentages):")
    print(
        f"  x={ls_bbox['x']:.2f}%, y={ls_bbox['y']:.2f}%, "
        f"w={ls_bbox['width']:.2f}%, h={ls_bbox['height']:.2f}%"
    )

    # Convert back to COCO
    final_coco = label_studio_to_coco(tasks, categories=["vehicle"])
    print("\nConverted back to COCO:")
    print(f"  Bbox: {final_coco['annotations'][0]['bbox']}")

    print("\nVerification:")
    original_bbox = original_coco["annotations"][0]["bbox"]
    final_bbox = final_coco["annotations"][0]["bbox"]
    max_diff = max(abs(o - f) for o, f in zip(original_bbox, final_bbox))
    print(f"  Maximum coordinate difference: {max_diff:.2f} pixels")
    print(
        f"  Round-trip successful: {max_diff < 1.0}"
    )  # Should be < 1 pixel due to rounding


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("COCO S3 Loader - Format Converter Examples")
    print("=" * 60)

    example_coco_to_label_studio()
    example_label_studio_to_coco()
    example_round_trip_conversion()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
