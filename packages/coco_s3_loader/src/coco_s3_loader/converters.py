"""
Converters between COCO S3 format and Label Studio JSON format.

This module provides functions to convert between COCO-format annotations
with S3 URLs and Label Studio's JSON export format for object detection tasks.
"""

import json
from typing import Any, Dict, List, Optional


def coco_to_label_studio(
    coco_data: Dict[str, Any],
    project_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert COCO format annotations to Label Studio JSON format.

    Args:
        coco_data: COCO format dictionary with 'images', 'annotations', and 'categories'
        project_id: Optional project ID to include in the output

    Returns:
        List of Label Studio task dictionaries

    Example:
        >>> coco_data = {
        ...     "images": [{"id": 1, "file_name": "s3://bucket/img.jpg", "width": 640, "height": 480}],
        ...     "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200]}],
        ...     "categories": [{"id": 1, "name": "person"}]
        ... }
        >>> tasks = coco_to_label_studio(coco_data)
    """
    # Build category lookup
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    # Group annotations by image_id
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Convert images to Label Studio tasks
    tasks = []
    for idx, image in enumerate(coco_data.get("images", [])):
        image_id = image["id"]
        image_url = image["file_name"]
        width = image.get("width", 0)
        height = image.get("height", 0)

        # Build annotation results
        results = []
        for ann in annotations_by_image.get(image_id, []):
            # COCO bbox format: [x, y, width, height] in pixels
            x, y, w, h = ann["bbox"]

            # Convert to Label Studio format (percentages)
            x_percent = (x / width * 100) if width > 0 else 0
            y_percent = (y / height * 100) if height > 0 else 0
            width_percent = (w / width * 100) if width > 0 else 0
            height_percent = (h / height * 100) if height > 0 else 0

            category_name = categories.get(ann["category_id"], "unknown")

            result = {
                "id": f"result_{ann['id']}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [category_name],
                },
            }
            results.append(result)

        # Create Label Studio task
        task = {
            "id": image_id,
            "data": {"image": image_url},
            "annotations": [
                {
                    "id": f"annotation_{image_id}",
                    "result": results,
                }
            ] if results else [],
        }

        # Add optional project field
        if project_id is not None:
            task["project"] = project_id

        tasks.append(task)

    return tasks


def label_studio_to_coco(
    tasks: List[Dict[str, Any]],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert Label Studio JSON format to COCO format annotations.

    Args:
        tasks: List of Label Studio task dictionaries
        categories: Optional list of category names. If not provided,
                   categories will be extracted from the annotations.

    Returns:
        COCO format dictionary with 'images', 'annotations', and 'categories'

    Example:
        >>> tasks = [{
        ...     "id": 1,
        ...     "data": {"image": "s3://bucket/img.jpg"},
        ...     "annotations": [{
        ...         "result": [{
        ...             "type": "rectanglelabels",
        ...             "value": {
        ...                 "x": 10, "y": 20, "width": 30, "height": 40,
        ...                 "rectanglelabels": ["person"]
        ...             },
        ...             "original_width": 640,
        ...             "original_height": 480
        ...         }]
        ...     }]
        ... }]
        >>> coco_data = label_studio_to_coco(tasks)
    """
    # Collect all unique category names
    category_names = set(categories) if categories else set()
    if not categories:
        # Extract categories from annotations
        for task in tasks:
            for annotation in task.get("annotations", []):
                for result in annotation.get("result", []):
                    if result.get("type") == "rectanglelabels":
                        labels = result.get("value", {}).get("rectanglelabels", [])
                        category_names.update(labels)

    # Build category list with IDs
    category_list = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(sorted(category_names))
    ]
    category_name_to_id = {cat["name"]: cat["id"] for cat in category_list}

    # Convert tasks to COCO format
    images = []
    annotations = []
    annotation_id = 1

    for task in tasks:
        task_id = task.get("id", 0)
        image_url = task.get("data", {}).get("image", "")

        # Get image dimensions from first annotation result if available
        width = None
        height = None
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") == "rectanglelabels":
                    width = result.get("original_width")
                    height = result.get("original_height")
                    if width and height:
                        break
            if width and height:
                break

        # If dimensions not found in results, set defaults
        if width is None or height is None:
            width = 0
            height = 0

        # Create image entry
        image_entry = {
            "id": task_id,
            "file_name": image_url,
            "width": width,
            "height": height,
        }
        images.append(image_entry)

        # Process annotations
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") == "rectanglelabels":
                    value = result.get("value", {})

                    # Get original dimensions from result
                    original_width = result.get("original_width", width)
                    original_height = result.get("original_height", height)

                    # Label Studio format: percentages
                    x_percent = value.get("x", 0)
                    y_percent = value.get("y", 0)
                    width_percent = value.get("width", 0)
                    height_percent = value.get("height", 0)

                    # Convert to COCO format (pixels)
                    x = (x_percent / 100) * original_width
                    y = (y_percent / 100) * original_height
                    w = (width_percent / 100) * original_width
                    h = (height_percent / 100) * original_height

                    # Get category
                    labels = value.get("rectanglelabels", [])
                    if not labels:
                        continue

                    category_name = labels[0]  # Take first label
                    category_id = category_name_to_id.get(category_name, 1)

                    # Create annotation entry
                    ann_entry = {
                        "id": annotation_id,
                        "image_id": task_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                    annotations.append(ann_entry)
                    annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": category_list,
    }


def coco_file_to_label_studio_file(
    coco_file_path: str,
    output_file_path: str,
    project_id: Optional[int] = None,
) -> None:
    """
    Convert a COCO JSON file to Label Studio JSON file.

    Args:
        coco_file_path: Path to input COCO JSON file
        output_file_path: Path to output Label Studio JSON file
        project_id: Optional project ID to include in the output

    Example:
        >>> coco_file_to_label_studio_file(
        ...     "coco_annotations.json",
        ...     "label_studio_tasks.json"
        ... )
    """
    with open(coco_file_path, "r") as f:
        coco_data = json.load(f)

    tasks = coco_to_label_studio(coco_data, project_id=project_id)

    with open(output_file_path, "w") as f:
        json.dump(tasks, f, indent=2)


def label_studio_file_to_coco_file(
    label_studio_file_path: str,
    output_file_path: str,
    categories: Optional[List[str]] = None,
) -> None:
    """
    Convert a Label Studio JSON file to COCO JSON file.

    Args:
        label_studio_file_path: Path to input Label Studio JSON file
        output_file_path: Path to output COCO JSON file
        categories: Optional list of category names

    Example:
        >>> label_studio_file_to_coco_file(
        ...     "label_studio_tasks.json",
        ...     "coco_annotations.json",
        ...     categories=["person", "car"]
        ... )
    """
    with open(label_studio_file_path, "r") as f:
        tasks = json.load(f)

    coco_data = label_studio_to_coco(tasks, categories=categories)

    with open(output_file_path, "w") as f:
        json.dump(coco_data, f, indent=2)
