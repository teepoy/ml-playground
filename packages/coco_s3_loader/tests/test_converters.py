"""
Unit tests for COCO to Label Studio format converters.

These tests cover:
- COCO to Label Studio conversion
- Label Studio to COCO conversion
- File-based conversions
- Edge cases and error handling
"""

import json
import os
import tempfile

import pytest

from coco_s3_loader.converters import (
    coco_file_to_label_studio_file,
    coco_to_label_studio,
    label_studio_file_to_coco_file,
    label_studio_to_coco,
)


# Test fixtures
@pytest.fixture
def sample_coco_data():
    """Create sample COCO data."""
    return {
        "images": [
            {
                "id": 1,
                "file_name": "s3://test-bucket/images/img1.jpg",
                "height": 480,
                "width": 640,
            },
            {
                "id": 2,
                "file_name": "s3://test-bucket/images/img2.jpg",
                "height": 600,
                "width": 800,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 200],
                "area": 40000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [320, 240, 160, 120],
                "area": 19200,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }


@pytest.fixture
def sample_label_studio_tasks():
    """Create sample Label Studio tasks."""
    return [
        {
            "id": 1,
            "data": {"image": "s3://test-bucket/images/img1.jpg"},
            "annotations": [
                {
                    "id": "annotation_1",
                    "result": [
                        {
                            "id": "result_1",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 640,
                            "original_height": 480,
                            "value": {
                                "x": 15.625,  # 100/640 * 100
                                "y": 20.833333333333336,  # 100/480 * 100
                                "width": 31.25,  # 200/640 * 100
                                "height": 41.666666666666664,  # 200/480 * 100
                                "rotation": 0,
                                "rectanglelabels": ["person"],
                            },
                        },
                        {
                            "id": "result_2",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 640,
                            "original_height": 480,
                            "value": {
                                "x": 50.0,  # 320/640 * 100
                                "y": 50.0,  # 240/480 * 100
                                "width": 25.0,  # 160/640 * 100
                                "height": 25.0,  # 120/480 * 100
                                "rotation": 0,
                                "rectanglelabels": ["car"],
                            },
                        },
                    ],
                }
            ],
        },
        {
            "id": 2,
            "data": {"image": "s3://test-bucket/images/img2.jpg"},
            "annotations": [
                {
                    "id": "annotation_2",
                    "result": [
                        {
                            "id": "result_3",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 800,
                            "original_height": 600,
                            "value": {
                                "x": 6.25,  # 50/800 * 100
                                "y": 8.333333333333334,  # 50/600 * 100
                                "width": 12.5,  # 100/800 * 100
                                "height": 16.666666666666668,  # 100/600 * 100
                                "rotation": 0,
                                "rectanglelabels": ["person"],
                            },
                        },
                    ],
                }
            ],
        },
    ]


# Test cases
class TestCocoToLabelStudio:
    """Test COCO to Label Studio conversion."""

    def test_basic_conversion(self, sample_coco_data):
        """Test basic conversion from COCO to Label Studio."""
        tasks = coco_to_label_studio(sample_coco_data)

        assert len(tasks) == 2
        assert tasks[0]["id"] == 1
        assert tasks[0]["data"]["image"] == "s3://test-bucket/images/img1.jpg"
        assert len(tasks[0]["annotations"]) == 1
        assert len(tasks[0]["annotations"][0]["result"]) == 2

    def test_bbox_conversion_to_percentages(self, sample_coco_data):
        """Test that bounding boxes are correctly converted to percentages."""
        tasks = coco_to_label_studio(sample_coco_data)

        # First bbox: [100, 100, 200, 200] on 640x480 image
        result = tasks[0]["annotations"][0]["result"][0]
        value = result["value"]

        # Calculate expected values from input data
        bbox = sample_coco_data["annotations"][0]["bbox"]
        img_width = sample_coco_data["images"][0]["width"]
        img_height = sample_coco_data["images"][0]["height"]
        expected_x = (bbox[0] / img_width) * 100
        expected_y = (bbox[1] / img_height) * 100
        expected_width = (bbox[2] / img_width) * 100
        expected_height = (bbox[3] / img_height) * 100

        # Verify conversion accuracy
        assert abs(value["x"] - expected_x) < 0.001
        assert abs(value["y"] - expected_y) < 0.001
        assert abs(value["width"] - expected_width) < 0.001
        assert abs(value["height"] - expected_height) < 0.001

    def test_category_labels(self, sample_coco_data):
        """Test that category labels are correctly mapped."""
        tasks = coco_to_label_studio(sample_coco_data)

        result1 = tasks[0]["annotations"][0]["result"][0]
        result2 = tasks[0]["annotations"][0]["result"][1]

        assert result1["value"]["rectanglelabels"] == ["person"]
        assert result2["value"]["rectanglelabels"] == ["car"]

    def test_original_dimensions(self, sample_coco_data):
        """Test that original dimensions are preserved."""
        tasks = coco_to_label_studio(sample_coco_data)

        result = tasks[0]["annotations"][0]["result"][0]
        assert result["original_width"] == 640
        assert result["original_height"] == 480

    def test_empty_annotations(self):
        """Test conversion with empty annotations."""
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "s3://bucket/img.jpg",
                    "width": 640,
                    "height": 480,
                }
            ],
            "annotations": [],
            "categories": [],
        }

        tasks = coco_to_label_studio(coco_data)

        assert len(tasks) == 1
        assert len(tasks[0]["annotations"]) == 0

    def test_with_project_id(self, sample_coco_data):
        """Test conversion with project ID."""
        tasks = coco_to_label_studio(sample_coco_data, project_id=42)

        assert tasks[0]["project"] == 42
        assert tasks[1]["project"] == 42

    def test_without_project_id(self, sample_coco_data):
        """Test conversion without project ID."""
        tasks = coco_to_label_studio(sample_coco_data)

        assert "project" not in tasks[0]
        assert "project" not in tasks[1]


class TestLabelStudioToCoco:
    """Test Label Studio to COCO conversion."""

    def test_basic_conversion(self, sample_label_studio_tasks):
        """Test basic conversion from Label Studio to COCO."""
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        assert len(coco_data["images"]) == 2
        assert len(coco_data["annotations"]) == 3
        assert len(coco_data["categories"]) == 2

    def test_bbox_conversion_to_pixels(self, sample_label_studio_tasks):
        """Test that bounding boxes are correctly converted to pixels."""
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        # First annotation: x=15.625%, y≈20.833%, w=31.25%, h≈41.667% on 640x480
        ann = coco_data["annotations"][0]
        bbox = ann["bbox"]

        # Expected: x=100, y=100, w=200, h=200
        assert abs(bbox[0] - 100) < 1
        assert abs(bbox[1] - 100) < 1
        assert abs(bbox[2] - 200) < 1
        assert abs(bbox[3] - 200) < 1

    def test_category_extraction(self, sample_label_studio_tasks):
        """Test that categories are correctly extracted."""
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        category_names = {cat["name"] for cat in coco_data["categories"]}
        assert "person" in category_names
        assert "car" in category_names

    def test_category_ids(self, sample_label_studio_tasks):
        """Test that category IDs are assigned correctly."""
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        # Categories should be sorted alphabetically and assigned sequential IDs
        categories = sorted(coco_data["categories"], key=lambda x: x["id"])
        assert categories[0]["name"] == "car"  # Alphabetically first
        assert categories[0]["id"] == 1
        assert categories[1]["name"] == "person"
        assert categories[1]["id"] == 2

    def test_with_predefined_categories(self, sample_label_studio_tasks):
        """Test conversion with predefined categories."""
        coco_data = label_studio_to_coco(
            sample_label_studio_tasks, categories=["person", "car", "dog"]
        )

        assert len(coco_data["categories"]) == 3
        category_names = {cat["name"] for cat in coco_data["categories"]}
        assert category_names == {"person", "car", "dog"}

    def test_image_dimensions(self, sample_label_studio_tasks):
        """Test that image dimensions are correctly extracted."""
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        img1 = coco_data["images"][0]
        assert img1["width"] == 640
        assert img1["height"] == 480

        img2 = coco_data["images"][1]
        assert img2["width"] == 800
        assert img2["height"] == 600

    def test_empty_tasks(self):
        """Test conversion with empty tasks."""
        coco_data = label_studio_to_coco([])

        assert len(coco_data["images"]) == 0
        assert len(coco_data["annotations"]) == 0
        assert len(coco_data["categories"]) == 0

    def test_task_without_annotations(self):
        """Test conversion with task that has no annotations."""
        tasks = [
            {
                "id": 1,
                "data": {"image": "s3://bucket/img.jpg"},
                "annotations": [],
            }
        ]

        coco_data = label_studio_to_coco(tasks)

        assert len(coco_data["images"]) == 1
        assert len(coco_data["annotations"]) == 0


class TestRoundTripConversion:
    """Test round-trip conversions between formats."""

    # Tolerance for floating point comparisons (in pixels)
    BBOX_TOLERANCE = 1.0

    def test_coco_to_label_studio_to_coco(self, sample_coco_data):
        """Test round-trip conversion: COCO -> Label Studio -> COCO."""
        # Convert to Label Studio
        tasks = coco_to_label_studio(sample_coco_data)

        # Convert back to COCO
        # Need to provide categories to ensure consistent ordering
        original_categories = [cat["name"] for cat in sample_coco_data["categories"]]
        coco_data = label_studio_to_coco(tasks, categories=original_categories)

        # Check structure
        assert len(coco_data["images"]) == len(sample_coco_data["images"])
        assert len(coco_data["annotations"]) == len(sample_coco_data["annotations"])
        assert len(coco_data["categories"]) == len(sample_coco_data["categories"])

        # Check bounding boxes (allowing for small floating point errors)
        for i, original_ann in enumerate(sample_coco_data["annotations"]):
            converted_ann = coco_data["annotations"][i]
            for j in range(4):
                assert (
                    abs(original_ann["bbox"][j] - converted_ann["bbox"][j])
                    < self.BBOX_TOLERANCE
                )

    def test_label_studio_to_coco_to_label_studio(self, sample_label_studio_tasks):
        """Test round-trip conversion: Label Studio -> COCO -> Label Studio."""
        # Convert to COCO
        coco_data = label_studio_to_coco(sample_label_studio_tasks)

        # Convert back to Label Studio
        tasks = coco_to_label_studio(coco_data)

        # Check structure
        assert len(tasks) == len(sample_label_studio_tasks)

        # Check that we have the same number of annotations
        total_original = sum(
            len(task["annotations"][0]["result"])
            for task in sample_label_studio_tasks
            if task.get("annotations")
        )
        total_converted = sum(
            len(task["annotations"][0]["result"])
            for task in tasks
            if task.get("annotations")
        )
        assert total_original == total_converted


class TestFileBasedConversions:
    """Test file-based conversion functions."""

    def test_coco_file_to_label_studio_file(self, sample_coco_data):
        """Test file-based COCO to Label Studio conversion."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as coco_file:
            json.dump(sample_coco_data, coco_file)
            coco_path = coco_file.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as ls_file:
                ls_path = ls_file.name

            try:
                coco_file_to_label_studio_file(coco_path, ls_path)

                # Read and verify output
                with open(ls_path, "r") as f:
                    tasks = json.load(f)

                assert len(tasks) == 2
                assert tasks[0]["data"]["image"] == "s3://test-bucket/images/img1.jpg"
            finally:
                os.unlink(ls_path)
        finally:
            os.unlink(coco_path)

    def test_label_studio_file_to_coco_file(self, sample_label_studio_tasks):
        """Test file-based Label Studio to COCO conversion."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as ls_file:
            json.dump(sample_label_studio_tasks, ls_file)
            ls_path = ls_file.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as coco_file:
                coco_path = coco_file.name

            try:
                label_studio_file_to_coco_file(ls_path, coco_path)

                # Read and verify output
                with open(coco_path, "r") as f:
                    coco_data = json.load(f)

                assert len(coco_data["images"]) == 2
                assert len(coco_data["annotations"]) == 3
                assert len(coco_data["categories"]) == 2
            finally:
                os.unlink(coco_path)
        finally:
            os.unlink(ls_path)

    def test_file_conversion_with_project_id(self, sample_coco_data):
        """Test file conversion with project ID."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as coco_file:
            json.dump(sample_coco_data, coco_file)
            coco_path = coco_file.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as ls_file:
                ls_path = ls_file.name

            try:
                coco_file_to_label_studio_file(coco_path, ls_path, project_id=100)

                with open(ls_path, "r") as f:
                    tasks = json.load(f)

                assert tasks[0]["project"] == 100
            finally:
                os.unlink(ls_path)
        finally:
            os.unlink(coco_path)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_dimensions(self):
        """Test handling of zero dimensions."""
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "s3://bucket/img.jpg",
                    "width": 0,
                    "height": 0,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 0, 0],
                    "area": 0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
        }

        tasks = coco_to_label_studio(coco_data)
        assert len(tasks) == 1
        # Should handle division by zero gracefully
        assert tasks[0]["annotations"][0]["result"][0]["value"]["x"] == 0

    def test_missing_category(self):
        """Test handling of missing category."""
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "s3://bucket/img.jpg",
                    "width": 640,
                    "height": 480,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 999,  # Non-existent category
                    "bbox": [100, 100, 200, 200],
                    "area": 40000,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "human"}],
        }

        tasks = coco_to_label_studio(coco_data)
        # Should use "unknown" for missing categories
        assert (
            tasks[0]["annotations"][0]["result"][0]["value"]["rectanglelabels"][0]
            == "unknown"
        )

    def test_multiple_labels_per_bbox(self):
        """Test Label Studio task with multiple labels (takes first)."""
        tasks = [
            {
                "id": 1,
                "data": {"image": "s3://bucket/img.jpg"},
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "rectanglelabels",
                                "original_width": 640,
                                "original_height": 480,
                                "value": {
                                    "x": 10,
                                    "y": 10,
                                    "width": 20,
                                    "height": 20,
                                    "rectanglelabels": [
                                        "label1",
                                        "label2",
                                    ],  # Multiple labels
                                },
                            }
                        ]
                    }
                ],
            }
        ]

        coco_data = label_studio_to_coco(tasks)
        # Should extract categories from multi-label annotations
        category_names = {cat["name"] for cat in coco_data["categories"]}
        assert "label1" in category_names
        assert "label2" in category_names
        # Annotation should use first label
        assert coco_data["annotations"][0]["category_id"] in [1, 2]

    def test_task_without_dimensions(self):
        """Test Label Studio task without original dimensions."""
        tasks = [
            {
                "id": 1,
                "data": {"image": "s3://bucket/img.jpg"},
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "rectanglelabels",
                                # No original_width/original_height
                                "value": {
                                    "x": 10,
                                    "y": 10,
                                    "width": 20,
                                    "height": 20,
                                    "rectanglelabels": ["object"],
                                },
                            }
                        ]
                    }
                ],
            }
        ]

        coco_data = label_studio_to_coco(tasks)
        # Should handle missing dimensions by setting to 0
        assert coco_data["images"][0]["width"] == 0
        assert coco_data["images"][0]["height"] == 0
