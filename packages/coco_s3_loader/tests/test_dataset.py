"""
Unit tests for CocoS3Dataset.

These tests cover:
- JSON parsing
- S3 URL handling
- Error handling for inaccessible URLs
- Image transformations
- Batching with DataLoader
"""

import json
import os
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from coco_s3_loader import CocoS3Dataset


# Test fixtures
@pytest.fixture
def sample_coco_data():
    """Create sample COCO JSON data."""
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
                "height": 480,
                "width": 640,
            },
            {
                "id": 3,
                "file_name": "s3://test-bucket/images/img3.jpg",
                "height": 480,
                "width": 640,
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
                "bbox": [300, 300, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 150, 150],
                "area": 22500,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }


@pytest.fixture
def temp_annotation_file(sample_coco_data):
    """Create a temporary COCO annotation file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_coco_data, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    mock_client = Mock()
    
    def get_object_side_effect(Bucket, Key):
        # Create a dummy image
        img = Image.new("RGB", (640, 480), color="red")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        
        return {"Body": Mock(read=lambda: img_byte_arr.getvalue())}
    
    mock_client.get_object.side_effect = get_object_side_effect
    return mock_client


@pytest.fixture
def simple_transform():
    """Create a simple transform pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# Test cases
class TestCocoS3DatasetInitialization:
    """Test dataset initialization."""
    
    def test_initialization_success(self, temp_annotation_file, mock_s3_client):
        """Test successful dataset initialization."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        assert len(dataset) == 3
        assert len(dataset.images) == 3
        assert len(dataset.annotations) == 3
        assert len(dataset.categories) == 2
    
    def test_invalid_annotation_file(self, mock_s3_client):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            CocoS3Dataset(
                annotation_file="nonexistent.json",
                s3_client=mock_s3_client,
            )
    
    def test_invalid_handle_errors(self, temp_annotation_file, mock_s3_client):
        """Test initialization with invalid handle_errors parameter."""
        with pytest.raises(ValueError, match="handle_errors must be"):
            CocoS3Dataset(
                annotation_file=temp_annotation_file,
                handle_errors="invalid",
                s3_client=mock_s3_client,
            )
    
    def test_annotations_by_image_mapping(self, temp_annotation_file, mock_s3_client):
        """Test that annotations are correctly mapped to images."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        # Image 1 should have 2 annotations
        assert len(dataset.annotations_by_image[1]) == 2
        # Image 2 should have 1 annotation
        assert len(dataset.annotations_by_image[2]) == 1
        # Image 3 should have 0 annotations
        assert len(dataset.annotations_by_image.get(3, [])) == 0


class TestCocoS3DatasetLoading:
    """Test image loading functionality."""
    
    def test_load_image_success(self, temp_annotation_file, mock_s3_client):
        """Test successful image loading."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        image, annotation = dataset[0]
        
        assert isinstance(image, Image.Image)
        assert annotation["image_info"]["id"] == 1
        assert len(annotation["annotations"]) == 2
    
    def test_load_image_with_transform(
        self, temp_annotation_file, mock_s3_client, simple_transform
    ):
        """Test image loading with transforms."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            transform=simple_transform,
            s3_client=mock_s3_client,
        )
        
        image, annotation = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)  # C, H, W
    
    def test_s3_url_parsing(self, temp_annotation_file, mock_s3_client):
        """Test S3 URL parsing."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        # Load an image to trigger S3 URL parsing
        dataset[0]
        
        # Verify S3 client was called with correct bucket and key
        mock_s3_client.get_object.assert_called()
        call_args = mock_s3_client.get_object.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Key"] == "images/img1.jpg"
    
    def test_invalid_s3_url_format(self, temp_annotation_file):
        """Test handling of invalid S3 URL format."""
        # Create dataset with invalid URL
        invalid_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "http://example.com/image.jpg",  # Not S3 URL
                    "height": 480,
                    "width": 640,
                }
            ],
            "annotations": [],
            "categories": [],
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            mock_client = Mock()
            dataset = CocoS3Dataset(
                annotation_file=temp_path,
                handle_errors="raise",
                s3_client=mock_client,
            )
            
            with pytest.raises(ValueError, match="Invalid S3 URL format"):
                dataset[0]
        finally:
            os.unlink(temp_path)


class TestCocoS3DatasetErrorHandling:
    """Test error handling modes."""
    
    def test_error_handling_raise(self, temp_annotation_file):
        """Test 'raise' error handling mode."""
        mock_client = Mock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "get_object"
        )
        
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            handle_errors="raise",
            s3_client=mock_client,
        )
        
        with pytest.raises(ClientError):
            dataset[0]
    
    def test_error_handling_return_none(self, temp_annotation_file):
        """Test 'return_none' error handling mode."""
        mock_client = Mock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "get_object"
        )
        
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            handle_errors="return_none",
            s3_client=mock_client,
        )
        
        result = dataset[0]
        assert result is None
    
    def test_error_handling_skip(self, temp_annotation_file):
        """Test 'skip' error handling mode."""
        mock_client = Mock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "get_object"
        )
        
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            handle_errors="skip",
            s3_client=mock_client,
        )
        
        # In skip mode, failed items return None when accessed
        result = dataset[0]
        assert result is None
    
    def test_no_credentials_error(self, temp_annotation_file):
        """Test handling of missing AWS credentials."""
        mock_client = Mock()
        mock_client.get_object.side_effect = NoCredentialsError()
        
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            handle_errors="raise",
            s3_client=mock_client,
        )
        
        with pytest.raises(NoCredentialsError):
            dataset[0]


class TestCocoS3DatasetBatching:
    """Test DataLoader batching functionality."""
    
    def test_dataloader_batching(
        self, temp_annotation_file, mock_s3_client, simple_transform
    ):
        """Test using dataset with PyTorch DataLoader."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            transform=simple_transform,
            s3_client=mock_s3_client,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: x,  # Use custom collate for dict annotations
        )
        
        batches = list(dataloader)
        assert len(batches) == 2  # 3 images / batch_size 2 = 2 batches
        assert len(batches[0]) == 2  # First batch has 2 items
        assert len(batches[1]) == 1  # Second batch has 1 item
    
    def test_dataloader_shuffle(
        self, temp_annotation_file, mock_s3_client, simple_transform
    ):
        """Test DataLoader with shuffle enabled."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            transform=simple_transform,
            s3_client=mock_s3_client,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x,
        )
        
        # Just verify we can iterate through shuffled data
        items = list(dataloader)
        assert len(items) == 3


class TestCocoS3DatasetUtilityMethods:
    """Test utility methods."""
    
    def test_get_categories(self, temp_annotation_file, mock_s3_client):
        """Test get_categories method."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        categories = dataset.get_categories()
        assert len(categories) == 2
        assert categories[0]["name"] == "person"
        assert categories[1]["name"] == "car"
    
    def test_get_image_info(self, temp_annotation_file, mock_s3_client):
        """Test get_image_info method."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        image_info = dataset.get_image_info(0)
        assert image_info["id"] == 1
        assert image_info["file_name"] == "s3://test-bucket/images/img1.jpg"
        assert image_info["height"] == 480
        assert image_info["width"] == 640
    
    def test_get_image_info_out_of_range(self, temp_annotation_file, mock_s3_client):
        """Test get_image_info with invalid index."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        with pytest.raises(IndexError):
            dataset.get_image_info(999)
    
    def test_len(self, temp_annotation_file, mock_s3_client):
        """Test __len__ method."""
        dataset = CocoS3Dataset(
            annotation_file=temp_annotation_file,
            s3_client=mock_s3_client,
        )
        
        assert len(dataset) == 3
