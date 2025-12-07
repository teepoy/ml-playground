"""
COCO S3 Dataset implementation.

This module provides a PyTorch Dataset class for loading COCO-format datasets
with S3 URLs as image paths. It handles S3 image loading, error handling,
and image transformations.
"""

import json
import logging
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3
import torch
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CocoS3Dataset(Dataset):
    """
    PyTorch Dataset for COCO-format annotations with S3 image URLs.

    This dataset loads images from S3 URLs specified in COCO JSON annotations
    and provides error handling for inaccessible images. It supports standard
    PyTorch transforms and can be used with DataLoader for batching.

    Attributes:
        annotation_file (str): Path to COCO JSON annotation file
        transform (Optional[Callable]): Optional transform to apply to images
        handle_errors (str): Error handling mode ('skip', 'raise', 'return_none')
        s3_client: Boto3 S3 client for downloading images
        images (List[Dict]): List of image metadata from COCO JSON
        annotations_by_image (Dict): Mapping of image_id to annotations
        valid_indices (List[int]): Indices of valid images (after error handling)

    Example:
        >>> from coco_s3_loader import CocoS3Dataset
        >>> from torchvision import transforms
        >>>
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>>
        >>> dataset = CocoS3Dataset(
        ...     annotation_file='annotations.json',
        ...     transform=transform,
        ...     handle_errors='skip'
        ... )
        >>>
        >>> image, annotation = dataset[0]
    """

    def __init__(
        self,
        annotation_file: str,
        transform: Optional[Callable] = None,
        handle_errors: str = "skip",
        s3_client: Optional[Any] = None,
    ):
        """
        Initialize the COCO S3 Dataset.

        Args:
            annotation_file: Path to COCO JSON annotation file
            transform: Optional callable transform to apply to images
            handle_errors: How to handle image load errors:
                - 'skip': Return None for failed images (same as 'return_none').
                         These should be filtered out when processing batches.
                - 'raise': Raise exception on load failure
                - 'return_none': Return None for failed images
            s3_client: Optional pre-configured boto3 S3 client. If None,
                      a new client will be created using default credentials.

        Raises:
            FileNotFoundError: If annotation file doesn't exist
            ValueError: If handle_errors is not one of the valid options
            json.JSONDecodeError: If annotation file is not valid JSON
        """
        if handle_errors not in ("skip", "raise", "return_none"):
            raise ValueError(
                f"handle_errors must be 'skip', 'raise', or 'return_none', "
                f"got '{handle_errors}'"
            )

        self.annotation_file = annotation_file
        self.transform = transform
        self.handle_errors = handle_errors

        # Initialize S3 client
        self.s3_client = s3_client if s3_client is not None else boto3.client("s3")

        # Load annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.images = coco_data.get("images", [])
        self.annotations = coco_data.get("annotations", [])
        self.categories = coco_data.get("categories", [])

        # Build image_id to annotations mapping
        self.annotations_by_image: Dict[int, List[Dict]] = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)

        # Track all indices as valid - images are loaded lazily on access
        # Failed loads will return None in 'skip' and 'return_none' modes
        self.valid_indices = list(range(len(self.images)))

        logger.info(
            f"Loaded COCO dataset with {len(self.images)} images, "
            f"{len(self.annotations)} annotations, and {len(self.categories)} categories"
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, Dict[str, Any]], None]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image, annotation_dict) where:
                - image is a torch.Tensor (if transform includes ToTensor)
                  or PIL Image
                - annotation_dict contains image metadata and annotations
            Returns None if handle_errors='return_none' and load fails.

        Raises:
            IndexError: If idx is out of range
            Exception: If handle_errors='raise' and image load fails
        """
        if idx >= len(self.valid_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        actual_idx = self.valid_indices[idx]
        image_info = self.images[actual_idx]

        try:
            # Load image from S3
            image = self._load_image_from_s3(image_info["file_name"])

            # Apply transforms if provided
            if self.transform is not None:
                image = self.transform(image)

            # Prepare annotation data
            image_id = image_info["id"]
            annotation_data = {
                "image_info": image_info,
                "annotations": self.annotations_by_image.get(image_id, []),
            }

            return image, annotation_data

        except Exception as e:
            logger.warning(
                f"Failed to load image {image_info.get('file_name', 'unknown')}: {e}"
            )

            if self.handle_errors == "raise":
                raise
            elif self.handle_errors == "return_none":
                return None
            else:  # skip mode
                # This shouldn't happen in normal iteration since we skip invalid indices
                # But if someone accesses by index directly, return None
                return None

    def _load_image_from_s3(self, s3_url: str) -> Image.Image:
        """
        Load an image from an S3 URL.

        Args:
            s3_url: S3 URL in format 's3://bucket-name/key/path'

        Returns:
            PIL Image object

        Raises:
            ValueError: If URL is not a valid S3 URL
            ClientError: If S3 access fails
            NoCredentialsError: If AWS credentials are not configured
        """
        # Parse S3 URL
        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        # Remove 's3://' prefix and split into bucket and key
        s3_path = s3_url[5:]  # Remove 's3://'
        parts = s3_path.split("/", 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        bucket, key = parts

        # Download image from S3
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise ClientError(
                {"Error": {"Code": error_code, "Message": str(e)}}, "get_object"
            ) from e
        except NoCredentialsError as e:
            raise NoCredentialsError() from e

    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get the list of categories from the COCO dataset.

        Returns:
            List of category dictionaries with 'id', 'name', and 'supercategory'
        """
        return self.categories

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for an image without loading it.

        Args:
            idx: Index of the image

        Returns:
            Dictionary with image metadata (id, file_name, height, width, etc.)

        Raises:
            IndexError: If idx is out of range
        """
        if idx >= len(self.valid_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        actual_idx = self.valid_indices[idx]
        return self.images[actual_idx]
