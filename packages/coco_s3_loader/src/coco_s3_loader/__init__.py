"""
COCO S3 Loader - PyTorch-compatible dataloader for COCO datasets with S3 URL support.

This package provides a PyTorch Dataset implementation for loading COCO-format
image datasets where image paths are S3 URLs. It includes error handling for
inaccessible URLs, support for image transformations, and full PyTorch DataLoader
compatibility for batching.

It also includes converters between COCO format and Label Studio JSON format.
"""

from .converters import (
    coco_file_to_label_studio_file,
    coco_to_label_studio,
    label_studio_file_to_coco_file,
    label_studio_to_coco,
)
from .dataset import CocoS3Dataset

__version__ = "0.1.0"
__all__ = [
    "CocoS3Dataset",
    "coco_to_label_studio",
    "label_studio_to_coco",
    "coco_file_to_label_studio_file",
    "label_studio_file_to_coco_file",
]
