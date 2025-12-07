"""
COCO S3 Loader - PyTorch-compatible dataloader for COCO datasets with S3 URL support.

This package provides a PyTorch Dataset implementation for loading COCO-format
image datasets where image paths are S3 URLs. It includes error handling for
inaccessible URLs, support for image transformations, and full PyTorch DataLoader
compatibility for batching.
"""

from .dataset import CocoS3Dataset

__version__ = "0.1.0"
__all__ = ["CocoS3Dataset"]
