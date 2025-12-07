"""
Image Alignment Component for Gradio and PyQt6
A custom component that allows aligning two images using keyboard controls
"""
from .alignment_app import alignment_interface, create_image_overlay
from .coco_alignment_app import COCOImageAlignmentApp

__version__ = "0.1.0"
__all__ = ['alignment_interface', 'create_image_overlay', 'COCOImageAlignmentApp']
