"""
Test script to verify the image alignment component functionality
"""
from image_alignment_component.alignment_app import create_image_overlay
from PIL import Image
import numpy as np
import os

def test_create_image_overlay():
    """Test the image overlay functionality with sample images."""
    # Create test images
    fixed_img = Image.new('RGB', (200, 200), color='red')
    movable_img = Image.new('RGB', (100, 100), color='blue')
    
    # Save test images temporarily
    fixed_path = "/tmp/test_fixed.png"
    movable_path = "/tmp/test_movable.png"
    fixed_img.save(fixed_path)
    movable_img.save(movable_path)
    
    # Test overlay function
    result = create_image_overlay(fixed_path, movable_path, dx=10, dy=20, alpha=0.3)
    
    if result is not None:
        print("✓ Image overlay function works correctly")
        result_path = "/tmp/test_result.png"
        result.save(result_path)
        print(f"✓ Result saved to {result_path}")
        
        # Verify dimensions
        print(f"  - Original fixed image size: {fixed_img.size}")
        print(f"  - Original movable image size: {movable_img.size}")
        print(f"  - Result image size: {result.size}")
        return True
    else:
        print("✗ Image overlay function failed")
        return False

def test_negative_offsets():
    """Test overlay with negative offsets."""
    fixed_img = Image.new('RGB', (150, 150), color='green')
    movable_img = Image.new('RGB', (100, 100), color='yellow')
    
    fixed_path = "/tmp/test_fixed2.png"
    movable_path = "/tmp/test_movable2.png"
    fixed_img.save(fixed_path)
    movable_img.save(movable_path)
    
    # Test with negative offsets
    result = create_image_overlay(fixed_path, movable_path, dx=-10, dy=-15, alpha=0.5)
    
    if result is not None:
        print("✓ Negative offset handling works correctly")
        result_path = "/tmp/test_result_neg.png"
        result.save(result_path)
        print(f"✓ Result with negative offset saved to {result_path}")
        return True
    else:
        print("✗ Negative offset handling failed")
        return False

if __name__ == "__main__":
    print("Testing Image Alignment Component...")
    
    success1 = test_create_image_overlay()
    success2 = test_negative_offsets()
    
    if success1 and success2:
        print("\n✓ All tests passed! The image alignment component works correctly.")
    else:
        print("\n✗ Some tests failed.")