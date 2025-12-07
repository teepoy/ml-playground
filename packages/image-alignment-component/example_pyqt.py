"""
Example usage of the PyQt6 Image Alignment Component for COCO Dataset
"""
from image_alignment_component import COCOImageAlignmentApp
from PyQt6.QtWidgets import QApplication
import sys


def main():
    app = QApplication(sys.argv)
    window = COCOImageAlignmentApp()
    window.show()
    print("PyQt6 Image Alignment Tool is running...")
    print("Controls:")
    print("- Use arrow keys to move the overlay image")
    print("- Press Enter to submit alignment")
    print("- Select different images from the dropdowns")
    print("- Use sliders for fine adjustments")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()