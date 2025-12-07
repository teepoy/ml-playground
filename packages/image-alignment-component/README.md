# Image Alignment Component

A dual-purpose image alignment tool that allows users to align two images by overlaying them and using keyboard controls to adjust positioning. Available in both Gradio web and PyQt6 desktop versions.

## Features

1. **Image Overlay**: Takes two images and overlays them with 30% alpha for the movable image
2. **Keyboard Controls**: Use arrow keys to move the movable image pixel by pixel
3. **Mouse Controls**: Buttons for precise movement control
4. **Submission**: Press Enter (or click Submit) to return the alignment shift values
5. **COCO Dataset Integration**: Select images from a COCO dataset
6. **Fine Adjustment**: Slider controls for precise alignment

## Versions Available

### Web Version (Gradio)
- Browser-based interface
- Upload any two images
- Works in web browsers

### Desktop Version (PyQt6)
- Native desktop application
- COCO dataset integration
- More responsive controls
- Available on multiple platforms

## Installation

```bash
pip install image-alignment-component
```

## Usage Examples

### Gradio Web Version
```python
from image_alignment_component import alignment_interface

# Launch the web interface
app = alignment_interface()
app.launch()
```

### PyQt6 Desktop Version
```python
from image_alignment_component import COCOImageAlignmentApp
from PyQt6.QtWidgets import QApplication
import sys

# Create and run the desktop application
app = QApplication(sys.argv)
window = COCOImageAlignmentApp()
window.show()
sys.exit(app.exec())
```

## Controls

- **Arrow Keys**: Move movable image by 1 pixel in respective direction
- **Enter Key**: Submit the current alignment values
- **Directional Buttons**: Alternative mouse-based movement
- **Sliders**: Fine-tune X and Y positions
- **Image Selection**: Choose from COCO dataset images

## API

The component returns a JSON object with the format:
```json
{
  "x": <horizontal_shift_pixels>,
  "y": <vertical_shift_pixels>
}
```

This can be used to measure alignment accuracy between two images for various computer vision tasks.