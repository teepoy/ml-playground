"""
Example usage of the Image Alignment Component (Gradio Web Version)
"""
from image_alignment_component import alignment_interface

# Create and launch the alignment interface
if __name__ == "__main__":
    app = alignment_interface()

    print("Image Alignment Tool is running...")
    print("Open your browser and go to: http://localhost:7860")
    print("\nInstructions:")
    print("1. Upload a fixed image and a movable image")
    print("2. Click 'Load Images' to initialize the overlay")
    print("3. Use arrow keys to adjust the overlay position")
    print("4. Press Enter when images are properly aligned")
    print("5. The alignment shift values will be displayed")

    app.launch()