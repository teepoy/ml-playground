"""
Image Alignment Component for Gradio
A custom component that allows aligning two images using keyboard controls
"""
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64


def create_image_overlay(fixed_img_path, movable_img_path, dx=0, dy=0, alpha=0.3):
    """Create an overlay image with the movable image positioned at dx, dy."""
    if fixed_img_path is None or movable_img_path is None:
        return None
    
    # Load images
    fixed_img = Image.open(fixed_img_path).convert('RGBA')
    movable_img = Image.open(movable_img_path).convert('RGBA')
    
    # Make movable image semi-transparent
    movable_img = movable_img.copy()
    movable_array = np.array(movable_img)
    movable_array[:, :, 3] = int(255 * alpha)  # Set alpha channel
    movable_img = Image.fromarray(movable_array)
    
    # Create a canvas with the size of the fixed image
    result = fixed_img.convert('RGBA')
    
    # Paste the movable image at the specified offset
    paste_x = dx
    paste_y = dy
    
    # Handle negative offsets by expanding canvas if needed
    if paste_x < 0 or paste_y < 0:
        result = Image.new('RGBA', 
                          (max(result.width, movable_img.width + abs(paste_x)), 
                           max(result.height, movable_img.height + abs(paste_y))), 
                          (0, 0, 0, 0))
        result.paste(fixed_img, (-min(0, paste_x), -min(0, paste_y)))
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
    
    result.paste(movable_img, (paste_x, paste_y), movable_img)
    
    # Convert back to RGB for Gradio compatibility
    result = result.convert('RGB')
    
    return result


def alignment_interface():
    """Create the image alignment interface."""
    with gr.Blocks(title="Image Alignment Tool") as demo:
        gr.Markdown("""
        # Image Alignment Accuracy Collector
        
        Upload two images and use the controls to align them. 
        The movable image has 30% opacity to enable overlay comparison.
        """)
        
        # State variables
        shift_x = gr.State(value=0)
        shift_y = gr.State(value=0)
        
        with gr.Row():
            with gr.Column(scale=1):
                fixed_image = gr.Image(label="Fixed Image", type="filepath")
                movable_image = gr.Image(label="Movable Image", type="filepath")
                load_btn = gr.Button("Load Images")
            
            with gr.Column(scale=1):
                overlay_output = gr.Image(label="Overlay Preview", interactive=False)
                
                with gr.Row():
                    left_btn = gr.Button("← Left")
                    right_btn = gr.Button("Right →")
                    up_btn = gr.Button("↑ Up")
                    down_btn = gr.Button("↓ Down")
                
                with gr.Row():
                    x_shift_input = gr.Number(label="X Shift", value=0, precision=0)
                    y_shift_input = gr.Number(label="Y Shift", value=0, precision=0)
                    update_btn = gr.Button("Update Position")
                
                with gr.Row():
                    submit_btn = gr.Button("Submit Alignment (Enter)", variant="primary")
                    reset_btn = gr.Button("Reset")
                
                result = gr.JSON(label="Alignment Result")
        
        # Function to update overlay with current shifts
        def update_overlay(fixed_path, movable_path, dx, dy):
            if fixed_path and movable_path:
                overlay = create_image_overlay(fixed_path, movable_path, int(dx), int(dy), alpha=0.3)
                return overlay
            return None
        
        # Function to update shifts and overlay
        def update_shifts(fixed_path, movable_path, dx, dy):
            overlay = create_image_overlay(fixed_path, movable_path, int(dx), int(dy), alpha=0.3)
            return overlay, int(dx), int(dy)
        
        # Keyboard controls using Gradio's key events
        def shift_left(fixed_path, movable_path, current_x, current_y):
            new_x = current_x - 1
            overlay = create_image_overlay(fixed_path, movable_path, new_x, current_y, alpha=0.3)
            return overlay, new_x, current_y
        
        def shift_right(fixed_path, movable_path, current_x, current_y):
            new_x = current_x + 1
            overlay = create_image_overlay(fixed_path, movable_path, new_x, current_y, alpha=0.3)
            return overlay, new_x, current_y
        
        def shift_up(fixed_path, movable_path, current_x, current_y):
            new_y = current_y - 1
            overlay = create_image_overlay(fixed_path, movable_path, current_x, new_y, alpha=0.3)
            return overlay, current_x, new_y
        
        def shift_down(fixed_path, movable_path, current_x, current_y):
            new_y = current_y + 1
            overlay = create_image_overlay(fixed_path, movable_path, current_x, new_y, alpha=0.3)
            return overlay, current_x, new_y
        
        def submit_alignment(dx, dy):
            return {"x": int(dx), "y": int(dy)}
        
        def reset_shifts(fixed_path, movable_path):
            overlay = create_image_overlay(fixed_path, movable_path, 0, 0, alpha=0.3)
            return overlay, 0, 0, {"x": 0, "y": 0}
        
        # Button events
        load_btn.click(
            fn=lambda f, m: update_overlay(f, m, 0, 0),
            inputs=[fixed_image, movable_image],
            outputs=overlay_output
        )
        
        left_btn.click(
            fn=shift_left,
            inputs=[fixed_image, movable_image, shift_x, shift_y],
            outputs=[overlay_output, shift_x, shift_y]
        ).then(
            fn=lambda x, y: (x, y),
            inputs=[shift_x, shift_y],
            outputs=[x_shift_input, y_shift_input]
        )
        
        right_btn.click(
            fn=shift_right,
            inputs=[fixed_image, movable_image, shift_x, shift_y],
            outputs=[overlay_output, shift_x, shift_y]
        ).then(
            fn=lambda x, y: (x, y),
            inputs=[shift_x, shift_y],
            outputs=[x_shift_input, y_shift_input]
        )
        
        up_btn.click(
            fn=shift_up,
            inputs=[fixed_image, movable_image, shift_x, shift_y],
            outputs=[overlay_output, shift_x, shift_y]
        ).then(
            fn=lambda x, y: (x, y),
            inputs=[shift_x, shift_y],
            outputs=[x_shift_input, y_shift_input]
        )
        
        down_btn.click(
            fn=shift_down,
            inputs=[fixed_image, movable_image, shift_x, shift_y],
            outputs=[overlay_output, shift_x, shift_y]
        ).then(
            fn=lambda x, y: (x, y),
            inputs=[shift_x, shift_y],
            outputs=[x_shift_input, y_shift_input]
        )
        
        update_btn.click(
            fn=update_shifts,
            inputs=[fixed_image, movable_image, x_shift_input, y_shift_input],
            outputs=[overlay_output, shift_x, shift_y]
        )
        
        submit_btn.click(
            fn=submit_alignment,
            inputs=[shift_x, shift_y],
            outputs=result
        )
        
        reset_btn.click(
            fn=reset_shifts,
            inputs=[fixed_image, movable_image],
            outputs=[overlay_output, shift_x, shift_y, result]
        )
        
        # Define JavaScript for keyboard controls
        keyboard_js = """
        document.addEventListener('keydown', function(e) {
            // Only respond to keys when not in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            // Get all buttons by class or other attributes
            const buttons = Array.from(document.querySelectorAll('button'));
            let targetBtn = null;

            if (e.key === 'ArrowLeft') {
                targetBtn = buttons.find(btn => btn.textContent.includes('← Left'));
            } else if (e.key === 'ArrowRight') {
                targetBtn = buttons.find(btn => btn.textContent.includes('Right →'));
            } else if (e.key === 'ArrowUp') {
                targetBtn = buttons.find(btn => btn.textContent.includes('↑ Up'));
            } else if (e.key === 'ArrowDown') {
                targetBtn = buttons.find(btn => btn.textContent.includes('↓ Down'));
            } else if (e.key === 'Enter') {
                targetBtn = buttons.find(btn => btn.textContent.includes('Submit Alignment'));
            }

            if (targetBtn) {
                e.preventDefault();
                targetBtn.click();
            }
        });
        """

        demo.load(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            js=keyboard_js
        )
    
    return demo


if __name__ == "__main__":
    app = alignment_interface()
    app.launch()