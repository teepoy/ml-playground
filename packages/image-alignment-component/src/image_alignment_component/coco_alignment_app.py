"""
Image Alignment Tool for COCO Dataset
A PyQt6 application that allows users to align two images from a COCO dataset
using keyboard controls to measure alignment accuracy.
"""
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QSlider, QGroupBox, QFrame)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QKeyEvent
from PyQt6.QtCore import Qt, QRectF
from typing import Optional
import json
import os


class COCOImageAlignmentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COCO Image Alignment Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # State variables
        self.fixed_image_path = None
        self.movable_image_path = None
        self.dx = 0  # x shift
        self.dy = 0  # y shift
        self.alpha = 0.3  # transparency for movable image
        
        # COCO dataset images (will be populated later)
        self.coco_images = []
        self.current_image_index = 0
        
        self.init_ui()
        self.load_coco_dataset()
    
    def init_ui(self):
        """Initialize the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for image display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 3)
    
    def create_control_panel(self):
        """Create the control panel with image selection and controls."""
        panel = QGroupBox("Controls")
        layout = QVBoxLayout(panel)
        
        # Image selection comboboxes
        fixed_label = QLabel("Fixed Image:")
        self.fixed_combo = QComboBox()
        self.fixed_combo.currentTextChanged.connect(self.on_fixed_image_changed)
        
        movable_label = QLabel("Movable Image:")
        self.movable_combo = QComboBox()
        self.movable_combo.currentTextChanged.connect(self.on_movable_image_changed)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("Previous Pair")
        next_btn = QPushButton("Next Pair")
        prev_btn.clicked.connect(self.on_prev_pair)
        next_btn.clicked.connect(self.on_next_pair)
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        
        # Movement controls
        movement_group = QGroupBox("Movement")
        movement_layout = QVBoxLayout(movement_group)
        
        # Directional buttons
        btn_layout = QHBoxLayout()
        up_btn = QPushButton("↑ Up")
        down_btn = QPushButton("↓ Down")
        left_btn = QPushButton("← Left")
        right_btn = QPushButton("→ Right")
        
        up_btn.clicked.connect(self.move_up)
        down_btn.clicked.connect(self.move_down)
        left_btn.clicked.connect(self.move_left)
        right_btn.clicked.connect(self.move_right)
        
        btn_layout.addWidget(left_btn)
        btn_layout.addWidget(up_btn)
        btn_layout.addWidget(down_btn)
        btn_layout.addWidget(right_btn)
        
        # Slider controls for fine adjustments
        x_slider_group = QGroupBox("X Adjustment")
        x_slider_layout = QVBoxLayout(x_slider_group)
        self.x_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_slider.setRange(-200, 200)
        self.x_slider.setValue(0)
        self.x_slider.valueChanged.connect(self.on_x_slider_changed)
        x_slider_layout.addWidget(self.x_slider)
        
        y_slider_group = QGroupBox("Y Adjustment")
        y_slider_layout = QVBoxLayout(y_slider_group)
        self.y_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_slider.setRange(-200, 200)
        self.y_slider.setValue(0)
        self.y_slider.valueChanged.connect(self.on_y_slider_changed)
        y_slider_layout.addWidget(self.y_slider)
        
        # Submit button
        submit_btn = QPushButton("Submit Alignment (Enter)")
        submit_btn.clicked.connect(self.submit_alignment)
        
        # Result display
        self.result_label = QLabel("Current Shift: (0, 0)")
        
        # Add all widgets to layout
        layout.addWidget(fixed_label)
        layout.addWidget(self.fixed_combo)
        layout.addWidget(movable_label)
        layout.addWidget(self.movable_combo)
        layout.addLayout(nav_layout)
        layout.addWidget(movement_group)
        movement_layout.addLayout(btn_layout)
        movement_layout.addWidget(x_slider_group)
        movement_layout.addWidget(y_slider_group)
        layout.addWidget(submit_btn)
        layout.addWidget(self.result_label)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """Create the display panel for image overlay."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        layout = QVBoxLayout(panel)
        
        # Graphics view for image display
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # Add view to layout
        layout.addWidget(self.view)
        
        # Info label
        self.info_label = QLabel("Select images and use arrow keys to align. Press Enter to submit.")
        layout.addWidget(self.info_label)
        
        return panel
    
    def load_coco_dataset(self):
        """Load COCO dataset images from a specified directory."""
        # For demonstration, we'll create some mock COCO image paths
        # In a real scenario, you would load from an actual COCO dataset
        # For now, I'll create mock image paths - in a real app you'd load from a real COCO dataset
        example_images = [
            "image_001.jpg", "image_002.jpg", "image_003.jpg",
            "image_004.jpg", "image_005.jpg", "image_006.jpg"
        ]

        # Add them to the comboboxes
        for img in example_images:
            self.fixed_combo.addItem(img)
            self.movable_combo.addItem(img)

        if example_images:
            self.fixed_combo.setCurrentIndex(0)
            self.movable_combo.setCurrentIndex(1)
            # Set initial image paths
            self.fixed_image_path = f"/path/to/coco/images/{example_images[0]}"
            self.movable_image_path = f"/path/to/coco/images/{example_images[1]}"
    
    def on_fixed_image_changed(self, text):
        """Handle fixed image selection change."""
        # In a real implementation, this would load the selected image
        # For now, we'll just update the path with a placeholder
        self.fixed_image_path = f"/path/to/coco/images/{text}"
        self.update_display()
    
    def on_movable_image_changed(self, text):
        """Handle movable image selection change."""
        # In a real implementation, this would load the selected image
        # For now, we'll just update the path with a placeholder
        self.movable_image_path = f"/path/to/coco/images/{text}"
        self.update_display()
    
    def update_display(self):
        """Update the image display with the current overlay."""
        if not self.fixed_image_path or not self.movable_image_path:
            return
        
        # Clear the scene
        self.scene.clear()
        
        # Load fixed image (background)
        fixed_pixmap = QPixmap(self.fixed_image_path)
        if fixed_pixmap.isNull():
            # Create a placeholder pixmap if image doesn't exist
            fixed_pixmap = QPixmap(400, 400)
            fixed_pixmap.fill(Qt.GlobalColor.lightGray)
        
        # Add fixed image to scene
        fixed_item = self.scene.addPixmap(fixed_pixmap)
        fixed_item.setZValue(0)  # Background
        
        # Load movable image (overlay)
        movable_pixmap = QPixmap(self.movable_image_path)
        if movable_pixmap.isNull():
            # Create a placeholder pixmap if image doesn't exist
            movable_pixmap = QPixmap(300, 300)
            movable_pixmap.fill(Qt.GlobalColor.red)
        
        # Create a transparent version of the movable image
        transparent_pixmap = self.make_transparent(movable_pixmap, self.alpha)
        
        # Add movable image to scene
        movable_item = self.scene.addPixmap(transparent_pixmap)
        movable_item.setZValue(1)  # Foreground
        movable_item.setPos(self.dx, self.dy)  # Apply current offset
        
        # Set the scene size to fit the content
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        
        # Update the slider values to match current position
        self.x_slider.setValue(self.dx)
        self.y_slider.setValue(self.dy)
        
        # Update result label
        self.result_label.setText(f"Current Shift: ({self.dx}, {self.dy})")
    
    def make_transparent(self, pixmap: QPixmap, alpha: float) -> QPixmap:
        """Create a transparent version of the pixmap."""
        result = QPixmap(pixmap.size())
        result.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(result)
        painter.setOpacity(alpha)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        
        return result
    
    def move_up(self):
        """Move the movable image up."""
        self.dy -= 1
        # Update the slider as well to maintain synchronization
        self.y_slider.setValue(self.dy)
        self.update_display()

    def move_down(self):
        """Move the movable image down."""
        self.dy += 1
        # Update the slider as well to maintain synchronization
        self.y_slider.setValue(self.dy)
        self.update_display()

    def move_left(self):
        """Move the movable image left."""
        self.dx -= 1
        # Update the slider as well to maintain synchronization
        self.x_slider.setValue(self.dx)
        self.update_display()

    def move_right(self):
        """Move the movable image right."""
        self.dx += 1
        # Update the slider as well to maintain synchronization
        self.x_slider.setValue(self.dx)
        self.update_display()
    
    def on_x_slider_changed(self, value):
        """Handle X slider change."""
        self.dx = value
        self.update_display()
    
    def on_y_slider_changed(self, value):
        """Handle Y slider change."""
        self.dy = value
        self.update_display()
    
    def submit_alignment(self):
        """Submit the current alignment and print results."""
        result = {"x": self.dx, "y": self.dy}
        print(f"Alignment submitted: {result}")
        self.result_label.setText(f"Submitted: {result}")
    
    def on_prev_pair(self):
        """Go to the previous image pair."""
        current_fixed_idx = self.fixed_combo.currentIndex()
        current_movable_idx = self.movable_combo.currentIndex()
        
        if current_fixed_idx > 0:
            self.fixed_combo.setCurrentIndex(current_fixed_idx - 1)
        elif current_movable_idx > 0:
            self.movable_combo.setCurrentIndex(current_movable_idx - 1)
    
    def on_next_pair(self):
        """Go to the next image pair."""
        current_fixed_idx = self.fixed_combo.currentIndex()
        current_movable_idx = self.movable_combo.currentIndex()
        
        if current_fixed_idx < self.fixed_combo.count() - 1:
            self.fixed_combo.setCurrentIndex(current_fixed_idx + 1)
        elif current_movable_idx < self.movable_combo.count() - 1:
            self.movable_combo.setCurrentIndex(current_movable_idx + 1)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for image alignment."""
        key = event.key()

        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Submit on Enter key
            self.submit_alignment()
            event.accept()
        elif event.key() == Qt.Key.Key_Up:
            # Move up
            self.move_up()
            event.accept()
        elif event.key() == Qt.Key.Key_Down:
            # Move down
            self.move_down()
            event.accept()
        elif event.key() == Qt.Key.Key_Left:
            # Move left
            self.move_left()
            event.accept()
        elif event.key() == Qt.Key.Key_Right:
            # Move right
            self.move_right()
            event.accept()
        else:
            # For any other keys, call parent implementation
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    window = COCOImageAlignmentApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()