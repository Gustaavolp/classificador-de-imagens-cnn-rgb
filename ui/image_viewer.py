from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize

class ImageViewer(QWidget):
    """Widget for viewing images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setText("No image")
        
        layout.addWidget(self.image_label)
    
    def load_image(self, image_path):
        """Load and display an image from the given path."""
        pixmap = QPixmap(image_path)
        
        if not pixmap.isNull():
            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Failed to load image")
    
    def clear(self):
        """Clear the current image."""
        self.image_label.clear()
        self.image_label.setText("No image")
    
    def resizeEvent(self, event):
        """Handle resize events to scale the image appropriately."""
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            pixmap = QPixmap(self.image_label.pixmap())
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        
        super().resizeEvent(event)