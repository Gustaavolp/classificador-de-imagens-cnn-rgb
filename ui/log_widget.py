import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import Qt, QDateTime

class LogWidget(QWidget):
    """Widget for displaying log messages."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_ui()
        self.setup_logger()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Log")
        layout.addWidget(title)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
    
    def setup_logger(self):
        """Set up the Python logger for the application."""
        self.logger = logging.getLogger('NeuralNetApp')
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        file_handler = logging.FileHandler('logs/application.log')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log(self, message, level="info"):
        """Add a message to the log."""
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        
        # Set color based on level
        color = "black"
        if level == "error":
            color = "red"
            self.logger.error(message)
        elif level == "warning":
            color = "orange"
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Add to QTextEdit
        self.log_text.append(f'<span style="color:{color}">{timestamp} {message}</span>')