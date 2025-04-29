from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QSlider, QLineEdit, QGroupBox, QSpinBox)
from PySide6.QtCore import Qt

class RGBSelector(QGroupBox):
    """Widget for selecting RGB color ranges."""
    
    def __init__(self, name="Attribute", parent=None):
        super().__init__(parent)
        self.setTitle(name)
        
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Name field
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nome:"))
        self.name_edit = QLineEdit(self.title())
        name_layout.addWidget(self.name_edit)
        main_layout.addLayout(name_layout)
        
        # Folga (margem)
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Margem (±):"))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(1, 50)
        self.margin_spin.setValue(10)
        self.margin_spin.valueChanged.connect(self.update_ranges)
        margin_layout.addWidget(self.margin_spin)
        main_layout.addLayout(margin_layout)
        
        # RGB sliders
        self.create_rgb_sliders(main_layout)
    
    def create_rgb_sliders(self, parent_layout):
        # Red
        red_group = QGroupBox("Vermelho")
        red_layout = QVBoxLayout()
        
        red_slider_layout = QHBoxLayout()
        red_slider_layout.addWidget(QLabel("Valor:"))
        
        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setRange(0, 255)
        self.r_slider.setValue(128)
        self.r_slider.valueChanged.connect(self.update_ranges)
        
        self.r_value_label = QLabel("128")
        
        red_slider_layout.addWidget(self.r_slider)
        red_slider_layout.addWidget(self.r_value_label)
        red_layout.addLayout(red_slider_layout)
        
        # Mostrar intervalo calculado
        self.r_range_label = QLabel("Intervalo: 118-138")
        red_layout.addWidget(self.r_range_label)
        
        red_group.setLayout(red_layout)
        parent_layout.addWidget(red_group)
        
        # Green
        green_group = QGroupBox("Verde")
        green_layout = QVBoxLayout()
        
        green_slider_layout = QHBoxLayout()
        green_slider_layout.addWidget(QLabel("Valor:"))
        
        self.g_slider = QSlider(Qt.Horizontal)
        self.g_slider.setRange(0, 255)
        self.g_slider.setValue(128)
        self.g_slider.valueChanged.connect(self.update_ranges)
        
        self.g_value_label = QLabel("128")
        
        green_slider_layout.addWidget(self.g_slider)
        green_slider_layout.addWidget(self.g_value_label)
        green_layout.addLayout(green_slider_layout)
        
        # Mostrar intervalo calculado
        self.g_range_label = QLabel("Intervalo: 118-138")
        green_layout.addWidget(self.g_range_label)
        
        green_group.setLayout(green_layout)
        parent_layout.addWidget(green_group)
        
        # Blue
        blue_group = QGroupBox("Azul")
        blue_layout = QVBoxLayout()
        
        blue_slider_layout = QHBoxLayout()
        blue_slider_layout.addWidget(QLabel("Valor:"))
        
        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setRange(0, 255)
        self.b_slider.setValue(128)
        self.b_slider.valueChanged.connect(self.update_ranges)
        
        self.b_value_label = QLabel("128")
        
        blue_slider_layout.addWidget(self.b_slider)
        blue_slider_layout.addWidget(self.b_value_label)
        blue_layout.addLayout(blue_slider_layout)
        
        # Mostrar intervalo calculado
        self.b_range_label = QLabel("Intervalo: 118-138")
        blue_layout.addWidget(self.b_range_label)
        
        blue_group.setLayout(blue_layout)
        parent_layout.addWidget(blue_group)
        
        # Connect signals
        self.r_slider.valueChanged.connect(lambda v: self.r_value_label.setText(str(v)))
        self.g_slider.valueChanged.connect(lambda v: self.g_value_label.setText(str(v)))
        self.b_slider.valueChanged.connect(lambda v: self.b_value_label.setText(str(v)))
        
        # Inicializar os intervalos
        self.update_ranges()
    
    def update_ranges(self):
        """Atualiza os intervalos baseado nos valores dos sliders e na folga."""
        margin = self.margin_spin.value()
        
        # Calcular intervalos para R
        r_value = self.r_slider.value()
        r_min = max(0, r_value - margin)
        r_max = min(255, r_value + margin)
        self.r_range_label.setText(f"Intervalo: {r_min}-{r_max}")
        
        # Calcular intervalos para G
        g_value = self.g_slider.value()
        g_min = max(0, g_value - margin)
        g_max = min(255, g_value + margin)
        self.g_range_label.setText(f"Intervalo: {g_min}-{g_max}")
        
        # Calcular intervalos para B
        b_value = self.b_slider.value()
        b_min = max(0, b_value - margin)
        b_max = min(255, b_value + margin)
        self.b_range_label.setText(f"Intervalo: {b_min}-{b_max}")
    
    # Métodos para compatibilidade com o código existente
    def r_slider_min(self):
        margin = self.margin_spin.value()
        return max(0, self.r_slider.value() - margin)
        
    def r_slider_max(self):
        margin = self.margin_spin.value()
        return min(255, self.r_slider.value() + margin)
        
    def g_slider_min(self):
        margin = self.margin_spin.value()
        return max(0, self.g_slider.value() - margin)
        
    def g_slider_max(self):
        margin = self.margin_spin.value()
        return min(255, self.g_slider.value() + margin)
        
    def b_slider_min(self):
        margin = self.margin_spin.value()
        return max(0, self.b_slider.value() - margin)
        
    def b_slider_max(self):
        margin = self.margin_spin.value()
        return min(255, self.b_slider.value() + margin)