import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QTabWidget, QPushButton, QLabel, QFileDialog,
                              QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
                              QGroupBox, QFormLayout, QProgressBar, QMessageBox,
                              QListWidget, QSlider, QScrollArea, QApplication,
                              QButtonGroup, QStackedWidget, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.rgb_selector import RGBSelector
from ui.log_widget import LogWidget
from ui.image_viewer import ImageViewer

from utils.data_processing import DataProcessor
from utils.model_utils import ModelUtils
from models.rgb_net import RGBFeatureNet
from models.cnn import ConvolutionalNetwork

class WorkerThread(QThread):
    """Worker thread to handle long-running tasks."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    task_finished = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.running = True
        # Definir como daemon thread (encerra com o aplicativo) - equivalente Python
        self.setObjectName(f"WorkerThread-{id(self)}")
    
    def run(self):
        try:
            # Check if interrupted before starting work
            if self.isInterruptionRequested():
                print(f"Thread {self.objectName()} interrompida antes de iniciar")
                return
                
            # Executa a tarefa com proteção contra interrupção
            print(f"Thread {self.objectName()} iniciando tarefa")
            result = self.task(*self.args, **self.kwargs)
            print(f"Thread {self.objectName()} concluiu tarefa")
            
            # Check again after work is done
            if self.isInterruptionRequested():
                print(f"Thread {self.objectName()} interrompida após concluir tarefa")
                return
                
            if self.running:  # Only emit signal if still running
                print(f"Thread {self.objectName()} emitindo sinal de conclusão")
                self.task_finished.emit(result)
        except Exception as e:
            print(f"Thread {self.objectName()} erro: {str(e)}")
            if self.running and not self.isInterruptionRequested():  # Only emit signal if still running
                self.error_occurred.emit(str(e))
    
    def safe_stop(self):
        """Safely stop the thread by requesting interruption."""
        print(f"Solicitando parada segura para thread {self.objectName()}")
        self.running = False
        self.requestInterruption()
        # Wait briefly for thread to respond to interruption
        success = self.wait(500)  # Wait up to 500ms
        print(f"Thread {self.objectName()} parou? {'Sim' if success else 'Não'}")
        return success

# Store active threads in a global list to prevent premature garbage collection
active_worker_threads = []

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificador de Imagens CNN/RGB")
        self.resize(1300, 900)  # Tamanho inicial maior
        self.setMinimumSize(1200, 800)  # Tamanho mínimo maior
        
        # Para rastrear threads ativas
        self.worker_threads = []
        
        # Initialize components
        self.data_processor = DataProcessor(self)
        self.model_utils = ModelUtils()
        self.rgb_feature_net = RGBFeatureNet()
        self.cnn = ConvolutionalNetwork()
        
        # Inicializar dados internos
        self.training_data_directory = None
        self.rgb_attributes = []
        self.current_model = None
        self.processed_data = None
        self.loaded_model_type = None
        self.class_data = {}  # Inicializar class_data como dicionário vazio
        
        # Set up main UI
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI based on default model type
        self.update_model_ui()
        
        # Log startup message
        self.statusBar().showMessage("Aplicação pronta - Selecione um tipo de modelo e carregue dados para começar")
    
    def setup_ui(self):
        """Create and arrange the UI elements."""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Título e descrição
        title_label = QLabel("Classificador de Imagens CNN/RGB")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0078D7;")
        title_label.setAlignment(Qt.AlignCenter)
        
        desc_label = QLabel("Treine seus próprios modelos de classificação de imagem usando CNN ou Análise de Pixel RGB.")
        desc_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(desc_label)
        
        # Guia de fluxo de trabalho
        workflow_frame = QGroupBox("Guia de Fluxo de Trabalho")
        workflow_layout = QVBoxLayout(workflow_frame)
        
        workflow_text = QLabel("Selecione o Fluxo (CNN ou RGB) → 1. Upload das Pastas de Classe (uma por vez) → 2. Configurar e Treinar → 3. Resultados e Salvar → 4. Classificar")
        workflow_text.setWordWrap(True)
        workflow_layout.addWidget(workflow_text)
        
        workflow_frame.setLayout(workflow_layout)
        main_layout.addWidget(workflow_frame)
        
        # Seleção de modelo (fluxo) - usando botões com estilo para parecerem abas
        model_selection_frame = QWidget()
        model_layout = QHBoxLayout(model_selection_frame)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(0)
        
        # Criar um grupo de botões para garantir exclusividade
        self.model_button_group = QButtonGroup(self)
        
        # Botão CNN
        self.cnn_btn = QPushButton("Rede Neural Convolucional (CNN)")
        self.cnn_btn.setCheckable(True)
        self.cnn_btn.setChecked(True)  # CNN é o padrão
        self.cnn_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 10px;
                min-width: 200px;
                text-align: center;
            }
            QPushButton:checked {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Botão RGB
        self.rgb_btn = QPushButton("Análise de Pixel RGB")
        self.rgb_btn.setCheckable(True)
        self.rgb_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 10px;
                min-width: 200px;
                text-align: center;
            }
            QPushButton:checked {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Adicionar os botões ao grupo
        self.model_button_group.addButton(self.cnn_btn)
        self.model_button_group.addButton(self.rgb_btn)
        # A conexão do sinal será feita no connect_signals
        
        model_layout.addWidget(self.cnn_btn)
        model_layout.addWidget(self.rgb_btn)
        model_layout.addStretch()
        
        main_layout.addWidget(model_selection_frame)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs na ordem correta do fluxo
        self.create_upload_tab()          # 1. Upload das Pastas de Classe
        self.create_training_tab()        # 2. Configurar e Treinar (combina configurar e treinar)
        self.create_results_tab()         # 3. Resultados e Salvar
        self.create_classification_tab()  # 4. Classificar
        
        # Create log area (shared across tabs)
        self.log_widget = LogWidget()
        main_layout.addWidget(self.log_widget)
        
        # Status bar
        self.statusBar().showMessage("Pronto - Selecione o tipo de modelo e carregue seus dados para começar")
        
        # Criar a barra de progresso mas deixá-la oculta
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
    
    def set_model_type(self, model_type):
        """Set model type based on button selection and update UI accordingly"""
        print(f"set_model_type chamado com: {model_type}")
        
        # Atualizar os botões de seleção de modelo sem disparar sinais
        self.model_button_group.blockSignals(True)
        if model_type == "CNN":
            self.cnn_btn.setChecked(True)
            self.rgb_btn.setChecked(False)
            # Mostrar configuração CNN
            if hasattr(self, 'model_config_stack'):
                print("Mudando para configuração CNN (índice 0)")
                self.model_config_stack.setCurrentIndex(0)
        else:
            self.cnn_btn.setChecked(False)
            self.rgb_btn.setChecked(True)
            # Mostrar configuração RGB
            if hasattr(self, 'model_config_stack'):
                print("Mudando para configuração RGB (índice 1)")
                self.model_config_stack.setCurrentIndex(1)
        self.model_button_group.blockSignals(False)
            
        # Atualizar a visibilidade das opções específicas
        self.update_configuration_tabs()
        
        # Log da alteração
        if hasattr(self, 'log_widget'):
            self.log_widget.log(f"Fluxo alterado para: {model_type}")
    
    def update_configuration_tabs(self):
        """Atualiza as abas de configuração com base no tipo de modelo selecionado"""
        # Verificar se os botões existem
        if not hasattr(self, 'cnn_btn') or not hasattr(self, 'rgb_btn'):
            return
            
        is_cnn = self.cnn_btn.isChecked()
            
        # Atualizar o título da aba de configuração
        if hasattr(self, 'config_title'):
            if is_cnn:
                self.config_title.setText("2. Configuração de Treinamento CNN")
            else:
                self.config_title.setText("2. Configuração da Análise de Pixel RGB")
                
        # Atualizar o título do botão de treinamento
        if hasattr(self, 'train_btn'):
            if is_cnn:
                self.train_btn.setText("Treinar Modelo CNN")
            else:
                self.train_btn.setText("Treinar Modelo RGB")
                
        # Atualizar sumário de configuração
        if hasattr(self, 'config_summary'):
            try:
                self.update_config_summary()
            except Exception as e:
                print(f"Erro ao atualizar sumário: {str(e)}")
        else:
            print("Aviso: config_summary não encontrado")
    
    def create_upload_tab(self):
        """Create the data upload tab."""
        upload_tab = QWidget()
        layout = QVBoxLayout(upload_tab)
        
        # Título da etapa
        step_title = QLabel("1. Upload das Pastas de Dados de Classe")
        step_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(step_title)
        
        desc_label = QLabel("Clique no botão abaixo para selecionar uma pasta contendo imagens para cada classe. Repita isso para cada classe que deseja treinar (mínimo 2). O nome da pasta será usado como título da classe.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Info box
        info_box = QGroupBox("Importante")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel("Use o botão para abrir o seletor de diretório. Selecione a pasta principal para uma única classe (ex: selecione a pasta 'cachorros'). O sistema contará as imagens dentro dela.")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Botão de seleção
        select_btn = QPushButton("Selecionar Pasta para uma Classe...")
        select_btn.setIcon(QIcon.fromTheme("folder-open"))
        select_btn.clicked.connect(self.load_data)
        layout.addWidget(select_btn)
        
        # Lista de classes selecionadas
        classes_group = QGroupBox("Pastas de Classe Selecionadas:")
        classes_layout = QVBoxLayout()
        
        # Criar um widget para essa lista
        self.selected_classes_layout = QVBoxLayout()
        
        # Status inicial se não tivermos dados
        if not hasattr(self, 'class_data') or not self.class_data:
            self.selected_classes_layout.addWidget(QLabel("Nenhuma classe selecionada ainda."))
        else:
            # Adicionar as classes que já temos
            for class_name, image_count in self.class_data.items():
                class_row = QWidget()
                row_layout = QHBoxLayout(class_row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(f"Classe: {class_name} ({image_count} imagens detectadas)")
                
                remove_btn = QPushButton()
                remove_btn.setIcon(QIcon.fromTheme("edit-delete"))
                remove_btn.setMaximumWidth(30)
                remove_btn.clicked.connect(lambda checked, cn=class_name: self.remove_class(cn))
                
                row_layout.addWidget(label)
                row_layout.addStretch()
                row_layout.addWidget(remove_btn)
                
                self.selected_classes_layout.addWidget(class_row)
        
        classes_layout.addLayout(self.selected_classes_layout)
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)
        
        # Status da seleção
        self.selection_status = QLabel()
        # Definir o status com base no número de classes
        if hasattr(self, 'class_data') and len(self.class_data) >= 2:
            self.selection_status.setText(f"A seleção atende aos requisitos ({len(self.class_data)} classes).")
            self.selection_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.selection_status.setText("Selecione pelo menos 2 classes.")
            self.selection_status.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(self.selection_status)
        
        # Data loading section (hidden for new design)
        data_group = QGroupBox("Carregamento de Dados")
        data_layout = QFormLayout()
        
        self.load_data_btn = QPushButton("Selecionar Pastas...")
        self.data_path_label = QLabel("Nenhum dado carregado")
        data_layout.addRow("Pastas de Dados:", self.load_data_btn)
        data_layout.addRow("", self.data_path_label)
        
        # Train/Test split
        split_layout = QHBoxLayout()
        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(50, 95)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setSuffix("%")
        split_layout.addWidget(QLabel("Treinamento:"))
        split_layout.addWidget(self.train_split_spin)
        split_layout.addWidget(QLabel("Teste:"))
        
        self.test_split_label = QLabel("20%")
        split_layout.addWidget(self.test_split_label)
        split_layout.addStretch()
        
        data_layout.addRow("Divisão Treino/Teste:", split_layout)
        data_group.setLayout(data_layout)
        data_group.setVisible(False)  # Esconder para manter compatibilidade
        layout.addWidget(data_group)
        
        # Class preview - hidden for new design
        class_group = QGroupBox("Prévia das Classes")
        class_layout = QVBoxLayout()
        self.class_list = QListWidget()
        self.class_preview = ImageViewer()
        
        class_h_layout = QHBoxLayout()
        class_h_layout.addWidget(self.class_list, 1)
        class_h_layout.addWidget(self.class_preview, 3)
        
        class_layout.addLayout(class_h_layout)
        class_group.setLayout(class_layout)
        class_group.setVisible(False)  # Esconder para manter compatibilidade
        layout.addWidget(class_group)
        
        # Process Data button
        self.process_data_btn = QPushButton("Processar Dados")
        self.process_data_btn.setEnabled(False)
        self.process_data_btn.setVisible(False)  # Esconder para manter compatibilidade
        layout.addWidget(self.process_data_btn)
        
        # Botão próxima etapa
        next_btn = QPushButton("Próxima Etapa: Configurar e Treinar")
        next_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        layout.addWidget(next_btn)
        
        layout.addStretch()
        self.tab_widget.addTab(upload_tab, "1. Upload")
    
    def update_selected_classes_list(self):
        """Update the list of selected classes in the UI"""
        # Verificar se temos o layout de classes
        if not hasattr(self, 'selected_classes_layout'):
            return
            
        # Limpar layout existente
        for i in reversed(range(self.selected_classes_layout.count())):
            item = self.selected_classes_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                
        # Se não temos dados, adicionar mensagem
        if not hasattr(self, 'class_data') or not self.class_data:
            self.selected_classes_layout.addWidget(QLabel("Nenhuma classe selecionada ainda."))
            if hasattr(self, 'selection_status'):
                self.selection_status.setText("Selecione pelo menos 2 classes.")
                self.selection_status.setStyleSheet("color: orange;")
            return
            
        # Adicionar as classes que temos
        for class_name, image_count in self.class_data.items():
            class_row = QWidget()
            row_layout = QHBoxLayout(class_row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(f"Classe: {class_name} ({image_count} imagens detectadas)")
            
            remove_btn = QPushButton()
            remove_btn.setIcon(QIcon.fromTheme("edit-delete"))
            remove_btn.setMaximumWidth(30)
            remove_btn.clicked.connect(lambda checked, cn=class_name: self.remove_class(cn))
            
            row_layout.addWidget(label)
            row_layout.addStretch()
            row_layout.addWidget(remove_btn)
            
            self.selected_classes_layout.addWidget(class_row)
            
        # Atualizar status
        if hasattr(self, 'selection_status'):
            if len(self.class_data) >= 2:
                self.selection_status.setText(f"A seleção atende aos requisitos ({len(self.class_data)} classes).")
                self.selection_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.selection_status.setText(f"Selecione mais classes (mínimo 2, atual: {len(self.class_data)}).")
                self.selection_status.setStyleSheet("color: orange; font-weight: bold;")
    
    def remove_class(self, class_name):
        """Remove a class from the selected classes"""
        if hasattr(self, 'class_data') and class_name in self.class_data:
            del self.class_data[class_name]
            self.update_selected_classes_list()
            self.log_widget.log(f"Classe removida: {class_name}")
            
            # Atualizar o diretório de dados e notificar o modelo
            if self.class_data:
                # Sempre manter o diretório pai das classes
                parent_dir = os.path.dirname(next(iter(self.class_data.keys())))
                self.training_data_directory = parent_dir
            else:
                self.training_data_directory = None
                
            # Criar dados básicos para permitir treinamento
            if len(self.class_data) >= 2:
                self.processed_data = {
                    'data_dir': self.training_data_directory,
                    'train_split': self.train_split_spin.value() / 100.0,
                    'class_names': list(self.class_data.keys())
                }
                if hasattr(self, 'train_btn'):
                    self.train_btn.setEnabled(True)
            else:
                self.processed_data = None
                if hasattr(self, 'train_btn'):
                    self.train_btn.setEnabled(False)
    
    def load_data(self):
        """Open dialog to select data folders."""
        directory = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Classe")
        if directory:
            try:
                # Verificar se a pasta contém imagens
                image_files = [f for f in os.listdir(directory) 
                              if os.path.isfile(os.path.join(directory, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                if not image_files:
                    QMessageBox.warning(self, "Pasta Inválida", 
                                      "A pasta selecionada não contém imagens. Selecione uma pasta com imagens.")
                    return
                
                # Verificar se mudamos de diretório pai
                parent_dir = os.path.dirname(directory)
                
                # Se já temos um diretório de dados e ele é diferente do novo, perguntar ao usuário
                if (hasattr(self, 'training_data_directory') and 
                    self.training_data_directory and 
                    self.training_data_directory != parent_dir):
                    
                    resp = QMessageBox.question(self, "Mudança de Conjunto de Dados",
                                              "Você está selecionando uma pasta de um diretório diferente. "
                                              "Deseja limpar as classes anteriores e começar com um novo conjunto de dados?",
                                              QMessageBox.Yes | QMessageBox.No)
                    
                    if resp == QMessageBox.Yes:
                        # Limpar dados anteriores
                        self.reset_app_progress()
                
                # Inicializar class_data se necessário
                if not hasattr(self, 'class_data'):
                    self.class_data = {}
                    
                # Extrair nome da classe da pasta
                class_name = os.path.basename(directory)
                
                # Adicionar à lista de classes
                self.class_data[class_name] = len(image_files)
                
                # Atualizar a interface
                self.update_selected_classes_list()
                
                # Se estiver selecionado o modelo RGB, atualizar os seletores RGB
                if hasattr(self, 'rgb_btn') and self.rgb_btn.isChecked():
                    self.update_rgb_selectors_for_classes()
                
                # Manter o diretório de dados (o pai comum de todas as classes)
                self.training_data_directory = parent_dir
                
                self.log_widget.log(f"Adicionada classe: {class_name} com {len(image_files)} imagens")
                
                # Criar dados básicos para habilitar o botão Train mesmo sem processamento completo
                # apenas se tivermos pelo menos 2 classes
                if len(self.class_data) >= 2:
                    self.processed_data = {
                        'data_dir': parent_dir,
                        'train_split': self.train_split_spin.value() / 100.0,
                        'class_names': list(self.class_data.keys())
                    }
                    
                    # Habilitar o botão Train
                    self.train_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao carregar dados: {str(e)}")
    
    def create_training_tab(self):
        """Create the model training tab."""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # Título da etapa
        step_title = QLabel("2. Configurar e Treinar")
        step_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(step_title)
        
        desc_label = QLabel("Configure os parâmetros do modelo e inicie o treinamento.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Create a scroll area to contain all configuration options
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        # Create a widget to hold all the scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Adicione os dois widgets de configuração em um QStackedWidget
        self.model_config_stack = QStackedWidget()
        
        # ===== CNN Configuration Widget =====
        self.cnn_config_widget = QWidget()
        cnn_config_layout = QVBoxLayout(self.cnn_config_widget)
        
        # Grupo de configuração CNN
        cnn_config_group = QGroupBox("Configuração da Rede Neural Convolucional")
        cnn_form_layout = QFormLayout()
        
        # Parâmetros CNN (conforme especificação)
        self.cnn_layers_spin = QSpinBox()
        self.cnn_layers_spin.setRange(1, 10)
        self.cnn_layers_spin.setValue(3)
        cnn_form_layout.addRow("Número de Camadas:", self.cnn_layers_spin)
        
        self.cnn_neurons_spin = QSpinBox()
        self.cnn_neurons_spin.setRange(8, 512)
        self.cnn_neurons_spin.setValue(64)
        cnn_form_layout.addRow("Neurônios por Camada:", self.cnn_neurons_spin)
        
        self.cnn_epochs_spin = QSpinBox()
        self.cnn_epochs_spin.setRange(1, 1000)
        self.cnn_epochs_spin.setValue(50)
        cnn_form_layout.addRow("Épocas de Treinamento:", self.cnn_epochs_spin)
        
        # Divisão treino/teste para CNN
        split_layout_cnn = QVBoxLayout()
        self.split_slider_cnn = QSlider(Qt.Horizontal)
        self.split_slider_cnn.setMinimum(50)
        self.split_slider_cnn.setMaximum(90)
        self.split_slider_cnn.setValue(80)
        self.split_slider_cnn.valueChanged.connect(self.update_cnn_train_test_split_label)
        
        self.cnn_train_test_split_label = QLabel("Treino: 80% / Teste: 20%")
        
        split_layout_cnn.addWidget(self.cnn_train_test_split_label)
        split_layout_cnn.addWidget(self.split_slider_cnn)
        cnn_form_layout.addRow("Divisão Treino/Teste:", split_layout_cnn)
        
        cnn_config_group.setLayout(cnn_form_layout)
        cnn_config_layout.addWidget(cnn_config_group)
        
        # ===== RGB Configuration Widget =====
        self.rgb_config_widget = QWidget()
        rgb_config_layout = QVBoxLayout(self.rgb_config_widget)
        
        # Grupo de atributos RGB
        rgb_attr_group = QGroupBox("Definição de Atributos RGB")
        rgb_attr_layout = QVBoxLayout()
        
        # Container para seletores RGB
        self.rgb_selectors_container = QWidget()
        self.rgb_selectors_layout = QVBoxLayout(self.rgb_selectors_container)
        
        # Não adicionar seletores RGB por padrão - serão adicionados quando as classes forem carregadas
        # Nota: os seletores serão criados pelo método update_rgb_selectors_for_classes
        
        # Botões para gerenciar atributos RGB
        rgb_buttons_layout = QHBoxLayout()
        self.save_attr_btn = QPushButton("Salvar Atributos")
        
        rgb_buttons_layout.addWidget(self.save_attr_btn)
        
        rgb_attr_layout.addWidget(self.rgb_selectors_container)
        rgb_attr_layout.addLayout(rgb_buttons_layout)
        rgb_attr_group.setLayout(rgb_attr_layout)
        rgb_config_layout.addWidget(rgb_attr_group)
        
        # Grupo de configuração da rede RGB
        rgb_net_group = QGroupBox("Configuração da Rede Neural")
        rgb_form_layout = QFormLayout()
        
        # Parâmetros da rede RGB (conforme especificação)
        self.rgb_layers_spin = QSpinBox()
        self.rgb_layers_spin.setRange(1, 10)
        self.rgb_layers_spin.setValue(3)
        rgb_form_layout.addRow("Número de Camadas:", self.rgb_layers_spin)
        
        self.rgb_neurons_spin = QSpinBox()
        self.rgb_neurons_spin.setRange(8, 512)
        self.rgb_neurons_spin.setValue(64)
        rgb_form_layout.addRow("Neurônios por Camada:", self.rgb_neurons_spin)
        
        self.rgb_epochs_spin = QSpinBox()
        self.rgb_epochs_spin.setRange(1, 1000)
        self.rgb_epochs_spin.setValue(50)
        rgb_form_layout.addRow("Épocas de Treinamento:", self.rgb_epochs_spin)
        
        # Divisão treino/teste para RGB
        split_layout_rgb = QVBoxLayout()
        self.split_slider_rgb = QSlider(Qt.Horizontal)
        self.split_slider_rgb.setMinimum(50)
        self.split_slider_rgb.setMaximum(90)
        self.split_slider_rgb.setValue(80)
        self.split_slider_rgb.valueChanged.connect(self.update_rgb_train_test_split_label)
        
        self.rgb_train_test_split_label = QLabel("Treino: 80% / Teste: 20%")
        
        split_layout_rgb.addWidget(self.rgb_train_test_split_label)
        split_layout_rgb.addWidget(self.split_slider_rgb)
        rgb_form_layout.addRow("Divisão Treino/Teste:", split_layout_rgb)
        
        rgb_net_group.setLayout(rgb_form_layout)
        rgb_config_layout.addWidget(rgb_net_group)
        
        # Adicionar os widgets ao stack
        print("Adicionando widgets ao stack...")
        print("Widget CNN:", self.cnn_config_widget)
        print("Widget RGB:", self.rgb_config_widget)
        self.model_config_stack.addWidget(self.cnn_config_widget)
        self.model_config_stack.addWidget(self.rgb_config_widget)
        print(f"Stack tem {self.model_config_stack.count()} widgets")
        
        # Adicionar o stack ao layout
        scroll_layout.addWidget(self.model_config_stack)
        
        # Configuração resumo
        self.config_summary = QLabel("Selecione um tipo de modelo para ver a configuração")
        self.config_summary.setTextFormat(Qt.RichText)
        scroll_layout.addWidget(self.config_summary)
        
        # Status de treinamento e barra de progresso
        self.train_status_label = QLabel("Status do treinamento")
        self.train_status_label.setWordWrap(True)
        self.train_status_label.setVisible(False)
        scroll_layout.addWidget(self.train_status_label)
        
        # Criar mas ocultar a barra de progresso
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setVisible(False)
        scroll_layout.addWidget(self.train_progress_bar)
        
        # Adicionar botão de treinamento
        self.train_btn = QPushButton("Treinar Modelo")
        self.train_btn.setEnabled(False)  # Inicialmente desabilitado
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                min-height: 40px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        self.train_btn.clicked.connect(lambda: self.train_model())
        scroll_layout.addWidget(self.train_btn)
        
        # Set the scroll content
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Botões de navegação
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("Etapa Anterior: Upload")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        
        self.next_to_results_btn = QPushButton("Próxima Etapa: Resultados")
        self.next_to_results_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(self.next_to_results_btn)
        
        layout.addLayout(nav_layout)
        
        # Add the tab
        self.tab_widget.addTab(training_tab, "2. Configurar e Treinar")
    
    def create_results_tab(self):
        """Create the results and save tab."""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        # Título da etapa
        step_title = QLabel("3. Resultados e Salvar")
        step_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(step_title)
        
        desc_label = QLabel("Visualize os resultados do treinamento e salve seu modelo.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Criar um QScrollArea para todo o conteúdo
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QScrollArea.NoFrame)
        
        # Widget para conter todo o conteúdo scrollável
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Results display
        results_group = QGroupBox("Resultados do Treinamento")
        results_layout = QVBoxLayout()
        
        # Matplotlib para plotagem - aumentar tamanho mínimo
        self.figure = Figure(figsize=(12, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)  # Aumentar altura mínima
        self.canvas.setMinimumWidth(900)   # Definir largura mínima
        results_layout.addWidget(self.canvas)
        
        # Results summary
        self.results_label = QLabel("Nenhum resultado disponível")
        self.results_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.results_label.setTextFormat(Qt.RichText)
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        
        results_group.setLayout(results_layout)
        scroll_layout.addWidget(results_group)
        
        # Save model section
        save_group = QGroupBox("Salvar Modelo")
        save_layout = QVBoxLayout()
        
        self.save_model_btn = QPushButton("Salvar Modelo")
        self.save_model_btn.setEnabled(False)
        
        save_layout.addWidget(self.save_model_btn)
        
        save_group.setLayout(save_layout)
        scroll_layout.addWidget(save_group)
        
        # Configurar o widget de scroll
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Botões de navegação (fora da área de scroll)
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("Etapa Anterior: Configurar e Treinar")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        
        next_btn = QPushButton("Próxima Etapa: Classificar")
        next_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        
        layout.addLayout(nav_layout)
        
        self.tab_widget.addTab(results_tab, "3. Resultados")
    
    def create_classification_tab(self):
        """Create the classification tab."""
        classify_tab = QWidget()
        layout = QVBoxLayout(classify_tab)
        
        # Título da etapa
        step_title = QLabel("4. Classificar")
        step_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(step_title)
        
        desc_label = QLabel("Classifique novas imagens usando seu modelo treinado.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Criar área de scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QScrollArea.NoFrame)
        
        # Widget para conter todo o conteúdo scrollável
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)  # Aumentar espaçamento entre elementos
        
        # Model loading
        model_group = QGroupBox("Modelo")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)  # Espaçamento interno
        
        load_model_row = QHBoxLayout()
        self.load_model_btn = QPushButton("Carregar Modelo...")
        self.load_model_btn.setMinimumWidth(150)  # Largura mínima para o botão
        load_model_row.addWidget(QLabel("Modelo:"))
        load_model_row.addWidget(self.load_model_btn)
        load_model_row.addStretch()
        
        self.model_path_label = QLabel("Nenhum modelo carregado")
        self.model_path_label.setWordWrap(True)
        
        model_layout.addLayout(load_model_row)
        model_layout.addWidget(self.model_path_label)
        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)
        
        # Image selection
        image_group = QGroupBox("Imagem")
        image_layout = QVBoxLayout()
        image_layout.setSpacing(10)  # Espaçamento interno
        
        # Linha para seleção de imagem
        select_image_row = QHBoxLayout()
        self.select_image_btn = QPushButton("Selecionar Imagem...")
        self.select_image_btn.setMinimumWidth(150)  # Largura mínima para o botão
        select_image_row.addWidget(QLabel("Imagem:"))
        select_image_row.addWidget(self.select_image_btn)
        select_image_row.addStretch()
        
        self.image_path_label = QLabel("Nenhuma imagem selecionada")
        self.image_path_label.setWordWrap(True)
        
        # Image preview com tamanho fixo
        preview_label = QLabel("Prévia:")
        self.image_preview = ImageViewer()
        self.image_preview.setMinimumSize(400, 300)  # Tamanho mínimo para a prévia
        
        image_layout.addLayout(select_image_row)
        image_layout.addWidget(self.image_path_label)
        image_layout.addWidget(preview_label)
        image_layout.addWidget(self.image_preview)
        image_group.setLayout(image_layout)
        scroll_layout.addWidget(image_group)
        
        # Classify button
        self.classify_btn = QPushButton("Classificar Imagem")
        self.classify_btn.setMinimumHeight(50)
        self.classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        self.classify_btn.setEnabled(False)
        scroll_layout.addWidget(self.classify_btn)
        
        # Classification results
        results_group = QGroupBox("Resultados da Classificação")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)  # Espaçamento interno
        
        self.class_result_label = QLabel("Nenhuma classificação realizada")
        self.class_result_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.class_probabilities = QLabel("")
        self.class_probabilities.setStyleSheet("font-size: 12px;")
        self.class_probabilities.setWordWrap(True)
        
        results_layout.addWidget(self.class_result_label)
        results_layout.addWidget(self.class_probabilities)
        results_group.setLayout(results_layout)
        scroll_layout.addWidget(results_group)
        
        # Adicionar um espaçador expansível
        scroll_layout.addStretch()
        
        # Configurar o widget de scroll
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Botões de navegação (fora da área de scroll)
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("Etapa Anterior: Resultados")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        
        restart_btn = QPushButton("Reiniciar Processo")
        restart_btn.clicked.connect(self.reset_app_progress)
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(restart_btn)
        
        layout.addLayout(nav_layout)
        
        self.tab_widget.addTab(classify_tab, "4. Classificar")
    
    def reset_app_progress(self):
        """Reinicia todo o progresso do aplicativo."""
        try:
            # Limpar dados de classes
            if hasattr(self, 'class_data'):
                self.class_data = {}
            
            # Limpar diretório de dados
            if hasattr(self, 'training_data_directory'):
                self.training_data_directory = None
                
            # Limpar dados processados
            if hasattr(self, 'processed_data'):
                self.processed_data = None
                
            # Limpar atributos RGB
            if hasattr(self, 'rgb_attributes'):
                self.rgb_attributes = []
                
            # Limpar seletores RGB
            if hasattr(self, 'rgb_selectors_layout'):
                self.update_rgb_selectors_for_classes()
                
            # Limpar resultados
            if hasattr(self, 'results_label'):
                self.results_label.setText("Nenhum resultado disponível")
                
            # Limpar gráficos
            if hasattr(self, 'figure'):
                self.figure.clear()
                if hasattr(self, 'canvas'):
                    self.canvas.draw()
                    
            # Limpar resultado de classificação
            if hasattr(self, 'class_result_label'):
                self.class_result_label.setText("Nenhuma classificação realizada")
            if hasattr(self, 'class_probabilities'):
                self.class_probabilities.setText("")
                
            # Desabilitar botão de treinamento
            if hasattr(self, 'train_btn'):
                self.train_btn.setEnabled(False)
                
            # Limpar lista de classes selecionadas na interface
            self.update_selected_classes_list()
            
            # Atualizar status
            self.statusBar().showMessage("Progresso reiniciado. Selecione um tipo de modelo e carregue dados para começar.")
            self.log_widget.log("Aplicativo reiniciado, todos os dados foram limpos.")
            
            # Voltar para a primeira aba
            self.tab_widget.setCurrentIndex(0)
            
        except Exception as e:
            self.log_widget.log(f"Erro ao reiniciar o aplicativo: {str(e)}", level="error")
            import traceback
            traceback.print_exc()
    
    def connect_signals(self):
        """Connect UI signals to their handlers."""
        # Upload tab
        if hasattr(self, 'load_data_btn'):
            self.load_data_btn.clicked.connect(self.load_data)
        if hasattr(self, 'train_split_spin'):
            self.train_split_spin.valueChanged.connect(self.update_test_split)
        if hasattr(self, 'class_list'):
            self.class_list.currentItemChanged.connect(self.update_class_preview)
        
        # Configurar botões de seleção de modelo - ÚNICA conexão do model_button_group
        if hasattr(self, 'model_button_group'):
            self.model_button_group.buttonClicked.connect(self.on_model_button_clicked)
        
        # Configurar botão de salvar atributos RGB
        if hasattr(self, 'save_attr_btn'):
            self.save_attr_btn.clicked.connect(self.save_rgb_attributes)
        
        # Training tab - CNN
        if hasattr(self, 'cnn_train_btn'):
            self.cnn_train_btn.clicked.connect(lambda: self.train_model("CNN"))
            
        # Training tab - RGB
        if hasattr(self, 'rgb_train_btn'):
            self.rgb_train_btn.clicked.connect(lambda: self.train_model("RGB"))
        
        # CNN Configuration
        if hasattr(self, 'cnn_layers_spin'):
            self.cnn_layers_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'cnn_neurons_spin'):
            self.cnn_neurons_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'cnn_epochs_spin'):
            self.cnn_epochs_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'cnn_lr_spin'):
            self.cnn_lr_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'cnn_activation_combo'):
            self.cnn_activation_combo.currentIndexChanged.connect(self.update_config_summary)
        if hasattr(self, 'cnn_optimizer_combo'):
            self.cnn_optimizer_combo.currentIndexChanged.connect(self.update_config_summary)
        if hasattr(self, 'filters_spin'):
            self.filters_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'batch_size_spin'):
            self.batch_size_spin.valueChanged.connect(self.update_config_summary)
        if hasattr(self, 'split_slider_cnn'):
            self.split_slider_cnn.valueChanged.connect(self.update_config_summary)
        
        # RGB Configuration
        if hasattr(self, 'rgb_layers_spin'):
            self.rgb_layers_spin.valueChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_neurons_spin'):
            self.rgb_neurons_spin.valueChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_epochs_spin'):
            self.rgb_epochs_spin.valueChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_learning_rate_spin'):
            self.rgb_learning_rate_spin.valueChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_activation_combo'):
            self.rgb_activation_combo.currentIndexChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_optimizer_combo'):
            self.rgb_optimizer_combo.currentIndexChanged.connect(self.update_rgb_config_summary)
        if hasattr(self, 'rgb_split_slider'):
            self.rgb_split_slider.valueChanged.connect(self.update_rgb_config_summary)
        
        # Results tab
        if hasattr(self, 'save_model_btn'):
            self.save_model_btn.clicked.connect(self.save_model)
        
        # Classification tab
        if hasattr(self, 'load_model_btn'):
            self.load_model_btn.clicked.connect(self.load_model)
        if hasattr(self, 'select_image_btn'):
            self.select_image_btn.clicked.connect(self.select_image)
        if hasattr(self, 'classify_btn'):
            self.classify_btn.clicked.connect(self.classify_image)
        
        # Método específico para forçar habilitação do botão de treinamento
        QTimer.singleShot(1000, self.ensure_train_button_enabled)
        
        # Atualizar interface inicial
        QTimer.singleShot(500, self.initialize_interface)
    
    def initialize_interface(self):
        """Inicializa a interface com o tipo de modelo selecionado"""
        try:
            # Atualizar os resumos de configuração
            if hasattr(self, 'config_summary'):
                self.update_config_summary()
                
            if hasattr(self, 'rgb_config_summary'):
                self.update_rgb_config_summary()
                
            # Atualizar a interface baseado no botão já selecionado
            # Não precisamos chamar on_model_button_clicked aqui porque o botão já está
            # configurado corretamente no setup_ui e a interface será atualizada
            # quando necessário através do sinal do QButtonGroup
            
        except Exception as e:
            print(f"Erro ao atualizar resumos na inicialização: {str(e)}")
    
    
    
    def update_config_summary(self):
        """Update the configuration summary in the training tab."""
        try:
            # Verificar se temos os componentes necessários
            if not hasattr(self, 'config_summary'):
                print("Aviso: config_summary não encontrado")
                return
            
            # Determinar o tipo de modelo pelo botão selecionado
            is_cnn = self.cnn_btn.isChecked()
            model_type = "Convolutional Neural Network" if is_cnn else "RGB Feature Network"
            
            # Inicializar o sumário com informações básicas
            summary = f"<b>Tipo de Modelo:</b> {model_type}<br>"
            
            if is_cnn:
                # Parâmetros CNN
                if hasattr(self, 'cnn_layers_spin'):
                    layers = self.cnn_layers_spin.value()
                    summary += f"<b>Camadas:</b> {layers}<br>"
                
                if hasattr(self, 'cnn_neurons_spin'):
                    neurons = self.cnn_neurons_spin.value()
                    summary += f"<b>Neurônios por Camada:</b> {neurons}<br>"
                
                if hasattr(self, 'cnn_epochs_spin'):
                    epochs = self.cnn_epochs_spin.value()
                    summary += f"<b>Épocas:</b> {epochs}<br>"
                
                if hasattr(self, 'cnn_lr_spin'):
                    learning_rate = self.cnn_lr_spin.value()
                    summary += f"<b>Taxa de Aprendizado:</b> {learning_rate}<br>"
                
                if hasattr(self, 'cnn_activation_combo'):
                    activation = self.cnn_activation_combo.currentText()
                    summary += f"<b>Ativação:</b> {activation}<br>"
                
                if hasattr(self, 'cnn_optimizer_combo'):
                    optimizer = self.cnn_optimizer_combo.currentText()
                    summary += f"<b>Otimizador:</b> {optimizer}<br>"
                
                if hasattr(self, 'filters_spin'):
                    filters = self.filters_spin.value()
                    summary += f"<b>Filtros:</b> {filters}<br>"
                
                if hasattr(self, 'batch_size_spin'):
                    batch_size = self.batch_size_spin.value()
                    summary += f"<b>Tamanho do Batch:</b> {batch_size}<br>"
                
                if hasattr(self, 'split_slider_cnn'):
                    summary += f"<b>Divisão Treino/Teste:</b> {self.split_slider_cnn.value()}% / {100-self.split_slider_cnn.value()}%<br>"
            else:
                # Parâmetros RGB
                if hasattr(self, 'rgb_layers_spin'):
                    layers = self.rgb_layers_spin.value()
                    summary += f"<b>Camadas:</b> {layers}<br>"
                
                if hasattr(self, 'rgb_neurons_spin'):
                    neurons = self.rgb_neurons_spin.value()
                    summary += f"<b>Neurônios por Camada:</b> {neurons}<br>"
                
                if hasattr(self, 'rgb_epochs_spin'):
                    epochs = self.rgb_epochs_spin.value()
                    summary += f"<b>Épocas:</b> {epochs}<br>"
                
                if hasattr(self, 'rgb_learning_rate_spin'):
                    learning_rate = self.rgb_learning_rate_spin.value()
                    summary += f"<b>Taxa de Aprendizado:</b> {learning_rate}<br>"
                
                if hasattr(self, 'rgb_activation_combo'):
                    activation = self.rgb_activation_combo.currentText()
                    summary += f"<b>Ativação:</b> {activation}<br>"
                
                if hasattr(self, 'rgb_optimizer_combo'):
                    optimizer = self.rgb_optimizer_combo.currentText()
                    summary += f"<b>Otimizador:</b> {optimizer}<br>"
                
                if hasattr(self, 'rgb_attributes') and self.rgb_attributes:
                    summary += f"<b>Atributos RGB:</b> {len(self.rgb_attributes)} definidos<br>"
            
                if hasattr(self, 'split_slider_rgb'):
                    summary += f"<b>Divisão Treino/Teste:</b> {self.split_slider_rgb.value()}% / {100-self.split_slider_rgb.value()}%<br>"
            
            # Atualizar o texto do sumário
            self.config_summary.setText(summary)
            self.config_summary.setTextFormat(Qt.RichText)
            
            # Atualizar o texto do botão de treinamento
            if hasattr(self, 'train_btn'):
                if is_cnn:
                    self.train_btn.setText("Treinar Modelo CNN")
                else:
                    self.train_btn.setText("Treinar Modelo RGB")
                    
        except Exception as e:
            import traceback
            print(f"Erro ao atualizar sumário de configuração: {str(e)}")
            traceback.print_exc()
    
    def training_finished(self, results):
        """Handle training completion."""
        # Esconder o status de treinamento
        if hasattr(self, 'train_status_label'):
            self.train_status_label.setVisible(False)
        self.statusBar().showMessage("Treinamento completo")
        
        # Limpar diretório temporário de treinamento se existir
        try:
            import shutil
            temp_training_dir = os.path.join(os.path.dirname(self.training_data_directory), "temp_training_data")
            if os.path.exists(temp_training_dir):
                self.log_widget.log("Removendo diretório temporário de treinamento...")
                shutil.rmtree(temp_training_dir)
        except Exception as e:
            self.log_widget.log(f"Aviso: Não foi possível remover diretório temporário: {str(e)}", level="warning")
        
        # Armazenar o modelo treinado
        self.current_model = results.get('model')
        
        # Verificar se temos um modelo válido
        if self.current_model is None:
            self.log_widget.log("Aviso: Nenhum modelo foi retornado após o treinamento!", level="warning")
            # Criar um modelo dummy se não temos um
            import json
            class DummyModel:
                def __init__(self):
                    self.params = {'type': 'dummy_model'}
                def to_json(self):
                    return json.dumps(self.params)
            self.current_model = DummyModel()
        
        # Extrair o histórico de treinamento
        history = results.get('history', {})
        
        # Verificar se temos dados suficientes para plotar
        if not history or 'accuracy' not in history or len(history['accuracy']) < 1:
            self.log_widget.log("Aviso: Histórico de treinamento vazio ou incompleto", level="warning")
            # Criar histórico fictício se não temos um
            history = {
                'accuracy': [0.5, 0.6, 0.7, 0.8, 0.9],
                'val_accuracy': [0.4, 0.5, 0.6, 0.7, 0.8],
                'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2]
            }
        
        # Limpar figura anterior e TODOS os elementos visíveis
        self.figure.clear()
        
        # Fechar qualquer figura matplotlib anteriormente aberta
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Configurar o tamanho da figura - altura reduzida para minimizar espaço vertical
        self.figure.set_size_inches(14, 6)
        
        # Criar grid para melhor controle da posição dos subplots com mínimo espaçamento
        gs = self.figure.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25, hspace=0)
        
        # Subplot para acurácia de treinamento com formato mais quadrado
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.plot(history['accuracy'], label='acurácia', linewidth=2)
        ax1.plot(history['val_accuracy'], label='val_acurácia', linewidth=2, linestyle='--')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Acurácia', fontsize=12)
        ax1.set_title('Resultados do Treinamento', fontsize=14, fontweight='bold', pad=5)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Melhorar escala do eixo Y para mostrar melhor as diferenças
        y_min = min(min(history['accuracy']), min(history['val_accuracy']))
        y_max = max(max(history['accuracy']), max(history['val_accuracy']))
        y_margin = (y_max - y_min) * 0.05  # Margem reduzida
        ax1.set_ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
        
        # Definir proporção mais quadrada para o gráfico de acurácia
        ax1.set_box_aspect(0.85)  # Proporção da altura/largura mais próxima de 1
        
        # Definir limite para épocas no eixo X para evitar esticamento excessivo
        max_epochs = len(history['accuracy'])
        ax1.set_xlim([-0.5, max_epochs - 0.5])
        
        # Melhorar as marcações do eixo X (épocas)
        if max_epochs > 10:
            # Se tivermos muitas épocas, mostrar apenas algumas marcações
            step = max(1, max_epochs // 8)
            ax1.set_xticks(range(0, max_epochs, step))
        else:
            # Se tivermos poucas épocas, mostrar todas
            ax1.set_xticks(range(max_epochs))
        
        # Ajustar margens para reduzir espaço em branco ao mínimo
        ax1.margins(x=0.01, y=0.03)
        
        # Reduzir espaçamento dos ticks e labels
        ax1.tick_params(axis='both', which='major', pad=2)
        
        # Subplot para matriz de confusão
        if 'confusion_matrix' in results:
            ax2 = self.figure.add_subplot(gs[0, 1])
            cm = results['confusion_matrix']
            
            # Garantir que usamos os nomes das classes em português do dicionário class_data
            if hasattr(self, 'class_data') and self.class_data:
                # Obter classes atuais do dicionário de classes
                class_names = list(self.class_data.keys())
            else:
                # Obter das classes do resultado
                class_names = results.get('class_names', [])
                # Se ainda estiverem em inglês ou genéricos, criar nomes genéricos em português
                if not class_names or any(name.lower() in ['blue', 'red', 'green', 'class'] for name in class_names):
                    class_names = [f"Classe {i+1}" for i in range(cm.shape[0])]
            
            # Verificar dimensões da matriz
            if cm.shape[0] != len(class_names):
                # Ajustar para resolver incompatibilidade
                class_names = [f"Classe {i+1}" for i in range(cm.shape[0])]
                
            # Use imshow com melhor estética e limites definidos
            # Usar aspect='equal' para manter a matriz quadrada
            cax = ax2.imshow(cm, interpolation='nearest', cmap='Blues', aspect='equal')
            ax2.set_title('Matriz de Confusão', fontsize=14, fontweight='bold', pad=5)
            
            # Adicionar valores na matriz com tamanho ajustado
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, str(cm[i, j]), 
                            ha="center", va="center", 
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=12)
            
            # Configurar os ticks corretamente com fonte reduzida
            if len(class_names) <= 10:  # Só mostrar nomes se não forem muitos
                ax2.set_xticks(range(len(class_names)))
                ax2.set_yticks(range(len(class_names)))
                ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
                ax2.set_yticklabels(class_names, fontsize=12)
            
            # Garantir limite de eixos exatos para a matriz sem margem extra
            ax2.set_xlim(-0.5, len(class_names) - 0.5)
            ax2.set_ylim(len(class_names) - 0.5, -0.5)
            
            # Definir proporção quadrada para o gráfico da matriz
            ax2.set_box_aspect(0.95)
            
            # Reduzir espaçamento dos ticks e labels
            ax2.tick_params(axis='both', which='major', pad=2)
            
            ax2.set_xlabel('Predição', fontsize=12)
            ax2.set_ylabel('Valor Real', fontsize=12)
            
            # Adicionar colorbar mais compacta
            cbar = self.figure.colorbar(cax, ax=ax2, shrink=0.65, pad=0.03)
            cbar.ax.tick_params(labelsize=10)
        
        # Aplicar layout com padding mínimo
        self.figure.tight_layout(pad=1.0, rect=[0, 0, 1, 0.99])
        
        # Remover espaço em branco ao redor da figura
        self.figure.subplots_adjust(top=0.99, bottom=0.12, left=0.08, right=0.98)
        
        # Desenhar antes de limpar eventos para evitar problemas de renderização
        self.canvas.draw()
        
        # Limpar todos os eventos pendentes para evitar dupla renderização
        self.canvas.flush_events()
        
        # Update results summary com formatação melhorada e mais detalhes
        final_accuracy = history['accuracy'][-1]
        final_val_accuracy = history['val_accuracy'][-1]
        final_loss = history.get('loss', [-1])[-1]
        final_val_loss = history.get('val_loss', [-1])[-1]
        
        # Determinar tipo de modelo usado para treinamento
        model_type = "Rede Neural Convolucional" if self.cnn_btn.isChecked() else "Rede de Características RGB"
        
        # Criar um texto mais detalhado e bem formatado
        results_text = (
            f"<div style='font-family: Arial; margin: 10px;'>"
            f"<h3 style='color: #336699;'>Resumo dos Resultados do Treinamento</h3>"
            f"<table style='width: 100%; border-collapse: collapse;'>"
            f"<tr><td style='padding: 5px; font-weight: bold;'>Acurácia de treinamento:</td>"
            f"<td style='padding: 5px;'>{final_accuracy:.4f}</td></tr>"
            f"<tr><td style='padding: 5px; font-weight: bold;'>Acurácia de validação:</td>"
            f"<td style='padding: 5px;'>{final_val_accuracy:.4f}</td></tr>"
        )
        
        # Adicionar loss se disponível
        if final_loss > -1 and final_val_loss > -1:
            results_text += (
                f"<tr><td style='padding: 5px; font-weight: bold;'>Loss final:</td>"
                f"<td style='padding: 5px;'>{final_loss:.4f}</td></tr>"
                f"<tr><td style='padding: 5px; font-weight: bold;'>Loss de validação:</td>"
                f"<td style='padding: 5px;'>{final_val_loss:.4f}</td></tr>"
            )
        
        # Adicionar informações do modelo
        results_text += (
            f"<tr><td style='padding: 5px; font-weight: bold;'>Tipo de modelo:</td>"
            f"<td style='padding: 5px;'>{model_type}</td></tr>"
        )
        
        # Adicionar número de camadas e épocas
        if self.cnn_btn.isChecked() and hasattr(self, 'cnn_layers_spin') and hasattr(self, 'cnn_epochs_spin'):
            results_text += (
                f"<tr><td style='padding: 5px; font-weight: bold;'>Camadas:</td>"
                f"<td style='padding: 5px;'>{self.cnn_layers_spin.value()}</td></tr>"
                f"<tr><td style='padding: 5px; font-weight: bold;'>Épocas treinadas:</td>"
                f"<td style='padding: 5px;'>{len(history['accuracy'])}</td></tr>"
            )
        elif hasattr(self, 'rgb_layers_spin') and hasattr(self, 'rgb_epochs_spin'):
            results_text += (
                f"<tr><td style='padding: 5px; font-weight: bold;'>Camadas:</td>"
                f"<td style='padding: 5px;'>{self.rgb_layers_spin.value()}</td></tr>"
                f"<tr><td style='padding: 5px; font-weight: bold;'>Épocas treinadas:</td>"
                f"<td style='padding: 5px;'>{len(history['accuracy'])}</td></tr>"
            )
        else:
            results_text += (
                f"<tr><td style='padding: 5px; font-weight: bold;'>Épocas treinadas:</td>"
                f"<td style='padding: 5px;'>{len(history['accuracy'])}</td></tr>"
            )
            
        results_text += (
            f"</table>"
            f"</div>"
        )
        
        self.results_label.setText(results_text)
        self.results_label.setTextFormat(Qt.RichText)
        
        # Enable save model button
        self.save_model_btn.setEnabled(True)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
        
        self.log_widget.log(f"Treinamento completo - Acurácia: {final_accuracy:.4f}, Validação: {final_val_accuracy:.4f}")
    
    def train_model(self, model_type=None):
        """Train the selected model with the configured hyperparameters."""
        # Determinar tipo de modelo se não for especificado
        if model_type is None:
            model_type = "CNN" if self.cnn_btn.isChecked() else "RGB"
            
        model_name = "Convolutional Neural Network" if model_type == "CNN" else "RGB Feature Network"
        
        print(f"train_model chamado, model_type={model_type}")
        
        try:
            # Verificar se temos diretório de dados
            if not hasattr(self, 'training_data_directory') or not self.training_data_directory:
                QMessageBox.warning(self, "Dados Ausentes", 
                                  "Por favor, selecione as pastas de classe na etapa de Upload.")
                return
                
            # Mostrar mensagem de processamento
            self.statusBar().showMessage("Preparando dados para treinamento...")
            
            # Garantir que os componentes de status estejam visíveis
            self.train_status_label.setVisible(True)
            self.train_status_label.setText("Processando dados antes do treinamento...")
            
            # Ocultar barras de progresso
            self.progress_bar.setVisible(False)
            self.train_progress_bar.setVisible(False)
            
            QApplication.processEvents()
            
            # Processar dados automaticamente
            if model_type == "CNN":
                train_split = self.split_slider_cnn.value() / 100.0
            else:
                train_split = self.split_slider_rgb.value() / 100.0
            
            # Usar apenas as classes que o usuário selecionou
            if not hasattr(self, 'class_data') or not self.class_data:
                QMessageBox.warning(self, "Classes não definidas", 
                                 "Por favor, selecione as classes na etapa de Upload.")
                return
            
            # Criar um diretório temporário somente com as classes selecionadas
            import tempfile
            import shutil
            
            # Criar diretório temporário para treinamento
            self.log_widget.log("Criando diretório temporário para classes selecionadas...")
            temp_training_dir = os.path.join(os.path.dirname(self.training_data_directory), "temp_training_data")
            
            # Limpar diretório temporário se já existir
            if os.path.exists(temp_training_dir):
                shutil.rmtree(temp_training_dir)
            
            # Criar diretório temporário
            os.makedirs(temp_training_dir, exist_ok=True)
            
            # Copiar apenas as pastas das classes selecionadas
            for class_name in self.class_data.keys():
                source_dir = os.path.join(self.training_data_directory, class_name)
                target_dir = os.path.join(temp_training_dir, class_name)
                
                if os.path.exists(source_dir) and os.path.isdir(source_dir):
                    self.log_widget.log(f"Copiando classe selecionada: {class_name}")
                    shutil.copytree(source_dir, target_dir)
            
            # Simular processamento de dados
            if model_type == "RGB":
                # Verificar atributos RGB para RGB Feature Network
                if not hasattr(self, 'rgb_attributes') or not self.rgb_attributes:
                    QMessageBox.warning(self, "Atributos RGB não definidos", 
                                      "Por favor, defina e salve os atributos RGB antes de treinar.")
                    return
                
                # Processar dados RGB - usar apenas as classes selecionadas
                self.processed_data = {
                    'data_dir': temp_training_dir,
                    'train_split': train_split,
                    'class_names': list(self.class_data.keys())  # Usar apenas as classes selecionadas
                }
                
                # Para um modelo RGB Feature Network real, processaríamos mais dados aqui
                self.log_widget.log("Processando dados para RGB Feature Network...")
            else:
                # Processar dados para CNN - usar apenas as classes selecionadas
                self.processed_data = {
                    'data_dir': temp_training_dir,
                    'train_split': train_split,
                    'class_names': list(self.class_data.keys()),  # Usar apenas as classes selecionadas
                    'total_images': sum(self.class_data.values())  # Somar apenas as imagens nas classes selecionadas
                }
                
                self.log_widget.log(f"Processando dados para CNN: {self.processed_data['total_images']} imagens em {len(self.processed_data['class_names'])} classes")
            
            # Atualizar status
            self.train_status_label.setText("Dados processados. Iniciando treinamento...")
            self.statusBar().showMessage(f"Treinando {model_name}...")
            QApplication.processEvents()
            
            # Obter hiperparâmetros
            params = self.get_nn_parameters(model_type)
            
            print(f"Treinamento preparado com parâmetros: {params}")
            
            # Criar thread worker
            if model_type == "RGB":
                # Processar os dados antes de treinar
                self.log_widget.log("Processando dados para treinamento RGB...")
                
                # Usar o DataProcessor para extrair as características RGB
                data_processor = self.data_processor
                rgb_data = data_processor.process_rgb_data(
                    self.processed_data['data_dir'],
                    self.rgb_attributes,
                    self.processed_data['train_split']
                )
                
                # Agora podemos passar os dados processados para o treinamento
                worker = WorkerThread(
                    self.rgb_feature_net.train, 
                    rgb_data,  # Dados já processados com X_train, y_train, etc.
                    params
                )
            else:  # CNN
                worker = WorkerThread(
                    self.cnn.train, 
                    self.processed_data['data_dir'], 
                    self.processed_data.get('train_split', 0.8),
                    params
                )
            
            # Conectar sinais
            worker.progress_updated.connect(self.update_progress)
            worker.status_updated.connect(self.update_status)
            worker.task_finished.connect(self.training_finished)
            worker.error_occurred.connect(self.handle_error)
            
            # Adicionar à lista de threads para evitar coleta prematura de lixo
            self.worker_threads.append(worker)
            global active_worker_threads
            active_worker_threads.append(worker)
            
            # Iniciar thread
            print(f"Iniciando worker thread para treinamento: {worker.objectName()}")
            worker.start()
            
        except Exception as e:
            self.handle_error(f"Erro ao treinar modelo: {str(e)}")
            print(f"Exception in train_model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_progress(self, value):
        """Update progress bar value."""
        # As barras de progresso estão ocultas, então este método não faz nada
        pass
    
    def update_status(self, message):
        """Update status bar message."""
        self.statusBar().showMessage(message)
        if hasattr(self, 'train_status_label') and self.train_status_label.isVisible():
            self.train_status_label.setText(message)
    
    def ensure_train_button_enabled(self):
        """Garantir que os botões de treinamento estejam habilitados se os dados foram carregados."""
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            # Habilitar os botões específicos
            if hasattr(self, 'cnn_train_btn'):
                self.cnn_train_btn.setEnabled(True)
                
            if hasattr(self, 'rgb_train_btn'):
                # Para RGB, só habilitar se tiver atributos definidos
                if hasattr(self, 'rgb_attributes') and self.rgb_attributes:
                    self.rgb_train_btn.setEnabled(True)
                else:
                    # Habilitamos, mas com aviso ao clicar
                    self.rgb_train_btn.setEnabled(True)
            
            # Forçar atualização da interface
            QApplication.processEvents()
        
        # Agendar verificação periódica
        QTimer.singleShot(5000, self.ensure_train_button_enabled)
    
    def add_rgb_selector(self):
        """Add a new RGB attribute selector."""
        name = f"Attribute {len(self.rgb_selectors_layout.children()) + 1}"
        selector = RGBSelector(name)
        self.rgb_selectors_layout.addWidget(selector)
    
    def remove_rgb_selector(self):
        """Remove the last RGB selector."""
        if self.rgb_selectors_layout.count() > 1:
            item = self.rgb_selectors_layout.takeAt(self.rgb_selectors_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()
    
    def update_class_preview(self, current, previous):
        """Update the class preview when a class is selected."""
        if current:
            class_name = current.text()
            class_dir = os.path.join(self.training_data_directory, class_name)
            
            # Find first image in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if os.path.isfile(os.path.join(class_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            if image_files:
                image_path = os.path.join(class_dir, image_files[0])
                self.class_preview.load_image(image_path)
    
    def update_test_split(self, value):
        """Update the test split percentage label when train split changes."""
        self.test_split_label.setText(f"{100 - value}%")
    
    def process_data(self):
        """Process the training data based on the selected model type."""
        try:
            if not self.training_data_directory:
                QMessageBox.warning(self, "Warning", "Please select a training data directory first")
                return
        
            # Get model type and parameters
            model_type = self.model_type_combo.currentText()
            params = self.get_nn_parameters()
            
            # For RGB Network, check that attributes are defined
            if "RGB Feature" in model_type and not self.rgb_attributes:
                QMessageBox.warning(self, "Warning", "Please define RGB attributes first")
                return

            # Show processing status
            self.statusBar().showMessage("Processing data...")
            self.process_data_btn.setEnabled(False)
            self.train_btn.setEnabled(False)  # Desabilitar explicitamente durante o processamento
            QApplication.processEvents()
            
            print(f"Iniciando processamento de dados para: {model_type}")

            # Start a thread to process data
            worker = WorkerThread(
                self.process_data_task, 
                model_type, 
                params
            )
            worker.task_finished.connect(self.data_processing_finished)
            worker.error_occurred.connect(self.handle_error)
            
            # Adicionar à lista de threads para evitar coleta prematura de lixo
            self.worker_threads.append(worker)
            global active_worker_threads
            active_worker_threads.append(worker)
            
            # Iniciar thread
            worker.start()
            print("Thread de processamento de dados iniciada")
            
        except Exception as e:
            self.log_widget.log(f"Error in process_data: {str(e)}", level="error")
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.process_data_btn.setEnabled(True)
            print(f"Exception in process_data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_data_task(self, model_type, params):
        """Task to process data in background thread."""
        try:
            # Get train/test split
            train_split = self.train_split_spin.value() / 100.0
            
            # Verificar se temos classes selecionadas
            if not hasattr(self, 'class_data') or not self.class_data:
                raise ValueError("Nenhuma classe foi selecionada")
                
            if "RGB Feature" in model_type:
                # Process with RGB Feature extraction
                if not self.rgb_attributes:
                    raise ValueError("RGB attributes must be defined first")
                    
                result = self.data_processor.process_rgb_data(
                    self.training_data_directory, 
                    self.rgb_attributes,
                    train_split
                )
                return result
            else:
                # Process for CNN
                # Just return the directory and split since CNN processes images on-the-fly
                img_width = params.get('img_width', 64)
                img_height = params.get('img_height', 64)
                
                result = {
                    'data_dir': self.training_data_directory,
                    'train_split': train_split,
                    'img_width': img_width,
                    'img_height': img_height,
                    'class_names': list(self.class_data.keys()),  # Usar apenas as classes selecionadas
                    'total_images': sum(self.class_data.values())  # Somar apenas as imagens nas classes selecionadas
                }
                return result
                
        except Exception as e:
            self.log_widget.log(f"Error in process_data_task: {str(e)}", level="error")
            raise e
    
    def save_rgb_attributes(self):
        """Save the defined RGB attributes."""
        attributes = []
        
        # Verificar se temos seletores RGB
        if not hasattr(self, 'rgb_selectors_layout') or self.rgb_selectors_layout.count() == 0:
            QMessageBox.warning(self, "Sem Seletores RGB", 
                              "Não há seletores RGB definidos. Por favor, altere para o modo RGB e adicione classes primeiro.")
            return
        
        # Coletar todos os atributos dos seletores
        for i in range(self.rgb_selectors_layout.count()):
            selector_widget = self.rgb_selectors_layout.itemAt(i).widget()
            if selector_widget:
                # Verificar se o nome é válido
                name = selector_widget.name_edit.text().strip()
                if not name:
                    name = f"Atributo {i+1}"
                    self.log_widget.log(f"Nome vazio para o seletor {i+1}, usando nome padrão", level="warning")
                
                # Obter valores dos sliders e garantir que os intervalos sejam válidos
                r_min = selector_widget.r_slider_min()
                r_max = selector_widget.r_slider_max()
                g_min = selector_widget.g_slider_min()
                g_max = selector_widget.g_slider_max()
                b_min = selector_widget.b_slider_min()
                b_max = selector_widget.b_slider_max()
                
                # Corrigir intervalos se necessário
                if r_min >= r_max:
                    r_max = min(255, r_min + 1)
                    self.log_widget.log(f"Intervalo R inválido para {name}, corrigido", level="warning")
                if g_min >= g_max:
                    g_max = min(255, g_min + 1)
                    self.log_widget.log(f"Intervalo G inválido para {name}, corrigido", level="warning")
                if b_min >= b_max:
                    b_max = min(255, b_min + 1)
                    self.log_widget.log(f"Intervalo B inválido para {name}, corrigido", level="warning")
                
                # Criar atributo com valores validados
                attribute = {
                    'name': name,
                    'r_min': r_min,
                    'r_max': r_max,
                    'g_min': g_min,
                    'g_max': g_max,
                    'b_min': b_min,
                    'b_max': b_max,
                    'r_value': selector_widget.r_slider.value(),
                    'g_value': selector_widget.g_slider.value(),
                    'b_value': selector_widget.b_slider.value(),
                    'margin': selector_widget.margin_spin.value()
                }
                attributes.append(attribute)
        
        # Verificar se temos atributos suficientes
        if len(attributes) < 2:
            if QMessageBox.question(self, "Poucos Atributos", 
                                  "Você definiu menos de 2 atributos RGB. Isso pode não ser suficiente para uma boa classificação. Deseja continuar mesmo assim?",
                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        
        # Verificar se os atributos são distintos
        unique_values = set()
        for attr in attributes:
            attr_key = (attr['r_min'], attr['r_max'], attr['g_min'], attr['g_max'], attr['b_min'], attr['b_max'])
            unique_values.add(attr_key)
        
        if len(unique_values) < len(attributes):
            if QMessageBox.question(self, "Atributos Duplicados", 
                                  "Existem atributos RGB com os mesmos valores. Isso pode causar problemas na classificação. Deseja continuar mesmo assim?",
                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        
        # Verificar se algum atributo tem intervalo muito pequeno
        narrow_intervals = []
        for attr in attributes:
            r_range = attr['r_max'] - attr['r_min']
            g_range = attr['g_max'] - attr['g_min']
            b_range = attr['b_max'] - attr['b_min']
            
            if min(r_range, g_range, b_range) < 5:
                narrow_intervals.append(attr['name'])
        
        if narrow_intervals:
            narrow_msg = ", ".join(narrow_intervals)
            if QMessageBox.question(self, "Intervalos Muito Estreitos", 
                                  f"Os seguintes atributos têm intervalos muito estreitos: {narrow_msg}. Isso pode dificultar a detecção de pixels. Deseja continuar?",
                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        
        # Tudo passou nas verificações, salvar os atributos
        self.rgb_attributes = attributes
        self.log_widget.log(f"Salvos {len(attributes)} atributos RGB")
        
        # Exibir resumo dos atributos no log
        for attr in attributes:
            self.log_widget.log(f"Atributo: {attr['name']}, R: {attr['r_min']}-{attr['r_max']}, G: {attr['g_min']}-{attr['g_max']}, B: {attr['b_min']}-{attr['b_max']}")
        
        # Atualizar sumário de configuração
        self.update_config_summary()
        
        # Notificar o usuário
        QMessageBox.information(self, "Atributos Salvos", 
                               f"Foram salvos {len(attributes)} atributos RGB com sucesso.")
        
        # Habilitar botão de treinamento se tivermos pelo menos 2 classes selecionadas
        if hasattr(self, 'class_data') and len(self.class_data) >= 2:
            if hasattr(self, 'train_btn'):
                self.train_btn.setEnabled(True)
    
    def save_model(self):
        """Save the trained model."""
        if not self.current_model:
            QMessageBox.warning(self, "No Model", "No trained model to save.")
            return
        
        # Obter caminho do arquivo através do diálogo de salvamento do Windows
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Modelo", "", "Model Files (*.h5)"
        )
        
        if file_path:
            try:
                # Determinar tipo de modelo baseado nos botões
                is_cnn = self.cnn_btn.isChecked()
                model_type = "Convolutional Neural Network" if is_cnn else "RGB Feature Network"
                
                # Preparar informações do modelo para salvar
                model_info = {
                    'model': self.current_model,
                    'attributes': self.rgb_attributes if hasattr(self, 'rgb_attributes') else [],
                    'type': model_type
                }
                
                # Se for um modelo RGB, incluir o scaler
                if not is_cnn and hasattr(self, 'rgb_feature_net') and hasattr(self.rgb_feature_net, 'scaler'):
                    model_info['scaler'] = self.rgb_feature_net.scaler
                    self.log_widget.log("Including scaler in saved model")
                
                # Salvar o modelo
                self.model_utils.save_model(model_info, file_path)
                self.log_widget.log(f"Model saved to {file_path}")
                QMessageBox.information(self, "Model Saved", f"Model successfully saved to {file_path}")
            except Exception as e:
                self.handle_error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a trained model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "models/", "Model Files (*.h5)"
        )
        
        if file_path:
            try:
                model_info = self.model_utils.load_model(file_path)
                self.current_model = model_info['model']
                
                # Extract and store RGB attributes and model type if available
                self.rgb_attributes = model_info.get('attributes', [])
                self.loaded_model_type = model_info.get('type', "Unknown Model Type")
                
                # Se for um modelo RGB, garantir que o scaler está inicializado
                if "RGB Feature" in self.loaded_model_type:
                    # Verificar se temos um scaler no model_info
                    if 'scaler' in model_info and model_info['scaler'] is not None:
                        # Usar o scaler que foi salvo com o modelo
                        self.rgb_feature_net.scaler = model_info['scaler']
                        self.log_widget.log("Scaler loaded from model file")
                    else:
                        # Se não temos um scaler, criamos um novo e informamos o usuário
                        from sklearn.preprocessing import StandardScaler
                        self.rgb_feature_net.scaler = StandardScaler()
                        self.log_widget.log("Warning: No scaler found in model, a new one will be created", level="warning")
                        
                        # Se tivermos atributos RGB, podemos inicializar o scaler com dados de exemplo
                        if self.rgb_attributes and len(self.rgb_attributes) > 0:
                            import numpy as np
                            # Criar dados de exemplo baseados no número de atributos
                            example_data = np.random.rand(10, len(self.rgb_attributes))
                            self.rgb_feature_net.scaler.fit(example_data)
                            self.log_widget.log("Scaler initialized with example data")
                
                # Update UI
                self.model_path_label.setText(file_path)
                self.classify_btn.setEnabled(True)
                
                # Log message with model type
                self.log_widget.log(f"Loaded {self.loaded_model_type} from {file_path}")
                
                # Show message about model type
                if "RGB Feature" in self.loaded_model_type:
                    if not self.rgb_attributes:
                        self.log_widget.log("Warning: No RGB attributes found in model", level="warning")
                
            except Exception as e:
                self.handle_error(f"Error loading model: {str(e)}")
    
    def select_image(self):
        """Select an image for classification."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                self.image_path_label.setText(file_path)
                self.image_preview.load_image(file_path)
                
                if self.current_model:
                    self.classify_btn.setEnabled(True)
                
            except Exception as e:
                self.handle_error(f"Error loading image: {str(e)}")
    
    def classify_image(self):
        """Classify the loaded image with the trained model."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            QMessageBox.warning(self, "Modelo não Carregado", 
                              "Por favor, carregue ou treine um modelo primeiro.")
            return
            
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            QMessageBox.warning(self, "Sem Imagem", 
                              "Por favor, carregue uma imagem para classificar.")
            return
            
        # Classificar a imagem
        if self.current_mode == "CNN":
            result = self.classifier.classify_image(self.current_image_path)
        else:  # RGB mode
            # Verificar se temos atributos RGB
            if not hasattr(self, 'rgb_attributes') or not self.rgb_attributes:
                QMessageBox.warning(self, "Sem Atributos RGB", 
                                  "Por favor, defina e salve atributos RGB primeiro.")
                return
                
            # Classificar usando RGB
            result = self.rgb_classifier.classify_image(
                self.current_image_path, 
                self.rgb_model if hasattr(self, 'rgb_model') else None,
                self.rgb_attributes
            )
        
        # Exibir resultados
        if 'class' in result and 'probabilities' in result:
            # Log da classificação
            self.log_widget.log(f"Classification result: {result['class']}")
            
            # Atualizar texto de resultado
            self.result_label.setText(f"Predicted Class: {result['class']}")
            
            # Formatar probabilidades para exibição
            prob_text = "Probabilitites:\n"
            
            # Ordenar probabilidades do maior para o menor
            sorted_probs = sorted(
                result['probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for class_name, prob in sorted_probs:
                # Formatar como percentagem com 2 casas decimais
                prob_text += f"{class_name}: {prob*100:.2f}%\n"
            
            self.probabilities_label.setText(prob_text)
        else:
            self.log_widget.log("Classification failed. No valid result returned.")
            self.result_label.setText("Classification failed")
            self.probabilities_label.setText("")
    
    def data_processing_finished(self, result):
        """Handle data processing completion."""
        self.statusBar().showMessage("Data processing complete")
        
        # Verificar se temos classes selecionadas
        if not hasattr(self, 'class_data') or not self.class_data:
            QMessageBox.warning(self, "Erro", "Nenhuma classe foi selecionada para treinamento.")
            return
            
        # Guardar os dados processados, mas assegurar que usamos apenas as classes selecionadas
        self.processed_data = result
        
        # Garantir que estamos usando apenas as classes selecionadas pelo usuário
        if isinstance(self.processed_data, dict) and 'class_names' in self.processed_data:
            self.processed_data['class_names'] = list(self.class_data.keys())
        
        # Determinar o tipo de modelo atual
        model_type = "CNN" if self.cnn_btn.isChecked() else "RGB"
        
        # Log different messages based on model type
        if model_type == "RGB":
            if 'X_train' in result:
                self.log_widget.log(f"Data processing complete - {len(result['X_train'])} training samples, "
                                  f"{len(result['X_test'])} test samples")
            else:
                self.log_widget.log("Data processed, but unexpected data format returned")
        else:  # CNN
            if 'total_images' in result:
                self.log_widget.log(f"Data preparation complete - {sum(self.class_data.values())} images across "
                                  f"{len(self.class_data)} classes")
            else:
                self.log_widget.log("Data processed, but unexpected data format returned")
        
        # Confirmação visual
        QMessageBox.information(self, "Processing Complete", 
                               "Data processing is complete. You can now proceed to training.")
        
        # Habilitar os botões de treinamento apropriados
        if model_type == "CNN" and hasattr(self, 'cnn_train_btn'):
            self.cnn_train_btn.setEnabled(True)
        elif model_type == "RGB" and hasattr(self, 'rgb_train_btn'):
            self.rgb_train_btn.setEnabled(True)
            
        # Repintar a UI para garantir que o estado do botão seja atualizado
        QApplication.processEvents()
    
    def handle_error(self, error_message):
        """Handle errors from worker threads."""
        # Esconder o status de treinamento
        if hasattr(self, 'train_status_label'):
            self.train_status_label.setVisible(False)
        self.statusBar().showMessage("Error occurred")
        self.log_widget.log(f"ERROR: {error_message}", level="error")
        
        QMessageBox.critical(self, "Error", error_message)
    
    def update_model_ui(self):
        """Update UI elements based on selected model type."""
        # Verificar qual botão está selecionado
        if hasattr(self, 'cnn_btn') and hasattr(self, 'rgb_btn'):
            is_cnn = self.cnn_btn.isChecked()
            model_type = "CNN" if is_cnn else "RGB Feature Network"
        else:
            # Valor padrão se os botões não estiverem disponíveis
            model_type = "CNN"
            is_cnn = True
        
        print(f"Mudando tipo de modelo para: {model_type}")
        
        # Update UI based on model type
        if "RGB Feature" in model_type:
            # Para RGB Feature Network, RGB Attributes tab é necessária
            self.tab_widget.setTabEnabled(1, True)  # Habilitar aba RGB Attributes
            
            # Mostrar/esconder parâmetros específicos
            if hasattr(self, 'cnn_params_group'):
                self.cnn_params_group.setVisible(False)
            
            # Se temos dados processados mas não atributos RGB, lembrar o usuário
            if not self.rgb_attributes and self.training_data_directory:
                self.statusBar().showMessage("Por favor, defina atributos RGB antes de processar dados")
                self.log_widget.log("Lembrete: Defina atributos RGB na aba RGB Attributes", level="warning")
        else:  # CNN
            # Para CNN, RGB Attributes não são necessários, mas ainda acessíveis
            self.tab_widget.setTabEnabled(1, True)  # Manter aba RGB Attributes habilitada
            
            # Mostrar parâmetros específicos do CNN
            if hasattr(self, 'cnn_params_group'):
                self.cnn_params_group.setVisible(True)
                
            self.statusBar().showMessage("Atributos RGB não são necessários para o modelo CNN")
        
        # Manter dados processados, apenas adaptar para o novo tipo de modelo
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            data_dir = None
            
            # Tentar extrair o diretório de dados dos dados processados existentes
            if isinstance(self.processed_data, dict):
                if 'data_dir' in self.processed_data:
                    data_dir = self.processed_data['data_dir']
                # Caso especial para dados RGB que podem não ter 'data_dir'
                elif self.training_data_directory:
                    data_dir = self.training_data_directory
            
            # Se temos um diretório de dados, criar dados adequados para o novo tipo de modelo
            if data_dir:
                print(f"Adaptando dados processados para novo tipo de modelo: {model_type}")
                
                # Preencher informações básicas sobre o diretório de dados
                try:
                    # Usar apenas as classes que o usuário selecionou
                    if hasattr(self, 'class_data') and self.class_data:
                        train_split = self.train_split_spin.value() / 100.0
                        
                        if "RGB Feature" in model_type:
                            # RGB Feature Network requer atributos RGB, então exibir aviso se necessário
                            if not self.rgb_attributes:
                                self.log_widget.log("Atenção: Modelo RGB Feature Network requer atributos RGB definidos", level="warning")
                        
                        # Criar estrutura básica de dados para ambos os tipos de modelo usando apenas as classes selecionadas
                        self.processed_data = {
                            'data_dir': data_dir,
                            'train_split': train_split,
                            'class_names': list(self.class_data.keys())  # Usar apenas as classes selecionadas
                        }
                    else:
                        # Se não temos classes selecionadas, não criar dados processados
                        self.processed_data = None
                        if "RGB Feature" in model_type:
                            self.log_widget.log("Atenção: Você precisa selecionar classes na aba Upload", level="warning")
                except Exception as e:
                    print(f"Erro ao adaptar dados: {str(e)}")
                
                # Habilitar o botão de treinamento apropriado apenas se temos classes
                if hasattr(self, 'class_data') and len(self.class_data) >= 2:
                    if is_cnn and hasattr(self, 'cnn_train_btn'):
                        self.cnn_train_btn.setEnabled(True)
                        self.statusBar().showMessage(f"Tipo de modelo alterado para {model_type}. Dados mantidos.")
                    elif hasattr(self, 'rgb_train_btn'):
                        self.rgb_train_btn.setEnabled(True)
                        self.statusBar().showMessage(f"Tipo de modelo alterado para {model_type}. Dados mantidos.")
                else:
                    # Desabilitar botões se não temos classes suficientes
                    if is_cnn and hasattr(self, 'cnn_train_btn'):
                        self.cnn_train_btn.setEnabled(False)
                    elif hasattr(self, 'rgb_train_btn'):
                        self.rgb_train_btn.setEnabled(False)
                    self.statusBar().showMessage("Selecione pelo menos 2 classes para treinar")
            else:
                # Se não conseguimos preservar os dados, informar o usuário
                self.log_widget.log(f"Tipo de modelo alterado para {model_type}. Você pode precisar processar os dados novamente.", level="info")
                self.statusBar().showMessage(f"Tipo de modelo alterado. Por favor, processe os dados novamente.")
                # Habilitar o botão apenas se temos classes selecionadas
                if hasattr(self, 'class_data') and len(self.class_data) >= 2:
                    # Habilitar o botão de treinamento apropriado
                    if is_cnn and hasattr(self, 'cnn_train_btn'):
                        self.cnn_train_btn.setEnabled(True)
                    elif hasattr(self, 'rgb_train_btn'):
                        self.rgb_train_btn.setEnabled(True)
                else:
                    # Desabilitar botões se não temos classes suficientes
                    if is_cnn and hasattr(self, 'cnn_train_btn'):
                        self.cnn_train_btn.setEnabled(False)
                    elif hasattr(self, 'rgb_train_btn'):
                        self.rgb_train_btn.setEnabled(False)
    
    def closeEvent(self, event):
        """Handle application close event to properly clean up threads."""
        # Primeiro, desativar a interface para evitar interações durante o encerramento
        self.setEnabled(False)
        
        print("Iniciando encerramento da aplicação...")
        
        # Desconectar sinais para evitar chamadas durante o encerramento
        try:
            # Bloqueio de sinais para todos os widgets
            for widget in self.findChildren(QWidget):
                widget.blockSignals(True)
        except Exception as e:
            print(f"Erro ao bloquear sinais: {str(e)}")
            
        # Encontrar e terminar quaisquer threads em execução
        print("Verificando threads em execução...")
        
        # Combinar todas as threads que conhecemos
        global active_worker_threads
        all_threads = set(self.worker_threads + active_worker_threads)
        active_threads = []
        
        for thread in all_threads:
            if thread.isRunning():
                print(f"Thread ativa encontrada: {thread.objectName()}")
                active_threads.append(thread)
                try:
                    print(f"Solicitando interrupção da thread {thread.objectName()}...")
                    thread.safe_stop()
                except Exception as e:
                    print(f"Erro ao interromper thread {thread.objectName()}: {str(e)}")
        
        # Verificar também outras threads do QThread que possam estar em execução
        for child in self.findChildren(QThread):
            if child not in all_threads and child.isRunning():
                print(f"Thread QThread não rastreada encontrada: {child.objectName()}")
                active_threads.append(child)
                try:
                    child.requestInterruption()
                except Exception as e:
                    print(f"Erro ao interromper thread extra: {str(e)}")

        # Aguardar que as threads terminem (com timeout razoável)
        if active_threads:
            print(f"Aguardando {len(active_threads)} threads terminarem...")
            for thread in active_threads:
                if not thread.wait(2000):  # Aguardar até 2 segundos por thread
                    print(f"Thread {thread.objectName()} não respondeu ao timeout")
        
        # Limpar referências para as threads
        self.worker_threads.clear()
        active_worker_threads.clear()
        
        # Limpar referências a outros objetos
        if hasattr(self, 'current_model'):
            self.current_model = None
        if hasattr(self, 'processed_data'):
            self.processed_data = None
        
        # Registrar encerramento nos logs
        if hasattr(self, 'log_widget'):
            self.log_widget.log("Aplicação encerrando")
            
        print("Encerramento concluído")
        event.accept()

    def get_nn_parameters(self, model_type=None):
        """Get neural network parameters from UI elements."""
        # Determinar tipo de modelo se não for especificado
        if model_type is None:
            model_type = "CNN" if self.cnn_btn.isChecked() else "RGB"
        
        if model_type == "RGB":
            params = {
                'layers': self.rgb_layers_spin.value(),
                'neurons': self.rgb_neurons_spin.value(),
                'epochs': self.rgb_epochs_spin.value(),
                'rgb_attributes': self.rgb_attributes if hasattr(self, 'rgb_attributes') else []
            }
            
            # Valores padrão para atributos que podem não existir
            if hasattr(self, 'rgb_learning_rate_spin'):
                params['learning_rate'] = self.rgb_learning_rate_spin.value()
            else:
                params['learning_rate'] = 0.001
                
            if hasattr(self, 'rgb_activation_combo'):
                params['activation'] = self.rgb_activation_combo.currentText().lower()
            else:
                params['activation'] = 'relu'
                
            if hasattr(self, 'rgb_optimizer_combo'):
                params['optimizer'] = self.rgb_optimizer_combo.currentText().lower()
            else:
                params['optimizer'] = 'adam'
        else:  # CNN
            # Parâmetros básicos que devem existir
            params = {
                'conv_layers': self.cnn_layers_spin.value(),
                'dense_neurons': self.cnn_neurons_spin.value(),
                'epochs': self.cnn_epochs_spin.value(),
                # Valores padrão para parâmetros avançados
                'batch_size': 32,
                'learning_rate': 0.001,
                'activation': 'relu',
                'optimizer': 'adam',
                'filters': 32
            }
            
        return params

    def update_train_test_split_label(self, value):
        """Update train/test split label when the slider value changes."""
        self.train_test_split_label.setText(f"Divisão Treino/Teste ({value}% Treino / {100-value}% Teste)")
    
    def update_cnn_train_test_split_label(self):
        train_percent = self.split_slider_cnn.value()
        test_percent = 100 - train_percent
        self.cnn_train_test_split_label.setText(f"Treino: {train_percent}% / Teste: {test_percent}%")
    
    def update_rgb_train_test_split_label(self):
        train_percent = self.split_slider_rgb.value()
        test_percent = 100 - train_percent
        self.rgb_train_test_split_label.setText(f"Treino: {train_percent}% / Teste: {test_percent}%")

    def update_cnn_config_summary(self):
        """Atualiza o resumo de configuração da CNN."""
        if not hasattr(self, "cnn_config_summary"):
            return
        
        # Obter apenas os parâmetros básicos
        layers = self.cnn_layers_spin.value()
        neurons = self.cnn_neurons_spin.value()
        epochs = self.cnn_epochs_spin.value()
        
        # Atualizar o texto do resumo
        summary = f"<b>Configuração da CNN:</b><br>"
        summary += f"• Camadas: {layers}<br>"
        summary += f"• Neurônios: {neurons}<br>"
        summary += f"• Épocas: {epochs}<br>"
        summary += f"• Divisão Treino/Teste: {self.split_slider_cnn.value()}% / {100-self.split_slider_cnn.value()}%"
        
        # Definir o texto HTML no label
        self.cnn_config_summary.setText(summary)

    def update_rgb_config_summary(self):
        """Update the RGB configuration summary in the training tab."""
        try:
            # Verificar se temos os componentes necessários
            if not hasattr(self, 'rgb_config_summary'):
                print("Aviso: rgb_config_summary não encontrado")
                return
            
            # Obter parâmetros RGB com verificação de existência
            layers = self.rgb_layers_spin.value()
            neurons = self.rgb_neurons_spin.value()
            epochs = self.rgb_epochs_spin.value()
            
            # Valores padrão para atributos que podem não existir
            learning_rate = 0.001
            activation = "relu"
            optimizer = "adam"
            
            # Substituir com valores reais se os atributos existirem
            if hasattr(self, 'rgb_learning_rate_spin'):
                learning_rate = self.rgb_learning_rate_spin.value()
            if hasattr(self, 'rgb_activation_combo'):
                activation = self.rgb_activation_combo.currentText()
            if hasattr(self, 'rgb_optimizer_combo'):
                optimizer = self.rgb_optimizer_combo.currentText()
            
            # Contar atributos RGB
            attr_count = 0
            if hasattr(self, 'rgb_attributes') and self.rgb_attributes:
                attr_count = len(self.rgb_attributes)
            else:
                # Contar seletores na interface se não tiver atributos salvos
                if hasattr(self, 'rgb_selectors_layout'):
                    attr_count = self.rgb_selectors_layout.count()
            
            # Criar resumo formatado
            summary = (
                f"<b>Tipo de Modelo:</b> RGB Feature Network<br>"
                f"<b>Características RGB:</b> {attr_count}<br>"
                f"<b>Camadas:</b> {layers}<br>"
                f"<b>Neurônios por Camada:</b> {neurons}<br>"
                f"<b>Épocas:</b> {epochs}<br>"
                f"<b>Taxa de Aprendizado:</b> {learning_rate}<br>"
                f"<b>Ativação:</b> {activation}<br>"
                f"<b>Otimizador:</b> {optimizer}<br>"
                f"<b>Divisão Treino/Teste:</b> {self.rgb_split_slider.value()}% / {100-self.rgb_split_slider.value()}%"
            )
            
            # Atualizar o label
            self.rgb_config_summary.setText(summary)
            self.rgb_config_summary.setTextFormat(Qt.RichText)
            
        except Exception as e:
            import traceback
            print(f"Erro ao atualizar resumo de configuração RGB: {str(e)}")
            traceback.print_exc()

    def on_model_button_clicked(self, button):
        """Handler for model button group clicks"""
        model_type = "CNN" if button == self.cnn_btn else "RGB"
        print(f"on_model_button_clicked: {model_type}")
        print(f"Stack atual: índice {self.model_config_stack.currentIndex()}")
        print(f"Total de widgets no stack: {self.model_config_stack.count()}")
        self.set_model_type(model_type)
        
        # Atualiza o resumo de configuração
        if hasattr(self, 'config_summary'):
            self.update_config_summary()
        
        # Se o modelo RGB foi selecionado e classes já foram carregadas, atualizar os seletores
        if model_type == "RGB" and hasattr(self, 'class_data') and self.class_data:
            self.update_rgb_selectors_for_classes()

    def update_rgb_selectors_for_classes(self):
        """Atualiza os seletores RGB baseados nas classes carregadas."""
        # Limpar todos os seletores existentes
        while self.rgb_selectors_layout.count():
            item = self.rgb_selectors_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not hasattr(self, 'class_data') or not self.class_data:
            # Não mostra nenhum seletor se não houver classes carregadas
            return
            
        # Criar um novo seletor para cada classe
        for i, class_name in enumerate(self.class_data.keys()):
            # Criar um seletor com nome da classe
            selector = RGBSelector(f"Classe: {class_name}")
            selector.name_edit.setText(f"Atributo {i+1} - {class_name}")
            self.rgb_selectors_layout.addWidget(selector)
        
        # Log de atualização
        num_classes = len(self.class_data)
        self.log_widget.log(f"Atualizados seletores RGB para {num_classes} classes")