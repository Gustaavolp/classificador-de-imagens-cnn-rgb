import sys
import os
import signal
import traceback
import time
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from ui.main_window import MainWindow

def exception_hook(exctype, value, tb):
    """Capturar exceções não tratadas e mostrá-las em uma mensagem de erro."""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f"ERRO NÃO TRATADO: {error_msg}")
    app = QApplication.instance()
    if app:
        QMessageBox.critical(None, "Erro Crítico", 
                           f"Ocorreu um erro não tratado:\n\n{str(value)}\n\nDetalhes completos foram impressos no console.")
    sys.exit(1)

def main():
    """Main application entry point."""
    try:
        # Configurar hook para exceções não tratadas
        sys.excepthook = exception_hook
        
        # Create application directory structure if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Create application instance
        app = QApplication(sys.argv)
        app.setApplicationName("Neural Network Feature Extractor")
        
        # Set stylesheet (optional - for better styling)
        # with open("ui/style.qss", "r") as f:
        #     app.setStyleSheet(f.read())
        
        # Create and show the main window
        window = MainWindow()
        window.show()
        
        # Adicionar temporizador para manter a aplicação aberta por pelo menos 30 segundos para testes
        keepalive_timer = QTimer()
        keepalive_timer.timeout.connect(lambda: print("Aplicação em execução..."))
        keepalive_timer.start(5000)  # A cada 5 segundos
        
        # Set up signal handling for clean shutdown
        def signal_handler(signum, frame):
            print(f"Sinal {signum} recebido, encerrando aplicação...")
            keepalive_timer.stop()
            window.close()
            QTimer.singleShot(1000, app.quit)  # Dar um tempo para o encerramento
        
        # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("Aplicação iniciada e em execução...")
        
        # Start the application event loop
        return app.exec()
    except Exception as e:
        print(f"ERRO NA INICIALIZAÇÃO: {str(e)}")
        traceback.print_exc()  # Imprimir stacktrace detalhado
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Erro ao Iniciar Aplicação", 
                           f"Ocorreu um erro ao iniciar a aplicação:\n\n{str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())