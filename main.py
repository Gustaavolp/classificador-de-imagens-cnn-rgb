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
    """Ponto de entrada principal da aplicação."""
    try:
        # Configurar hook para exceções não tratadas
        sys.excepthook = exception_hook
        
        # Criar estrutura de diretórios da aplicação se não existir
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Criar instância da aplicação
        app = QApplication(sys.argv)
        app.setApplicationName("Classificador de Imagens CNN/RGB")
        
        # Definir folha de estilo (opcional - para melhor estilização)
        # with open("ui/style.qss", "r") as f:
        #     app.setStyleSheet(f.read())
        
        # Criar e mostrar a janela principal
        window = MainWindow()
        window.show()
        
        # Adicionar temporizador para manter a aplicação aberta por pelo menos 30 segundos para testes
        keepalive_timer = QTimer()
        keepalive_timer.timeout.connect(lambda: print("Aplicação em execução..."))
        keepalive_timer.start(5000)  # A cada 5 segundos
        
        # Configurar manipulação de sinais para encerramento adequado
        def signal_handler(signum, frame):
            print(f"Sinal {signum} recebido, encerrando aplicação...")
            keepalive_timer.stop()
            window.close()
            QTimer.singleShot(1000, app.quit)  # Dar um tempo para o encerramento
        
        # Registrar manipuladores de sinal para SIGINT (Ctrl+C) e SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("Aplicação iniciada e em execução...")
        
        # Iniciar o loop de eventos da aplicação
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