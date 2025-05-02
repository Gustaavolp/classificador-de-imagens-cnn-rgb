import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class RGBFeatureNet:
    """Rede neural para classificação baseada em características RGB."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = None
    
    def build_model(self, input_shape, num_classes, layers=3, neurons=16, 
                   activation='relu', learning_rate=0.001, optimizer='adam'):
        """
        Construir o modelo de rede neural.
        
        Args:
            input_shape: Formato dos atributos de entrada
            num_classes: Número de classes de saída
            layers: Número de camadas ocultas
            neurons: Número de neurônios por camada oculta
            activation: Função de ativação a ser usada
            learning_rate: Taxa de aprendizado para o otimizador
            optimizer: Otimizador a ser usado ('adam', 'sgd', ou 'rmsprop')
            
        Returns:
            Modelo Keras compilado
        """
        model = Sequential()
        
        # Camada de entrada
        model.add(Dense(neurons, activation=activation, input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        
        # Camadas ocultas
        for _ in range(layers - 1):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(0.2))
        
        # Camada de saída
        model.add(Dense(num_classes, activation='softmax'))
        
        # Configurar otimizador
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = RMSprop(learning_rate=learning_rate)
        
        # Compilar modelo
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data, params):
        """
        Treinar a rede neural com os dados fornecidos.
        
        Args:
            data: Dicionário com X_train, y_train, X_test, y_test
            params: Dicionário com hiperparâmetros
            
        Returns:
            Dicionário com resultados do treinamento
        """
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Armazenar os nomes das classes originais
        if 'class_names' in data:
            self.class_names = data['class_names']
            print(f"Classes para treinamento RGB: {self.class_names}")
        else:
            # Criar nomes genéricos se nenhum nome for fornecido
            self.class_names = [f"Classe {i+1}" for i in range(y_train.shape[1])]
            print(f"Nomes de classes não fornecidos, usando valores genéricos: {self.class_names}")
        
        # Escalonar características
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Construir modelo
        self.model = self.build_model(
            input_shape=X_train.shape[1],
            num_classes=y_train.shape[1],
            layers=params.get('layers', 3),
            neurons=params.get('neurons', 16),
            activation=params.get('activation', 'relu'),
            learning_rate=params.get('learning_rate', 0.001),
            optimizer=params.get('optimizer', 'adam')
        )
        
        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=params.get('epochs', 100),
            batch_size=32,
            verbose=1
        )
        
        # Avaliar modelo
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Gerar matriz de confusão
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Verificar se a matriz de confusão tem a dimensão correta
        if cm.shape[0] != len(self.class_names):
            print(f"ALERTA: Dimensão da matriz de confusão ({cm.shape[0]}) não corresponde ao número de classes ({len(self.class_names)})")
            # Ajustar nomes de classes se necessário
            adjusted_class_names = [f"Classe {i+1}" for i in range(cm.shape[0])]
            print(f"Ajustando nomes de classes para: {adjusted_class_names}")
            self.class_names = adjusted_class_names
        
        return {
            'model': self.model,
            'history': history.history,
            'accuracy': accuracy,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'confusion_matrix': cm
        }
    
    def classify_image(self, image_path, model, rgb_attributes):
        """
        Classificar uma única imagem usando o modelo treinado.
        
        Args:
            image_path: Caminho para a imagem
            model: Modelo Keras treinado
            rgb_attributes: Lista de atributos RGB
            
        Returns:
            Dicionário com resultados da classificação
        """
        from utils.data_processing import DataProcessor
        
        # Extrair características
        data_processor = DataProcessor()
        features = data_processor.extract_image_features(image_path, rgb_attributes)
        
        # Escalonar características
        if hasattr(self, 'scaler') and self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Fazer predição
        prediction = model.predict(features)[0]
        
        # Obter nomes das classes se não estiverem já armazenados
        if self.class_names is None and hasattr(model, 'output_names'):
            self.class_names = model.output_names
        
        # Criar dicionário de resultado
        if self.class_names:
            predicted_class = self.class_names[np.argmax(prediction)]
            probabilities = {cls: float(prob) for cls, prob in zip(self.class_names, prediction)}
        else:
            predicted_class = f"Classe {np.argmax(prediction)}"
            probabilities = {f"Classe {i}": float(prob) for i, prob in enumerate(prediction)}
        
        return {
            'class': predicted_class,
            'probabilities': probabilities,
            'raw_prediction': prediction
        }