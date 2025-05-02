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
        
        # Verificar se temos dados suficientes
        print(f"Dimensões dos dados - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Dimensões das labels - y_train: {y_train.shape}, y_test: {y_test.shape}")
        
        # Verificar valores extremos ou NaN
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            print("AVISO: Encontrados valores NaN nos dados!")
            # Substituir NaN por zeros
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)
        
        # Verificar range dos dados
        print(f"Range de valores em X_train: Min={X_train.min()}, Max={X_train.max()}")
        
        # Escalonar características com mais robustez
        try:
            # Tentar ajustar o scaler aos dados
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            print("Dados escalados com sucesso")
        except Exception as e:
            print(f"Erro ao escalar dados: {e}")
            # Fallback para escalonamento manual se StandardScaler falhar
            if X_train.shape[1] > 0:  # Verificar se temos características
                # Escalonar manualmente para [0,1]
                x_min = X_train.min(axis=0)
                x_max = X_train.max(axis=0)
                
                # Evitar divisão por zero
                range_values = x_max - x_min
                range_values[range_values == 0] = 1.0
                
                X_train = (X_train - x_min) / range_values
                X_test = np.clip((X_test - x_min) / range_values, 0, 1)
                print("Dados escalados manualmente")
        
        # Construir modelo com regularização para evitar overfitting
        neurons = params.get('neurons', 16)
        activation = params.get('activation', 'relu')
        
        self.model = Sequential()
        
        # Camada de entrada com regularização
        self.model.add(Dense(neurons, 
                             activation=activation, 
                             input_shape=(X_train.shape[1],),
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        self.model.add(Dropout(0.3))  # Aumento do dropout para reduzir overfitting
        
        # Camadas ocultas
        for _ in range(params.get('layers', 3) - 1):
            self.model.add(Dense(neurons, 
                                activation=activation,
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            self.model.add(Dropout(0.3))
        
        # Camada de saída
        self.model.add(Dense(y_train.shape[1], activation='softmax'))
        
        # Configurar otimizador com menor learning rate para maior estabilidade
        learning_rate = params.get('learning_rate', 0.001) * 0.5  # Reduzir para maior estabilidade
        
        if params.get('optimizer', 'adam').lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif params.get('optimizer', 'adam').lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)  # Adicionar momentum
        else:
            opt = RMSprop(learning_rate=learning_rate, rho=0.9)
        
        # Compilar modelo
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Resumo do modelo
        self.model.summary()
        
        # Early stopping para evitar overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduzir batch size para maior estabilidade
        batch_size = min(32, X_train.shape[0] // 4) if X_train.shape[0] > 4 else 2
        print(f"Usando batch size: {batch_size}")
        
        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=params.get('epochs', 100),
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop]
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
        
        # Mostrar matriz de confusão para depuração
        print("\nMatriz de Confusão:")
        print(cm)
        print("Nomes das classes:", self.class_names)
        
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