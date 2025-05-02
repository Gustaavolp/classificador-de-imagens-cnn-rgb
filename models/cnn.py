import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix
from PIL import Image

class ConvolutionalNetwork:
    """Rede Neural Convolucional para classificação de imagens."""
    
    def __init__(self):
        self.model = None
        self.class_names = None
        self.img_width = 64
        self.img_height = 64
    
    def build_model(self, num_classes, params=None):
        """
        Construir e compilar o modelo CNN.
        
        Args:
            num_classes: Número de classes de saída
            params: Dicionário com hiperparâmetros opcionais
            
        Returns:
            Modelo Keras compilado
        """
        if params is None:
            params = {}
            
        # Obter parâmetros com valores padrão
        layers = params.get('layers', 4)
        neurons = params.get('neurons', 64)
        activation = params.get('activation', 'relu')
        learning_rate = params.get('learning_rate', 0.001)
        optimizer_name = params.get('optimizer', 'adam')
        img_width = params.get('img_width', self.img_width)
        img_height = params.get('img_height', self.img_height)
        
        # Definir variáveis de instância
        self.img_width = img_width
        self.img_height = img_height
        
        # Criar modelo
        model = Sequential()
        
        # Primeira camada convolucional
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', 
                         input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Segunda camada convolucional
        model.add(Conv2D(64, (3, 3), activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Terceira camada convolucional (opcional com base na profundidade do modelo)
        if layers > 3:
            model.add(Conv2D(128, (3, 3), activation=activation, padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Camada de achatamento (Flatten)
        model.add(Flatten())
        
        # Camadas densas
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(0.5))
        
        # Adicionar mais camadas densas com base no parâmetro
        for _ in range(layers - 3):
            model.add(Dense(neurons // 2, activation=activation))
            model.add(Dropout(0.3))
        
        # Camada de saída
        model.add(Dense(num_classes, activation='softmax'))
        
        # Configurar otimizador
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        else:  # rmsprop
            optimizer = RMSprop(learning_rate=learning_rate)
            
        # Compilar modelo
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_dir, train_split=0.8, params=None):
        """
        Treinar a CNN no diretório de dados fornecido.
        
        Args:
            data_dir: Diretório contendo subdiretórios de classes com imagens
            train_split: Proporção de dados a usar para treinamento (0.0 a 1.0)
            params: Dicionário com hiperparâmetros
            
        Returns:
            Dicionário com resultados do treinamento
        """
        if params is None:
            params = {}
            
        # Obter parâmetros com valores padrão
        batch_size = params.get('batch_size', 8)
        epochs = params.get('epochs', 200)
        img_width = params.get('img_width', self.img_width)
        img_height = params.get('img_height', self.img_height)
        
        # Atualizar variáveis de instância
        self.img_width = img_width
        self.img_height = img_height
        
        # Aumento de dados para conjunto de treinamento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=1-train_split  # Definir divisão de validação
        )
        
        # Gerador para dados de treinamento
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Gerador para dados de validação
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Armazenar nomes das classes
        self.class_names = list(train_generator.class_indices.keys())
        
        # Construir modelo
        num_classes = len(self.class_names)
        self.model = self.build_model(num_classes, params)
        
        # Treinar modelo
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        # Avaliar no conjunto de validação
        validation_generator.reset()
        y_pred = []
        y_true = []
        
        # Predizer em lotes
        for i in range(validation_generator.samples // batch_size + 1):
            try:
                x, y = next(validation_generator)
                pred = self.model.predict(x)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(np.argmax(y, axis=1))
            except StopIteration:
                break
                
        # Recortar para o tamanho real da validação
        y_pred = y_pred[:validation_generator.samples]
        y_true = y_true[:validation_generator.samples]
        
        # Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Retornar resultados
        return {
            'model': self.model,
            'history': history.history,
            'class_names': self.class_names,
            'confusion_matrix': cm,
            'accuracy': history.history['val_accuracy'][-1] if history.history['val_accuracy'] else 0
        }
    
    def classify_image(self, image_path, model):
        """
        Classificar uma única imagem usando o modelo treinado.
        
        Args:
            image_path: Caminho para a imagem
            model: Modelo treinado
            
        Returns:
            Dicionário com resultados da classificação
        """
        # Carregar e preparar a imagem
        img_width, img_height = self.img_width, self.img_height
        
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((img_width, img_height))
            img_array = np.array(img) / 255.0  # Normalizar para [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de lote
            
            # Fazer a predição
            predictions = model.predict(img_array)
            
            # Encontrar classe predita
            predicted_class_index = np.argmax(predictions[0])
            
            # Obter nomes das classes
            if hasattr(self, 'class_names') and self.class_names:
                class_names = self.class_names
            else:
                # Se os nomes das classes não estiverem disponíveis, usar índices
                num_classes = predictions.shape[1]
                class_names = [f"Classe {i}" for i in range(num_classes)]
            
            # Criar dicionário de probabilidades
            probabilities = {}
            for i, prob in enumerate(predictions[0]):
                class_name = class_names[i] if i < len(class_names) else f"Classe {i}"
                probabilities[class_name] = float(prob)
            
            # Classe predita
            predicted_class = class_names[predicted_class_index] if predicted_class_index < len(class_names) else f"Classe {predicted_class_index}"
            
            # Retornar resultados
            return {
                'class': predicted_class,
                'probabilities': probabilities,
                'raw_prediction': predictions[0].tolist()
            }
        except Exception as e:
            print(f"Erro ao classificar imagem: {str(e)}")
            return {
                'class': 'Erro',
                'probabilities': {},
                'error': str(e)
            }