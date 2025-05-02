import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

class RGBFeatureNet:
    """Rede neural para classificação baseada em características RGB."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = None
        self.input_shape = None
        self.feature_weights = None
        self.use_direct_classification = True  # Nova flag para usar classificação direta
    
    def build_model(self, input_shape, num_classes, layers=3, neurons=32, 
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
        print(f"Construindo modelo com: input_shape={input_shape}, num_classes={num_classes}, layers={layers}, neurons={neurons}")
        
        # Armazenar o formato de entrada para uso futuro
        self.input_shape = input_shape
        
        model = Sequential()
        
        # Camada de entrada - aumentar o tamanho para lidar com dados mais complexos
        model.add(Dense(neurons*2, activation=activation, input_shape=(input_shape,)))
        model.add(Dropout(0.3))  # Aumentar dropout para reduzir overfitting
        
        # Camadas ocultas - usar tamanho decrescente para criar um funil
        for i in range(layers - 1):
            layer_size = neurons * 2 // (i+1)  # Reduzir tamanho gradualmente
            layer_size = max(num_classes*2, layer_size)  # Garantir tamanho mínimo
            
            model.add(Dense(layer_size, activation=activation))
            model.add(Dropout(0.2))
        
        # Camada de saída com softmax
        model.add(Dense(num_classes, activation='softmax'))
        
        # Configurar otimizador com clipping para evitar explosão de gradientes
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate, clipnorm=1.0, momentum=0.9)
        else:
            opt = RMSprop(learning_rate=learning_rate, clipnorm=1.0)
        
        # Compilar modelo
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Resumo do modelo
        model.summary()
        
        return model
    
    def verify_attributes(self, rgb_attributes):
        """
        Verificar se os atributos RGB são válidos.
        
        Args:
            rgb_attributes: Lista de atributos RGB a verificar
            
        Returns:
            True se válidos, False caso contrário
        """
        if not rgb_attributes or len(rgb_attributes) == 0:
            print("ERRO: Nenhum atributo RGB fornecido")
            return False
            
        for i, attr in enumerate(rgb_attributes):
            # Verificar campos obrigatórios
            required_fields = ['name', 'r_min', 'r_max', 'g_min', 'g_max', 'b_min', 'b_max']
            for field in required_fields:
                if field not in attr:
                    print(f"ERRO: Atributo {i} não tem o campo obrigatório '{field}'")
                    return False
                    
            # Verificar intervalos RGB válidos
            if attr['r_min'] >= attr['r_max']:
                print(f"ERRO: Atributo {attr['name']} tem r_min >= r_max ({attr['r_min']} >= {attr['r_max']})")
                return False
            if attr['g_min'] >= attr['g_max']:
                print(f"ERRO: Atributo {attr['name']} tem g_min >= g_max ({attr['g_min']} >= {attr['g_max']})")
                return False
            if attr['b_min'] >= attr['b_max']:
                print(f"ERRO: Atributo {attr['name']} tem b_min >= b_max ({attr['b_min']} >= {attr['b_max']})")
                return False
                
        print(f"Verificação de atributos RGB: {len(rgb_attributes)} atributos válidos")
        return True
    
    def direct_classification(self, features, class_names):
        """
        Classificação direta baseada em regras, sem usar rede neural.
        
        Args:
            features: Características extraídas da imagem
            class_names: Nomes das classes
            
        Returns:
            Array de probabilidades para cada classe
        """
        print(f"Usando classificação direta com features: {features}")
        
        # Verificar se temos features e classes
        if len(features) == 0 or len(class_names) == 0:
            print("ERRO: Sem características ou classes para classificação direta")
            # Retornar probabilidades uniformes
            return np.ones(len(class_names)) / len(class_names)
        
        # Verificar se todas as features são zero
        if np.all(features == 0):
            print("ALERTA: Todas as características são zero, impossível classificar com precisão")
            # Retornar probabilidades uniformes
            return np.ones(len(class_names)) / len(class_names)
        
        # Se temos pesos específicos de features, usá-los
        if self.feature_weights is not None and len(self.feature_weights) == len(features):
            weighted_features = features * self.feature_weights
            print(f"Usando pesos de características: {self.feature_weights}")
            print(f"Características ponderadas: {weighted_features}")
        else:
            # Se não temos pesos, usar as features diretamente
            weighted_features = features
        
        # Normalizar features - garantir que somem 1.0
        if np.sum(weighted_features) > 0:
            normalized_features = weighted_features / np.sum(weighted_features)
        else:
            # Se a soma for zero, usar distribuição uniforme
            normalized_features = np.ones_like(weighted_features) / len(weighted_features)
        
        # Mapear características para classes - este é o ponto mais crítico
        num_features = len(normalized_features)
        num_classes = len(class_names)
        
        # Criar matriz de mapeamento feature -> class (inicialmente uniforme)
        feature_to_class = np.zeros((num_features, num_classes))
        
        if num_features == num_classes:
            # Caso perfeito: uma característica por classe
            for i in range(num_features):
                feature_to_class[i, i] = 1.0
            print("Mapeamento direto de características para classes (1:1)")
        else:
            # Caso com números diferentes de características e classes
            # Distribuir características uniformemente entre as classes
            for i in range(num_features):
                # Atribuir esta característica principalmente à classe correspondente
                class_idx = i % num_classes
                feature_to_class[i, class_idx] = 0.8  # 80% para a classe principal
                
                # Distribuir o resto uniformemente entre as outras classes
                other_classes = [j for j in range(num_classes) if j != class_idx]
                if other_classes:
                    remaining = 0.2
                    for j in other_classes:
                        feature_to_class[i, j] = remaining / len(other_classes)
            
            print(f"Mapeamento de {num_features} características para {num_classes} classes")
        
        # Calcular probabilidades
        # P(classe) = soma(P(característica) * P(classe|característica))
        probabilities = np.zeros(num_classes)
        
        for i in range(num_classes):
            # Para cada classe, somar a contribuição de cada característica
            class_prob = 0
            for j in range(num_features):
                class_prob += normalized_features[j] * feature_to_class[j, i]
            probabilities[i] = class_prob
        
        # Garantir que as probabilidades somem 1.0
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # Fallback para distribuição uniforme
            probabilities = np.ones(num_classes) / num_classes
        
        # Enfatizar ainda mais a classe mais provável
        max_idx = np.argmax(probabilities)
        probabilities = probabilities ** 2  # Elevar ao quadrado aumenta a diferença
        probabilities = probabilities / np.sum(probabilities)  # Renormalizar
        
        print(f"Probabilidades calculadas: {probabilities}")
        return probabilities
    
    def train(self, data, params):
        """
        Treinar a rede neural com os dados fornecidos.
        
        Args:
            data: Dicionário com X_train, y_train, X_test, y_test
            params: Dicionário com hiperparâmetros
            
        Returns:
            Dicionário com resultados do treinamento
        """
        # Verificar se devemos usar classificação direta
        self.use_direct_classification = params.get('use_direct_classification', True)
        print(f"Modo de classificação: {'Direta baseada em regras' if self.use_direct_classification else 'Rede Neural'}")
        
        # Verificar se temos dados de RGB
        if 'rgb_attributes' in params:
            rgb_attributes = params['rgb_attributes']
            if not self.verify_attributes(rgb_attributes):
                print("ALERTA: Atributos RGB inválidos, isso pode afetar o treinamento")
        
        # Debug: Verificar formato dos dados
        print("Shape dos dados de treinamento:")
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Verificar se temos features suficientes
        if X_train.shape[1] <= 1:
            print("ALERTA: Apenas uma característica disponível. Isso pode limitar o desempenho do modelo.")
            
        # Verificar valores em X_train
        print(f"X_train - min: {X_train.min()}, max: {X_train.max()}, média: {X_train.mean()}")
        
        # Verificar distribuição das classes
        print("Distribuição das classes:")
        for i in range(y_train.shape[1]):
            count = np.sum(y_train[:, i])
            print(f"Classe {i}: {count} exemplos ({count/len(y_train)*100:.1f}%)")
        
        # Armazenar os nomes das classes originais
        if 'class_names' in data:
            self.class_names = data['class_names']
            print(f"Classes para treinamento RGB: {self.class_names}")
        else:
            # Criar nomes genéricos se nenhum nome for fornecido
            self.class_names = [f"Classe {i+1}" for i in range(y_train.shape[1])]
            print(f"Nomes de classes não fornecidos, usando valores genéricos: {self.class_names}")
        
        # Se estivermos usando classificação direta, não precisamos treinar uma rede
        if self.use_direct_classification:
            print("Usando classificação direta baseada em regras, pulando treinamento de rede neural")
            
            # Calcular médias das características por classe para usar como referência
            feature_means = []
            for i in range(y_train.shape[1]):  # Para cada classe
                # Selecionar exemplos desta classe
                class_indices = np.argmax(y_train, axis=1) == i
                if np.any(class_indices):
                    # Calcular média das características para esta classe
                    class_features = X_train[class_indices]
                    class_mean = np.mean(class_features, axis=0)
                    feature_means.append(class_mean)
                else:
                    # Se não temos exemplos desta classe, usar zeros
                    feature_means.append(np.zeros(X_train.shape[1]))
            
            # Criar matriz de características por classe
            self.feature_means = np.array(feature_means)
            print(f"Médias de características por classe:\n{self.feature_means}")
            
            # Criar pesos de características baseados na variância entre classes
            # Características com maior variância entre classes são mais discriminativas
            feature_vars = np.var(self.feature_means, axis=0)
            self.feature_weights = feature_vars / np.sum(feature_vars) if np.sum(feature_vars) > 0 else np.ones(X_train.shape[1])
            print(f"Pesos das características: {self.feature_weights}")
            
            # Avaliar com o conjunto de teste
            y_pred = []
            for i in range(len(X_test)):
                probs = self.direct_classification(X_test[i], self.class_names)
                y_pred.append(probs)
            y_pred = np.array(y_pred)
            
            # Calcular acurácia
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(y_pred_classes == y_true_classes)
            
            # Criar matriz de confusão
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            print(f"Matriz de confusão:\n{cm}")
            
            # Criar um pseudo-histórico para compatibilidade
            history = {
                'accuracy': [accuracy],
                'val_accuracy': [accuracy],
                'loss': [0.0],
                'val_loss': [0.0]
            }
            
            return {
                'model': None,  # Não temos modelo de rede neural
                'history': history,
                'accuracy': accuracy,
                'loss': 0.0,
                'scaler': None,  # Não usamos scaler
                'class_names': self.class_names,
                'confusion_matrix': cm,
                'input_shape': X_train.shape[1],
                'feature_weights': self.feature_weights,
                'feature_means': self.feature_means,
                'use_direct_classification': True
            }
        
        # Código para treinar rede neural, caso use_direct_classification = False
        # ... Resto do código original para treinamento de rede neural ...
        # Escalonar características usando scaler
        print("Escalonando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Debug: verificar valores após escalonamento
        print(f"X_train_scaled - min: {X_train_scaled.min()}, max: {X_train_scaled.max()}, média: {X_train_scaled.mean()}")
        
        # Obter hiperparâmetros
        layers = params.get('layers', 3)
        neurons = params.get('neurons', 32)
        activation = params.get('activation', 'relu')
        learning_rate = params.get('learning_rate', 0.001)
        optimizer = params.get('optimizer', 'adam')
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 16)
        
        # Ajustar hiperparâmetros baseado no tamanho dos dados
        if len(X_train) < 50:
            print("Conjunto de dados pequeno, ajustando hiperparâmetros...")
            batch_size = min(batch_size, max(4, len(X_train) // 5))
            print(f"Batch size ajustado para {batch_size}")
        
        # Construir modelo
        self.model = self.build_model(
            input_shape=X_train.shape[1],
            num_classes=y_train.shape[1],
            layers=layers,
            neurons=neurons,
            activation=activation,
            learning_rate=learning_rate,
            optimizer=optimizer
        )
        
        # Configurar early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Treinar modelo
        print(f"Iniciando treinamento com {epochs} épocas, batch_size={batch_size}")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Avaliar modelo
        print("Avaliando modelo no conjunto de teste...")
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Acurácia final: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Verificar predições no conjunto de teste
        print("Fazendo predições no conjunto de teste...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Verificar valores das predições
        print(f"Predições - min: {y_pred.min()}, max: {y_pred.max()}, média: {y_pred.mean()}")
        
        # Verificar se temos uma distribuição razoável nas predições (não apenas um valor constante)
        pred_stds = np.std(y_pred, axis=0)
        print(f"Desvio padrão por classe: {pred_stds}")
        if np.all(pred_stds < 0.001):
            print("ALERTA: Baixa variação nas predições. O modelo pode não estar aprendendo adequadamente.")
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("Verificando distribuição das previsões:")
        unique, counts = np.unique(y_pred_classes, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Classe {cls}"
            print(f"{class_name}: {count} predições ({count/len(y_pred_classes)*100:.1f}%)")
        
        # Gerar matriz de confusão
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print("Matriz de confusão:")
        print(cm)
        
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
            'loss': loss,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'confusion_matrix': cm,
            'input_shape': X_train.shape[1],
            'use_direct_classification': False
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
        
        print(f"Classificando imagem: {image_path}")
        
        # Verificar se os atributos RGB são válidos
        if not self.verify_attributes(rgb_attributes):
            print("ALERTA: Atributos RGB inválidos, isso pode afetar a classificação")
        
        # Comparar com o formato esperado pelo modelo
        if hasattr(self, 'input_shape') and self.input_shape is not None:
            if len(rgb_attributes) != self.input_shape:
                print(f"ALERTA: Número de atributos RGB ({len(rgb_attributes)}) não corresponde ao formato de entrada do modelo ({self.input_shape})")
        
        # Extrair características
        print("Extraindo características...")
        data_processor = DataProcessor()
        features = data_processor.extract_image_features(image_path, rgb_attributes)
        features = features.flatten()  # Garantir que seja 1D
        
        print(f"Características extraídas: {features}")
        
        # Verificar se estamos usando classificação direta ou rede neural
        if self.use_direct_classification:
            # Classificação direta baseada em regras
            prediction = self.direct_classification(features, self.class_names)
        else:
            # Classificação com rede neural
            # Escalonar características usando o mesmo scaler do treinamento
            if hasattr(self, 'scaler') and self.scaler is not None:
                print("Escalonando características...")
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                print(f"Características escalonadas: {features_scaled}")
            else:
                print("ALERTA: Não foi encontrado um scaler, usando características não escalonadas")
                features_scaled = features.reshape(1, -1)
            
            # Fazer predição
            print("Realizando predição...")
            prediction = model.predict(features_scaled)[0]
        
        print(f"Valores de predição: {prediction}")
        print(f"Soma das predições: {np.sum(prediction)}")
        
        # Criar dicionário de resultado com todas as probabilidades
        predicted_class_idx = np.argmax(prediction)
        
        # Verificar se o índice está dentro dos limites
        if predicted_class_idx >= len(self.class_names):
            print(f"ERRO: Índice de classe prevista ({predicted_class_idx}) fora dos limites dos nomes de classe ({len(self.class_names)})")
            predicted_class = f"Classe {predicted_class_idx+1}"
        else:
            predicted_class = self.class_names[predicted_class_idx]
        
        # Criar dicionário de probabilidades
        probabilities = {}
        for i, prob in enumerate(prediction):
            if i < len(self.class_names):
                probabilities[self.class_names[i]] = float(prob)
            else:
                probabilities[f"Classe {i+1}"] = float(prob)
        
        print(f"Classe prevista: {predicted_class} (índice {predicted_class_idx})")
        print(f"Probabilidades: {probabilities}")
        
        return {
            'class': predicted_class,
            'probabilities': probabilities,
            'raw_prediction': prediction
        }