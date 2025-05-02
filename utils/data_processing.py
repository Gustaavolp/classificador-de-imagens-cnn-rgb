import os
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

class DataProcessor(QObject):
    """Class for processing image data and extracting features."""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def process_images(self, data_dir, rgb_attributes, train_split=0.8):
        """
        Process images from data directory, extract RGB features, and split into train/test sets.
        
        Args:
            data_dir: Directory containing class subdirectories
            rgb_attributes: List of RGB range attributes to extract
            train_split: Fraction of data to use for training (default: 0.8)
            
        Returns:
            Dictionary with train/test data
        """
        self.status_updated.emit("Scanning class directories...")
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(class_dirs) < 2:
            raise ValueError("At least two class directories are required")
        
        # Prepare empty lists for data
        all_features = []
        all_labels = []
        
        # Process each class directory
        total_dirs = len(class_dirs)
        for i, class_dir in enumerate(class_dirs):
            self.status_updated.emit(f"Processing class: {class_dir}")
            self.progress_updated.emit(int((i / total_dirs) * 100))
            
            class_path = os.path.join(data_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            total_images = len(image_files)
            for j, img_file in enumerate(image_files):
                if j % max(1, int(total_images / 10)) == 0:
                    self.status_updated.emit(f"Class {class_dir}: Processing image {j+1}/{total_images}")
                
                try:
                    img_path = os.path.join(class_path, img_file)
                    features = self.extract_features(img_path, rgb_attributes)
                    all_features.append(features)
                    all_labels.append(class_dir)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        self.status_updated.emit("Creating dataset...")
        
        # Create DataFrame
        feature_columns = []
        for attr in rgb_attributes:
            feature_columns.append(f"{attr['name']}")
            
        # Convert feature lists to proper DataFrame
        df = pd.DataFrame(all_features, columns=feature_columns)
        df['class'] = all_labels
        
        # Split into train and test sets
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        for class_name in df['class'].unique():
            class_df = df[df['class'] == class_name]
            n_train = int(len(class_df) * train_split)
            
            # Garantir que temos dados suficientes para treinamento e teste
            if n_train == 0:
                n_train = 1  # pelo menos 1 para treinamento
            elif n_train == len(class_df):
                n_train = max(1, len(class_df) - 1)  # pelo menos 1 para teste
            
            train_df = pd.concat([train_df, class_df.iloc[:n_train]])
            test_df = pd.concat([test_df, class_df.iloc[n_train:]])
        
        # Shuffle datasets
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        
        # Preparar os dados para o formato correto para treinamento
        X_train = train_df.drop('class', axis=1).values
        
        # Usar as classes exclusivamente em ordem alfabética para garantir consistência
        unique_classes = sorted(df['class'].unique())
        
        # Criar dicionário para mapear classes
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        
        # Criar one-hot encoding manualmente para manter consistência nos nomes das classes
        y_train = np.zeros((len(train_df), len(unique_classes)))
        for i, cls in enumerate(train_df['class']):
            y_train[i, class_mapping[cls]] = 1
            
        X_test = test_df.drop('class', axis=1).values
        
        # Codificar os dados de teste da mesma forma
        y_test = np.zeros((len(test_df), len(unique_classes)))
        for i, cls in enumerate(test_df['class']):
            y_test[i, class_mapping[cls]] = 1
        
        # Salvar dados processados para análise posterior
        train_df.to_csv('data/train_data.csv', index=False)
        test_df.to_csv('data/test_data.csv', index=False)
        
        # Registrar informações das classes
        print(f"Classes processadas: {unique_classes}")
        print(f"Mapeamento de classes: {class_mapping}")
        
        self.status_updated.emit("Data processing complete")
        self.progress_updated.emit(100)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': unique_classes,
            'feature_names': feature_columns
        }
    
    def extract_features(self, image_path, rgb_attributes):
        """
        Extract RGB features from a single image.
        
        Args:
            image_path: Path to the image file
            rgb_attributes: List of RGB range attributes to extract
            
        Returns:
            List of extracted feature values
        """
        # Verificar se os atributos RGB são válidos
        if not rgb_attributes or len(rgb_attributes) == 0:
            raise ValueError("Nenhum atributo RGB fornecido para extração")
        
        # Load image
        print(f"Carregando imagem: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get image dimensions and basic info
        height, width, channels = image.shape
        print(f"Dimensões da imagem: {width}x{height}, {channels} canais")
        
        # Convert from BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calcular histograma de cores para diagnóstico
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Normalizar histogramas
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        
        # Calcular valores médios, máximos e mínimos para cada canal
        r_avg = np.mean(image[:,:,0])
        g_avg = np.mean(image[:,:,1])
        b_avg = np.mean(image[:,:,2])
        
        r_max = np.max(image[:,:,0])
        g_max = np.max(image[:,:,1])
        b_max = np.max(image[:,:,2])
        
        r_min = np.min(image[:,:,0])
        g_min = np.min(image[:,:,1])
        b_min = np.min(image[:,:,2])
        
        print(f"Canal R: média={r_avg:.1f}, min={r_min}, max={r_max}")
        print(f"Canal G: média={g_avg:.1f}, min={g_min}, max={g_max}")
        print(f"Canal B: média={b_avg:.1f}, min={b_min}, max={b_max}")
        
        # Extract features for each attribute
        features = []
        total_pixels = image.shape[0] * image.shape[1]
        
        for i, attr in enumerate(rgb_attributes):
            # Verificar se o atributo tem todos os campos necessários
            required_fields = ['name', 'r_min', 'r_max', 'g_min', 'g_max', 'b_min', 'b_max']
            for field in required_fields:
                if field not in attr:
                    raise ValueError(f"Atributo {i} não possui o campo obrigatório '{field}'")
            
            # Create mask for the RGB range
            lower_bound = np.array([attr['r_min'], attr['g_min'], attr['b_min']])
            upper_bound = np.array([attr['r_max'], attr['g_max'], attr['b_max']])
            
            print(f"Aplicando atributo {attr['name']}: RGB faixa [{lower_bound}] a [{upper_bound}]")
            
            # Verificar se os limites são válidos
            if np.any(lower_bound >= upper_bound):
                print(f"ALERTA: Limites inválidos para atributo {attr['name']}: [{lower_bound}] a [{upper_bound}]")
                # Corrigir automaticamente limites problemáticos
                for j in range(3):
                    if lower_bound[j] >= upper_bound[j]:
                        upper_bound[j] = min(255, lower_bound[j] + 1)
                print(f"Limites corrigidos: [{lower_bound}] a [{upper_bound}]")
            
            # Cálculo aprimorado da característica usando vários métodos
            
            # Método 1: Contagem de pixels na faixa (abordagem tradicional)
            mask = cv2.inRange(image, lower_bound, upper_bound)
            pixel_count = np.sum(mask > 0)
            percentage = pixel_count / total_pixels
            
            # Método 2: Peso baseado na distância do pixel ao centro da faixa
            # Quanto mais próximo do centro, maior o peso
            center_r = (attr['r_min'] + attr['r_max']) / 2
            center_g = (attr['g_min'] + attr['g_max']) / 2
            center_b = (attr['b_min'] + attr['b_max']) / 2
            
            # Calcular contribuição do histograma para esta faixa
            r_contribution = np.sum(hist_r[attr['r_min']:attr['r_max']+1])
            g_contribution = np.sum(hist_g[attr['g_min']:attr['g_max']+1])
            b_contribution = np.sum(hist_b[attr['b_min']:attr['b_max']+1])
            
            # Combinar canais - usar o menor valor para garantir que todos os canais contribuam
            # Isso faz com que a cor tenha que estar presente em todos os canais para pontuar alto
            rgb_combined = min(r_contribution, g_contribution, b_contribution) * 3
            
            # Ponderar mais para canais dominantes no atributo
            r_range = attr['r_max'] - attr['r_min']
            g_range = attr['g_max'] - attr['g_min']
            b_range = attr['b_max'] - attr['b_min']
            
            # Se algum canal tem um intervalo pequeno, ele é mais específico e deve ter mais peso
            r_weight = 1.0 / max(r_range, 1) if r_range > 0 else 0
            g_weight = 1.0 / max(g_range, 1) if g_range > 0 else 0
            b_weight = 1.0 / max(b_range, 1) if b_range > 0 else 0
            
            # Normalizar pesos
            total_weight = r_weight + g_weight + b_weight
            if total_weight > 0:
                r_weight /= total_weight
                g_weight /= total_weight
                b_weight /= total_weight
            else:
                r_weight = g_weight = b_weight = 1/3
            
            # Calcular característica final com todos os métodos combinados
            # Damos mais peso para a proporção simples de pixels (método 1)
            final_percentage = percentage * 0.7 + rgb_combined * 0.3
            
            # Acentuar as diferenças elevando ao quadrado - isso torna valores pequenos ainda menores
            # e valores grandes mais evidentes, facilitando a classificação
            final_percentage = final_percentage ** 0.5  # Usar raiz quadrada para suavizar diferenças
            
            # Garantir valor mínimo para evitar zeros
            final_percentage = max(final_percentage, 0.0001)
            
            print(f"Atributo {attr['name']}: {pixel_count} pixels ({percentage*100:.2f}%), valor final: {final_percentage:.4f}")
            
            # Adicionar característica
            features.append(final_percentage)
            
            # Verificar se temos pelo menos algum pixel válido
            if pixel_count == 0:
                print(f"ALERTA: Nenhum pixel encontrado para o atributo {attr['name']}!")
                
                # Usar um valor pequeno baseado em quão próxima a cor média da imagem está da faixa
                r_dist = max(0, min(r_avg - attr['r_max'], attr['r_min'] - r_avg))
                g_dist = max(0, min(g_avg - attr['g_max'], attr['g_min'] - g_avg))
                b_dist = max(0, min(b_avg - attr['b_max'], attr['b_min'] - b_avg))
                
                # Quanto menor a distância, mais próximo da faixa
                total_dist = r_dist + g_dist + b_dist
                if total_dist < 100:  # Se estiver relativamente próximo
                    fallback_value = 0.1 * (1 - total_dist/300)  # Valor entre 0 e 0.1 baseado na proximidade
                    print(f"Atribuindo valor de proximidade: {fallback_value:.4f}")
                    features[-1] = fallback_value  # Substituir o último valor adicionado
        
        # Verificar se todas as características são iguais (problema potencial)
        if len(features) > 1 and all(f == features[0] for f in features):
            print("ALERTA: Todas as características têm o mesmo valor! Isso causará problemas na classificação.")
            
            # Tentar fazer uma classificação direta baseada nos valores médios RGB da imagem
            normalized_features = []
            
            # Para cada atributo, calcular quão bem ele corresponde à cor média da imagem
            for i, attr in enumerate(rgb_attributes):
                center_r = (attr['r_min'] + attr['r_max']) / 2
                center_g = (attr['g_min'] + attr['g_max']) / 2
                center_b = (attr['b_min'] + attr['b_max']) / 2
                
                # Calcular distância da cor média ao centro do atributo
                dist = np.sqrt(((r_avg - center_r)/255)**2 + ((g_avg - center_g)/255)**2 + ((b_avg - center_b)/255)**2)
                
                # Inverter a distância: quanto maior, pior a correspondência
                match_score = max(0, 1 - dist)
                
                # Aplicar uma função exponencial para aumentar a diferença
                match_score = match_score ** 2
                
                normalized_features.append(match_score)
            
            # Normalizar para que a soma seja 1
            total = sum(normalized_features)
            if total > 0:
                normalized_features = [f/total for f in normalized_features]
            
            print(f"Características ajustadas baseadas na cor média: {normalized_features}")
            features = normalized_features
            
        # Normalizar as características para que somem 1.0
        # Isso ajuda a interpretar como probabilidades relativas
        feature_sum = sum(features)
        if feature_sum > 0:
            normalized_features = [f / feature_sum for f in features]
            print(f"Características normalizadas: {normalized_features}")
            return normalized_features
        
        print(f"Características extraídas: {features}")
        return features
    
    def extract_image_features(self, image_path, rgb_attributes):
        """
        Extract features from a single image for classification.
        
        Args:
            image_path: Path to the image file
            rgb_attributes: List of RGB range attributes
            
        Returns:
            Numpy array of features
        """
        features = self.extract_features(image_path, rgb_attributes)
        return np.array(features).reshape(1, -1)
        
    def prepare_cnn_data(self, data_dir, train_split=0.8):
        """
        Prepare data for CNN training without feature extraction.
        This simply validates the data directory and returns information
        about the classes for CNN training.
        
        Args:
            data_dir: Directory containing class subdirectories
            train_split: Fraction of data to use for training (default: 0.8)
            
        Returns:
            Dictionary with class information
        """
        self.status_updated.emit("Scanning class directories...")
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(class_dirs) < 2:
            raise ValueError("At least two class directories are required")
            
        # Count images in each class
        class_counts = {}
        total_dirs = len(class_dirs)
        total_images = 0
        
        for i, class_dir in enumerate(class_dirs):
            self.status_updated.emit(f"Scanning class: {class_dir}")
            self.progress_updated.emit(int((i / total_dirs) * 50))
            
            class_path = os.path.join(data_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            class_counts[class_dir] = len(image_files)
            total_images += len(image_files)
        
        # Simulate full progress
        self.progress_updated.emit(100)
        self.status_updated.emit("Data directory validated for CNN training")
        
        # Return dictionary with information about the data
        return {
            'data_dir': data_dir,
            'class_names': list(class_counts.keys()),
            'class_counts': class_counts,
            'total_images': total_images,
            'train_split': train_split
        }
    
    def process_rgb_data(self, data_dir, rgb_attributes, train_split=0.8):
        """
        Alias for process_images to maintain compatibility with new API.
        Process images from data directory, extract RGB features, and split into train/test sets.
        
        Args:
            data_dir: Directory containing class subdirectories
            rgb_attributes: List of RGB range attributes to extract
            train_split: Fraction of data to use for training (default: 0.8)
            
        Returns:
            Dictionary with train/test data
        """
        return self.process_images(data_dir, rgb_attributes, train_split)