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
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert from BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features for each attribute
        features = []
        
        for attr in rgb_attributes:
            # Create mask for the RGB range
            lower_bound = np.array([attr['r_min'], attr['g_min'], attr['b_min']])
            upper_bound = np.array([attr['r_max'], attr['g_max'], attr['b_max']])
            
            mask = cv2.inRange(image, lower_bound, upper_bound)
            
            # Calculate percentage of pixels in the range
            pixel_count = np.sum(mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            percentage = pixel_count / total_pixels
            
            features.append(percentage)
        
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