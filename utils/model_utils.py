import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

class ModelUtils:
    """Utilities for saving and loading models."""
    
    def save_model(self, model_info, filepath):
        """
        Save model and related information.
        
        Args:
            model_info: Dictionary with model, scaler, and other info
            filepath: Path to save the model
        """
        # Create the directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine if it's a direct classifier or regular NN model
        if model_info.get('type') == 'rgb_direct_classifier' or hasattr(model_info.get('model', None), 'params'):
            # É um modelo baseado em regras
            print("Salvando modelo baseado em regras")
            self.save_direct_classifier(model_info, filepath)
        else:
            # É um modelo keras normal
            print("Salvando modelo neural Keras")
            self.save_neural_model(model_info, filepath)
    
    def save_direct_classifier(self, model_info, filepath):
        """Salva um modelo classificador direto baseado em regras."""
        # Extrair modelo (que já contém os parâmetros) ou criar estrutura de dados
        if 'model' in model_info and hasattr(model_info['model'], 'params'):
            model_data = model_info['model'].params
        else:
            # Criar estrutura para salvar
            model_data = {
                'type': 'rgb_direct_classifier',
                'class_names': model_info.get('class_names', []),
                'feature_weights': model_info.get('feature_weights', None),
                'feature_means': model_info.get('feature_means', None),
                'input_shape': model_info.get('input_shape', None),
                'use_direct_classification': True
            }
            
            # Converter arrays numpy para listas
            if isinstance(model_data['feature_weights'], np.ndarray):
                model_data['feature_weights'] = model_data['feature_weights'].tolist()
            if isinstance(model_data['feature_means'], np.ndarray):
                model_data['feature_means'] = model_data['feature_means'].tolist()
        
        # Adicionar atributos RGB se estiverem disponíveis
        if 'attributes' in model_info:
            model_data['attributes'] = model_info['attributes']
            
        # Salvar como JSON
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Modelo baseado em regras salvo em: {filepath}")
    
    def save_neural_model(self, model_info, filepath):
        """Salva um modelo neural Keras."""
        # Extract required components
        keras_model = model_info['model']
        
        # Save Keras model
        keras_model.save(filepath)
        
        # Check if we have a scaler to save
        if 'scaler' in model_info and model_info['scaler'] is not None:
            # We can't save the scaler in the h5 file directly, so save it separately
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(model_info['scaler'], f)
        
        # Save additional information like attributes, type, etc.
        metadata = {
            'type': model_info.get('type', 'unknown'),
            'attributes': model_info.get('attributes', []),
            'class_names': model_info.get('class_names', [])
        }
        
        # Save metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Modelo neural salvo em: {filepath}")
        print(f"Metadados salvos em: {metadata_path}")
    
    def load_model(self, filepath):
        """
        Load model and related information.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Dictionary with model, scaler, and other info
        """
        # Check if it's a JSON file (direct classifier) or h5 file (neural model)
        if filepath.lower().endswith('.json'):
            return self.load_direct_classifier(filepath)
        else:
            return self.load_neural_model(filepath)
    
    def load_direct_classifier(self, filepath):
        """Carrega um modelo classificador direto baseado em regras."""
        # Carregar dados do modelo
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Verificar se é um modelo baseado em regras
        if model_data.get('type') != 'rgb_direct_classifier':
            raise ValueError(f"O arquivo {filepath} não contém um modelo classificador direto válido")
        
        # Converter listas para arrays numpy
        if 'feature_weights' in model_data and model_data['feature_weights'] is not None:
            model_data['feature_weights'] = np.array(model_data['feature_weights'])
        if 'feature_means' in model_data and model_data['feature_means'] is not None:
            model_data['feature_means'] = np.array(model_data['feature_means'])
        
        # Criar um objeto SimpleModel para compatibilidade com o resto do código
        class SimpleModel:
            def __init__(self, params):
                self.params = params
                
            def to_json(self):
                return json.dumps(self.params)
        
        # Retornar informações do modelo
        return {
            'model': SimpleModel(model_data),
            'type': 'RGB Feature Network',
            'attributes': model_data.get('attributes', []),
            'class_names': model_data.get('class_names', []),
            'feature_weights': model_data.get('feature_weights', None),
            'feature_means': model_data.get('feature_means', None),
            'use_direct_classification': True
        }
    
    def load_neural_model(self, filepath):
        """Carrega um modelo neural Keras."""
        # Load Keras model
        keras_model = load_model(filepath)
        
        # Check for scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Create model info
        model_info = {
            'model': keras_model,
            'scaler': scaler,
            'type': metadata.get('type', 'unknown'),
            'attributes': metadata.get('attributes', []),
            'class_names': metadata.get('class_names', [])
        }
        
        return model_info