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
        print(f"Tentando carregar modelo de: {filepath}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
            
        # Detectar tipo de arquivo baseado na extensão e conteúdo
        try:
            # Se o arquivo termina com .json, tentar carregar como modelo baseado em regras
            if filepath.lower().endswith('.json'):
                print("Detectado arquivo JSON, tentando carregar como modelo baseado em regras")
                return self.load_direct_classifier(filepath)
                
            # Se o arquivo termina com .h5, tentar carregar como modelo Keras
            elif filepath.lower().endswith('.h5'):
                print("Detectado arquivo H5, tentando carregar como modelo Keras")
                try:
                    return self.load_neural_model(filepath)
                except Exception as e:
                    print(f"Erro ao carregar como modelo neural: {str(e)}")
                    # Tentar carregar como pickle se falhar como H5
                    return self.load_pickle_model(filepath)
            
            # Tentativa de descobrir o tipo por conteúdo
            else:
                print("Formato de arquivo não reconhecido, tentando detectar pelo conteúdo")
                # Primeiro tente como JSON
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)  # Só verificando se é JSON válido
                    return self.load_direct_classifier(filepath)
                except:
                    # Se não for JSON, tente como H5
                    try:
                        return self.load_neural_model(filepath)
                    except:
                        # Último recurso: tentar como pickle
                        return self.load_pickle_model(filepath)
        except Exception as e:
            # Mensagem de erro detalhada
            print(f"Erro ao carregar modelo de {filepath}: {str(e)}")
            # Rethrow com mais informações
            raise ValueError(f"Não foi possível carregar o modelo. Formato não suportado ou arquivo corrompido: {str(e)}")
    
    def load_pickle_model(self, filepath):
        """Carrega um modelo salvo em formato pickle."""
        print(f"Tentando carregar como modelo pickle: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # Determinar o que foi carregado
            if isinstance(model_data, dict):
                # Se for um dicionário, provavelmente é um modelo completo com metadados
                print("Carregado dicionário de modelo a partir de pickle")
                if 'model' in model_data:
                    return model_data
                else:
                    # Criar estrutura esperada
                    return {
                        'model': model_data,
                        'type': 'unknown',
                        'attributes': [],
                        'class_names': []
                    }
            else:
                # Se não for um dicionário, assume-se que é apenas o modelo
                print("Carregado objeto modelo a partir de pickle")
                return {
                    'model': model_data,
                    'type': 'unknown',
                    'attributes': [],
                    'class_names': []
                }
        except Exception as e:
            print(f"Erro ao carregar modelo pickle: {str(e)}")
            raise ValueError(f"Falha ao carregar modelo pickle: {str(e)}")
    
    def load_direct_classifier(self, filepath):
        """Carrega um modelo classificador direto baseado em regras."""
        print(f"Carregando modelo baseado em regras de: {filepath}")
        try:
            # Carregar dados do modelo
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Verificar se é um modelo baseado em regras
            if model_data.get('type') != 'rgb_direct_classifier':
                print(f"Aviso: O arquivo {filepath} não tem o tipo esperado. Tipo encontrado: {model_data.get('type', 'indefinido')}")
                # Mesmo assim, vamos tentar usá-lo
                
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
                    
                def predict(self, X):
                    # Implementar previsão dummy para compatibilidade
                    print("SimpleModel.predict chamado - retornando valores fictícios")
                    return np.random.rand(X.shape[0], len(self.params.get('class_names', ['Class1', 'Class2'])))
            
            # Adicionar metadados necessários se não existirem
            if 'type' not in model_data:
                model_data['type'] = 'rgb_direct_classifier'
            if 'class_names' not in model_data or not model_data['class_names']:
                model_data['class_names'] = ['Classe 1', 'Classe 2']
            
            # Retornar informações do modelo
            result = {
                'model': SimpleModel(model_data),
                'type': 'RGB Feature Network',
                'attributes': model_data.get('attributes', []),
                'class_names': model_data.get('class_names', []),
                'feature_weights': model_data.get('feature_weights', None),
                'feature_means': model_data.get('feature_means', None),
                'use_direct_classification': True
            }
            
            print(f"Modelo baseado em regras carregado com sucesso. Classes: {result['class_names']}")
            return result
            
        except Exception as e:
            print(f"Erro ao carregar modelo baseado em regras: {str(e)}")
            raise ValueError(f"Falha ao carregar modelo JSON: {str(e)}")
    
    def load_neural_model(self, filepath):
        """Carrega um modelo neural Keras."""
        print(f"Carregando modelo neural de: {filepath}")
        try:
            # Tentativa de carregar o modelo
            try:
                # Tentar carregar com abordagem padrão
                keras_model = load_model(filepath)
            except Exception as first_error:
                print(f"Erro na primeira tentativa: {str(first_error)}")
                # Tentar com opções personalizadas
                try:
                    keras_model = load_model(filepath, compile=False)
                    print("Modelo carregado sem compilação")
                except Exception as second_error:
                    print(f"Erro na segunda tentativa: {str(second_error)}")
                    # Tentar opção com custom_objects
                    try:
                        custom_objects = {}  # Definir objetos personalizados se necessário
                        keras_model = load_model(filepath, compile=False, custom_objects=custom_objects)
                        print("Modelo carregado com objetos personalizados")
                    except Exception as third_error:
                        # Se todas as tentativas falharem, levantar a exceção original
                        raise first_error
            
            # Check for scaler
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            scaler = None
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    print(f"Scaler carregado de: {scaler_path}")
                except Exception as e:
                    print(f"Aviso: Não foi possível carregar o scaler: {str(e)}")
            
            # Load metadata if available
            metadata_path = filepath.replace('.h5', '_metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"Metadados carregados de: {metadata_path}")
                except Exception as e:
                    print(f"Aviso: Não foi possível carregar os metadados: {str(e)}")
            
            # Create model info
            model_info = {
                'model': keras_model,
                'scaler': scaler,
                'type': metadata.get('type', 'unknown'),
                'attributes': metadata.get('attributes', []),
                'class_names': metadata.get('class_names', [])
            }
            
            print(f"Modelo neural carregado com sucesso. Tipo: {model_info['type']}")
            return model_info
            
        except Exception as e:
            print(f"Erro ao carregar modelo neural: {str(e)}")
            raise ValueError(f"Falha ao carregar modelo Keras: {str(e)}")