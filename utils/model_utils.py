import os
import pickle
import tensorflow as tf
import numpy as np

class ModelUtils:
    """Utility class for model operations like saving and loading."""
    
    def __init__(self):
        pass
    
    def save_model(self, model_info, file_path):
        """
        Save a trained model and its metadata.
        
        Args:
            model_info: Dictionary with model and metadata
            file_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Extract model and metadata
        model = model_info['model']
        metadata = {k: v for k, v in model_info.items() if k != 'model'}
        
        # Save model
        if isinstance(model, tf.keras.Model):
            model.save(file_path)
            
            # Save metadata separately
            metadata_path = file_path + '.meta'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        else:
            # For non-Keras models, save everything in pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_info, f)
    
    def load_model(self, file_path):
        """
        Load a trained model and its metadata.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Dictionary with model and metadata
        """
        # Check if this is a Keras model or pickle
        if file_path.endswith('.h5'):
            try:
                # Load Keras model
                model = tf.keras.models.load_model(file_path)
                
                # Load metadata
                metadata_path = file_path + '.meta'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                else:
                    metadata = {}
                
                # Combine model and metadata
                result = {'model': model, **metadata}
                return result
            except:
                # Fall back to pickle if it's not a Keras model
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        else:
            # Assume it's a pickle file
            with open(file_path, 'rb') as f:
                return pickle.load(f)