import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class RGBFeatureNet:
    """Neural network for RGB feature-based classification."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = None
    
    def build_model(self, input_shape, num_classes, layers=3, neurons=16, 
                   activation='relu', learning_rate=0.001, optimizer='adam'):
        """
        Build the neural network model.
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            layers: Number of hidden layers
            neurons: Number of neurons per hidden layer
            activation: Activation function to use
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer to use ('adam', 'sgd', or 'rmsprop')
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(neurons, activation=activation, input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        
        # Hidden layers
        for _ in range(layers - 1):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        # Configure optimizer
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = RMSprop(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data, params):
        """
        Train the neural network on the provided data.
        
        Args:
            data: Dictionary with X_train, y_train, X_test, y_test
            params: Dictionary with hyperparameters
            
        Returns:
            Dictionary with training results
        """
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        self.class_names = data['class_names']
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(
            input_shape=X_train.shape[1],
            num_classes=y_train.shape[1],
            layers=params.get('layers', 3),
            neurons=params.get('neurons', 16),
            activation=params.get('activation', 'relu'),
            learning_rate=params.get('learning_rate', 0.001),
            optimizer=params.get('optimizer', 'adam')
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=params.get('epochs', 100),
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Generate confusion matrix
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
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
        Classify a single image using the trained model.
        
        Args:
            image_path: Path to the image
            model: Trained Keras model
            rgb_attributes: List of RGB attributes
            
        Returns:
            Dictionary with classification results
        """
        from utils.data_processing import DataProcessor
        
        # Extract features
        data_processor = DataProcessor()
        features = data_processor.extract_image_features(image_path, rgb_attributes)
        
        # Scale features
        if hasattr(self, 'scaler') and self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get class names if not already stored
        if self.class_names is None and hasattr(model, 'output_names'):
            self.class_names = model.output_names
        
        # Create result dictionary
        if self.class_names:
            predicted_class = self.class_names[np.argmax(prediction)]
            probabilities = {cls: float(prob) for cls, prob in zip(self.class_names, prediction)}
        else:
            predicted_class = f"Class {np.argmax(prediction)}"
            probabilities = {f"Class {i}": float(prob) for i, prob in enumerate(prediction)}
        
        return {
            'class': predicted_class,
            'probabilities': probabilities,
            'raw_prediction': prediction
        }