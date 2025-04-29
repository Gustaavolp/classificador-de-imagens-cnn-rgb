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
    """Convolutional Neural Network for image classification."""
    
    def __init__(self):
        self.model = None
        self.class_names = None
        self.img_width = 64
        self.img_height = 64
    
    def build_model(self, num_classes, params=None):
        """
        Build and compile the CNN model.
        
        Args:
            num_classes: Number of output classes
            params: Dictionary with optional hyperparameters
            
        Returns:
            Compiled Keras model
        """
        if params is None:
            params = {}
            
        # Get parameters with defaults
        layers = params.get('layers', 4)
        neurons = params.get('neurons', 64)
        activation = params.get('activation', 'relu')
        learning_rate = params.get('learning_rate', 0.001)
        optimizer_name = params.get('optimizer', 'adam')
        img_width = params.get('img_width', self.img_width)
        img_height = params.get('img_height', self.img_height)
        
        # Set instance variables
        self.img_width = img_width
        self.img_height = img_height
        
        # Create model
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', 
                         input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Second convolutional layer
        model.add(Conv2D(64, (3, 3), activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Third convolutional layer (optional based on model depth)
        if layers > 3:
            model.add(Conv2D(128, (3, 3), activation=activation, padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten layer
        model.add(Flatten())
        
        # Dense layers
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(0.5))
        
        # Add more dense layers based on parameter
        for _ in range(layers - 3):
            model.add(Dense(neurons // 2, activation=activation))
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        # Configure optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        else:  # rmsprop
            optimizer = RMSprop(learning_rate=learning_rate)
            
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_dir, train_split=0.8, params=None):
        """
        Train the CNN on the provided data directory.
        
        Args:
            data_dir: Directory containing class subdirectories with images
            train_split: Proportion of data to use for training (0.0 to 1.0)
            params: Dictionary with hyperparameters
            
        Returns:
            Dictionary with training results
        """
        if params is None:
            params = {}
            
        # Get parameters with defaults
        batch_size = params.get('batch_size', 8)
        epochs = params.get('epochs', 200)
        img_width = params.get('img_width', self.img_width)
        img_height = params.get('img_height', self.img_height)
        
        # Update instance variables
        self.img_width = img_width
        self.img_height = img_height
        
        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=1-train_split  # Set validation split
        )
        
        # Generator for training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Generator for validation data
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        
        # Build model
        num_classes = len(self.class_names)
        self.model = self.build_model(num_classes, params)
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        # Evaluate on validation set
        validation_generator.reset()
        y_pred = []
        y_true = []
        
        # Predict on batches
        for i in range(validation_generator.samples // batch_size + 1):
            try:
                x, y = next(validation_generator)
                pred = self.model.predict(x)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(np.argmax(y, axis=1))
            except StopIteration:
                break
                
        # Trim to actual validation size
        y_pred = y_pred[:validation_generator.samples]
        y_true = y_true[:validation_generator.samples]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Return results
        return {
            'model': self.model,
            'history': history.history,
            'class_names': self.class_names,
            'confusion_matrix': cm,
            'accuracy': history.history['val_accuracy'][-1] if history.history['val_accuracy'] else 0
        }
    
    def classify_image(self, image_path, model):
        """
        Classify a single image using the trained model.
        
        Args:
            image_path: Path to the image
            model: Trained Keras model
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Load and preprocess the image
            img = Image.open(image_path)
            img = img.resize((self.img_width, self.img_height))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            
            # Get class names if not already stored
            if not self.class_names:
                if hasattr(model, 'output_names'):
                    self.class_names = model.output_names
                else:
                    self.class_names = [f"Class {i}" for i in range(predictions.shape[1])]
            
            # Create result dictionary
            predicted_class = self.class_names[predicted_class_idx]
            probabilities = {cls: float(predictions[0][i]) for i, cls in enumerate(self.class_names)}
            
            return {
                'class': predicted_class,
                'probabilities': probabilities,
                'raw_prediction': predictions[0]
            }
            
        except Exception as e:
            raise Exception(f"Error classifying image: {str(e)}")