import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

def create_improved_localization_model(input_shape, num_classes, l2_reg=0.01, dropout_rate=0.3):
    """
    Create the improved localization model.
    
    Args:
        input_shape (tuple): Shape of input data (height, width, channels)
        num_classes (int): Number of output classes (coordinates)
        l2_reg (float): L2 regularization factor
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled model
    """
    # Input validation
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be 3D (height, width, channels), got {input_shape}")
    if num_classes <= 0:
        raise ValueError(f"Number of classes must be positive, got {num_classes}")
    if not (0 <= dropout_rate <= 1):
        raise ValueError(f"Dropout rate must be between 0 and 1, got {dropout_rate}")
    if l2_reg < 0:
        raise ValueError(f"L2 regularization must be non-negative, got {l2_reg}")

    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block with increased filters
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second convolutional block with increased filters
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Third convolutional block
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers with increased capacity
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes)(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with consistent optimizer settings
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='mse',
                 metrics=['mae'])
    
    return model

def get_training_callbacks():
    """
    Get training callbacks with consistent settings:
    - Learning rate reduction
    - Early stopping
    - Model checkpointing
    """
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'results/improved_localization/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks 