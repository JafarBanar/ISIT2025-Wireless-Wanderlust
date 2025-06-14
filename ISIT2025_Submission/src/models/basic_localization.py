import tensorflow as tf
import keras
from keras import layers, models, regularizers

@keras.utils.register_keras_serializable()
class BasicLocalizationModel(keras.Model):
    def __init__(self, model_input_shape, num_classes=2, l2_reg=0.01, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.model_input_shape = model_input_shape
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # Define layers with reduced size
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.dropout3 = layers.Dropout(self.dropout_rate)
        
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(256, activation='relu',
                                 kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(128, activation='relu',
                                 kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn5 = layers.BatchNormalization()
        self.dropout5 = layers.Dropout(self.dropout_rate)
        
        self.output_layer = layers.Dense(self.num_classes)
        
        # Compile the model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='mse',
            metrics=['mae']
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)
        
        x = self.dense2(x)
        x = self.bn5(x, training=training)
        x = self.dropout5(x, training=training)
        
        return self.output_layer(x)

    def get_training_callbacks(self):
        return [
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
                'models/basic_localization/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

    def get_config(self):
        config = super().get_config()
        # Ensure model_input_shape is saved as a tuple
        config.update({
            "model_input_shape": tuple(self.model_input_shape),
            "num_classes": self.num_classes,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure model_input_shape is a tuple
        if isinstance(config.get("model_input_shape"), list):
            config["model_input_shape"] = tuple(config["model_input_shape"])
        return cls(**config)

def create_basic_localization_model(input_shape, num_classes, l2_reg=0.01, dropout_rate=0.3):
    """
    Create the basic localization model.
    
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

    return BasicLocalizationModel(input_shape, num_classes, l2_reg, dropout_rate) 