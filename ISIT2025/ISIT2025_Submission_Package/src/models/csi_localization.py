import tensorflow as tf
import numpy as np
import dataclasses
import enum

@dataclasses.dataclass
class CSIModelConfig:
    input_shape: tuple
    num_classes: int = 2
    l2_reg: float = 0.01
    dropout_rate: float = 0.3

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention = tf.keras.layers.Dense(1, activation='tanh')
        
    def call(self, inputs):
        attention_weights = self.attention(inputs)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

class CSIModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes=2, l2_reg=0.01, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self._model_input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # CSI Feature Extraction
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
        
        # Attention mechanism for CSI features
        self.attention = AttentionLayer()
        
        # LSTM for trajectory awareness
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.lstm_attention = AttentionLayer()
        
        # Dense layers for final prediction
        self.dense1 = tf.keras.layers.Dense(256, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.dense2 = tf.keras.layers.Dense(128, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.dropout5 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.output_layer = tf.keras.layers.Dense(self.num_classes)
        
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
        # CSI feature extraction
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
        
        # Apply attention to CSI features
        x = self.attention(x)
        
        # LSTM for trajectory awareness
        x = tf.expand_dims(x, axis=1)  # Add sequence dimension
        x = self.lstm(x)
        x = self.lstm_attention(x)
        
        # Final prediction layers
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
                'models/csi_localization/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/csi_localization',
                histogram_freq=1
            )
        ]

def create_csi_localization_model(input_shape, num_classes=2, l2_reg=0.01, dropout_rate=0.3):
    """
    Create the CSI localization model.
    
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

    return CSIModel(input_shape, num_classes, l2_reg, dropout_rate)

class CSIModelType(enum.Enum):
    VANILLA = "vanilla"
    TRAJECTORY = "trajectory"
    GRANT_FREE = "grant_free" 