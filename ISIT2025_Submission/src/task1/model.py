import tensorflow as tf
from tensorflow.keras import layers

class SimpleLocalizationModel(tf.keras.Model):
    """Simple neural network for CSI-based localization."""
    
    def __init__(self, input_shape=(4, 8, 16, 2)):
        super().__init__()
        
        # CNN layers for CSI processing
        self.conv1 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool3D((1, 2, 2))
        
        self.conv2 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool3D((1, 2, 2))
        
        self.conv3 = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        
        # Dense layers for position prediction
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.position = layers.Dense(2)  # x, y coordinates
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.position(x)

# For script usage
if __name__ == "__main__":
    model = SimpleLocalizationModel()
    model.build(input_shape=(None, 4, 8, 16, 2))
    model.summary() 