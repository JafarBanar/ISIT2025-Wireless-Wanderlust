import tensorflow as tf
from tensorflow.keras import layers, Model

class CSILocalizationModel(Model):
    def __init__(self):
        super(CSILocalizationModel, self).__init__()
        
        # Input shape: (batch_size, 4, 8, 16, 2) for CSI data
        # Encoder
        self.conv1 = layers.Conv3D(32, (2, 2, 2), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv3D(64, (2, 2, 2), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv3D(128, (2, 2, 2), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        
        # Shared layers
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        self.dense3 = layers.Dense(128, activation='relu')
        self.bn6 = layers.BatchNormalization()
        
        # Position prediction head
        self.position_head = layers.Dense(2)  # x, y coordinates
        
        # Consistency regularization head
        self.consistency_head = layers.Dense(128, activation='relu')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.bn4(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn5(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.bn6(x, training=training)
        
        # Position prediction
        position = self.position_head(x)
        
        # Consistency prediction
        consistency = self.consistency_head(x)
        
        return position, consistency
    
    def build_graph(self):
        x = tf.keras.Input(shape=(4, 8, 16, 2))
        return Model(inputs=[x], outputs=self.call(x)) 