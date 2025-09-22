import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class OptimizedCSILocalizationModel(Model):
    def __init__(self):
        super(OptimizedCSILocalizationModel, self).__init__()
        
        # Best hyperparameters from tuning
        self.l2_reg = 0.00043799
        self.dropout_rate = 0.2
        
        # Enhanced encoder with residual connections
        self.conv1a = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.conv1b = layers.Conv3D(32, (3, 3, 3), padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn1 = layers.BatchNormalization()
        
        self.conv2a = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.conv2b = layers.Conv3D(64, (3, 3, 3), padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn2 = layers.BatchNormalization()
        
        self.conv3a = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.conv3b = layers.Conv3D(128, (3, 3, 3), padding='same',
                                  kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn3 = layers.BatchNormalization()
        
        # Global average pooling instead of flatten for better spatial invariance
        self.global_pool = layers.GlobalAveragePooling3D()
        
        # Enhanced dense layers with residual connections
        self.dense1 = layers.Dense(512, kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn4 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(256, kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn5 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        self.dense3 = layers.Dense(128, kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn6 = layers.BatchNormalization()
        self.act3 = layers.Activation('relu')
        
        # Position prediction head with improved accuracy
        self.position_dense = layers.Dense(64, activation='relu',
                                         kernel_regularizer=regularizers.l2(self.l2_reg))
        self.position_head = layers.Dense(2)  # x, y coordinates
        
        # Enhanced consistency regularization
        self.consistency_dense = layers.Dense(64, activation='relu',
                                            kernel_regularizer=regularizers.l2(self.l2_reg))
        self.consistency_head = layers.Dense(128, activation='relu')
        
    def residual_block(self, x, conv_a, conv_b, bn, training=False):
        shortcut = x
        x = conv_a(x)
        x = conv_b(x)
        x = bn(x, training=training)
        # Add shortcut if shapes match
        if shortcut.shape[-1] == x.shape[-1]:
            x = layers.Add()([shortcut, x])
        return tf.nn.relu(x)
    
    def call(self, inputs, training=False):
        # Enhanced forward pass with residual connections
        x = self.residual_block(inputs, self.conv1a, self.conv1b, self.bn1, training)
        x = self.residual_block(x, self.conv2a, self.conv2b, self.bn2, training)
        x = self.residual_block(x, self.conv3a, self.conv3b, self.bn3, training)
        
        x = self.global_pool(x)
        
        # Dense layers with residual connections
        x1 = self.dense1(x)
        x1 = self.bn4(x1, training=training)
        x1 = self.act1(x1)
        x1 = self.dropout1(x1, training=training)
        
        x2 = self.dense2(x1)
        x2 = self.bn5(x2, training=training)
        x2 = self.act2(x2)
        x2 = self.dropout2(x2, training=training)
        
        x3 = self.dense3(x2)
        x3 = self.bn6(x3, training=training)
        x3 = self.act3(x3)
        
        # Enhanced position prediction
        pos = self.position_dense(x3)
        position = self.position_head(pos)
        
        # Enhanced consistency prediction
        cons = self.consistency_dense(x3)
        consistency = self.consistency_head(cons)
        
        return position, consistency
    
    def build_graph(self):
        x = tf.keras.Input(shape=(4, 8, 16, 2))
        return Model(inputs=[x], outputs=self.call(x)) 