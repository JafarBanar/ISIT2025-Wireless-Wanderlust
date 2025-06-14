import torch
import torch.nn as nn
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LocalTransmissionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN layers for CSI processing
        self.cnn = nn.Sequential(
            # Input: (batch_size, 8, 16, 2)
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 128 * 2 * 4  # After pooling
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability of transmission
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 8, 16, 2)
        # Rearrange for Conv2d: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Apply CNN
        x = self.cnn(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply FC layers
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CentralFusionModel(nn.Module):
    def __init__(self, n_antennas: int = 4):
        super().__init__()
        self.n_antennas = n_antennas
        
        # CNN layers for CSI processing
        self.cnn = nn.Sequential(
            # Input: (batch_size, n_antennas, 8, 16, 2)
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 128 * 1 * 2 * 4  # After pooling
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Output: (x, y) coordinates
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_antennas, 8, 16, 2)
        # Rearrange for Conv3d: (batch_size, channels, depth, height, width)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply CNN
        x = self.cnn(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply FC layers
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CombinedLoss(nn.Module):
    def __init__(self, transmission_weight: float = 0.1):
        super().__init__()
        self.mae = nn.L1Loss()
        self.transmission_weight = transmission_weight
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                transmission_probs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss (MAE + R90 + transmission cost).
        
        Args:
            predictions: Predicted positions (batch_size, 2)
            targets: Ground truth positions (batch_size, 2)
            transmission_probs: List of transmission probabilities for each antenna
            
        Returns:
            Tuple of (total_loss, mae_loss, r90_loss, transmission_loss)
        """
        # Calculate MAE
        mae_loss = self.mae(predictions, targets)
        
        # Calculate R90
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
        r90_loss = torch.quantile(distances, 0.9)
        
        # Calculate transmission cost
        transmission_loss = sum(prob.mean() for prob in transmission_probs)
        
        # Combined loss
        total_loss = 0.5 * mae_loss + 0.5 * r90_loss + self.transmission_weight * transmission_loss
        
        return total_loss, mae_loss, r90_loss, transmission_loss 

class LocalModel(tf.keras.Model):
    """Local model that processes CSI and decides transmission policy."""
    def __init__(self, input_shape=(4, 8, 16, 2)):
        super().__init__()
        
        # CNN for CSI processing
        self.conv1 = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool3D((1, 2, 2))
        
        self.conv2 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool3D((1, 2, 2))
        
        # Dense layers for transmission decision
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.transmit_prob = layers.Dense(1, activation='sigmoid')  # Probability of transmission
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.transmit_prob(x)

class CentralModel(tf.keras.Model):
    """Central model that processes transmitted CSI and predicts positions."""
    def __init__(self, input_shape=(4, 8, 16, 2)):
        super().__init__()
        
        # CNN for CSI processing
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

class GrantFreeModel:
    """Combined model for grant-free random access."""
    def __init__(self):
        self.local_model = LocalModel()
        self.central_model = CentralModel()
        
        # Build models to initialize weights
        self.local_model.build(input_shape=(None, 4, 8, 16, 2))
        self.central_model.build(input_shape=(None, 4, 8, 16, 2))
        
        # Compile models
        self.local_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.central_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='mae',
            metrics=['mae']
        )
    
    def train_step(self, csi, position):
        """Custom training step that simulates the grant-free process."""
        # Local model decides whether to transmit
        transmit_probs = self.local_model(csi)
        transmit_mask = tf.cast(transmit_probs > 0.5, tf.float32)
        
        # Only transmitted CSI reaches the central model
        transmitted_csi = csi * transmit_mask
        
        # Central model predicts position
        position_pred = self.central_model(transmitted_csi)
        
        # Calculate losses
        transmission_loss = tf.keras.losses.binary_crossentropy(
            transmit_mask, transmit_probs
        )
        position_loss = tf.keras.losses.mae(position, position_pred)
        
        # Combined loss
        total_loss = transmission_loss + position_loss
        
        return {
            'total_loss': total_loss,
            'transmission_loss': transmission_loss,
            'position_loss': position_loss
        }

# For script usage
if __name__ == "__main__":
    model = GrantFreeModel()
    print("Local model summary:")
    model.local_model.summary()
    print("\nCentral model summary:")
    model.central_model.summary() 