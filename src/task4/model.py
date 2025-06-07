import torch
import torch.nn as nn
from typing import Tuple, List

class CSICompressor(nn.Module):
    def __init__(self, compression_rate: float = 0.25):
        super().__init__()
        self.compression_rate = compression_rate
        
        # Calculate compressed size
        original_size = 8 * 16 * 2  # 8 antennas * 16 frequencies * 2 (real/imag)
        self.compressed_size = int(original_size * compression_rate)
        
        # Encoder
        self.encoder = nn.Sequential(
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
        
        # Calculate encoder output size
        self.encoder_output_size = 128 * 2 * 4  # After pooling
        
        # Compression layers
        self.compression = nn.Sequential(
            nn.Linear(self.encoder_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.compressed_size)
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
        
        # Apply encoder
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply compression
        x = self.compression(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LocalTransmissionPolicy(nn.Module):
    def __init__(self, compressed_size: int):
        super().__init__()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(compressed_size, 256),
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
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CentralFusionModel(nn.Module):
    def __init__(self, n_antennas: int = 4, compressed_size: int = 32):
        super().__init__()
        self.n_antennas = n_antennas
        
        # Decompression layers
        self.decompression = nn.Sequential(
            nn.Linear(compressed_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # CNN layers for processing decompressed data
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
        # Input shape: (batch_size, n_antennas, compressed_size)
        batch_size = x.size(0)
        
        # Decompress each antenna's data
        decompressed = []
        for i in range(self.n_antennas):
            antenna_data = x[:, i]  # (batch_size, compressed_size)
            decompressed.append(self.decompression(antenna_data))
        
        # Stack and reshape
        x = torch.stack(decompressed, dim=1)  # (batch_size, n_antennas, 1024)
        x = x.view(batch_size, self.n_antennas, 8, 16, 2)  # Reshape to original CSI format
        
        # Rearrange for Conv3d: (batch_size, channels, depth, height, width)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply CNN
        x = self.cnn(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
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