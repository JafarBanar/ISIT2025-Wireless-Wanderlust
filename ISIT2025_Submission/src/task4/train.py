import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, List

from ..utils.data_utils import load_csi_data, preprocess_csi, calculate_mae, calculate_r90
from .model import CSICompressor, LocalTransmissionPolicy, CentralFusionModel, CombinedLoss

class CSIDataset(Dataset):
    def __init__(self, csi_data: np.ndarray, positions: np.ndarray):
        self.csi_data = csi_data  # Shape: (n_samples, 4, 8, 16, 2)
        self.positions = positions  # Shape: (n_samples, 2)
        
    def __len__(self) -> int:
        return len(self.csi_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.csi_data[idx]).float(), torch.from_numpy(self.positions[idx]).float()

def train_epoch(compressor: nn.Module,
                local_policies: List[nn.Module],
                central_model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    # Set models to training mode
    compressor.train()
    for policy in local_policies:
        policy.train()
    central_model.train()
    
    total_loss = 0
    total_mae = 0
    total_r90 = 0
    total_transmission = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Compress CSI data
        compressed_data = []
        transmission_probs = []
        transmitted_data = []
        
        for i in range(data.size(1)):  # For each antenna
            # Get CSI data for this antenna
            antenna_data = data[:, i]  # Shape: (batch_size, 8, 16, 2)
            
            # Compress data
            compressed = compressor(antenna_data)
            compressed_data.append(compressed)
            
            # Get transmission probability
            prob = local_policies[i](compressed)
            transmission_probs.append(prob)
            
            # Sample transmission decision
            transmission = (torch.rand_like(prob) < prob).float()
            
            # Store transmitted data
            transmitted_data.append(compressed * transmission)
        
        # Stack transmitted data
        transmitted_data = torch.stack(transmitted_data, dim=1)  # Shape: (batch_size, 4, compressed_size)
        
        # Get predictions from central model
        output = central_model(transmitted_data)
        
        # Calculate loss
        loss, mae_loss, r90_loss, transmission_loss = criterion(output, target, transmission_probs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_mae += mae_loss.item()
        total_r90 += r90_loss.item()
        total_transmission += transmission_loss.item()
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'mae': total_mae / n_batches,
        'r90': total_r90 / n_batches,
        'transmission': total_transmission / n_batches
    }

def validate(compressor: nn.Module,
            local_policies: List[nn.Module],
            central_model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Dict[str, float]:
    """Validate the model."""
    # Set models to evaluation mode
    compressor.eval()
    for policy in local_policies:
        policy.eval()
    central_model.eval()
    
    total_loss = 0
    total_mae = 0
    total_r90 = 0
    total_transmission = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Compress CSI data
            compressed_data = []
            transmission_probs = []
            transmitted_data = []
            
            for i in range(data.size(1)):  # For each antenna
                # Get CSI data for this antenna
                antenna_data = data[:, i]  # Shape: (batch_size, 8, 16, 2)
                
                # Compress data
                compressed = compressor(antenna_data)
                compressed_data.append(compressed)
                
                # Get transmission probability
                prob = local_policies[i](compressed)
                transmission_probs.append(prob)
                
                # Sample transmission decision
                transmission = (torch.rand_like(prob) < prob).float()
                
                # Store transmitted data
                transmitted_data.append(compressed * transmission)
            
            # Stack transmitted data
            transmitted_data = torch.stack(transmitted_data, dim=1)  # Shape: (batch_size, 4, compressed_size)
            
            # Get predictions from central model
            output = central_model(transmitted_data)
            
            # Calculate loss
            loss, mae_loss, r90_loss, transmission_loss = criterion(output, target, transmission_probs)
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae_loss.item()
            total_r90 += r90_loss.item()
            total_transmission += transmission_loss.item()
    
    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'mae': total_mae / n_batches,
        'r90': total_r90 / n_batches,
        'transmission': total_transmission / n_batches
    }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    n_antennas = 4
    compression_rate = 0.25  # 25% of original size
    
    compressor = CSICompressor(compression_rate).to(device)
    local_policies = [LocalTransmissionPolicy(compressor.compressed_size).to(device) for _ in range(n_antennas)]
    central_model = CentralFusionModel(n_antennas, compressor.compressed_size).to(device)
    
    # Check parameter counts
    compressor_params = compressor.count_parameters()
    local_params = sum(policy.count_parameters() for policy in local_policies)
    central_params = central_model.count_parameters()
    print(f"Compressor has {compressor_params:,} parameters")
    print(f"Local policies have {local_params:,} parameters")
    print(f"Central model has {central_params:,} parameters")
    assert compressor_params <= 40_000_000, f"Compressor exceeds 40M parameter limit: {compressor_params:,}"
    assert local_params <= 20_000_000, f"Local policies exceed 20M parameter limit: {local_params:,}"
    assert central_params <= 60_000_000, f"Central model exceeds 60M parameter limit: {central_params:,}"
    
    # Load and preprocess data
    csi_data, positions = load_csi_data('data/dichasus-cf0x.h5')
    csi_tensor = preprocess_csi(csi_data)
    positions_tensor = torch.from_numpy(positions).float()
    
    # Create datasets
    dataset = CSIDataset(csi_tensor.numpy(), positions_tensor.numpy())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize optimizer and loss
    all_params = []
    all_params.extend(compressor.parameters())
    for policy in local_policies:
        all_params.extend(policy.parameters())
    all_params.extend(central_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)
    criterion = CombinedLoss(transmission_weight=0.1)
    
    # Initialize tensorboard
    writer = SummaryWriter('runs/task4')
    
    # Training loop
    n_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(compressor, local_policies, central_model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(compressor, local_policies, central_model, val_loader, criterion, device)
        
        # Log metrics
        for metric, value in train_metrics.items():
            writer.add_scalar(f'train/{metric}', value, epoch)
        for metric, value in val_metrics.items():
            writer.add_scalar(f'val/{metric}', value, epoch)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Save compressor
            torch.save({
                'epoch': epoch,
                'model_state_dict': compressor.state_dict(),
                'val_loss': best_val_loss,
            }, 'models/task4_compressor_best.pth')
            # Save local policies
            for i, policy in enumerate(local_policies):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'val_loss': best_val_loss,
                }, f'models/task4_local_policy_{i}_best.pth')
            # Save central model
            torch.save({
                'epoch': epoch,
                'model_state_dict': central_model.state_dict(),
                'val_loss': best_val_loss,
            }, 'models/task4_central_best.pth')
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, "
              f"R90: {train_metrics['r90']:.4f}, Transmission: {train_metrics['transmission']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"R90: {val_metrics['r90']:.4f}, Transmission: {val_metrics['transmission']:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main() 