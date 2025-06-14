import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')

# Create output directory
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

def create_layer_box(ax, x, y, width, height, label, color='#3498db', alpha=0.7):
    """Create a box representing a neural network layer."""
    rect = patches.Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha,
                           edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    
    # Add label
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=8,
            color='black', fontweight='bold')

def create_connection(ax, x1, y1, x2, y2, color='gray', alpha=0.3):
    """Create a connection line between layers."""
    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1)

def plot_task1_architecture():
    """Plot architecture for Task 1 (Vanilla Localization)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer dimensions
    layer_width = 2
    layer_height = 0.5
    spacing = 0.3
    
    # Input layer
    create_layer_box(ax, 0, 2, layer_width, layer_height, 
                    'CSI Input\n(64x64x2)', color='#2ecc71')
    
    # CNN layers
    create_layer_box(ax, 3, 2, layer_width, layer_height, 
                    'Conv3D\n(32 filters)', color='#3498db')
    create_layer_box(ax, 3, 1.2, layer_width, layer_height, 
                    'Conv3D\n(64 filters)', color='#3498db')
    create_layer_box(ax, 3, 0.4, layer_width, layer_height, 
                    'Conv3D\n(128 filters)', color='#3498db')
    
    # Dense layers
    create_layer_box(ax, 6, 1.2, layer_width, layer_height, 
                    'Dense\n(256 units)', color='#e74c3c')
    create_layer_box(ax, 6, 0.4, layer_width, layer_height, 
                    'Dense\n(2 units)', color='#e74c3c')
    
    # Connections
    create_connection(ax, 2, 2.25, 3, 2.25)
    create_connection(ax, 5, 2.25, 6, 1.45)
    create_connection(ax, 5, 1.45, 6, 0.65)
    
    # Add pooling and activation annotations
    ax.text(2.5, 1.7, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(2.5, 0.7, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(2.5, -0.3, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(5.5, 0.7, 'ReLU', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_title('Task 1: Vanilla Localization Architecture')
    
    plt.savefig(output_dir / 'task1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_task2_architecture():
    """Plot architecture for Task 2 (Trajectory-Aware)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer dimensions
    layer_width = 2
    layer_height = 0.5
    spacing = 0.3
    
    # Input layers
    create_layer_box(ax, 0, 3, layer_width, layer_height, 
                    'CSI Input\n(64x64x2)', color='#2ecc71')
    create_layer_box(ax, 0, 2, layer_width, layer_height, 
                    'Previous Positions\n(5x2)', color='#2ecc71')
    
    # CNN branch
    create_layer_box(ax, 3, 3, layer_width, layer_height, 
                    'Conv3D\n(32 filters)', color='#3498db')
    create_layer_box(ax, 3, 2.2, layer_width, layer_height, 
                    'Conv3D\n(64 filters)', color='#3498db')
    create_layer_box(ax, 3, 1.4, layer_width, layer_height, 
                    'Conv3D\n(128 filters)', color='#3498db')
    
    # LSTM branch
    create_layer_box(ax, 3, 1, layer_width, layer_height, 
                    'LSTM\n(64 units)', color='#9b59b6')
    
    # Fusion and output layers
    create_layer_box(ax, 6, 1.7, layer_width, layer_height, 
                    'Concatenate', color='#f1c40f')
    create_layer_box(ax, 6, 1, layer_width, layer_height, 
                    'Dense\n(256 units)', color='#e74c3c')
    create_layer_box(ax, 6, 0.3, layer_width, layer_height, 
                    'Dense\n(2 units)', color='#e74c3c')
    
    # Connections
    create_connection(ax, 2, 3.25, 3, 3.25)
    create_connection(ax, 2, 2.25, 3, 2.25)
    create_connection(ax, 5, 3.25, 6, 1.95)
    create_connection(ax, 5, 1.25, 6, 1.25)
    create_connection(ax, 5, 1.25, 6, 0.55)
    
    # Add annotations
    ax.text(2.5, 2.7, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.9, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.1, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(5.5, 0.7, 'ReLU', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Task 2: Trajectory-Aware Architecture')
    
    plt.savefig(output_dir / 'task2_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_task3_architecture():
    """Plot architecture for Task 3 (Grant-Free RA)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer dimensions
    layer_width = 2
    layer_height = 0.5
    spacing = 0.3
    
    # Input layers
    create_layer_box(ax, 0, 3, layer_width, layer_height, 
                    'CSI Input\n(64x64x2)', color='#2ecc71')
    create_layer_box(ax, 0, 2, layer_width, layer_height, 
                    'User Activity\n(1)', color='#2ecc71')
    
    # CNN branch
    create_layer_box(ax, 3, 3, layer_width, layer_height, 
                    'Conv3D\n(32 filters)', color='#3498db')
    create_layer_box(ax, 3, 2.2, layer_width, layer_height, 
                    'Conv3D\n(64 filters)', color='#3498db')
    
    # Attention branch
    create_layer_box(ax, 3, 1.4, layer_width, layer_height, 
                    'Self-Attention\n(64 units)', color='#e67e22')
    
    # Fusion and output layers
    create_layer_box(ax, 6, 2.2, layer_width, layer_height, 
                    'Concatenate', color='#f1c40f')
    create_layer_box(ax, 6, 1.4, layer_width, layer_height, 
                    'Dense\n(128 units)', color='#e74c3c')
    create_layer_box(ax, 6, 0.6, layer_width, layer_height, 
                    'Dense\n(2 units)', color='#e74c3c')
    
    # Connections
    create_connection(ax, 2, 3.25, 3, 3.25)
    create_connection(ax, 2, 2.25, 3, 2.25)
    create_connection(ax, 5, 3.25, 6, 2.45)
    create_connection(ax, 5, 1.65, 6, 1.65)
    create_connection(ax, 5, 1.65, 6, 0.85)
    
    # Add annotations
    ax.text(2.5, 2.7, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.9, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(5.5, 1.1, 'ReLU', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Task 3: Grant-Free RA Architecture')
    
    plt.savefig(output_dir / 'task3_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_task4_architecture():
    """Plot architecture for Task 4 (Feature Selection + RA)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer dimensions
    layer_width = 2
    layer_height = 0.5
    spacing = 0.3
    
    # Input layers
    create_layer_box(ax, 0, 3, layer_width, layer_height, 
                    'CSI Input\n(64x64x2)', color='#2ecc71')
    create_layer_box(ax, 0, 2, layer_width, layer_height, 
                    'Selected Features\n(32)', color='#2ecc71')
    
    # Feature selection branch
    create_layer_box(ax, 3, 3, layer_width, layer_height, 
                    'Feature Selection\n(32 features)', color='#1abc9c')
    
    # CNN branch
    create_layer_box(ax, 3, 2, layer_width, layer_height, 
                    'Conv3D\n(32 filters)', color='#3498db')
    create_layer_box(ax, 3, 1.2, layer_width, layer_height, 
                    'Conv3D\n(64 filters)', color='#3498db')
    
    # Fusion and output layers
    create_layer_box(ax, 6, 2.2, layer_width, layer_height, 
                    'Concatenate', color='#f1c40f')
    create_layer_box(ax, 6, 1.4, layer_width, layer_height, 
                    'Dense\n(128 units)', color='#e74c3c')
    create_layer_box(ax, 6, 0.6, layer_width, layer_height, 
                    'Dense\n(2 units)', color='#e74c3c')
    
    # Connections
    create_connection(ax, 2, 3.25, 3, 3.25)
    create_connection(ax, 2, 2.25, 3, 2.25)
    create_connection(ax, 5, 3.25, 6, 2.45)
    create_connection(ax, 5, 2.25, 6, 1.65)
    create_connection(ax, 5, 1.65, 6, 0.85)
    
    # Add annotations
    ax.text(2.5, 2.7, 'Feature Selection', ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.9, 'MaxPool3D + ReLU', ha='center', va='center', fontsize=8)
    ax.text(5.5, 1.1, 'ReLU', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Task 4: Feature Selection + RA Architecture')
    
    plt.savefig(output_dir / 'task4_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_architecture():
    """Plot a combined view of all architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Plot each architecture in a subplot
    plot_functions = [
        plot_task1_architecture,
        plot_task2_architecture,
        plot_task3_architecture,
        plot_task4_architecture
    ]
    
    for ax, plot_func in zip(axes, plot_functions):
        # Create a new figure for each architecture
        temp_fig = plt.figure(figsize=(10, 6))
        plot_func()
        plt.close(temp_fig)
        
        # Load the saved image and display it in the subplot
        task_num = plot_functions.index(plot_func) + 1
        img = plt.imread(output_dir / f'task{task_num}_architecture.png')
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_architectures.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating model architecture visualizations...")
    
    # Generate individual architecture plots
    plot_task1_architecture()
    plot_task2_architecture()
    plot_task3_architecture()
    plot_task4_architecture()
    
    # Generate combined architecture plot
    plot_combined_architecture()
    
    print(f"Plots saved to {output_dir}") 