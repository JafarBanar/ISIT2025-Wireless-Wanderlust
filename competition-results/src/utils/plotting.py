import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss=None, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_loss: List of training loss values
        val_loss: Optional list of validation loss values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 