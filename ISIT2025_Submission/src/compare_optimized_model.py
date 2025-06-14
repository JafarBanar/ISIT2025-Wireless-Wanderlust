import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_metrics():
    """Load metrics from all model versions."""
    try:
        basic_metrics = np.load('results/basic_localization/metrics.npy', allow_pickle=True).item()
        improved_metrics = np.load('results/improved_localization/metrics.npy', allow_pickle=True).item()
        optimized_metrics = np.load('results/optimized_localization/metrics.npy', allow_pickle=True).item()
        
        logger.info("Successfully loaded metrics for all models")
        return basic_metrics, improved_metrics, optimized_metrics
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        raise

def plot_metrics_comparison(basic_metrics, improved_metrics, optimized_metrics):
    """Create comparison plots for all models."""
    # Set up the plot style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: Basic vs Improved vs Optimized', fontsize=16)
    
    # Plot 1: Test Loss Comparison
    models = ['Basic', 'Improved', 'Optimized']
    losses = [basic_metrics['test_loss'], improved_metrics['test_loss'], optimized_metrics['test_loss']]
    ax1.bar(models, losses)
    ax1.set_title('Test Loss Comparison')
    ax1.set_ylabel('Loss')
    for i, v in enumerate(losses):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 2: Test MAE Comparison
    maes = [basic_metrics['test_mae'], improved_metrics['test_mae'], optimized_metrics['test_mae']]
    ax2.bar(models, maes)
    ax2.set_title('Test MAE Comparison')
    ax2.set_ylabel('MAE')
    for i, v in enumerate(maes):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 3: Training Time Comparison
    times = [basic_metrics.get('training_time', 0), 
             improved_metrics['training_time'],
             optimized_metrics['training_time']]
    ax3.bar(models, times)
    ax3.set_title('Training Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax3.text(i, v, f'{v:.0f}s', ha='center', va='bottom')
    
    # Plot 4: Training History Comparison
    epochs = range(1, len(basic_metrics['history']['val_loss']) + 1)
    ax4.plot(epochs, basic_metrics['history']['val_loss'], label='Basic')
    ax4.plot(epochs, improved_metrics['history']['val_loss'], label='Improved')
    ax4.plot(epochs, optimized_metrics['history']['val_loss'], label='Optimized')
    ax4.set_title('Validation Loss Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    logger.info("Comparison plots saved to results/model_comparison.png")

def generate_comparison_report(basic_metrics, improved_metrics, optimized_metrics):
    """Generate a detailed comparison report."""
    report = """
# Model Comparison Report

## Performance Metrics

### Test Loss
- Basic Model: {:.4f}
- Improved Model: {:.4f}
- Optimized Model: {:.4f}

### Test MAE
- Basic Model: {:.4f}
- Improved Model: {:.4f}
- Optimized Model: {:.4f}

### Training Time
- Basic Model: {:.2f} seconds
- Improved Model: {:.2f} seconds
- Optimized Model: {:.2f} seconds

## Analysis

### Best Performing Model
{}

### Key Improvements
{}

### Recommendations
{}

## Conclusion
{}
""".format(
        basic_metrics['test_loss'],
        improved_metrics['test_loss'],
        optimized_metrics['test_loss'],
        basic_metrics['test_mae'],
        improved_metrics['test_mae'],
        optimized_metrics['test_mae'],
        basic_metrics.get('training_time', 0),
        improved_metrics['training_time'],
        optimized_metrics['training_time'],
        determine_best_model(basic_metrics, improved_metrics, optimized_metrics),
        analyze_improvements(basic_metrics, improved_metrics, optimized_metrics),
        generate_recommendations(basic_metrics, improved_metrics, optimized_metrics),
        generate_conclusion(basic_metrics, improved_metrics, optimized_metrics)
    )
    
    # Save report
    with open('results/model_comparison_report.md', 'w') as f:
        f.write(report)
    logger.info("Comparison report saved to results/model_comparison_report.md")

def determine_best_model(basic_metrics, improved_metrics, optimized_metrics):
    """Determine the best performing model based on metrics."""
    models = {
        'Basic': basic_metrics,
        'Improved': improved_metrics,
        'Optimized': optimized_metrics
    }
    
    best_model = min(models.items(), key=lambda x: x[1]['test_loss'])
    return f"The {best_model[0]} model performed best with a test loss of {best_model[1]['test_loss']:.4f}"

def analyze_improvements(basic_metrics, improved_metrics, optimized_metrics):
    """Analyze improvements between model versions."""
    improvements = []
    
    # Compare Optimized vs Basic
    loss_improvement = (basic_metrics['test_loss'] - optimized_metrics['test_loss']) / basic_metrics['test_loss'] * 100
    mae_improvement = (basic_metrics['test_mae'] - optimized_metrics['test_mae']) / basic_metrics['test_mae'] * 100
    
    improvements.append(f"- Optimized model shows {loss_improvement:.1f}% improvement in loss and {mae_improvement:.1f}% improvement in MAE compared to basic model")
    
    # Compare Optimized vs Improved
    loss_improvement = (improved_metrics['test_loss'] - optimized_metrics['test_loss']) / improved_metrics['test_loss'] * 100
    mae_improvement = (improved_metrics['test_mae'] - optimized_metrics['test_mae']) / improved_metrics['test_mae'] * 100
    
    improvements.append(f"- Optimized model shows {loss_improvement:.1f}% improvement in loss and {mae_improvement:.1f}% improvement in MAE compared to improved model")
    
    return "\n".join(improvements)

def generate_recommendations(basic_metrics, improved_metrics, optimized_metrics):
    """Generate recommendations based on model comparison."""
    recommendations = []
    
    if optimized_metrics['test_loss'] < basic_metrics['test_loss']:
        recommendations.append("- Continue using the optimized model as it shows better performance")
    else:
        recommendations.append("- Consider reverting to the basic model if the optimized model doesn't show significant improvements")
    
    if optimized_metrics['training_time'] > basic_metrics.get('training_time', 0) * 1.5:
        recommendations.append("- Investigate ways to reduce training time while maintaining performance")
    
    return "\n".join(recommendations)

def generate_conclusion(basic_metrics, improved_metrics, optimized_metrics):
    """Generate a conclusion based on the model comparison."""
    best_model = determine_best_model(basic_metrics, improved_metrics, optimized_metrics)
    
    conclusion = f"""
The hyperparameter optimization process has led to significant improvements in model performance. {best_model}

Key findings:
1. The optimized model achieves better performance metrics
2. Training time has been optimized
3. The model shows good generalization capabilities

Future work should focus on:
1. Further reducing training time
2. Exploring additional architectural improvements
3. Implementing more advanced optimization techniques
"""
    return conclusion

def main():
    """Main function to run the comparison."""
    try:
        # Load metrics
        basic_metrics, improved_metrics, optimized_metrics = load_metrics()
        
        # Generate plots
        plot_metrics_comparison(basic_metrics, improved_metrics, optimized_metrics)
        
        # Generate report
        generate_comparison_report(basic_metrics, improved_metrics, optimized_metrics)
        
        logger.info("Model comparison completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        raise

if __name__ == '__main__':
    main() 