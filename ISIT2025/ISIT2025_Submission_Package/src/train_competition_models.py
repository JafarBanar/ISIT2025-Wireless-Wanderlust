import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.config.run_functions_eagerly(True)  # Enable eager execution for debugging
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from src.utils.data_loader import CompetitionDataLoader, SequenceDataLoader, load_and_preprocess_data
from src.models.basic_localization import create_basic_localization_model
from src.models.improved_localization import create_improved_localization_model
from src.models.optimized_localization import OptimizedLocalizationModel
from src.models.enhanced_localization import EnhancedLocalizationModel, ChannelAwareLocalizationModel
from src.utils.error_analysis import ErrorAnalyzer
from src.utils.visualization import ChannelVisualizer, RealTimeMonitor, AdvancedAnalyzer
from src.utils.report_generator import ReportGenerator
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Tuple

RESULTS_DIR = 'results/competition_models'
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_training_callbacks(model_name):
    """Get a list of callbacks for model training."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(RESULTS_DIR, 'logs', f'{model_name}_{timestamp}')
    
    return [
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, f'{model_name}_best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        # CSV logging
        tf.keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, f'{model_name}_training_log.csv')
        )
    ]

def train_and_evaluate(model, train_ds, val_ds, test_ds, model_name, is_sequence_model=False):
    """Train and evaluate a model, saving results and exporting the model."""
    print(f'\nTraining {model_name}...')
    
    # Get callbacks
    callbacks = get_training_callbacks(model_name)
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  # Reduced from 50 to 20 epochs
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f'\nEvaluating {model_name}...')
    metrics = model.evaluate(test_ds, return_dict=True)
    print(f'{model_name} test metrics:', metrics)
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, f'{model_name}_metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
    
    # Generate predictions for error analysis
    print(f'\nGenerating predictions for error analysis...')
    predictions = []
    ground_truth = []
    
    for batch in test_ds:
        if is_sequence_model:
            # For sequence models, use the last position in sequence
            x, y = batch
            pred = model.predict(x)
            predictions.extend(pred[:, -1, :])  # Last position in sequence
            ground_truth.extend(y[:, -1, :])    # Last position in sequence
        else:
            # For non-sequence models
            x, y = batch
            pred = model.predict(x)
            predictions.extend(pred)
            ground_truth.extend(y)
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Perform error analysis
    print(f'\nPerforming error analysis...')
    analyzer = ErrorAnalyzer()
    error_metrics, errors = analyzer.calculate_errors(predictions, ground_truth)
    analyzer.generate_report(
        model_name=model_name,
        metrics=error_metrics,
        errors=errors,
        predictions=predictions,
        ground_truth=ground_truth
    )
    
    # Export model in multiple formats
    print(f'\nExporting {model_name}...')
    
    # Save Keras model
    model.save(os.path.join(RESULTS_DIR, f'{model_name}_final.keras'))
    
    # Export as SavedModel
    export_dir = os.path.join(RESULTS_DIR, f'{model_name}_saved_model')
    model.export(export_dir, format='tf_saved_model')
    
    # Export as TFLite (if possible)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(RESULTS_DIR, f'{model_name}.tflite'), 'wb') as f:
            f.write(tflite_model)
    except Exception as e:
        print(f"Warning: Could not export to TFLite: {e}")
    
    return metrics, error_metrics

def train_channel_aware_model(
    train_csi: np.ndarray,
    train_pos: np.ndarray,
    val_csi: np.ndarray,
    val_pos: np.ndarray,
    test_csi: np.ndarray,
    test_pos: np.ndarray,
    model_config: Dict[str, Any],
    output_dir: str = 'results'
) -> Tuple[ChannelAwareLocalizationModel, Dict[str, Any]]:
    """Train the channel-aware localization model with enhanced monitoring and reporting."""
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize model
    model = ChannelAwareLocalizationModel(**model_config)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Initialize optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_csi),
        torch.FloatTensor(train_pos)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_csi),
        torch.FloatTensor(val_pos)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_csi),
        torch.FloatTensor(test_pos)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False
    )
    
    # Initialize visualization and monitoring tools
    visualizer = ChannelVisualizer(str(output_dir / 'visualizations'))
    monitor = RealTimeMonitor(str(output_dir / 'monitoring'))
    analyzer = AdvancedAnalyzer(str(output_dir / 'analysis'))
    report_generator = ReportGenerator(str(output_dir / 'reports'))
    
    # Training loop
    best_val_loss = float('inf')
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'channel_stats': [],
        'timestamps': []
    }
    
    logging.info("Starting training...")
    start_time = datetime.now()
    
    for epoch in range(model_config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        epoch_channel_stats = []
        
        for batch_csi, batch_pos in train_loader:
            batch_csi = batch_csi.cuda() if torch.cuda.is_available() else batch_csi
            batch_pos = batch_pos.cuda() if torch.cuda.is_available() else batch_pos
            
            optimizer.zero_grad()
            predictions, channel_quality = model(batch_csi)
            loss = nn.MSELoss()(predictions, batch_pos)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            epoch_channel_stats.append({
                'channel_quality': channel_quality.mean().item(),
                'collision_rate': (channel_quality < 0.5).float().mean().item()
            })
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_ground_truth = []
        val_channel_quality = []
        
        with torch.no_grad():
            for batch_csi, batch_pos in val_loader:
                batch_csi = batch_csi.cuda() if torch.cuda.is_available() else batch_csi
                batch_pos = batch_pos.cuda() if torch.cuda.is_available() else batch_pos
                
                predictions, channel_quality = model(batch_csi)
                loss = nn.MSELoss()(predictions, batch_pos)
                
                val_loss += loss.item()
                val_predictions.append(predictions.cpu().numpy())
                val_ground_truth.append(batch_pos.cpu().numpy())
                val_channel_quality.append(channel_quality.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_predictions = np.concatenate(val_predictions)
        val_ground_truth = np.concatenate(val_ground_truth)
        val_channel_quality = np.concatenate(val_channel_quality)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['channel_stats'].append({
            'channel_quality': np.mean([stats['channel_quality'] for stats in epoch_channel_stats]),
            'collision_rate': np.mean([stats['collision_rate'] for stats in epoch_channel_stats])
        })
        training_history['timestamps'].append((datetime.now() - start_time).total_seconds())
        
        # Update monitoring
        monitor.update_metrics(
            train_loss=train_loss,
            val_loss=val_loss,
            channel_quality=training_history['channel_stats'][-1]['channel_quality'],
            errors=np.sqrt(np.sum(np.square(val_predictions - val_ground_truth), axis=1)),
            channel_weights=model.channel_attention.weights.detach().cpu().numpy()
        )
        
        # Log progress
        logging.info(
            f"Epoch {epoch+1}/{model_config['epochs']} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Channel Quality: {training_history['channel_stats'][-1]['channel_quality']:.4f}"
        )
        
        # Generate visualizations every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualizer.plot_training_metrics(
                training_history['train_losses'],
                training_history['val_losses'],
                training_history['channel_stats']
            )
            visualizer.plot_channel_quality(
                val_channel_quality,
                training_history['channel_stats'][-1]['collision_rate']
            )
            visualizer.plot_channel_weights(
                model.channel_attention.weights.detach().cpu().numpy()
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
    
    # Final analysis
    logging.info("Training completed. Performing final analysis...")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    model.eval()
    
    # Test predictions
    test_predictions = []
    test_channel_quality = []
    
    with torch.no_grad():
        for batch_csi, _ in test_loader:
            batch_csi = batch_csi.cuda() if torch.cuda.is_available() else batch_csi
            predictions, channel_quality = model(batch_csi)
            test_predictions.append(predictions.cpu().numpy())
            test_channel_quality.append(channel_quality.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions)
    test_channel_quality = np.concatenate(test_channel_quality)
    
    # Generate final visualizations
    visualizer.plot_error_distribution(
        test_predictions,
        test_pos
    )
    visualizer.plot_trajectory_errors(
        test_predictions,
        test_pos
    )
    visualizer.plot_channel_correlation(
        test_channel_quality,
        np.sqrt(np.sum(np.square(test_predictions - test_pos), axis=1))
    )
    
    # Perform advanced analysis
    analysis_results = analyzer.analyze_model_performance(
        test_predictions,
        test_pos,
        test_channel_quality
    )
    
    # Generate reports
    training_report = report_generator.generate_training_report(
        training_history,
        model_config
    )
    
    analysis_report = report_generator.generate_analysis_report(
        analysis_results,
        test_predictions,
        test_pos,
        test_channel_quality
    )
    
    summary_report = report_generator.generate_summary_report(
        training_report,
        analysis_report
    )
    
    logging.info(f"Reports generated:")
    logging.info(f"- Training Report: {training_report['html']}")
    logging.info(f"- Analysis Report: {analysis_report['html']}")
    logging.info(f"- Summary Report: {summary_report['html']}")
    logging.info(f"- LaTeX Reports: {summary_report['latex']}")
    
    return model, training_history

def main():
    # Load and preprocess data
    train_csi, train_pos, val_csi, val_pos, test_csi, test_pos = load_and_preprocess_data()
    
    # Model configuration
    model_config = {
        'input_dim': train_csi.shape[1],
        'hidden_dim': 256,
        'output_dim': train_pos.shape[1],
        'num_channels': train_csi.shape[2],
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    }
    
    # Train model
    model, history = train_channel_aware_model(
        train_csi,
        train_pos,
        val_csi,
        val_pos,
        test_csi,
        test_pos,
        model_config
    )
    
    # Initialize analyzer and report generator
    analyzer = AdvancedAnalyzer(output_dir=os.path.join(RESULTS_DIR, 'final_analysis'))
    report_generator = ReportGenerator(output_dir=os.path.join(RESULTS_DIR, 'reports'))
    
    # Generate predictions for test set
    test_predictions = model.predict(test_csi)
    
    # Generate comprehensive analysis report
    stats = analyzer.generate_comprehensive_report(
        predictions=test_predictions['position'],
        ground_truth=test_pos,
        channel_quality=np.mean(test_predictions['channel_quality'], axis=1),
        timestamps=np.array(history['timestamps'])
    )
    
    # Generate reports
    training_report_path = report_generator.generate_training_report(
        history,
        model_config
    )
    
    analysis_report_path = report_generator.generate_analysis_report(
        analysis_results=stats,
        predictions=test_predictions['position'],
        ground_truth=test_pos,
        channel_quality=np.mean(test_predictions['channel_quality'], axis=1)
    )
    
    # Generate summary report
    summary_report_path = report_generator.generate_summary_report(
        training_report_path=training_report_path,
        analysis_report_path=analysis_report_path
    )
    
    # Log final statistics
    logging.info("\nFinal Analysis Statistics:")
    for metric, value in stats.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Log report paths
    logging.info("\nGenerated Reports:")
    logging.info(f"Training Report: {training_report_path}")
    logging.info(f"Analysis Report: {analysis_report_path}")
    logging.info(f"Summary Report: {summary_report_path}")
    
    # Save model
    model.localization_model.save('models/saved/channel_aware_model')
    
    logging.info("Training and analysis completed successfully!")

if __name__ == '__main__':
    main() 