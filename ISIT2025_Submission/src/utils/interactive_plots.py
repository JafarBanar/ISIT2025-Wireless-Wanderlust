import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import logging

class InteractivePlotter:
    """Generate interactive plots using Plotly."""
    
    def __init__(self, output_dir: str = 'results/reports/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_plot(self, fig: go.Figure, name: str) -> str:
        """Save a plot and return its relative path."""
        # Save as HTML for interactive viewing
        html_path = self.output_dir / f"{name}.html"
        fig.write_html(str(html_path))
        
        # Save as JSON for embedding in reports
        json_path = self.output_dir / f"{name}.json"
        fig.write_json(str(json_path))
        
        return str(html_path)
    
    def plot_training_metrics(self, train_losses: list, val_losses: list,
                            channel_qualities: list, epochs: list) -> str:
        """Create interactive training metrics plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training and Validation Losses', 'Channel Quality Over Time'),
            vertical_spacing=0.15
        )
        
        # Add training and validation losses
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name='Train Loss',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name='Val Loss',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Add channel quality
        fig.add_trace(
            go.Scatter(x=epochs, y=channel_qualities, name='Channel Quality',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Training Metrics",
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        
        return self._save_plot(fig, 'training_metrics')
    
    def plot_error_analysis(self, errors: np.ndarray, channel_quality: np.ndarray,
                          predictions: np.ndarray, ground_truth: np.ndarray) -> str:
        """Create interactive error analysis plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Distribution', 'Channel Quality vs Error',
                          'Trajectory Comparison', 'Error Heatmap'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                  [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=errors, name='Error Distribution',
                        nbinsx=50, showlegend=False),
            row=1, col=1
        )
        
        # Channel quality vs error
        fig.add_trace(
            go.Scatter(x=channel_quality, y=errors, mode='markers',
                      name='Channel Quality vs Error', marker=dict(size=8, opacity=0.5)),
            row=1, col=2
        )
        
        # Add trend line
        z = np.polyfit(channel_quality, errors, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=channel_quality, y=p(channel_quality),
                      name='Trend Line', line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # Trajectory comparison
        fig.add_trace(
            go.Scatter(x=ground_truth[:, 0], y=ground_truth[:, 1],
                      name='Ground Truth', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=predictions[:, 0], y=predictions[:, 1],
                      name='Predictions', line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Error heatmap
        x_bins = np.linspace(min(ground_truth[:, 0]), max(ground_truth[:, 0]), 20)
        y_bins = np.linspace(min(ground_truth[:, 1]), max(ground_truth[:, 1]), 20)
        H, xedges, yedges = np.histogram2d(
            ground_truth[:, 0], ground_truth[:, 1],
            bins=[x_bins, y_bins],
            weights=errors
        )
        
        fig.add_trace(
            go.Heatmap(z=H.T, x=xedges[:-1], y=yedges[:-1],
                      colorscale='Viridis', name='Error Heatmap'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Error Analysis",
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Error (m)", row=1, col=1)
        fig.update_xaxes(title_text="Channel Quality", row=1, col=2)
        fig.update_xaxes(title_text="X Position (m)", row=2, col=1)
        fig.update_xaxes(title_text="X Position (m)", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Error (m)", row=1, col=2)
        fig.update_yaxes(title_text="Y Position (m)", row=2, col=1)
        fig.update_yaxes(title_text="Y Position (m)", row=2, col=2)
        
        return self._save_plot(fig, 'error_analysis')
    
    def plot_channel_analysis(self, channel_stats: dict, channel_weights: np.ndarray) -> str:
        """Create interactive channel analysis plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Channel Quality Over Time', 'Channel Weights',
                          'Collision Rate', 'Channel Quality Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                  [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Channel quality over time
        fig.add_trace(
            go.Scatter(y=channel_stats['channel_quality'], name='Channel Quality',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Channel weights
        fig.add_trace(
            go.Bar(x=list(range(len(channel_weights))), y=channel_weights,
                  name='Channel Weights'),
            row=1, col=2
        )
        
        # Collision rate
        fig.add_trace(
            go.Scatter(y=channel_stats['collision_rate'], name='Collision Rate',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # Channel quality distribution
        fig.add_trace(
            go.Histogram(x=channel_stats['channel_quality'], name='Quality Distribution',
                        nbinsx=30),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Channel Analysis",
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_xaxes(title_text="Channel Index", row=1, col=2)
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_xaxes(title_text="Quality Score", row=2, col=2)
        
        fig.update_yaxes(title_text="Quality Score", row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=2)
        fig.update_yaxes(title_text="Collision Rate", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return self._save_plot(fig, 'channel_analysis') 