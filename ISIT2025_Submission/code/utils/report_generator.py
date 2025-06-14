import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .pdf_generator import PDFGenerator
from .interactive_plots import InteractivePlotter
from .latex_generator import LaTeXGenerator

class ReportGenerator:
    """Generate comprehensive reports for model training and analysis results."""
    
    def __init__(self, output_dir: str = 'results/reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = self.output_dir / f"report_{self.timestamp}"
        self.report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.report_dir / "figures").mkdir(exist_ok=True)
        (self.report_dir / "data").mkdir(exist_ok=True)
        (self.report_dir / "js").mkdir(exist_ok=True)
        
        # Initialize PDF generator
        self.pdf_generator = PDFGenerator(output_dir=str(self.report_dir / "pdf"))
        
        # Initialize plotters and generators
        self.plotter = InteractivePlotter(str(self.report_dir / "figures"))
        self.latex_generator = LaTeXGenerator(str(self.report_dir / 'latex'))
        
        # Create JavaScript utilities
        self._create_js_utilities()
    
    def _create_js_utilities(self):
        """Create JavaScript utility functions for interactive features."""
        js_content = """
        function toggleSection(sectionId) {
            const section = document.getElementById(sectionId);
            const button = document.querySelector(`button[onclick="toggleSection('${sectionId}')"]`);
            if (section.style.display === 'none') {
                section.style.display = 'block';
                button.textContent = 'Hide';
            } else {
                section.style.display = 'none';
                button.textContent = 'Show';
            }
        }
        
        function filterTable(tableId, inputId) {
            const input = document.getElementById(inputId);
            const filter = input.value.toUpperCase();
            const table = document.getElementById(tableId);
            const tr = table.getElementsByTagName("tr");
            
            for (let i = 1; i < tr.length; i++) {
                const td = tr[i].getElementsByTagName("td");
                let found = false;
                for (let j = 0; j < td.length; j++) {
                    if (td[j].textContent.toUpperCase().indexOf(filter) > -1) {
                        found = true;
                        break;
                    }
                }
                tr[i].style.display = found ? "" : "none";
            }
        }
        
        function sortTable(tableId, column) {
            const table = document.getElementById(tableId);
            const tbody = table.getElementsByTagName('tbody')[0];
            const rows = Array.from(tbody.getElementsByTagName('tr'));
            
            const sortedRows = rows.sort((a, b) => {
                const aValue = a.getElementsByTagName('td')[column].textContent;
                const bValue = b.getElementsByTagName('td')[column].textContent;
                return aValue.localeCompare(bValue, undefined, {numeric: true});
            });
            
            tbody.innerHTML = '';
            sortedRows.forEach(row => tbody.appendChild(row));
        }
        """
        
        js_path = self.report_dir / "js" / "utils.js"
        with open(js_path, 'w') as f:
            f.write(js_content)
    
    def _save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save a figure and return its relative path."""
        fig_path = self.report_dir / "figures" / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return f"figures/{name}.png"
    
    def _save_data(self, data: Any, name: str) -> str:
        """Save data and return its relative path."""
        data_path = self.report_dir / "data" / f"{name}.npy"
        np.save(data_path, data)
        return f"data/{name}.npy"
    
    def generate_training_report(self, training_history: Dict[str, Any], 
                               model_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate a training report with interactive plots."""
        # Generate interactive plots
        plot_paths = self.plotter.plot_training_metrics(
            training_history['train_losses'],
            training_history['val_losses'],
            training_history['channel_stats']
        )
        
        # Save training data
        data_paths = {
            'train_losses': self._save_data(training_history['train_losses'], 'train_losses'),
            'val_losses': self._save_data(training_history['val_losses'], 'val_losses'),
            'channel_stats': self._save_data(training_history['channel_stats'], 'channel_stats')
        }
        
        # Create HTML content
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .plot {{ width: 100%; height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .search-box {{ margin: 10px 0; padding: 5px; width: 200px; }}
        .toggle-btn {{ margin: 10px 0; padding: 5px 10px; cursor: pointer; }}
    </style>
    <script src="js/utils.js"></script>
</head>
<body>
    <h1>Training Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="section">
        <button class="toggle-btn" onclick="toggleSection('config')">
            <i class="fas fa-cog"></i> Training Configuration
        </button>
        <div id="config" style="display: none;">
            <pre>{json.dumps(model_config, indent=2)}</pre>
        </div>
    </div>
    
    <div class="section">
        <h2>Training Metrics</h2>
        <div class="plot" id="training_metrics"></div>
    </div>
    
    <div class="section">
        <h2>Channel Analysis</h2>
        <div class="plot" id="channel_analysis"></div>
    </div>
    
    <div class="section">
        <h2>Detailed Metrics</h2>
        <input type="text" class="search-box" placeholder="Search metrics..." onkeyup="filterTable('metrics_table', this.value)">
        <table id="metrics_table">
            <tr>
                <th onclick="sortTable('metrics_table', 0)">Metric</th>
                <th onclick="sortTable('metrics_table', 1)">Value</th>
            </tr>
            <tr>
                <td>Final Training Loss</td>
                <td>{training_history['train_losses'][-1]:.4f}</td>
            </tr>
            <tr>
                <td>Final Validation Loss</td>
                <td>{training_history['val_losses'][-1]:.4f}</td>
            </tr>
            <tr>
                <td>Average Channel Quality</td>
                <td>{np.mean([stats['channel_quality'] for stats in training_history['channel_stats']]):.4f}</td>
            </tr>
            <tr>
                <td>Training Duration</td>
                <td>{training_history['timestamps'][-1] - training_history['timestamps'][0]:.2f} seconds</td>
            </tr>
        </table>
    </div>
    
    <script>
        // Load interactive plots
        fetch('{plot_paths["training_metrics"]}')
            .then(response => response.json())
            .then(data => Plotly.newPlot('training_metrics', data.data, data.layout));
        
        fetch('{plot_paths["channel_analysis"]}')
            .then(response => response.json())
            .then(data => Plotly.newPlot('channel_analysis', data.data, data.layout));
    </script>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.report_dir / 'training_report.html'
        with open(html_path, 'w') as f:
            f.write(content)
        
        # Generate LaTeX report
        latex_path = self.latex_generator.generate_training_report(
            training_history,
            model_config,
            {k: str(self.report_dir / v) for k, v in plot_paths.items()}
        )
        
        return {
            'html': str(html_path),
            'latex': latex_path,
            'plots': plot_paths,
            'data': data_paths
        }
    
    def generate_analysis_report(self, analysis_results: Dict[str, Any],
                               predictions: np.ndarray,
                               ground_truth: np.ndarray,
                               channel_quality: np.ndarray) -> Dict[str, str]:
        """Generate an analysis report with interactive plots."""
        # Generate interactive plots
        plot_paths = self.plotter.plot_error_analysis(
            predictions,
            ground_truth,
            channel_quality
        )
        
        # Save analysis data
        data_paths = {
            'predictions': self._save_data(predictions, 'predictions'),
            'ground_truth': self._save_data(ground_truth, 'ground_truth'),
            'channel_quality': self._save_data(channel_quality, 'channel_quality')
        }
        
        # Create HTML content
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .plot {{ width: 100%; height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .search-box {{ margin: 10px 0; padding: 5px; width: 200px; }}
        .toggle-btn {{ margin: 10px 0; padding: 5px 10px; cursor: pointer; }}
    </style>
    <script src="js/utils.js"></script>
</head>
<body>
    <h1>Analysis Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="section">
        <h2>Error Analysis</h2>
        <div class="plot" id="error_analysis"></div>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <input type="text" class="search-box" placeholder="Search metrics..." onkeyup="filterTable('metrics_table', this.value)">
        <table id="metrics_table">
            <tr>
                <th onclick="sortTable('metrics_table', 0)">Metric</th>
                <th onclick="sortTable('metrics_table', 1)">Value</th>
            </tr>
"""
        
        # Add metrics
        for metric, value in analysis_results.items():
            content += f"""            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td>{value:.4f}</td>
            </tr>
"""
        
        content += """        </table>
    </div>
    
    <div class="section">
        <button class="toggle-btn" onclick="toggleSection('error_stats')">
            <i class="fas fa-chart-bar"></i> Error Statistics
        </button>
        <div id="error_stats" style="display: none;">
            <div class="plot" id="error_distribution"></div>
        </div>
    </div>
    
    <div class="section">
        <button class="toggle-btn" onclick="toggleSection('channel_analysis')">
            <i class="fas fa-signal"></i> Channel Quality Analysis
        </button>
        <div id="channel_analysis" style="display: none;">
            <div class="plot" id="channel_quality"></div>
        </div>
    </div>
    
    <script>
        // Load interactive plots
        fetch('{plot_paths["error_analysis"]}')
            .then(response => response.json())
            .then(data => Plotly.newPlot('error_analysis', data.data, data.layout));
        
        fetch('{plot_paths["error_distribution"]}')
            .then(response => response.json())
            .then(data => Plotly.newPlot('error_distribution', data.data, data.layout));
        
        fetch('{plot_paths["channel_quality"]}')
            .then(response => response.json())
            .then(data => Plotly.newPlot('channel_quality', data.data, data.layout));
    </script>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.report_dir / 'analysis_report.html'
        with open(html_path, 'w') as f:
            f.write(content)
        
        # Generate LaTeX report
        latex_path = self.latex_generator.generate_analysis_report(
            analysis_results,
            predictions,
            ground_truth,
            channel_quality,
            {k: str(self.report_dir / v) for k, v in plot_paths.items()}
        )
        
        return {
            'html': str(html_path),
            'latex': latex_path,
            'plots': plot_paths,
            'data': data_paths
        }
    
    def generate_summary_report(self, training_report: Dict[str, str],
                              analysis_report: Dict[str, str]) -> Dict[str, str]:
        """Generate a summary report combining training and analysis results."""
        # Create HTML content
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Training Summary</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .report-link {{ display: block; margin: 10px 0; padding: 10px; background-color: #f5f5f5; text-decoration: none; color: #333; }}
        .report-link:hover {{ background-color: #e5e5e5; }}
    </style>
</head>
<body>
    <h1>Model Training Summary</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="section">
        <h2>Reports</h2>
        <a href="{training_report['html']}" class="report-link">
            <i class="fas fa-file-alt"></i> Training Report
        </a>
        <a href="{analysis_report['html']}" class="report-link">
            <i class="fas fa-chart-line"></i> Analysis Report
        </a>
    </div>
    
    <div class="section">
        <h2>Export Formats</h2>
        <a href="{training_report['latex']}" class="report-link">
            <i class="fas fa-file-pdf"></i> LaTeX Training Report
        </a>
        <a href="{analysis_report['latex']}" class="report-link">
            <i class="fas fa-file-pdf"></i> LaTeX Analysis Report
        </a>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.report_dir / 'summary_report.html'
        with open(html_path, 'w') as f:
            f.write(content)
        
        # Generate LaTeX summary report
        latex_path = self.latex_generator.generate_summary_report(
            training_report['latex'],
            analysis_report['latex']
        )
        
        return {
            'html': str(html_path),
            'latex': latex_path
        } 