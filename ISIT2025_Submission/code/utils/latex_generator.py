import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import logging

class LaTeXGenerator:
    """Generate LaTeX reports from analysis results."""
    
    def __init__(self, output_dir: str = 'results/reports/latex'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
            '<': r'\textless{}',
            '>': r'\textgreater{}',
        }
        return ''.join(special_chars.get(c, c) for c in str(text))
    
    def _format_number(self, num: float, precision: int = 4) -> str:
        """Format a number for LaTeX output."""
        if isinstance(num, (int, float)):
            return f"{num:.{precision}f}"
        return str(num)
    
    def generate_training_report(self, training_history: dict,
                               model_config: dict,
                               figure_paths: dict) -> str:
        """Generate a LaTeX report for training results."""
        # Create LaTeX content
        content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{xcolor}}
\\usepackage{{listings}}

\\title{{Training Report}}
\\author{{Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}}
\\date{{}}

\\begin{{document}}

\\maketitle

\\section{{Training Configuration}}
\\begin{{lstlisting}}[frame=single]
{json.dumps(model_config, indent=2)}
\\end{{lstlisting}}

\\section{{Training Metrics}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{figure_paths['training_metrics']}}}
    \\caption{{Training and validation losses over time}}
\\end{{figure}}

\\section{{Final Metrics}}
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{lr}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\midrule
        Training Loss & {self._format_number(training_history['train_losses'][-1])} \\\\
        Validation Loss & {self._format_number(training_history['val_losses'][-1])} \\\\
        Channel Quality & {self._format_number(np.mean([stats['channel_quality'] for stats in training_history['channel_stats']]))} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Final training metrics}}
\\end{{table}}

\\section{{Training Duration}}
Total training time: {self._format_number(training_history['timestamps'][-1] - training_history['timestamps'][0], 2)} seconds

\\end{{document}}
"""
        
        # Save LaTeX file
        tex_path = self.output_dir / f"training_report_{self.timestamp}.tex"
        with open(tex_path, 'w') as f:
            f.write(content)
        
        return str(tex_path)
    
    def generate_analysis_report(self, analysis_results: dict,
                               predictions: np.ndarray,
                               ground_truth: np.ndarray,
                               channel_quality: np.ndarray,
                               figure_paths: dict) -> str:
        """Generate a LaTeX report for analysis results."""
        # Calculate errors
        errors = np.sqrt(np.sum(np.square(predictions - ground_truth), axis=1))
        
        # Create LaTeX content
        content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{xcolor}}
\\usepackage{{listings}}

\\title{{Analysis Report}}
\\author{{Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}}
\\date{{}}

\\begin{{document}}

\\maketitle

\\section{{Performance Metrics}}
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{lr}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\midrule
"""
        
        # Add metrics
        for metric, value in analysis_results.items():
            content += f"        {self._escape_latex(metric.replace('_', ' ').title())} & {self._format_number(value)} \\\\\n"
        
        content += """        \\bottomrule
    \\end{tabular}
    \\caption{Performance metrics}
\\end{table}

\\section{{Error Analysis}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{figure_paths['error_analysis']}}}
    \\caption{{Error analysis plots}}
\\end{{figure}}

\\section{{Error Statistics}}
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{lr}}
        \\toprule
        \\textbf{{Statistic}} & \\textbf{{Value}} \\\\
        \\midrule
        Mean Error & {self._format_number(np.mean(errors))} m \\\\
        Median Error & {self._format_number(np.median(errors))} m \\\\
        95th Percentile & {self._format_number(np.percentile(errors, 95))} m \\\\
        99th Percentile & {self._format_number(np.percentile(errors, 99))} m \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Error statistics}}
\\end{{table}}

\\section{{Channel Quality Analysis}}
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{lr}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\midrule
        Average Channel Quality & {self._format_number(np.mean(channel_quality))} \\\\
        Channel Quality Std & {self._format_number(np.std(channel_quality))} \\\\
        Correlation with Error & {self._format_number(np.corrcoef(channel_quality, errors)[0,1])} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Channel quality analysis}}
\\end{{table}}

\\end{{document}}
"""
        
        # Save LaTeX file
        tex_path = self.output_dir / f"analysis_report_{self.timestamp}.tex"
        with open(tex_path, 'w') as f:
            f.write(content)
        
        return str(tex_path)
    
    def generate_summary_report(self, training_report_path: str,
                              analysis_report_path: str) -> str:
        """Generate a LaTeX summary report."""
        content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{xcolor}}

\\title{{Model Training Summary}}
\\author{{Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}}
\\date{{}}

\\begin{{document}}

\\maketitle

\\section{{Reports}}
\\begin{{itemize}}
    \\item \\href{{{training_report_path}}}{{Training Report}}
    \\item \\href{{{analysis_report_path}}}{{Analysis Report}}
\\end{{itemize}}

\\section{{Report Generation}}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

\\end{{document}}
"""
        
        # Save LaTeX file
        tex_path = self.output_dir / f"summary_report_{self.timestamp}.tex"
        with open(tex_path, 'w') as f:
            f.write(content)
        
        return str(tex_path) 