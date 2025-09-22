#!/usr/bin/env python3
"""
Create all figures with perfectly matched bar sizes
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set IEEE-compliant style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8
})

# Create output directory
output_dir = Path("ISIT2025_Final_Paper/figures")
output_dir.mkdir(parents=True, exist_ok=True)

def create_perfectly_matched_figures():
    """Create all figures with perfectly matched bar sizes"""
    
    # Data
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4']
    mae_values = [0.4320, 0.4721, 0.3237, 0.2517]
    r90_values = [0.8841, 1.0018, 0.8016, 0.6750]
    transmission_rates = [100, 100, 30.02, 25]
    combined_scores = [0.6177, 0.6310, 0.4867, 0.3787]
    
    # IEEE-compliant dimensions
    fig_width = 3.45
    fig_height = 2.6
    dpi = 600
    
    # Colors
    colors = ['#4472C4', '#E7A614', '#70AD47', '#FF6B6B']
    
    # Common settings for all figures
    bar_width = 0.6
    y_scale_factor = 1.2
    
    # (a) MAE Performance Comparison
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    bars1 = ax1.bar(tasks, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8, width=bar_width)
    ax1.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) MAE Performance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_ylim(0, max(mae_values) * y_scale_factor)
    
    # Add value labels
    for bar, value in zip(bars1, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(mae_values) * 0.05,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_perfect.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # (b) R90 Performance Comparison
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    bars2 = ax2.bar(tasks, r90_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8, width=bar_width)
    ax2.set_ylabel('R90 (m)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) R90 Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_ylim(0, max(r90_values) * y_scale_factor)
    
    # Add value labels
    for bar, value in zip(bars2, r90_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(r90_values) * 0.05,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r90_perfect.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # (c) Transmission Rate Analysis - PERFECTLY MATCHED
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    bars3 = ax3.bar(tasks, transmission_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8, width=bar_width)
    ax3.set_ylabel('Transmission Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Transmission Efficiency', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.set_ylim(0, max(transmission_rates) * y_scale_factor)  # Same scaling as others
    
    # Add value labels
    for bar, value in zip(bars3, transmission_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(transmission_rates) * 0.05,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight bandwidth reduction
    ax3.annotate('70% Reduction', xy=(2, 30), xytext=(2.5, 60),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, color='red', fontweight='bold')
    ax3.annotate('75% Reduction', xy=(3, 25), xytext=(3.5, 50),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transmission_perfect.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # (d) Combined Score Evaluation
    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    bars4 = ax4.bar(tasks, combined_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8, width=bar_width)
    ax4.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Overall Performance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.set_ylim(0, max(combined_scores) * y_scale_factor)
    
    # Add value labels
    for bar, value in zip(bars4, combined_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(combined_scores) * 0.05,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_perfect.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print("✅ Created PERFECTLY MATCHED figures:")
    print("📏 All figures use:")
    print(f"   - Bar width: {bar_width}")
    print(f"   - Y-axis scaling: {y_scale_factor}x max value")
    print(f"   - Label positioning: 5% of max value above bars")
    print(f"   - Figure size: {fig_width}\" x {fig_height}\"")
    print(f"   - DPI: {dpi}")
    print("\n📁 PERFECT files:")
    print(f"   - {output_dir / 'mae_perfect.png'}")
    print(f"   - {output_dir / 'r90_perfect.png'}")
    print(f"   - {output_dir / 'transmission_perfect.png'}")
    print(f"   - {output_dir / 'combined_perfect.png'}")

if __name__ == "__main__":
    create_perfectly_matched_figures()
