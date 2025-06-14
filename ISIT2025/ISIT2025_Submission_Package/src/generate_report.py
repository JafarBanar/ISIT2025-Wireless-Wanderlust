import os

RESULTS_DIR = 'results/competition_models'
REPORT_PATH = os.path.join(RESULTS_DIR, 'summary_report.md')
MODEL_NAMES = ['basic_localization', 'improved_localization', 'optimized_localization']


def read_metrics(model_name):
    metrics_path = os.path.join(RESULTS_DIR, f'{model_name}_metrics.txt')
    if not os.path.exists(metrics_path):
        return None
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                metrics[k.strip()] = v.strip()
    return metrics


def main():
    with open(REPORT_PATH, 'w') as report:
        report.write('# Competition Model Results Summary\n\n')
        for model_name in MODEL_NAMES:
            report.write(f'## {model_name.replace("_", " ").title()}\n')
            metrics = read_metrics(model_name)
            if metrics:
                for k, v in metrics.items():
                    report.write(f'- **{k}**: {v}\n')
                # Optionally add plot links if available
                plot_path = os.path.join(RESULTS_DIR, f'{model_name}_training_plot.png')
                if os.path.exists(plot_path):
                    report.write(f'![Training Plot]({plot_path})\n')
            else:
                report.write('No metrics found.\n')
            report.write('\n')
        report.write('---\n')
        report.write('Generated automatically by generate_report.py\n')

if __name__ == '__main__':
    main() 