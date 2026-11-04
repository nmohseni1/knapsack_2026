"""Read and plot cached final results without simulation or post-processing."""
import json
from pathlib import Path


def load_final_results(run_directory):
    return json.loads((Path(run_directory) / 'analysis' / 'summary.json').read_text())


def plot_final_results(run_directory, metric='best_value'):
    """Read cached summaries only. Does not write files or contact a service.

    Metrics: best_value, ar_top8000, ar_top20000, ar_all_feasible,
    feasible_probability. AR denominators are saved Gurobi incumbents, not
    certified optima. Empty feasible sets appear as gaps, not zero quality.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    data = load_final_results(run_directory)
    labels = {'best_value': 'Best feasible value',
              'ar_top8000': 'Top 8,000 feasible shots / reference',
              'ar_top20000': 'Top 20,000 feasible shots / reference',
              'ar_all_feasible': 'Mean feasible value / reference',
              'feasible_probability': 'Feasible-shot fraction'}
    if metric not in labels:
        raise ValueError(f'Choose a metric from {list(labels)}')
    stages = [('before', 'Without post-processing'),
              ('repair', 'Feasibility repair'),
              ('completion', 'Repair + greedy filling')]
    colours = ['tab:olive', 'tab:cyan', 'tab:purple']
    markers = ['o', 's', '^']
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    for ax, (stage, title) in zip(axes, stages):
        all_p = set()
        for (label, series), colour, marker in zip(data['series'].items(), colours, markers):
            depths = sorted(map(int, series))
            all_p.update(depths)
            vals = [series[str(p)][stage][metric] for p in depths]
            ax.plot(depths, [np.nan if v is None else v for v in vals],
                    color=colour, marker=marker, markersize=8, linewidth=2, label=label)
        warm = data['baselines']['warm_start'][stage][metric]
        if warm is not None:
            ax.axhline(warm, color='tab:green', linestyle='--', label='Warm start')
        if metric == 'best_value':
            ax.axhline(data['reference']['value'], color='black', linestyle='--', label='Gurobi reference')
        elif metric.startswith('ar_'):
            ax.axhline(1.0, color='black', linestyle='--', label='Saved reference = 1')
        ax.set_title(title, fontsize=19)
        ax.set_xlabel('QAOA depth p', fontsize=24)
        ax.set_xticks(sorted(all_p))
        ax.tick_params(axis='both', labelsize=17)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=13, frameon=False)
    axes[0].set_ylabel(labels[metric], fontsize=21)
    fig.tight_layout()
    return fig, axes
