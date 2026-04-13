import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'
_RATIO_LIMIT = 1e-8
_TS_FIG_WIDTH_IN_PER_COL = 2
_TS_BOX_POS_MODEL = 0.24
_TS_BOX_POS_EMP = 0.76
_TS_XLIM = (-0.05, 1.05)

def draw_distribution(ax, data, pos, color):

    y = np.asarray(data, dtype=float)
    y = np.log(np.clip(1 + y / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT))
    p5, p25, p50, p75, p95 = np.percentile(y, [5, 25, 50, 75, 95])

    box_hw = 0.10
    cap_hw = 0.11
    box_h = max(p75 - p25, 1e-9)
    ax.add_patch(mpatches.Rectangle((pos - box_hw, p25), 2 * box_hw, box_h, facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.38, zorder=4))
    ax.plot([pos, pos], [p5, p25], color=color, linewidth=1.4, solid_capstyle='round', zorder=3)
    ax.plot([pos, pos], [p75, p95], color=color, linewidth=1.4, solid_capstyle='round', zorder=3)
    ax.plot([pos - cap_hw, pos + cap_hw], [p5, p5], color=color, linewidth=1.4, zorder=3)
    ax.plot([pos - cap_hw, pos + cap_hw], [p95, p95], color=color, linewidth=1.4, zorder=3)
    ax.scatter([pos], [p50], color='white', s=28, zorder=6, edgecolors=color, linewidths=1.2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlim(_TS_XLIM[0], _TS_XLIM[1])
    ax.set_xticks([_TS_BOX_POS_MODEL, _TS_BOX_POS_EMP])
    ax.set_xticklabels(['M', 'E'])
    ax.set_ylabel('Cumulative d-b  (%)')

def plot_each_panel(ax, df_model_results, emp_cumsum, color):
    model_cumsum = df_model_results['model_cumulative']
    emp_data = df_model_results[emp_cumsum]
    draw_distribution(ax, model_cumsum, _TS_BOX_POS_MODEL, 'gray')
    draw_distribution(ax, emp_data, _TS_BOX_POS_EMP, color)

    # Set axis ticks using utilities
    all_raw = np.concatenate([model_cumsum.values, emp_data.values])
    all_raw = all_raw[np.isfinite(all_raw)]
    ratios = np.clip(1 + all_raw / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT)
    bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ratio_pct(
        [ratios.min(), ratios.max()])
    ax.set_ylim(bounds)
    ax.set_yticks(ticks_vals)
    ax.set_yticklabels([format_percent(p) for p in pct_labels])


def plot_figure1_2_col2(plot_type):

    df_results = pd.read_csv(f'./data/input/growth_rate_country_boxplotLikeDistribution.csv')
    model_list = df_results['model_name'].unique()

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)

    for model_i in models_to_plot:
        df_model_results = df_results[df_results['model_name'] == model_i]
        fig, axs = plt.subplots(2, 2, figsize=(6, 8), sharex=True, constrained_layout=True)
        plot_each_panel(axs[0, 0], df_model_results, 'burke_growth', 'firebrick')
        plot_each_panel(axs[0, 1], df_model_results, 'burke_level', 'royalblue')
        plot_each_panel(axs[1, 0], df_model_results, 'newell_growth', 'firebrick')
        plot_each_panel(axs[1, 1], df_model_results, 'newell_level', 'royalblue')
        fig.savefig(f'./data/output/Fig1_2_col2_{plot_type}_{model_i}.pdf', dpi=300)
        plt.close(fig)
