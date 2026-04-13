import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent


color_range = (-0.2, 0.2)
cmap = plt.colormaps['managua']
norm = colors.TwoSlopeNorm(vmin=color_range[0], vcenter=0.0, vmax=color_range[1])

_LN10 = np.log(10)


def _plot_diagnoal_line(ax, gdp_form):
    xlim = ax.get_xlim()
    # 1:1 line
    ax.plot(xlim, xlim, color='gray', linestyle=':', linewidth=1, zorder=1)

    if gdp_form == 'growth':
        # 1:10 and 10:1 lines
        ax.plot(xlim, [xlim[0] + np.log10(10), xlim[1] + np.log10(10)], color='gray', linestyle=':', linewidth=1, zorder=1)
        ax.plot(xlim, [xlim[0] - np.log10(10), xlim[1] - np.log10(10)], color='gray', linestyle=':', linewidth=1, zorder=1)
    elif gdp_form == 'level':
        # 1:2, 1:4, 2:1, and 4:1 lines
        ax.plot(xlim, [xlim[0] + np.log10(2), xlim[1] + np.log10(2)], color='gray', linestyle=':', linewidth=1, zorder=1)
        ax.plot(xlim, [xlim[0] + np.log10(4), xlim[1] + np.log10(4)], color='gray', linestyle=':', linewidth=1, zorder=1)
        ax.plot(xlim, [xlim[0] - np.log10(2), xlim[1] - np.log10(2)], color='gray', linestyle=':', linewidth=1, zorder=1)
        ax.plot(xlim, [xlim[0] - np.log10(4), xlim[1] - np.log10(4)], color='gray', linestyle=':', linewidth=1, zorder=1)
    # Horizontal and vertical lines
    ax.axhline(0, color='black', linestyle='--', linewidth=1, zorder=5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=5)


def plot_this_panel(df_model_results, case_name, fig, ax):

    X = df_model_results['model_log10ratio']
    Y = df_model_results[f'{case_name}_log10ratio']

    areas = df_model_results['area']
    scatter_sizes = areas / np.max(areas) * 500

    ratio_y_x = 10.0 ** (Y - X)
    color_vals = np.log10(ratio_y_x)

    ax.scatter(X, Y, s=scatter_sizes, c=color_vals, cmap=cmap, norm=norm, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Compute ratio-percent ticks from data
    all_log10 = np.concatenate([X.values, Y.values])
    all_log10 = all_log10[np.isfinite(all_log10)]
    all_ratios = 10.0 ** all_log10
    bounds_ln, ticks_ln, pct_labels = get_axis_bounds_and_ticks_ratio_pct(
        [all_ratios.min(), all_ratios.max()])
    # Convert from ln space to log10 space
    bounds = [b / _LN10 for b in bounds_ln]
    ticks = [t / _LN10 for t in ticks_ln]
    labels = [format_percent(p) for p in pct_labels]

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_aspect('equal')

    gdp_form = 'growth' if 'growth' in case_name else 'level'
    _plot_diagnoal_line(ax, gdp_form)

    ax.set_xlabel('Empirical GPP change  (full vs BGC)')
    ax.set_ylabel('Model GPP change  (full vs BGC)')


def fig3_plot_scatter():

    df_results = pd.read_csv(f'./data/input/figure3_data.csv')

    model_list = df_results['model_name'].unique()
    n_models = len(model_list)
    fig, axs = plt.subplots(n_models, 2, figsize=(9, 10), sharex='col', sharey='col', constrained_layout=True)
    for model_idx, model_i in enumerate(model_list):
        df_model_results = df_results[df_results['model_name'] == model_i]
        plot_this_panel(df_model_results, 'burke_level',  fig, axs[model_idx, 0])
        plot_this_panel(df_model_results, 'newell_level',   fig, axs[model_idx, 1])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.02, shrink=0.5, label='log10(model / empirical)')

    fig.savefig('./data/output/Fig3_scatter.pdf', dpi=300)
    plt.close(fig)
