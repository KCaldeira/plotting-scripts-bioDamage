import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'
_RATIO_LIMIT = 1e-8
_CD_FIG_WIDTH_IN = 6.3
_BASELINE_TAS_YEAR_LO = 2004
_BASELINE_TAS_YEAR_HI = 2014
_CD_IN_PER_COUNTRY = 0.14
_CD_FIG_WIDTH_MAX_IN = 56.0
_CD_BOX_HW = 0.18
_CD_CAP_HW = 0.20
_CD_SCATTER_S = 28


def _draw_distribution_country(ax, data_native, pos, color):
    y = np.asarray(data_native, dtype=float)
    y = np.log(np.clip(1 + y / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT))

    model, pCentral, p5, p25, p75, p95 = y[0], y[1], y[2], y[3], y[4], y[5]
    box_h = max(p75 - p25, 1e-9)
    ax.add_patch(
        mpatches.Rectangle(
            (pos - _CD_BOX_HW, p25),
            2 * _CD_BOX_HW,
            box_h,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.38,
            zorder=3,
        )
    )
    ax.plot([pos, pos], [p5, p25], color=color, linewidth=1.4, solid_capstyle='round', zorder=3)
    ax.plot([pos, pos], [p75, p95], color=color, linewidth=1.4, solid_capstyle='round', zorder=3)
    ax.plot([pos - _CD_CAP_HW, pos + _CD_CAP_HW], [p5, p5], color=color, linewidth=1.4, zorder=3)
    ax.plot([pos - _CD_CAP_HW, pos + _CD_CAP_HW], [p95, p95], color=color, linewidth=1.4, zorder=3)
    ax.scatter([pos], [pCentral], color='white', s=_CD_SCATTER_S, zorder=5, edgecolors=color, linewidths=1.2)
    ax.scatter([pos], [model], color='black', s=5, zorder=8, marker='D')


def plot_each_panel(ax, df_model_results, emp_cumsum, color):

    country_selected = df_model_results['region_list']
    n_sel = len(country_selected)

    model_cumsum = np.array(df_model_results['model_cumulative'].values.tolist())
    emp_cumsum_central = np.array(df_model_results[emp_cumsum].values.tolist())
    emp_cumsum_5 = np.array(df_model_results[f'{emp_cumsum}_5'].values.tolist())
    emp_cumsum_25 = np.array(df_model_results[f'{emp_cumsum}_25'].values.tolist())
    emp_cumsum_75 = np.array(df_model_results[f'{emp_cumsum}_75'].values.tolist())
    emp_cumsum_95 = np.array(df_model_results[f'{emp_cumsum}_95'].values.tolist())

    for ci in range(n_sel):
        col_y = np.array([model_cumsum[ci], emp_cumsum_central[ci], emp_cumsum_5[ci], emp_cumsum_25[ci], emp_cumsum_75[ci], emp_cumsum_95[ci]])
        _draw_distribution_country(ax, col_y, ci, color)
    ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=4)

    # Set axis ticks using utilities
    all_raw = np.concatenate([model_cumsum, emp_cumsum_central, emp_cumsum_5, emp_cumsum_95])
    all_raw = all_raw[np.isfinite(all_raw)]
    ratios = np.clip(1 + all_raw / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT)
    bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ratio_pct(
        [ratios.min(), ratios.max()])
    ax.set_ylim(bounds)
    ax.set_yticks(ticks_vals)
    ax.set_yticklabels([format_percent(p) for p in pct_labels])

def plot_figure1_2_col3(plot_type):

    df_results = pd.read_csv(f'./data/input/growth_rate_country_barPlotDistribution.csv')
    model_list = df_results['model_name'].unique()

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)

    for model_i in models_to_plot:

        model_i = 'CNRM-ESM2-1'

        df_model_results = df_results[df_results['model_name'] == model_i]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
        plot_each_panel(axs[0, 0], df_model_results, 'burke_growth', 'firebrick')
        plot_each_panel(axs[0, 1], df_model_results, 'burke_level', 'royalblue')
        plot_each_panel(axs[1, 0], df_model_results, 'newell_growth', 'firebrick')
        plot_each_panel(axs[1, 1], df_model_results, 'newell_level', 'royalblue')
        fig.savefig(f'./data/output/Fig1_2_col3_{plot_type}_{model_i}.pdf', dpi=300)
        plt.close(fig)
