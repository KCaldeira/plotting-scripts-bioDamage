import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'
_RATIO_LIMIT = 1e-8

def plot_this_panel(df_model_results, case_name, ax, color):

    years = df_model_results['years']

    raw_model = df_model_results['model_cumulative'].values
    raw_central = df_model_results[case_name].values
    raw_5 = df_model_results[case_name + '_5'].values
    raw_25 = df_model_results[case_name + '_25'].values
    raw_75 = df_model_results[case_name + '_75'].values
    raw_95 = df_model_results[case_name + '_95'].values

    def _pct_to_ln(pct):
        ratio = np.clip(1 + pct / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT)
        return np.log(ratio)

    model_projection = _pct_to_ln(raw_model)
    empirical_central = _pct_to_ln(raw_central)
    empirical_5 = _pct_to_ln(raw_5)
    empirical_25 = _pct_to_ln(raw_25)
    empirical_75 = _pct_to_ln(raw_75)
    empirical_95 = _pct_to_ln(raw_95)

    ax.plot(years, model_projection,  color='black', linewidth=2.0, linestyle='-', zorder=10)
    ax.plot(years, empirical_central, color=color,   linewidth=2.0, linestyle='-', zorder=10)
    ax.fill_between(years, empirical_5, empirical_95, facecolor=color, edgecolor='none', alpha=0.1)
    ax.fill_between(years, empirical_25, empirical_75, facecolor=color, edgecolor='none', alpha=0.3)
    ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=5)

    # Set axis ticks using utilities
    all_raw = np.concatenate([raw_model, raw_central, raw_5, raw_95])
    all_raw = all_raw[np.isfinite(all_raw)]
    ratios = np.clip(1 + all_raw / 100, _RATIO_LIMIT, 1.0 / _RATIO_LIMIT)
    bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ratio_pct(
        [ratios.min(), ratios.max()])
    ax.set_ylim(bounds)
    ax.set_yticks(ticks_vals)
    ax.set_yticklabels([format_percent(p) for p in pct_labels])


def plot_figure1_2_col1(plot_type):

    df_results = pd.read_csv(f'./data/input/growth_rate_global_mean_timeSeries.csv')
    model_list = df_results['model_name'].unique()

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)

    for model_i in models_to_plot:
        df_model_results = df_results[df_results['model_name'] == model_i]
        df_model_results = df_model_results.sort_values(by='years')
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
        plot_this_panel(df_model_results, 'burke_growth', axs[0, 0], 'firebrick')
        plot_this_panel(df_model_results, 'burke_level', axs[0, 1], 'royalblue')
        plot_this_panel(df_model_results, 'newell_growth', axs[1, 0], 'firebrick')
        plot_this_panel(df_model_results, 'newell_level', axs[1, 1], 'royalblue')
        fig.savefig(f'./data/output/Fig1_2_col1_{plot_type}_{model_i}.pdf', dpi=300)
        plt.close(fig)
