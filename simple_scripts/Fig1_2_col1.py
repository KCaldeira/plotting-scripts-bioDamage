import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent
from amount_arcsinh_ticks import get_axis_bounds_and_ticks_arcsinh, format_amount

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'

def plot_this_panel(df_model_results, case_name, ax, color, yscale):

    years = df_model_results['years']

    raw_model = df_model_results['model_cumulative'].values
    raw_central = df_model_results[case_name].values
    raw_5 = df_model_results[case_name + '_5'].values
    raw_25 = df_model_results[case_name + '_25'].values
    raw_75 = df_model_results[case_name + '_75'].values
    raw_95 = df_model_results[case_name + '_95'].values

    if yscale == 'arcsinh':
        model_projection = np.arcsinh(raw_model)
        empirical_central = np.arcsinh(raw_central)
        empirical_5 = np.arcsinh(raw_5)
        empirical_25 = np.arcsinh(raw_25)
        empirical_75 = np.arcsinh(raw_75)
        empirical_95 = np.arcsinh(raw_95)
    else:  # ratio
        model_projection = np.log(1 + raw_model / 100)
        empirical_central = np.log(1 + raw_central / 100)
        empirical_5 = np.log(1 + raw_5 / 100)
        empirical_25 = np.log(1 + raw_25 / 100)
        empirical_75 = np.log(1 + raw_75 / 100)
        empirical_95 = np.log(1 + raw_95 / 100)

    ax.plot(years, model_projection,  color='black', linewidth=2.0, linestyle='-', zorder=10)
    ax.plot(years, empirical_central, color=color,   linewidth=2.0, linestyle='-', zorder=10)
    ax.fill_between(years, empirical_5, empirical_95, facecolor=color, edgecolor='none', alpha=0.1)
    ax.fill_between(years, empirical_25, empirical_75, facecolor=color, edgecolor='none', alpha=0.3)
    ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=5)

    # Set axis ticks using utilities
    all_raw = np.concatenate([raw_model, raw_central, raw_5, raw_95])
    all_raw = all_raw[np.isfinite(all_raw)]

    if yscale == 'arcsinh':
        bounds, ticks_vals, amount_labels = get_axis_bounds_and_ticks_arcsinh(
            [all_raw.min(), all_raw.max()], scale=1.0)
        ax.set_ylim(bounds)
        ax.set_yticks(ticks_vals)
        ax.set_yticklabels([format_amount(a) for a in amount_labels])
    else:  # ratio
        ratios = 1 + all_raw / 100
        ratios = ratios[ratios > 0]
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
        plot_this_panel(df_model_results, 'burke_growth', axs[0, 0], 'firebrick', 'arcsinh')
        plot_this_panel(df_model_results, 'burke_level', axs[0, 1], 'royalblue', 'ratio')
        plot_this_panel(df_model_results, 'newell_growth', axs[1, 0], 'firebrick', 'arcsinh')
        plot_this_panel(df_model_results, 'newell_level', axs[1, 1], 'royalblue', 'ratio')
        fig.savefig(f'./data/output/Fig1_2_col1_{plot_type}_{model_i}.pdf', dpi=300)
        plt.close(fig)
