import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'

def plot_this_panel(df_model_results, case_name, ax, color, yscale):

    years = df_model_results['years']

    if yscale == 'arcsinh':
        model_projection = np.arcsinh(df_model_results['model_cumulative'])
        empirical_central = np.arcsinh(df_model_results[case_name])
        empirical_5 = np.arcsinh(df_model_results[case_name + '_5'])
        empirical_25 = np.arcsinh(df_model_results[case_name + '_25'])
        empirical_75 = np.arcsinh(df_model_results[case_name + '_75'])
        empirical_95 = np.arcsinh(df_model_results[case_name + '_95']) 
    else:
        model_projection = df_model_results['model_cumulative']
        empirical_central = df_model_results[case_name]
        empirical_5 = df_model_results[case_name + '_5']
        empirical_25 = df_model_results[case_name + '_25']
        empirical_75 = df_model_results[case_name + '_75']
        empirical_95 = df_model_results[case_name + '_95'] 

    ax.plot(years, model_projection,  color='black', linewidth=2.0, linestyle='-', zorder=10)
    ax.plot(years, empirical_central, color=color,   linewidth=2.0, linestyle='-', zorder=10)
    ax.fill_between(years, empirical_5, empirical_95, facecolor=color, edgecolor='none', alpha=0.1)
    ax.fill_between(years, empirical_25, empirical_75, facecolor=color, edgecolor='none', alpha=0.3)
    ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=5)


def plot_figure1_2_col1(plot_type): 

    df_results = pd.read_csv(f'./simple_scripts/growth_rate_global_mean_timeSeries.csv')
    model_list = df_results['model_name'].unique() 

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)

    for model_i in models_to_plot:
        df_model_results = df_results[df_results['model_name'] == model_i]
        df_model_results = df_model_results.sort_values(by='years')
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='col', constrained_layout=True)
        plot_this_panel(df_model_results, 'burke_growth', axs[0, 0], 'firebrick', 'arcsinh')
        plot_this_panel(df_model_results, 'burke_level', axs[0, 1], 'royalblue', 'linear')
        plot_this_panel(df_model_results, 'newell_growth', axs[1, 0], 'firebrick', 'arcsinh')
        plot_this_panel(df_model_results, 'newell_level', axs[1, 1], 'royalblue', 'linear')
        plt.show() 