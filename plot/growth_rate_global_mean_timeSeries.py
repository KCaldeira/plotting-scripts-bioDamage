import matplotlib.pyplot as plt
import numpy as np

from plot.func_shared_plotting import (
    _pool_bounds,
    _apply_yaxis_growth_arcsinh,
    _apply_yaxis_level_linear,
    _ribbon_aligned_bounds,
)

gdp_form_color = {'growth': 'firebrick', 'level': 'royalblue'}

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'

def plot_this_panel(years, model_projection, input_allCases, ax, color, yscale):

    print ('model simulation', model_projection[-1])
    print ('empirical projection', input_allCases[0, -1], np.percentile(input_allCases, 5,  axis=0)[-1], np.percentile(input_allCases, 95,  axis=0)[-1])

    if yscale == 'linear':
        pass
    elif yscale == 'arcsinh':
        model_projection = np.arcsinh(model_projection)
        input_allCases   = np.arcsinh(input_allCases)
    ax.plot(years, model_projection,     color='black', linewidth=2.0, linestyle='-', zorder=10)
    ax.plot(years, input_allCases[0, :], color=color,   linewidth=2.0, linestyle='-', zorder=10)
    ax.fill_between(years, np.percentile(input_allCases, 5,  axis=0), np.percentile(input_allCases, 95, axis=0), facecolor=color, edgecolor='none', alpha=0.1)
    ax.fill_between(years, np.percentile(input_allCases, 25, axis=0), np.percentile(input_allCases, 75, axis=0), facecolor=color, edgecolor='none', alpha=0.3)
    ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=5)

def _log_excess_to_pct(log_excess):
    """Map log-growth excess X = log g_full − log g_BGC to percent: (e^X − 1)·100."""
    return (np.exp(log_excess) - 1) * 100

def _burke_d_minus_b_series(results_bootstrap_dict, projection_central, bootstrap_num):
    years                     = projection_central['years']
    weights_projection        = projection_central['weights_projection']
    plot_years                = years[1:]
    model_projection_regional = np.array(projection_central['model_simulation_projection'])
    model_references_regional = np.array(projection_central['model_simulation_references'])
    base_growth               = np.log(model_references_regional[:, 1:] / model_references_regional[:, :-1])
    model_log_agr             = np.log(model_projection_regional[:, 1:] / model_projection_regional[:, :-1])

    #### Log excess at grid cells → area-weighted global mean per year (still in log space)
    log_excess_model   = model_log_agr - base_growth
    global_log_model   = np.average(log_excess_model, weights=weights_projection[:, 1:], axis=0)

    all_emp_corrected = [np.array(projection_central['empirical_projection_corrected'])] + \
                        [np.array(results_bootstrap_dict[f'projection_{i+1}']['empirical_projection_corrected']) for i in range(bootstrap_num)]
    
    global_logs_emp = np.array([
        np.average(np.log(ec[:, 1:] / ec[:, :-1]) - base_growth, weights=weights_projection[:, 1:], axis=0)
        for ec in all_emp_corrected
    ])

    #### Annual panel: convert each year’s global-mean log excess to %
    model_d_minus_b_global = _log_excess_to_pct(global_log_model)
    d_minus_b_all          = _log_excess_to_pct(global_logs_emp)

    #### Cumulative panel: cumsum of global-mean log excess, then convert to %
    model_d_minus_b_cumsum = _log_excess_to_pct(np.cumsum(global_log_model))
    d_minus_b_cumsum_all   = _log_excess_to_pct(np.cumsum(global_logs_emp, axis=1))

    return plot_years, model_d_minus_b_global, d_minus_b_all, model_d_minus_b_cumsum, d_minus_b_cumsum_all

def _newell_d_minus_b_series(spec_list, form_specs):
    """Same d-b pipeline as Burke: cell log excess → global mean → cumsum in log space → (e^X−1)·100."""
    spec0                  = spec_list[0]
    years                  = np.array(spec0['years'])
    weights_projection     = np.array(spec0['weights_projection'])
    plot_years             = years[1:]
    model_projection_regional = np.array(spec0['model_simulation_projection'])
    model_references_regional = np.array(spec0['model_simulation_references'])
    base_growth            = np.log(model_references_regional[:, 1:] / model_references_regional[:, :-1])
    model_log_agr          = np.log(model_projection_regional[:, 1:] / model_projection_regional[:, :-1])
    global_log_model       = np.average(model_log_agr - base_growth, weights=weights_projection[:, 1:], axis=0)
    global_logs_emp = np.array([
        np.average(
            np.log(np.array(s['empirical_projection_corrected'])[:, 1:] / np.array(s['empirical_projection_corrected'])[:, :-1]) - base_growth,
            weights=weights_projection[:, 1:], axis=0)
        for s in form_specs
    ])
    annual_specs = _log_excess_to_pct(global_logs_emp)
    model_ann    = _log_excess_to_pct(global_log_model)
    cum_specs    = _log_excess_to_pct(np.cumsum(global_logs_emp, axis=1))
    model_cum    = _log_excess_to_pct(np.cumsum(global_log_model))
    median_ann   = np.median(annual_specs, axis=0)
    median_cum   = np.median(cum_specs, axis=0)
    d_for_plot   = np.vstack([median_ann, annual_specs])
    d_cum_plot   = np.vstack([median_cum, cum_specs])
    return plot_years, model_ann, d_for_plot, model_cum, d_cum_plot


def _prepare_growth_rate_timeseries(self, plot_type):
    """One pass: compute every Burke/Newell d-b series once, pool y-bounds for shared axes, return caches."""
    results_dict = self.results_dict
    model_list = self.model_list
    bootstrap_num = self.bootstrap_num

    if plot_type == 'main':
        models = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models = list(model_list)

    growth_annual_bounds = []
    growth_cum_bounds = []
    level_annual_bounds = []
    level_cum_bounds = []

    burke_series = {}
    newell_series = {}

    burke_growth_acts = [a for a in results_dict if a.startswith('burke_') and 'growth' in a]
    burke_level_acts = [a for a in results_dict if a.startswith('burke_') and 'level' in a]
    newell_acts = [a for a in results_dict if a.startswith('newell_')]

    for action_name in burke_growth_acts:
        for model_i in models:
            results_bootstrap_dict = results_dict[action_name][model_i]
            projection_central = results_bootstrap_dict['projection_main']
            tup = _burke_d_minus_b_series(results_bootstrap_dict, projection_central, bootstrap_num)
            burke_series[(action_name, model_i)] = tup
            _, ma, da, mc, dc = tup
            growth_annual_bounds.append(_ribbon_aligned_bounds(ma, da))
            growth_cum_bounds.append(_ribbon_aligned_bounds(mc, dc))

    for action_name in burke_level_acts:
        for model_i in models:
            results_bootstrap_dict = results_dict[action_name][model_i]
            projection_central = results_bootstrap_dict['projection_main']
            tup = _burke_d_minus_b_series(results_bootstrap_dict, projection_central, bootstrap_num)
            burke_series[(action_name, model_i)] = tup
            _, ma, da, mc, dc = tup
            level_annual_bounds.append(_ribbon_aligned_bounds(ma, da))
            level_cum_bounds.append(_ribbon_aligned_bounds(mc, dc))

    for action_name in newell_acts:
        for model_i in models:
            spec_list = results_dict[action_name][model_i]
            for gdp_form in ('growth', 'level'):
                form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
                if len(form_specs) == 0:
                    continue
                tup = _newell_d_minus_b_series(spec_list, form_specs)
                newell_series[(action_name, gdp_form, model_i)] = tup
                _, ma, da, mc, dc = tup
                if gdp_form == 'growth':
                    growth_annual_bounds.append(_ribbon_aligned_bounds(ma, da))
                    growth_cum_bounds.append(_ribbon_aligned_bounds(mc, dc))
                else:
                    level_annual_bounds.append(_ribbon_aligned_bounds(ma, da))
                    level_cum_bounds.append(_ribbon_aligned_bounds(mc, dc))

    ylim_shared = {}
    # sym = plot_type != 'main'
    # sym = True
    sym = False
    if len(growth_annual_bounds) > 0:
        ylim_shared['growth'] = {
            'annual': _pool_bounds(growth_annual_bounds),
            'cum': _pool_bounds(growth_cum_bounds),
            'symmetric': sym,
        }
    if len(level_annual_bounds) > 0:
        ylim_shared['level'] = {
            'annual': _pool_bounds(level_annual_bounds),
            'cum': _pool_bounds(level_cum_bounds),
        }
    return burke_series, newell_series, ylim_shared


def burke_subType(self, plot_type, ylim_shared, burke_series):
    results_dict = self.results_dict
    action_names = results_dict.keys()
    action_names = [a for a in action_names if a.startswith('burke_')]
    model_list = self.model_list

    for action_name in action_names:

        print (action_name)

        if plot_type == 'main':
            models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
        else:
            models_to_plot = list(model_list)

        n_models = len(models_to_plot)
        color    = gdp_form_color['growth'] if 'growth' in action_name else gdp_form_color['level']
        fig, axs = plt.subplots(n_models, 2, figsize=(10, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
        axs      = np.atleast_2d(axs)
        print(f'\n{action_name}  ({plot_type}:  {n_models} model(s))')

        series_list = [burke_series[(action_name, model_i)] for model_i in models_to_plot]

        #### Y-axis data range (shared with Newell when ylim_shared provided)
        effect_key = 'growth' if 'growth' in action_name else 'level'
        use_shared = ylim_shared is not None and effect_key in ylim_shared

        if use_shared:
            amin_a, amax_a = ylim_shared[effect_key]['annual']
            amin_c, amax_c = ylim_shared[effect_key]['cum']
            growth_symmetric = ylim_shared[effect_key].get('symmetric', plot_type != 'main')
        elif plot_type == 'main':
            _, model_ann, d_all, model_cum, d_cum_all = series_list[0]
            amin_a, amax_a = _ribbon_aligned_bounds(model_ann, d_all)
            amin_c, amax_c = _ribbon_aligned_bounds(model_cum, d_cum_all)
            growth_symmetric = False
        else:
            annual_bounds = [_ribbon_aligned_bounds(ma, da) for _, ma, da, _, _ in series_list]
            cum_bounds    = [_ribbon_aligned_bounds(mc, dc) for _, _, _, mc, dc in series_list]
            amin_a, amax_a = _pool_bounds(annual_bounds)
            amin_c, amax_c = _pool_bounds(cum_bounds)
            growth_symmetric = True

        for model_idx, model_i in enumerate(models_to_plot):
            plot_years, model_d_minus_b_global, d_minus_b_all, model_d_minus_b_cumsum, d_minus_b_cumsum_all = series_list[model_idx]
            years = results_dict[action_name][model_i]['projection_main']['years']

            if 'growth' in action_name:
                plot_this_panel(plot_years, model_d_minus_b_global, d_minus_b_all,        axs[model_idx, 0], color, 'arcsinh')
                plot_this_panel(plot_years, model_d_minus_b_cumsum, d_minus_b_cumsum_all, axs[model_idx, 1], color, 'arcsinh')
            elif 'level' in action_name:
                plot_this_panel(plot_years, model_d_minus_b_global, d_minus_b_all,        axs[model_idx, 0], color, 'linear')
                plot_this_panel(plot_years, model_d_minus_b_cumsum, d_minus_b_cumsum_all, axs[model_idx, 1], color, 'linear')
            axs[model_idx, 0].set_xlim(years[0], years[-1])
            axs[model_idx, 0].set_ylabel('d-b  (% yr⁻¹)')
            axs[model_idx, 1].set_ylabel('Cumulative d-b  (%)')
            axs[model_idx, 0].set_title(f'{model_i}  —  (A) d-b  (annual)')
            axs[model_idx, 1].set_title(f'{model_i}  —  (B) d-b  (cumulative)')

        if 'growth' in action_name:
            _apply_yaxis_growth_arcsinh(axs[0, 0], amin_a, amax_a, plot_type, growth_symmetric)
            _apply_yaxis_growth_arcsinh(axs[0, 1], amin_c, amax_c, plot_type, growth_symmetric)
        elif 'level' in action_name:
            _apply_yaxis_level_linear(axs[0, 0], amin_a, amax_a, plot_type)
            _apply_yaxis_level_linear(axs[0, 1], amin_c, amax_c, plot_type)

        # if plot_type == 'main': plt.savefig(f'burke_growth_rate_global_mean_{action_name}_Main.svg')
        # if plot_type == 'SI': plt.savefig(f'burke_growth_rate_global_mean_{action_name}_SI.svg')
        # plt.clf()
        plt.show() 


def newell_subType(self, plot_type, ylim_shared, newell_series):

    results_dict = self.results_dict
    action_names = list(results_dict.keys())
    action_names = [a for a in action_names if a.startswith('newell_')]
    model_list   = self.model_list

    for action_name in action_names:

        print (action_name)

        for gdp_form, color in gdp_form_color.items():

            if plot_type == 'main':
                models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
            else:
                models_to_plot = list(model_list)

            series_list = []
            models_with_data = []
            for model_i in models_to_plot:
                key = (action_name, gdp_form, model_i)
                if key not in newell_series:
                    continue
                models_with_data.append(model_i)
                series_list.append(newell_series[key])

            if len(series_list) == 0:
                print(f'  skip  {action_name}  {gdp_form}  (no matching specs)')
                continue

            n_models = len(series_list)
            fig, axs = plt.subplots(n_models, 2, figsize=(10, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
            axs      = np.atleast_2d(axs)
            print(f'\n{action_name}  ({gdp_form})  ({plot_type}:  {n_models} model(s))')

            use_shared = ylim_shared is not None and gdp_form in ylim_shared

            if use_shared:
                amin_a, amax_a = ylim_shared[gdp_form]['annual']
                amin_c, amax_c = ylim_shared[gdp_form]['cum']
                growth_symmetric = ylim_shared[gdp_form].get('symmetric', plot_type != 'main')
            elif plot_type == 'main':
                _, ma, da, mc, dc = series_list[0]
                amin_a, amax_a = _ribbon_aligned_bounds(ma, da)
                amin_c, amax_c = _ribbon_aligned_bounds(mc, dc)
                growth_symmetric = False
            else:
                annual_bounds = [_ribbon_aligned_bounds(ma, da) for _, ma, da, _, _ in series_list]
                cum_bounds    = [_ribbon_aligned_bounds(mc, dc) for _, _, _, mc, dc in series_list]
                amin_a, amax_a = _pool_bounds(annual_bounds)
                amin_c, amax_c = _pool_bounds(cum_bounds)
                growth_symmetric = True

            for model_idx, model_i in enumerate(models_with_data):
                plot_years, model_ann, d_for_plot, model_cum, d_cum_plot = series_list[model_idx]
                years = np.array(results_dict[action_name][model_i][0]['years'])

                if 'growth' in gdp_form:
                    plot_this_panel(plot_years, model_ann, d_for_plot, axs[model_idx, 0], color, 'arcsinh')
                    plot_this_panel(plot_years, model_cum, d_cum_plot, axs[model_idx, 1], color, 'arcsinh')
                elif 'level' in gdp_form:
                    plot_this_panel(plot_years, model_ann, d_for_plot, axs[model_idx, 0], color, 'linear')
                    plot_this_panel(plot_years, model_cum, d_cum_plot, axs[model_idx, 1], color, 'linear')
                axs[model_idx, 0].set_xlim(years[0], years[-1])
                axs[model_idx, 0].set_ylabel('d-b  (% yr⁻¹)')
                axs[model_idx, 1].set_ylabel('Cumulative d-b  (%)')
                axs[model_idx, 0].set_title(f'{model_i}  —  (A) d-b  (annual)')
                axs[model_idx, 1].set_title(f'{model_i}  —  (B) d-b  (cumulative)')

            if 'growth' in gdp_form:
                _apply_yaxis_growth_arcsinh(axs[0, 0], amin_a, amax_a, plot_type, growth_symmetric)
                _apply_yaxis_growth_arcsinh(axs[0, 1], amin_c, amax_c, plot_type, growth_symmetric)
            elif 'level' in gdp_form:
                _apply_yaxis_level_linear(axs[0, 0], amin_a, amax_a, plot_type)
                _apply_yaxis_level_linear(axs[0, 1], amin_c, amax_c, plot_type)

            # if plot_type == 'main': plt.savefig(f'newell_growth_rate_global_mean_{action_name}_{gdp_form}_Main.svg')
            # if plot_type == 'SI': plt.savefig(f'newell_growth_rate_global_mean_{action_name}_{gdp_form}_SI.svg')
            # plt.clf()
            plt.show() 

def growth_rate_global_mean_timeSeries(self):
    for plot_type in ('main', 'SI'):
        burke_series, newell_series, ylim_shared = _prepare_growth_rate_timeseries(self, plot_type)
        burke_subType(self, plot_type, ylim_shared, burke_series)
        newell_subType(self, plot_type, ylim_shared, newell_series) 