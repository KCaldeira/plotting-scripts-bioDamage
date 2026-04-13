import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from plot.func_shared_plotting import (
    _pool_bounds,
    _apply_yaxis_growth_arcsinh,
    _apply_yaxis_level_linear,
    _ribbon_aligned_bounds,
)

gdp_form_color = {'growth': 'firebrick', 'level': 'royalblue'}

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'
_TS_FIG_WIDTH_IN_PER_COL = 2
_TS_BOX_POS_MODEL = 0.24
_TS_BOX_POS_EMP = 0.76
_TS_XLIM = (-0.05, 1.05)


def _log_excess_to_pct(log_excess):
    """Map log-growth excess X = log g_full − log g_BGC to percent: (e^X − 1)·100."""
    return (np.exp(log_excess) - 1) * 100


def compute_cumsum(data_2d, base_growth):
    """Per-country cumulative d-b: cumsum of log excess in log space, then (e^X−1)·100; valid rows only."""
    log_excess = np.log(data_2d[:, 1:] / data_2d[:, :-1]) - base_growth
    log_cum = np.cumsum(log_excess, axis=1)
    cum_pct = _log_excess_to_pct(log_cum)
    valid = ~np.any(np.isnan(cum_pct), axis=1)
    return cum_pct[valid]


def cumulative_surface(data_2d, base_growth):
    """Full grid (no row drop); same pipeline as compute_cumsum for stacking Newell specs."""
    log_excess = np.log(data_2d[:, 1:] / data_2d[:, :-1]) - base_growth
    log_cum = np.cumsum(log_excess, axis=1)
    return _log_excess_to_pct(log_cum)


def _native_vertical_span_for_draw_distribution(data_1d):
    """Native p5–p95 cross-country span (same as draw_distribution whiskers).

    Using min/max would let a single extreme country widen the shared y-axis; pooled
    limits should follow the plotted 5–95% envelope so axes-main stays near the visible range.
    """
    d = np.asarray(data_1d, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError('no finite values for ts y bounds')
    p5, p95 = np.percentile(d, [5, 95])
    return float(p5), float(p95)


def _ts_panel_native_bounds(model_1d, emp_1d):
    lo_m, hi_m = _native_vertical_span_for_draw_distribution(model_1d)
    lo_e, hi_e = _native_vertical_span_for_draw_distribution(emp_1d)
    return min(lo_m, lo_e), max(hi_m, hi_e)


def _prepare_country_distribution_ts(self, plot_type):

    results_dict = self.results_dict
    target_year = 2100
    models = [m for m in self.model_list if m == _MAIN_ONLY_MODEL] if plot_type == 'main' else list(self.model_list)

    growth_bounds = []
    level_bounds = []
    burke_cache = {}
    newell_cache = {}

    for action_name in [a for a in results_dict if a.startswith('burke_')]:
        for model_i in models:
            pc = results_dict[action_name][model_i]['projection_main']
            years = np.array(pc['years'])
            plot_years = years[1:]
            idx = int(np.searchsorted(plot_years, target_year))
            emp_corrected = np.array(pc['empirical_projection_corrected'])
            model_projection = np.array(pc['model_simulation_projection'])
            model_references = np.array(pc['model_simulation_references'])
            base_growth = np.log(model_references[:, 1:] / model_references[:, :-1])
            emp_cumsum = compute_cumsum(emp_corrected, base_growth)
            model_cumsum = compute_cumsum(model_projection, base_growth)
            burke_cache[(action_name, model_i)] = (years, plot_years, model_cumsum, emp_cumsum)
            b = _ts_panel_native_bounds(model_cumsum[:, idx], emp_cumsum[:, idx])
            if 'growth' in action_name:
                growth_bounds.append(b)
            else:
                level_bounds.append(b)

    for action_name in [a for a in results_dict if a.startswith('newell_')]:
        for model_i in models:
            spec_list = results_dict[action_name][model_i]
            spec0 = spec_list[0]
            years = np.array(spec0['years'])
            plot_years = years[1:]
            idx = int(np.searchsorted(plot_years, target_year))
            model_projection = np.array(spec0['model_simulation_projection'])
            model_references = np.array(spec0['model_simulation_references'])
            base_growth = np.log(model_references[:, 1:] / model_references[:, :-1])
            model_cumsum = compute_cumsum(model_projection, base_growth)
            for gdp_form in ('growth', 'level'):
                form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
                if len(form_specs) == 0:
                    continue
                emp_cumsums = np.array([cumulative_surface(np.array(s['empirical_projection_corrected']), base_growth) for s in form_specs])
                emp_median_country = np.median(emp_cumsums, axis=0)
                valid_emp = ~np.any(np.isnan(emp_median_country), axis=1)
                newell_cache[(action_name, gdp_form, model_i)] = (years, plot_years, model_cumsum, emp_median_country, valid_emp)
                b = _ts_panel_native_bounds(model_cumsum[:, idx], emp_median_country[valid_emp, idx])
                if gdp_form == 'growth':
                    growth_bounds.append(b)
                else:
                    level_bounds.append(b)

    ylim_shared = {}
    # sym = plot_type != 'main'
    sym = False 
    if len(growth_bounds) > 0:
        ylim_shared['growth'] = {'cum': _pool_bounds(growth_bounds), 'symmetric': sym}
    if len(level_bounds) > 0:
        ylim_shared['level'] = {'cum': _pool_bounds(level_bounds)}
    return burke_cache, newell_cache, ylim_shared


def draw_distribution(ax, data, pos, color, yscale):
    """Boxplot-style: p50 marker, p25–p75 box, p5–p95 whiskers."""
    y = np.asarray(data, dtype=float)
    if yscale == 'arcsinh':
        y = np.arcsinh(y)
    p5, p25, p50, p75, p95 = np.percentile(y, [5, 25, 50, 75, 95])
    print (f'{yscale}  {np.sinh(p5)}, {np.sinh(p25)}, {np.sinh(p50)}, {np.sinh(p75)}, {np.sinh(p95)}') if yscale == 'arcsinh' else print (f'{yscale}  {p5}, {p25}, {p50}, {p75}, {p95}')
    
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


def burke_subType(self, plot_type, ylim_shared, burke_cache):

    results_dict = self.results_dict
    action_names = results_dict.keys()
    action_names = [a for a in action_names if a.startswith('burke_')]
    model_list   = self.model_list
    target_year  = 2100

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)

    n_models = len(models_to_plot)
    n_actions = len(action_names)
    fig, axes = plt.subplots(n_models, n_actions, figsize=(_TS_FIG_WIDTH_IN_PER_COL * n_actions, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
    axes = np.array(axes).reshape(n_models, n_actions)

    for model_idx, model_i in enumerate(models_to_plot):
        for action_idx, action_name in enumerate(action_names):

            print (model_i, action_name)

            ax    = axes[model_idx, action_idx]
            color = gdp_form_color['growth'] if 'growth' in action_name else gdp_form_color['level']
            years, plot_years, model_cumsum, emp_cumsum = burke_cache[(action_name, model_i)]
            yscale = 'arcsinh' if 'growth' in action_name else 'linear'

            idx = np.searchsorted(plot_years, target_year)
            draw_distribution(ax, model_cumsum[:, idx], _TS_BOX_POS_MODEL, 'gray', yscale)
            draw_distribution(ax, emp_cumsum[:, idx], _TS_BOX_POS_EMP, color, yscale)
            ax.set_title(f'{model_i}  —  {action_name}')

    use_growth = ylim_shared is not None and 'growth' in ylim_shared
    use_level = ylim_shared is not None and 'level' in ylim_shared
    for action_idx, action_name in enumerate(action_names):
        ax0 = axes[0, action_idx]
        if 'growth' in action_name and use_growth:
            amin_g, amax_g = ylim_shared['growth']['cum']
            growth_symmetric = ylim_shared['growth'].get('symmetric', plot_type != 'main')
            _apply_yaxis_growth_arcsinh(ax0, amin_g, amax_g, plot_type, growth_symmetric)
        elif 'level' in action_name and use_level:
            amin_l, amax_l = ylim_shared['level']['cum']
            _apply_yaxis_level_linear(ax0, amin_l, amax_l, plot_type)


    suffix = 'Main' if plot_type == 'main' else 'SI'
    # plt.savefig(f'burke_growth_rate_countries_discrete_ts_{action_name}_{suffix}.svg')
    # plt.clf()
    plt.show()


def newell_subType(self, plot_type, ylim_shared, newell_cache):

    results_dict = self.results_dict
    action_names = [a for a in results_dict.keys() if a.startswith('newell_')]
    target_year  = 2100

    models_to_plot = [m for m in self.model_list if m == _MAIN_ONLY_MODEL] if plot_type == 'main' else list(self.model_list)
    n_models = len(models_to_plot)
    n_actions = len(action_names)
    n_cols = n_actions * 2
    fig, axes = plt.subplots(n_models, n_cols, figsize=(_TS_FIG_WIDTH_IN_PER_COL * n_cols, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
    axes = np.array(axes).reshape(n_models, n_cols)

    for model_idx, model_i in enumerate(models_to_plot):
        for action_idx, action_name in enumerate(action_names):

            print (model_i, action_name)

            spec_list = results_dict[action_name][model_i]

            for gdp_idx, (gdp_form, color) in enumerate(gdp_form_color.items()):
                col_idx = action_idx * 2 + gdp_idx
                ax = axes[model_idx, col_idx]

                key = (action_name, gdp_form, model_i)
                if key not in newell_cache:
                    ax.axis('off')
                    continue

                years, plot_years, model_cumsum, emp_median_country, valid_emp = newell_cache[key]
                form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
                yscale = 'arcsinh' if gdp_form == 'growth' else 'linear'

                idx = np.searchsorted(plot_years, target_year)
                draw_distribution(ax, model_cumsum[:, idx], _TS_BOX_POS_MODEL, 'gray', yscale)
                draw_distribution(ax, emp_median_country[valid_emp, idx], _TS_BOX_POS_EMP, color, yscale)
                ax.set_title(f'{model_i}  —  {action_name}  ({gdp_form},  {len(form_specs)} specs)')

    use_growth = ylim_shared is not None and 'growth' in ylim_shared
    use_level = ylim_shared is not None and 'level' in ylim_shared
    for action_idx in range(n_actions):
        for gdp_idx, gdp_form in enumerate(gdp_form_color.keys()):
            col_idx = action_idx * 2 + gdp_idx
            ax0 = axes[0, col_idx]
            if not ax0.get_visible():
                continue
            if gdp_form == 'growth' and use_growth:
                amin_g, amax_g = ylim_shared['growth']['cum']
                growth_symmetric = ylim_shared['growth'].get('symmetric', plot_type != 'main')
                _apply_yaxis_growth_arcsinh(ax0, amin_g, amax_g, plot_type, growth_symmetric)
            elif gdp_form == 'level' and use_level:
                amin_l, amax_l = ylim_shared['level']['cum']
                _apply_yaxis_level_linear(ax0, amin_l, amax_l, plot_type)

    suffix = 'Main' if plot_type == 'main' else 'SI'
    # plt.savefig(f'newell_growth_rate_country_distribution_ts_{plot_type}_{suffix}.svg')
    # plt.clf()
    plt.show() 


def growth_rate_country_boxplotLikeDistribution(self):
    # for plot_type in ('main', 'SI'):
    for plot_type in ['main']:
    # for plot_type in ['SI']:
        burke_cache, newell_cache, ylim_ts = _prepare_country_distribution_ts(self, plot_type)
        burke_subType(self, plot_type, ylim_ts, burke_cache)
        newell_subType(self, plot_type, ylim_ts, newell_cache)