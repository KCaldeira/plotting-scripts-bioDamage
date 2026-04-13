
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


from plot.func_shared_plotting import (
    _pool_bounds,
    _apply_yaxis_growth_arcsinh,
    _apply_yaxis_level_linear,
    _ribbon_aligned_bounds,
)


_PERCENTILES = (0, 5, 25, 50, 75, 90, 100)

gdp_form_color = {'growth': 'firebrick', 'level': 'royalblue'}

_MAIN_ONLY_MODEL = 'ACCESS-ESM1-5'
_CD_FIG_WIDTH_IN = 6.3
_CD_BOX_HW = 0.18
_CD_CAP_HW = 0.20
_CD_SCATTER_S = 28


def _draw_distribution_box_at_x(ax, y, pos, color):
    """Boxplot-style in data y: p50 marker, p25–p75 box, p5–p95 whiskers (y already in axis coordinates)."""
    y = np.asarray(y, dtype=float)
    p5, p25, p50, p75, p95 = np.percentile(y, [5, 25, 50, 75, 95])
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
    ax.scatter([pos], [p50], color='white', s=_CD_SCATTER_S, zorder=5, edgecolors=color, linewidths=1.2)


def _draw_distribution_country(ax, data_native, pos, color, yscale):
    """Native % values → same y mapping as before (arcsinh growth / linear level), then box glyphs."""
    y = np.asarray(data_native, dtype=float)
    if yscale == 'arcsinh':
        y = np.arcsinh(y)
    elif yscale == 'linear':
        pass
    else:
        raise ValueError(yscale)
    _draw_distribution_box_at_x(ax, y, pos, color)


def _country_indices_at_percentiles(values_1d, weights_1d, percentiles):
    """Countries sorted by value; percentile p is the first country whose cumulative weight share reaches p/100."""
    v = np.asarray(values_1d, dtype=float)
    w = np.asarray(weights_1d, dtype=float)
    if v.shape[0] != w.shape[0]:
        raise ValueError('values_1d and weights_1d must have the same length')
    idx_all = np.arange(v.shape[0])
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v_valid = v[mask]
    w_valid = w[mask]
    idx_valid = idx_all[mask]
    order = np.argsort(v_valid)
    w_sorted = w_valid[order]
    idx_sorted = idx_valid[order]
    if len(order) == 0:
        raise ValueError('no countries with finite cumulative impact and positive weights')
    cum_w = np.cumsum(w_sorted)
    w_sum = float(cum_w[-1])
    out = []
    for p in percentiles:
        target = min(p / 100.0 * w_sum, w_sum)
        k = int(np.searchsorted(cum_w, target, side='left'))
        k = min(k, len(idx_sorted) - 1)
        out.append(int(idx_sorted[k]))
    return out







def _burke_d_minus_b_series(results_bootstrap_dict, bootstrap_num, weight_mode):
    projection_central        = results_bootstrap_dict['projection_main'] 
    model_projection_regional = np.array(projection_central['model_simulation_projection'])
    model_references_regional = np.array(projection_central['model_simulation_references'])
    base_growth               = np.log(model_references_regional[:, 1:] / model_references_regional[:, :-1])

    #### Climate model 
    model_log_agr                     = np.log(model_projection_regional[:, 1:] / model_projection_regional[:, :-1])
    log_excess_model                  = model_log_agr - base_growth
    log_excess_model_cum_2100         = np.cumsum(log_excess_model, axis=1)[:, -1]
    years = np.array(projection_central['years'])
    iy_2100_hits = np.where(years == 2100)[0]
    iy_2100 = int(iy_2100_hits[0]) if len(iy_2100_hits) > 0 else int(len(years) - 1)
    if weight_mode == 'gpp':
        w_select = model_references_regional[:, iy_2100]
    elif weight_mode == 'area':
        w_select = np.array(projection_central['weights_projection'])[:, -1]
    else:
        raise ValueError(f'weight_mode must be "gpp" or "area", got {weight_mode!r}')
    country_selected = _country_indices_at_percentiles(log_excess_model_cum_2100, w_select, _PERCENTILES)
    log_excess_model_cum_2100_country = log_excess_model_cum_2100[country_selected]
    mc = (np.exp(log_excess_model_cum_2100_country) - 1) * 100

    #### Empirical 
    all_emp_corrected = [np.array(projection_central['empirical_projection_corrected'])] + \
                        [np.array(results_bootstrap_dict[f'projection_{i+1}']['empirical_projection_corrected']) for i in range(bootstrap_num)]
    
    log_excess_emp_cum_2100_country_all = [] 
    for ec in all_emp_corrected:
        log_excess_emp_ec = np.log(ec[:, 1:] / ec[:, :-1]) - base_growth
        log_excess_emp_ec_cum_2100 = np.cumsum(log_excess_emp_ec, axis=1)[:, -1] 
        log_excess_emp_ec_cum_2100_country = log_excess_emp_ec_cum_2100[country_selected]
        log_excess_emp_ec_cum_2100_country = (np.exp(log_excess_emp_ec_cum_2100_country) - 1) * 100
        log_excess_emp_cum_2100_country_all.append(log_excess_emp_ec_cum_2100_country)
    dc = np.array(log_excess_emp_cum_2100_country_all)
    return country_selected, mc, dc


def _newell_country_cum_2100_series(spec_list, form_specs, weight_mode):
    """Per-region cumulative log excess at 2100 (model from spec_list[0]); empirical rows one per form_spec."""
    spec0 = spec_list[0]
    model_projection_regional = np.array(spec0['model_simulation_projection'])
    model_references_regional = np.array(spec0['model_simulation_references'])
    base_growth = np.log(model_references_regional[:, 1:] / model_references_regional[:, :-1])
    model_log_agr = np.log(model_projection_regional[:, 1:] / model_projection_regional[:, :-1])
    log_excess_model = model_log_agr - base_growth
    log_excess_model_cum_2100 = np.cumsum(log_excess_model, axis=1)[:, -1]
    years = np.array(spec0['years'])
    iy_2100_hits = np.where(years == 2100)[0]
    iy_2100 = int(iy_2100_hits[0]) if len(iy_2100_hits) > 0 else int(len(years) - 1)
    if weight_mode == 'gpp':
        w_select = model_references_regional[:, iy_2100]
    elif weight_mode == 'area':
        w_select = np.array(spec0['weights_projection'])[:, -1]
    else:
        raise ValueError(f'weight_mode must be "gpp" or "area", got {weight_mode!r}')
    country_selected = _country_indices_at_percentiles(log_excess_model_cum_2100, w_select, _PERCENTILES)
    mc = (np.exp(log_excess_model_cum_2100[country_selected]) - 1) * 100
    rows = []
    for s in form_specs:
        ec = np.array(s['empirical_projection_corrected'])
        log_excess_emp = np.log(ec[:, 1:] / ec[:, :-1]) - base_growth
        log_excess_emp_cum_2100 = np.cumsum(log_excess_emp, axis=1)[:, -1]
        rows.append((np.exp(log_excess_emp_cum_2100[country_selected]) - 1) * 100)
    dc = np.array(rows)
    return country_selected, mc, dc


def _country_panel_ribbon_bounds_newell(mc, dc_native):
    emp_median = np.median(dc_native, axis=0, keepdims=True)
    stacked = np.vstack([emp_median, dc_native])
    return _ribbon_aligned_bounds(mc, stacked)


def _prepare_country_distribution_bar(self, plot_type, weight_mode):
    """One pass: Burke/Newell country percentile series + pooled y-bounds (native %)."""
    results_dict = self.results_dict
    bootstrap_num = self.bootstrap_num

    if plot_type == 'main':
        models = [m for m in self.model_list if m == _MAIN_ONLY_MODEL]
    else:
        models = list(self.model_list)

    growth_bounds = []
    level_bounds = []
    burke_cache = {}
    newell_cache = {}

    burke_growth_acts = [a for a in results_dict if a.startswith('burke_') and 'growth' in a]
    burke_level_acts = [a for a in results_dict if a.startswith('burke_') and 'level' in a]
    newell_acts = [a for a in results_dict if a.startswith('newell_')]

    for action_name in burke_growth_acts:
        for model_i in models:
            tup = _burke_d_minus_b_series(results_dict[action_name][model_i], bootstrap_num, weight_mode)
            burke_cache[(action_name, model_i)] = tup
            _, mc, dc = tup
            growth_bounds.append(_ribbon_aligned_bounds(mc, dc))

    for action_name in burke_level_acts:
        for model_i in models:
            tup = _burke_d_minus_b_series(results_dict[action_name][model_i], bootstrap_num, weight_mode)
            burke_cache[(action_name, model_i)] = tup
            _, mc, dc = tup
            level_bounds.append(_ribbon_aligned_bounds(mc, dc))

    for action_name in newell_acts:
        for model_i in models:
            spec_list = results_dict[action_name][model_i]
            for gdp_form in ('growth', 'level'):
                form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
                if len(form_specs) == 0:
                    continue
                tup = _newell_country_cum_2100_series(spec_list, form_specs, weight_mode)
                newell_cache[(action_name, gdp_form, model_i)] = tup
                _, mc, dc = tup
                b = _country_panel_ribbon_bounds_newell(mc, dc)
                if gdp_form == 'growth':
                    growth_bounds.append(b)
                else:
                    level_bounds.append(b)

    ylim_shared = {}
    sym = plot_type != 'main'
    if len(growth_bounds) > 0:
        ylim_shared['growth'] = {'cum': _pool_bounds(growth_bounds), 'symmetric': sym}
    if len(level_bounds) > 0:
        ylim_shared['level'] = {'cum': _pool_bounds(level_bounds)}
    return burke_cache, newell_cache, ylim_shared


def burke_subType(self, plot_type, ylim_shared, weight_mode, burke_cache):
    results_dict = self.results_dict
    action_names = results_dict.keys()
    action_names = [a for a in action_names if a.startswith('burke_')]
    model_list = self.model_list

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)
    n_models = len(models_to_plot)

    fig, axs = plt.subplots(n_models, 2, figsize=(_CD_FIG_WIDTH_IN, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
    axs      = np.atleast_2d(axs)

    w_label = 'GPP (BGC) at 2100' if weight_mode == 'gpp' else 'Area (last year)'
    print(f'\n({plot_type}:  {n_models} model(s)) {models_to_plot}  |  percentile weights: {w_label}')
    fig.suptitle(f'Country selection: weighted by {w_label}', fontsize=11, y=1.02)

    for model_i in models_to_plot:
        for action_name in action_names:
            color = gdp_form_color['growth'] if 'growth' in action_name else gdp_form_color['level']
            results_bootstrap_dict = results_dict[action_name][model_i]
            country_selected, mc, dc = burke_cache[(action_name, model_i)]
            if 'growth' in action_name:
                mc_plot = np.arcsinh(mc)
            else:
                mc_plot = mc
            yscale = 'arcsinh' if 'growth' in action_name else 'linear'
            ax = axs[models_to_plot.index(model_i), 0] if 'growth' in action_name else axs[models_to_plot.index(model_i), 1]
            region_list = results_bootstrap_dict['projection_main']['region_list']
            x_labels = [str(region_list[r]) for r in country_selected]
            n_sel = len(country_selected)
            ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=4)
            for ci in range(n_sel):
                v = dc[:, ci]
                emp_native = v[1:] if v.shape[0] > 1 else v
                _draw_distribution_country(ax, emp_native, ci, color, yscale)
                ax.scatter(ci, mc_plot[ci], color='black', s=36, zorder=8, marker='D')
            ax.set_xticks(np.arange(n_sel))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_xlim(-0.5, n_sel - 0.5)

    use_growth = ylim_shared is not None and 'growth' in ylim_shared
    use_level = ylim_shared is not None and 'level' in ylim_shared
    if use_growth:
        amin_g, amax_g = ylim_shared['growth']['cum']
        growth_symmetric = ylim_shared['growth'].get('symmetric', plot_type != 'main')
        _apply_yaxis_growth_arcsinh(axs[0, 0], amin_g, amax_g, plot_type, growth_symmetric)
    if use_level:
        amin_l, amax_l = ylim_shared['level']['cum']
        _apply_yaxis_level_linear(axs[0, 1], amin_l, amax_l, plot_type)

    suffix = 'Main' if plot_type == 'main' else 'SI'
    # plt.savefig(f'burke_growth_rate_country_distribution_weights-{weight_mode}_{suffix}.svg', bbox_inches='tight')
    # plt.clf()
    plt.show() 

def newell_subType(self, plot_type, ylim_shared, weight_mode, newell_cache):
    results_dict = self.results_dict
    action_names = [a for a in results_dict.keys() if a.startswith('newell_')]
    model_list = self.model_list

    if plot_type == 'main':
        models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
    else:
        models_to_plot = list(model_list)
    n_models = len(models_to_plot)

    w_label = 'GPP (BGC) at 2100' if weight_mode == 'gpp' else 'Area (last year)'

    for action_name in action_names:
        fig, axs = plt.subplots(n_models, 2, figsize=(_CD_FIG_WIDTH_IN, 3 * n_models), sharex=True, sharey='col', constrained_layout=True)
        axs = np.atleast_2d(axs)
        print(f'\n{action_name}  ({plot_type}:  {n_models} model(s))  |  percentile weights: {w_label}')
        fig.suptitle(f'{action_name}  —  country selection weighted by {w_label}', fontsize=11, y=1.02) 

        for mi, model_i in enumerate(models_to_plot):
            spec_list = results_dict[action_name][model_i]
            for col_idx, gdp_form in enumerate(('growth', 'level')):
                ax = axs[mi, col_idx]
                color = gdp_form_color[gdp_form]
                key = (action_name, gdp_form, model_i)
                if key not in newell_cache:
                    ax.axis('off')
                    continue
                country_selected, mc, dc = newell_cache[key]
                if gdp_form == 'growth':
                    mc_plot = np.arcsinh(mc)
                    dc_plot = np.arcsinh(dc)
                else:
                    mc_plot = mc
                    dc_plot = dc
                emp_median = np.median(dc_plot, axis=0)
                region_list = spec_list[0]['region_list']
                x_labels = [str(region_list[r]) for r in country_selected]
                n_sel = len(country_selected)
                ax.axhline(y=0.0, color='black', linewidth=1.3, linestyle='--', zorder=4)
                for ci in range(n_sel):
                    col_y = np.concatenate([[emp_median[ci]], dc_plot[:, ci]])
                    _draw_distribution_box_at_x(ax, col_y, ci, color)
                    ax.scatter(ci, mc_plot[ci], color='black', s=36, zorder=8, marker='D')
                ax.set_xticks(np.arange(n_sel))
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
                ax.set_xlim(-0.5, n_sel - 0.5)

        use_growth = ylim_shared is not None and 'growth' in ylim_shared
        use_level = ylim_shared is not None and 'level' in ylim_shared 
        if use_growth:
            amin_g, amax_g = ylim_shared['growth']['cum']
            growth_symmetric = ylim_shared['growth'].get('symmetric', plot_type != 'main')
            _apply_yaxis_growth_arcsinh(axs[0, 0], amin_g, amax_g, plot_type, growth_symmetric)
        if use_level:
            amin_l, amax_l = ylim_shared['level']['cum']
            _apply_yaxis_level_linear(axs[0, 1], amin_l, amax_l, plot_type)

        suffix = 'Main' if plot_type == 'main' else 'SI'
        # plt.savefig(f'newell_growth_rate_country_distribution_weights-{weight_mode}_{suffix}.svg', bbox_inches='tight')
        # plt.clf()
        plt.show() 


def growth_rate_country_boxplotLikeSelectedCountries(self):
    for plot_type in ['main', 'SI']:
        for weight_mode in ['gpp']:
            burke_cache, newell_cache, ylim_country = _prepare_country_distribution_bar(self, plot_type, weight_mode)
            burke_subType(self, plot_type, ylim_country, weight_mode, burke_cache)
            newell_subType(self, plot_type, ylim_country, weight_mode, newell_cache)