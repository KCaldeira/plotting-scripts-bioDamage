import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plot.growth_rate_global_mean_timeSeries import _MAIN_ONLY_MODEL


def _gpp_violin_panel_bundle(results_dict, model_i, gdp_form, burke_growth_acts, burke_level_acts, newell_acts, x_labels):

    """Arrays and star lists for one model × (growth | level) panel; also all y used for autoscale."""
    bar_colors = {'growth': 'firebrick', 'level': 'royalblue'}
    color = bar_colors[gdp_form]
    b_acts = burke_growth_acts if gdp_form == 'growth' else burke_level_acts

    proj_main = results_dict[b_acts[0]][model_i]['projection_main']
    years = np.array(proj_main['years'])
    i0 = np.searchsorted(years, 2080)
    i1 = np.searchsorted(years, 2101)

    refs = np.array(proj_main['model_simulation_references'])
    mod = np.array(proj_main['model_simulation_projection'])
    emp = np.array(proj_main['empirical_projection_corrected'])
    wp = np.array(proj_main['weights_projection'])
    ref_mean = np.mean(refs[:, i0:i1], axis=1)
    area_w = np.mean(wp[:, i0:i1], axis=1)

    r_model = np.mean(mod[:, i0:i1], axis=1) / ref_mean
    r_burke = np.mean(emp[:, i0:i1], axis=1) / ref_mean
    ok_m = np.isfinite(r_model) & (r_model > 0)
    ok_b = np.isfinite(r_burke) & (r_burke > 0)
    y_model = np.log10(r_model[ok_m])
    y_burke = np.log10(r_burke[ok_b])

    spec_list = results_dict[newell_acts[0]][model_i]
    spec0 = spec_list[0]
    years_n = np.array(spec0['years'])
    j0 = np.searchsorted(years_n, 2080)
    j1 = np.searchsorted(years_n, 2101)
    form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
    star_newell = None
    if len(form_specs) == 0:
        y_newell = np.array([], dtype=float)
    else:
        refs_n = np.array(spec0['model_simulation_references'])
        wp_n = np.array(spec0['weights_projection'])
        ref_n = np.mean(refs_n[:, j0:j1], axis=1)
        area_w_n = np.mean(wp_n[:, j0:j1], axis=1)
        stack = np.array([
            np.mean(np.array(s['empirical_projection_corrected'])[:, j0:j1], axis=1) / ref_n
            for s in form_specs
        ])
        r_new = np.median(stack, axis=0)
        ok_n = np.isfinite(r_new) & (r_new > 0)
        y_newell = np.log10(r_new[ok_n])
        m_nw = np.isfinite(r_new) & (r_new > 0) & np.isfinite(area_w_n) & (area_w_n > 0)
        star_newell = float(np.average(np.log10(r_new[m_nw]), weights=area_w_n[m_nw]))

    datasets, positions, labels_use = [], [], []
    star_x, star_y, star_ec, star_fc = [], [], [], []
    y_all = []

    if y_model.size > 0:
        datasets.append(y_model)
        positions.append(1.0)
        labels_use.append(x_labels[0])
        star_x.append(1.0)
        m_wm = np.isfinite(r_model) & (r_model > 0) & np.isfinite(area_w) & (area_w > 0)
        sy = float(np.average(np.log10(r_model[m_wm]), weights=area_w[m_wm]))
        star_y.append(sy)
        star_ec.append('black')
        star_fc.append('white')
        y_all.extend(y_model.tolist())
        y_all.append(sy)
    if y_burke.size > 0:
        datasets.append(y_burke)
        positions.append(2.0)
        labels_use.append(x_labels[1])
        star_x.append(2.0)
        m_wb = np.isfinite(r_burke) & (r_burke > 0) & np.isfinite(area_w) & (area_w > 0)
        sy = float(np.average(np.log10(r_burke[m_wb]), weights=area_w[m_wb]))
        star_y.append(sy)
        star_ec.append(color)
        star_fc.append('white')
        y_all.extend(y_burke.tolist())
        y_all.append(sy)
    if y_newell.size > 0:
        datasets.append(y_newell)
        positions.append(3.0)
        labels_use.append(x_labels[2])
        star_x.append(3.0)
        star_y.append(star_newell)
        star_ec.append(color)
        star_fc.append(color)
        y_all.extend(y_newell.tolist())
        y_all.append(star_newell)

    return {
        'color': color,
        'datasets': datasets,
        'positions': positions,
        'labels_use': labels_use,
        'star_x': star_x,
        'star_y': star_y,
        'star_ec': star_ec,
        'star_fc': star_fc,
        'y_all': y_all,
    }


def gpp_country_violin(self):

    for plot_type in ('main', 'SI'):
        results_dict = self.results_dict
        action_names = list(results_dict.keys())
        model_list = self.model_list

        burke_growth_acts = [a for a in action_names if a.startswith('burke_') and 'growth' in a]
        burke_level_acts = [a for a in action_names if a.startswith('burke_') and 'level' in a]
        newell_acts = [a for a in action_names if a.startswith('newell_')]

        ylim_cfg = {
            'growth': {'lim': (-1.0, 1.0), 'ticks': [-1.0, -0.5, 0, 0.5, 1.0]},
            'level': {'lim': (-0.2, 0.0), 'ticks': [-0.2, -0.1, 0.0]},
        }
        x_labels = ['Model\n(per country)', 'Burke central\n(per country)', 'Newell median\n(per country)']

        if plot_type == 'main':
            models_to_plot = [m for m in model_list if m == _MAIN_ONLY_MODEL]
            if len(models_to_plot) == 0:
                raise ValueError(f'plot_type main requires {_MAIN_ONLY_MODEL!r} in model_list')
        else:
            models_to_plot = list(model_list)

        n_models = len(models_to_plot)
        fig_w_in = 6.0
        fig_h_in = fig_w_in * n_models / 2.0
        fig, axes = plt.subplots(n_models, 2, figsize=(fig_w_in, fig_h_in), sharey='col', constrained_layout=True)
        axes = np.array(axes).reshape(n_models, 2)

        si_bundle_cache = {}
        ylim_si_growth = None
        ylim_si_level = None
        if plot_type == 'SI':
            ys_si_growth = []
            ys_si_level = []
            for model_i in models_to_plot:
                bg = _gpp_violin_panel_bundle(
                    results_dict, model_i, 'growth', burke_growth_acts, burke_level_acts, newell_acts, x_labels
                )
                bl = _gpp_violin_panel_bundle(
                    results_dict, model_i, 'level', burke_growth_acts, burke_level_acts, newell_acts, x_labels
                )
                si_bundle_cache[(model_i, 'growth')] = bg
                si_bundle_cache[(model_i, 'level')] = bl
                ys_si_growth.extend(bg['y_all'])
                ys_si_level.extend(bl['y_all'])
            if len(ys_si_growth) == 0:
                ylim_si_growth = None
            else:
                lo, hi = float(min(ys_si_growth)), float(max(ys_si_growth))
                sp = hi - lo
                pad = max(0.06 * sp, 0.02) if sp > 0 else 0.05
                ylim_si_growth = (lo - pad, hi + pad)
            if len(ys_si_level) == 0:
                ylim_si_level = None
            else:
                lo, hi = float(min(ys_si_level)), float(max(ys_si_level))
                sp = hi - lo
                pad = max(0.06 * sp, 0.02) if sp > 0 else 0.05
                ylim_si_level = (lo - pad, hi + pad)

        for model_idx, model_i in enumerate(models_to_plot):
            if plot_type == 'main':
                bg = _gpp_violin_panel_bundle(
                    results_dict, model_i, 'growth', burke_growth_acts, burke_level_acts, newell_acts, x_labels
                )
                bl = _gpp_violin_panel_bundle(
                    results_dict, model_i, 'level', burke_growth_acts, burke_level_acts, newell_acts, x_labels
                )
                yg, yl = bg['y_all'], bl['y_all']
                if len(yg) == 0:
                    ylim_main_growth = None
                else:
                    lo, hi = float(min(yg)), float(max(yg))
                    sp = hi - lo
                    pad = max(0.06 * sp, 0.02) if sp > 0 else 0.05
                    ylim_main_growth = (lo - pad, hi + pad)
                if len(yl) == 0:
                    ylim_main_level = None
                else:
                    lo, hi = float(min(yl)), float(max(yl))
                    sp = hi - lo
                    pad = max(0.06 * sp, 0.02) if sp > 0 else 0.05
                    ylim_main_level = (lo - pad, hi + pad)
            else:
                ylim_main_growth = None
                ylim_main_level = None

            for col_idx, gdp_form in enumerate(['growth', 'level']):
                ax = axes[model_idx, col_idx]
                if plot_type == 'main':
                    panel = bg if gdp_form == 'growth' else bl
                else:
                    panel = si_bundle_cache[(model_i, gdp_form)]
                color = panel['color']
                datasets = panel['datasets']
                positions = panel['positions']
                labels_use = panel['labels_use']
                star_x = panel['star_x']
                star_y = panel['star_y']
                star_ec = panel['star_ec']
                star_fc = panel['star_fc']

                if len(datasets) > 0:
                    vp = ax.violinplot(
                        datasets,
                        positions=positions,
                        widths=0.5,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    for b, body in enumerate(vp['bodies']):
                        pos = positions[b]
                        if pos == 1.0:
                            fc, ec, al = '0.25', 'black', 0.75
                        elif pos == 2.0:
                            fc, ec, al = color, color, 0.55
                        else:
                            fc, ec, al = color, color, 0.35
                        body.set_facecolor(fc)
                        body.set_edgecolor(ec)
                        body.set_alpha(al)
                        body.set_linewidth(1.2)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in vp and vp[partname] is not None:
                            vp[partname].set_visible(False)

                if len(star_x) > 0:
                    ax.scatter(
                        star_x,
                        star_y,
                        marker='*',
                        s=110,
                        zorder=5,
                        facecolors=star_fc,
                        edgecolors=star_ec,
                        linewidths=1.4,
                    )

                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
                if len(positions) > 0:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels_use, rotation=12, ha='right')
                ax.set_xlim(0.35, 3.65)

                if plot_type == 'main':
                    ylim_row = ylim_main_growth if gdp_form == 'growth' else ylim_main_level
                else:
                    ylim_row = ylim_si_growth if gdp_form == 'growth' else ylim_si_level
                if ylim_row is not None:
                    ax.set_ylim(ylim_row)
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=4))
                else:
                    ax.set_ylim(ylim_cfg[gdp_form]['lim'])
                    ax.set_yticks(ylim_cfg[gdp_form]['ticks'])

                ax.set_box_aspect(1)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{10**y:.2f}'))
                ax.set_ylabel('GPP / BGC ratio')
                ax.set_title(f'{model_i}  —  {gdp_form}  effect  (2080–2100)')

        # plt.show()
        plt.savefig(f'gpp_country_violin_{plot_type}.svg')
        plt.clf()
