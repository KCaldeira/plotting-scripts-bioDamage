import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from utils.func_shared import get_land_ocean_areacella


gdp_form_color = {'growth': 'firebrick', 'level': 'royalblue'}


def _plot_diagnoal_line(ax, ticks):
    # 1:1 line 
    ax.plot([ticks[0], ticks[-1]], [ticks[0], ticks[-1]], color='gray', linestyle=':', linewidth=1, zorder=1) 
    # 1:2 line
    # ax.plot([ticks[0], ticks[-1]], [ticks[0] * 2, ticks[-1] * 2], color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.plot([ticks[0], ticks[-1]], [ticks[0] + np.log10(2), ticks[-1] + np.log10(2)], color='gray', linestyle=':', linewidth=1, zorder=1)
    # 1:4 line
    # ax.plot([ticks[0], ticks[-1]], [ticks[0] * 4, ticks[-1] * 4], color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.plot([ticks[0], ticks[-1]], [ticks[0] + np.log10(4), ticks[-1] + np.log10(4)], color='gray', linestyle=':', linewidth=1, zorder=1)
    # 2:1 line
    # ax.plot([ticks[0], ticks[-1]], [ticks[0] / 2, ticks[-1] / 2], color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.plot([ticks[0], ticks[-1]], [ticks[0] - np.log10(2), ticks[-1] - np.log10(2)], color='gray', linestyle=':', linewidth=1, zorder=1)
    # 4:1 line
    # ax.plot([ticks[0], ticks[-1]], [ticks[0] / 4, ticks[-1] / 4], color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.plot([ticks[0], ticks[-1]], [ticks[0] - np.log10(4), ticks[-1] - np.log10(4)], color='gray', linestyle=':', linewidth=1, zorder=1)
    # Horizontal and vertical lines 
    ax.axhline(0, color='black', linestyle='--', linewidth=1, zorder=5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=5)


# color_range = (-1.0, 1.0)
# cmap = plt.colormaps['managua']
# norm = colors.TwoSlopeNorm(vmin=color_range[0], vcenter=0.0, vmax=color_range[1])

color_range = (-0.2, 0.2)
cmap = plt.colormaps['managua']
norm = colors.TwoSlopeNorm(vmin=color_range[0], vcenter=0.0, vmax=color_range[1])


def burke_subType(self, ylim_country):

    results_dict = self.results_dict
    action_names = list(results_dict.keys())
    action_names = [a for a in action_names if a.startswith('burke_')] 
    model_list   = self.model_list

    #### Axis config: same log10 ranges as burke_gpp_countries_map colorbars
    axis_cfg = {
        'growth': [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0],
        'level':  [-0.6, -0.3, 0, 0.3],
    }

    n_models  = len(model_list)
    n_actions = len(action_names)
    fig, axes = plt.subplots(n_models, n_actions, figsize=(4 * n_actions, 4 * n_models), constrained_layout=True)
    axes = np.array(axes).reshape(n_models, n_actions)

    for model_idx, model_i in enumerate(model_list):
        for action_idx, action_name in enumerate(action_names):

            ax = axes[model_idx, action_idx]

            projection_central = results_dict[action_name][model_i]['projection_main']
            years              = np.array(projection_central['years'])
            model_projection   = np.array(projection_central['model_simulation_projection'])
            model_references   = np.array(projection_central['model_simulation_references'])
            emp_corrected      = np.array(projection_central['empirical_projection_corrected'])
            weights_projection = np.array(projection_central['weights_projection'])
            areas              = weights_projection[:, 0]

            #### Mean GPP over 2080–2100
            idx_start = np.searchsorted(years, 2080)
            idx_end   = np.searchsorted(years, 2101)
            ref_mean  = np.mean(model_references[:, idx_start:idx_end], axis=1)

            X = np.log10(np.mean(emp_corrected[:, idx_start:idx_end],  axis=1) / ref_mean)
            Y = np.log10(np.mean(model_projection[:, idx_start:idx_end], axis=1) / ref_mean)

            #### Clip to 5th–95th percentile to remove extreme outliers
            for arr in (X, Y):
                p5, p95 = np.nanpercentile(arr, [5, 95])
                arr[arr < p5]  = np.nan
                arr[arr > p95] = np.nan

            valid         = ~np.isnan(X) & ~np.isnan(Y)
            X_v, Y_v      = X[valid], Y[valid]
            areas_v       = areas[valid]
            scatter_sizes = areas_v / np.max(areas_v) * 500

            ratio_y_x = 10.0 ** (Y_v - X_v)
            color_vals = np.log10(ratio_y_x)
            print (np.min(color_vals), np.max(color_vals))

            ax.scatter(X_v, Y_v, s=scatter_sizes, c=color_vals, cmap=cmap, norm=norm,
                       alpha=0.6, edgecolors='black', linewidth=0.5)

            #### Axis style
            gdp_form = 'growth' if 'growth' in action_name else 'level' 
            ticks  = axis_cfg[gdp_form]
            labels = [f'{10**t:.2f}' for t in ticks]

            _plot_diagnoal_line(ax, ticks)

            ax.set_xlim(ticks[0], ticks[-1])
            ax.set_ylim(ticks[0], ticks[-1])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_aspect('equal')
            ax.set_xlabel('Empirical GPP ratio  (full / BGC)')
            ax.set_ylabel('Model GPP ratio  (full / BGC)')
            ax.set_title(f'{model_i}  —  {action_name}  (2080–2100)')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.02, shrink=0.5,
                 label='log10(model / empirical)')

    plt.show()
    # plt.savefig('burke_gpp_countries_scatter.svg')
    # plt.clf()



def newell_subType(self, ylim_country):

    results_dict = self.results_dict
    action_names = list(results_dict.keys())
    action_names = [a for a in action_names if a.startswith('newell_')]
    model_list   = self.model_list
    gdp_forms    = list(gdp_form_color.keys())   # ['growth', 'level']

    #### Same axis config as burke_gpp_countries_scatter / burke_gpp_countries_map
    axis_cfg = {
        'growth': [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0],
        'level':  [-0.6, -0.3, 0, 0.3],
    }


    #### Layout: rows = models,  cols = action × gdp_form  (growth then level per action)
    n_models  = len(model_list)
    n_cols    = len(action_names) * len(gdp_forms)
    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(4 * n_cols, 4 * n_models),
                             constrained_layout=True)
    axes = np.array(axes).reshape(n_models, n_cols)

    for model_idx, model_i in enumerate(model_list):
        for action_idx, action_name in enumerate(action_names):

            results_modelI = results_dict[action_name][model_i]
            spec0          = results_modelI[0]

            years            = np.array(spec0['years'])
            model_projection = np.array(spec0['model_simulation_projection'])
            model_references = np.array(spec0['model_simulation_references'])
            weights          = np.array(spec0['weights_projection'])
            areas            = weights[:, 0]

            #### Mean GPP over 2080–2100
            idx_start = np.searchsorted(years, 2080)
            idx_end   = np.searchsorted(years, 2101)
            ref_mean  = np.mean(model_references[:, idx_start:idx_end], axis=1)

            Y = np.log10(np.mean(model_projection[:, idx_start:idx_end], axis=1) / ref_mean)

            for form_idx, gdp_form in enumerate(gdp_forms):

                col_idx = action_idx * len(gdp_forms) + form_idx
                ax      = axes[model_idx, col_idx]

                #### Empirical: per-country median log10 ratio across matching specs
                form_specs = [s for s in results_modelI if gdp_form in s['spec']['gdp_form']]
                X = np.median(np.array([
                    np.log10(np.mean(np.array(s['empirical_projection_corrected'])[:, idx_start:idx_end], axis=1) / ref_mean)
                    for s in form_specs
                ]), axis=0)   # [n_countries]

                #### Clip to 5th–95th percentile
                X_plot = X.copy()
                Y_plot = Y.copy()
                for arr in (X_plot, Y_plot):
                    p5, p95 = np.nanpercentile(arr, [5, 95])
                    arr[arr < p5]  = np.nan
                    arr[arr > p95] = np.nan

                valid         = ~np.isnan(X_plot) & ~np.isnan(Y_plot)
                X_v, Y_v      = X_plot[valid], Y_plot[valid]
                areas_v       = areas[valid]
                scatter_sizes = areas_v / np.max(areas_v) * 500

                ratio_y_x = 10.0 ** (Y_v - X_v)
                color_vals = np.log10(ratio_y_x)
                print (np.min(color_vals), np.max(color_vals))

                ax.scatter(X_v, Y_v, s=scatter_sizes, c=color_vals, cmap=cmap, norm=norm,
                           alpha=0.6, edgecolors='black', linewidth=0.5)

                ticks  = axis_cfg[gdp_form]
                labels = [f'{10**t:.2f}' for t in ticks]

                _plot_diagnoal_line(ax, ticks)

                ax.set_xlim(ticks[0], ticks[-1])
                ax.set_ylim(ticks[0], ticks[-1])
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                ax.set_aspect('equal')
                ax.set_xlabel('Empirical GPP ratio  (full / BGC)')
                ax.set_ylabel('Model GPP ratio  (full / BGC)')
                ax.set_title(f'{model_i}  —  {action_name}  ({gdp_form},  {len(form_specs)} specs,  2080–2100)')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.02, shrink=0.5,
                 label='log10(model / empirical)')

    plt.show()
    # plt.savefig('newell_gpp_countries_scatter.svg')
    # plt.clf()


def gpp_country_scatter(self):
    # ylim_country = _compute_shared_ylim_country_distribution(self, plot_type, weight_mode)
    ylim_country = ''
    burke_subType(self, ylim_country)
    newell_subType(self, ylim_country)
