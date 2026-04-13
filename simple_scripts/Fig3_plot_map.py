import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import numpy as np
import pickle
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent

_LN10 = np.log(10)


def _spectral_white_center_cmap():
    """Spectral-like ramp with true white at the midpoint (log10 ratio = 0 -> ratio 1)."""
    spec = plt.colormaps['Spectral']
    stops = [
        (0.0, spec(0.02)),
        (0.40, spec(0.34)),
        (0.5, (1.0, 1.0, 1.0)),
        (0.60, spec(0.66)),
        (1.0, spec(0.98)),
    ]
    pairs = [(x, tuple(np.asarray(c, dtype=float)[:3])) for x, c in stops]
    return colors.LinearSegmentedColormap.from_list('Spectral_white_center', pairs, N=256)

GPP_COUNTRY_MAP_CMAP = _spectral_white_center_cmap()


def _ratio_pct_colorbar_params(rmin, rmax):
    """Generate colorbar norm, ticks, and labels from ratio range using the utility."""
    bounds_ln, ticks_ln, pct_labels = get_axis_bounds_and_ticks_ratio_pct([rmin, rmax])
    bounds_log10 = [b / _LN10 for b in bounds_ln]
    ticks_log10 = [t / _LN10 for t in ticks_ln]
    labels = [format_percent(p) for p in pct_labels]
    norm = colors.Normalize(vmin=bounds_log10[0], vmax=bounds_log10[1])
    return norm, ticks_log10, labels


def add_panel(ax, data, norm, ticks, labels, title, lon, lat):
    mp = ax.pcolormesh(lon, lat, data, cmap=GPP_COUNTRY_MAP_CMAP, shading='auto', norm=norm, transform=ccrs.PlateCarree())
    cb = plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal', ticks=ticks)
    cb.set_ticklabels(labels)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.set_title(title)


def fig3_plot_map():

    dict_region_info = pickle.load(open(f'./data/input/figure3_map_data.pickle', 'rb'))

    model_list = dict_region_info.keys()
    n_models = len(model_list)

    #### Growth scale: ratios ~0.1 to ~10 (log10 range [-1, 1])
    norm1, ticks1, labels1 = _ratio_pct_colorbar_params(0.1, 10.0)

    #### Level scale: ratios ~0.5 to ~2 (log10 range [-0.3, 0.3])
    norm2, ticks2, labels2 = _ratio_pct_colorbar_params(0.5, 2.0)

    for model_i in model_list:

        dict_model_region_info = dict_region_info[model_i]

        #### Get region info based on Burke_growth
        model_log10ratio = dict_model_region_info['model_log10ratio']
        burke_growth_log10ratio = dict_model_region_info['burke_growth_log10ratio']
        burke_level_log10ratio = dict_model_region_info['burke_level_log10ratio']
        newell_growth_log10ratio = dict_model_region_info['newell_growth_log10ratio']
        newell_level_log10ratio = dict_model_region_info['newell_level_log10ratio']
        lat, lon = dict_model_region_info['lat'], dict_model_region_info['lon']

        fig, axes = plt.subplots(3, 2, figsize=(11, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
        add_panel(axes[0, 0], model_log10ratio,         norm1, ticks1, labels1, 'model_log10ratio',         lon, lat)
        add_panel(axes[0, 1], model_log10ratio,         norm2, ticks2, labels2, 'model_log10ratio',         lon, lat)
        add_panel(axes[1, 0], burke_growth_log10ratio,  norm1, ticks1, labels1, 'burke_growth_log10ratio',  lon, lat)
        add_panel(axes[1, 1], burke_level_log10ratio,   norm2, ticks2, labels2, 'burke_level_log10ratio',   lon, lat)
        add_panel(axes[2, 0], newell_growth_log10ratio, norm1, ticks1, labels1, 'newell_growth_log10ratio', lon, lat)
        add_panel(axes[2, 1], newell_level_log10ratio,  norm2, ticks2, labels2, 'newell_level_log10ratio',  lon, lat)

        fig.savefig(f'./data/output/Fig3_map_{model_i}.pdf', dpi=300)
        plt.close(fig)
