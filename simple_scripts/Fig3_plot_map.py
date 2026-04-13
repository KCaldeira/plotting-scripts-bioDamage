import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import numpy as np
import pickle


def _spectral_white_center_cmap():
    """Spectral-like ramp with true white at the midpoint (log10 ratio = 0 → ratio 1)."""
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


def add_panel(ax, data, norm, ticks, labels, title, lon, lat):
    mp = ax.pcolormesh(lon, lat, data, cmap=GPP_COUNTRY_MAP_CMAP, shading='auto', norm=norm, transform=ccrs.PlateCarree())
    cb = plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal', ticks=ticks)
    cb.set_ticklabels(labels)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.set_title(title)


def fig3_plot_map():

    dict_region_info = pickle.load(open(f'./simple_scripts/figure3_map_data.pickle', 'rb'))

    model_list = dict_region_info.keys()
    n_models = len(model_list)

    #### Pre-defined growth scale color scale
    norm1   = colors.Normalize(vmin=-1.0, vmax=1.0)
    ticks1  = [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0] 
    labels1 = [f'{10**t:.2f}' for t in ticks1]

    #### Pre-defined level scale color scale
    norm2   = colors.Normalize(vmin=-0.3, vmax=0.3)
    ticks2  = [-0.3, -0.15, 0, 0.15, 0.3]
    labels2 = [f'{10**t:.2f}' for t in ticks2]        

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
 
        plt.show()
        # plt.savefig('burke_gpp_countries_map.pdf', dpi=300)
        # plt.clf()