import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from utils.func_shared import get_land_ocean_areacella


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


def assign_ratio_map(ratio_per_country, lat, lon, region_list, reg_lookup, reg_mask):
    ratio_per_country = np.array(ratio_per_country)
    shape = np.zeros([len(lat), len(lon)])
    for reg_i in region_list:
        count = region_list.index(reg_i)
        if np.isnan(ratio_per_country[count]):
            continue
        reg_number = reg_lookup.loc[reg_lookup['name'] == reg_i, 'number'].tolist()
        mask_i = np.copy(reg_mask)
        mask_i = np.ma.filled(np.ma.masked_not_equal(mask_i, reg_number) * 0 + 1, 0)
        shape  = shape + ratio_per_country[count] * mask_i
    return shape

def to_log10_map(ratio_map, land_mask):
    ratio_masked = np.ma.masked_where(ratio_map == 0, ratio_map) * land_mask
    return np.ma.log10(ratio_masked)

def add_panel(ax, data, norm, ticks, labels, title, lon, lat):
    mp = ax.pcolormesh(lon, lat, data, cmap=GPP_COUNTRY_MAP_CMAP, shading='auto', norm=norm, transform=ccrs.PlateCarree())
    cb = plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal', ticks=ticks)
    cb.set_ticklabels(labels)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.set_title(title)


def burke_subType(self, ylim_country):

    results_dict  = self.results_dict
    model_list    = self.model_list
    model_data    = self.path_root + 'CMIP6_annual/'
    analysis_data = self.path_root + 'analysisOutput/'

    for model_i in model_list:

        #### Get masks
        ds_land_ocean_areacella = get_land_ocean_areacella(model_i, model_data, analysis_data)
        land_mask = ds_land_ocean_areacella['land_mask'].values
        land_mask = np.ma.masked_equal(land_mask, 0)

        proj_growth = results_dict['burke_growth'][model_i]['projection_main']
        proj_level  = results_dict['burke_level'][model_i]['projection_main']

        years            = np.array(proj_growth['years'])
        region_list      = proj_growth['region_list']
        reg_lookup       = proj_growth['reg_lookup']
        reg_mask         = proj_growth['reg_mask']
        lat              = proj_growth['lat']
        lon              = proj_growth['lon']
        model_projection = np.array(proj_growth['model_simulation_projection'])
        model_references = np.array(proj_growth['model_simulation_references'])

        #### Year indices for 2080–2100
        idx_start = np.searchsorted(years, 2080)
        idx_end   = np.searchsorted(years, 2101)   # exclusive upper bound → includes 2100

        #### Per-country mean GPP over 2080–2100 and ratio relative to BGC reference
        ref_mean    = np.mean(model_references[:, idx_start:idx_end], axis=1)
        model_ratio = np.mean(model_projection[:, idx_start:idx_end], axis=1) / ref_mean
        growth_ratio = np.mean(np.array(proj_growth['empirical_projection_corrected'])[:, idx_start:idx_end], axis=1) / ref_mean
        level_ratio  = np.mean(np.array(proj_level['empirical_projection_corrected'])[:, idx_start:idx_end],  axis=1) / ref_mean

        model_log10  = to_log10_map(assign_ratio_map(model_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask)
        growth_log10 = to_log10_map(assign_ratio_map(growth_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask)
        level_log10  = to_log10_map(assign_ratio_map(level_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask)

        #### Color scale definitions
        norm1   = colors.Normalize(vmin=-1.0, vmax=1.0)
        ticks1  = [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0]
        labels1 = [f'{10**t:.2f}' for t in ticks1]

        norm2   = colors.Normalize(vmin=-0.3, vmax=0.3)
        ticks2  = [-0.3, -0.15, 0, 0.15, 0.3]
        labels2 = [f'{10**t:.2f}' for t in ticks2]        

        fig, axes = plt.subplots(2, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
        axes = axes.ravel()

        add_panel(axes[0], model_log10,  norm1, ticks1, labels1, f'{model_i}  —  (A) Model  [growth scale]  GPP ratio 2080–2100', lon, lat)
        add_panel(axes[1], model_log10,  norm2, ticks2, labels2, f'{model_i}  —  (B) Model  [level scale]  GPP ratio 2080–2100', lon, lat)
        add_panel(axes[2], growth_log10, norm1, ticks1, labels1, f'{model_i}  —  (C) Growth effect  GPP ratio 2080–2100', lon, lat)
        add_panel(axes[3], level_log10,  norm2, ticks2, labels2, f'{model_i}  —  (D) Level effect  GPP ratio 2080–2100', lon, lat)

        plt.show()
        # plt.savefig(f'burke_gpp_countries_map_{model_i}.pdf', dpi=300)
        # plt.clf()


def newell_subType(self, ylim_country):

    results_dict  = self.results_dict
    model_list    = self.model_list
    model_data    = self.path_root + 'CMIP6_annual/'
    analysis_data = self.path_root + 'analysisOutput/'

    for model_i in model_list:

        #### Get masks
        ds_land_ocean_areacella = get_land_ocean_areacella(model_i, model_data, analysis_data)
        land_mask = ds_land_ocean_areacella['land_mask'].values
        land_mask = np.ma.masked_equal(land_mask, 0)

        #### Spatial info and model simulation come from spec0 of the first action
        results_modelI   = results_dict["newell_all800"][model_i]
        spec0            = results_modelI[0]
        years            = np.array(spec0['years'])
        region_list      = spec0['region_list']
        reg_lookup       = spec0['reg_lookup']
        reg_mask         = spec0['reg_mask']
        lat              = spec0['lat']
        lon              = spec0['lon']
        model_projection = np.array(spec0['model_simulation_projection'])
        model_references = np.array(spec0['model_simulation_references'])

        #### Year indices for 2080–2100
        idx_start = np.searchsorted(years, 2080)
        idx_end   = np.searchsorted(years, 2101)   # exclusive → includes 2100

        ref_mean    = np.mean(model_references[:, idx_start:idx_end], axis=1)
        model_ratio = np.mean(model_projection[:, idx_start:idx_end], axis=1) / ref_mean

        #### Per-country median ratio across growth / level specs
        growth_specs = [s for s in results_modelI if 'growth' in s['spec']['gdp_form']]
        level_specs  = [s for s in results_modelI if 'level'  in s['spec']['gdp_form']]

        growth_ratio = np.median(np.array([
            np.mean(np.array(s['empirical_projection_corrected'])[:, idx_start:idx_end], axis=1) / ref_mean
            for s in growth_specs
        ]), axis=0)   # [n_countries]

        level_ratio = np.median(np.array([
            np.mean(np.array(s['empirical_projection_corrected'])[:, idx_start:idx_end], axis=1) / ref_mean
            for s in level_specs
        ]), axis=0)   # [n_countries]

        model_log10  = to_log10_map(assign_ratio_map(model_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask)
        growth_log10 = to_log10_map(assign_ratio_map(growth_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask)
        level_log10  = to_log10_map(assign_ratio_map(level_ratio, lat, lon, region_list, reg_lookup, reg_mask), land_mask) 

        #### Same color scales as burke_gpp_countries_map
        norm1   = colors.Normalize(vmin=-1.0, vmax=1.0)
        ticks1  = [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0] 
        labels1 = [f'{10**t:.2f}' for t in ticks1]

        norm2   = colors.Normalize(vmin=-0.3, vmax=0.3)
        ticks2  = [-0.3, -0.15, 0, 0.15, 0.3]
        labels2 = [f'{10**t:.2f}' for t in ticks2]        

        fig, axes = plt.subplots(2, 2, figsize=(14, 6),
                                 subplot_kw={'projection': ccrs.PlateCarree()},
                                 constrained_layout=True)
        axes = axes.ravel()

        add_panel(axes[0], model_log10,  norm1, ticks1, labels1, f'{model_i}  —  (A) Model  [growth scale]  GPP ratio 2080–2100', lon, lat)
        add_panel(axes[1], model_log10,  norm2, ticks2, labels2, f'{model_i}  —  (B) Model  [level scale]  GPP ratio 2080–2100', lon, lat)
        add_panel(axes[2], growth_log10, norm1, ticks1, labels1, f'{model_i}  —  (C) Growth effect  GPP ratio 2080–2100  (median of {len(growth_specs)} specs)', lon, lat)
        add_panel(axes[3], level_log10,  norm2, ticks2, labels2, f'{model_i}  —  (D) Level effect  GPP ratio 2080–2100  (median of {len(level_specs)} specs)', lon, lat)

        plt.show()
        # plt.savefig(f'newell_gpp_countries_map_{model_i}.pdf', dpi=300)
        # plt.clf()



def gpp_country_map(self):
    # ylim_country = _compute_shared_ylim_country_distribution(self, plot_type, weight_mode)
    ylim_country = ''
    burke_subType(self, ylim_country)
    newell_subType(self, ylim_country)
