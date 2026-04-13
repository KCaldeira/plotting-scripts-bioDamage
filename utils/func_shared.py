import pandas as pd, numpy as np, os
import xcdat as xc, xarray as xr
from scipy.signal import butter, filtfilt

def get_dummy_reg(var, pf, drop_first=True):
    var_dummy = pd.get_dummies(var, prefix=pf, drop_first=drop_first, dtype=int) 
    return var_dummy

def construct_file_name(model_name, grid_type, case_name, var):
    return f'{grid_type}_{str(var)}_{model_name}_{case_name}.nc'

def get_netcdf_var(filename): 
    return xc.open_dataset(filename, decode_times=False) 
    
def calculate_growth(df_base, key_old, key_new, geo_level):
    pd_growth = df_base[['model', 'year', 'region', key_old]].copy()
    pd_growth.sort_values(by=['model', 'region', 'year'], inplace=True)
    pd_growth[key_new] = pd_growth.groupby(['model', 'region'])[key_old].transform(lambda x: x.diff())
    pd_growth.drop(columns=[key_old], inplace=True)
    return pd_growth 

def calculate_growth_ln(df_base, key_old, key_new): 
    pd_growth = df_base[['model', 'year', 'region', key_old]].copy()
    pd_growth.sort_values(by=['model', 'region', 'year'], inplace=True)
    pd_growth[key_new] = pd_growth.groupby(['model', 'region'])[key_old].transform(lambda x: np.log(x).diff())
    pd_growth.drop(columns=[key_old], inplace=True) 
    return pd_growth

def get_dummy_pre(var, var_name_list, pd_length, mode='fre'):
    df_var = var[var_name_list]
    column_sums = df_var.sum()
    total_num = sum(column_sums)
    column_ratios = column_sums / total_num
    if pd_length > 1: new_dummies = np.tile(column_ratios, (pd_length, 1))
    new_dummies = pd.DataFrame(new_dummies, columns=var_name_list)
    return new_dummies 

def get_cell_area(lat, lon): 
    #### Cell area 
    bounds_lat = lat.getBounds()
    bounds_lon = lon.getBounds()
    R_earth = 6.371*10**6
    cell_areas = np.zeros((len(lat),len(lon)))
    for ii in range(len(lat)):
        for jj in range(len(lon)):
            cell_areas[ii,jj] = 2*np.pi*R_earth**2*np.absolute(np.sin(bounds_lat[ii,1]*np.pi/180)-np.sin(bounds_lat[ii,0]*np.pi/180))*np.absolute(bounds_lon[jj,1]-bounds_lon[jj,0])/360
    cell_areas = cell_areas / 1e6
    return cell_areas 

def butter_filter(series, cutoff, order=3, btype='low'):
    """Apply Butterworth filter to a 1D pandas Series."""
    series = series.dropna()
    if len(series) < (order * 3):  # too short
        return pd.Series(np.nan, index=series.index)
    
    b, a = butter(order, cutoff, btype=btype)
    y = filtfilt(b, a, series.values)
    return pd.Series(y, index=series.index)

def lowpass(series, cutoff=1/10, order=3):
    return butter_filter(series, cutoff, order, btype='low')

def highpass(series, cutoff=1/10, order=3):
    return butter_filter(series, cutoff, order, btype='high')

def get_land_ocean_areacella(model_to_examine, model_data, analysis_data):
    land_ocean_areacella_fileName = f'land_ocean_areacella_ds_{model_to_examine}.nc'
    if land_ocean_areacella_fileName in os.listdir(analysis_data + 'sub_region_masks'):
        ds_land_ocean_areacella = xc.open_dataset(os.path.join(analysis_data, 'sub_region_masks', land_ocean_areacella_fileName), 
                                                decode_times=False)
    else:
        #### Land mask and ocean mask from sftlf file
        sftlf_file_name = f'{model_data}sftlf_fx_{model_to_examine}_piControl.nc' 
        ds_sftlf = xc.open_dataset(sftlf_file_name, decode_times=False) 
        sftlf = ds_sftlf['sftlf'] 
        land_mask = xr.where(sftlf >= 50.0, 1.0, 0.0)
        ocean_mask = xr.where(sftlf < 50.0, 1.0, 0.0) 
        #### Cell area from areacella file 
        areacella_file_name = f'{model_data}areacella_fx_{model_to_examine}_piControl.nc' 
        ds_areacella = xc.open_dataset(areacella_file_name, decode_times=False) 
        areacella = ds_areacella['areacella']
        #### Store these variables in xArray dataset
        ds_land_ocean_areacella = xr.Dataset({
                "land_mask": land_mask,
                "ocean_mask": ocean_mask,
                "areacella": areacella},
                coords={
                    "lat": ds_sftlf["lat"],
                    "lon": ds_sftlf["lon"],
                    "lat_bnds": (("lat", "bnds"), ds_sftlf["lat_bnds"].data),
                    "lon_bnds": (("lon", "bnds"), ds_sftlf["lon_bnds"].data)})
        #### Save to nc files
        ds_land_ocean_areacella.to_netcdf(os.path.join(analysis_data, 'sub_region_masks', land_ocean_areacella_fileName))
    return ds_land_ocean_areacella 

def get_years_from_scenario(scenario_name, time_length):
    if scenario_name in ['historical', 'hist-bgc']:
        years = np.arange(1850, 2015)
    elif scenario_name in ['ssp585', 'ssp585-bgc']:
        years = np.arange(2015, 2101)
    elif scenario_name in ['1pctCO2', '1pctCO2-bgc']:
        years = np.arange(1850, 1850 + time_length)
    elif scenario_name == 'piControl':
        years = np.arange(time_length) - time_length + 1850
    return years

def apply_mask_and_average(ds, var_name, mask):
    ds_masked = ds.copy()
    ds_masked[var_name] = ds_masked[var_name] * mask
    return ds_masked.spatial.average(var_name)[var_name].values

def calculate_lai_weighted_average(var_values, lai_m2, mask):
    masked_var = var_values * mask
    masked_lai_m2 = lai_m2 * mask
    weighted_sum = np.sum(masked_var * masked_lai_m2, axis=(1, 2))
    total_lai = np.sum(masked_lai_m2, axis=(1, 2)) 
    return np.where(total_lai > 0, weighted_sum / total_lai, np.nan)

def calculate_lai_weighted_total(var_values, lai_values, area_cell, mask):
    var_total = var_values * area_cell * mask
    lai_m2 = lai_values * area_cell * mask
    total_var = np.sum(var_total, axis=(1, 2))
    total_lai = np.sum(lai_m2, axis=(1, 2))
    return np.where(total_lai > 0, total_var / total_lai, np.nan) 

def shared_process(data_subset, target_variable):
    #### Get full year information before any actions 
    years = np.sort(np.unique(data_subset['year'].values.tolist())) 
    #### Remove nan, inf, and abnormal values based on spatial level
    data_subset = data_subset.dropna() 
    #### For region-scale analysis, consider only regions with full-year data 
    full_years = set(years) 
    grouped = data_subset.groupby(['model', 'region']) 
    data_subset = grouped.filter(lambda x: set(x['year']) >= full_years) 
    #### Calculate percentage growth based on spatial level
    gpp_growth_pct = calculate_growth_ln(data_subset, target_variable, 'pct_growth_' + target_variable) 
    data_subset = data_subset.merge(gpp_growth_pct, on=['model', 'year', 'region'], how='left')  
    return data_subset

def define_default_parameters():
    return {
        "model_list": ["ACCESS-ESM1-5", "CNRM-ESM2-1", "MIROC-ES2L"],
        "grid_type": "gridRaw",
        "spatial_level": "country",
        "pr_scale": "ln", 
        "gpp_filter": False,
        "gpp_threshold": 0.0,
        "gpp_threshold_sign": 1,
        "tas_filter": False,
        "tas_threshold": 0.0,
        "tas_threshold_sign": 1, 
        "force_redo_entireflow": False,
        "force_redo_regression": False,
        "force_redo_projection": False,
        "impact_contribution": "full", 
        "newell_combination_choice": "all800", 
        "target_variable": "gpp",
        "weighting_method": "area",
        "path_root": "/Users/duanlei/Desktop/File/Research/Github_local/DataCenter/clab_burke_biogeophysical_variables/",
        "reduce_time_step": False, 
        "reduce_time_step_value": -1,
        "reduce_time_step_method": "mean",
        "bootstrap_num": 0, 
        "add_constant": False,
        "country_distribution_selection_method": "percentile",
        }