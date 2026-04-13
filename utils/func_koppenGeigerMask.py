import numpy as np, pandas as pd
import pickle, os
import xarray as xr
from scipy.stats import mode 


def check_lat_direction(lat, name): 
    lat_tar_dir = "increasing" if lat[0] < lat[-1] else "decreasing"
    print(f"{name} is {lat_tar_dir} (starts at {lat[0]}, ends at {lat[-1]})")
    return lat_tar_dir


def check_lon_range(lon, name):
    if np.any(lon < 0):
        lon_tar_range = "[-180, 180]"
    else:
        lon_tar_range = "[0, 360]" 
    print(f"{name} is in {lon_tar_range} (min: {lon.min()}, max: {lon.max()})")
    return lon_tar_range


def get_KoppenGeiger_mask(ds_land_ocean_areacella, model_to_examine, analysis_data, check_plot=False): 

    kg_mask_fileName = f'KoppenGeiger_mask_{model_to_examine}.pickle' 

    if kg_mask_fileName in os.listdir(analysis_data + 'sub_region_masks/'):
        with open(os.path.join(analysis_data, 'sub_region_masks', kg_mask_fileName), 'rb') as f:
            kg_mask, kg_lookup = pickle.load(f)
    
    else:
        #### Get lat/lon from target dataset
        lat_tar = ds_land_ocean_areacella["lat"].values
        lon_tar = ds_land_ocean_areacella["lon"].values

        #### Step 1, get the Koppen-Geiger class data in source resolution:
        fname = f'{analysis_data}sub_region_maps/koppen_geiger_nc/1991_2020/koppen_geiger_0p1.nc' 
        ds_kg = xr.open_dataset(fname)
        kg_class = ds_kg['kg_class'].values.squeeze()
        lat_src = ds_kg['lat'].values
        lon_src = ds_kg['lon'].values
        ds_kg.close()

        #### Step 2, check the direction of latitude:
        lat_tar_dir = check_lat_direction(lat_tar, 'lat_tar') 
        lat_src_dir = check_lat_direction(lat_src, 'lat_src')
        lat_src_transfer = lat_src.copy()
        if lat_src_dir != lat_tar_dir: 
            print("Flipping lat_src and corresponding var_src to match lat_tar direction...")
            lat_src_transfer = lat_src[::-1]
            kg_class = np.flip(kg_class, axis=0)

        #### Step 3, check the direction of longitude: 
        lon_tar_range = check_lon_range(lon_tar, 'lon_tar')
        lon_src_range = check_lon_range(lon_src, 'lon_src')
        lon_src_transfer = lon_src.copy()
        if lon_src_range != lon_tar_range:
            print("Converting lon_src and reordering var_src to match lon_tar range...")
            if lon_tar_range == "[0, 360]":
                lon_src_transfer = (lon_src + 360) % 360
            else:
                lon_src_transfer = (lon_src + 180) % 360 - 180
            sort_idx = np.argsort(lon_src_transfer)
            lon_src_transfer = lon_src_transfer[sort_idx]
            kg_class = kg_class[:, sort_idx]

        #### Step 4, do the mode-based re-mapping: 
        kg_class_tar = np.full((len(lat_tar), len(lon_tar)), fill_value=np.nan) 
        lat_res = np.abs(lat_tar[1] - lat_tar[0]) 
        lon_res = np.abs(lon_tar[1] - lon_tar[0]) 
        for i in range(lat_tar.shape[0]): 
            for j in range(lon_tar.shape[0]):
                lat_c, lon_c = lat_tar[i], lon_tar[j] 
                lat_min, lat_max = lat_c - lat_res/2, lat_c + lat_res/2
                lon_min, lon_max = lon_c - lon_res/2, lon_c + lon_res/2
                lat_idx = np.where((lat_src_transfer >= lat_min) & (lat_src_transfer <= lat_max))[0]
                lon_idx = np.where((lon_src_transfer >= lon_min) & (lon_src_transfer <= lon_max))[0] 
                values_in_box = kg_class[np.ix_(lat_idx, lon_idx)].flatten() 
                valid_values = values_in_box[~np.isnan(values_in_box)]
                if valid_values.size > 0: 
                    mode_result = mode(valid_values, keepdims=False)
                    mode_value = mode_result.mode
                    if np.isscalar(mode_value):
                        kg_class_tar[i, j] = mode_value
                    elif mode_value.size > 0:
                        kg_class_tar[i, j] = mode_value.item() if mode_value.ndim == 0 else mode_value[0]
        whereisnan = np.isnan(kg_class_tar) 
        kg_class_tar = np.ma.masked_where(whereisnan, kg_class_tar)
        kg_mask = np.ma.masked_equal(kg_class_tar, 0) 

        #### Step 5, specify names and create lookup table: 
        kg_names = ['Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 
                    'Dsa', 'Dsb', 'Dsc', 'Dsd', 'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF']
        kg_lookup = pd.DataFrame({
            "name": kg_names,
            "number": list(range(1, len(kg_names) + 1))
        })

        #### Save to pickle file
        with open(os.path.join(analysis_data, 'sub_region_masks', kg_mask_fileName), 'wb') as f: 
            pickle.dump([kg_mask, kg_lookup], f) 

        #### Check masks
        check_plot = False
        if check_plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
            mp = ax.pcolor(lon_tar, lat_tar, kg_mask, cmap='viridis', transform=ccrs.PlateCarree())
            plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal') 
            plt.show() 

    return kg_mask, kg_lookup 