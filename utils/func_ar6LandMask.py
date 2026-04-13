import numpy as np, pandas as pd
import pickle, os, regionmask


def get_ar6_land_mask(ds_land_ocean_areacella, model_to_examine, analysis_data, check_plot=False): 
    ar6_land_mask_fileName = f'ar6_land_mask_{model_to_examine}.pickle' 
    if ar6_land_mask_fileName in os.listdir(analysis_data + 'sub_region_masks'):
        with open(os.path.join(analysis_data, 'sub_region_masks', ar6_land_mask_fileName), 'rb') as f:
            ar6_land_mask, ar6_land_lookup = pickle.load(f)
    else:
        lat_1d = ds_land_ocean_areacella["lat"].values
        lon_1d = ds_land_ocean_areacella["lon"].values
        ar6_land = regionmask.defined_regions.ar6.land
        #### Get ar6 land mask 
        ar6_land_mask = np.array(ar6_land.mask(lon_1d, lat_1d))
        whereisnan = np.isnan(ar6_land_mask) 
        ar6_land_mask = np.ma.masked_where(whereisnan, ar6_land_mask)
        #### Get ar6 land lookup table 
        ar6_land_lookup = pd.DataFrame({
            "name": ar6_land.names, 
            "number": ar6_land.numbers,
            "abbrev": ar6_land.abbrevs})
        #### Save to pickle file 
        with open(os.path.join(analysis_data, 'sub_region_masks', ar6_land_mask_fileName), 'wb') as f:
            pickle.dump([ar6_land_mask, ar6_land_lookup], f)
        #### Check masks
        if check_plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            mp = ax.pcolor(lon_1d, lat_1d, ar6_land_mask, cmap='tab20', transform=ccrs.PlateCarree())
            plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal') 
            plt.show() 
    return ar6_land_mask, ar6_land_lookup


def get_ar6_ocean_mask(ds_land_ocean_areacella, model_to_examine, analysis_data, check_plot=False): 
    ar6_ocean_mask_fileName = f'ar6_ocean_mask_{model_to_examine}.pickle' 
    if ar6_ocean_mask_fileName in os.listdir(analysis_data + 'sub_region_masks'):
        with open(os.path.join(analysis_data, 'sub_region_masks', ar6_ocean_mask_fileName), 'rb') as f:
            ar6_ocean_mask, ar6_ocean_lookup = pickle.load(f)
    else:
        lat_1d = ds_land_ocean_areacella["lat"].values
        lon_1d = ds_land_ocean_areacella["lon"].values
        ar6_ocean = regionmask.defined_regions.ar6.ocean
        #### Get ar6 ocean mask 
        ar6_ocean_mask = np.array(ar6_ocean.mask(lon_1d, lat_1d))
        whereisnan = np.isnan(ar6_ocean_mask) 
        ar6_ocean_mask = np.ma.masked_where(whereisnan, ar6_ocean_mask)
        #### Get ar6 ocean lookup table 
        ar6_ocean_lookup = pd.DataFrame({
            "name": ar6_ocean.names, 
            "number": ar6_ocean.numbers,
            "abbrev": ar6_ocean.abbrevs})
        #### Save to pickle file 
        with open(os.path.join(analysis_data, 'sub_region_masks', ar6_ocean_mask_fileName), 'wb') as f:
            pickle.dump([ar6_ocean_mask, ar6_ocean_lookup], f)
        #### Check masks
        if check_plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            mp = ax.pcolor(lon_1d, lat_1d, ar6_ocean_mask, cmap='tab20', transform=ccrs.PlateCarree())
            plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal') 
            plt.show() 
    return ar6_ocean_mask, ar6_ocean_lookup

