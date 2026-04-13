import numpy as np, pandas as pd
import pickle, os, regionmask

def get_country_mask(ds_land_ocean_areacella, model_to_examine, analysis_data, check_plot=False): 
    country_mask_fileName = f'country_mask_{model_to_examine}.pickle' 
    if country_mask_fileName in os.listdir(analysis_data + 'sub_region_masks'):
        with open(os.path.join(analysis_data, 'sub_region_masks', country_mask_fileName), 'rb') as f:
            country_mask, country_lookup = pickle.load(f)
    else:
        lat_1d = ds_land_ocean_areacella["lat"].values  # shape (nlat,)
        lon_1d = ds_land_ocean_areacella["lon"].values  # shape (nlon,)
        countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
        #### Get country mask 
        country_mask = np.array(countries.mask(lon_1d, lat_1d))
        whereisnan = np.isnan(country_mask) 
        country_mask = np.ma.masked_where(whereisnan, country_mask)
        #### Get country lookup table 
        country_lookup = pd.DataFrame({
            "name": countries.names, 
            "number": countries.numbers,
            "abbrev": countries.abbrevs})
        #### Save to pickle file 
        with open(os.path.join(analysis_data, 'sub_region_masks', country_mask_fileName), 'wb') as f:
            pickle.dump([country_mask, country_lookup], f)
        #### Check masks
        if check_plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
            mp = ax.pcolor(lon_1d, lat_1d, country_mask, cmap='viridis', transform=ccrs.PlateCarree())
            plt.colorbar(mp, ax=ax, extend='both', shrink=0.5, orientation='horizontal') 
            plt.show() 
    return country_mask, country_lookup 