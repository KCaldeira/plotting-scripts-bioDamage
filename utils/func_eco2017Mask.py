import numpy as np, pandas as pd
import pickle, os, regionmask
import geopandas as gp
from shapely.ops import unary_union

def get_eco2017_mask(ds_land_ocean_areacella, model_to_examine, analysis_data, check_plot=False): 

    ecoregion2017_mask_filename = f'ecoregion2017_mask_{model_to_examine}.pickle'

    if ecoregion2017_mask_filename in os.listdir(analysis_data + 'sub_region_masks'):
        with open(os.path.join(analysis_data, 'sub_region_masks', ecoregion2017_mask_filename), 'rb') as f:
            ecoregion2017_mask, ecoregion2017_lookup = pickle.load(f)
    else:
        lat_1d = ds_land_ocean_areacella["lat"].values  # shape (nlat,)
        lon_1d = ds_land_ocean_areacella["lon"].values  # shape (nlon,)
        fname = f'{analysis_data}sub_region_maps/Ecoregions2017.zip'
        ecoregions = gp.read_file(fname)
        biome_groups = ecoregions.groupby('BIOME_NUM')
        biome_numbers = [] 
        biome_names = [] 
        biome_shapes = [] 
        for biome_num, group in biome_groups:
            biome_numbers.append(int(biome_num))
            biome_names.append(group['BIOME_NAME'].iloc[0])  # Get one name
            merged_geom = unary_union(group.geometry)
            biome_shapes.append(merged_geom)
        regions = regionmask.Regions(
                        outlines=biome_shapes,
                        numbers=biome_numbers,
                        names=biome_names,
                        abbrevs=[str(n) for n in biome_numbers]  # Optional
                    )
        mask_object = regions.mask(lon_1d, lat_1d) 
        ecoregion2017_mask = mask_object.values  # Convert to a regular numpy array
        whereisnan = np.isnan(ecoregion2017_mask)
        ecoregion2017_mask = np.ma.masked_where(whereisnan, ecoregion2017_mask)

        ecoregion2017_names = regions.names 
        ecoregion2017_numbers = regions.numbers
        ecoregion2017_abbrevs = regions.abbrevs
        
        ecoregion2017_lookup = pd.DataFrame({
            "name": ecoregion2017_names, 
            "number": ecoregion2017_numbers, 
            "abbrev": ecoregion2017_abbrevs})

        with open(os.path.join(analysis_data, 'sub_region_masks', ecoregion2017_mask_filename), 'wb') as f:
            pickle.dump([ecoregion2017_mask, ecoregion2017_lookup], f)
        
    return ecoregion2017_mask, ecoregion2017_lookup 