from utils.func_shared import construct_file_name
from utils.func_shared import get_years_from_scenario
from utils.func_shared import apply_mask_and_average
from utils.func_shared import calculate_lai_weighted_average
from utils.func_shared import calculate_lai_weighted_total
from utils.func_shared import get_land_ocean_areacella
from utils.func_shared import get_netcdf_var
from utils.func_shared import shared_process
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import butter, filtfilt
import sys, os, pickle, numpy as np, pandas as pd


def reduce_time_step(self, pd_to_use):

    reduce_time_step = self.reduce_time_step
    reduce_time_step_value = self.reduce_time_step_value
    reduce_time_step_method = self.reduce_time_step_method

    if reduce_time_step:

        if reduce_time_step_method == 'mean':
            
            #### keep 1/n years data
            # pd_to_use['year_group'] = pd_to_use['year'].apply(lambda x: x // reduce_time_step_value)
            # pd_to_use = pd_to_use.groupby(['model', 'region', 'year_group']).mean().reset_index()
            # pd_to_use = pd_to_use.drop(columns=['year_group']) 
            #### moving average
            numeric_cols = [col for col in pd_to_use.columns if col not in ['model', 'region', 'year']]
            def rolling_mean_group(group):
                group = group.sort_values('year').copy()
                for col in numeric_cols:
                    group[col] = group[col].rolling(window=reduce_time_step_value, center=True, min_periods=1).mean()
                return group
            pd_to_use = pd_to_use.groupby(['model', 'region'], group_keys=False).apply(rolling_mean_group)

        if reduce_time_step_method == 'LOESS':
            numeric_cols = [col for col in pd_to_use.columns if col not in ['model', 'region', 'year']]
            def loess_smooth_group(group):
                group = group.sort_values('year').copy() 
                frac = reduce_time_step_value / len(group)
                for col in numeric_cols:
                    group[col] = lowess(group[col].values, group['year'].values, frac=frac, return_sorted=False)
                return group
            pd_to_use = pd_to_use.groupby(['model', 'region'], group_keys=False).apply(loess_smooth_group)
        
        if reduce_time_step_method == 'Butterworth':
            numeric_cols = [col for col in pd_to_use.columns if col not in ['model', 'region', 'year']]
            b, a = butter(N=3, Wn=1.0 / reduce_time_step_value, btype='low', fs=1.0)
            def butterworth_smooth_group(group):
                group = group.sort_values('year').copy()
                for col in numeric_cols:
                    col_values = group[col].values.astype(float)
                    nan_mask = np.isnan(col_values)
                    col_values[~nan_mask] = filtfilt(b, a, col_values[~nan_mask])
                    group[col] = col_values
                return group
            pd_to_use = pd_to_use.groupby(['model', 'region'], group_keys=False).apply(butterworth_smooth_group)

        if reduce_time_step_method == 'Butterworth_high_pass':
            numeric_cols = [col for col in pd_to_use.columns if col not in ['model', 'region', 'year']]
            b, a = butter(N=3, Wn=1.0 / reduce_time_step_value, btype='high', fs=1.0)
            def butterworth_high_pass_group(group):
                group = group.sort_values('year').copy()
                for col in numeric_cols:
                    col_values = group[col].values.astype(float)
                    nan_mask = np.isnan(col_values)
                    col_values[~nan_mask] = filtfilt(b, a, col_values[~nan_mask])
                    group[col] = col_values
                return group
            pd_to_use = pd_to_use.groupby(['model', 'region'], group_keys=False).apply(butterworth_high_pass_group)

    return pd_to_use 


def get_model_pd(self, model_to_examine, scenario_to_get): 
    
    #### Get basic information
    grid_type = self.grid_type
    spatial_level = self.spatial_level 
    target_variable = self.target_variable
    pr_scale = self.pr_scale
    path_root = self.path_root 
    analysis_data = path_root + 'analysisOutput/'
    model_data = path_root + 'CMIP6_annual/'
    weighting_method = self.weighting_method
    
    #### Check if the data already exists
    file_name_pd = f'PDdf_{grid_type}_{model_to_examine}_{scenario_to_get}_{spatial_level}.pickle' 
    local_data_path = analysis_data + 'output/load_model_data/' 
    if file_name_pd in os.listdir(local_data_path): 
        with open(os.path.join(local_data_path, file_name_pd), 'rb') as f: 
            pd_subset, years = pickle.load(f) 
 
    else:
        #########################################################################################################
        #### Get global variables 
        #########################################################################################################

        ## Get land ocean mask and cell area
        ds_land_ocean_areacella = get_land_ocean_areacella(model_to_examine, model_data, analysis_data) 
        area_cell = ds_land_ocean_areacella['areacella'].values
        land_mask = ds_land_ocean_areacella['land_mask'].values 
        land_mask = np.ma.masked_equal(land_mask, 0.0) 
        ## Get common variables for all models 
        ds_tas = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'tas')) 
        ds_tas['tas'] = ds_tas['tas'] - 273.15 # °C
        ds_pr = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'pr')) 
        ds_pr['pr'] = ds_pr['pr'] * 86400.0 # mm/day
        ds_gpp = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'gpp')) 
        ds_gpp['gpp'] = ds_gpp['gpp'] * 86400.0 * 1000.0 # gC/m2/day
        ds_npp = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'npp')) 
        ds_npp['npp'] = ds_npp['npp'] * 86400.0 * 1000.0 # gC/m2/day
        ds_lai = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'lai'))
        ## Get carbon pools for specific models only
        if model_to_examine in ['ACCESS-ESM1-5', 'CNRM-ESM2-1']:
            ds_cLitter = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'cLitter')) 
            ds_cLitter['cLitter'] = ds_cLitter['cLitter'] * 1000.0 # gC/m2
            ds_cVeg = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'cVeg')) 
            ds_cVeg['cVeg'] = ds_cVeg['cVeg'] * 1000.0 # gC/m2
            ds_cSoil = get_netcdf_var(model_data + grid_type + '/' + construct_file_name(model_to_examine, grid_type, scenario_to_get, 'cSoil')) 
            ds_cSoil['cSoil'] = ds_cSoil['cSoil'] * 1000.0 # gC/m2 
        ## Global average
        tas_global = ds_tas.spatial.average('tas')['tas'].values
        pr_global = ds_pr.spatial.average('pr')['pr'].values
        gpp_global_land = apply_mask_and_average(ds_gpp, 'gpp', land_mask)
        npp_global_land = apply_mask_and_average(ds_npp, 'npp', land_mask) 
        ## Global LAI weighted average
        lai_m2 = ds_lai['lai'].values * area_cell # m2
        total_area_global = float(np.sum(area_cell)) # m2
        total_lai_global = np.sum(lai_m2 * land_mask, axis=(1, 2)) # m2 
        tasLAI_global_land = calculate_lai_weighted_average(ds_tas['tas'].values, lai_m2, land_mask)
        prLAI_global_land = calculate_lai_weighted_average(ds_pr['pr'].values, lai_m2, land_mask)
        gppLAI_global_land = calculate_lai_weighted_total(ds_gpp['gpp'].values, ds_lai['lai'].values, area_cell, land_mask)
        nppLAI_global_land = calculate_lai_weighted_total(ds_npp['npp'].values, ds_lai['lai'].values, area_cell, land_mask)
        ## Carbon pools for specific models
        if model_to_examine in ['ACCESS-ESM1-5', 'CNRM-ESM2-1']:
            cLitter_global_land = apply_mask_and_average(ds_cLitter, 'cLitter', land_mask)
            cVeg_global_land = apply_mask_and_average(ds_cVeg, 'cVeg', land_mask)
            cSoil_global_land = apply_mask_and_average(ds_cSoil, 'cSoil', land_mask)
            cLitterLAI_global_land = calculate_lai_weighted_total(ds_cLitter['cLitter'].values, ds_lai['lai'].values, area_cell, land_mask)
            cVegLAI_global_land = calculate_lai_weighted_total(ds_cVeg['cVeg'].values, ds_lai['lai'].values, area_cell, land_mask)
            cSoilLAI_global_land = calculate_lai_weighted_total(ds_cSoil['cSoil'].values, ds_lai['lai'].values, area_cell, land_mask)
        ## Year information 
        years = get_years_from_scenario(scenario_to_get, ds_tas['time'].shape[0]) 


        #########################################################################################################
        #### Get regional variables 
        #########################################################################################################

        ## Get regional mask and lookup table
        if spatial_level == 'country':       from utils.func_countryMask      import get_country_mask       as get_mask 
        if spatial_level == 'eco2017':       from utils.func_eco2017Mask      import get_eco2017_mask       as get_mask 
        if spatial_level == 'KoppenGeiger':  from utils.func_koppenGeigerMask import get_KoppenGeiger_mask  as get_mask 
        if spatial_level == 'AR6Land':       from utils.func_ar6LandMask      import get_ar6_land_mask      as get_mask 
        reg_mask, reg_lookup = get_mask(ds_land_ocean_areacella, model_to_examine, analysis_data) 
        unique_mask_numbers = np.unique(reg_mask).compressed() 
        unique_mask_numbers = np.sort(unique_mask_numbers[~np.isnan(unique_mask_numbers)])
        print (unique_mask_numbers)
        print ('total unique mask numbers: ', len(unique_mask_numbers)) 

        ## Create global row for dataframe
        if model_to_examine in ['ACCESS-ESM1-5', 'CNRM-ESM2-1']:
            data_dict = {
                'year': years, 'region': 'global', 'model': model_to_examine, 'area': total_area_global, 'lai': total_lai_global,
                'tas': tas_global, 'pr': pr_global, 'gpp': gpp_global_land, 'npp': npp_global_land, 
                'tasLAI': tasLAI_global_land, 'prLAI': prLAI_global_land, 'gppLAI': gppLAI_global_land, 'nppLAI': nppLAI_global_land,
                'cLitter': cLitter_global_land, 'cVeg': cVeg_global_land, 'cSoil': cSoil_global_land,
                'cLitterLAI': cLitterLAI_global_land, 'cVegLAI': cVegLAI_global_land, 'cSoilLAI': cSoilLAI_global_land  
            }
            columns = ['year', 'region', 'model', 'area', 'lai', 
                       'tas', 'pr', 'gpp', 'npp', 'tasLAI', 'prLAI', 'gppLAI', 'nppLAI', 
                       'cLitter', 'cVeg', 'cSoil', 'cLitterLAI', 'cVegLAI', 'cSoilLAI']
        else:
            data_dict = {
                'year': years, 'region': 'global', 'model': model_to_examine, 'area': total_area_global, 'lai': total_lai_global,
                'tas': tas_global, 'pr': pr_global, 'gpp': gpp_global_land, 'npp': npp_global_land, 
                'tasLAI': tasLAI_global_land, 'prLAI': prLAI_global_land, 'gppLAI': gppLAI_global_land, 'nppLAI': nppLAI_global_land
            }
            columns = ['year', 'region', 'model', 'area', 'lai', 
                       'tas', 'pr', 'gpp', 'npp', 'tasLAI', 'prLAI', 'gppLAI', 'nppLAI']
        pd_subset = pd.DataFrame(data_dict, columns=columns)

        ## Loop through regions
        for reg_idx in unique_mask_numbers: 
            ## Get regional name and mask  
            reg_name = reg_lookup.loc[reg_lookup['number'] == reg_idx, 'name'].values[0] 
            reg_number = reg_lookup.loc[reg_lookup['number'] == reg_idx, 'number'].values[0] 
            print (reg_idx, '--- reg_name: ', reg_name, reg_number) 
            mask_i = np.copy(reg_mask) 
            mask_i = np.ma.masked_not_equal(mask_i, reg_number) * 0.0 + 1.0
            combined_mask = mask_i * land_mask 
            ## Get regional average 
            tas_reg = apply_mask_and_average(ds_tas, 'tas', combined_mask) 
            pr_reg = apply_mask_and_average(ds_pr, 'pr', combined_mask) 
            gpp_reg = apply_mask_and_average(ds_gpp, 'gpp', combined_mask) 
            npp_reg = apply_mask_and_average(ds_npp, 'npp', combined_mask) 
            tasLAI_reg = calculate_lai_weighted_average(ds_tas['tas'].values, lai_m2, combined_mask)
            prLAI_reg = calculate_lai_weighted_average(ds_pr['pr'].values, lai_m2, combined_mask)
            gppLAI_reg = calculate_lai_weighted_total(ds_gpp['gpp'].values, ds_lai['lai'].values, area_cell, combined_mask)
            nppLAI_reg = calculate_lai_weighted_total(ds_npp['npp'].values, ds_lai['lai'].values, area_cell, combined_mask)                
            ## Get regional area and LAI 
            reg_area = float(np.sum(area_cell * combined_mask))
            reg_lai = np.sum(lai_m2 * combined_mask, axis=(1, 2))
            ## Add carbon pools for specific models
            if model_to_examine in ['ACCESS-ESM1-5', 'CNRM-ESM2-1']:
                cLitter_reg = apply_mask_and_average(ds_cLitter, 'cLitter', combined_mask)
                cVeg_reg = apply_mask_and_average(ds_cVeg, 'cVeg', combined_mask)
                cSoil_reg = apply_mask_and_average(ds_cSoil, 'cSoil', combined_mask)
                cLitterLAI_reg = calculate_lai_weighted_total(ds_cLitter['cLitter'].values, ds_lai['lai'].values, area_cell, combined_mask)
                cVegLAI_reg = calculate_lai_weighted_total(ds_cVeg['cVeg'].values, ds_lai['lai'].values, area_cell, combined_mask)
                cSoilLAI_reg = calculate_lai_weighted_total(ds_cSoil['cSoil'].values, ds_lai['lai'].values, area_cell, combined_mask)
                data_dict = {
                    'year': years, 'region': reg_name, 'model': model_to_examine, 'area': reg_area, 'lai': reg_lai,
                    'tas': tas_reg, 'pr': pr_reg, 'gpp': gpp_reg, 'npp': npp_reg, 
                    'tasLAI': tasLAI_reg, 'prLAI': prLAI_reg, 'gppLAI': gppLAI_reg, 'nppLAI': nppLAI_reg,
                    'cLitter': cLitter_reg, 'cVeg': cVeg_reg, 'cSoil': cSoil_reg, 
                    'cLitterLAI': cLitterLAI_reg, 'cVegLAI': cVegLAI_reg, 'cSoilLAI': cSoilLAI_reg
                }
                columns = ['year', 'region', 'model', 'area', 'lai', 
                           'tas', 'pr', 'gpp', 'npp', 'tasLAI', 'prLAI', 'gppLAI', 'nppLAI', 
                           'cLitter', 'cVeg', 'cSoil', 'cLitterLAI', 'cVegLAI', 'cSoilLAI']
            else:
                data_dict = {
                    'year': years, 'region': reg_name, 'model': model_to_examine, 'area': reg_area, 'lai': reg_lai,
                    'tas': tas_reg, 'pr': pr_reg, 'gpp': gpp_reg, 'npp': npp_reg, 
                    'tasLAI': tasLAI_reg, 'prLAI': prLAI_reg, 'gppLAI': gppLAI_reg, 'nppLAI': nppLAI_reg
                }
                columns = ['year', 'region', 'model', 'area', 'lai', 
                           'tas', 'pr', 'gpp', 'npp', 'tasLAI', 'prLAI', 'gppLAI', 'nppLAI']
            pd_tmp = pd.DataFrame(data_dict, columns=columns) 

            #### Now let's put some constraints on what regions to keep:
            #### 1. If the region has any nan tas data, remove it
            #### 2. If the region has any nan gpp/npp data, remove it
            #### 3. If the region has any minimum gpp/npp value less than 0.01, remove it
            if pd_tmp['tas'].isna().any():
                print (f'Warning: pd_tmp has nan tas data for region {reg_name}') 
                continue
            if pd_tmp['gpp'].isna().any() or pd_tmp['npp'].isna().any():
                print (f'Warning: pd_tmp has nan gpp/npp data for region {reg_name}') 
                continue
            if pd_tmp['gpp'].min() < 0.001 or pd_tmp['npp'].min() < 0.001:
                print (f'Warning: pd_tmp has minimum gpp/npp value less than 0.001 for region {reg_name}') 
                continue
            pd_subset = pd.concat([pd_subset, pd_tmp], ignore_index=True)

        #### Transfer precipitation units 
        if pr_scale == 'ln': 
            pd_subset['pr'] = np.log(pd_subset['pr']) 
            pd_subset['prLAI'] = np.log(pd_subset['prLAI']) 
        else:
            print ('no change on precipitation, using absolute value') 

        #### Save raw dataframe as pickle file 
        with open(os.path.join(local_data_path, file_name_pd), 'wb') as f:
            pickle.dump([pd_subset, years], f) 

    #### Now reduce the column to use only the target carbon variable and climate variables
    if weighting_method == 'lai':
        target_variable_to_use = target_variable + 'LAI'
    else:
        target_variable_to_use = target_variable
    pd_to_use = pd_subset[['model', 'region', 'year', 'area', 'lai', 'tas', 'pr', target_variable_to_use]].copy()
    pd_to_use.rename(columns={target_variable_to_use: target_variable}, inplace=True)
    pd_to_use = pd_to_use.sort_values(by=['model', 'region', 'year'])
    pd_to_use = shared_process(pd_to_use, target_variable) 

    #### Reduce time step if needed 
    pd_RTS = reduce_time_step(self, pd_to_use)

    return pd_RTS 


def bootstrap_regression_data(pd_in_use, bootstrap_idx): 
    """
    Now bootstrap the regression data if needed 
    """
    if bootstrap_idx > 0: 
        ## Save and remove global region
        df_global = pd_in_use[pd_in_use['region'] == 'global'].copy()
        df_global["region_boot"] = 'global'
        pd_in_use = pd_in_use[pd_in_use['region'] != 'global']
        ## Get unique regions
        grouped = pd_in_use.groupby('region') 
        unique_regions = pd_in_use['region'].unique() 
        ## Use random seed to bootstrap the data 
        seed = bootstrap_idx + 42 
        rng = np.random.default_rng(seed) 
        resampled_regions = rng.choice(unique_regions, size=len(unique_regions), replace=True)
        resampled_regions = np.sort(resampled_regions) 
        ## Label the resampled regions for easy identification 
        s = pd.Series(resampled_regions) 
        counts = s.groupby(s).cumcount() 
        resampled_regions_labeled = s + (counts + 1).astype(str)
        resampled_regions_labeled[counts == 0] = s[counts == 0]
        resampled_regions_labeled = resampled_regions_labeled.values.tolist() 
        ## Create a new dataframe with the resampled regions 
        dfs = [] 
        for base_region, boot_region in zip(resampled_regions, resampled_regions_labeled):
            df_region = grouped.get_group(base_region).copy()
            df_region["region_boot"] = boot_region
            dfs.append(df_region) 
        ## Add global region back
        dfs.append(df_global)
        ## Concatenate the new dataframe
        df_in_use_new = pd.concat(dfs, ignore_index=True)
    else:
        df_in_use_new = pd_in_use 

    unique_regions = df_in_use_new['region'].unique() 
    print ("unique_fraction =", len(unique_regions) / 138 * 100, "%") 

    return df_in_use_new 
    


def load_model_data(self, model_to_examine, bootstrap_idx): 

    #### ----------------------------------------------------------------------------------------
    #### ------------------------------------------------------- Deal with regression data 
    #### ----------------------------------------------------------------------------------------
    
    #### Load regression data and remove global
    pd_regression = get_model_pd(self, model_to_examine, self.scenario_regression)
    pd_regression = bootstrap_regression_data(pd_regression, bootstrap_idx) 

    #### Filter data based on countries
    if self.gpp_filter: 
        if bootstrap_idx > 0:
            raise ValueError("GPP filter is not supported for bootstrap data")
        pd_regression_mean = pd_regression.groupby(['region'])['gpp'].mean() * self.gpp_threshold_sign
        pd_regression_mean = pd_regression_mean[pd_regression_mean < self.gpp_threshold * self.gpp_threshold_sign]
        pd_regression = pd_regression[~pd_regression['region'].isin(pd_regression_mean.index)] 
    if self.tas_filter:
        if bootstrap_idx > 0: 
            raise ValueError("tas filter is not supported for bootstrap data")
        pd_regression_mean = pd_regression.groupby(['region'])['tas'].mean() * self.tas_threshold_sign
        pd_regression_mean = pd_regression_mean[pd_regression_mean < self.tas_threshold * self.tas_threshold_sign]
        pd_regression = pd_regression[~pd_regression['region'].isin(pd_regression_mean.index)] 

    #### Calculate knots for cubic spline regression 
    knots_tas = np.quantile(pd_regression['tas'][~pd_regression['tas'].isna()], [0.2, 0.4, 0.6, 0.8]).tolist()
    knots_pr = np.quantile(pd_regression['pr'][~pd_regression['pr'].isna()], [0.2, 0.4, 0.6, 0.8]).tolist() 

    #### Calculate climate data defined as averagers between 1960 and 2000 for each region 
    pd_regression_sub = pd_regression[['region', 'year', 'tas', 'pr']].copy()
    pd_climate = pd_regression_sub[(pd_regression_sub['year'] >= 1960) & (pd_regression_sub['year'] <= 2000)].groupby(['region']).mean().reset_index()
    pd_climate = pd_climate[['region', 'tas', 'pr']].rename(columns={'tas': 'tas_climate', 'pr': 'pr_climate'})
    pd_regression = pd_regression.merge(pd_climate, on='region', how='left') 


    #### ----------------------------------------------------------------------------------------------------------  
    #### ------------------------------------------------------- Deal with reference and projection data 
    #### ----------------------------------------------------------------------------------------------------------  

    #### Load reference and projection data and the attachment data 
    pd_projection_future = get_model_pd(self, model_to_examine, self.scenario_projection) 
    pd_references_future = get_model_pd(self, model_to_examine, self.scenario_references) 
    pd_projection_attach = get_model_pd(self, model_to_examine, self.attach_projection) 
    pd_references_attach = get_model_pd(self, model_to_examine, self.attach_references) 

    #### Find shared regions among all cases 
    shared_regions_all = (pd.Index(pd_projection_future['region'].dropna().astype(str).str.strip().unique())
            .intersection(pd.Index(pd_references_future['region'].dropna().astype(str).str.strip().unique()))
            .intersection(pd.Index(pd_projection_attach['region'].dropna().astype(str).str.strip().unique()))
            .intersection(pd.Index(pd_references_attach['region'].dropna().astype(str).str.strip().unique()))
            )   
    shared_regions_all = shared_regions_all.tolist()
    pd_projection_future = pd_projection_future[pd_projection_future['region'].isin(shared_regions_all)]
    pd_references_future = pd_references_future[pd_references_future['region'].isin(shared_regions_all)]
    pd_projection_attach = pd_projection_attach[pd_projection_attach['region'].isin(shared_regions_all)]
    pd_references_attach = pd_references_attach[pd_references_attach['region'].isin(shared_regions_all)]

    #### Now attach the attachment data to the projection data 
    if self.attach_method == 'full':
        pd_projection = pd.concat([pd_projection_future, pd_projection_attach], ignore_index=True).sort_values(["model", "region", "year"])
        pd_references = pd.concat([pd_references_future, pd_references_attach], ignore_index=True).sort_values(["model", "region", "year"])
    elif self.attach_method == 'no':
        pd_projection = pd_projection_future.sort_values(["model", "region", "year"])
        pd_references = pd_references_future.sort_values(["model", "region", "year"])
    else:
        print ('Error, specify how to deal with nan values in projection data')
        sys.exit() 

    #### Merge reference climate data to the projection and references data
    pd_projection = pd_projection.merge(pd_climate, on='region', how='left')
    pd_references = pd_references.merge(pd_climate, on='region', how='left') 

    #### Create and return a dictionary with all necessary data
    data_dict = {'pd_regression': pd_regression, 
                 'pd_projection': pd_projection, 
                 'pd_references': pd_references, 
                 'knots_tas': knots_tas, 
                 'knots_pr': knots_pr} 

    return data_dict 