from utils.func_shared import get_land_ocean_areacella
import os, pickle, numpy as np, pandas as pd 

def construct_projection_data(pd_in, target_variable, regression_type, knots_tas, knots_pr): 
    if regression_type.startswith('burke_'):   from mainAnalysis.constructBurkeXY   import construct_XY
    if regression_type.startswith('newell_'):  from mainAnalysis.constructNewellXY  import construct_XY
    if regression_type.startswith('harding_'): from mainAnalysis.constructHardingXY import construct_XY
    if regression_type.startswith('kalkuhl_'): from mainAnalysis.constructKalkuhlXY import construct_XY
    if regression_type.startswith('newellExtended_'): from mainAnalysis.constructNewellExtendedXY import construct_XY
    pd_inuse = construct_XY(pd_in, target_variable, regression_type, 'projection', knots_tas, knots_pr) 
    return pd_inuse 

def fitting_func(fitting_dict, pd_in, type='full'):
    X_reg = fitting_dict['X']
    cols = X_reg.columns
    X_proj = pd_in.reindex(columns=cols, fill_value=0)
    model = fitting_dict['model']
    prj_value = model.predict(X_proj)
    prj_value_numpy = np.array(prj_value)
    return prj_value_numpy 

def do_projection2(self, data_dict, model_to_examine, regression_type, bootstrap_idx): 
    target_variable = self.target_variable
    spatial_level = self.spatial_level
    scenario_FUL_part2 = self.scenario_FUL_part2
    projection_start_year = self.projection_start_year
    projection_end_year = self.projection_end_year
    force_redo_projection = self.force_redo_projection
    impact_contribution = self.impact_contribution 
    model_data = self.path_root + 'CMIP6_annual/'
    analysis_data = self.path_root + 'analysisOutput/'
    weighting_method = self.weighting_method 

    if regression_type.startswith('burke_'):   local_data_path = analysis_data + 'output_proj/burke/' 
    if regression_type.startswith('newell_'):  local_data_path = analysis_data + 'output_proj/newell/' 
    if regression_type.startswith('harding_'): local_data_path = analysis_data + 'output_proj/harding/' 
    if regression_type.startswith('kalkuhl_'): local_data_path = analysis_data + 'output_proj/kalkuhl/' 
    if regression_type.startswith('newellExtended_'): local_data_path = analysis_data + 'output_proj/newellExtended/' 


    #### Get data 
    pd_projection_references = construct_projection_data(data_dict['pd_references'], target_variable, regression_type, data_dict['knots_tas'], data_dict['knots_pr'])
    pd_projection_projection = construct_projection_data(data_dict['pd_projection'], target_variable, regression_type, data_dict['knots_tas'], data_dict['knots_pr'])
    if projection_start_year > 0:
        pd_projection_references = pd_projection_references[pd_projection_references['year'] >= projection_start_year]
        pd_projection_projection = pd_projection_projection[pd_projection_projection['year'] >= projection_start_year] 
    if projection_end_year > 0:
        pd_projection_references = pd_projection_references[pd_projection_references['year'] <= projection_end_year]
        pd_projection_projection = pd_projection_projection[pd_projection_projection['year'] <= projection_end_year]

    #### Get masks
    ds_land_ocean_areacella = get_land_ocean_areacella(model_to_examine, model_data, analysis_data) 
    lat, lon = ds_land_ocean_areacella['lat'].values, ds_land_ocean_areacella['lon'].values
    if spatial_level == 'country':       from utils.func_countryMask      import get_country_mask       as get_mask 
    if spatial_level == 'eco2017':       from utils.func_eco2017Mask      import get_eco2017_mask       as get_mask 
    if spatial_level == 'KoppenGeiger':  from utils.func_koppenGeigerMask import get_KoppenGeiger_mask  as get_mask 
    if spatial_level == 'AR6Land':       from utils.func_ar6LandMask      import get_ar6_land_mask      as get_mask 
    reg_mask, reg_lookup = get_mask(ds_land_ocean_areacella, model_to_examine, analysis_data) 

    #### Find shared regions in reference and projection data to use; 
    unique_region_references = np.unique(pd_projection_references['region']).tolist() 
    unique_region_projection = np.unique(pd_projection_projection['region']).tolist() 
    shared_regions_FUL_BGC = (pd.Index(unique_region_references).intersection(pd.Index(unique_region_projection)))   
    unique_region_list = sorted(shared_regions_FUL_BGC.tolist())
    pd_projection_references = pd_projection_references[pd_projection_references['region'].isin(unique_region_list)]
    pd_projection_projection = pd_projection_projection[pd_projection_projection['region'].isin(unique_region_list)]
    years = np.sort(np.unique(pd_projection_references['year'].values.tolist())) 
    #### Get projection results 
    regional_results_radonl = np.zeros([len(unique_region_list), len(years)]) 
    regional_results_fulimp = np.zeros([len(unique_region_list), len(years)]) 
    regional_results_prjres = np.zeros([len(unique_region_list), len(years)]) 
    weighting_regions_list1 = np.zeros([len(unique_region_list), len(years)]) 
    weighting_regions_list2 = np.zeros([len(unique_region_list), len(years)]) 
    for reg_i in unique_region_list: 
        #### Get climate impact; 
        pd_radonl_reg = pd_projection_references[pd_projection_references['region'] == reg_i]
        pd_fulimp_reg = pd_projection_projection[pd_projection_projection['region'] == reg_i]
        #### Get projection results; 
        Y_name = f'{target_variable}' 
        reg_radonl = np.array(pd_radonl_reg[Y_name].values.tolist()) 
        reg_fulimp = np.array(pd_fulimp_reg[Y_name].values.tolist()) 
        reg_prjres = np.zeros(len(years))
        # reg_prjres[0] = reg_radonl[0]
        reg_prjres[0] = reg_fulimp[0] 
        # base_growth = reg_radonl[1:] / reg_radonl[:-1] - 1 # 2016 to 2100, in which 2016 represents changes from 2015 to 2016
        base_growth = reg_fulimp[1:] / reg_fulimp[:-1] - 1 # 2016 to 2100, in which 2016 represents changes from 2015 to 2016
        for i in range(1, len(years)): 
            if i == 1:
                actual_growth = 1 + base_growth[i-1] + 0.01
            else:
                actual_growth = 1 + base_growth[i-1]
            reg_prjres[i] = reg_prjres[i-1] * actual_growth 
        
        #### Save results 
        regional_results_radonl[unique_region_list.index(reg_i)] = reg_radonl
        regional_results_fulimp[unique_region_list.index(reg_i)] = reg_fulimp 
        regional_results_prjres[unique_region_list.index(reg_i)] = reg_prjres 
        #### Save weights
        if weighting_method == 'area':
            area_reg = np.array(pd_projection_projection[pd_projection_projection['region'] == reg_i]['area'].values.tolist())
            weighting_regions_list1[unique_region_list.index(reg_i)] = area_reg  
            weighting_regions_list2[unique_region_list.index(reg_i)] = area_reg  
        if weighting_method == 'lai':
            lai_reference = np.array(pd_projection_references[pd_projection_references['region'] == reg_i]['lai'].values.tolist())
            weighting_regions_list1[unique_region_list.index(reg_i)] = lai_reference  
            lai_projection = np.array(pd_projection_projection[pd_projection_projection['region'] == reg_i]['lai'].values.tolist())
            weighting_regions_list2[unique_region_list.index(reg_i)] = lai_projection  

        #### Calculate global average
        results_dict = {
            'region_list': unique_region_list,
            'years': years,
            'regional_results_radonl': regional_results_radonl,
            'regional_results_fulimp': regional_results_fulimp,
            'regional_results_prjres': regional_results_prjres,
            'weighting_radonl': weighting_regions_list1, 
            'weighting_fulimp': weighting_regions_list2, 
            'reg_mask': reg_mask,
            'reg_lookup': reg_lookup,
            'lat': lat,
            'lon': lon} 

    return results_dict  