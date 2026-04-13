from utils.func_shared import get_land_ocean_areacella
import os, pickle, numpy as np, pandas as pd 

def construct_projection_data(pd_in, target_variable, regression_type, knots_tas, knots_pr): 
    if regression_type.startswith('burke_'):   from mainAnalysis.constructBurkeXY   import construct_XY
    if regression_type.startswith('newell_'):  from mainAnalysis.constructNewellXY  import construct_XY
    if regression_type.startswith('harding_'): from mainAnalysis.constructHardingXY import construct_XY
    if regression_type.startswith('kalkuhl_'): from mainAnalysis.constructKalkuhlXY import construct_XY
    if regression_type.startswith('newellExtended_'): from mainAnalysis.constructNewellExtendedXY import construct_XY
    if regression_type.startswith('ken_'):     from mainAnalysis.constructKenXY     import construct_XY
    pd_inuse = construct_XY(pd_in, target_variable, regression_type, 'projection', knots_tas, knots_pr) 
    return pd_inuse 

def fitting_func(fitting_dict, pd_in, type='full'):
    X_reg = fitting_dict['X']
    cols = X_reg.columns
    X_proj = pd_in.reindex(columns=cols, fill_value=0)
    model = fitting_dict['model']
    prj_value = model.predict(X_proj)
    prj_value_numpy = np.exp(np.array(prj_value))
    return prj_value_numpy 

def do_projection(self, fitting_regression, data_dict, model_to_examine, regression_type, bootstrap_idx): 
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

    file_name_projection_results = f'{model_to_examine}_{target_variable}_{weighting_method}_{spatial_level}_{scenario_FUL_part2}_{regression_type}_{projection_start_year}_{projection_end_year}_{bootstrap_idx}.pickle' 
    if regression_type.startswith('burke_'):   local_data_path = analysis_data + 'output_proj/burke/' 
    if regression_type.startswith('newell_'):  local_data_path = analysis_data + 'output_proj/newell/' 
    if regression_type.startswith('harding_'): local_data_path = analysis_data + 'output_proj/harding/' 
    if regression_type.startswith('kalkuhl_'): local_data_path = analysis_data + 'output_proj/kalkuhl/' 
    if regression_type.startswith('newellExtended_'): local_data_path = analysis_data + 'output_proj/newellExtended/' 
    if regression_type.startswith('ken_'): local_data_path = analysis_data + 'output_proj/ken/' 
    os.makedirs(local_data_path, exist_ok=True)

    if file_name_projection_results in os.listdir(local_data_path) and force_redo_projection == False:
        with open(os.path.join(local_data_path, file_name_projection_results), 'rb') as f: 
            results_dict = pickle.load(f) 
    else:
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
        if spatial_level == 'country':       from utils.func_countryMask   import get_country_mask       as get_mask 
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
        regional_results_prjBur = np.zeros([len(unique_region_list), len(years)]) 
        regional_results_prjKen = np.zeros([len(unique_region_list), len(years)]) 
        weighting_regions_list1 = np.zeros([len(unique_region_list), len(years)]) 
        weighting_regions_list2 = np.zeros([len(unique_region_list), len(years)]) 
        for reg_i in unique_region_list: 

            #### Before diving into different equations, we set the shared model results here 
            pd_radonl_reg = pd_projection_references[pd_projection_references['region'] == reg_i]
            pd_fulimp_reg = pd_projection_projection[pd_projection_projection['region'] == reg_i]
            reg_radonl = np.array(pd_radonl_reg[target_variable].values.tolist()) 
            reg_fulimp = np.array(pd_fulimp_reg[target_variable].values.tolist()) 
            regional_results_radonl[unique_region_list.index(reg_i)] = reg_radonl
            regional_results_fulimp[unique_region_list.index(reg_i)] = reg_fulimp 

            #########################################################
            #### Apply Burke's assumption here
            #########################################################
            # if regression_type in ["ken_level_eq14", "ken_growth_eq22", "ken_growth_eq23"]: 
            if regression_type.startswith('ken_'): 
                #### Have converted log Y to Y in fitting_func
                b = fitting_func(fitting_regression, pd_radonl_reg, impact_contribution) 
                d = fitting_func(fitting_regression, pd_fulimp_reg, impact_contribution)
                #### The first way of projection 
                reg_prjBur = np.zeros(len(years)) 
                for i in range(0, len(years)): 
                    Y_differ = d[i] - b[i]
                    reg_prjBur[i] = reg_radonl[i] + Y_differ 
                #### The second way of projection 
                reg_prjKen = np.zeros(len(years))
                for i in range(0, len(years)): 
                    Y_differ = d[i]
                    reg_prjKen[i] = reg_radonl[i] + Y_differ 
                #### Save results 
                regional_results_prjBur[unique_region_list.index(reg_i)] = reg_prjBur
                regional_results_prjKen[unique_region_list.index(reg_i)] = reg_prjKen 

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

        #### Get outputs 
        results_dict = {
            'region_list': unique_region_list,
            'years': years,
            'regional_results_radonl': regional_results_radonl,
            'regional_results_fulimp': regional_results_fulimp,
            'regional_results_prjBur': regional_results_prjBur,
            'regional_results_prjKen': regional_results_prjKen,
            'weighting_radonl': weighting_regions_list1, 
            'weighting_fulimp': weighting_regions_list2, 
            'reg_mask': reg_mask,
            'reg_lookup': reg_lookup,
            'lat': lat,
            'lon': lon} 

        with open(os.path.join(local_data_path, file_name_projection_results), 'wb') as f: 
            pickle.dump(results_dict, f) 

    return results_dict  