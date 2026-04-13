from utils.func_shared import get_land_ocean_areacella
import os, pickle, numpy as np, pandas as pd 

def fitting_func(fitting_dict, pd_in, impact_contribution): 

    X_reg = fitting_dict['X']
    cols = X_reg.columns
    model = fitting_dict['model']
    X_proj = pd_in.reindex(columns=cols, fill_value=0)

    if impact_contribution == 'full':
        prj_full_impact = np.array(model.predict(X_proj))
        return prj_full_impact
    else:
        tas_cols = [c for c in cols if c.startswith('tas') or c.startswith('T_')]
        pr_cols = [c for c in cols if c.startswith('pr') or c.startswith('P_')]
        if impact_contribution == 'tas':
            X_proj_tas = X_proj.copy()
            for col in pr_cols:
                X_proj_tas[col] = 0 
            prj_tas_impact = np.array(model.predict(X_proj_tas))
            return prj_tas_impact
        elif impact_contribution == 'pr':
            X_proj_pr = X_proj.copy()
            for col in tas_cols:
                X_proj_pr[col] = 0 
            prj_pr_impact = np.array(model.predict(X_proj_pr))
            return prj_pr_impact
        elif impact_contribution == '30T':
            X_proj_30T = X_proj.copy()
            if 'tas' in pd_in.columns:
                tas_original = pd_in['tas']
                tas_capped = np.minimum(tas_original, 30) 
                for col in tas_cols:
                    if col == 'tas': 
                        X_proj_30T[col] = tas_capped
                    elif col == 'tas2':
                        X_proj_30T[col] = tas_capped**2
                    elif 'poly1' in col:
                        X_proj_30T[col] = tas_capped
                    elif 'poly2' in col:
                        X_proj_30T[col] = tas_capped**2
                    elif 'poly3' in col:
                        X_proj_30T[col] = tas_capped**3
            prj_30T_impact = np.array(model.predict(X_proj_30T))
            return prj_30T_impact

def do_projection(self, fitting_regression, data_dict, model_to_examine, regression_type, bootstrap_idx): 
    target_variable = self.target_variable
    Y_name = f'{target_variable}' 
    spatial_level = self.spatial_level
    scenario_projection = self.scenario_projection 
    scenario_references = self.scenario_references
    projection_start_year = self.projection_start_year
    projection_end_year = self.projection_end_year
    force_redo_projection = self.force_redo_projection
    model_data = self.path_root + 'CMIP6_annual/'
    analysis_data = self.path_root + 'analysisOutput/'
    weighting_method = self.weighting_method 
    add_constant = self.add_constant
    impact_contribution = self.impact_contribution 

    file_name_projection_results = f'{model_to_examine}_{target_variable}_{weighting_method}_{spatial_level}_{scenario_projection}_{scenario_references}_{regression_type}_{projection_start_year}_{projection_end_year}_{bootstrap_idx}.pickle' 
    if regression_type.startswith('burke_'):   local_data_path = analysis_data + 'output/projection_results/burke/' 
    if regression_type.startswith('newell_'):  local_data_path = analysis_data + 'output/projection_results/newell/' 
    if regression_type.startswith('harding_'): local_data_path = analysis_data + 'output/projection_results/harding/' 
    if regression_type.startswith('kalkuhl_'): local_data_path = analysis_data + 'output/projection_results/kalkuhl/' 
    os.makedirs(local_data_path, exist_ok=True) 

    if file_name_projection_results in os.listdir(local_data_path) and force_redo_projection == False:
        with open(os.path.join(local_data_path, file_name_projection_results), 'rb') as f: 
            results_dict = pickle.load(f) 
    else:

        #########################################################################################################
        #### Get data 
        #########################################################################################################

        #### Get data and remove global 
        pd_projection = data_dict['pd_projection']; pd_projection = pd_projection[pd_projection['region'] != 'global'].copy()
        pd_references = data_dict['pd_references']; pd_references = pd_references[pd_references['region'] != 'global'].copy() 
        knots_tas, knots_pr = data_dict['knots_tas'], data_dict['knots_pr']

        #### Construct projection data 
        if regression_type.startswith('burke_'):   from mainAnalysis.constructBurkeXY   import construct_XY
        if regression_type.startswith('newell_'):  from mainAnalysis.constructNewellXY  import construct_XY
        if regression_type.startswith('harding_'): from mainAnalysis.constructHardingXY import construct_XY
        if regression_type.startswith('kalkuhl_'): from mainAnalysis.constructKalkuhlXY import construct_XY
        X_projection = construct_XY(pd_projection, target_variable, regression_type, 'projection', add_constant, knots_tas, knots_pr) 
        X_references = construct_XY(pd_references, target_variable, regression_type, 'projection', add_constant, knots_tas, knots_pr) 
        if projection_start_year > 0:
            X_projection = X_projection[X_projection['year'] >= projection_start_year]
            X_references = X_references[X_references['year'] >= projection_start_year] 
        if projection_end_year > 0:
            X_projection = X_projection[X_projection['year'] <= projection_end_year]
            X_references = X_references[X_references['year'] <= projection_end_year]
        
        #### Get year information 
        year_list_projection = np.unique(np.array(X_projection['year'].values.tolist())) 
        year_list_references = np.unique(np.array(X_references['year'].values.tolist())) 
        if np.max(year_list_projection) != np.max(year_list_references) and np.min(year_list_projection) != np.min(year_list_references):
            raise ValueError('Year range of projection and references data are not the same')
        # years = np.arange(np.min(year_list_projection), np.max(year_list_projection) + 1) 
        years = np.sort(year_list_projection)
        
        #### Get masks
        ds_land_ocean_areacella = get_land_ocean_areacella(model_to_examine, model_data, analysis_data) 
        lat, lon = ds_land_ocean_areacella['lat'].values, ds_land_ocean_areacella['lon'].values
        if spatial_level == 'country':       from utils.func_countryMask      import get_country_mask       as get_mask 
        if spatial_level == 'eco2017':       from utils.func_eco2017Mask      import get_eco2017_mask       as get_mask 
        if spatial_level == 'KoppenGeiger':  from utils.func_koppenGeigerMask import get_KoppenGeiger_mask  as get_mask 
        if spatial_level == 'AR6Land':       from utils.func_ar6LandMask      import get_ar6_land_mask      as get_mask 
        reg_mask, reg_lookup = get_mask(ds_land_ocean_areacella, model_to_examine, analysis_data) 

        #### Find shared regions in reference and projection data to use; 
        unique_region_projection = np.unique(X_projection['region']).tolist() 
        unique_region_references = np.unique(X_references['region']).tolist() 
        shared_regions_X = (pd.Index(unique_region_projection).intersection(pd.Index(unique_region_references)))   
        unique_region_list = sorted(shared_regions_X.tolist())
        X_projection = X_projection[X_projection['region'].isin(unique_region_list)]
        X_references = X_references[X_references['region'].isin(unique_region_list)]

        #########################################################################################################
        #### Do projection and separate the impact of T and P 
        #########################################################################################################
        model_projection = np.zeros([len(unique_region_list), len(years)]) #### Model simulation projection
        model_references = np.zeros([len(unique_region_list), len(years)]) #### Model simulation references 
        empirical_projection_corrected = np.zeros([len(unique_region_list), len(years)]) #### Empirical projection full impact
        empirical_projection_burke = np.zeros([len(unique_region_list), len(years)])
        weights_projection = np.zeros([len(unique_region_list), len(years)]) #### Weights for fulimp and prjres
        weights_references = np.zeros([len(unique_region_list), len(years)]) #### Weights for radonl

        #### Get projection results 
        for reg_i in unique_region_list: 

            #### Get climate impact 
            pd_fulimp_reg = X_projection[X_projection['region'] == reg_i]
            pd_radonl_reg = X_references[X_references['region'] == reg_i] 
            d = fitting_func(fitting_regression, pd_fulimp_reg, impact_contribution)
            b = fitting_func(fitting_regression, pd_radonl_reg, impact_contribution)

            #### Initilize the results
            reg_model_projection = np.array(pd_fulimp_reg[Y_name].values.tolist()) 
            reg_model_references = np.array(pd_radonl_reg[Y_name].values.tolist()) 
            reg_empirical_projection_corrected = np.zeros(len(years)) 
            reg_empirical_projection_corrected[0] = np.log(reg_model_projection[0]) 
            reg_empirical_projection_burke = np.zeros(len(years)) 
            reg_empirical_projection_burke[0] = reg_model_projection[0] 

            #### Growth rate from reference 
            base_growth_corrected = np.log(reg_model_references[1:] / reg_model_references[:-1])
            base_growth_burke = reg_model_references[1:] / reg_model_references[:-1] 
            
            #### Time step iteration 
            for i in range(1, len(years)): 
                actual_growth_corrected = base_growth_corrected[i-1] + d[i] - b[i]
                actual_growth_burke = base_growth_burke[i-1] + d[i] - b[i]
                reg_empirical_projection_corrected[i] = reg_empirical_projection_corrected[i-1] + actual_growth_corrected 
                reg_empirical_projection_burke[i] = reg_empirical_projection_burke[i-1] * actual_growth_burke 
            reg_empirical_projection_corrected = np.exp(reg_empirical_projection_corrected)

            #### Save results 
            model_projection[unique_region_list.index(reg_i)] = reg_model_projection 
            model_references[unique_region_list.index(reg_i)] = reg_model_references 
            empirical_projection_corrected[unique_region_list.index(reg_i)] = reg_empirical_projection_corrected 
            empirical_projection_burke[unique_region_list.index(reg_i)] = reg_empirical_projection_burke 
            if weighting_method == 'area':
                area_reg = np.array(X_projection[X_projection['region'] == reg_i]['area'].values.tolist())
                weights_projection[unique_region_list.index(reg_i)] = area_reg  
                weights_references[unique_region_list.index(reg_i)] = area_reg 
            if weighting_method == 'lai': 
                lai_projection = np.array(X_projection[X_projection['region'] == reg_i]['lai'].values.tolist()) 
                lai_references = np.array(X_references[X_references['region'] == reg_i]['lai'].values.tolist()) 
                weights_projection[unique_region_list.index(reg_i)] = lai_projection 
                weights_references[unique_region_list.index(reg_i)] = lai_references 

        #### Get outputs 
        results_dict = { 
            'region_list': unique_region_list,
            'years': years,
            'model_simulation_projection': model_projection,
            'model_simulation_references': model_references,
            'empirical_projection_corrected': empirical_projection_corrected,
            'empirical_projection_burke': empirical_projection_burke,
            'weights_projection': weights_projection, 
            'weights_references': weights_references, 
            'reg_mask': reg_mask,
            'reg_lookup': reg_lookup,
            'lat': lat,
            'lon': lon} 

        with open(os.path.join(local_data_path, file_name_projection_results), 'wb') as f: 
            pickle.dump(results_dict, f) 

    return results_dict  