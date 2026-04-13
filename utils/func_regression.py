import os, pickle, numpy as np 
import statsmodels.api as sm 

def do_regression(self, data_dict, model_to_examine, regression_type, bootstrap_idx): 

    target_variable = self.target_variable
    spatial_level = self.spatial_level
    scenario_regression = self.scenario_regression 
    regression_start_year = self.regression_start_year
    regression_end_year = self.regression_end_year
    force_redo_regression = self.force_redo_regression
    analysis_data = self.path_root + 'analysisOutput/' 
    weighting_method = self.weighting_method
    add_constant = self.add_constant 

    file_name_regression_results = f'{model_to_examine}_{target_variable}_{weighting_method}_{spatial_level}_{scenario_regression}_{regression_type}_{regression_start_year}_{regression_end_year}_{bootstrap_idx}.pickle' 
    if regression_type.startswith('burke_'):   local_data_path = analysis_data + 'output/regression_results/burke/' 
    if regression_type.startswith('newell_'):  local_data_path = analysis_data + 'output/regression_results/newell/' 
    if regression_type.startswith('harding_'): local_data_path = analysis_data + 'output/regression_results/harding/' 
    if regression_type.startswith('kalkuhl_'): local_data_path = analysis_data + 'output/regression_results/kalkuhl/' 
    if regression_type.startswith('ken_'): local_data_path = analysis_data + 'output/regression_results/ken/' 
    os.makedirs(local_data_path, exist_ok=True)

    if file_name_regression_results in os.listdir(local_data_path) and force_redo_regression == False:
        with open(os.path.join(local_data_path, file_name_regression_results), 'rb') as f: 
            fitting_results = pickle.load(f) 
    else: 
        pd_in = data_dict['pd_regression'] 
        pd_in = pd_in[pd_in['region'] != 'global'].copy() 
        knots_tas = data_dict['knots_tas'] 
        knots_pr = data_dict['knots_pr'] 

        if regression_start_year > 0: pd_in = pd_in[pd_in['year'] >= regression_start_year] 
        if regression_end_year > 0: pd_in = pd_in[pd_in['year'] <= regression_end_year] 
        if regression_type.startswith('burke_'):   from mainAnalysis.constructBurkeXY   import construct_XY
        if regression_type.startswith('newell_'):  from mainAnalysis.constructNewellXY  import construct_XY
        # if regression_type.startswith('harding_'): from mainAnalysis.constructHardingXY import construct_XY 
        # if regression_type.startswith('kalkuhl_'): from mainAnalysis.constructKalkuhlXY import construct_XY
        # if regression_type.startswith('ken_'):     from mainAnalysis.constructKenXY import construct_XY
        
        X, Y = construct_XY(pd_in, target_variable, regression_type, 'regression', add_constant, knots_tas, knots_pr) 
        if add_constant: X = sm.add_constant(X, has_constant='add') 
        model = sm.OLS(Y, X).fit() 

        #### Get metrics
        y = np.array(Y)
        y_pred = np.array(model.predict(X))
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2) 
        R2 = 1 - ss_res / ss_tot 
        RMSE = np.sqrt(model.mse_resid) 
        AIC = model.aic
        BIC = model.bic
        n = model.nobs
        k_eff = model.model.rank
        AICc = AIC + (2 * k_eff * (k_eff + 1)) / (n - k_eff - 1)
        #### Get output 
        fitting_results = {} 
        fitting_results['X'] = X
        fitting_results['Y'] = Y
        fitting_results['model'] = model 
        fitting_results['R2'] = R2
        fitting_results['RMSE'] = RMSE
        fitting_results['AIC'] = AIC
        fitting_results['BIC'] = BIC
        fitting_results['AICc'] = AICc 
        
        with open(os.path.join(local_data_path, file_name_regression_results), 'wb') as f: 
            pickle.dump(fitting_results, f) 

    return fitting_results 