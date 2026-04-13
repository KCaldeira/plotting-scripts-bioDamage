from numba.cuda.nvvmutils import declare_atomic_add_float64
import numpy as np 
from scipy import linalg
from scipy.optimize import minimize_scalar

def fit_ols(y, X):
    # Solve using normal equations (much faster than lstsq for large matrices)
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    # Compute residuals
    y_pred = X @ beta
    residuals = y - y_pred
    # Degrees of freedom
    n = len(y)
    p = X.shape[1]
    df = n - p
    # Residual variance
    sse = np.sum(residuals ** 2)
    sigma_squared = sse / df
    # Covariance matrix of beta (reuse XtX)
    XtX_inv = linalg.inv(XtX)
    cov_matrix = sigma_squared * XtX_inv
    return beta, residuals, sigma_squared, cov_matrix
 
def fit_ApproachDJ_persistence_conjoined(data, h4_bounds=np.array([0.0, 1.0])):

    #### Let's focus on T only for now 

    # Remove global region
    data = data[data['region'] != 'global'].copy()
    data = data[data['year'] >= 1960]
    data = data[data['year'] <= 2014]

    # Num of observations (full data, including NaN-y rows)
    n_obs = data.shape[0]

    # Region information 
    unique_region = data['region'].unique()
    iso_to_idx = {iso: i for i, iso in enumerate(unique_region)}
    n_region = len(unique_region) 
    region_idx = data['region'].map(iso_to_idx).values.astype(np.int32)

    # Time information 
    year = data['year'].values.astype(np.int32)
    actual_year_min = int(min(year))
    actual_year_max = int(max(year))
    year_mid = (actual_year_min + actual_year_max) / 2
    time = (year - year_mid).astype(np.float64)

    # y (may contain NaN for the first observation of each country)
    y_full = data['pct_growth_gpp'].values.astype(np.float64)
    valid_mask = ~np.isnan(y_full)
    y = y_full[valid_mask]

    # active_years = only years that have at least one valid (non-NaN) y observation
    # this excludes the very first year (e.g. 1850) which has NaN for all countries
    active_years = np.sort(np.unique(year[valid_mask]))
    active_year_to_idx = {yr: i for i, yr in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters: 
    n_climate_params = 2
    n_j_params = 3 * (n_region - 1)
    n_k_params = n_active_years
    n_total_params = n_climate_params + n_j_params + n_k_params    ## Flag here to add P

    # Pre-compute constant parts of design matrix (country trends and year effects)
    # Built over FULL data; rows with NaN y will be sliced out before OLS
    X_base = np.zeros((n_obs, n_total_params))

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = region_idx[i] 
        if c > 0:
            t = time[i]
            col_base = n_climate_params + 3 * (c - 1)    ## Flag here to add P
            X_base[i, col_base] = 1.0        
            X_base[i, col_base + 1] = t      
            X_base[i, col_base + 2] = t * t  

    # Year fixed effects (only active years, i.e. years with valid y)
    k_col_start = n_climate_params + n_j_params    ## Flag here to add P
    for i in range(n_obs):
        yr = year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X_base[i, k_col_start + yr_idx] = 1.0

    T = data['tas'].values.astype(np.float64)

    # Compute T_linear and P_linear at first year for each country (for pre-history correction)
    # This provides a smoothed baseline instead of using noisy actual T(first_year)
    def compute_linear_at_first_year(data, var_name):
        X_linear_first = np.zeros(n_obs)
        var = data[var_name].values.astype(np.float64)   # BUG FIX 4: float64
        for c in range(n_region):
            #### Get data 
            region_mask = region_idx == c
            region_indices = np.where(region_mask)[0]
            t_region = time[region_indices]
            X_region = var[region_indices]
            #### Perform regression 
            n_c = len(t_region)
            X_lin = np.column_stack([np.ones(n_c), t_region])
            coeffs, _, _, _ = linalg.lstsq(X_lin, X_region)
            a, b = coeffs
            #### Get projections 
            years_for_region = year[region_indices]
            first_year_idx = np.argmin(years_for_region)
            t_first = t_region[first_year_idx]
            X_at_first = a + b * t_first
            X_linear_first[region_mask] = X_at_first
        return X_linear_first
    T_linear_first = compute_linear_at_first_year(data, 'tas') 

    # 1D optimization: grid search then Brent's method
    def compute_persistence_accumulators(data, h4):
        temp = data['tas'].values.astype(np.float64)   # BUG FIX 4: float64
        decay = 1 - h4
        A_T_lag = np.zeros(n_obs)
        A_T2_lag = np.zeros(n_obs)
        for c in range(n_region):
            region_mask = region_idx == c
            region_indices = np.where(region_mask)[0]
            years_for_region = year[region_indices]
            sorted_order = np.argsort(years_for_region)
            sorted_indices = region_indices[sorted_order]
            # Compute accumulators
            A_T = 0.0
            A_T2 = 0.0
            for i, idx in enumerate(sorted_indices):
                T_val = temp[idx]
                T2_val = T_val ** 2
                if i == 0:
                    A_T_lag[idx] = 0.0
                    A_T2_lag[idx] = 0.0
                    A_T = T_val
                    A_T2 = T2_val
                else:
                    A_T_lag[idx] = A_T
                    A_T2_lag[idx] = A_T2
                    A_T = T_val + decay * A_T
                    A_T2 = T2_val + decay * A_T2
        return A_T_lag, A_T2_lag

    def compute_pre_first_year_correction(data, h4, T_values):
        if T_values is None: T_values = data['tas'].values.astype(np.float64) 
        decay = 1 - h4
        correction_T = np.zeros(n_obs)
        correction_T2 = np.zeros(n_obs) 
        for c in range(n_region):
            region_mask = region_idx == c
            region_indices = np.where(region_mask)[0]
            years_for_region = year[region_indices]
            sorted_order = np.argsort(years_for_region)
            sorted_indices = region_indices[sorted_order]
            sorted_years = years_for_region[sorted_order]
            first_year = sorted_years[0]
            first_idx = sorted_indices[0]
            T_first = T_values[first_idx]
            T2_first = T_first ** 2
            # Compute correction for each year
            for i, idx in enumerate(sorted_indices):
                years_since_first = sorted_years[i] - first_year
                decay_factor = decay ** years_since_first
                correction_T[idx] = decay_factor * T_first
                correction_T2[idx] = decay_factor * T2_first
        return correction_T, correction_T2

    def compute_sse_for_h4(h4_val):
        A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
        correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T_linear_first)
        # Modified temperature regressors (computed over full data for correct accumulator history)
        X1 = T - h4_val * A_T_lag - correction_T
        X2 = T**2 - h4_val * A_T2_lag - correction_T2
        # Build design matrix over full data, then filter to valid (non-NaN y) rows
        X = X_base.copy()
        X[:, 0] = X1
        X[:, 1] = X2
        X_valid = X[valid_mask]
        beta_ols, _, _, _ = np.linalg.lstsq(X_valid, y, rcond=None)
        y_pred = X_valid @ beta_ols
        sse = np.sum((y - y_pred) ** 2)
        return sse

    h4_grid = np.linspace(h4_bounds[0], h4_bounds[1], 21)
    sse_grid = [compute_sse_for_h4(h4_val) for h4_val in h4_grid]
    best_grid_idx = np.argmin(sse_grid)

    search_lo = h4_grid[max(0, best_grid_idx - 1)]
    search_hi = h4_grid[min(len(h4_grid) - 1, best_grid_idx + 1)]

    result = minimize_scalar(
        compute_sse_for_h4,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    h4_opt = result.x 

    # Re-fit at optimal h4 to get coefficients and statistics
    # Use T_linear_first for pre-history correction (consistent with optimization)
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_opt)
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_opt, T_linear_first)

    X1 = T - h4_opt * A_T_lag - correction_T
    X2 = T**2 - h4_opt * A_T2_lag - correction_T2

    X_opt = X_base.copy()
    X_opt[:, 0] = X1
    X_opt[:, 1] = X2
    # Filter to valid (non-NaN y) rows for OLS
    X_opt_valid = X_opt[valid_mask]

    beta, residuals, sigma_sq, cov = fit_ols(y, X_opt_valid)

    h1 = beta[0]
    h2 = beta[1] 

    return h4_opt, h1, h2 

    