import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from utils.func_shared import get_land_ocean_areacella


def _cumulative_d_b_pct_surface(data_2d, base_growth):
    """Per-region cumulative d-b (%): log excess vs BGC reference growth, cumsum in log space, (e^X−1)·100."""
    log_excess = np.log(data_2d[:, 1:] / data_2d[:, :-1]) - base_growth
    log_cum = np.cumsum(log_excess, axis=1)
    return (np.exp(log_cum) - 1) * 100


def burke_number_spatial_statistics_subType(self):

    results_dict = self.results_dict
    action_names = [a for a in results_dict.keys() if a.startswith('burke_')]
    model_list = self.model_list
    ts_year = 2100

    for model_i in model_list:

        print()
        print('=' * 80)
        print(f'Model: {model_i}')
        print('=' * 80)

        for action_name in action_names:

            print()
            print(f'--- {action_name} ---')

            projection_central = results_dict[action_name][model_i]['projection_main']

            years = np.array(projection_central['years'])
            plot_years = years[1:]
            idx = int(np.searchsorted(plot_years, ts_year))

            model_simulation_projection = np.array(projection_central['model_simulation_projection'])
            model_simulation_references = np.array(projection_central['model_simulation_references'])
            empirical_projection_corrected = np.array(projection_central['empirical_projection_corrected'])
            weights_projection = projection_central['weights_projection']

            regional_areas = weights_projection[:, 0]

            base_growth = np.log(model_simulation_references[:, 1:] / model_simulation_references[:, :-1])
            emp_cumulative_pct = _cumulative_d_b_pct_surface(empirical_projection_corrected, base_growth)
            mod_cumulative_pct = _cumulative_d_b_pct_surface(model_simulation_projection, base_growth)

            empirical_change = emp_cumulative_pct[:, idx]
            model_change = mod_cumulative_pct[:, idx]

            valid_mask_both = np.isfinite(empirical_change) & np.isfinite(model_change)
            empirical_change_both = empirical_change[valid_mask_both]
            model_change_both = model_change[valid_mask_both]
            areas_both = regional_areas[valid_mask_both]
            
            sign_empirical = np.sign(empirical_change_both)
            sign_model = np.sign(model_change_both)
            same_sign_mask = sign_empirical * sign_model > 0
            diff_sign_mask = sign_empirical * sign_model < 0
            
            fraction_regions_same = np.sum(same_sign_mask) / len(empirical_change_both) * 100
            fraction_regions_diff = np.sum(diff_sign_mask) / len(empirical_change_both) * 100
            
            fraction_area_same = np.sum(areas_both[same_sign_mask]) / np.sum(areas_both) * 100
            fraction_area_diff = np.sum(areas_both[diff_sign_mask]) / np.sum(areas_both) * 100
            
            valid_mask_empirical = np.isfinite(empirical_change)
            empirical_change_valid = empirical_change[valid_mask_empirical]
            areas_empirical = regional_areas[valid_mask_empirical]
            weights_empirical = areas_empirical / np.sum(areas_empirical)

            valid_mask_model = np.isfinite(model_change)
            model_change_valid = model_change[valid_mask_model]
            areas_model = regional_areas[valid_mask_model]
            weights_model = areas_model / np.sum(areas_model)
            
            mean_empirical = np.average(empirical_change_valid, weights=weights_empirical)
            mean_model = np.average(model_change_valid, weights=weights_model)
            
            empirical_anom = empirical_change_valid - mean_empirical
            model_anom = model_change_valid - mean_model
            
            std_empirical = np.sqrt(np.average(empirical_anom**2, weights=weights_empirical))
            std_model = np.sqrt(np.average(model_anom**2, weights=weights_model))
            
            weights_both = areas_both / np.sum(areas_both)
            mean_empirical_both = np.average(empirical_change_both, weights=weights_both)
            mean_model_both = np.average(model_change_both, weights=weights_both)
            empirical_anom_both = empirical_change_both - mean_empirical_both
            model_anom_both = model_change_both - mean_model_both
            std_empirical_both = np.sqrt(np.average(empirical_anom_both**2, weights=weights_both))
            std_model_both = np.sqrt(np.average(model_anom_both**2, weights=weights_both))
            
            if std_empirical_both > 0 and std_model_both > 0:
                cov = np.average(empirical_anom_both * model_anom_both, weights=weights_both)
                corr = cov / (std_empirical_both * std_model_both)
            else:
                corr = np.nan

            print(f'\nMetric: cumulative d-b at {ts_year} (%), projection_main (central).')
            print(f'\n1. Sign Agreement (comparing {len(empirical_change_both)} regions with both valid):')
            print(f'   Same sign:      {np.sum(same_sign_mask):3d} regions ({fraction_regions_same:5.1f}%)')
            print(f'   Different sign: {np.sum(diff_sign_mask):3d} regions ({fraction_regions_diff:5.1f}%)')

            print(f'\n2. Area-weighted Sign Agreement:')
            print(f'   Same sign area:      {fraction_area_same:5.1f}%')
            print(f'   Different sign area: {fraction_area_diff:5.1f}%')

            print(f'\n3. Spatial Standard Deviation (area-weighted, % cumulative d-b):')
            print(f'   Empirical (central): {std_empirical:.4f}  (n={len(empirical_change_valid)} regions)')
            print(f'   Model projection:    {std_model:.4f}  (n={len(model_change_valid)} regions)')
            
            print(f'\n4. Spatial Correlation (area-weighted, {len(empirical_change_both)} regions):')
            print(f'   Correlation: {corr:.4f}')
            print()




def newell_number_spatial_statistics_subType(self):

    results_dict = self.results_dict
    newell_actions = [a for a in results_dict.keys() if a.startswith('newell_')]
    model_list = self.model_list
    ts_year = 2100

    for model_i in model_list:

        print()
        print('=' * 80)
        print(f'Model: {model_i}')
        print('=' * 80)

        for action_name in newell_actions:

            spec_list = results_dict[action_name][model_i]
            spec0 = spec_list[0]

            years = np.array(spec0['years'])
            plot_years = years[1:]
            idx = int(np.searchsorted(plot_years, ts_year))

            model_simulation_projection = np.array(spec0['model_simulation_projection'])
            model_simulation_references = np.array(spec0['model_simulation_references'])
            weights_projection = spec0['weights_projection']
            regional_areas = weights_projection[:, 0]

            base_growth = np.log(model_simulation_references[:, 1:] / model_simulation_references[:, :-1])
            mod_cumulative_pct = _cumulative_d_b_pct_surface(model_simulation_projection, base_growth)
            model_change = mod_cumulative_pct[:, idx]

            for gdp_form in ('growth', 'level'):

                form_specs = [s for s in spec_list if gdp_form in s['spec']['gdp_form']]
                print()
                print(f'--- {action_name}  ({gdp_form}, {len(form_specs)} specs) ---')

                if len(form_specs) == 0:
                    print('   (no specs for this gdp_form; skip)')
                    print()
                    continue

                emp_surfaces = np.stack(
                    [
                        _cumulative_d_b_pct_surface(np.array(s['empirical_projection_corrected']), base_growth)
                        for s in form_specs
                    ],
                    axis=0,
                )
                empirical_change = np.median(emp_surfaces, axis=0)[:, idx]

                valid_mask_both = np.isfinite(empirical_change) & np.isfinite(model_change)
                empirical_change_both = empirical_change[valid_mask_both]
                model_change_both = model_change[valid_mask_both]
                areas_both = regional_areas[valid_mask_both]

                sign_empirical = np.sign(empirical_change_both)
                sign_model = np.sign(model_change_both)
                same_sign_mask = sign_empirical * sign_model > 0
                diff_sign_mask = sign_empirical * sign_model < 0

                fraction_regions_same = np.sum(same_sign_mask) / len(empirical_change_both) * 100
                fraction_regions_diff = np.sum(diff_sign_mask) / len(empirical_change_both) * 100

                fraction_area_same = np.sum(areas_both[same_sign_mask]) / np.sum(areas_both) * 100
                fraction_area_diff = np.sum(areas_both[diff_sign_mask]) / np.sum(areas_both) * 100

                valid_mask_empirical = np.isfinite(empirical_change)
                empirical_change_valid = empirical_change[valid_mask_empirical]
                areas_empirical = regional_areas[valid_mask_empirical]
                weights_empirical = areas_empirical / np.sum(areas_empirical)

                valid_mask_model = np.isfinite(model_change)
                model_change_valid = model_change[valid_mask_model]
                areas_model = regional_areas[valid_mask_model]
                weights_model = areas_model / np.sum(areas_model)

                mean_empirical = np.average(empirical_change_valid, weights=weights_empirical)
                mean_model = np.average(model_change_valid, weights=weights_model)

                empirical_anom = empirical_change_valid - mean_empirical
                model_anom = model_change_valid - mean_model

                std_empirical = np.sqrt(np.average(empirical_anom**2, weights=weights_empirical))
                std_model = np.sqrt(np.average(model_anom**2, weights=weights_model))

                weights_both = areas_both / np.sum(areas_both)
                mean_empirical_both = np.average(empirical_change_both, weights=weights_both)
                mean_model_both = np.average(model_change_both, weights=weights_both)
                empirical_anom_both = empirical_change_both - mean_empirical_both
                model_anom_both = model_change_both - mean_model_both
                std_empirical_both = np.sqrt(np.average(empirical_anom_both**2, weights=weights_both))
                std_model_both = np.sqrt(np.average(model_anom_both**2, weights=weights_both))

                if std_empirical_both > 0 and std_model_both > 0:
                    cov = np.average(empirical_anom_both * model_anom_both, weights=weights_both)
                    corr = cov / (std_empirical_both * std_model_both)
                else:
                    corr = np.nan 

                print(f'\nMetric: cumulative d-b at {ts_year} (%); empirical = cross-spec median per country.')
                print(f'\n1. Sign Agreement (comparing {len(empirical_change_both)} regions with both valid):')
                print(f'   Same sign:      {np.sum(same_sign_mask):3d} regions ({fraction_regions_same:5.1f}%)')
                print(f'   Different sign: {np.sum(diff_sign_mask):3d} regions ({fraction_regions_diff:5.1f}%)')

                print(f'\n2. Area-weighted Sign Agreement:')
                print(f'   Same sign area:      {fraction_area_same:5.1f}%')
                print(f'   Different sign area: {fraction_area_diff:5.1f}%')

                print(f'\n3. Spatial Standard Deviation (area-weighted, % cumulative d-b):')
                print(f'   Empirical (median across specs): {std_empirical:.4f}  (n={len(empirical_change_valid)} regions)')
                print(f'   Model projection:                 {std_model:.4f}  (n={len(model_change_valid)} regions)')

                print(f'\n4. Spatial Correlation (area-weighted, {len(empirical_change_both)} regions):')
                print(f'   Correlation: {corr:.4f}')
                print()


def number_spatial_statistics(self):
    burke_number_spatial_statistics_subType(self)
    newell_number_spatial_statistics_subType(self)
