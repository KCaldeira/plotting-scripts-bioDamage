import argparse
import os, json 
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle 
from plot.growth_rate_global_mean_timeSeries               import growth_rate_global_mean_timeSeries
from plot.growth_rate_country_barPlotDistribution          import growth_rate_country_barPlotDistribution
from plot.growth_rate_country_boxplotLikeDistribution      import growth_rate_country_boxplotLikeDistribution
from plot.growth_rate_country_boxplotLikeSelectedCountries import growth_rate_country_boxplotLikeSelectedCountries
from plot.gpp_country_map     import gpp_country_map
from plot.gpp_country_scatter import gpp_country_scatter
from plot.gpp_country_violin  import gpp_country_violin
from joblib import dump, load


class SimpleNamespace:
    def __init__(self, d):
        self.__dict__.update(d)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration file from the specified path.
    """

    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config 


def run_pipeline(config):

    #### Read data
    data = load('main_analysis_state.joblib')
    main_analysis = SimpleNamespace(data)

    #### Plotting 
    action = 'action_Burke_1'
    plotting_list = config[action]['plotting_list']
    for plotting_name in plotting_list.keys(): 
        if plotting_list[plotting_name]: 
            if plotting_name == 'growth_rate_global_mean_timeSeries':               growth_rate_global_mean_timeSeries(main_analysis)
            if plotting_name == 'growth_rate_country_barPlotDistribution':          growth_rate_country_barPlotDistribution(main_analysis)
            if plotting_name == 'growth_rate_country_boxplotLikeDistribution':      growth_rate_country_boxplotLikeDistribution(main_analysis) 
            if plotting_name == 'growth_rate_country_boxplotLikeSelectedCountries': growth_rate_country_boxplotLikeSelectedCountries(main_analysis) 
            if plotting_name == 'gpp_country_map':     gpp_country_map(main_analysis)
            if plotting_name == 'gpp_country_scatter': gpp_country_scatter(main_analysis)
            if plotting_name == 'gpp_country_violin':  gpp_country_violin(main_analysis)


def main():

    """Main entry point for integrated processing pipeline."""

    #### Parse the arguments
    parser = argparse.ArgumentParser(
                description="Apply Empirical Response Functions on Model-simulated Land Biosphere Data",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
                    Examples:
                    python main.py run_full_workflow_001.json
                    python main.py config.json --example-para "xxxx" """
                    )
    parser.add_argument('config',                      help='Path to configuration JSON file, required')
    parser.add_argument('--example-para',              help='Example parameters to add') 
    args = parser.parse_args() 

    #### Load the config
    config = load_config(args.config)

    #### Run the pipeline
    run_pipeline(config) 

if __name__ == "__main__":
    main() 