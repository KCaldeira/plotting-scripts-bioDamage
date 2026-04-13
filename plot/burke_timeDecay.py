import matplotlib.pyplot as plt
import numpy as np


def burke_timeDecay_distribution(self):

    model_list = self.model_list
    results_dict = self.results_dict

    fig, axes = plt.subplots(1, len(model_list), figsize=(5 * len(model_list), 4), sharey=True, sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, model_i in zip(axes, model_list):

        df_results = results_dict['burke_timeDecay'][model_i]
        f_decay_list = np.array(df_results['f_decay'].values.tolist()) 
        central_estimate = f_decay_list[0]
        bootstrapping_estimate = f_decay_list[1:]
        percentile_5,  percentile_95 = np.percentile(bootstrapping_estimate,  5), np.percentile(bootstrapping_estimate, 95)
        percentile_25, percentile_75 = np.percentile(bootstrapping_estimate, 25), np.percentile(bootstrapping_estimate, 75)

        print (f'{1/central_estimate} [{1/percentile_5}, {1/percentile_95}]')

        #### Histogram of bootstrap estimates
        ax.hist(bootstrapping_estimate, bins=30, density=True, color='steelblue', alpha=0.6, edgecolor='none')

        #### Shaded percentile ranges of bootstrap estimates
        ax.axvspan(percentile_5,  percentile_95, alpha=0.15, color='steelblue', label='5th–95th pct')
        ax.axvspan(percentile_25, percentile_75, alpha=0.30, color='steelblue', label='25th–75th pct')

        #### Vertical line for central estimate
        ax.axvline(central_estimate, color='black', linestyle='--', linewidth=2, label='Central estimate')

        ax.set_xlabel('f_decay', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(model_i, fontsize=12) 
        ax.legend(fontsize=9)
        
    axes[0].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
    # plt.savefig('burke_timeDecay_distribution.svg')
    # plt.clf()


def burke_timeDecay_responseFunc(self):

    model_list   = self.model_list
    results_dict = self.results_dict

    t_examine = np.arange(-5, 30.1, 0.1)
    length_t  = len(t_examine)
    color     = 'steelblue'

    n_models = len(model_list)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(5 * n_models, 4),
                             sharey=True, sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, model_i in zip(axes, model_list):

        df_results = results_dict['burke_timeDecay'][model_i]
        h1_list    = np.array(df_results['h1'].values.tolist()).astype(float)
        h2_list    = np.array(df_results['h2'].values.tolist()).astype(float)
        num_row    = len(h1_list)

        full_results = np.zeros([num_row, length_t])
        for row_i in range(num_row):
            tas_response     = h1_list[row_i] * t_examine + h2_list[row_i] * t_examine**2
            optimal_temp     = -h1_list[row_i] / (2 * h2_list[row_i])
            optimal_response = h1_list[row_i] * optimal_temp + h2_list[row_i] * optimal_temp**2
            full_results[row_i, :] = (np.exp(tas_response - optimal_response) - 1) * 100

        ax.plot(t_examine, full_results[0], color=color, linewidth=2, linestyle='-')
        ax.fill_between(t_examine,
                        np.percentile(full_results[1:],  5, axis=0),
                        np.percentile(full_results[1:], 95, axis=0),
                        facecolor=color, edgecolor='none', alpha=0.1)
        ax.fill_between(t_examine,
                        np.percentile(full_results[1:], 25, axis=0),
                        np.percentile(full_results[1:], 75, axis=0),
                        facecolor=color, edgecolor='none', alpha=0.3)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5, zorder=5)
        ax.set_xlim(-5, 30)
        ax.set_ylim(-100, 5)
        ax.set_aspect((30 - (-5)) / (5 - (-100)))
        ax.set_xlabel('Temperature anomaly  (°C)')
        ax.set_title(model_i)

    axes[0].set_ylabel('GPP response  (%)')
    plt.show()
    # plt.savefig('burke_timeDecay_responseFunc.svg')
    # plt.clf()
