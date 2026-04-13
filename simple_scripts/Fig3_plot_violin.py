import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
from ratio_percent_ticks import get_axis_bounds_and_ticks_ratio_pct, format_percent

_LN10 = np.log(10)


def _weighted_mean_log10(y, weights):
    ok = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.any(ok):
        return float(np.average(y[ok], weights=weights[ok]))
    ok2 = np.isfinite(y)
    if np.any(ok2):
        return float(np.nanmean(y[ok2]))
    return None


def plot_violin_panel(df_model_results, ax, title=None):
    """One panel: Model, Burke central, and Newell median (level, per country)."""
    color = 'royalblue'
    y_model = df_model_results['model_log10ratio'].to_numpy()
    y_burke = df_model_results['burke_level_log10ratio'].to_numpy()
    y_newell = df_model_results['newell_level_log10ratio'].to_numpy()
    area = df_model_results['area'].to_numpy()

    datasets, positions, labels_use = [], [], []
    star_x, star_y, star_ec, star_fc = [], [], [], []
    y_all = []

    x_labels = [
        'Model\n(per country)',
        'Burke central\n(per country)',
        'Newell median\n(per country)',
    ]
    series = [
        (y_model, 1.0, 'model'),
        (y_burke, 2.0, 'burke'),
        (y_newell, 3.0, 'newell'),
    ]

    for y_arr, pos, kind in series:
        ok = np.isfinite(y_arr)
        if not np.any(ok):
            continue
        datasets.append(y_arr[ok])
        positions.append(pos)
        labels_use.append(x_labels[int(pos) - 1])
        sy = _weighted_mean_log10(y_arr, area)
        if sy is not None:
            star_x.append(pos)
            star_y.append(sy)
            if kind == 'model':
                star_ec.append('black')
                star_fc.append('white')
            elif kind == 'burke':
                star_ec.append(color)
                star_fc.append('white')
            else:
                star_ec.append(color)
                star_fc.append(color)
        y_all.extend(y_arr[ok].tolist())
        if sy is not None:
            y_all.append(sy)

    if len(datasets) > 0:
        vp = ax.violinplot(
            datasets,
            positions=positions,
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for b, body in enumerate(vp['bodies']):
            pos = positions[b]
            if pos == 1.0:
                fc, ec, al = '0.25', 'black', 0.75
            elif pos == 2.0:
                fc, ec, al = color, color, 0.55
            else:
                fc, ec, al = color, color, 0.35
            body.set_facecolor(fc)
            body.set_edgecolor(ec)
            body.set_alpha(al)
            body.set_linewidth(1.2)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in vp and vp[partname] is not None:
                vp[partname].set_visible(False)

    if len(star_x) > 0:
        ax.scatter(
            star_x,
            star_y,
            marker='*',
            s=110,
            zorder=5,
            facecolors=star_fc,
            edgecolors=star_ec,
            linewidths=1.4,
        )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
    if len(positions) > 0:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_use, rotation=12, ha='right')
    ax.set_xlim(0.35, 3.65)

    # Set y-axis ticks using ratio_percent_ticks utility
    if len(y_all) > 0:
        y_arr = np.array(y_all)
        ratios = 10.0 ** y_arr[np.isfinite(y_arr)]
        bounds_ln, ticks_ln, pct_labels = get_axis_bounds_and_ticks_ratio_pct(
            [ratios.min(), ratios.max()])
        bounds = [b / _LN10 for b in bounds_ln]
        ticks_log10 = [t / _LN10 for t in ticks_ln]
        ax.set_ylim(bounds)
        ax.set_yticks(ticks_log10)
        ax.set_yticklabels([format_percent(p) for p in pct_labels])
    else:
        ax.set_ylim(-0.2, 0.0)
        ax.set_yticks([-0.2, -0.1, 0.0])

    ax.set_box_aspect(1)
    ax.set_ylabel('GPP change  (full vs BGC)')
    if title is not None:
        ax.set_title(title)


def fig3_plot_violin():

    df_results = pd.read_csv(f'./data/input/figure3_data.csv')

    model_list = df_results['model_name'].unique()
    n_models = len(model_list)
    fig_w_in = 6.0
    fig_h_in = fig_w_in * n_models * 0.55
    fig, axs = plt.subplots(n_models, 1, figsize=(fig_w_in, fig_h_in), constrained_layout=True)
    axs = np.atleast_1d(axs)

    for model_idx, model_i in enumerate(model_list):
        df_model_results = df_results[df_results['model_name'] == model_i]
        plot_violin_panel(
            df_model_results,
            axs[model_idx],
            title=f'{model_i}  —  level  (2080–2100)',
        )

    fig.savefig('./data/output/Fig3_violin.pdf', dpi=300)
    plt.close(fig)
