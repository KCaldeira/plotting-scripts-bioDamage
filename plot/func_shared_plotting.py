import os
import sys 
import numpy as np 

_axes_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'axes-main'))
if _axes_main not in sys.path:
    sys.path.insert(0, _axes_main)
from amount_arcsinh_ticks import get_axis_bounds_and_ticks_arcsinh
from axes import get_axis_bounds_and_ticks


def _pool_bounds(bounds_list):
    los, his = zip(*bounds_list)
    return float(min(los)), float(max(his))


def _apply_yaxis_growth_arcsinh(ax_ref, amin, amax, plot_type, symmetric):
    padding = 0 if plot_type == 'main' else 0
    buffer = 0 if plot_type == 'main' else 0
    bounds, ticks, amount_labels = get_axis_bounds_and_ticks_arcsinh(
        [amin, amax], scale=1.0, padding=padding, symmetric=symmetric, buffer=buffer)
    ax_ref.set_ylim(bounds)
    ax_ref.set_yticks(ticks)
    ax_ref.set_yticklabels([f'{a:g}' for a in amount_labels])


def _apply_yaxis_level_linear(ax_ref, ymin, ymax, plot_type):
    padding = 0 if plot_type == 'main' else 0
    bounds, ticks_vals = get_axis_bounds_and_ticks([ymin, ymax], padding=padding)
    ax_ref.set_ylim(bounds)
    ax_ref.set_yticks(ticks_vals)
    ax_ref.set_yticklabels([f'{t:g}' for t in ticks_vals])


def _ribbon_aligned_bounds(model_projection, input_all_cases):
    """Min/max of what plot_this_panel draws: black line, first colored line, and 5–95% ribbons."""
    m = np.asarray(model_projection, dtype=float)
    a = np.asarray(input_all_cases, dtype=float)
    lo = float(np.min(m))
    hi = float(np.max(m)) 
    row0 = a[0, :]
    lo = min(lo, float(np.min(row0)))
    hi = max(hi, float(np.max(row0)))
    p5 = np.percentile(a, 5, axis=0)
    p95 = np.percentile(a, 95, axis=0)
    lo = min(lo, float(np.min(p5)))
    hi = max(hi, float(np.max(p95)))
    return lo, hi