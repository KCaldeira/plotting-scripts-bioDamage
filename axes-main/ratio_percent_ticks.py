# Ratio-to-percent tick generation for matplotlib axes.
#
# Accepts raw ratio data (where 1.0 = no change), log-transforms internally,
# then computes buffered bounds and nicely-spaced tick marks labeled as
# percentage change.
#
# Standalone usage:
#     plot_min, plot_max, tick_locs, tick_labels = choose_ratio_percent_ticks(rmin, rmax)
#
# High-level usage:
#     bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ratio_pct(data, padding=0.05)
#     ax.set_ylim(bounds)
#     ax.set_yticks(ticks_vals)
#     ax.set_yticklabels([f'{p:g}%' for p in pct_labels])

import math

import numpy as np


def percent_to_log(p):
    """Convert percent change to natural log ratio."""
    return math.log(1 + p / 100.0)


def log_to_percent(x):
    """Convert natural log ratio to percent change."""
    return 100 * (math.exp(x) - 1)


def ratio_to_percent(r):
    """Convert a ratio to percent change.  ratio 1.0 = 0%, ratio 2.0 = +100%."""
    return 100 * (r - 1)


def format_percent(p):
    """Format percent labels cleanly."""
    if abs(p) < 1e-12:
        return "0%"
    sign = "+" if p > 0 else ""
    if abs(p) >= 1:
        return f"{sign}{round(p):.0f}%"
    else:
        return f"{sign}{p:.2f}%"


def build_percent_candidates():
    """Build a library of 'nice' percent ticks.

    Negative side:
        -90, -98, -99, -99.5
        plus survival powers of 10 ( -99.9, -99.99, ... )

    Positive side:
        1-2-5 sequence (10%, 20%, 50%, 100%, 200%, 500%, ...)
    """
    perc = set()

    # basic negative ticks
    perc.update([-90, -98, -99, -99.5])

    # survival powers of 10: -99.9, -99.99, -99.999, ...
    for k in range(1, 8):
        s = 10 ** (-k)
        p = -100 * (1 - s)
        perc.add(p)

    # small and mid-range negative values
    perc.update([-80, -75, -70, -60, -50, -30, -20, -10, -5, -2, -1])

    # zero
    perc.add(0)

    # positive 1-2-3-5 sequence
    for scale in [1, 10, 100, 1000, 10000]:
        for base in [1, 2, 3, 5]:
            perc.add(base * scale)

    # keep values > -100
    perc = [p for p in perc if p > -100]

    return sorted(perc)


def _find_nearest_candidate(target, candidates_log):
    """Find the candidate in log space nearest to target."""
    idx = np.searchsorted(candidates_log, target)
    # Check the two nearest candidates
    best_idx = idx
    best_dist = float('inf')
    for i in [max(0, idx - 1), min(len(candidates_log) - 1, idx)]:
        dist = abs(candidates_log[i] - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def choose_ratio_percent_ticks(rmin, rmax, symmetric=False, buffer=0.05):
    """Choose nicely-spaced tick marks for a ratio axis with percent labels.

    Accepts raw ratio values (1.0 = no change), log-transforms internally,
    then finds boundary ticks from nice candidates and fills interior with
    7 total ticks proportionally allocated between negative and positive sides.

    Args:
        rmin: minimum ratio value (e.g. 0.5 for a 50% decline)
        rmax: maximum ratio value (e.g. 2.0 for a 100% increase)
        symmetric: if True, make bounds symmetric around zero (in log space)
        buffer: fractional buffer to add beyond data range (default 0.05)

    Returns:
        plot_min: adjusted minimum bound in log space
        plot_max: adjusted maximum bound in log space
        tick_locations: list of tick positions in log space
        tick_labels: list of formatted percentage strings
    """
    # Convert ratios to log space
    xmin = math.log(rmin)
    xmax = math.log(rmax)

    if xmax < xmin:
        xmin, xmax = xmax, xmin

    # ensure zero included if one-sided
    if xmin > 0:
        xmin = 0
    if xmax < 0:
        xmax = 0

    # Step 1: Compute buffered bounds
    if symmetric:
        y_extreme = max(xmax, -xmin)
        y_max_new = y_extreme * (1 + buffer)
        y_min_new = -y_max_new
    else:
        y_max_new = xmax * (1 + buffer)
        y_min_new = xmin * (1 + buffer)  # xmin is negative, so this makes it more negative

    # Build candidates in log space
    perc_candidates = build_percent_candidates()
    candidates = [(p, percent_to_log(p)) for p in perc_candidates]
    candidates.sort(key=lambda t: t[1])
    candidates_log = np.array([x for _, x in candidates])
    candidates_pct = [p for p, _ in candidates]

    # Step 2: Find boundary ticks (outermost nice ticks)
    # Strategy: look for a nice candidate in a window near the buffered bound.
    # If none found, snap to the nearest candidate (inward is fine — plot bounds
    # are set independently to cover the buffered data range).

    # Upper bound: look for nice candidate in [y_max_new * (1 - 2*buffer), y_max_new]
    upper_inner = y_max_new * (1 - 2 * buffer)
    upper_candidates = [(i, candidates_log[i]) for i in range(len(candidates_log))
                        if upper_inner <= candidates_log[i] <= y_max_new]
    if upper_candidates:
        max_tick_idx = upper_candidates[-1][0]  # pick the outermost one in range
    else:
        max_tick_idx = _find_nearest_candidate(y_max_new, candidates_log)

    # Lower bound: look for nice candidate in [y_min_new, y_min_new * (1 - 2*buffer)]
    lower_inner = y_min_new * (1 - 2 * buffer)  # closer to zero since y_min_new is negative
    lower_candidates = [(i, candidates_log[i]) for i in range(len(candidates_log))
                        if y_min_new <= candidates_log[i] <= lower_inner]
    if lower_candidates:
        min_tick_idx = lower_candidates[0][0]  # pick the outermost one (most negative)
    else:
        min_tick_idx = _find_nearest_candidate(y_min_new, candidates_log)

    min_tick = candidates_log[min_tick_idx]
    max_tick = candidates_log[max_tick_idx]

    # Step 3: Fill interior ticks (7 total)
    # 3 fixed: min_tick, 0, max_tick
    # 4 additional allocated proportionally between negative and positive sides
    full_span = max_tick - min_tick  # min_tick is negative, so this is the full range
    n_pos = round(max_tick / full_span * 4) if full_span > 0 else 2
    n_neg = 4 - n_pos

    targets = [min_tick]
    for k in range(1, n_neg + 1):
        targets.append(k / (n_neg + 1) * min_tick)
    targets.append(0.0)
    for k in range(1, n_pos + 1):
        targets.append(k / (n_pos + 1) * max_tick)
    targets.append(max_tick)

    # Snap each target to nearest candidate, deduplicate
    seen = set()
    ticks = []
    for target in targets:
        idx = _find_nearest_candidate(target, candidates_log)
        if idx not in seen:
            seen.add(idx)
            ticks.append((candidates_pct[idx], candidates_log[idx]))

    ticks.sort(key=lambda t: t[1])

    # Ensure zero is included
    if not any(abs(x) < 1e-12 for _, x in ticks):
        zero_idx = _find_nearest_candidate(0.0, candidates_log)
        if abs(candidates_log[zero_idx]) < 1e-12:
            ticks.append((0, 0.0))
            ticks.sort(key=lambda t: t[1])

    # Plot bounds cover both the outermost ticks and the buffered data range
    plot_min = min(ticks[0][1], y_min_new)
    plot_max = max(ticks[-1][1], y_max_new)

    tick_locations = [x for _, x in ticks]
    tick_labels = [format_percent(p) for p, _ in ticks]

    return plot_min, plot_max, tick_locations, tick_labels


def get_axis_bounds_and_ticks_ratio_pct(data, padding=0.0, symmetric=False, buffer=0.05):
    """Calculate axis bounds and ticks for ratio data with percentage-change labels.

    Accepts raw ratio values (where 1.0 = no change, 2.0 = +100%, 0.5 = -50%),
    log-transforms internally, then computes tick positions in log space with
    corresponding percentage labels.

    Args:
        data: iterable of ratio values (e.g. [0.8, 1.5] meaning -20% to +50%).
            Often just [min_ratio, max_ratio].
        padding: fractional padding added to each side of the data range
            (in log space)
        symmetric: if True, make bounds symmetric around zero (in log space)
        buffer: fractional buffer for tick boundary selection (default 0.05)

    Returns:
        bounds: [min, max] in log space, suitable for ax.set_ylim
        ticks_vals: numpy array of tick positions in log space
        pct_labels: list of numeric percentage values at each tick
            (e.g. [-25, 0, 25]). Use with
            ax.set_yticklabels([f'{p:g}%' for p in pct_labels]).
    """
    finite_data = [x for x in data if np.isfinite(x)]
    # Convert ratios to log space
    log_data = [math.log(x) for x in finite_data if x > 0]
    xmin = min(log_data) if log_data else -0.1
    xmax = max(log_data) if log_data else 0.1
    span = xmax - xmin
    xmin -= span * padding
    xmax += span * padding

    # Convert padded log bounds back to ratios for choose_ratio_percent_ticks
    rmin = math.exp(xmin)
    rmax = math.exp(xmax)

    plot_min, plot_max, tick_locations, _ = choose_ratio_percent_ticks(
        rmin, rmax, symmetric=symmetric, buffer=buffer
    )

    # Add a small margin so outermost ticks aren't clipped by matplotlib
    margin = (plot_max - plot_min) * 0.02
    bounds = [plot_min - margin, plot_max + margin]
    ticks_vals = np.array(tick_locations)
    pct_labels = [log_to_percent(x) for x in tick_locations]

    return bounds, ticks_vals, pct_labels
