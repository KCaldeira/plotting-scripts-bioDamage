# Amount-to-arcsinh tick generation for matplotlib axes.
#
# Accepts raw amount data (positive, negative, or zero), transforms via
# arcsinh(amount / scale) internally, then computes buffered bounds and
# nicely-spaced tick marks labeled as amount values.
#
# The scale parameter controls the transition from linear behavior near
# zero to logarithmic behavior far from zero.
#
# Standalone usage:
#     plot_min, plot_max, tick_locs, tick_labels = choose_amount_arcsinh_ticks(amin, amax, scale)
#
# High-level usage:
#     bounds, ticks_vals, amount_labels = get_axis_bounds_and_ticks_arcsinh(data, scale)
#     ax.set_ylim(bounds)
#     ax.set_yticks(ticks_vals)
#     ax.set_yticklabels([f'{a:g}' for a in amount_labels])

import math

import numpy as np


def amount_to_arcsinh(a, scale):
    """Convert an amount value to arcsinh-transformed space."""
    return math.asinh(a / scale)


def arcsinh_to_amount(x, scale):
    """Convert an arcsinh-transformed value back to amount."""
    return scale * math.sinh(x)


def format_amount(a):
    """Format amount labels cleanly."""
    if abs(a) < 1e-12:
        return "0"
    sign = "+" if a > 0 else ""
    return f"{sign}{a:g}"


def build_amount_candidates():
    """Build a library of 'nice' amount ticks.

    Uses a 1-2-5 sequence across magnitudes 10^-4 to 10^7,
    both positive and negative, plus zero.
    """
    amounts = set()
    amounts.add(0)

    for exp in range(-4, 8):  # 10^-4 to 10^7
        for base in [1, 2, 5]:
            val = base * 10**exp
            amounts.add(val)
            amounts.add(-val)

    return sorted(amounts)


def _find_nearest_candidate(target, candidates_transformed):
    """Find the candidate in transformed space nearest to target."""
    idx = np.searchsorted(candidates_transformed, target)
    best_idx = idx
    best_dist = float('inf')
    for i in [max(0, idx - 1), min(len(candidates_transformed) - 1, idx)]:
        dist = abs(candidates_transformed[i] - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def choose_amount_arcsinh_ticks(amin, amax, scale, symmetric=False, buffer=0.05):
    """Choose nicely-spaced tick marks for an arcsinh-transformed amount axis.

    Accepts raw amount values, transforms via arcsinh(amount / scale) internally,
    then finds boundary ticks from nice candidates and fills interior with
    7 total ticks proportionally allocated between negative and positive sides.

    Args:
        amin: minimum amount value
        amax: maximum amount value
        scale: arcsinh scale parameter (controls linear-to-log transition)
        symmetric: if True, make bounds symmetric around zero (in arcsinh space)
        buffer: fractional buffer to add beyond data range (default 0.05)

    Returns:
        plot_min: adjusted minimum bound in arcsinh space
        plot_max: adjusted maximum bound in arcsinh space
        tick_locations: list of tick positions in arcsinh space
        tick_labels: list of formatted amount strings
    """
    # Convert amounts to arcsinh space
    xmin = amount_to_arcsinh(amin, scale)
    xmax = amount_to_arcsinh(amax, scale)

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

    # Build candidates in arcsinh space
    amount_candidates = build_amount_candidates()
    candidates = [(a, amount_to_arcsinh(a, scale)) for a in amount_candidates]
    candidates.sort(key=lambda t: t[1])
    candidates_transformed = np.array([x for _, x in candidates])
    candidates_amounts = [a for a, _ in candidates]

    # Step 2: Find boundary ticks (outermost nice ticks)
    # Upper bound: look for nice candidate in [y_max_new * (1 - 2*buffer), y_max_new]
    upper_inner = y_max_new * (1 - 2 * buffer)
    upper_candidates = [(i, candidates_transformed[i]) for i in range(len(candidates_transformed))
                        if upper_inner <= candidates_transformed[i] <= y_max_new]
    if upper_candidates:
        max_tick_idx = upper_candidates[-1][0]
    else:
        max_tick_idx = _find_nearest_candidate(y_max_new, candidates_transformed)

    # Lower bound: look for nice candidate in [y_min_new, y_min_new * (1 - 2*buffer)]
    lower_inner = y_min_new * (1 - 2 * buffer)  # closer to zero since y_min_new is negative
    lower_candidates = [(i, candidates_transformed[i]) for i in range(len(candidates_transformed))
                        if y_min_new <= candidates_transformed[i] <= lower_inner]
    if lower_candidates:
        min_tick_idx = lower_candidates[0][0]
    else:
        min_tick_idx = _find_nearest_candidate(y_min_new, candidates_transformed)

    min_tick = candidates_transformed[min_tick_idx]
    max_tick = candidates_transformed[max_tick_idx]

    # Step 3: Fill interior ticks (7 total)
    # 3 fixed: min_tick, 0, max_tick
    # 4 additional allocated proportionally between negative and positive sides
    full_span = max_tick - min_tick
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
        idx = _find_nearest_candidate(target, candidates_transformed)
        if idx not in seen:
            seen.add(idx)
            ticks.append((candidates_amounts[idx], candidates_transformed[idx]))

    ticks.sort(key=lambda t: t[1])

    # Ensure zero is included
    if not any(abs(x) < 1e-12 for _, x in ticks):
        zero_idx = _find_nearest_candidate(0.0, candidates_transformed)
        if abs(candidates_transformed[zero_idx]) < 1e-12:
            ticks.append((0, 0.0))
            ticks.sort(key=lambda t: t[1])

    # Plot bounds cover both the outermost ticks and the buffered data range
    plot_min = min(ticks[0][1], y_min_new)
    plot_max = max(ticks[-1][1], y_max_new)

    tick_locations = [x for _, x in ticks]
    tick_labels = [format_amount(a) for a, _ in ticks]

    return plot_min, plot_max, tick_locations, tick_labels


def get_axis_bounds_and_ticks_arcsinh(data, scale, padding=0.0, symmetric=False, buffer=0.05):
    """Calculate axis bounds and ticks for amount data with arcsinh transform.

    Accepts raw amount values (positive, negative, or zero), transforms via
    arcsinh(amount / scale), then computes tick positions in arcsinh space with
    corresponding amount labels.

    Args:
        data: iterable of amount values (e.g. [-5, 100]).
            Often just [min_amount, max_amount].
        scale: arcsinh scale parameter (controls linear-to-log transition)
        padding: fractional padding added to each side of the data range
            (in arcsinh space)
        symmetric: if True, make bounds symmetric around zero (in arcsinh space)
        buffer: fractional buffer for tick boundary selection (default 0.05)

    Returns:
        bounds: [min, max] in arcsinh space, suitable for ax.set_ylim
        ticks_vals: numpy array of tick positions in arcsinh space
        amount_labels: list of numeric amount values at each tick
            (e.g. [-5, -2, 0, 10, 50, 100]). Use with
            ax.set_yticklabels([f'{a:g}' for a in amount_labels]).
    """
    finite_data = [x for x in data if np.isfinite(x)]
    # Convert amounts to arcsinh space
    arcsinh_data = [amount_to_arcsinh(x, scale) for x in finite_data]
    xmin = min(arcsinh_data) if arcsinh_data else -0.1
    xmax = max(arcsinh_data) if arcsinh_data else 0.1
    span = xmax - xmin
    xmin -= span * padding
    xmax += span * padding

    # Convert padded arcsinh bounds back to amounts
    amin = arcsinh_to_amount(xmin, scale)
    amax = arcsinh_to_amount(xmax, scale)

    plot_min, plot_max, tick_locations, _ = choose_amount_arcsinh_ticks(
        amin, amax, scale, symmetric=symmetric, buffer=buffer
    )

    # Add a small margin so outermost ticks aren't clipped by matplotlib
    margin = (plot_max - plot_min) * 0.02
    bounds = [plot_min - margin, plot_max + margin]
    ticks_vals = np.array(tick_locations)
    amount_labels = [arcsinh_to_amount(x, scale) for x in tick_locations]

    return bounds, ticks_vals, amount_labels
