"""Demo script for arcsinh axis — scatter plot with positive and negative values."""

import numpy as np
from axes import plot_amounts_arcsinh

np.random.seed(42)

x = np.arange(0, 1.001, 0.001)  # 1001 points from 0 to 1
y =  np.random.normal(loc=0, scale=1, size=len(x))
y = np.exp(np.abs(y))  # assume log normal distribution

# Set scale to cover 95% of the data (middle 95%)
scale = np.percentile(np.abs(y), 97.5)

# Set axis range to cover 99.9% of the data
p_low, p_high = np.percentile(y, [0.05, 99.95])

fig, ax = plot_amounts_arcsinh(x, y, scale=scale,
                               title=f'Arcsinh scatterplot demo (scale={scale:.0f})',
                               x_axis_label='x', y_axis_label='Amount')

# Replace lines with scatter points
ax.lines[0].remove()
transformed_y = np.arcsinh(y / scale)
ax.scatter(x, transformed_y, color='steelblue', s=5, zorder=3)

# Override y-axis bounds to cover 99% of data
from amount_arcsinh_ticks import get_axis_bounds_and_ticks_arcsinh
bounds, ticks, amount_labels = get_axis_bounds_and_ticks_arcsinh([p_low, p_high], scale=scale)
ax.set_ylim(bounds)
ax.set_yticks(ticks)
ax.set_yticklabels([f'{a:g}' for a in amount_labels])

fig.savefig('plots/demo_arcsinh_scatterplot.png', dpi=150, bbox_inches='tight')
print('Saved plots/demo_arcsinh_scatterplot.png')
