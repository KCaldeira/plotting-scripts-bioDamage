"""Demo script for plot_ratios_shaded() — generates 4 PNGs showing each column-count case."""

import numpy as np
from axes import plot_ratios_shaded

x = np.linspace(0, 10, 200)

# Fan-shaped spread that widens with x
center = 1.0 + 0.3 * np.sin(x)  # median ratio oscillates around 1.0

def spread(x, width):
    """Create a symmetric spread that grows with x."""
    return width * x / x.max()

# --- 1 column: plain line ---
fig, ax = plot_ratios_shaded(x, center, title='1 column — plain line',
                             x_axis_label='x', y_axis_label='Change')
fig.savefig('plots/demo_shaded_1col.png', dpi=150, bbox_inches='tight')
print('Saved demo_shaded_1col.png')

# --- 3 columns: one shaded band + median ---
data3 = np.column_stack([
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
])
fig, ax = plot_ratios_shaded(x, data3, title='3 columns — one band',
                             x_axis_label='x', y_axis_label='Change',
                             color='steelblue')
fig.savefig('plots/demo_shaded_3col.png', dpi=150, bbox_inches='tight')
print('Saved demo_shaded_3col.png')

# --- 5 columns: two nested bands + median ---
data5 = np.column_stack([
    center - spread(x, 0.60),
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
    center + spread(x, 0.60),
])
fig, ax = plot_ratios_shaded(x, data5, title='5 columns — two nested bands',
                             x_axis_label='x', y_axis_label='Change',
                             color='darkorange')
fig.savefig('plots/demo_shaded_5col.png', dpi=150, bbox_inches='tight')
print('Saved demo_shaded_5col.png')

# --- 7 columns: two bands + thin outer lines + median ---
data7 = np.column_stack([
    center - spread(x, 0.80),
    center - spread(x, 0.60),
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
    center + spread(x, 0.60),
    center + spread(x, 0.80),
])
fig, ax = plot_ratios_shaded(x, data7, title='7 columns — bands + thin outer lines',
                             x_axis_label='x', y_axis_label='Change',
                             color='seagreen')
fig.savefig('plots/demo_shaded_7col.png', dpi=150, bbox_inches='tight')
print('Saved demo_shaded_7col.png')
