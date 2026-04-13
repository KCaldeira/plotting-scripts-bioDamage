"""Demo script for plot_ratios() — generates 4 PNGs matching demo_plot_shaded.py cases."""

import numpy as np
from axes import plot_ratios

x = np.linspace(0, 10, 200)

# Fan-shaped spread that widens with x
center = 1.0 + 0.3 * np.sin(x)  # median ratio oscillates around 1.0

def spread(x, width):
    """Create a symmetric spread that grows with x."""
    return width * x / x.max()

# --- 1 column: plain line ---
fig, ax = plot_ratios(x, center, title='1 column — plain line',
                      x_axis_label='x', y_axis_label='Change')
fig.savefig('plots/demo_plot_1col.png', dpi=150, bbox_inches='tight')
print('Saved demo_plot_1col.png')

# --- 3 columns: three lines ---
data3 = np.column_stack([
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
])
fig, ax = plot_ratios(x, data3, title='3 columns — three lines',
                      x_axis_label='x', y_axis_label='Change')
fig.savefig('plots/demo_plot_3col.png', dpi=150, bbox_inches='tight')
print('Saved demo_plot_3col.png')

# --- 5 columns: five lines ---
data5 = np.column_stack([
    center - spread(x, 0.60),
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
    center + spread(x, 0.60),
])
fig, ax = plot_ratios(x, data5, title='5 columns — five lines',
                      x_axis_label='x', y_axis_label='Change')
fig.savefig('plots/demo_plot_5col.png', dpi=150, bbox_inches='tight')
print('Saved demo_plot_5col.png')

# --- 7 columns: seven lines ---
data7 = np.column_stack([
    center - spread(x, 0.80),
    center - spread(x, 0.60),
    center - spread(x, 0.30),
    center,
    center + spread(x, 0.30),
    center + spread(x, 0.60),
    center + spread(x, 0.80),
])
fig, ax = plot_ratios(x, data7, title='7 columns — seven lines',
                      x_axis_label='x', y_axis_label='Change')
fig.savefig('plots/demo_plot_7col.png', dpi=150, bbox_inches='tight')
print('Saved demo_plot_7col.png')
