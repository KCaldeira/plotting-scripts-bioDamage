# axes

Utility functions for calculating axis bounds, tick marks, and plotting data with percentage-change axes (log scale) or arcsinh-transformed axes.

## Usage from Another Project

Add the `axes` directory to your Python path, then import what you need:

```python
import sys
sys.path.insert(0, '/path/to/axes')

from axes import plot_ratios, plot_ratios_shaded
from axes import plot_amounts_arcsinh, plot_amounts_shaded_arcsinh
from axes import get_axis_bounds_and_ticks, get_axis_bounds_and_ticks_ratio_pct
from axes import get_axis_bounds_and_ticks_arcsinh
```

Alternatively, create a symlink to the `axes` directory from your project:

```bash
ln -s /path/to/axes /path/to/your_project/axes
```

Then import directly:

```python
from axes import plot_ratios, plot_amounts_arcsinh
```

### When to use which transform

| Data type | Function family | Transform | Labels |
|-----------|----------------|-----------|--------|
| Ratios (1.0 = no change) | `plot_ratios`, `get_axis_bounds_and_ticks_ratio_pct` | `ln(ratio)` | Percentage change (e.g. +50%, -20%) |
| Amounts (pos/neg/zero) | `plot_amounts_arcsinh`, `get_axis_bounds_and_ticks_arcsinh` | `arcsinh(amount / scale)` | Raw amounts (e.g. +100, -5) |
| Simple numeric data | `get_axis_bounds_and_ticks` | None (linear) | Raw values |

### Choosing the `scale` parameter for arcsinh

The `scale` parameter controls where the arcsinh axis transitions from linear (near zero) to logarithmic (far from zero). Values within `[-scale, scale]` appear roughly linear; values beyond that are compressed logarithmically.

A practical approach: set `scale` based on a percentile of your data:

```python
scale = np.percentile(np.abs(data), 97.5)  # 95% of data in the linear region
```

## Plotting Functions

### `plot_ratios(x_data, ratio_data, ..., colors=None)`

Plots ratio data (where 1.0 = no change) with a log-ratio y-axis labeled as percentage change. Each column of `ratio_data` is drawn as a separate line.

```python
import numpy as np
from axes import plot_ratios

x = np.arange(10)
ratios = np.column_stack([1 + 0.05 * x, 1 - 0.03 * x])
fig, ax = plot_ratios(x, ratios, x_axis_label='Year', y_axis_label='Change',
                      title='Two series', colors=['steelblue', 'darkorange'])
```

Optional `colors` argument accepts a list of matplotlib colors, one per column.

### `plot_ratios_shaded(x_data, ratio_data, ..., color=None)`

Plots ratio data with shaded uncertainty bands. The number of columns determines the visual style:

| Columns | Rendering |
|---------|-----------|
| 1 | Solid line |
| 3 | Shaded band + median line |
| 5 | Two nested bands (lighter outer, darker inner) + median line |
| 7 | Two nested bands + thin outer lines + median line |

All elements use a single color, distinguished by alpha and line width.

```python
import numpy as np
from axes import plot_ratios_shaded

x = np.linspace(0, 10, 100)
center = 1.0 + 0.2 * np.sin(x)
spread = 0.3 * x / x.max()
data5 = np.column_stack([center - 2*spread, center - spread, center,
                         center + spread, center + 2*spread])
fig, ax = plot_ratios_shaded(x, data5, title='Uncertainty bands', color='seagreen')
```

Optional `color` argument accepts any matplotlib color spec (default: first color from the default cycle).

### `plot_amounts_arcsinh(x_data, amount_data, scale, ..., colors=None)`

Plots amount data (positive, negative, or zero) with an arcsinh-transformed y-axis. The `scale` parameter controls the transition from linear behavior near zero to logarithmic behavior far from zero. Each column of `amount_data` is drawn as a separate line.

```python
import numpy as np
from axes import plot_amounts_arcsinh

x = np.arange(10)
amounts = np.column_stack([x * 0.5, -x * 0.3])
fig, ax = plot_amounts_arcsinh(x, amounts, scale=2, x_axis_label='Year', y_axis_label='Change',
                               title='Two series', colors=['steelblue', 'darkorange'])
```

### `plot_amounts_shaded_arcsinh(x_data, amount_data, scale, ..., color=None)`

Plots amount data with shaded uncertainty bands using arcsinh transform. Same column conventions as `plot_ratios_shaded` (1, 3, 5, or 7 columns).

```python
import numpy as np
from axes import plot_amounts_shaded_arcsinh

x = np.linspace(0, 10, 100)
center = 2 * np.sin(x)
spread = 0.5 * x / x.max()
data5 = np.column_stack([center - 2*spread, center - spread, center,
                         center + spread, center + 2*spread])
fig, ax = plot_amounts_shaded_arcsinh(x, data5, scale=1, title='Arcsinh bands', color='seagreen')
```

## Axis Utility Functions

### `get_axis_bounds_and_ticks(data, padding=0.1)`

Computes axis bounds and evenly spaced "nice" tick positions for numerical data. Automatically selects tick spacing based on the data range and anchors ticks at zero when the range spans zero.

```python
from axes import get_axis_bounds_and_ticks

bounds, ticks = get_axis_bounds_and_ticks([-0.2, 0.8], padding=0.1)
```

### `get_axis_bounds_and_ticks_ratio_pct(data, padding=0.1)`

For ratio data (where 1.0 = no change, 2.0 = +100%, 0.5 = -50%), computes axis bounds and tick positions in log space with percentage-change labels. The function log-transforms the ratio data internally.

```python
import numpy as np
from axes import get_axis_bounds_and_ticks_ratio_pct

data = [0.5, 3.0]  # ratios: -50% to +200%
bounds, ticks, pct_labels = get_axis_bounds_and_ticks_ratio_pct(data, padding=0.0)
# pct_labels contains nice percentage values like [-50, -20, 0, 50, 100, 200]
```

### `get_axis_bounds_and_ticks_arcsinh(data, scale, padding=0.0)`

For amount data (positive, negative, or zero), computes axis bounds and tick positions in arcsinh space with amount labels. The `scale` parameter controls the linear-to-logarithmic transition.

```python
from axes import get_axis_bounds_and_ticks_arcsinh

bounds, ticks, amount_labels = get_axis_bounds_and_ticks_arcsinh([-5, 100], scale=2)
# amount_labels contains nice amount values like [-5, -2, 0, 10, 50, 100]
```

## Demo Scripts

- `demo_plot.py` — generates demo PNGs for `plot_ratios()` with 1, 3, 5, and 7 columns
- `demo_plot_shaded.py` — generates demo PNGs for `plot_ratios_shaded()` with 1, 3, 5, and 7 columns
- `demo_arcsinh_scatterplot.py` — scatter plot demo for arcsinh-transformed axis

Output is saved to the `plots/` subdirectory.

## Requirements

- Python 3
- NumPy
- Matplotlib
