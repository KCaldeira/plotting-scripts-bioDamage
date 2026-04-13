# Send-to-Ken plotting bundle

Small, self-contained slice of the land-biosphere / climate-damage workflow: **configuration-driven figure generation** from a saved analysis state. It targets the main plots for the damage–bio paper (growth-rate panels, country distributions, GPP maps and related views).

## What it does

- **`main_json.py`** loads a JSON config, reads a serialized analysis object from **`main_analysis_state.joblib`**, wraps it for attribute-style access, and calls plotting routines listed under `plotting_list` in the config.
- Plotting code lives under **`plot/`**. Shared axis helpers from the vendored **`axes-main/`** package are added to `sys.path` automatically by `plot/func_shared_plotting.py`.
- **`utils/`** holds masking, NetCDF loading, regression, and projection helpers used by plots and by other entry points in the repo.

## Requirements

- **Python** 3.10+ is a practical minimum for recent `xcdat` / `xarray` stacks.
- **System libraries** for **Cartopy** (GEOS, PROJ); installing via **conda-forge** often avoids build pain:

  ```bash
  conda create -n bio_plots python=3.11
  conda activate bio_plots
  conda install -c conda-forge cartopy xcdat xarray netcdf4 scipy pandas matplotlib statsmodels joblib regionmask geopandas shapely
  ```

  Alternatively, from this directory:

  ```bash
  pip install -r requirements.txt
  ```

  If Cartopy fails to install with pip, use conda-forge for Cartopy and its dependencies.

## Data file

Place **`main_analysis_state.joblib`** in the **project root** (same directory as `main_json.py`). It must be compatible with the plotting functions (same keys/structure they expect on the loaded object). The joblib payload is wrapped in a small `SimpleNamespace`-like helper so code can use attribute access (e.g. `main_analysis.some_key`).

## Usage

Run from the repository root so imports resolve (`plot.*`, `utils.*`):

```bash
cd /path/to/send_to_ken_folder
python main_json.py run_main_workflow.json
```

### Configuration JSON

The workflow section used by `main_json.py` is **`action_Burke_1`**. Inside it, **`plotting_list`** is a dictionary of plot names to booleans. Set a key to `true` to run that figure, `false` to skip. Example keys:

| Key | Script |
|-----|--------|
| `growth_rate_global_mean_timeSeries` | Global mean time series |
| `growth_rate_country_boxplotLikeDistribution` | Country distribution (boxplot-like) |
| `growth_rate_country_barPlotDistribution` | Country bar distribution |
| `growth_rate_country_boxplotLikeSelectedCountries` | Selected countries |
| `gpp_country_map` | GPP map |
| `gpp_country_scatter` | GPP scatter |
| `gpp_country_violin` | GPP violin |

See **`run_main_workflow.json`** for a full example, including `run_metadata` and other action fields (some are for documentation or a wider pipeline; only `plotting_list` drives `main_json.py` today).

## Simple figure pipeline (`simple_json.py`)

For paper figures, you can bypass `main_json.py` and the `plotting_list` config: **`simple_json.py`** is a small entry point that builds **cached datasets** under **`simple_scripts/`** and calls the matching plot helpers. Edit `main()` in `simple_json.py` to turn individual figures on or off (several blocks are commented as templates).

### What you need

- **`main_analysis_state.joblib`** in the **project root**, same as the main workflow (`simple_scripts/read_original_data.py` loads it via `joblib.load`).
- Run from the **repository root** so paths like `./simple_scripts/` resolve.

### How it works

1. **Data scripts** read the joblib state once (or skip if a cache file already exists) and write **CSV / pickle** artifacts next to the scripts.
2. **Plot scripts** read those artifacts only (fast iteration on layout and style).

| Target | Data helper | Output cache | Plot helper(s) |
|--------|-------------|--------------|----------------|
| Figures 1–2 (template) | `simple_scripts/Fig1_2_data.py` → `get_figure1_2_data()` | `simple_scripts/figure1_2_data.csv` | `Fig1_2_plot.py` → `plot_figure1_2(df, 'main' \| 'SI')` |
| Figure 3 (country table) | `Fig3_data.py` → `get_figure3_data()` | `simple_scripts/figure3_data.csv` | `Fig3_plot_scatter.py` → `fig3_plot_scatter(df)`; **`Fig3_plot_violin.py`** → `fig3_plot_violin(df)` |
| Figure 3 (maps) | `Fig3_map_data.py` → `get_figure3_map_data()` | `simple_scripts/figure3_map_data.pickle` | `Fig3_plot_map.py` → `fig3_plot_map(dict)` |

**Figure 3 CSV columns** (per country and model) include `region_list`, `model_name`, `area`, `model_log10ratio`, Burke empirical columns (`burke_growth_log10ratio`, `burke_level_log10ratio`), and Newell medians (`newell_growth_log10ratio`, `newell_level_log10ratio`) for 2080–2100 mean GPP ratios relative to BGC. The violin script combines **model**, **Burke central (level)**, and **Newell median (level)** in one panel per climate model.

**Figure 3 maps** require gridded land masks: `Fig3_map_data.py` expects **`land_ocean_areacella_ds_<model>.nc`** files under `simple_scripts/` (see that module for naming).

### Usage

```bash
cd /path/to/send_to_ken_folder
python simple_json.py
```

The default `main()` in the repo may point at a single figure (e.g. Figure 3 violin); uncomment other imports and calls to regenerate data or produce scatter, map, or Fig 1–2 outputs.

## Repository layout

| Path | Role |
|------|------|
| `main_json.py` | CLI: config + joblib load + plot dispatch |
| `simple_json.py` | Optional: cached `simple_scripts/` data + lightweight Fig 1–3 plot entry |
| `run_main_workflow.json` | Example config |
| `simple_scripts/` | Figure-specific data builders, caches (`figure3_data.csv`, `figure3_map_data.pickle`, …), and `read_original_data.py` |
| `plot/` | Figure scripts and `func_shared_plotting.py` |
| `axes-main/` | Axis tick / scale helpers used by plots |
| `utils/` | Data I/O, masks, regression, projections |
| `action_mainAnalysis.py` | Alternate entry that wires additional plots (e.g. Burke time decay, spatial stats) |

## Notes

- **Joblib vs pickle:** This bundle loads **`main_analysis_state.joblib`** with `joblib.load`. If you only have an older **pickle** (`.pkl`), you may need to regenerate the joblib file from the upstream analysis code, or adjust `run_pipeline` to match your serialization format.
- **Paths inside the saved state:** Plots may still expect absolute paths to NetCDF or mask files that existed on the machine that produced the joblib; if figures fail with missing files, check those paths in the loaded object or in `utils/func_shared.py` helpers.
