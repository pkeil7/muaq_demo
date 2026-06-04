# Interactive Demo Package

This folder contains all code and config for the interactive XGBoost what-if demo.

Find the streamlit app here: `https://muaq-demo-openday.streamlit.app/`

## Contents

- `xgb_whatif.py`: core predictor and override logic
- `whatif_config.py`: runtime config schema/loader
- `whatif_service.py`: web-safe service facade for scenario execution
- `xgb_whatif_demo.py`: notebook UI launcher (ipywidgets)
- `whatif_runtime_config.json`: runtime defaults and bounds
- `app.py`: streamlit app file
- `__init__.py`: package exports

## Notebook Usage

From `notebooks/interactive_demo.ipynb` (or any notebook under `notebooks/`):

```python
import sys
sys.path.append("..")

from interactive_demo.xgb_whatif_demo import launch_simple_whatif
```

Launch with explicit paths:

```python
launch_simple_whatif(
    model_path="/path/to/model",
    grid_data_path="/path/to/grid_data.nc",
)
```

Or launch via runtime config defaults:

```python
launch_simple_whatif(runtime_config_path="../interactive_demo/whatif_runtime_config.json")
```

## Service Usage (Streamlit/Backend)

Use the service layer directly for app code:

```python
from interactive_demo.whatif_service import XGBWhatIfService, ScenarioRequest

service = XGBWhatIfService.from_paths(
    model_path="/path/to/model",
    grid_data_path="/path/to/grid_data.nc",
    config_path="interactive_demo/whatif_runtime_config.json",
)

request = ScenarioRequest(
    time_index=0,
    hour_override=12,
    mod_offset=0.0,
    mod_scale=1.0,
    weather_feature="era5l_t2m",
    weather_offset=2.0,
    weather_scale=1.0,
)

baseline, scenario, delta = service.run_scenario(request)
```

Or use config-defined default paths:

```python
service = XGBWhatIfService.from_config("interactive_demo/whatif_runtime_config.json")
```

## Streamlit App

Run on localhost:

```bash
streamlit run interactive_demo/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open from another machine on the same network:

```text
http://<YOUR-MAC-LAN-IP>:8501
```

Find your LAN IP on macOS:

```bash
ipconfig getifaddr en0
```

If you are on Ethernet, use:

```bash
ipconfig getifaddr en1
```

If others still cannot connect, verify:

- macOS Firewall allows incoming connections for Python/Streamlit
- both devices are on the same network/VLAN
- port 8501 is not blocked by local security software

## Public Hosting (Simple)

For the simplest public sharing, use Streamlit Community Cloud.

Set these fields in `interactive_demo/whatif_runtime_config.json` for deployment:

- `default_model_path`: `SpatialFullSet`
- `default_grid_data_path`: `interactive_demo/demo_subset.nc`

### Streamlit Community Cloud steps

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from your repo.
3. Set:
     - Main file path: `interactive_demo/app.py`
     - Python version: `3.10`
     - Requirements file: `interactive_demo/requirements_streamlit_cloud.txt`
4. Deploy.

After deploy, share the generated public app URL.

### Startup NetCDF Download (Python)

`interactive_demo/app.py` now checks whether `default_grid_data_path` exists locally.
If missing, it downloads the file at startup using Python (`urllib`) and writes it
to that configured local path.

Set one of these in Streamlit Cloud (App settings -> Secrets):

```toml
GRID_DATA_URL = "https://.../demo_subset_30km_1week.nc"
```

Optional auth header (for protected storage endpoints):

```toml
GRID_DATA_AUTH_HEADER = "Authorization: Bearer <token>"
```


Keep `default_grid_data_path` in `whatif_runtime_config.json` as the local
destination path where the file should be cached (for example,
`demo_data/demo_subset_30km_1week.nc`).

Before launching, set these fields in `interactive_demo/whatif_runtime_config.json`:

- `default_model_path`
- `default_grid_data_path`

## Runtime Config Notes

Edit `whatif_runtime_config.json` to set:

- `default_model_path`
- `default_grid_data_path`
- optional feature allowlist (`selectable_weather_features`)
- UI bounds (`mod_*`, `weather_scale_*`, optional `weather_offset_bounds`)
- display->internal unit conversion (`weather_offset_internal_scales`)
