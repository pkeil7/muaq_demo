"""Minimal notebook-friendly UI for obs-only XGBoost what-if predictions.

Usage in a notebook cell:

import sys
sys.path.append("..")
from interactive_demo.xgb_whatif_demo import launch_simple_whatif

launch_simple_whatif(
    model_path="../models/your_xgb_model.json",
    grid_data_path="/path/to/grid_data.nc",
)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


from whatif_config import load_runtime_config
from whatif_service import ScenarioRequest, XGBWhatIfService
from xgb_whatif import XGBWhatIfPredictor
from parameters import FEATURE_LABELS, WEATHER_OFFSET_UNITS, PRESSURE_FEATURES, TEMPERATURE_FEATURES


def feature_label(feature_name: str) -> str:
    """Return a user-friendly label for a model feature key."""
    return FEATURE_LABELS.get(feature_name, feature_name)


def build_time_options(predictor: XGBWhatIfPredictor, n_steps: int) -> list[tuple[str, int]]:
    """Build selection options as (display label, timestep index)."""
    if "time" not in predictor.ds.coords:
        return [(f"step {idx}", idx) for idx in range(n_steps)]

    time_values = predictor.ds["time"].values
    options: list[tuple[str, int]] = []
    for idx, value in enumerate(time_values):
        if np.issubdtype(time_values.dtype, np.datetime64):
            label = np.datetime_as_string(value, unit="m")
        else:
            label = str(value)
        options.append((label, idx))
    return options


def build_weather_offset_ranges(
    predictor: XGBWhatIfPredictor,
    weather_features: list[str],
) -> dict[str, dict[str, float | str]]:
    """Build offset slider ranges from full-domain min/max values per weather feature."""
    ranges: dict[str, dict[str, float | str]] = {}
    for feature in weather_features:
        arr = predictor.ds[feature]
        vmin = float(arr.min(skipna=True).item())
        vmax = float(arr.max(skipna=True).item())
        unit = WEATHER_OFFSET_UNITS.get(feature, "")

        if feature in PRESSURE_FEATURES:
            # Display pressure offsets in hPa but keep Pa in the model override.
            display_min = float(np.rint(vmin / 100.0))
            display_max = float(np.rint(vmax / 100.0))
            scale_to_internal = 100.0
        elif feature in TEMPERATURE_FEATURES:
            # Display temperature in Celsius while retaining additive behavior internally.
            display_min = float(np.rint(vmin - 273.15))
            display_max = float(np.rint(vmax - 273.15))
            scale_to_internal = 1.0
        else:
            display_min = float(np.rint(vmin))
            display_max = float(np.rint(vmax))
            scale_to_internal = 1.0

        if display_max < display_min:
            display_min, display_max = display_max, display_min

        symmetric_bound = float(max(abs(display_min), abs(display_max)))
        display_min = -symmetric_bound
        display_max = symmetric_bound

        ranges[feature] = {
            "min": display_min,
            "max": display_max,
            "step": 1.0,
            "readout_format": ".0f",
            "description": f"weather offset ({unit})" if unit else "weather offset",
            "unit": unit,
            "scale_to_internal": scale_to_internal,
        }
    return ranges


def launch_simple_whatif(
    model_path: str | None = None,
    grid_data_path: str | None = None,
    runtime_config_path: str | None = None,
) -> XGBWhatIfPredictor:
    """Launch a minimal interactive what-if explorer in Jupyter."""
    runtime_config = load_runtime_config(runtime_config_path)

    resolved_model_path = model_path or runtime_config.default_model_path
    resolved_grid_data_path = grid_data_path or runtime_config.default_grid_data_path
    if not resolved_model_path or not resolved_grid_data_path:
        raise ValueError(
            "model_path and grid_data_path must be provided, either directly or via runtime config."
        )

    predictor = XGBWhatIfPredictor(
        model_path=resolved_model_path,
        grid_data_path=resolved_grid_data_path,
        cache_dtype=np.float16,
    )
    service = XGBWhatIfService(predictor=predictor, config=runtime_config)

    n_steps = max(1, predictor.num_timesteps())
    weather_features = service.available_weather_features()
    weather_options = [(feature_label(name), name) for name in weather_features]
    weather_offset_ranges = build_weather_offset_ranges(predictor, weather_features)

    for feature_name in weather_features:
        configured_bounds = service.weather_offset_bounds(feature_name)
        if configured_bounds is not None:
            weather_offset_ranges[feature_name]["min"] = float(configured_bounds[0])
            weather_offset_ranges[feature_name]["max"] = float(configured_bounds[1])

        weather_offset_ranges[feature_name]["scale_to_internal"] = float(
            runtime_config.weather_offset_internal_scales.get(
                feature_name,
                weather_offset_ranges[feature_name]["scale_to_internal"],
            )
        )

    time_options = build_time_options(predictor, n_steps)
    label_style = {"description_width": "170px"}

    time_selector = widgets.SelectionSlider(
        description="date/time",
        options=time_options,
        value=0,
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
        style=label_style,
    )
    hour_slider = widgets.IntSlider(
        description="scenario hour",
        min=0,
        max=23,
        step=1,
        value=12,
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
        style=label_style,
    )
    mod_offset = widgets.FloatSlider(
        description="BG NO2 offset",
        min=float(runtime_config.mod_offset_min),
        max=float(runtime_config.mod_offset_max),
        step=0.5,
        value=0.0,
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px", min_width="300px"),
        style=label_style,
    )
    mod_scale = widgets.FloatSlider(
        description="BG NO2 scale",
        min=float(runtime_config.mod_scale_min),
        max=float(runtime_config.mod_scale_max),
        step=0.01,
        value=1.0,
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="400px", min_width="300px"),
        style=label_style,
    )
    weather_var = widgets.Dropdown(
        description="ERA5",
        options=[("none", "none")] + weather_options,
        value="none",
        layout=widgets.Layout(width="360px"),
        style=label_style,
    )
    weather_offset = widgets.FloatSlider(
        description="weather offset",
        min=-20.0,
        max=20.0,
        step=0.1,
        value=0.0,
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px", min_width="300px"),
        style=label_style,
    )
    weather_scale = widgets.FloatSlider(
        description="weather scale",
        min=float(runtime_config.weather_scale_min),
        max=float(runtime_config.weather_scale_max),
        step=0.01,
        value=1.0,
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="400px", min_width="300px"),
        style=label_style,
    )
    weather_offset_unit = widgets.HTML(value="", layout=widgets.Layout(width="80px"))

    output = widgets.Output()
    suppress_render = {"active": False}

    def update_weather_offset_control(selected_weather: str) -> None:
        config = weather_offset_ranges.get(selected_weather)
        if config is None:
            suppress_render["active"] = True
            weather_offset.description = "weather offset"
            weather_offset.min = -20.0
            weather_offset.max = 20.0
            weather_offset.step = 0.5
            weather_offset.readout_format = ".1f"
            weather_offset.value = 0.0
            weather_offset_unit.value = ""
            suppress_render["active"] = False
            return

        min_val = float(config["min"])
        max_val = float(config["max"])
        default_val = 0.0 if min_val <= 0.0 <= max_val else min_val

        suppress_render["active"] = True
        weather_offset.description = str(config["description"])
        weather_offset.min = min_val
        weather_offset.max = max_val
        weather_offset.step = float(config["step"])
        weather_offset.readout_format = str(config["readout_format"])
        weather_offset.value = default_val
        unit = str(config.get("unit", ""))
        weather_offset_unit.value = f"<span style='padding-left:6px'>{unit}</span>" if unit else ""
        suppress_render["active"] = False

    def get_selected_time_hour(time_index: int) -> int | None:
        if "time" not in predictor.ds.coords:
            return None
        time_values = predictor.ds["time"].values
        if not np.issubdtype(time_values.dtype, np.datetime64):
            return None
        selected_time = time_values[int(time_index)]
        return int(selected_time.astype("datetime64[h]").astype(int) % 24)

    def sync_hour_with_selected_time(time_index: int) -> None:
        hour = get_selected_time_hour(time_index)
        if hour is None:
            return
        suppress_render["active"] = True
        hour_slider.value = hour
        suppress_render["active"] = False

    def render(*_args):
        selected_weather = weather_var.value
        request = ScenarioRequest(
            time_index=int(time_selector.value),
            hour_override=int(hour_slider.value),
            mod_offset=float(mod_offset.value),
            mod_scale=float(mod_scale.value),
            weather_feature=None if selected_weather == "none" else selected_weather,
            weather_offset=float(weather_offset.value),
            weather_scale=float(weather_scale.value),
        )
        baseline, scenario, delta = service.run_scenario(request)

        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

            vmin = float(min(np.nanmin(baseline), np.nanmin(scenario)))
            vmax = float(max(np.nanmax(baseline), np.nanmax(scenario)))
            dmax = float(np.nanmax(np.abs(delta)))
            if dmax == 0.0:
                dmax = 1.0

            im0 = axes[0].imshow(baseline, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
            axes[0].set_title("Baseline")
            axes[0].invert_yaxis()
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            fig.colorbar(im0, ax=axes[0], shrink=0.8)

            im1 = axes[1].imshow(scenario, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
            axes[1].set_title("Scenario")
            axes[1].invert_yaxis()
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(im1, ax=axes[1], shrink=0.8)

            im2 = axes[2].imshow(delta, origin="lower", vmin=-dmax, vmax=dmax, cmap="RdBu_r")
            axes[2].set_title("Delta (scenario - baseline)")
            axes[2].invert_yaxis()
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.8)
            cbar2.set_label(r"$NO_2 / \mu g m^{-3}$")

            plt.show()

    row_1 = widgets.HBox(
        [time_selector, hour_slider],
        layout=widgets.Layout(gap="30px", margin="0 0 14px 0"),
    )
    row_2 = widgets.HBox(
        [mod_offset, mod_scale],
        layout=widgets.Layout(gap="30px", margin="0 0 8px 0"),
    )
    row_3 = widgets.HBox(
        [weather_var, widgets.HBox([weather_offset, weather_offset_unit]), weather_scale],
        layout=widgets.Layout(gap="10px"),
    )

    controls = widgets.VBox([row_1, row_2, row_3])

    def on_control_change(change):
        if suppress_render["active"]:
            return
        render()

    def on_weather_change(change):
        if suppress_render["active"]:
            return
        update_weather_offset_control(str(change["new"]))
        render()

    def on_time_change(change):
        if suppress_render["active"]:
            return
        sync_hour_with_selected_time(int(change["new"]))
        render()

    time_selector.observe(on_time_change, names="value")
    weather_var.observe(on_weather_change, names="value")

    for w in (
        hour_slider,
        mod_offset,
        mod_scale,
        weather_offset,
        weather_scale,
    ):
        w.observe(on_control_change, names="value")

    display(controls, output)
    update_weather_offset_control(str(weather_var.value))
    sync_hour_with_selected_time(int(time_selector.value))
    render()
    return predictor
