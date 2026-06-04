"""Streamlit app scaffold for interactive XGBoost what-if exploration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


from whatif_service import ScenarioRequest, XGBWhatIfService
from parameters import FEATURE_LABELS, PRESSURE_FEATURES, TEMPERATURE_FEATURES, WEATHER_OFFSET_UNITS


@st.cache_resource(show_spinner=False)
def load_service(config_path: str) -> XGBWhatIfService:
    return XGBWhatIfService.from_config(config_path)


def _secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if value:
        return str(value)
    return str(os.getenv(name, default))


def _load_runtime_payload(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Runtime config JSON must contain a top-level object.")
    return payload


def _download_file(url: str, destination: Path, auth_header: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url)
    if auth_header:
        if ":" not in auth_header:
            raise ValueError("GRID_DATA_AUTH_HEADER must look like 'Header-Name: value'.")
        header_name, header_value = auth_header.split(":", 1)
        request.add_header(header_name.strip(), header_value.strip())

    with urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_grid_data_available(config_path: str) -> None:
    payload = _load_runtime_payload(config_path)
    raw_grid_path = str(payload.get("default_grid_data_path", "")).strip()
    if not raw_grid_path:
        return

    grid_path = Path(raw_grid_path)
    if not grid_path.is_absolute():
        grid_path = (Path.cwd() / grid_path).resolve()

    if grid_path.exists():
        return

    download_url = _secret_or_env("GRID_DATA_URL")
    auth_header = _secret_or_env("GRID_DATA_AUTH_HEADER")
    if not download_url:
        raise FileNotFoundError(
            f"Grid data not found at {grid_path}. Set GRID_DATA_URL in Streamlit secrets or environment."
        )

    with st.spinner(f"Downloading NetCDF to {grid_path} ..."):
        _download_file(download_url, grid_path, auth_header=auth_header)


def feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def time_options(service: XGBWhatIfService) -> list[tuple[str, int]]:
    ds = service.predictor.ds
    n_steps = service.predictor.num_timesteps()
    if "time" not in ds.coords:
        return [(f"step {idx}", idx) for idx in range(n_steps)]

    values = ds["time"].values
    options: list[tuple[str, int]] = []
    for idx, value in enumerate(values):
        if np.issubdtype(values.dtype, np.datetime64):
            label = np.datetime_as_string(value, unit="m")
        else:
            label = str(value)
        options.append((label, idx))
    return options


def selected_hour(service: XGBWhatIfService, time_index: int) -> int:
    ds = service.predictor.ds
    if "time" not in ds.coords:
        return 12
    values = ds["time"].values
    if not np.issubdtype(values.dtype, np.datetime64):
        return 12
    selected = values[int(time_index)]
    return int(selected.astype("datetime64[h]").astype(int) % 24)


def weather_offset_bounds(service: XGBWhatIfService, feature_name: str) -> tuple[float, float, str]:
    configured = service.weather_offset_bounds(feature_name)
    if configured is not None:
        lo, hi = configured
        unit = WEATHER_OFFSET_UNITS.get(feature_name, "")
        return float(lo), float(hi), unit

    arr = service.predictor.ds[feature_name]
    vmin = float(arr.min(skipna=True).item())
    vmax = float(arr.max(skipna=True).item())

    if feature_name in PRESSURE_FEATURES:
        vmin /= 100.0
        vmax /= 100.0
    elif feature_name in TEMPERATURE_FEATURES:
        vmin -= 273.15
        vmax -= 273.15

    bound = float(max(abs(np.rint(vmin)), abs(np.rint(vmax))))
    unit = WEATHER_OFFSET_UNITS.get(feature_name, "")
    return -bound, bound, unit


def _to_display_units(feature_name: str, values: np.ndarray) -> np.ndarray:
    if feature_name in PRESSURE_FEATURES:
        return values / 100.0
    if feature_name in TEMPERATURE_FEATURES:
        return values - 273.15
    return values


def weather_feature_maps(
    service: XGBWhatIfService,
    request: ScenarioRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None:
    if request.weather_feature is None or request.weather_feature == "none":
        return None

    feature_name = request.weather_feature
    arr = service.predictor.ds[feature_name]
    if "time" in arr.dims:
        baseline_raw = arr.isel(time=int(request.time_index)).values
    else:
        baseline_raw = arr.values

    baseline_raw = np.nan_to_num(baseline_raw, nan=0.0).astype(np.float32, copy=False)
    internal_offset = service.to_internal_weather_offset(feature_name, request.weather_offset)
    scenario_raw = (baseline_raw * float(request.weather_scale)) + float(internal_offset)

    baseline_display = _to_display_units(feature_name, baseline_raw)
    scenario_display = _to_display_units(feature_name, scenario_raw)
    difference = scenario_display - baseline_display

    label = feature_label(feature_name)
    unit = WEATHER_OFFSET_UNITS.get(feature_name, "")
    return baseline_display, scenario_display, difference, label, unit


def draw_maps(
    baseline: np.ndarray,
    scenario: np.ndarray,
    difference: np.ndarray,
    weather_maps: tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None = None,
) -> plt.Figure:
    nrows = 2 if weather_maps is not None else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    vmin = float(min(np.nanmin(baseline), np.nanmin(scenario)))
    vmax = float(max(np.nanmax(baseline), np.nanmax(scenario)))
    dmax = float(np.nanmax(np.abs(difference)))
    if dmax == 0.0:
        dmax = 1.0

    im0 = axes[0, 0].imshow(baseline, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 0].set_title("NO2 Baseline")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(scenario, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 1].set_title("NO2 Scenario")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(difference, origin="lower", vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    axes[0, 2].set_title("Difference (scenario - baseline)")
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    cbar2.set_label(r"$NO_2 / \mu g m^{-3}$")

    if weather_maps is not None:
        weather_baseline, weather_scenario, weather_difference, weather_label, weather_unit = weather_maps

        w_vmin = float(min(np.nanmin(weather_baseline), np.nanmin(weather_scenario)))
        w_vmax = float(max(np.nanmax(weather_baseline), np.nanmax(weather_scenario)))
        w_dmax = float(np.nanmax(np.abs(weather_difference)))
        if w_dmax == 0.0:
            w_dmax = 1.0

        im3 = axes[1, 0].imshow(weather_baseline, origin="lower", vmin=w_vmin, vmax=w_vmax, cmap="viridis")
        axes[1, 0].set_title(f"{weather_label} Baseline")
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        cbar3 = fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        if weather_unit:
            cbar3.set_label(weather_unit)

        im4 = axes[1, 1].imshow(weather_scenario, origin="lower", vmin=w_vmin, vmax=w_vmax, cmap="viridis")
        axes[1, 1].set_title(f"{weather_label} Scenario")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        cbar4 = fig.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        if weather_unit:
            cbar4.set_label(weather_unit)

        im5 = axes[1, 2].imshow(weather_difference, origin="lower", vmin=-w_dmax, vmax=w_dmax, cmap="RdBu_r")
        axes[1, 2].set_title("Difference (scenario - baseline)")
        axes[1, 2].invert_yaxis()
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        cbar5 = fig.colorbar(im5, ax=axes[1, 2], shrink=0.8)
        if weather_unit:
            cbar5.set_label(weather_unit)

    return fig


def main() -> None:
    st.set_page_config(page_title="XGBoost What-If Explorer", layout="wide")
    st.title("Interactive XGBoost What-If Explorer")

    default_cfg = str(Path(__file__).with_name("whatif_runtime_config.json"))
    config_path = st.sidebar.text_input("Runtime config path", value=default_cfg)

    try:
        ensure_grid_data_available(config_path)
    except Exception as exc:
        st.error(f"Failed during startup data check/download: {exc}")
        st.stop()

    try:
        service = load_service(config_path)
    except Exception as exc:
        st.error(f"Failed to load service from config: {exc}")
        st.stop()

    options = time_options(service)
    label_to_idx = {label: idx for label, idx in options}
    st.sidebar.header("Scenario Controls")
    selected_time_label = st.sidebar.select_slider(
        "date/time",
        options=[label for label, _ in options],
        value=options[0][0],
    )
    time_index = int(label_to_idx[selected_time_label])

    auto_hour = selected_hour(service, time_index)
    scenario_hour = st.sidebar.slider("scenario hour", min_value=0, max_value=23, value=auto_hour)

    cfg = service.config
    mod_offset = st.sidebar.slider(
        "Background NO2 offset",
        min_value=float(cfg.mod_offset_min),
        max_value=float(cfg.mod_offset_max),
        value=0.0,
        step=0.5,
    )
    mod_scale = st.sidebar.slider(
        "Background NO2 scale",
        min_value=float(cfg.mod_scale_min),
        max_value=float(cfg.mod_scale_max),
        value=1.0,
        step=0.01,
    )

    weather_features = service.available_weather_features()
    weather_choices = ["none"] + weather_features
    selected_weather = st.sidebar.selectbox(
        "weather variable",
        options=weather_choices,
        format_func=lambda x: "none" if x == "none" else feature_label(x),
    )

    if selected_weather == "none":
        weather_offset = 0.0
        weather_scale = 1.0
        weather_unit = ""
    else:
        lo, hi, weather_unit = weather_offset_bounds(service, selected_weather)
        weather_offset = st.sidebar.slider(
            f"weather offset ({weather_unit})" if weather_unit else "weather offset",
            min_value=float(lo),
            max_value=float(hi),
            value=0.0,
            step=1.0,
        )
        weather_scale = st.sidebar.slider(
            "weather scale",
            min_value=float(cfg.weather_scale_min),
            max_value=float(cfg.weather_scale_max),
            value=1.0,
            step=0.01,
        )

    request = ScenarioRequest(
        time_index=time_index,
        hour_override=int(scenario_hour),
        mod_offset=float(mod_offset),
        mod_scale=float(mod_scale),
        weather_feature=None if selected_weather == "none" else selected_weather,
        weather_offset=float(weather_offset),
        weather_scale=float(weather_scale),
    )

    baseline, scenario, difference = service.run_scenario(request)
    weather_maps = weather_feature_maps(service, request)

    fig = draw_maps(baseline, scenario, difference, weather_maps=weather_maps)
    st.pyplot(fig, clear_figure=True)

    diagnostics = {
        "min": float(np.nanmin(difference)),
        "mean": float(np.nanmean(difference)),
        "max": float(np.nanmax(difference)),
        "p05": float(np.nanpercentile(difference, 5)),
        "p95": float(np.nanpercentile(difference, 95)),
    }
    st.markdown(
        """
        <div style="max-width: 290px; margin-top: -6px; padding: 6px 10px; border: 1px solid #d8dbe2; border-radius: 8px; font-size: 0.78rem; line-height: 1.25;">
            <strong>Difference diagnostics (scenario - baseline)</strong><br>
            min: {min:.3f}<br>
            mean: {mean:.3f}<br>
            max: {max:.3f}<br>
            p05: {p05:.3f}<br>
            p95: {p95:.3f}
        </div>
        """.format(**diagnostics),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="position: fixed; right: 16px; bottom: 16px; z-index: 999; max-width: 290px; padding: 6px 10px; border: 1px solid #d8dbe2; border-radius: 8px; font-size: 0.78rem; line-height: 1.25; background: white;">
            <strong>github.com/pkeil7/muaq_demo</strong><br>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
