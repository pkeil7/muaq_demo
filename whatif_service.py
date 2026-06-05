"""Web-safe inference service layer for XGBoost what-if exploration.

This wrapper keeps serving concerns in one importable module:
- loading runtime config
- building predictor instances
- applying scenario overrides
- optional filtering of selectable weather features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parameters import PRESSURE_FEATURES
from whatif_config import WhatIfRuntimeConfig, load_runtime_config
from xgb_whatif import WhatIfOverrides, XGBWhatIfPredictor


OFFSET_MIN = -180.0
OFFSET_MAX = 180.0


@dataclass
class ScenarioRequest:
    """Single what-if scenario request in display-space units."""

    time_index: int
    hour_override: int | None = None
    mod_offset: float = 0.0
    mod_scale: float = 1.0
    weather_feature: str | None = None
    weather_offset: float = 0.0
    weather_scale: float = 1.0
    industry_distance: float | None = None
    city_centre_landcover: int | None = None


class XGBWhatIfService:
    """Thin service facade for web/notebook what-if prediction calls."""

    def __init__(self, predictor: XGBWhatIfPredictor, config: WhatIfRuntimeConfig | None = None) -> None:
        self.predictor = predictor
        self.config = config or WhatIfRuntimeConfig()

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        grid_data_path: str,
        config_path: str | Path | None = None,
        cache_dtype: np.dtype = np.float16,
    ) -> "XGBWhatIfService":
        config = load_runtime_config(config_path)
        predictor = XGBWhatIfPredictor(
            model_path=model_path,
            grid_data_path=grid_data_path,
            cache_dtype=cache_dtype,
        )
        return cls(predictor=predictor, config=config)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        cache_dtype: np.dtype = np.float16,
    ) -> "XGBWhatIfService":
        config = load_runtime_config(config_path)
        if not config.default_model_path or not config.default_grid_data_path:
            raise ValueError(
                "Runtime config must define default_model_path and default_grid_data_path "
                "when using from_config()."
            )
        predictor = XGBWhatIfPredictor(
            model_path=config.default_model_path,
            grid_data_path=config.default_grid_data_path,
            cache_dtype=cache_dtype,
        )
        return cls(predictor=predictor, config=config)

    def available_weather_features(self) -> list[str]:
        available = self.predictor.available_weather_features()
        if not self.config.selectable_weather_features:
            return available
        allowed = set(self.config.selectable_weather_features)
        return [name for name in available if name in allowed]

    def weather_offset_bounds(self, feature_name: str) -> tuple[float, float] | None:
        bounds = self.config.weather_offset_bounds.get(feature_name)
        if not bounds or len(bounds) != 2:
            return None
        lo = float(bounds[0])
        hi = float(bounds[1])
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def to_internal_weather_offset(self, feature_name: str, display_offset: float) -> float:
        default_scale = 100.0 if feature_name in PRESSURE_FEATURES else 1.0
        scale = float(self.config.weather_offset_internal_scales.get(feature_name, default_scale))
        return float(display_offset) * scale

    def city_centre_box_indices(self) -> tuple[int, int, int, int]:
        ny = int(self.predictor.ny)
        nx = int(self.predictor.nx)

        configured = self.config.city_centre_box_indices
        if configured is not None and len(configured) == 4:
            y0, y1, x0, x1 = [int(v) for v in configured]
        else:
            # Fallback: central quarter of the grid.
            y0, y1 = ny // 4, (3 * ny) // 4
            x0, x1 = nx // 4, (3 * nx) // 4

        y0 = max(0, min(y0, ny))
        y1 = max(y0, min(y1, ny))
        x0 = max(0, min(x0, nx))
        x1 = max(x0, min(x1, nx))
        return y0, y1, x0, x1

    def run_scenario(self, request: ScenarioRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weather_offset = {}
        weather_scale = {}
        feature_set = {}
        box_feature_set = {}
        box_indices = None

        if request.weather_feature is not None and request.weather_feature != "none":
            clamped_offset = float(np.clip(float(request.weather_offset), OFFSET_MIN, OFFSET_MAX))
            weather_offset[request.weather_feature] = self.to_internal_weather_offset(
                request.weather_feature,
                clamped_offset,
            )
            if self.config.weather_scale_enabled:
                weather_scale[request.weather_feature] = float(request.weather_scale)

        if request.industry_distance is not None:
            feature_set["clc_industry_distance"] = float(request.industry_distance)

        if request.city_centre_landcover is not None:
            box_indices = self.city_centre_box_indices()
            landcover_value = int(request.city_centre_landcover)
            box_feature_set["clc_category"] = float(landcover_value)

            if landcover_value == 3:
                box_feature_set["clc_vegetation_distance"] = 0.0
            elif landcover_value == 4:
                box_feature_set["clc_agriculture_distance"] = 0.0
            elif landcover_value == 5:
                box_feature_set["clc_water_distance"] = 0.0

            for feat in self.predictor.selected_features:
                if feat.startswith("ghsl_") and feat != "ghsl_urban_core":
                    box_feature_set[feat] = 0.0

        overrides = WhatIfOverrides(
            mod_offset=float(request.mod_offset),
            mod_scale=float(request.mod_scale),
            hour_override=request.hour_override,
            weather_offset=weather_offset,
            weather_scale=weather_scale,
            feature_set=feature_set,
            box_indices=box_indices,
            box_feature_set=box_feature_set,
        )

        baseline, scenario = self.predictor.predict_baseline_and_scenario(
            time_index=int(request.time_index),
            overrides=overrides,
        )
        delta = scenario - baseline
        return baseline, scenario, delta


__all__ = [
    "ScenarioRequest",
    "XGBWhatIfService",
]
