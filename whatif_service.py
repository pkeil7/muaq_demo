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

try:
    from .whatif_config import WhatIfRuntimeConfig, load_runtime_config
    from .xgb_whatif import WhatIfOverrides, XGBWhatIfPredictor
except ImportError:
    from whatif_config import WhatIfRuntimeConfig, load_runtime_config
    from xgb_whatif import WhatIfOverrides, XGBWhatIfPredictor


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
        scale = float(self.config.weather_offset_internal_scales.get(feature_name, 1.0))
        return float(display_offset) * scale

    def run_scenario(self, request: ScenarioRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weather_offset = {}
        weather_scale = {}

        if request.weather_feature is not None and request.weather_feature != "none":
            weather_offset[request.weather_feature] = self.to_internal_weather_offset(
                request.weather_feature,
                request.weather_offset,
            )
            weather_scale[request.weather_feature] = float(request.weather_scale)

        overrides = WhatIfOverrides(
            mod_offset=float(request.mod_offset),
            mod_scale=float(request.mod_scale),
            hour_override=request.hour_override,
            weather_offset=weather_offset,
            weather_scale=weather_scale,
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
