"""Simple interactive what-if inference for gridded XGBoost NO2 predictions.

v1 scope:
- target variable is fixed to obs
- ctm_residual logic is intentionally ignored
- prediction uses float32 inputs for XGBoost
- optional cache in float16 to reduce memory pressure
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr
import xgboost as xgb

# Hardcoded features. Needs to be adapted if this changes
FEATURE_MAPPING = {
    0: "mod",
    1: "unixtime",
    2: "hour",
    3: "daytype",
    4: "era5_blh",
    5: "era5l_t2m",
    6: "era5l_sp",
    7: "era5l_rh",
    8: "era5l_ws",
    9: "era5l_wd",
    10: "ghsl_urban_core",
    11: "ghsl_population",
    12: "ghsl_building_height",
    13: "ghsl_building_volume",
    14: "ghsl_built_surface",
    15: "clc_category",
    16: "clc_urban_distance",
    17: "clc_industry_distance",
    18: "clc_agriculture_distance",
    19: "clc_vegetation_distance",
    20: "clc_water_distance",
    21: "osm_intersections",
    22: "osm_major_distance",
    23: "osm_residential_length",
    24: "osm_residential_distance",
}

DEFAULT_WEATHER_FEATURES = (
    "era5_blh",
    "era5l_t2m",
    "era5l_sp",
    "era5l_rh",
    "era5l_ws",
    "era5l_wd",
)

@dataclass
class WhatIfOverrides:
    """Overrides for one what-if prediction run."""

    mod_offset: float = 0.0
    mod_scale: float = 1.0
    hour_override: Optional[int] = None
    weather_offset: Dict[str, float] = field(default_factory=dict)
    weather_scale: Dict[str, float] = field(default_factory=dict)


class XGBWhatIfPredictor:
    """XGBoost grid predictor with lightweight what-if controls."""

    def __init__(
        self,
        model_path: str,
        grid_data_path: str,
        cache_dtype: np.dtype = np.float16,
    ) -> None:
        """Load the model and grid dataset, then initialize feature metadata and cache."""
        self.model_path = model_path
        self.grid_data_path = grid_data_path
        self.cache_dtype = cache_dtype

        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        self.ds = xr.open_dataset(grid_data_path)
        self.ny = int(self.ds.sizes["y"])
        self.nx = int(self.ds.sizes["x"])

        self.selected_features = list(FEATURE_MAPPING.values())
        self.feature_index = list(FEATURE_MAPPING.keys())
        # check that all selected features are present in the dataset
        missing = [feat for feat in self.selected_features if feat not in self.ds.data_vars]
        if missing:
            raise ValueError(f"Selected features missing from dataset: {missing}")

        # Cache for baseline feature matrices by timestep, keyed by time index. Values are [n_points, n_features] arrays.
        self._matrix_cache: Dict[int, np.ndarray] = {}

    def _feature_array(self, feature: str, time_index: int) -> np.ndarray:
        """Extract one feature grid for a timestep and convert NaNs to float32 zeros."""
        arr = self.ds[feature]
        if "time" in arr.dims:
            values = arr.isel(time=time_index).values
        else:
            values = arr.values
        return np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)

    def build_baseline_matrix(self, time_index: int) -> np.ndarray:
        """Build flattened feature matrix for one timestep."""
        columns = [self._feature_array(feat, time_index).ravel() for feat in self.selected_features]
        matrix = np.stack(columns, axis=1)
        if self.cache_dtype is not None:
            return matrix.astype(self.cache_dtype, copy=False)
        return matrix

    def get_cached_baseline_matrix(self, time_index: int) -> np.ndarray:
        """Return cached baseline features for a timestep, building once on first access."""
        if time_index not in self._matrix_cache:
            self._matrix_cache[time_index] = self.build_baseline_matrix(time_index)
        return self._matrix_cache[time_index]

    def _apply_overrides(self, X: np.ndarray, overrides: WhatIfOverrides) -> np.ndarray:
        """Apply scenario overrides in-place to a feature matrix copy before prediction."""
        if "mod" in self.feature_index:
            idx = self.feature_index["mod"]
            X[:, idx] = (X[:, idx] * float(overrides.mod_scale)) + float(overrides.mod_offset)

        if overrides.hour_override is not None and "hour" in self.feature_index:
            hour_idx = self.feature_index["hour"]
            current_hour = float(np.nanmedian(X[:, hour_idx]))
            X[:, hour_idx] = float(overrides.hour_override)

            # Keep unixtime roughly aligned with hour shifts if that feature exists.
            if "unixtime" in self.feature_index:
                ut_idx = self.feature_index["unixtime"]
                delta_h = float(overrides.hour_override) - current_hour
                X[:, ut_idx] = X[:, ut_idx] + (delta_h * 3600.0)

        for feat, offset in overrides.weather_offset.items():
            if feat in self.feature_index:
                X[:, self.feature_index[feat]] = X[:, self.feature_index[feat]] + float(offset)

        for feat, scale in overrides.weather_scale.items():
            if feat in self.feature_index:
                X[:, self.feature_index[feat]] = X[:, self.feature_index[feat]] * float(scale)

        return X

    def predict_timestep(self, time_index: int, overrides: Optional[WhatIfOverrides] = None) -> np.ndarray:
        """Predict one timestep for the full grid, returning [y, x]."""
        base = self.get_cached_baseline_matrix(time_index)
        X = base.astype(np.float32, copy=True)

        if overrides is not None:
            X = self._apply_overrides(X, overrides)

        pred = self.model.predict(X)
        return pred.reshape(self.ny, self.nx)

    def predict_baseline_and_scenario(
        self,
        time_index: int,
        overrides: Optional[WhatIfOverrides] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute baseline and scenario predictions for the same timestep."""
        baseline = self.predict_timestep(time_index=time_index, overrides=None)
        scenario = self.predict_timestep(time_index=time_index, overrides=overrides)
        return baseline, scenario

    def available_weather_features(self) -> List[str]:
        """List weather features available in both defaults and the active model inputs."""
        return [name for name in DEFAULT_WEATHER_FEATURES if name in self.feature_index]

    def clear_cache(self) -> None:
        """Clear cached timestep feature matrices to free memory."""
        self._matrix_cache.clear()

    def num_timesteps(self) -> int:
        """Return the number of timesteps in the loaded dataset."""
        return int(self.ds.sizes.get("time", 1))

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x and y coordinate vectors for plotting or mapping outputs."""
        return self.ds["x"].values, self.ds["y"].values


__all__ = [
    "XGBWhatIfPredictor",
    "WhatIfOverrides",
    "DEFAULT_WEATHER_FEATURES",
]
