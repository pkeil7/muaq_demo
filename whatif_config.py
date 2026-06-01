"""Runtime configuration for the interactive XGBoost what-if service.

This module is deployment-safe: it contains only inference-time defaults and bounds,
without any training code or training-time dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class WhatIfRuntimeConfig:
    """Minimal runtime configuration for notebook and web what-if apps."""

    default_model_path: str | None = None
    default_grid_data_path: str | None = None
    selectable_weather_features: list[str] = field(default_factory=list)

    mod_offset_min: float = -30.0
    mod_offset_max: float = 30.0
    mod_scale_min: float = 0.5
    mod_scale_max: float = 1.5
    weather_scale_min: float = 0.5
    weather_scale_max: float = 1.5

    # Optional fixed display-space bounds per weather variable.
    # Example: {"era5l_ws": [-8.0, 8.0], "era5l_sp": [-40.0, 40.0]}
    weather_offset_bounds: dict[str, list[float]] = field(default_factory=dict)

    # Scale factor to convert display offset units to internal feature units.
    # Example: pressure in hPa display but Pa internal -> 100.0
    weather_offset_internal_scales: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WhatIfRuntimeConfig":
        return cls(
            default_model_path=payload.get("default_model_path"),
            default_grid_data_path=payload.get("default_grid_data_path"),
            selectable_weather_features=list(payload.get("selectable_weather_features", [])),
            mod_offset_min=float(payload.get("mod_offset_min", -30.0)),
            mod_offset_max=float(payload.get("mod_offset_max", 30.0)),
            mod_scale_min=float(payload.get("mod_scale_min", 0.5)),
            mod_scale_max=float(payload.get("mod_scale_max", 1.5)),
            weather_scale_min=float(payload.get("weather_scale_min", 0.5)),
            weather_scale_max=float(payload.get("weather_scale_max", 1.5)),
            weather_offset_bounds=dict(payload.get("weather_offset_bounds", {})),
            weather_offset_internal_scales={
                k: float(v) for k, v in dict(payload.get("weather_offset_internal_scales", {})).items()
            },
        )

    @classmethod
    def from_json(cls, config_path: str | Path) -> "WhatIfRuntimeConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Runtime config JSON must contain a top-level object.")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_model_path": self.default_model_path,
            "default_grid_data_path": self.default_grid_data_path,
            "selectable_weather_features": self.selectable_weather_features,
            "mod_offset_min": self.mod_offset_min,
            "mod_offset_max": self.mod_offset_max,
            "mod_scale_min": self.mod_scale_min,
            "mod_scale_max": self.mod_scale_max,
            "weather_scale_min": self.weather_scale_min,
            "weather_scale_max": self.weather_scale_max,
            "weather_offset_bounds": self.weather_offset_bounds,
            "weather_offset_internal_scales": self.weather_offset_internal_scales,
        }


def load_runtime_config(config_path: str | Path | None = None) -> WhatIfRuntimeConfig:
    """Load runtime config from JSON path or return defaults when omitted."""
    if config_path is None:
        return WhatIfRuntimeConfig()
    return WhatIfRuntimeConfig.from_json(config_path)
