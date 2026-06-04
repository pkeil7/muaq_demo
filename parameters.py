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


FEATURE_LABELS = {
    "mod": "Background NO2 concentrations",
    "era5_blh": "Boundary layer height",
    "era5l_t2m": "2 metre temperature",
    "era5l_sp": "Surface pressure",
    "era5l_rh": "Relative humidity",
    "era5l_ws": "Wind speed",
    "era5l_wd": "Wind direction",
}

PRESSURE_FEATURES = {"era5l_sp"}
TEMPERATURE_FEATURES = {"era5l_t2m"}
WIND_SPEED_FEATURES = {"era5l_ws"}

WEATHER_OFFSET_UNITS = {
    "era5_blh": "m",
    "era5l_t2m": "C",
    "era5l_sp": "hPa",
    "era5l_rh": "%",
    "era5l_ws": "m/s",
    "era5l_wd": "degrees",
}

