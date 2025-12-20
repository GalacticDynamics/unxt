"""Generate sample NetCDF file with unit metadata for documentation examples."""

from pathlib import Path

import xarray as xr

# Create sample dataset with unit metadata
ds = xr.Dataset(
    {
        "temperature": (
            ["time", "location"],
            [[273.15, 293.15, 313.15], [275.0, 295.0, 315.0]],
            {"units": "K", "long_name": "Temperature", "standard_name": "temperature"},
        ),
        "pressure": (
            ["time", "location"],
            [[101325.0, 102000.0, 103000.0], [101500.0, 102200.0, 103200.0]],
            {"units": "Pa", "long_name": "Pressure", "standard_name": "air_pressure"},
        ),
        "distance": (
            ["location"],
            [0.0, 100.0, 200.0],
            {"units": "m", "long_name": "Distance from origin"},
        ),
    },
    coords={
        "time": (
            ["time"],
            [0.0, 3600.0],
            {"units": "s", "long_name": "Time since start"},
        ),
        "location": (
            ["location"],
            [0, 1, 2],
            {"long_name": "Measurement location index"},
        ),
    },
    attrs={
        "title": "Sample atmospheric measurements",
        "description": (
            "Example dataset with physical units for unxt-xarray documentation"
        ),
        "conventions": "CF-1.8",
    },
)

# Save to NetCDF file
output_path = Path(__file__).parent / "sample_data.nc"
ds.to_netcdf(output_path)
