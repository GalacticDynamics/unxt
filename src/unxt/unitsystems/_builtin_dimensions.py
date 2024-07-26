"""Built-in dimensions."""

__all__: list[str] = []

import astropy.units as u

dimensionless = u.get_physical_type("dimensionless")
length = u.get_physical_type("length")
mass = u.get_physical_type("mass")
time = u.get_physical_type("time")
speed = u.get_physical_type("speed")
angle = u.get_physical_type("angle")
