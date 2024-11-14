"""Built-in dimensions."""

__all__: list[str] = []

from unxt._src.dimensions.core import dimensions

dimensionless = dimensions("dimensionless")
length = dimensions("length")
mass = dimensions("mass")
time = dimensions("time")
speed = dimensions("speed")
angle = dimensions("angle")
amount = dimensions("amount of substance")
current = dimensions("electrical current")
temperature = dimensions("temperature")
luminous_intensity = dimensions("luminous intensity")
pressure = dimensions("pressure")
force = dimensions("force")
energy = dimensions("energy")
dynamic_viscosity = dimensions("dynamic viscosity")
kinematic_viscosity = dimensions("kinematic viscosity")
