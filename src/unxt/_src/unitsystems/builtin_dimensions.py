"""Built-in dimensions."""

__all__: tuple[str, ...] = ()

from unxt._src.dimensions import dimension

dimensionless = dimension("dimensionless")
length = dimension("length")
mass = dimension("mass")
time = dimension("time")
speed = dimension("speed")
angle = dimension("angle")
amount = dimension("amount of substance")
current = dimension("electrical current")
temperature = dimension("temperature")
luminous_intensity = dimension("luminous intensity")
pressure = dimension("pressure")
force = dimension("force")
energy = dimension("energy")
dynamic_viscosity = dimension("dynamic viscosity")
kinematic_viscosity = dimension("kinematic viscosity")
