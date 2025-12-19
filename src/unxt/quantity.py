"""Quantities in JAX.

This module provides JAX-compatible quantity types with automatic unit handling,
conversions, and a rich set of mathematical operations that preserve dimensional
correctness.

## Core Classes

- **`Quantity`**: The main quantity class with dimension parametrization and
  full unit checking. Aliased as `Q` for convenience.
- **`BareQuantity`**: Lightweight quantity without dimension parametrization.
- **`Angle`**: Specialized quantity type for angular measurements with wrapping
  support.

## Key Functions

- **`uconvert`**: Convert a quantity to different units.
- **`ustrip`**: Extract the numerical value from a quantity.
- **`is_unit_convertible`**: Check if two units are convertible to each other.
- **`wrap_to`**: Wrap angular quantities to a specified range.

## Examples

### Creating quantities with units:

>>> import unxt as u
>>> import jax
>>> import jax.numpy as jnp

>>> distance = u.Quantity(100, "m")
>>> distance
Quantity(Array(100, dtype=int32, ...), unit='m')

Create from arrays

>>> velocities = u.Quantity([10, 20, 30], "m/s")
>>> velocities
Quantity(Array([10, 20, 30], dtype=int32), unit='m / s')

Unit conversions and arithmetic:

Convert units

>>> distance_km = u.uconvert("km", distance)
>>> distance_km
Quantity(Array(0.1, dtype=float32, ...), unit='km')

Arithmetic preserves units.

>>> time = u.Q(5, "s")  # use Quantity alias
>>> velocity = distance / time
>>> velocity
Quantity(Array(20., dtype=float32, ...), unit='m / s')

Strip units for numerical operations

>>> u.ustrip("m", distance)
Array(100, dtype=int32, ...)


### Working with angles:

Create angle quantities

>>> theta = u.Angle(180, "deg")
>>> theta
Angle(Array(180, dtype=int32, ...), unit='deg')

Convert to radians

>>> theta_rad = u.uconvert("rad", theta)
>>> theta_rad
Angle(Array(3.1415927, dtype=float32, ...), unit='rad')

Wrap angles to a range

>>> angle = u.Angle(450, "deg")
>>> wrapped = u.quantity.wrap_to(angle, u.Angle(0, "deg"), u.Angle(360, "deg"))
>>> wrapped
Angle(Array(90, dtype=int32, ...), unit='deg')


### Advanced usage with JAX transformations:

Quantities work with JAX transformations

>>> def kinetic_energy(mass, velocity):
...     return 0.5 * mass * velocity**2

>>> mass = u.Q(2.0, "kg")
>>> vel = u.Q(10.0, "m/s")
>>> energy = kinetic_energy(mass, vel)
>>> energy
Quantity(Array(100., dtype=float32, ...), unit='m2 kg / s2')

Convert to standard energy units

>>> u.uconvert("J", energy)
Quantity(Array(100., dtype=float32, ...), unit='m2 kg / s2')

JIT compilation works seamlessly

>>> @jax.jit
... def compute_force(mass, accel):
...     return mass * accel
>>> force = compute_force(u.Q(5.0, "kg"), u.Q(9.8, "m/s^2"))
>>> force
Quantity(Array(49., dtype=float32, ...), unit='kg m / s2')

"""

__all__ = (
    # Core
    "Quantity",
    "Q",  # convenience alias
    # Base
    "AbstractQuantity",
    # Fast
    "BareQuantity",
    "UncheckedQuantity",
    # Angles
    "AbstractAngle",
    "Angle",
    # Base Parametric
    "AbstractParametricQuantity",
    # Functional
    "uconvert",
    "ustrip",
    "is_unit_convertible",
    "wrap_to",
    "is_any_quantity",
    "convert_to_quantity_value",
    "AllowValue",
)


from .setup_package import install_import_hook

with install_import_hook("unxt.quantity"):
    from ._src.quantity.angle import Angle
    from ._src.quantity.api import is_unit_convertible, uconvert, ustrip, wrap_to
    from ._src.quantity.base import AbstractQuantity, is_any_quantity
    from ._src.quantity.base_angle import AbstractAngle
    from ._src.quantity.base_parametric import AbstractParametricQuantity
    from ._src.quantity.flag import AllowValue
    from ._src.quantity.quantity import Q, Quantity
    from ._src.quantity.unchecked import BareQuantity, UncheckedQuantity
    from ._src.quantity.value import convert_to_quantity_value

    # isort: split
    # Register dispatches and conversions
    from ._src.quantity import (
        register_api,
        register_conversions,
        register_dispatches,
        register_primitives,
    )

# Clean up namespace
del (
    register_conversions,
    register_api,
    register_dispatches,
    register_primitives,
    install_import_hook,
)
