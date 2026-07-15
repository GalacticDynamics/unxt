"""Quantities in JAX.

This module provides JAX-compatible quantity types with automatic unit handling,
conversions, and a rich set of mathematical operations that preserve dimensional
correctness.

## Core Classes

- **`Quantity`**: The default quantity class, no dimension parametrization,
  aliased as `Q`.
- **`StaticQuantity`**: Quantity holding a static NumPy value, for use as a JAX
  static argument.
- **`Angle`**: Specialized quantity type for angular measurements with wrapping
  support.

Dimension-parametrized quantities with runtime checking (`ParametricQuantity`,
aliased `PQ`) now live in the separate `unxts.parametric` package.

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

>>> distance = u.Q(100, "m")
>>> distance
Quantity(Array(100, dtype=int32...), unit='m')

Create from arrays

>>> velocities = u.Q([10, 20, 30], "m/s")
>>> velocities
Quantity(Array([10, 20, 30], dtype=int32), unit='m / s')

Unit conversions and arithmetic:

Convert units

>>> distance_km = u.uconvert("km", distance)
>>> distance_km
Quantity(Array(0.1, dtype=float32...), unit='km')

Arithmetic preserves units.

>>> time = u.Q(5, "s")  # use Quantity alias
>>> velocity = distance / time
>>> velocity
Quantity(Array(20., dtype=float32...), unit='m / s')

Strip units for numerical operations

>>> u.ustrip("m", distance)
Array(100, dtype=int32...)


### Working with angles:

Create angle quantities

>>> theta = u.Angle(180, "deg")
>>> theta
Angle(Array(180, dtype=int32...), unit='deg')

Convert to radians

>>> theta_rad = u.uconvert("rad", theta)
>>> theta_rad
Angle(Array(3.1415927, dtype=float32...), unit='rad')

Wrap angles to a range

>>> angle = u.Angle(450, "deg")
>>> wrapped = u.quantity.wrap_to(angle, u.Angle(0, "deg"), u.Angle(360, "deg"))
>>> wrapped
Angle(Array(90, dtype=int32...), unit='deg')


### Advanced usage with JAX transformations:

Quantities work with JAX transformations

>>> def kinetic_energy(mass, velocity):
...     return 0.5 * mass * velocity**2

>>> mass = u.Q(2.0, "kg")
>>> vel = u.Q(10.0, "m/s")
>>> energy = kinetic_energy(mass, vel)
>>> energy
Quantity(Array(100., dtype=float32...), unit='m2 kg / s2')

Convert to standard energy units

>>> u.uconvert("J", energy)
Quantity(Array(100., dtype=float32...), unit='J')

JIT compilation works seamlessly

>>> @jax.jit
... def compute_force(mass, accel):
...     return mass * accel
>>> force = compute_force(u.Q(5.0, "kg"), u.Q(9.8, "m/s^2"))
>>> force
Quantity(Array(49., dtype=float32...), unit='kg m / s2')

"""
# pylint: disable=duplicate-code
# The ``BareQuantity`` deprecation shim in ``__getattr__`` imports ``warnings``
# lazily on purpose; silence the module-level lint for that intentional pattern.
# pylint: disable=import-outside-toplevel

__all__ = (
    # Core
    "Quantity",
    "Q",  # convenience alias
    "StaticQuantity",
    "StaticValue",
    # Base
    "AbstractQuantity",
    # Angles
    "AbstractAngle",
    "Angle",
    # Functional
    "uconvert_value",
    "uconvert",
    "ustrip",
    "is_unit_convertible",
    "wrap_to",
    "is_any_quantity",
    "convert_to_quantity_value",
    "AllowValue",
    # NumPy ufunc registry
    "register_ufunc",
)


from .setup_package import install_import_hook

with install_import_hook("unxt.quantity"):
    from ._src.quantity import (
        AbstractAngle,
        AbstractQuantity,
        AllowValue,
        Angle,
        Q,
        Quantity,
        StaticQuantity,
        StaticValue,
        convert_to_quantity_value,
        is_any_quantity,
        register_ufunc,
    )
    from unxt_api import is_unit_convertible, uconvert, uconvert_value, ustrip, wrap_to

# Clean up namespace
del install_import_hook


_MOVED_TO_PARAMETRIC = {"ParametricQuantity", "PQ", "AbstractParametricQuantity"}


def __getattr__(name: str) -> object:
    if name in _MOVED_TO_PARAMETRIC:
        msg = (
            f"`{name}` moved to the `unxts.parametric` package. Install it "
            "(`pip install unxts.parametric`) and use "
            f"`from unxts.parametric import {name}`."
        )
        raise AttributeError(msg)
    if name == "BareQuantity":
        import warnings  # noqa: PLC0415

        warnings.warn(
            "`BareQuantity` has been renamed to `Quantity` and is now the "
            "default quantity class (unxt v2). The parametric class formerly "
            "named `Quantity` is now `ParametricQuantity`. `BareQuantity` "
            "will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return Quantity
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
