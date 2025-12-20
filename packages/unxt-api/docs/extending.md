# Extending `unxt`

This guide shows how to extend `unxt` using the multiple dispatch system
provided by `unxt-api`.

## Understanding the Dispatch System

`unxt` uses [plum](https://beartype.github.io/plum/) for multiple dispatch,
which means:

- Functions can have **multiple implementations** for different argument types
- The **runtime types** of arguments determine which implementation executes
- You can **add new implementations** without modifying unxt's source code

## Quick Example

Let's say you have a custom quantity type and want it to work with unxt's
functions:

```python
from plum import dispatch
import unxt as u


class Temperature:
    """Custom temperature quantity."""

    def __init__(self, value, unit="K"):
        self.value = value
        self.unit_str = unit

    def __repr__(self):
        return f"Temperature({self.value}, {self.unit_str!r})"


# Register how to get units from Temperature
@dispatch
def unit_of(obj: Temperature, /) -> u.AbstractUnit:
    """Get unit from Temperature."""
    return u.unit(obj.unit_str)


# Register how to get dimension from Temperature
@dispatch
def dimension_of(obj: Temperature, /) -> u.AbstractDimension:
    """Get dimension from Temperature."""
    return u.dimension("temperature")


# Now it works!
temp = Temperature(300, "K")
u.unit_of(temp)  # Unit("K")
u.dimension_of(temp)  # PhysicalType('temperature')
```

## Registering Dispatch Functions

### Step 1: Import the Abstract Function

Import the abstract function from `unxt_api`:

```python
from unxt_api import unit_of, dimension_of, uconvert
```

### Step 2: Use `@dispatch` Decorator

Register your implementation with type annotations:

```python
from plum import dispatch


@dispatch
def unit_of(obj: Temperature, /) -> u.AbstractUnit:
    """Docstring explaining this specific implementation."""
    return u.unit(obj.unit_str)
```

### Step 3: Implement the Logic

Provide the concrete implementation for your type:

```python
@dispatch
def unit_of(obj: Temperature, /) -> u.AbstractUnit:
    """Get unit from Temperature object."""
    return u.unit(obj.unit_str)
```

## Common Extension Patterns

### Adding Unit Support to Custom Types

If you have a type that represents a physical quantity:

```python
from plum import dispatch
import unxt as u


class Distance:
    def __init__(self, meters):
        self.meters = meters

    def __repr__(self):
        return f"Distance({self.meters} m)"


@dispatch
def unit_of(obj: Distance, /) -> u.AbstractUnit:
    """Distance is always in meters."""
    return u.unit("m")


@dispatch
def dimension_of(obj: Distance, /) -> u.AbstractDimension:
    """Distance has length dimension."""
    return u.dimension("length")


@dispatch
def ustrip(to_unit: u.AbstractUnit, obj: Distance, /):
    """Convert Distance to specified unit and return value."""
    # Convert meters to target unit
    in_meters = u.Q(obj.meters, "m")
    return u.ustrip(to_unit, in_meters)


@dispatch
def uconvert(to_unit: u.AbstractUnit, obj: Distance, /):
    """Convert Distance to specified unit."""
    in_meters = u.Q(obj.meters, "m")
    return u.uconvert(to_unit, in_meters)


# Usage
d = Distance(1000)
u.unit_of(d)  # Unit("m")
u.dimension_of(d)  # PhysicalType('length')
u.ustrip(u.unit("km"), d)  # Array(1., ...)
u.uconvert(u.unit("km"), d)  # Quantity(Array(1., ...), unit='km')
```

### Converting Between Custom Types and unxt.Quantity

Create bidirectional conversion:

```python
from plum import dispatch
import unxt as u


class Vector3D:
    """3D vector with units."""

    def __init__(self, x, y, z, unit="m"):
        self.x = x
        self.y = y
        self.z = z
        self.unit_str = unit

    def to_quantity(self):
        """Convert to unxt.Quantity."""
        import jax.numpy as jnp

        return u.Q(jnp.array([self.x, self.y, self.z]), self.unit_str)

    @classmethod
    def from_quantity(cls, q):
        """Create from unxt.Quantity."""
        if q.shape != (3,):
            raise ValueError("Expected 3-element quantity")
        v = q.value
        return cls(v[0], v[1], v[2], str(q.unit))


@dispatch
def unit_of(obj: Vector3D, /) -> u.AbstractUnit:
    return u.unit(obj.unit_str)


@dispatch
def dimension_of(obj: Vector3D, /) -> u.AbstractDimension:
    return u.dimension_of(u.unit(obj.unit_str))


@dispatch
def uconvert(to_unit: u.AbstractUnit, obj: Vector3D, /):
    """Convert Vector3D to new units."""
    q = obj.to_quantity()
    converted_q = u.uconvert(to_unit, q)
    return Vector3D.from_quantity(converted_q)


# Usage
v = Vector3D(1, 2, 3, "km")
v_meters = u.uconvert(u.unit("m"), v)
print(v_meters.x, v_meters.unit_str)  # 1000.0 m
```

### Supporting Custom Unit Systems

Extend `unitsystem_of` for your types:

```python
from plum import dispatch
import unxt as u


class AstronomicalObject:
    """Object with an associated unit system."""

    def __init__(self, position, velocity):
        self.position = position  # in kpc
        self.velocity = velocity  # in km/s
        self._system = u.unitsystem("kpc", "Myr", "Msun", "rad")


@dispatch
def unitsystem_of(obj: AstronomicalObject, /):
    """Get the astronomical unit system."""
    return obj._system


# Usage
obj = AstronomicalObject([1, 2, 3], [10, 20, 30])
sys = u.unitsystem_of(obj)
sys["length"]  # Unit("kpc")
sys["time"]  # Unit("Myr")
```

## Advanced Patterns

### Conditional Dispatch Based on Multiple Arguments

Use multiple type annotations for complex dispatch:

```python
from plum import dispatch
import unxt as u


class SpecialQuantity:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit


@dispatch
def uconvert(to_unit: str, obj: SpecialQuantity, /):
    """Convert SpecialQuantity when target is a string."""
    # Implementation for string units
    return SpecialQuantity(u.ustrip(to_unit, obj.value, obj.unit), to_unit)


@dispatch
def uconvert(to_unit: u.AbstractUnit, obj: SpecialQuantity, /):
    """Convert SpecialQuantity when target is AbstractUnit."""
    # Implementation for Unit objects
    return SpecialQuantity(u.ustrip(to_unit, obj.value, obj.unit), str(to_unit))
```

### Handling Multiple Dispatch Signatures

One function can have many implementations:

```python
from plum import dispatch
import unxt as u


@dispatch
def dimension_of(obj: Temperature, /) -> u.AbstractDimension:
    """Temperature dimension."""
    return u.dimension("temperature")


@dispatch
def dimension_of(obj: Distance, /) -> u.AbstractDimension:
    """Distance dimension."""
    return u.dimension("length")


@dispatch
def dimension_of(obj: Vector3D, /) -> u.AbstractDimension:
    """Vector3D dimension."""
    return u.dimension_of(u.unit(obj.unit_str))


# The correct implementation is chosen automatically
temp_dim = u.dimension_of(Temperature(300))  # PhysicalType('temperature')
dist_dim = u.dimension_of(Distance(100))  # PhysicalType('length')
vec_dim = u.dimension_of(Vector3D(1, 2, 3, "m"))  # PhysicalType('length')
```

### Fallback Implementations

Provide a generic fallback for types that don't have specific handling:

```python
from plum import dispatch


@dispatch
def unit_of(obj: object, /) -> u.AbstractUnit | None:
    """Fallback: objects without units return None."""
    return None


# This catches anything that doesn't have a more specific implementation
class MyClass:
    pass


u.unit_of(MyClass())  # None
```

## Debugging Dispatch

### Viewing All Implementations

See what implementations are registered:

```python
import unxt as u

# List all implementations of unit_of
u.unit_of.methods

# List all implementations of dimension_of
u.dimension_of.methods

# List all implementations of uconvert
u.uconvert.methods
```

### Understanding Dispatch Resolution

When you call a dispatch function, plum selects the most specific
implementation:

```python
from plum import dispatch
import unxt as u


class Animal:
    pass


class Dog(Animal):
    def __init__(self):
        self.mass_kg = 20


@dispatch
def unit_of(obj: Animal, /) -> u.AbstractUnit | None:
    """Generic animal - no units."""
    return None


@dispatch
def unit_of(obj: Dog, /) -> u.AbstractUnit:
    """Dogs have mass in kg."""
    return u.unit("kg")


# More specific implementation wins
animal = Animal()
dog = Dog()

u.unit_of(animal)  # None (uses Animal implementation)
u.unit_of(dog)  # Unit("kg") (uses Dog implementation)
```

### Checking Dispatch Ambiguity

Plum will raise an error if dispatch is ambiguous:

```python
from plum import dispatch


class A:
    pass


class B:
    pass


class C(A, B):
    pass


@dispatch
def my_func(obj: A, /):
    return "A"


@dispatch
def my_func(obj: B, /):
    return "B"


# This would be ambiguous - don't do this!
# my_func(C())  # AmbiguousLookupError


# Instead, add a specific implementation
@dispatch
def my_func(obj: C, /):
    return "C"


my_func(C())  # "C" - now unambiguous
```

## Best Practices

### 1. Use Type Annotations

Always use type annotations for dispatch to work:

```python
# Good
@dispatch
def unit_of(obj: Temperature, /) -> u.AbstractUnit:
    return u.unit("K")


# Bad - won't dispatch correctly
@dispatch
def unit_of(obj):  # Missing type annotation!
    return u.unit("K")
```

### 2. Document Each Implementation

Add docstrings explaining the specific behavior:

```python
@dispatch
def dimension_of(obj: Temperature, /) -> u.AbstractDimension:
    """Get dimension from Temperature.

    Temperature objects always have temperature dimension,
    regardless of their unit (K, C, F, etc.).
    """
    return u.dimension("temperature")
```

### 3. Be Consistent with Return Types

Keep return types consistent across implementations:

```python
# Good - both return AbstractUnit
@dispatch
def unit_of(obj: Temperature, /) -> u.AbstractUnit:
    return u.unit("K")


@dispatch
def unit_of(obj: Distance, /) -> u.AbstractUnit:
    return u.unit("m")
```

### 4. Handle Edge Cases

Consider what should happen with invalid inputs:

```python
@dispatch
def uconvert(to_unit: u.AbstractUnit, obj: Temperature, /):
    """Convert temperature to new unit."""
    # Check if conversion makes sense
    if not u.is_unit_convertible(to_unit, u.unit(obj.unit_str)):
        raise ValueError(f"Cannot convert temperature to {to_unit}")

    value = u.ustrip(to_unit, obj.value, obj.unit_str)
    return Temperature(value, str(to_unit))
```

### 5. Test Your Dispatch Functions

Write tests to ensure dispatch works correctly:

```python
def test_temperature_unit_of():
    """Test unit_of works for Temperature."""
    temp = Temperature(300, "K")
    unit = u.unit_of(temp)

    assert unit == u.unit("K")
    assert u.dimension_of(unit) == "temperature"


def test_temperature_conversion():
    """Test converting Temperature."""
    temp = Temperature(0, "degC")
    temp_k = u.uconvert("K", temp)

    assert temp_k.unit_str == "K"
    assert abs(temp_k.value - 273.15) < 1e-10
```

## Package Integration Examples

### Minimal Dependency Package

Create a package that defines custom types with minimal dependencies:

```python
# my_physics_package/core.py
"""Core physics types."""


class Force:
    def __init__(self, newtons):
        self.newtons = newtons


class Energy:
    def __init__(self, joules):
        self.joules = joules
```

```text
# my_physics_package/unxt_support.py
"""unxt integration - optional dependency."""

from .core import Force, Energy

# Try to import unxt dependencies
from plum import dispatch

try:
    import unxt as u
except ImportError:
    HAS_UNXT = False
else:
    HAS_UNXT = True

    # Register Force with unxt
    @dispatch
    def unit_of(obj: Force, /) -> u.AbstractUnit:
        return u.unit("N")

    @dispatch
    def dimension_of(obj: Force, /) -> u.AbstractDimension:
        return u.dimension("force")

    # Register Energy with unxt
    @dispatch
    def unit_of(obj: Energy, /) -> u.AbstractUnit:
        return u.unit("J")

    @dispatch
    def dimension_of(obj: Energy, /) -> u.AbstractDimension:
        return u.dimension("energy")
```

Users can use your package with or without unxt!

## See Also

- [unxt-api API Reference](index.md)
- [plum Documentation](https://beartype.github.io/plum/)
- [unxt Documentation](https://unxt.readthedocs.io/)
