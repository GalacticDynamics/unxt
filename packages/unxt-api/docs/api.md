# API Reference

## Dimensions

### `dimension(obj)`

Construct a dimension from various inputs.

**Abstract Signature:**

```python
from typing import Any
import plum


@plum.dispatch.abstract
def dimension(obj: Any, /) -> Any:
    """Construct the dimension."""
```

**Example Implementations** (in `unxt`):

- `dimension(dimension: AbstractDimension)` - Identity operation
- `dimension(s: str)` - Construct from string (e.g., `"length"`, `"velocity"`)

**Examples:**

```python
import unxt as u

# From string
length_dim = u.dimension("length")

# From unit
meter_dim = u.dimension_of(u.unit("m"))
```

### `dimension_of(obj)`

Return the dimension of an object.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def dimension_of(obj: Any, /) -> Any:
    """Return the dimension of the given units."""
```

**Example Implementations** (in `unxt`):

- `dimension_of(unit: AbstractUnit)` - Get dimension from unit
- `dimension_of(quantity: Quantity)` - Get dimension from quantity

**Examples:**

```python
import unxt as u

# From unit
u.dimension_of(u.unit("km"))  # PhysicalType('length')

# From quantity
q = u.Q(5, "kg")
u.dimension_of(q)  # PhysicalType('mass')
```

## Units

### `unit(obj)`

Construct units from various inputs.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def unit(obj: Any, /) -> u.AbstractUnit:
    """Construct the units from a units object."""
```

**Example Implementations** (in `unxt`):

- `unit(unit: AbstractUnit)` - Identity operation
- `unit(s: str)` - Parse unit from string (e.g., `"m"`, `"km/s"`)

**Examples:**

```python
import unxt as u

# From string
meter = u.unit("m")
velocity = u.unit("km/s")
```

### `unit_of(obj)`

Return the units of an object.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def unit_of(obj: Any, /) -> u.AbstractUnit:
    """Return the units of an object."""
```

**Example Implementations** (in `unxt`):

- `unit_of(obj: Any)` - Returns `None` for objects without units
- `unit_of(unit: AbstractUnit)` - Identity operation
- `unit_of(quantity: Quantity)` - Extract unit from quantity

**Examples:**

```python
import unxt as u

# From quantity
q = u.Q(5, "m")
u.unit_of(q)  # Unit("m")

# From non-quantity
u.unit_of(5)  # None
```

## Quantities

### `uconvert(to_unit, quantity)`

Convert a quantity to the specified units.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def uconvert(u: Any, x: Any, /) -> Any:
    """Convert the quantity to the specified units."""
```

**Example Implementations** (in `unxt`):

- `uconvert(to_unit: str | AbstractUnit, quantity: Quantity)` - Convert quantity
  to new units

**Examples:**

```python
import unxt as u

# Convert quantity
q = u.Q(1, "km")
u.uconvert("m", q)  # Quantity(Array(1000., ...), unit='m')
```

### `ustrip(quantity)` / `ustrip(to_unit, quantity)`

Strip units from a quantity, optionally converting first.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def ustrip(*args: Any) -> Any:
    """Strip the units from the quantity, first converting if necessary."""
```

**Example Implementations** (in `unxt`):

- `ustrip(quantity: Quantity)` - Get value in current units
- `ustrip(to_unit: str | AbstractUnit, quantity: Quantity)` - Convert then get
  value.

**Examples:**

```python
import unxt as u

# Strip current units
q = u.Q(5, "km")
u.ustrip(q)  # Array(5., ...)

# Convert then strip
u.ustrip("m", q)  # Array(5000., ...)
```

### `is_unit_convertible(to_unit, from_obj)`

Check if units are convertible.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def is_unit_convertible(to_unit: Any, from_: Any, /) -> bool:
    """Check if the units are convertible."""
```

**Example Implementations** (in `unxt`):

- `is_unit_convertible(to_unit: AbstractUnit, from_unit: AbstractUnit)` - Check
  unit compatibility
- `is_unit_convertible(to_unit: AbstractUnit, from_quantity: Quantity)` - Check
  if quantity can convert

**Examples:**

```python
import unxt as u

# Check unit compatibility
u.is_unit_convertible(u.unit("km"), u.unit("m"))  # True
u.is_unit_convertible(u.unit("kg"), u.unit("m"))  # False

# Check quantity
q = u.Q(5, "m")
u.is_unit_convertible(u.unit("km"), q)  # True
u.is_unit_convertible(u.unit("kg"), q)  # False
```

### `wrap_to(value, min, max)`

Wrap a value to the range [min, max).

**Abstract Signature:**

```python
@plum.dispatch.abstract
def wrap_to(x: Any, min: Any, max: Any, /) -> Any:
    """Wrap to the range [min, max)."""
```

**Example Implementations** (in `unxt`):

- `wrap_to(angle: Quantity, min: Quantity, max: Quantity)` - Wrap angles to
  range

**Examples:**

```python
import unxt as u

# Wrap angle to [0, 360) degrees
angle = u.Q(370, "deg")
min_angle = u.Q(0, "deg")
max_angle = u.Q(360, "deg")

u.quantity.wrap_to(angle, min_angle, max_angle)  # Angle(Array(10, ...), unit='deg')

# Can also use keyword arguments
u.quantity.wrap_to(angle, min=min_angle, max=max_angle)
```

## Unit Systems

### `unitsystem_of(obj)`

Return the unit system of an object.

**Abstract Signature:**

```python
@plum.dispatch.abstract
def unitsystem_of(obj: Any, /) -> Any:
    """Return the unit system of an object."""
```

**Example Implementations** (in `unxt`):

- `unitsystem_of(quantity: Quantity)` - Get unit system if quantity has one
- `unitsystem_of(unitsystem: AbstractUnitSystem)` - Identity operation

**Examples:**

```python
import unxt as u

# From unit system
sys = u.unitsystem("m", "s", "kg", "rad")
u.unitsystem_of(sys)  # Returns sys

# From quantity (if it has an associated system)
# This depends on how the quantity was constructed
```
