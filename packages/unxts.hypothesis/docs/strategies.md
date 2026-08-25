# Strategies

`unxts.hypothesis` provides [Hypothesis](https://hypothesis.readthedocs.io/) strategies for generating `unxt` objects. Every strategy is re-exported at the package top level; throughout this page it is imported as `ust`.

For how to _use_ these in a test suite, see [How to write property-based tests](testing-guide).

```python
import unxt as u
import unxts.hypothesis as ust
```

| Strategy             | Generates                                      |
| -------------------- | ---------------------------------------------- |
| `named_dimensions()` | a named physical dimension                     |
| `units()`            | a unit of a given dimension                    |
| `derived_units()`    | a unit dimensionally equivalent to a given one |
| `quantities()`       | a `Quantity` of a given unit, shape and dtype  |
| `angles()`           | an `Angle`, optionally wrapped to a range      |
| `unitsystems()`      | an `AbstractUnitSystem` from given base units  |
| `wrap_to()`          | a wrapped copy of a generated quantity         |

`st.from_type()` also works on `unxt` types directly; the registrations are listed at the end of this page.

## Dimensions and units

### `named_dimensions()`

Generate a named physical dimension from Astropy's physical type catalogue. This strategy samples from a curated set of 134 physical types and returns `u.AbstractDimension`. It pairs well with `units()` and `quantities()` for building dimension-aware tests.

**Examples:**

```python
from hypothesis import given
import unxt as u
import unxts.hypothesis as ust


# Any named dimension
@given(dim=ust.named_dimensions())
def test_named_dimension(dim):
    assert isinstance(dim, u.AbstractDimension)


# Units from any dimension
@given(unit=ust.units(ust.named_dimensions()))
def test_units_any_dimension(unit):
    assert u.dimension_of(unit) in [u.dimension(name) for name in ust.DIMENSION_NAMES]


# Quantities from any dimension
@given(q=ust.quantities(unit=ust.units(ust.named_dimensions())))
def test_quantities_any_dimension(q):
    assert isinstance(q, u.Q)
```

See also: `ust.DIMENSION_NAMES` for the full set of names, and `unxt.dimension` to construct dimensions directly from names. You can use `st.sampled_from(ust.DIMENSION_NAMES)` to create custom strategies using these names.

### `derived_units(base, *, integer_powers=True, max_complexity=3)`

Generate units that are dimensionally equivalent to a given base unit.

This is a lower-level strategy that generates units by combining the base unit's decomposed forms and adding cancelling factors. It's useful when you want to generate various representations of the same physical dimension.

**Parameters:**

- `base` (str | Unit | SearchStrategy): Base unit (e.g., "m", "s", "kg") or a hypothesis strategy that generates such units.
- `integer_powers` (bool): If True, only generate units with integer powers of base units (default: True).
- `max_complexity` (int): Maximum number of additional base unit factors to combine (default: 3). Higher values create more complex compound units.

**Returns:** `unxt.AbstractUnit`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust


# Generate units derived from meters
@given(unit=ust.derived_units("m"))
def test_length_derived(unit):
    assert u.dimension_of(unit) == u.dimension("length")


# derived_units takes a base *unit* (or a strategy of units), not a dimension
# name -- so pass unit strings like "m/s" (velocity) and "m/s2" (acceleration)
@given(unit=ust.derived_units(st.sampled_from(["m/s", "m/s2"])))
def test_velocity_derived(unit):
    assert u.dimension_of(unit) in (
        u.dimension("velocity"),
        u.dimension("acceleration"),
    )


# Control complexity
@given(unit=ust.derived_units("kg", max_complexity=1))
def test_simple_mass_units(unit):
    assert u.dimension_of(unit) == u.dimension("mass")
```

### `units(dimension=named_dimensions(), /, **kwargs)`

Generate random `Unit` objects from astropy.

**Parameters:**

- `dimension` (str | Dimension | SearchStrategy, positional-only): The physical dimension of the unit. Defaults to the `named_dimensions()` strategy (units drawn across many dimensions). Examples: `"length"`, `"velocity"`, `"energy"`.
- `**kwargs`: forwarded to `derived_units` — notably `integer_powers` (bool, default `True`) and `max_complexity` (int, default `3`; higher values give more complex compound units like `m^2/s`).

**Returns:** `unxt.AbstractUnit`

**Examples:**

```python
from hypothesis import given

import unxt as u
import unxts.hypothesis as ust


# Generate unit with low-complexity away from SI base units
@given(u=ust.units())
def test_any_unit(u):
    assert u is not None


# Generate length units
@given(length_unit=ust.units("length"))
def test_length_unit(length_unit):
    assert u.dimension_of(length_unit) == u.dimension("length")


# Generate complex compound units
@given(u=ust.units(max_complexity=3))
def test_complex_unit(u):
    assert u is not None
```

## Quantities, angles and unit systems

### `quantities(unit=units(...), *, shape=(), dtype=jnp.float32, ...)`

Generate random `Quantity` objects.

**Parameters:**

- `unit` (str | Unit | Dimension | st.SearchStrategy): Unit for the quantity. Defaults to the `units()` strategy (units drawn across many dimensions). Accepts:
  - `str`: a unit string (e.g., `"m"`, `"km/s"`)
  - `Unit`: a unit object
  - `Dimension`: a physical dimension (units of that dimension are generated)
  - Strategy: a Hypothesis strategy generating units (e.g., from `units()`)
- `shape` (int | tuple[int, ...] | st.SearchStrategy): Shape of the array. Defaults to `()` (scalar). Accepts an int (e.g., `3` → shape `(3,)`), a tuple (e.g., `(3, 3)`), or a strategy that generates shapes.
- `dtype` (np.dtype | st.SearchStrategy): Data type of the array. Defaults to `jnp.float32`.
- `elements`, `unique`, `static_value`, `quantity_cls`: forwarded to the underlying array strategy / quantity construction.

**Returns:** `unxt.Quantity`

**Examples:**

```python
from hypothesis import given, strategies as st

import numpy as np
import unxt as u
import unxts.hypothesis as ust


# Generate any quantity
@given(q=ust.quantities())
def test_any_quantity(q):
    assert q.value is not None
    assert q.unit is not None


# Generate scalar quantities
@given(q=ust.quantities(shape=()))
def test_scalar_quantity(q):
    assert q.ndim == 0


# Generate matrix quantities
@given(q=ust.quantities(shape=(3, 3)))
def test_matrix_quantity(q):
    assert q.shape == (3, 3)


# Generate quantities with specific units
@given(q=ust.quantities(unit="m"))
def test_length_quantity(q):
    assert u.dimension_of(q) == u.dimension("length")


# Generate quantities with varying units from a strategy
@given(q=ust.quantities(unit=ust.units("energy")))
def test_energy_quantity(q):
    assert u.dimension_of(q) == u.dimension("energy")


# Combine multiple parameters
@given(
    q=ust.quantities(
        shape=st.integers(1, 10),
        dtype=st.sampled_from([np.float32, np.float64]),
        unit=ust.units("length", max_complexity=1),
    )
)
def test_custom_quantity(q):
    assert q.dtype in (np.float32, np.float64)
    assert u.dimension_of(q) == u.dimension("length")
```

### `unitsystems(*units)`

Generate random `UnitSystem` objects.

**Parameters:**

- `*units` (str | Unit | st.SearchStrategy[Unit]): Variable number of unit specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `Unit`: Fixed unit object
  - Strategy: A Hypothesis strategy that generates units (e.g., from `units()`)

**Returns:** `unxt.AbstractUnitSystem`

**Examples:**

```python
from hypothesis import given

import unxts.hypothesis as ust


# Generate MKS system with fixed units
@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_mks_system(sys):
    assert len(sys) == 4


# Generate system with varying length unit
@given(sys=ust.unitsystems(ust.units("length"), "s", "kg", "rad"))
def test_varying_length_system(sys):
    # Length unit varies, others are fixed
    assert len(sys) == 4


# Generate system with multiple varying units
@given(sys=ust.unitsystems(ust.units("length"), ust.units("time"), "kg", "rad"))
def test_multiple_varying_units(sys):
    assert len(sys) == 4


# Generate galactic unit system
@given(sys=ust.unitsystems("kpc", "Myr", "Msun", "rad"))
def test_galactic_system(sys):
    assert len(sys) == 4
```

### `angles(*, wrap_to=None, **kwargs)`

Generate random `Angle` objects with optional wrapping bounds.

This is a specialized strategy for generating `unxt.Angle` instances, which are quantities with angle dimensions. Angles can optionally have wrapping bounds that keep values within a specified range (e.g., 0-360 degrees).

**Parameters:**

- `wrap_to` (tuple | st.SearchStrategy | None): Wrapping bounds for the angle. Can be:
  - `None` (default): No wrapping applied
  - `tuple`: Pair of `(min, max)` quantities defining the wrapping range
  - Strategy: A Hypothesis strategy that generates `(min, max)` tuples
- `**kwargs`: Additional keyword arguments passed to `quantities()`. Common options include `dtype`, `shape`, `elements`, `unique`. The `unit` and `quantity_cls` parameters are set automatically and should not be provided.

**Returns:** `unxt.Angle`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust


# Generate any angle
@given(angle=ust.angles())
def test_any_angle(angle):
    assert isinstance(angle, u.Angle)
    assert u.dimension_of(angle) == u.dimension("angle")


# Generate angles with wrapping to 0-360 degrees
@given(angle=ust.angles(wrap_to=(u.Q(0, "deg"), u.Q(360, "deg"))))
def test_wrapped_angle_degrees(angle):
    assert isinstance(angle, u.Angle)
    assert angle.wrap_to is not None


# Generate angles with wrapping to 0-2π radians
@given(angle=ust.angles(wrap_to=(u.Q(0, "rad"), u.Q(6.28318530718, "rad"))))
def test_wrapped_angle_radians(angle):
    assert isinstance(angle, u.Angle)
    assert 0 <= angle.value <= 6.28318530718


# Generate angles with specific shape
@given(angle=ust.angles(shape=(3,)))
def test_angle_array(angle):
    assert isinstance(angle, u.Angle)
    assert angle.shape == (3,)


# Generate angles with dynamic wrapping bounds
@given(angle=ust.angles(wrap_to=st.just((u.Q(-180, "deg"), u.Q(180, "deg"))), shape=()))
def test_angle_with_strategy_wrapping(angle):
    assert isinstance(angle, u.Angle)
    assert -180 <= angle.value <= 180
```

### `wrap_to(quantity, min, max)`

Generate wrapped quantities by constraining values to a specified range.

This strategy takes a quantity (or quantity strategy) and wraps the generated values to the range [min, max) using modular arithmetic. This is particularly useful for periodic quantities like angles.

**Parameters:**

- `quantity` (u.AbstractQuantity | st.SearchStrategy): Quantity or strategy that generates the base quantity to wrap.
- `min` (u.AbstractQuantity | st.SearchStrategy): Minimum value of the wrapping range (inclusive).
- `max` (u.AbstractQuantity | st.SearchStrategy): Maximum value of the wrapping range (exclusive).

**Returns:** `unxt.AbstractQuantity`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust


# Wrap angles to 0-360 degree range
@given(
    angle=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle),
        min=u.Q(0, "deg"),
        max=u.Q(360, "deg"),
    )
)
def test_wrapped_angle(angle):
    assert 0 <= angle.value < 360


# Wrap angles to -π to π range
@given(
    angle=ust.wrap_to(
        ust.quantities("rad", quantity_cls=u.Angle),
        min=u.Q(-3.14159, "rad"),
        max=u.Q(3.14159, "rad"),
    )
)
def test_wrapped_angle_symmetric(angle):
    assert -3.14159 <= angle.value < 3.14159


# Dynamic min/max using strategies
@given(
    angle=ust.wrap_to(
        ust.quantities("rad", quantity_cls=u.Angle),
        min=st.just(u.Q(0, "rad")),
        max=st.just(u.Q(6.28318530718, "rad")),
    )
)
def test_wrapped_angle_with_strategies(angle):
    assert 0 <= angle.value < 6.28318530718
```

Note: The `angles()` strategy provides a more convenient interface for generating wrapped angles and should be preferred for most use cases involving angle generation.

## Type strategy registration

The package automatically registers type strategies for Hypothesis's `st.from_type()` function, enabling automatic strategy generation for unxt types. This allows you to use type annotations directly in your tests without explicitly importing the strategy functions.

**Registered Types:**

- `u.AbstractQuantity` → uses `quantities()`
- `u.Angle` → uses `angles()`
- `u.AbstractUnitSystem` → uses `unitsystems()`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust  # Import to register strategies


# Hypothesis automatically uses the registered strategies
@given(q=st.from_type(u.AbstractQuantity))
def test_quantity_via_from_type(q):
    """Test quantities generated via st.from_type()."""
    assert isinstance(q, u.AbstractQuantity)
    assert u.dimension_of(q) is not None


@given(a=st.from_type(u.Angle))
def test_angle_via_from_type(a):
    """Test angles generated via st.from_type()."""
    assert isinstance(a, u.Angle)
    assert u.dimension_of(a) == u.dimension("angle")


@given(usys=st.from_type(u.AbstractUnitSystem))
def test_unitsystem_via_from_type(usys):
    """Test unit systems generated via st.from_type()."""
    assert isinstance(usys, u.AbstractUnitSystem)
```

This integration makes tests more concise and easier to read, especially when combined with type-annotated function signatures:

```python
from hypothesis import given, strategies as st

import unxt as u


def calculate_momentum(mass: u.Q, velocity: u.Q) -> u.Q:
    """Calculate momentum: p = m * v"""
    return mass * velocity


# Using st.from_type() for cleaner test code
@given(
    mass=st.from_type(u.AbstractQuantity),
    velocity=st.from_type(u.AbstractQuantity),
)
def test_momentum_dimensions(mass, velocity):
    """Momentum has the right dimensions."""
    momentum = calculate_momentum(mass, velocity)
    expected_dim = u.dimension_of(mass) * u.dimension_of(velocity)
    assert u.dimension_of(momentum) == expected_dim
```
