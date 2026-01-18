# unxt-hypothesis

```{toctree}
:maxdepth: 1
:hidden:

api
testing-guide
```

Hypothesis strategies for property-based testing with
[unxt](https://github.com/GalacticDynamics/unxt).

This package provides [Hypothesis](https://hypothesis.readthedocs.io/)
strategies for generating random `Quantity`, `Angle`, `Unit`, `Dimension`, and
`UnitSystem` objects for property-based testing.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt-hypothesis
```

:::

:::{tab-item} uv

````bash

```bash
uv add unxt-hypothesis
````

:::

::::

## Quick Start

```python
import jax
from hypothesis import given

import unxt as u
import unxt_hypothesis as ust


@given(dim=ust.named_dimensions())
def test_named_dimension(dim):
    """Test that named dimensions are generated correctly."""
    assert isinstance(dim, u.AbstractDimension)


@given(unit=ust.units("length"))
def test_unit_property(unit):
    """Test that units can be converted to strings."""
    assert isinstance(unit, u.AbstractUnit)


@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_unitsystem_property(sys):
    """Test that unit systems have expected base units."""
    assert isinstance(sys, u.AbstractUnitSystem)
    assert len(sys) == 4


@given(q=ust.quantities(unit="km/s"))
def test_quantity_property(q):
    """Test that all quantities have value and unit."""
    assert isinstance(q.value, jax.Array)
    assert q.unit == u.unit("km/s")


@given(angle=ust.angles())
def test_angle_property(angle):
    """Test that all angles have the angle dimension."""
    assert u.dimension_of(angle) == u.dimension("angle")
```

## Strategies

### `named_dimensions()`

Generate a named physical dimension from Astropy's physical type catalogue. This
strategy samples from a curated set of 134 physical types and returns
`u.AbstractDimension`. It pairs well with `units()` and `quantities()` for
building dimension-aware tests.

**Examples:**

```python
from hypothesis import given
import unxt as u
import unxt_hypothesis as ust


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
    assert isinstance(q, u.Quantity)
```

See also: `ust.DIMENSION_NAMES` for the full set of names, and `unxt.dimension`
to construct dimensions directly from names. You can use
`st.sampled_from(ust.DIMENSION_NAMES)` to create custom strategies using these
names.

### `derived_units(base, *, integer_powers=True, max_complexity=3)`

Generate units that are dimensionally equivalent to a given base unit.

This is a lower-level strategy that generates units by combining the base unit's
decomposed forms and adding cancelling factors. It's useful when you want to
generate various representations of the same physical dimension.

**Parameters:**

- `base` (str | Unit | SearchStrategy): Base unit (e.g., "m", "s", "kg") or a
  hypothesis strategy that generates such units.
- `integer_powers` (bool): If True, only generate units with integer powers of
  base units (default: True).
- `max_complexity` (int): Maximum number of additional base unit factors to
  combine (default: 3). Higher values create more complex compound units.

**Returns:** `unxt.AbstractUnit`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust


# Generate units derived from meters
@given(unit=ust.derived_units("m"))
def test_length_derived(unit):
    assert u.dimension_of(unit) == u.dimension("length")


# Generate units from a strategy
@given(unit=ust.derived_units(st.sampled_from(["velocity", "acceleration"])))
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

### `units(dimension=None, *, max_complexity=2, allow_non_integer_powers=False)`

Generate random `Unit` objects from astropy.

**Parameters:**

- `dimension` (str | Dimension | SearchStrategy | None): The physical dimension
  of the unit. If None, generates units from various dimensions. Examples:
  `"length"`, `"velocity"`, `"energy"`.
- `max_complexity` (int): Maximum complexity of compound units (default: 2).
  Higher values generate more complex compound units like `m^2/s`.
- `allow_non_integer_powers` (bool): Whether to allow non-integer powers in
  units (default: False). When True, can generate units like `m^0.5`.

**Returns:** `unxt.AbstractUnit`

**Examples:**

```python
from hypothesis import given

import unxt as u
import unxt_hypothesis as ust


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

### `quantities(*, shape=None, dtype=None, unit=None)`

Generate random `Quantity` objects.

**Parameters:**

- `shape` (int | tuple[int, ...] | st.SearchStrategy | None): Shape of the
  array. Can be:
  - `None` (default): Generates small arrays with various shapes
  - `int`: Scalar shape specification (e.g., `3` for shape `(3,)`)
  - `tuple`: Explicit shape (e.g., `(3, 3)` for a 3×3 matrix)
  - Strategy: A Hypothesis strategy that generates shapes
- `dtype` (np.dtype | st.SearchStrategy | None): Data type of the array.
  Defaults to `float32`.
- `unit` (str | Unit | st.SearchStrategy | None): Unit for the quantity. Can be:
  - `None` (default): Generates quantities with various common units
  - `str`: Specific unit string (e.g., `"m"`, `"km/s"`)
  - `Unit`: Specific unit object
  - Strategy: A Hypothesis strategy that generates units (e.g., from `units()`)

**Returns:** `unxt.Quantity`

**Examples:**

```python
from hypothesis import given, strategies as st

import numpy as np
import unxt as u
import unxt_hypothesis as ust


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

- `*units` (str | Unit | st.SearchStrategy[Unit]): Variable number of unit
  specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `Unit`: Fixed unit object
  - Strategy: A Hypothesis strategy that generates units (e.g., from `units()`)

**Returns:** `unxt.AbstractUnitSystem`

**Examples:**

```python
from hypothesis import given

import unxt_hypothesis as ust


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

This is a specialized strategy for generating `unxt.Angle` instances, which are
quantities with angle dimensions. Angles can optionally have wrapping bounds
that keep values within a specified range (e.g., 0-360 degrees).

**Parameters:**

- `wrap_to` (tuple | st.SearchStrategy | None): Wrapping bounds for the angle.
  Can be:
  - `None` (default): No wrapping applied
  - `tuple`: Pair of `(min, max)` quantities defining the wrapping range
  - Strategy: A Hypothesis strategy that generates `(min, max)` tuples
- `**kwargs`: Additional keyword arguments passed to `quantities()`. Common
  options include `dtype`, `shape`, `elements`, `unique`. The `unit` and
  `quantity_cls` parameters are set automatically and should not be provided.

**Returns:** `unxt.Angle`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust


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

This strategy takes a quantity (or quantity strategy) and wraps the generated
values to the range [min, max) using modular arithmetic. This is particularly
useful for periodic quantities like angles.

**Parameters:**

- `quantity` (u.AbstractQuantity | st.SearchStrategy): Quantity or strategy that
  generates the base quantity to wrap.
- `min` (u.AbstractQuantity | st.SearchStrategy): Minimum value of the wrapping
  range (inclusive).
- `max` (u.AbstractQuantity | st.SearchStrategy): Maximum value of the wrapping
  range (exclusive).

**Returns:** `unxt.AbstractQuantity`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust


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

Note: The `angles()` strategy provides a more convenient interface for
generating wrapped angles and should be preferred for most use cases involving
angle generation.

## Type Strategy Registration

The package automatically registers type strategies for Hypothesis's
`st.from_type()` function, enabling automatic strategy generation for unxt
types. This allows you to use type annotations directly in your tests without
explicitly importing the strategy functions.

**Registered Types:**

- `u.AbstractQuantity` → uses `quantities()`
- `u.Angle` → uses `angles()`
- `u.AbstractUnitSystem` → uses `unitsystems()`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust  # Import to register strategies


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

This integration makes tests more concise and easier to read, especially when
combined with type-annotated function signatures:

```python
from hypothesis import given, strategies as st

import unxt as u


def calculate_momentum(mass: u.Quantity, velocity: u.Quantity) -> u.Quantity:
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

## Advanced Usage

### Combining Strategies

The strategies are designed to work together seamlessly:

```python
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust


# Create quantities with units from a unit strategy
@given(unit=ust.units("length"), q=ust.quantities(unit=ust.units("length")))
def test_consistent_length_units(unit, q):
    """Both unit and q have length dimension."""
    assert u.dimension_of(unit) == u.dimension("length")
    assert u.dimension_of(q) == u.dimension("length")


# Create unit systems with varying complexity
@given(
    sys=ust.unitsystems(
        ust.units("length", max_complexity=1),
        ust.units("time", max_complexity=1),
        ust.units("mass", max_complexity=1),
        "rad",
    )
)
def test_simple_unit_system(sys):
    """Generate systems with simple base units only."""
    assert len(sys) == 4
```

### Testing Unitful Functions

Here's a complete example of using these strategies to test a physics function:

```python
import jax.numpy as jnp
from hypothesis import given, strategies as st

import unxt as u
import unxt_hypothesis as ust


def kinetic_energy(mass, velocity):
    """Calculate kinetic energy: KE = 0.5 * m * v^2"""
    return 0.5 * mass * velocity**2


@given(
    mass=ust.quantities(unit="kg", shape=()),
    velocity=ust.quantities(unit="m/s", shape=()),
)
def test_kinetic_energy_positive(mass, velocity):
    """Kinetic energy is always non-negative."""
    ke = kinetic_energy(mass, velocity)
    assert jnp.all(ke.value >= 0)
    # Check resulting unit is energy
    assert u.dimension_of(ke) == u.dimension("energy")


@given(
    mass=ust.quantities(unit="kg", shape=(10,)),
    velocity=ust.quantities(unit="m/s", shape=(10,)),
)
def test_kinetic_energy_vectorized(mass, velocity):
    """Kinetic energy works with arrays."""
    ke = kinetic_energy(mass, velocity)
    assert ke.shape == (10,)
    assert jnp.all(ke.value >= 0)
```

### Custom Dimension Strategies

Create reusable strategies for specific physical dimensions:

```python
from hypothesis import strategies as st

import unxt as u
import unxt_hypothesis as ust

# Strategy for astronomical distances
astro_distances = ust.quantities(
    st.sampled_from(["pc", "kpc", "Mpc", "AU", "lyr"]), shape=st.just(())
)

# Strategy for velocities in astronomy
astro_velocities = ust.quantities(
    st.sampled_from(["km/s", "m/s", "pc/Myr"]), shape=st.just(())
)

# Strategy for masses in astronomy
astro_masses = ust.quantities(st.sampled_from(["Msun", "kg", "g"]), shape=st.just(()))


@given(distance=astro_distances, velocity=astro_velocities)
def test_astronomical_function(distance, velocity):
    """Test with astronomy-specific units."""
    time = distance / velocity
    assert u.dimension_of(time) == u.dimension("time")
```

## See Also

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [unxt Documentation](https://unxt.readthedocs.io/)
- [Property-Based Testing](https://hypothesis.works/articles/what-is-property-based-testing/)

## License

BSD 3-Clause License. See the
[LICENSE](https://github.com/GalacticDynamics/unxt/blob/main/LICENSE) file in
the main repository for details.

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.
