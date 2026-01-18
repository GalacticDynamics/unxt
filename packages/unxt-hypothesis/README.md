# unxt-hypothesis

Hypothesis strategies for property-based testing with
[unxt](https://github.com/GalacticDynamics/unxt).

This package provides [Hypothesis](https://hypothesis.readthedocs.io/)
strategies for generating random `Quantity`, `Angle`, `Unit`, `Dimension`, and
`UnitSystem` objects for property-based testing.

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

Generate a named physical dimension from Astropy's physical type catalogue.
Returns `u.AbstractDimension`. Useful for dimension-parameterized tests and for
feeding into `units()` and `quantities()`.

**Examples:**

```python
from hypothesis import given
import unxt as u
import unxt_hypothesis as ust


# Generate any named physical dimension
@given(dim=ust.named_dimensions())
def test_named_dimension(dim):
    assert isinstance(dim, u.AbstractDimension)


# Create units from any named dimension
@given(unit=ust.units(ust.named_dimensions()))
def test_any_unit(unit):
    assert isinstance(unit, u.AbstractUnit)
    assert u.dimension_of(unit) in [u.dimension(name) for name in ust.DIMENSION_NAMES]


# Use with quantities for generic physics tests
@given(q=ust.quantities(unit=ust.units(ust.named_dimensions())))
def test_quantity_any_dimension(q):
    assert isinstance(q, u.Quantity)
    assert u.dimension_of(q) in [u.dimension(name) for name in ust.DIMENSION_NAMES]
```

See also: `ust.DIMENSION_NAMES` for the curated list of physical type names.

### `derived_units(base, *, integer_powers=True, max_complexity=3)`

Generate units that are dimensionally equivalent to a given base unit.

**Parameters:**

- `base` (str | Unit | SearchStrategy): Base unit (e.g., "m", "s", "kg") or a
  hypothesis strategy that generates such units.
- `integer_powers` (bool): If True, only generate units with integer powers of
  base units (default: True).
- `max_complexity` (int): Maximum number of additional base unit factors to
  combine (default: 3).

**Returns:** `unxt.AbstractUnit`

### `units(dimension=None, *, max_complexity=2, allow_non_integer_powers=False)`

Generate random `Unit` objects from astropy.

**Parameters:**

- `dimension` (str | Dimension | SearchStrategy | None): The physical dimension
  of the unit. If None, generates units from various dimensions. Examples:
  `"length"`, `"velocity"`, `"energy"`.
- `max_complexity` (int): Maximum complexity of compound units (default: 2).
- `allow_non_integer_powers` (bool): Whether to allow non-integer powers in
  units (default: False).

**Returns:** `unxt.AbstractUnit`

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

### `unitsystems(*units)`

Generate random `UnitSystem` objects.

**Parameters:**

- `*units` (str | Unit | st.SearchStrategy[Unit]): Variable number of unit
  specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `Unit`: Fixed unit object
  - Strategy: A Hypothesis strategy that generates units

**Returns:** `unxt.AbstractUnitSystem`

### `angles(*, wrap_to=None, **kwargs)`

Generate random `Angle` objects with optional wrapping bounds.

**Parameters:**

- `wrap_to` (tuple | st.SearchStrategy | None): Wrapping bounds for the angle.
  Can be `None` (no wrapping), a `(min, max)` tuple of quantities, or a
  strategy.
- `**kwargs`: Additional keyword arguments passed to `quantities()` (e.g.,
  `dtype`, `shape`, `elements`, `unique`).

**Returns:** `unxt.Angle`

**Example:**

```python
from hypothesis import given
import unxt as u
import unxt_hypothesis as ust


@given(angle=ust.angles())
def test_any_angle(angle):
    assert isinstance(angle, u.Angle)
    assert u.dimension_of(angle) == u.dimension("angle")


@given(angle=ust.angles(wrap_to=(u.Q(0, "deg"), u.Q(360, "deg"))))
def test_wrapped_angle(angle):
    assert isinstance(angle, u.Angle)
    assert angle.wrap_to is not None
```

### `wrap_to(quantity, min, max)`

Generate wrapped quantities by constraining values to a specified range.

**Parameters:**

- `quantity` (u.AbstractQuantity | st.SearchStrategy): Quantity or strategy that
  generates the base quantity to wrap.
- `min` (u.AbstractQuantity | st.SearchStrategy): Minimum value (inclusive).
- `max` (u.AbstractQuantity | st.SearchStrategy): Maximum value (exclusive).

**Returns:** `unxt.AbstractQuantity`

**Example:**

```python
from hypothesis import given
import unxt as u
import unxt_hypothesis as ust


@given(
    angle=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle),
        min=u.Q(0, "deg"),
        max=u.Q(360, "deg"),
    )
)
def test_wrapped_angle(angle):
    assert 0 <= angle.value < 360
```

Note: The `angles()` strategy provides a more convenient interface for
generating wrapped angles.

## Type Strategy Registration

The package automatically registers type strategies for Hypothesis's
`st.from_type()` function, enabling automatic strategy generation for unxt
types:

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


@given(q=st.from_type(u.Quantity))
def test_quantity_class_via_from_type(q):
    """Test Quantity instances generated via st.from_type()."""
    assert isinstance(q, u.Quantity)


@given(bq=st.from_type(u.quantity.BareQuantity))
def test_bare_quantity_via_from_type(bq):
    """Test BareQuantity instances generated via st.from_type()."""
    assert isinstance(bq, u.quantity.BareQuantity)


@given(sq=st.from_type(u.quantity.StaticQuantity))
def test_static_quantity_via_from_type(sq):
    """Test StaticQuantity instances generated via st.from_type()."""
    assert isinstance(sq, u.quantity.StaticQuantity)
    # StaticQuantity uses StaticValue wrapper
    assert isinstance(sq.value, u.quantity.StaticValue)


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

### Registered Types

The following types are automatically registered:

- `u.AbstractQuantity` → generates `Quantity` instances
- `u.Quantity` → generates `Quantity` instances with dimension checking
- `u.quantity.BareQuantity` → generates `BareQuantity` instances (no dimension
  checking)
- `u.quantity.StaticQuantity` → generates `StaticQuantity` instances with
  `StaticValue` wrapper (for non-traced values)
- `u.Angle` → generates `Angle` instances with angle dimension
- `u.AbstractUnitSystem` → generates unit systems

This integration allows you to use type annotations directly in your tests
without explicitly importing the strategy functions, making tests more concise
and easier to read.

## Examples

### Generate quantities with specific shapes

```python
from hypothesis import given, strategies as st

import unxt_hypothesis as ust


@given(q=ust.quantities(shape=(3, 3)))
def test_matrix_quantity(q):
    assert q.shape == (3, 3)


@given(q=ust.quantities(shape=()))
def test_scalar_quantity(q):
    assert q.ndim == 0
```

### Generate quantities with specific dimensions

```python
from hypothesis import given

import unxt as u
import unxt_hypothesis as ust


@given(q=ust.quantities(unit=ust.units("length")))
def test_length_quantity(q):
    assert u.dimension_of(q) == u.dimension("length")


@given(q=ust.quantities(unit=ust.units("energy")))
def test_energy_quantity(q):
    assert u.dimension_of(q) == u.dimension("energy")
```

### Testing Unitful Functions

Here's a complete example of using these strategies to test a physics function:

```python
import jax.numpy as jnp
from hypothesis import given

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

## Documentation

For full documentation and advanced examples, see:

- [unxt-hypothesis Documentation](https://unxt.readthedocs.io/en/latest/packages/unxt-hypothesis/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [unxt Documentation](https://unxt.readthedocs.io/)

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.

## Documentation

For comprehensive documentation, examples, and guides, see the
[unxt documentation](https://unxt.readthedocs.io/en/latest/guides/testing.html).

## License

BSD 3-Clause License. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.
