# unxt-hypothesis

Hypothesis strategies for property-based testing with
[unxt](https://github.com/GalacticDynamics/unxt).

This package provides [Hypothesis](https://hypothesis.readthedocs.io/)
strategies for generating random `Quantity`, `Unit`, and `UnitSystem` objects
for property-based testing.

## Quick Start

```python
from hypothesis import given

import unxt as u
import unxt_hypothesis as ust


@given(q=ust.quantities(unit="km/s"))
def test_quantity_property(q):
    """Test that all quantities have value and unit."""
    assert q.value is not None
    assert q.unit is not None


@given(u=ust.units("length"))
def test_unit_property(u):
    """Test that units can be converted to strings."""
    assert str(u) is not None


@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_unitsystem_property(sys):
    """Test that unit systems have expected base units."""
    assert len(sys) == 4
```

## Strategies

### `derived_units(base, *, integer_powers=True, max_complexity=3)`

Generate units that are dimensionally equivalent to a given base unit.

**Parameters:**

- `base` (str | apyu.UnitBase | SearchStrategy): Base unit (e.g., "m", "s",
  "kg") or a hypothesis strategy that generates such units.
- `integer_powers` (bool): If True, only generate units with integer powers of
  base units (default: True).
- `max_complexity` (int): Maximum number of additional base unit factors to
  combine (default: 3).

**Returns:** `unxt.AbstractUnit`

### `units(dimension=None, *, max_complexity=2, allow_non_integer_powers=False)`

Generate random `Unit` objects from astropy.

**Parameters:**

- `dimension` (str | apyu.PhysicalType | None): The physical dimension of the
  unit. If None, generates units from various dimensions. Examples: `"length"`,
  `"velocity"`, `"energy"`.
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
- `unit` (str | apyu.UnitBase | st.SearchStrategy | None): Unit for the
  quantity. Can be:
  - `None` (default): Generates quantities with various common units
  - `str`: Specific unit string (e.g., `"m"`, `"km/s"`)
  - `apyu.UnitBase`: Specific unit object
  - Strategy: A Hypothesis strategy that generates units (e.g., from `units()`)

**Returns:** `unxt.Quantity`

### `unitsystems(*units)`

Generate random `UnitSystem` objects.

**Parameters:**

- `*units` (str | apyu.UnitBase | st.SearchStrategy[apyu.UnitBase]): Variable
  number of unit specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `apyu.UnitBase`: Fixed unit object
  - Strategy: A Hypothesis strategy that generates units

**Returns:** `unxt.AbstractUnitSystem`

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
