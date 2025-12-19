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
strategies for generating random `Quantity`, `Unit`, and `UnitSystem` objects
for property-based testing.

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

This is a lower-level strategy that generates units by combining the base unit's
decomposed forms and adding cancelling factors. It's useful when you want to
generate various representations of the same physical dimension.

**Parameters:**

- `base` (str | apyu.UnitBase | SearchStrategy): Base unit (e.g., "m", "s",
  "kg") or a hypothesis strategy that generates such units.
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

- `dimension` (str | apyu.PhysicalType | None): The physical dimension of the
  unit. If None, generates units from various dimensions. Examples: `"length"`,
  `"velocity"`, `"energy"`.
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
- `unit` (str | apyu.UnitBase | st.SearchStrategy | None): Unit for the
  quantity. Can be:
  - `None` (default): Generates quantities with various common units
  - `str`: Specific unit string (e.g., `"m"`, `"km/s"`)
  - `apyu.UnitBase`: Specific unit object
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

- `*units` (str | apyu.UnitBase | st.SearchStrategy[apyu.UnitBase]): Variable
  number of unit specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `apyu.UnitBase`: Fixed unit object
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
