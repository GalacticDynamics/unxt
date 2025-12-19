# unxt-hypothesis

Hypothesis strategies for [unxt](https://github.com/GalacticDynamics/unxt) -
unitful quantities in JAX.

This package provides [Hypothesis](https://hypothesis.readthedocs.io/)
strategies for generating `unxt` objects for for property-based testing.

## Installation

```bash
uv add unxt-hypothesis
```

## Usage

```python
from hypothesis import given

import unxt_hypothesis as ust


@given(q=ust.quantities(unit="km/s"))
def test_quantity_property(q):
    # Test some property of quantities
    assert q.value is not None
    assert q.unit is not None


@given(u=ust.units("length"))
def test_unit_property(u):
    # Test some property of units
    assert u is not None


@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_unitsystem_property(sys):
    # Test some property of unit systems
    assert len(sys) == 4
```

### Custom strategies

You can customize the strategies:

```python
from hypothesis import strategies as st

import unxt as u
import unxt_hypothesis as ust


# Generate quantities with specific shapes
@given(q=ust.quantities(unit="m", shape=st.just((3, 3))))
def test_matrix_quantity(q):
    assert q.shape == (3, 3)


# Generate quantities with specific dimensions
@given(q=ust.quantities(unit=ust.units("length")))
def test_length_quantity(q):
    assert u.dimension_of(q) == u.dimension("length")
```

## API

### `quantities(draw, *, shape=None, dtype=None, unit=None)`

Generate random `Quantity` objects.

**Parameters:**

- `draw`: Hypothesis draw function
- `shape`: Strategy for array shapes (optional, defaults to small arrays)
- `dtype`: Strategy for array dtypes (optional, defaults to float32)
- `unit`: Strategy for unit strings (optional, defaults to common units)

**Returns:** A `unxt.Quantity` instance

### `units(draw, dimension=None, *, max_complexity=2, allow_non_integer_powers=False)`

Generate random `Unit` objects.

**Parameters:**

- `draw`: Hypothesis draw function
- `dimension`: Physical dimension (optional, e.g., `"length"`, `"velocity"`)
- `max_complexity`: Maximum complexity of compound units (default: 2)
- `allow_non_integer_powers`: Allow non-integer powers (default: False)

**Returns:** A `unxt.AbstractUnit` instance

### `unitsystems(*units)`

Generate random `UnitSystem` objects.

**Parameters:**

- `*units`: Variable number of unit specifications. Each can be:
  - `str`: Fixed unit string (e.g., `"m"`, `"kg"`)
  - `unxt.AbstractUnit`: Fixed unit object
  - Strategy: A Hypothesis strategy that generates units

**Returns:** A `unxt.AbstractUnitSystem` instance

**Example:**

```python
from hypothesis import given

import unxt_hypothesis as ust


# Fixed unit system
@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_mks_system(sys):
    assert len(sys) == 4


# Varying length unit, other units fixed
@given(sys=ust.unitsystems(ust.units("length"), "s", "kg", "rad"))
def test_varying_length(sys):
    assert len(sys) == 4
```

## Documentation

For comprehensive documentation, examples, and guides, see the
[unxt documentation](https://unxt.readthedocs.io/en/latest/guides/testing.html).

## License

BSD 3-Clause License. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.
