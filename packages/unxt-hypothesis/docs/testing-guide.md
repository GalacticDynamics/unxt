# Testing with Hypothesis

This guide shows how to use the `unxt-hypothesis` package for property-based
testing of code that uses `unxt` quantities.

## What is Property-Based Testing?

Property-based testing is a testing methodology where you specify properties
that should hold true for all inputs, and the testing framework (Hypothesis)
generates random test cases to verify those properties.

Instead of writing:

```python
import unxt as u


def test_addition():
    x, y = u.Q(5, "m"), u.Q(3, "m")
    assert x + y == y + x
```

You write:

```python
from hypothesis import given
import unxt_hypothesis as ust


@given(q1=ust.quantities("m"), q2=ust.quantities("m"))
def test_addition_commutative(q1, q2):
    """Addition of Quantity is commutative."""
    assert q1 + q2 == q2 + q1
```

Hypothesis will generate random test cases with different values, uncovering
edge cases you might not have thought of.

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

## Basic Examples

```python
from hypothesis import given, assume, strategies as st
import unxt_hypothesis as ust
import unxt as u
import jax
import jax.numpy as jnp
```

### Testing Quantity Properties

```python
@given(q=ust.quantities())  # any Quantity
def test_quantity_has_value_and_unit(q):
    """Every Quantity has a value and a unit."""
    assert q.value is not None
    assert q.unit is not None


@given(q=ust.quantities("m"))
def test_quantity_has_meter_units(q):
    """Scalar Quantity have ndim of 0."""
    assert q.unit == u.unit("m")


@given(q=ust.quantities(shape=(3,)))
def test_vector_quantity_has_correct_shape(q):
    """Vector Quantity have the expected shape."""
    assert q.shape == (3,)
```

### Testing Physical Laws

```python
@given(q1=ust.quantities("m", shape=()), q2=ust.quantities("m", shape=()))
def test_addition_commutative(q1, q2):
    """Addition of lengths is commutative."""
    assert jnp.allclose((q1 + q2).value, (q2 + q1).value)


@given(q=ust.quantities("m", shape=()))
def test_multiplication_identity(q):
    """Multiplying by 1 (dimensionless) preserves the quantity."""
    one = u.Q(1.0, "")
    result = q * one
    assert result.unit == q.unit
    assert jnp.allclose(result.value, q.value)


@given(mass=ust.quantities("kg", shape=()), velocity=ust.quantities("m/s", shape=()))
def test_kinetic_energy_units(mass, velocity):
    """Kinetic energy has correct units."""
    ke = 0.5 * mass * velocity**2
    # Energy has dimension of mass * length^2 / time^2
    assert u.dimension_of(ke) == "energy"
```

### Testing Unit Conversions

```python
@given(q=ust.quantities("m"))
def test_length_conversion_reversible(q):
    """Converting to another length unit and back is reversible."""
    in_km = q.uconvert("km")
    back_to_m = in_km.uconvert("m")
    assert jnp.allclose(q.value, back_to_m.value, rtol=1e-5)


@given(q=ust.quantities("m"))
def test_conversion_preserves_dimension(q):
    """Unit conversion preserves physical dimension."""
    converted = q.uconvert("km")
    assert u.dimension_of(converted) == u.dimension_of(q)
```

### Testing Array Operations

```python
@given(q=ust.quantities(shape=st.integers(1, 20)))
def test_sum_reduces_dimension(q):
    """Summing a quantity reduces one dimension."""
    total = jnp.sum(q)
    assert total.ndim == 0
    assert total.unit == q.unit


@given(q=ust.quantities(shape=(5, 5)))
def test_transpose_shape(q):
    """Transposing a matrix quantity swaps dimensions."""
    qt = jnp.transpose(q)
    assert qt.shape == (5, 5)
    assert qt.unit == q.unit


@given(q1=ust.quantities(shape=(3, 4)), q2=ust.quantities(shape=(4, 5)))
def test_matrix_multiplication_shape(q1, q2):
    """Matrix multiplication produces correct shape."""
    # Make dimensionless for matrix multiplication
    q1_dimensionless = q1.ustrip("")
    q2_dimensionless = q2.ustrip("")

    result = jnp.matmul(q1_dimensionless, q2_dimensionless)
    assert result.shape == (3, 5)
```

## Intermediate Examples

### Using Unit Strategies

```python
@given(length_unit=ust.units("length"), q=ust.quantities(ust.units("length")))
def test_all_lengths_convertible(length_unit, q):
    """All length Quantity can convert to any length unit."""
    converted = q.uconvert(length_unit)
    assert u.dimension_of(converted) == "length"


@given(velocity_unit=ust.units("velocity", max_complexity=1))
def test_simple_velocity_units(velocity_unit):
    """Simple velocity units have expected dimension."""
    assert u.dimension_of(velocity_unit) == "velocity"
    # Can decompose into length/time
    decomposed = u.unit(velocity_unit).decompose()
    assert "m" in str(decomposed) or "km" in str(decomposed)
    assert "s" in str(decomposed)
```

### Using Dimensions to Generate Units

You can pass a dimension directly to `quantities()` to generate quantities with
varying units of that dimension:

```python
@given(q=ust.quantities(u.dimension("length"), shape=3))
def test_length_quantities_from_dimension(q):
    """Test generating length quantities from dimension."""
    # Will create quantities with different length units (m, km, etc.)
    assert u.dimension_of(q) == u.dimension("length")
    assert q.shape == (3,)


@given(
    pos=ust.quantities(u.dimension("length"), shape=(3,)),
    vel=ust.quantities(u.dimension("velocity"), shape=(3,)),
)
def test_phase_space_from_dimensions(pos, vel):
    """Test creating phase space coordinates from dimensions."""
    # Units will vary, but dimensions are guaranteed
    assert u.dimension_of(pos) == u.dimension("length")
    assert u.dimension_of(vel) == u.dimension("velocity")
```

### Using Dimension Strategies

For even more flexibility, use strategies that generate different dimensions:

```python
@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("length"), u.dimension("mass")]),
        shape=(),
    )
)
def test_mixed_dimension_quantities(q):
    """Test with quantities of different dimensions."""
    dim = u.dimension_of(q)
    assert dim in (u.dimension("length"), u.dimension("mass"))


@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("velocity"), u.dimension("acceleration")]),
        shape=3,
    )
)
def test_kinematic_vectors(q):
    """Test vectors that could be velocity or acceleration."""
    dim = u.dimension_of(q)
    assert dim in (u.dimension("velocity"), u.dimension("acceleration"))
    assert q.shape == (3,)
```

### Constraining Value Ranges with Elements

The `elements` parameter allows you to control the range of values in generated
quantities. This is particularly useful for physical quantities with natural
constraints.

**Important:** When using custom `elements` strategies with `float32` dtype (the
default), always specify `width=32` in `st.floats()` to ensure compatibility
with JAX's array API.

#### Positive Distances

Distances are always non-negative:

```python
@given(
    q=ust.quantities(
        unit="kpc",
        shape=3,
        elements=st.floats(min_value=0, max_value=100, width=32),
    )
)
def test_galactic_positions(q):
    """Test 3D positions with reasonable galactic distances."""
    assert jnp.all(q.value >= 0)
    assert jnp.all(q.value <= 100)
    # Use position vector for some calculation
    distance = jnp.linalg.norm(q.value)
    assert distance >= 0
```

#### Longitude Angles (0 to 360°)

Longitude angles are typically constrained to [0, 360] degrees:

```python
@given(
    lon=ust.quantities(
        unit="deg",
        shape=(),
        elements=st.floats(min_value=0, max_value=360, allow_nan=False, width=32),
    )
)
def test_longitude_wrapping(lon):
    """Test longitude angle operations."""
    assert 0 <= lon.value <= 360
    # Test that wrapping works correctly
    wrapped = lon.value % 360
    assert 0 <= wrapped < 360
```

#### Latitude Angles (-90 to 90°)

Latitude angles are constrained to [-90, 90] degrees:

```python
@given(
    lat=ust.quantities(
        unit="deg",
        shape=100,
        elements=st.floats(min_value=-90, max_value=90, allow_nan=False, width=32),
    )
)
def test_latitude_constraints(lat):
    """Test latitude angle array."""
    assert jnp.all(lat.value >= -90)
    assert jnp.all(lat.value <= 90)
    # cos(lat) should always be positive
    assert jnp.all(jnp.cos(jnp.deg2rad(lat.value)) >= 0)
```

#### Physical Scales

Constrain values to physically meaningful ranges:

```python
@given(
    radius=ust.quantities(
        unit="m",
        shape=(),
        elements=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, width=32),
    )
)
def test_realistic_radius(radius):
    """Test with radii from millimeters to kilometers."""
    assert 1e-3 <= radius.value <= 1e3
    # Physical calculations
    area = 4 * 3.14159 * radius.value**2
    assert area > 0
```

### Using Dtype Strategies

The `dtype` parameter can also be a strategy, allowing you to test across
different numeric types:

```python
@given(
    q=ust.quantities(
        unit="m",
        dtype=st.sampled_from([jnp.float32, jnp.float64]),
        shape=(3,),
    )
)
def test_precision_independence(q):
    """Test that operations work with different precisions."""
    assert q.dtype in (jnp.float32, jnp.float64)
    # Operation should work regardless of dtype
    norm = jnp.linalg.norm(q.value)
    assert jnp.isfinite(norm)


@given(
    q=ust.quantities(
        unit="rad",
        dtype=st.sampled_from([jnp.float32, jnp.float64, jnp.complex64]),
        shape=(),
    )
)
def test_angle_with_various_dtypes(q):
    """Test that angle operations handle different dtypes."""
    # Even complex dtypes might be used in some contexts
    assert q.dtype in (jnp.float32, jnp.float64, jnp.complex64)
```

### Combining Strategies

You can combine dimension and dtype strategies for comprehensive testing:

```python
@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("length"), u.dimension("time")]),
        dtype=st.sampled_from([jnp.float32, jnp.float64]),
        shape=(5,),
    )
)
def test_combined_strategies(q):
    """Test with varying dimensions and dtypes."""
    # Both dimension and dtype will vary across test runs
    dim = u.dimension_of(q)
    assert dim in (u.dimension("length"), u.dimension("time"))
    assert q.dtype in (jnp.float32, jnp.float64)
    assert q.shape == (5,)
```

### Using Unit Systems

```python
@given(sys=ust.unitsystems("m", "s", "kg", "rad"))
def test_mks_system_consistency(sys):
    """MKS unit system has expected properties."""
    assert len(sys) == 4
    # Each dimension is represented
    dims = [str(u) for u in sys]
    assert "m" in dims
    assert "s" in dims
    assert "kg" in dims
    assert "rad" in dims


@given(
    sys=ust.unitsystems(ust.units("length"), "s", "kg", "rad"),
    q=ust.quantities(ust.units("length")),
)
def test_quantity_in_system_units(sys, q):
    """Quantities can be expressed in system units."""
    # The quantity should be expressible in the system's length unit
    length_unit = list(sys)[0]
    converted = q.uconvert(length_unit)
    assert u.dimension_of(converted) == "length"
```

## Advanced Patterns

### Testing Angle Quantities

Use the `quantities()` strategy with `quantity_cls=u.Angle` to generate angle
quantities. For wrapping angles to a specific range, use the `wrap_to()`
strategy.

```python
@given(angle=ust.quantities("rad", quantity_cls=u.Angle))
def test_angle_is_angle_type(angle):
    """Generated angles are Angle instances."""
    assert isinstance(angle, u.Angle)
    assert u.dimension_of(angle) == u.dimension("angle")


@given(angle=ust.quantities("deg", quantity_cls=u.Angle, shape=()))
def test_angle_in_degrees(angle):
    """Angles can be generated in different units."""
    assert angle.unit == u.unit("deg")
```

#### Wrapping Quantities to a Range

Use the `wrap_to()` strategy to wrap generated quantities to a specific [min,
max) range. This is particularly useful for angular quantities like longitude
and latitude:

```python
@given(
    lon=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle),
        min=u.Q(0, "deg"),
        max=u.Q(360, "deg"),
    )
)
def test_longitude_range(lon):
    """Longitude angles wrapped to [0, 360) degrees."""
    assert isinstance(lon, u.Angle)
    assert 0 <= lon.value < 360


@given(
    lat=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle, shape=()),
        min=u.Q(-90, "deg"),
        max=u.Q(90, "deg"),
    )
)
def test_latitude_range(lat):
    """Latitude angles wrapped to [-90, 90) degrees."""
    assert isinstance(lat, u.Angle)
    assert -90 <= lat.value < 90
```

The `wrap_to()` strategy can wrap any quantity, not just angles:

```python
@given(
    distance=ust.wrap_to(
        ust.quantities("kpc", shape=10), min=u.Q(0, "kpc"), max=u.Q(100, "kpc")
    )
)
def test_distance_range(distance):
    """Distances wrapped to [0, 100) kpc."""
    assert jnp.all(distance.value >= 0)
    assert jnp.all(distance.value < 100)
```

#### Using the quantity_cls Parameter

The `quantity_cls` parameter controls the type of quantity object created. By
default, it's `u.Quantity`, but you can specify `u.Angle` or other quantity
subclasses:

```python
# Generate Angle objects
@given(angle=ust.quantities("rad", quantity_cls=u.Angle, shape=3))
def test_angle_generation(angle):
    """Generate Angle instances using quantity_cls."""
    assert isinstance(angle, u.Angle)
    assert angle.unit == u.unit("rad")
    assert angle.shape == (3,)


# Generate plain Quantity objects (default)
@given(distance=ust.quantities("kpc", shape=()))
def test_distance_generation(distance):
    """Generate Quantity instances (default quantity_cls)."""
    assert isinstance(distance, u.Quantity)
    assert distance.unit == u.unit("kpc")


# Combine with other parameters
@given(
    angle=ust.quantities(
        "deg",
        quantity_cls=u.Angle,
        dtype=jnp.float64,
        elements=st.floats(min_value=0, max_value=360, width=64),
    )
)
def test_angle_with_constraints(angle):
    """Combine quantity_cls with dtype and element constraints."""
    assert isinstance(angle, u.Angle)
    assert angle.dtype == jnp.float64
    assert 0 <= angle.value <= 360
```

### Testing Coordinate Transformations

```python
@given(
    x=ust.quantities("m", shape=()),
    y=ust.quantities("m", shape=()),
    z=ust.quantities("m", shape=()),
)
def test_cartesian_to_spherical_radius(x, y, z):
    """Spherical radius is always non-negative."""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    assert jnp.all(r.value >= 0)
    assert u.dimension_of(r) == "length"


@given(
    r=ust.quantities("m", shape=()),
    theta=ust.quantities(
        "rad",
        quantity_cls=u.Angle,
        elements=st.floats(min_value=0, max_value=3.14159, width=32),
    ),
    phi=ust.quantities(
        "rad",
        quantity_cls=u.Angle,
        elements=st.floats(min_value=0, max_value=6.28318, width=32),
    ),
)
def test_spherical_to_cartesian_reversible(r, theta, phi):
    """Converting spherical to cartesian and back is reversible."""
    assume(r.value > 1e-10)  # Avoid numerical issues at origin

    # Convert to cartesian
    x = r * jnp.sin(theta.value) * jnp.cos(phi.value)
    y = r * jnp.sin(theta.value) * jnp.sin(phi.value)
    z = r * jnp.cos(theta.value)

    # Convert back
    r_back = jnp.sqrt(x**2 + y**2 + z**2)

    assert jnp.allclose(r.value, r_back.value, rtol=1e-5)
```

### Testing JAX Transformations

```python
@given(q=ust.quantities("m", shape=(10,)))
def test_vmap_preserves_units(q):
    """vmap over ust.quantities preserves units."""

    def square(x):
        return x**2

    # Apply vmap
    squared = jax.vmap(square)(q)

    assert squared.shape == q.shape
    assert squared.unit == q.unit**2


@given(q=ust.quantities("m", shape=()))
def test_grad_units(q):
    """Gradient of x^2 has correct units."""

    def f(x):
        return (x**2).ustrip("m^2")

    # Gradient w.r.t. a length should give length
    grad_f = jax.grad(f)
    result = grad_f(q.value)

    # Result should be 2*x, so same unit as input
    expected = 2 * q.value
    assert jnp.allclose(result, expected, rtol=1e-5)
```

### Filtering Invalid Cases

Use `hypothesis.assume()` to skip test cases that don't make sense:

```python
@given(
    numerator=ust.quantities("m", shape=()), denominator=ust.quantities("s", shape=())
)
def test_division_units(numerator, denominator):
    """Division produces correct units."""
    # Skip cases where denominator is too close to zero
    assume(jnp.abs(denominator.value) > 1e-10)

    result = numerator / denominator
    assert u.dimension_of(result) == "velocity"


@given(q=ust.quantities("m"))
def test_positive_values_only(q):
    """Test function that only works with positive values."""
    assume(jnp.all(q.value > 0))

    # Now safe to take logarithm
    log_q = jnp.log(q.value)
    assert jnp.all(jnp.isfinite(log_q))
```

## Best Practices

### 1. Start Simple

Begin with simple properties before testing complex behaviors:

```python
@given(q=ust.quantities())
def test_quantity_repr(q):
    """Quantities have a string representation."""
    assert repr(q) is not None
    assert "Quantity" in repr(q)
```

### 2. Use Appropriate Assumptions

Don't overuse `assume()` as it can slow down tests. Instead, generate
appropriate data:

```python
# Instead of this:
@given(q=ust.quantities("m"))
def test_bad(q):
    assume(q.value > 0)  # Will reject many cases
    # ...


# Do this:
@given(
    q=ust.quantities(
        unit="m",
        shape=st.just(()),
    )
)
def test_good(q):
    # Generate only positive values if needed
    q_positive = abs(q)
    # ...
```

### 3. Test Properties, Not Implementations

Focus on what should be true, not how it's computed:

```python
# Good - tests a property
@given(q1=ust.quantities("m"), q2=ust.quantities("m"))
def test_addition_commutative(q1, q2):
    assert jnp.allclose((q1 + q2).value, (q2 + q1).value)


# Less good - tests implementation
@given(q1=ust.quantities("m"), q2=ust.quantities("m"))
def test_addition_calls_add(q1, q2):
    with mock.patch("jax.numpy.add") as mock_add:
        q1 + q2
        mock_add.assert_called_once()  # Too implementation-specific
```

### 4. Use Descriptive Test Names

Make it clear what property is being tested:

```python
@given(q=ust.quantities("m"))
def test_length_conversion_to_km_preserves_magnitude_within_tolerance(q):
    """Converting meters to kilometers preserves the physical magnitude."""
    in_km = q.uconvert("km")
    assert jnp.allclose(q.value, in_km.value * 1000, rtol=1e-5)
```

### 5. Set Reasonable Limits

Use strategies wisely to avoid edge cases that aren't relevant:

```python
# Limit array sizes to reasonable values
@given(q=ust.quantities(shape=st.integers(1, 100), unit="m"))  # Not too large
def test_sum_preserves_units(q):
    total = jnp.sum(q)
    assert total.unit == q.unit
```

## Debugging Failed Tests

When Hypothesis finds a failing case, it will try to simplify it to a minimal
example and provide you with a `@example` decorator to reproduce it:

```
Falsifying example: test_addition_commutative(
    q1=Quantity(Array([0.], dtype=float32), unit='m'),
    q2=Quantity(Array([inf], dtype=float32), unit='m'),
)

You can reproduce this example by temporarily adding @example(q1=Quantity(...), q2=Quantity(...))
as a decorator on top of @given.
```

### Using `@example` to Reproduce Failures

The recommended approach is to use Hypothesis's `@example` decorator to force
the specific failing case to be tested. This ensures the example runs every time
and is compatible with Hypothesis's shrinking process:

```python
from hypothesis import given, example

import unxt as u
import unxt_hypothesis as ust


@given(q1=ust.quantities(), q2=ust.quantities())
@example(
    q1=u.Q(jnp.array([0.0], dtype=jnp.float32), "m"),
    q2=u.Q(jnp.array([jnp.inf], dtype=jnp.float32), "m"),
)
def test_addition_commutative(q1, q2):
    """Test that addition is commutative."""
    # This will run both the generated examples AND the specific failing case
    assert jnp.allclose((q1 + q2).value, (q2 + q1).value, equal_nan=True)
```

The `@example` decorator ensures that:

- The failing case is always tested, even if Hypothesis would otherwise miss it
- You can debug with the exact values that caused the failure
- The test remains property-based for other inputs

### Alternative: Standalone Debug Test

You can also copy the failing example into a separate test for debugging:

```python
def test_debug_specific_case():
    """Debug the specific failing case."""
    q1 = u.Q([0.0], "m")
    q2 = u.Q(jnp.inf, "m")

    # Add debugging
    print(f"q1 = {q1}")
    print(f"q2 = {q2}")

    result = q1 + q2
    print(f"result = {result}")
```

This approach is useful when you need to:

- Step through the code with a debugger
- Add extensive logging or inspection
- Temporarily isolate the failing case

For more on debugging strategies, see the Hypothesis documentation on
[Reproducing Failures](https://hypothesis.readthedocs.io/en/latest/reproducing.html)
and
[Testing Your Tests](https://hypothesis.readthedocs.io/en/latest/details.html#making-assumptions).

## See Also

- [Full API Reference](./api.md)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property-Based Testing Introduction](https://hypothesis.works/articles/what-is-property-based-testing/)
