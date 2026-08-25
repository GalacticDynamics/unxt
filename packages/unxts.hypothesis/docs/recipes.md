# How to combine strategies

This guide covers the composition patterns that come up once the individual strategies are not enough on their own: building strategies out of other strategies, driving a unitful function under test, and narrowing generation to a domain. For the strategies themselves, see [Strategies](strategies); for writing the tests around them, see [How to write property-based tests](testing-guide).

## Combining Strategies

The strategies are designed to work together seamlessly:

```python
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust


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

## Testing Unitful Functions

Here's a complete example of using these strategies to test a physics function:

```python
import jax.numpy as jnp
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust


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

## Custom Dimension Strategies

Create reusable strategies for specific physical dimensions:

```python
from hypothesis import strategies as st

import unxt as u
import unxts.hypothesis as ust

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
