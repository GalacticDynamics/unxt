# Unxt: The Sharp Bits

This guide covers common pitfalls and surprising behaviors when working with
`unxt` quantities in JAX. Like JAX itself, `unxt` has some "sharp bits" —
behaviors that might surprise you if you're coming from NumPy or standard Python
scientific computing.

```{tip}
If you're new to `unxt`, start with the [Quantity guide](quantity.md) first.
This guide assumes you're familiar with basic `unxt` usage.
```


## Pure Functions and Immutability

### ❌ Problem: Trying to Mutate Quantities

Coming from NumPy, you might expect to modify quantities in place:

```python
import jax.numpy as jnp
import unxt as u

# This doesn't work as expected!
q = u.Q([1.0, 2.0, 3.0], "m")
try:
    q[0] = u.Q(5.0, "m")  # ❌ Error or doesn't modify in place
except Exception as e:
    print(f"Error: {e}")
```

### ✅ Solution: Use Functional Updates

Quantities are **immutable**. Use JAX's functional update methods:

```python
q = u.Q([1.0, 2.0, 3.0], "m")
new_q = q.at[0].set(u.Q(5.0, "m"))
```

Or use {func}`dataclasses.replace` (or {func}`dataclassish.replace`) for more
complex updates:

```python
from dataclasses import replace

new_q = replace(q, value=q.value.at[0].set(5.0))
```

**Why?** JAX requires pure functions for transformations like `jit` and `grad`.
Immutability ensures your functions have no side effects.


## JAX Transformations and Units

### ✅ Dimension Checking Works in JIT

Good news! Dimension checking **does** work inside JIT:

```python
import jax


@jax.jit
def add_quantities(x, y):
    return x + y


length = u.Q(5.0, "m")
time = u.Q(2.0, "s")

# ✅ This will raise an error at trace time
try:
    add_quantities(length, time)
except Exception as e:
    print(e)
```

**Why it works:** The units are static on the Quantity PyTree. {mod}`unxt` can
catch dimension mismatches during tracing.

### ❌ Problem: Unit Specialization and Recompilation

The catch is that functions compile separately for each **unit**, not just dimension:

```python
@jax.jit
def add_lengths(x: u.Q, y: u.Q):
    return x + y


# First call: compiles for meters
result1 = add_lengths(u.Q(5.0, "m"), u.Q(3.0, "m"))

# Second call: RECOMPILES for kilometers (different unit!)
result2 = add_lengths(u.Q(1.0, "km"), u.Q(2.0, "km"))

# Third call: RECOMPILES for mixed units (m and km)
result3 = add_lengths(u.Q(5.0, "m"), u.Q(3.0, "km"))
```

### ✅ Solution: Convert Units Before JIT

To avoid recompilation, standardize units before calling JIT functions:

```python
@jax.jit
def add_lengths_m(x: u.Q, y: u.Q):
    """Expects both inputs in meters."""
    return x + y


# Convert to standard units before JIT
length_km = u.Q(3.0, "km")
length_m_input = length_km.uconvert("m")

result = add_lengths_m(u.Q(5.0, "m"), length_m_input)
```

**Key insight:** Dimensions are checked statically, but each unique combination
of units creates a new compiled version.

### ❌ Problem: Control Flow on Quantity Values

JAX control flow requires special handling, even with units:

```python
@jax.jit
def bad_clamp(x: u.Q):
    # ❌ Python if statement with traced values doesn't work
    if x.value > 10.0:
        return u.Q(10.0, x.unit)
    else:
        return x
```

### ✅ Solution: Use JAX Control Flow Primitives

Use `jax.lax.cond` for traced values, or use `jax.numpy.where`:

```python
import jax.lax


@jax.jit
def good_clamp(x: u.Q):
    # ✅ Use jax.lax.cond for control flow
    return jax.lax.cond(x.value > 10.0, lambda x: u.Q(10.0, x.unit), lambda x: x, x)


# Or use jax.numpy.where for simple cases
@jax.jit
def clamp_with_where(x: u.Q):
    # ✅ jnp.where works with quantities
    import quaxed.numpy as jnp

    return jnp.where(x.value > 10.0, u.Q(10.0, x.unit), x)
```

**Note:** Checking dimensions in control flow is fine because dimensions are static:

```python
@jax.jit
def process(x: u.Q):
    # ✅ This works! Dimension check happens at trace time
    if u.dimension_of(x) == u.dimension("length"):
        return x * 2  # This branch traces
    else:
        return x  # This branch is never traced for length inputs
```


## Static vs Dynamic Quantities

### ❌ Problem: Using Regular Quantities for Constants

JAX tracers add overhead for values that never change:

```python
@jax.jit
def kinetic_energy(mass, velocity):
    # ❌ This creates a tracer every time, even though 0.5 is constant
    half = u.Q(0.5, "")
    return half * mass * velocity**2
```

### ✅ Solution: Use Static Quantities for Constants

`StaticQuantity` tells JAX the value won't change:

```python
from unxt.quantity import StaticQuantity

# Define once, outside the function
HALF = StaticQuantity(0.5, "")


@jax.jit
def kinetic_energy(mass, velocity):
    # ✅ JAX knows this is constant, no tracer overhead
    return HALF * mass * velocity**2
```

**When to use `StaticQuantity`:**

- Physical constants (G, c, ℏ)
- Conversion factors
- Fixed parameters in calculations

**When to use regular `Quantity`:**

- Input data
- Intermediate results
- Anything that varies


## Dimension Checking Overhead

### ❌ Problem: Slow Tests or Development

Dimension checking uses `beartype` for runtime validation, which adds overhead:

```python
from hypothesis import given
import unxt_hypothesis as ust


@given(q=ust.quantities("m"))  # During dev/testing, this can be slow
def test_something(q):
    result = complex_calculation(q)  # Slow with type checking
    assert result.unit == u.unit("m/s")
```

### ✅ Solution: Control Runtime Type Checking

Set the environment variable to control checking:

```bash
# Disable for production (faster)
export UNXT_ENABLE_RUNTIME_TYPECHECKING=False

# Enable for testing (safer)
export UNXT_ENABLE_RUNTIME_TYPECHECKING=beartype.beartype
```

Or in code:

```python
import os

# Fast mode for production
os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "False"

# Safe mode for testing
os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
```

**Default:** Runtime checking is `False` unless you're running tests.


## Array Operations and Unit Preservation

### ❌ Problem: Operating on Quantities with JAX Functions

Most direct JAX operations don't work:

```python
import jax.numpy as jnp

q = u.Q([1.0, 2.0, 3.0], "m")

# ❌ These might not preserve units as expected
try:
    jnp.concatenate([q, q])
except Exception as e:
    print(f"Error: {e}")
```

### ✅ Solution: Use Quaxified Functions

Use `quaxed` for pre-quaxified JAX functions that handle units:

```python
import quaxed.numpy as jnp  # Note: quaxed, not jax

q = u.Q([1.0, 2.0, 3.0], "m")

# ✅ These preserve quantities correctly
result = jnp.concat([q, q])  # Still Quantity
result = jnp.stack([q, q])  # Still Quantity
```

**General rule:** Import from `quaxed` when working with `unxt` quantities:

```python
# ✅ Do this
import quaxed.numpy as jnp
from quaxed import lax
from quaxed.scipy import special

# ❌ Not this (unless you manually quaxify)
import jax.numpy as jnp
```

**Alternative:** You can also quaxify individual functions instead of using `quaxed`:

```python
import jax.numpy as jnp
import quax

# Quaxify a specific function
quaxified_sum = quax.quaxify(jnp.sum)

positions = u.Q([1.0, 2.0, 3.0], "m")
total = quaxified_sum(positions)  # Preserves units


# Or use as a decorator
@quax.quaxify
def my_function(x):
    return jnp.sum(x**2)


result = my_function(positions)  # Works with quantities
```


## Angle Wrapping Surprises

### ❌ Problem: Unexpected Angle Values

Angle wrapping can produce surprising results if you're not careful:

```python
angle1 = u.Angle(350.0, "deg")
angle2 = u.Angle(20.0, "deg")

# What do you expect?
result = angle1 + angle2  # ???
```

### ✅ Solution: Understand Wrapping Behavior

Angles have optional wrapping bounds:

```python
# Without wrapping: arithmetic works normally
angle1 = u.Angle(350.0, "deg")  # No wrapping
angle2 = u.Angle(20.0, "deg")
result = angle1 + angle2  # Angle(370.0, 'deg')

# With wrapping: results wrap to range
from unxt.quantity import wrap_to

angle1 = wrap_to(u.Angle(350.0, "deg"), u.Q(0.0, "deg"), u.Q(360.0, "deg"))
angle2 = wrap_to(u.Angle(20.0, "deg"), u.Q(0.0, "deg"), u.Q(360.0, "deg"))
result = angle1 + angle2  # Angle(10.0, 'deg')  # Wrapped!
```

**Best practice:** Be explicit about whether you want wrapping:

```python
# For phase angles, use wrapping
phase = wrap_to(u.Angle(185, "deg"), u.Q(-jnp.pi, "rad"), u.Q(jnp.pi, "rad"))

# For cumulative rotation, don't wrap
total_rotation = u.Angle(0.0, "rad")  # Can exceed 2π
```


## Mixing Quantity Types

### ❌ Problem: Confused by BareQuantity vs Quantity

Different quantity types have different guarantees:

```python
# What's the difference?
q1 = u.Q(5.0, "m")
q2 = u.quantity.BareQuantity(5.0, "m")
q3 = u.quantity.StaticQuantity(5.0, "m")
```

### ✅ Solution: Choose the Right Type

**`Quantity`** — Standard choice with full dimension checking:

```python
length = u.Q(5.0, "m")
time = u.Q(2.0, "s")
speed = length / time  # ✅ Creates Quantity with correct dimension
```

**`BareQuantity`** — No dimension checking, just unit tracking:

```python
# Use when you need raw speed, trust your dimensions
length = u.quantity.BareQuantity(5.0, "m")
time = u.quantity.BareQuantity(2.0, "s")
speed = length / time  # Faster, but no dimension validation
```

**`StaticQuantity`** — For compile-time constants:

```python
# Use for constants that won't change
G = u.quantity.StaticQuantity(6.674e-11, "m^3 kg^-1 s^-2")
```

**When to use each:**

| Type | Use Case | Dimension Checking | Performance |
|------|----------|-------------------|-------------|
| `Quantity` | Default choice | ✅ Full | Good |
| `BareQuantity` | Trust your math | ❌ None | Better |
| `StaticQuantity` | Constants | ✅ Full | Best (no tracer) |


## Unit System Conversions

### ❌ Problem: Implicit Unit System Mixing

Different unit systems can lead to subtle bugs:

```python
# Galactic dynamics mixing SI and galactic units
distance_kpc = u.Q(8.5, "kpc")
velocity_si = u.Q(220.0, "km/s")

# ❌ This works but might not be what you want
time = distance_kpc / velocity_si  # Mixed units!
```

### ✅ Solution: Use Unit Systems Explicitly

Define and stick to a unit system:

```python
from unxt.unitsystems import galactic

# Define your unit system
usys = galactic

# Convert consistently
distance = u.Q(8.5, "kpc").uconvert(usys["length"])
velocity = u.Q(220.0, "km/s").uconvert(usys["speed"])
time = distance / velocity  # Now in consistent units


# Or convert everything upfront
def to_galactic(q):
    """Convert quantity to galactic units."""
    return q.uconvert(usys[u.dimension_of(q)])


distance = to_galactic(u.Q(8.5, "kpc"))
velocity = to_galactic(u.Q(220.0, "km/s"))
```


## Performance Tips

### Use JAX Transformations Correctly

```python
# ❌ Slow: Creating quantities in hot loop
@jax.jit
def slow_function(values):
    result = 0
    for v in values:
        result += u.Q(v, "m")  # Creates new quantity each iteration
    return result


# ✅ Fast: Create quantity once
@jax.jit
def fast_function(values):
    quantities = u.Q(values, "m")  # Single creation
    return jnp.sum(quantities)
```

### Minimize Unit Conversions

```python
# ❌ Slow: Converting repeatedly
def process_data(positions_km):
    result = u.Q(0.0, "m")
    for pos_km in positions_km:
        pos_m = pos_km.uconvert("m")  # Conversion in loop
        result += pos_m
    return result


# ✅ Fast: Convert once
def process_data(positions_km):
    positions_m = positions_km.uconvert("m")  # Single conversion
    return jnp.sum(positions_m)
```


## Summary: Common Patterns

### ✅ Do This

```python
# Use quaxed imports
import quaxed.numpy as jnp

# Use floats for measurements
distance = u.Q(5.0, "m")

# Static quantities for constants
G = u.quantity.StaticQuantity(6.674e-11, "m^3 kg^-1 s^-2")


# Strip units at boundaries for functions that need raw arrays
def plot(quantity):
    plt.plot(quantity.ustrip("m"))


# Convert once, use many times
data_km = u.Q([1.0, 2.0, 3.0], "km")
data_m = data_km.uconvert("m")
```


## See Also

- [JAX Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Quantity Guide](quantity.md)
- [Type Checking Guide](type-checking.md)
- [Testing Guide](../packages/unxt-hypothesis/testing-guide.md)
