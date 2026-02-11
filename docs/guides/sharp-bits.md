# Unxt: The Sharp Bits

This guide covers common pitfalls and surprising behaviors when working with
`unxt` quantities in JAX. Like JAX itself, `unxt` has some "sharp bits" —
behaviors that might surprise you if you're coming from NumPy or non-JAX Python
scientific computing.

```{tip}
If you're new to `unxt`, start with the [Quantity guide](quantity.md) first.
This guide assumes you're familiar with basic `unxt` usage.
```


## Pure Functions and Immutability

### ❌ Problem: Trying to Mutate Quantities

Coming from NumPy or Astropy, you might expect to modify quantities in place:

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

::::{tab-set}

:::{tab-item} dataclasses

```python
from dataclasses import replace

new_q = replace(q, value=q.value.at[0].set(5.0))
```

:::

:::{tab-item} dataclassish

```python
from dataclassish import replace

new_q = replace(q, value=q.value.at[0].set(5.0))
```

:::

::::

**Why?** JAX requires pure functions for transformations like `jit` and `grad`.
Immutability ensures your functions have no side effects.


## JAX Control Flow

### ❌ Problem: Control Flow on Quantity Values

JAX control flow requires special handling, independent of unit considerations:

```python
import jax


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

## Operations on Quantities

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

### ✅ Dimension Checking Works in JIT

Good news! Dimensions are checked inside JIT:

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

### ❌ Problem: Units Triggering Recompilation

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

### ✅ Solution: Use Consistent Units

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

## Mixing Quantity Types

### ❌ Problem: Confused by BareQuantity vs Quantity

Different quantity types have different guarantees:

```python
# What's the difference?
q1 = u.Q(5.0, "m")
q2 = u.quantity.BareQuantity(5.0, "m")
q3 = u.quantity.StaticQuantity(5.0, "m")
```

### ❌ Problem: Quantities are Dynamic

```python
import functools as ft


@ft.partial(jax.jit, static_argnames=("constant",))
def function(x, *, constant=u.Quantity(3.26, "lyr")):
    ...
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

```python
@ft.partial(jax.jit, static_argnames=("constant",))
def function(x, *, constant=u.StaticQuantity(3.26, "lyr")):
    ...
```

**When to use each:**

| Type | Use Case | Dimension Checking | Performance |
|------|----------|-------------------|-------------|
| `Quantity` | Default choice | ✅ Full | Good |
| `BareQuantity` | Trust your math | ❌ None | Better |
| `StaticQuantity` | Constants | ✅ Full | Best (no tracer) |

## Dimension Checking Overhead

### ❌ Problem: Slow Tests or Development

Dimension checking uses `beartype` for runtime validation, which can add overhead:

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

## Quantity as a PyTree: JAX flattening overhead

See the [Performance Guide](../notebooks/my_example.md)

### ❌ Problem: Quantity is slower than Array

For most functions, Quantity input is slower than an Array.
This is because Quantities are PyTrees that combine a value and a unit.
When a PyTree passes through a {func}`jax.jit` boundary it is de-structured then re-structured. This process has an associated overhead.

```python
@jax.jit
@quax.quaxify
def func(x, y):
    return jnp.sum((x**3 - y**3) / (x**2 + y**2))


x, y = jnp.asarray([1, 2, 3]), jnp.asarray([4, 5, 6])
func(x, y)

# vs
qx, qy = u.Q(x, "m"), u.Q(y, "m")
func(x, y)
```

### ✅ Solution: Don't pass through the outermost `jax.jit` boundary

If the PyTree is formed within the jit context than all the nodes
of the PyTree (the static parts) are constant-folded by JAX and will
not contribute to the run-time, only the time for first compilation.

```python
@ft.partial(jax.jit, static_argnames=("usys",))
def func(x, y, *, usys):
    x = u.Q.from_(x, usys["length"])
    y = u.Q.from_(y, usys["length"])
    return quax.quaxify(jnp.sum)((x**3 - y**3) / (x**2 + y**2))


x, y = jnp.asarray([1, 2, 3]), jnp.asarray([4, 5, 6])
func(x, y, usys=u.unitsystems.si)
```

This only applies to the outer-most function.
Nesting jitted and quaxified functions are fine.
The outermost jit boundary handles the constant-folding.


## See Also

- [JAX Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Quantity Guide](quantity.md)
- [Type Checking Guide](type-checking.md)
- [Testing Guide](../packages/unxt-hypothesis/testing-guide.md)
