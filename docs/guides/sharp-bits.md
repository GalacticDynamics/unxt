# Unxt: The Sharp Bits

This guide covers common pitfalls and surprising behaviors when working with `unxt` quantities in JAX. Like JAX itself, `unxt` has some "sharp bits" — behaviors that might surprise you if you're coming from NumPy or non-JAX Python scientific computing.

```{tip}
If you're new to `unxt`, start with the [Quantity guide](./quantity) first.
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

Or use {func}`dataclasses.replace` (or {func}`dataclassish.replace`) for more complex updates:

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

**Why?** JAX requires pure functions for transformations like `jit` and `grad`. Immutability ensures your functions have no side effects.

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

### ⚠️ Angle conversions: `deg2rad`, `rad2deg`, and friends

`jnp.deg2rad`, `jnp.rad2deg`, `jnp.radians`, and `jnp.degrees` lower to a plain multiplication by a constant conversion factor (`x * pi/180` for `deg2rad`/`radians`, `x * 180/pi` for `rad2deg`/`degrees`). Under `quaxed`/`quax` that scales the _value_ but leaves the _unit label_ unchanged, so the quantity is silently mislabeled:

```python
import quaxed.numpy as jnp
import unxt as u

# ❌ scales the value but keeps 'deg' -> Quantity(3.14159, unit='deg')
_ = jnp.deg2rad(u.Q(180.0, "deg"))
```

Convert angles with `uconvert` (or the `.uconvert` method), which tracks units correctly:

```python
# ✅ converts degrees -> radians (value rescaled, unit label becomes 'rad')
_ = u.Q(180.0, "deg").uconvert("rad")

# ✅ converts radians -> degrees (value rescaled, unit label becomes 'deg')
_ = u.uconvert("deg", u.Q(3.14159, "rad"))
```

The NumPy entry points (`np.deg2rad(q)`, `np.rad2deg(q)`, ...) are handled correctly: they convert the angle and raise on a non-angle quantity.

### ⚠️ `jnp.where` adopts the quantity's unit for a raw-array branch

Selecting between a quantity and a **raw array** with `jnp.where` silently treats the raw array as being _in the quantity's unit_ — it does **not** reject the mix the way `jnp.concat` does:

```{code-block} python
>>> import quaxed.numpy as jnp
>>> import jax.numpy as jnp_raw
>>> import unxt as u

>>> cond = jnp_raw.asarray([True, False])
>>> q = u.Q([1.0, 2.0], "m")
>>> raw = jnp_raw.asarray([10.0, 20.0])
```

The raw `20.0` comes back as `20.0 m` — no error, no conversion:

```{code-block} python
>>> jnp.where(cond, q, raw)
Quantity(Array([ 1., 20.], dtype=float32), unit='m')
```

Contrast `jnp.concat`, which treats the raw array as dimensionless and rejects the incompatible mix:

```{code-block} python
>>> try:
...     jnp.concat([q, raw])
... except Exception as e:
...     print(type(e).__name__)
UnitConversionError
```

This inconsistency is **inherent**, not an oversight. JAX lowers both a user `where` _and_ its own masking operations (`triu`, `tril`, `trace`, and `where(mask, q, 0.0)`) to the same `select_n` primitive with a raw-array operand — and masking _relies_ on that raw zero-fill adopting the quantity's unit (filling with `0` should keep `m`, since zero is unit-agnostic). At the primitive level a genuine raw-data operand is indistinguishable from a masking zero-fill, so unxt cannot reject one without breaking the other.

Filling with a plain `0` is therefore fine and does the right thing:

```{code-block} python
>>> jnp.where(cond, q, 0.0)
Quantity(Array([1., 0.], dtype=float32), unit='m')
```

**Rule:** never rely on `jnp.where` to unit-check a raw-array branch. Convert the raw array to a `Quantity` with the unit you mean _before_ the `where`, so the units are checked explicitly:

```{code-block} python
>>> jnp.where(cond, q, u.Q(raw, "m"))
Quantity(Array([ 1., 20.], dtype=float32), unit='m')
```

Or reach for `unxt.experimental.where`, a strict alternative that **rejects** a raw-array branch outright (both branches must be quantities), so a bare array can never silently acquire a unit.

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

**Why it works:** The units are static on the Quantity PyTree. {mod}`unxt` can catch dimension mismatches during tracing.

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

**Key insight:** Dimensions are checked statically, but each unique combination of units creates a new compiled version.

### ℹ️ Note: `ParametricQuantity` multiplies pytree _types_ (not jit compilations)

Feeding `ParametricQuantity` (from the separate [`unxts.parametric`](../packages/unxts.parametric/index) package) of different dimensions into a jitted function does **not** add a recompilation per dimension — recompilation is driven by the **unit**, a static field for both classes. What the parametric class adds is a new Python class and pytree type per dimension. See [Parametric types multiply pytree types](../packages/unxts.parametric/sharp-bits) in the parametric guide for the full explanation.

## Mixing Quantity Types

### ❌ Problem: Confused by Quantity vs ParametricQuantity

Different quantity types have different guarantees:

```python
# What's the difference?
q1 = u.Q(5.0, "m")  # the default, lightweight Quantity
q2 = u.quantity.StaticQuantity(5.0, "m")  # static value, for constants
# ParametricQuantity (dimension in the type) lives in unxts.parametric
```

### ❌ Problem: Quantities are Dynamic

```python
import functools as ft


@ft.partial(jax.jit, static_argnames=("constant",))
def function(x, *, constant=u.Q(3.26, "lyr")):
    ...
```

### ✅ Solution: Choose the Right Type

**`Quantity`** — The default. A lightweight, non-parametric quantity that tracks units without encoding the physical dimension in its type:

```python
length = u.Q(5.0, "m")
time = u.Q(2.0, "s")
speed = length / time  # ✅ Fast; unit arithmetic without per-dimension classes
```

**`ParametricQuantity`** — Opt in when you want the physical dimension carried in the type (for dimension-specific `plum` dispatch and runtime dimension checking). It lives in the separate [`unxts.parametric`](../packages/unxts.parametric/index) package (`up.PQ`); see its [guide](../packages/unxts.parametric/quantity).

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

| Type | Use Case | Dimension in Type | Performance |
| --- | --- | --- | --- |
| `Quantity` | Default choice | ❌ None | Better |
| `ParametricQuantity` | Dimension-parametrized dispatch / runtime checking | ✅ Yes | Good (a distinct type per dimension) |
| `StaticQuantity` | Constants | ❌ None | Best (no tracer) |

`ParametricQuantity` lives in the separate [`unxts.parametric`](../packages/unxts.parametric/index) package; see its [guide](../packages/unxts.parametric/quantity) for construction, dimension checking, and dispatch.

## Mixing Astropy and `unxt` Quantities Under `jit`

### ❌ Problem: The Astropy Unit Disappears Inside `jit`

Arithmetic between an `astropy.units.Quantity` and a `unxt` quantity works **eagerly**, because astropy's `__array_ufunc__` handles the operation. Inside `jax.jit` it does not, and the failure is quiet:

Eagerly, astropy handles it and both units survive:

```{code-block} python
>>> import astropy.units as apyu
>>> import jax
>>> import unxt as u

>>> apy = apyu.Quantity(2.0, "km")
>>> q = u.Q(3.0, "m")

>>> apy * q
<Quantity 6. km m>
```

Under `jit`, `km` is silently dropped and the result claims `m`:

```{code-block} python
>>> jax.jit(lambda a, b: a * b)(apy, q)
Quantity(Array(6., dtype=float32), unit='m')
```

The magnitude survives but the unit does not, so `*` and `/` return a plausible-looking result that is wrong by the conversion factor — here a factor of 1000. `+` and `-` at least fail loudly:

```{code-block} python
>>> try:
...     jax.jit(lambda a, b: a + b)(apy, q)
... except apyu.UnitConversionError as e:
...     print(type(e).__name__)
UnitConversionError
```

This is not something `unxt` can intercept: `jax` converts the astropy `ndarray` subclass to a unitless tracer before any `unxt` code runs, so the unit is already gone by then. (Capturing the astropy quantity as a closure constant rather than passing it as an argument loses it too, by a different route.)

### ✅ Solution: Convert at the Boundary

Turn foreign quantities into `unxt` quantities _before_ they cross into a jitted function, with `u.Q.from_`:

```{code-block} python
>>> qa = u.Q.from_(apy)  # 2.0 km, now a unxt Quantity
>>> qa
Quantity(Array(2., dtype=float32), unit='km')
```

Both operators now behave, and agree with the eager result:

```{code-block} python
>>> jax.jit(lambda a, b: a * b)(qa, q)
Quantity(Array(6., dtype=float32), unit='km m')

>>> jax.jit(lambda a, b: a + b)(qa, q)
Quantity(Array(2.003, dtype=float32), unit='km')
```

As a rule: keep astropy quantities at the edges of your program and convert once on the way in. Mixed-library arithmetic working eagerly is not evidence that it will work under `jit`.

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

See the [Performance Guide](perf)

### ❌ Problem: Quantity is slower than Array

For most functions, Quantity input is slower than an Array. This is because Quantities are PyTrees that combine a value and a unit. When a PyTree passes through a {func}`jax.jit` boundary it is de-structured then re-structured. This process has an associated overhead.

```python
@jax.jit
@quax.quaxify
def func(x, y):
    return jnp.sum((x**3 - y**3) / (x**2 + y**2))


x, y = jnp.asarray([1, 2, 3]), jnp.asarray([4, 5, 6])
func(x, y)

# vs
qx, qy = u.Q(x, "m"), u.Q(y, "m")
func(qx, qy)
```

### ✅ Solution: Don't pass through the outermost `jax.jit` boundary

If the PyTree is formed within the jit context then all the nodes of the PyTree (the static parts) are constant-folded by JAX and will not contribute to the run-time, only the time for first compilation.

```python
@ft.partial(jax.jit, static_argnames=("usys",))
def func(x, y, *, usys):
    x = u.Q.from_(x, usys["length"])
    y = u.Q.from_(y, usys["length"])
    return quax.quaxify(jnp.sum)((x**3 - y**3) / (x**2 + y**2))


x, y = jnp.asarray([1, 2, 3]), jnp.asarray([4, 5, 6])
func(x, y, usys=u.unitsystems.si)
```

This only applies to the outer-most function. Nesting jitted and quaxified functions are fine. The outermost jit boundary handles the constant-folding.

## `ParametricQuantity` Equality: Arrays vs. `StaticValue`

A normal `ParametricQuantity` backed by a JAX array returns an element-wise boolean array from `==`, so it can't be a `jax.jit` `static_argnames` argument. Wrapping its value in a `StaticValue` makes `==` return a scalar `bool`, making the quantity hashable and usable as a static argument. This is a `ParametricQuantity` behavior (from the separate [`unxts.parametric`](../packages/unxts.parametric/index) package); see [Equality with `StaticValue`](../packages/unxts.parametric/sharp-bits) in the parametric guide.

## See Also

- [JAX Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Quantity Guide](./quantity)
- [Type Checking Guide](./type-checking)
- [Hypothesis strategies](../packages/unxts.hypothesis/index)
