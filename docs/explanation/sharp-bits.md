# The sharp bits

Like JAX itself, `unxt` has behaviours that surprise people arriving from NumPy or astropy. This page collects the ones that are _inherent_ — consequences of how units, JAX pytrees and JAX primitives fit together — and explains why each one is the way it is. Where there is a fix, it links to the guide that gives it.

If you are looking for a procedure rather than a reason, you probably want {doc}`../how-to/use-jax-functions` or {doc}`../how-to/optimize-performance`.

<!-- invisible-code-block: python
import jax
import unxt as u
-->

## Quantities are immutable, and that is not negotiable

`q[0] = ...` does not work. Quantities are frozen dataclasses, updated functionally through `.at[]` or `dataclasses.replace`.

This is not a stylistic choice inherited from Equinox. JAX transformations — `jit`, `grad`, `vmap` — require pure functions, and a mutable array threaded through a trace is a side effect the tracer cannot see. Immutability is what makes a `Quantity` safe to hand to any of them. See {doc}`../how-to/use-jax-functions` for the functional-update syntax.

## A dimensionful quantity will not silently become a bare array

Handing a `Quantity` to something that expects a plain array — `numpy.asarray`, or any library that calls it for you — raises rather than quietly dropping the unit:

```{code-block} python
>>> import numpy as np

>>> try:
...     np.asarray(u.Q([1.0, 2.0, 3.0], "m"))
... except Exception as e:
...     print(type(e).__name__)
UnitConversionError

```

This is deliberate, and it is the one sharp bit on this page that `unxt` chose rather than inherited. `__array__` has no way to ask the caller which unit it wants, so any answer it invents is a number whose meaning depends on information the caller never sees — the classic silent-unit-bug. Refusing is the only honest option.

A **dimensionless** quantity converts fine, because there is nothing to lose:

```{code-block} python
>>> np.asarray(u.Q([1.0, 2.0], ""))
array([1., 2.], dtype=float32)

```

The fix is to name the unit you mean, with `ustrip`:

```{code-block} python
>>> np.asarray(u.ustrip("m", u.Q([1.0, 2.0, 3.0], "m")))
array([1., 2., 3.], dtype=float32)

```

See {doc}`../how-to/convert-units`.

## Dimensions are checked inside `jit`, but units drive recompilation

Both halves of this follow from one fact: the `unit` field is **static** on the `Quantity` pytree. It lives in the aux data, not the leaves.

Because it is static, it is available at trace time, so `unxt` catches a dimension mismatch while tracing rather than producing a wrong answer at runtime. Adding a length to a time raises inside a jitted function, just as it does eagerly. Branching on `u.dimension_of(x)` inside a jitted function is likewise fine — the comparison resolves before any tracing of the branch not taken.

The same staticness means the unit is part of the `jit` cache key. A function called with metres and then with kilometres compiles **twice** — not because the dimension differs (it does not) but because the unit labels do:

```python
@jax.jit
def add_lengths(x, y):
    return x + y


add_lengths(u.Q(5.0, "m"), u.Q(3.0, "m"))  # compiles
add_lengths(u.Q(1.0, "km"), u.Q(2.0, "km"))  # recompiles
add_lengths(u.Q(5.0, "m"), u.Q(3.0, "km"))  # recompiles again
```

There is no way around this that keeps units in the type system, and it is rarely a problem in practice — programs tend to settle on a working unit. Where it does matter, standardise units before the jitted call.

## `deg2rad` and friends rescale the value but not the label

`jnp.deg2rad`, `jnp.rad2deg`, `jnp.radians` and `jnp.degrees` lower to a plain multiplication by a constant — `x * pi/180` for `deg2rad`. Under `quax` that scales the value and leaves the unit label untouched, so the result is silently mislabelled:

```python
import quaxed.numpy as jnp

# scales the value but keeps 'deg' -> Quantity(3.14159, unit='deg')
_ = jnp.deg2rad(u.Q(180.0, "deg"))
```

The primitive genuinely carries no unit information for `unxt` to act on, which is why this is a documented sharp edge rather than a bug with a fix pending. Use `uconvert` for angles — it is unit-aware by construction:

```python
u.Q(180.0, "deg").uconvert("rad")
u.uconvert("deg", u.Q(3.14159, "rad"))
```

The NumPy entry points (`np.deg2rad(q)`, `np.rad2deg(q)`) _are_ handled correctly: they convert the angle, and raise on a non-angle quantity.

## `jnp.where` lets a raw array adopt a unit

Selecting between a quantity and a **raw array** treats the raw array as being in the quantity's unit. It does not reject the mix the way `jnp.concat` does:

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

The inconsistency looks like an oversight and is not. JAX lowers a user `where` _and_ its own masking operations — `triu`, `tril`, `trace`, `where(mask, q, 0.0)` — to the same `select_n` primitive with a raw-array operand. Masking _relies_ on that raw zero-fill adopting the quantity's unit, since filling with `0` should keep `m`; zero is unit-agnostic. At the primitive level a genuine raw-data operand is indistinguishable from a masking zero-fill, so `unxt` cannot reject the one without breaking the other.

Filling with a plain `0` is therefore correct and does the right thing:

```{code-block} python
>>> jnp.where(cond, q, 0.0)
Quantity(Array([1., 0.], dtype=float32), unit='m')
```

The consequence for your code: never rely on `jnp.where` to unit-check a raw-array branch. Convert the raw array to a `Quantity` with the unit you mean, so the check happens explicitly —

```{code-block} python
>>> jnp.where(cond, q, u.Q(raw, "m"))
Quantity(Array([ 1., 20.], dtype=float32), unit='m')
```

— or use `unxt.experimental.where`, a strict alternative that requires both branches to be quantities.

## A quantity is a pytree, and pytrees cost something at a `jit` boundary

A `Quantity` is a pytree combining a value and a unit. Crossing a `jax.jit` boundary de-structures it and re-structures it on the way out, and that costs time — which is why a jitted function is measurably slower with `Quantity` inputs than with raw `Array` inputs.

The cost is _at the boundary_, not per operation. A quantity constructed **inside** a jitted context has its static parts constant-folded away by JAX, so it contributes only to compilation time, not to run time. That asymmetry is what makes the outer-wrapper pattern work: accept raw arrays at the outermost function, build quantities inside, and the unit handling compiles away entirely.

It is also a _fixed_ cost per outermost call. For a function processing a million-element array once it is irrelevant; for a scalar function called a million times it is not. {doc}`../how-to/optimize-performance` measures all of this.

## Astropy quantities lose their units inside `jit`

Mixing an `astropy.units.Quantity` with an `unxt` quantity works eagerly and fails silently under `jit`, because `jax` converts astropy's `ndarray` subclass to a unitless tracer before `unxt` sees it. This one has a clean fix — convert at the boundary — and it is covered in {doc}`../how-to/interoperate-with-astropy`.

## Which quantity class to reach for

`unxt` ships more than one quantity class, and the differences are about what goes into the _type_ rather than what the values do.

| Class | Dimension in the type | Cost | Reach for it when |
| --- | --- | --- | --- |
| `Quantity` | no | lowest | Always, unless one of the rows below applies. |
| `StaticQuantity` | no | lowest — never a tracer | The value is a compile-time constant. |
| `ParametricQuantity` | yes | a distinct class and pytree type per dimension | You need runtime dimension checking or dimension-specific `plum` dispatch. |

`ParametricQuantity` lives in the separate [`unxts.parametric`](../packages/unxts.parametric/index) package. A common worry is that feeding it quantities of different dimensions multiplies `jit` compilations — it does not, for the reason given above: recompilation is driven by the unit, which is static for either class. What it multiplies is Python classes and pytree node types. See {doc}`why-quantity-is-non-parametric` for the full argument, and [Parametric types multiply pytree types](../packages/unxts.parametric/sharp-bits) in the parametric package's own guide.

The equality semantics differ too: a `ParametricQuantity` backed by a JAX array returns an element-wise boolean array from `==` and so cannot be a `jax.jit` `static_argnames` argument, while wrapping its value in `StaticValue` makes `==` scalar and the quantity hashable. That is the same mechanism described in {doc}`equality-and-equivalence`.

## See also

- [JAX Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- {doc}`../how-to/use-jax-functions`, {doc}`../how-to/optimize-performance`
- {doc}`equality-and-equivalence`, {doc}`why-quantity-is-non-parametric`
