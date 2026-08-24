# How to use JAX functions with quantities

JAX functions accept JAX arrays. A `Quantity` is not one, so a raw JAX call rejects it:

```{code-block} python
>>> import jax.numpy as jnp
>>> import unxt as u

>>> x = u.Q([1., 2., 3.], "m")
>>> y = u.Q([4., 5., 6.], "m")

>>> try: jnp.square(x)
... except TypeError: print("not a pure JAX array")
not a pure JAX array

```

[`quax`](https://docs.kidger.site/quax/) is what bridges the gap. There are two ways to apply it, and they differ only in how much typing you do.

## Quaxify your own function

Decorate the outermost function with [`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify). Every JAX call _inside_ it then accepts quantities — you keep writing normal JAX:

```{code-block} python
:emphasize-lines: 3

>>> from quax import quaxify

>>> @quaxify
... def func(x, y):
...     return jnp.square(x) + jnp.multiply(x, y)  # normal JAX

>>> func(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

Use this when the unit-aware region of your program has a clear entry point, or when you need to pass a quantity into third-party code you do not control.

## Or import the pre-quaxified namespace

[`quaxed`][quaxed] is a drop-in replacement for much of JAX with `quaxify` already applied. If you are writing new code against `unxt`, importing it is less ceremony:

```{code-block} python
>>> import quaxed.numpy as jnp

>>> jnp.square(x) + jnp.multiply(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

The same holds for the rest of the namespace — `from quaxed import lax`, `from quaxed.scipy import special`. `quaxed` is entirely optional; it and manual `quaxify` can be mixed freely.

:::{attention}

`Quantity` should support **all** JAX functions. If you find one that does not, please open an issue on the [GitHub repository](https://github.com/GalacticDynamics/unxt).

:::

## Take care with functions that do not track units

Two cases are worth knowing before you hit them. `jnp.deg2rad` and friends rescale the value but leave the unit label alone, and `jnp.where` lets a raw array silently adopt a quantity's unit. Both are covered — with the reasons — in {doc}`../explanation/sharp-bits`.

## JIT it

`jax.jit` works through quantities without any extra step:

```{code-block} python
>>> from jax import jit

>>> jitted_func = jit(func)
>>> jitted_func(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

Note that a jitted function specializes per _unit_, not per dimension: calling it with metres and then with kilometres compiles twice. If that matters, convert to a common unit at the boundary — see {doc}`optimize-performance`.

## Differentiate it

Autodiff works the same way. Either wrap the JAX transform in `quaxify`:

```{code-block} python
>>> import jax

>>> def f(x: u.Q["length"], t: u.Q["time"]) -> u.Q["diffusivity"]:
...    return jnp.square(x) / t

>>> xq = u.Q(1.0, "m")
>>> tq = u.Q(4.0, "s")

>>> grad_f = quaxify(jax.grad(f))
>>> grad_f(xq, tq)
Quantity(Array(0.5, dtype=float32...), unit='m / s')

```

or use `quaxed`'s pre-wrapped transforms:

```{code-block} python
>>> import quaxed as qjax

>>> qjax.grad(f)(xq, tq)
Quantity(Array(0.5, dtype=float32...), unit='m / s')

>>> qjax.jacfwd(f)(xq, tq)
Quantity(Array(0.5, dtype=float32...), unit='m / s')

>>> qjax.hessian(f)(xq, tq)
Quantity(Array(0.5, dtype=float32...), unit='1 / s')

```

## Update arrays functionally

Quantities are immutable, as JAX requires. In-place assignment does not work; use the `.at[]` syntax, which converts the incoming value for you:

```{code-block} python
>>> q = u.Q([1., 2, 3, 4], "m")
>>> q.at[2].set(u.Q(30.1, "cm"))
Quantity(Array([1.   , 2.   , 0.301, 4.   ], dtype=float32), unit='m')

```

For structural edits, `dataclasses.replace` (or `dataclassish.replace`) works on a quantity like any other dataclass:

```{code-block} python
>>> from dataclasses import replace
>>> replace(q, value=q.value.at[0].set(5.0))
Quantity(Array([5., 2., 3., 4.], dtype=float32), unit='m')

```

## Branch on values with JAX control flow

A Python `if` on a traced value fails inside `jit`, exactly as it does for plain JAX arrays. Use `jax.lax.cond` or `jnp.where`:

```{code-block} python
>>> import quaxed.numpy as qnp

>>> @jax.jit
... def clamp(x):
...     return qnp.where(x > u.Q(10.0, "m"), u.Q(10.0, "m"), x)

>>> clamp(u.Q([5.0, 50.0], "m"))
Quantity(Array([ 5., 10.], dtype=float32), unit='m')

```

Branching on a _dimension_ is a different matter and is fine: dimensions are static and resolve at trace time, so both `u.dimension_of(x)` comparisons and the branch they select are decided before any tracing of the untaken path.

## See also

- {doc}`../explanation/sharp-bits` — where units and JAX surprise you, and why.
- {doc}`optimize-performance` — keeping the wrapper overhead off your hot path.
- {doc}`../reference/quantity` — the full `Quantity` surface.

[quaxed]: https://quaxed.readthedocs.io/en/latest/
