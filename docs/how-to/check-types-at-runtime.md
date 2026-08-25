# How to check quantity types at runtime

[jaxtyping-link]: https://pypi.org/project/jaxtyping/

A `Quantity` carries its **dtype** and **shape** in its type, so [`jaxtyping`][jaxtyping-link] annotations work on it directly and are checked both statically and at runtime.

```{code-block} python

from jaxtyping import Float

import unxt as u

def velocity(
    x: Float[u.Quantity, "N"],
    t: Float[u.Quantity, "N"],
) -> Float[u.Quantity, "N"]:
    return x / t

```

Annotations alone are inert — something has to enforce them. There are two ways to turn enforcement on.

## Turn it on for a single function

To check one function without touching the rest of your program, decorate it with `jaxtyping`'s [`jaxtyped`](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.jaxtyped) and a typechecker:

```{code-block} python
>>> from jaxtyping import Shaped, jaxtyped
>>> from beartype import beartype as typechecker

>>> import unxt as u

>>> @jaxtyped(typechecker=typechecker)
... def velocity(
...     x: Shaped[u.Quantity, "N"],
...     t: Shaped[u.Quantity, "N"],
... ) -> Shaped[u.Quantity, "N"]:
...     return x / t

>>> x = u.Q([2.], "m")
>>> t = u.Q([1.], "s")

>>> velocity(x, t)
Quantity(Array([2.], dtype=float32), unit='m / s')

```

The check earns its keep when an argument violates the annotation. Both parameters above share the axis name `"N"`, so mismatched shapes are rejected before the body runs:

```{code-block} python
>>> x2 = u.Q([2.0, 3.0], "m")  # shape (2,)
>>> t2 = u.Q([1.0], "s")       # shape (1,)

>>> try:
...     velocity(x2, t2)
... except Exception as e:
...     print(type(e).__name__)
TypeCheckError

```

Without a typechecker active the call would simply broadcast `(2,)` against `(1,)` and return a wrong answer quietly.

## Turn it on for everything

To check all of `unxt` — and, through `jaxtyping`'s [import hook](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook), your own annotated functions — set an environment variable to any runtime typechecker backend `jaxtyping` supports:

```{code-block} bash

export UNXT_ENABLE_RUNTIME_TYPECHECKING="beartype.beartype"

```

The variable takes three values:

| Value | Effect |
| --- | --- |
| `"False"` | No import hook is installed. **The default**, absent the variable. |
| `"None"` | Import hook installed with no typechecker: only `@jaxtyped`-decorated functions are checked. |
| any other string | Used as the `jaxtyping` typechecker, e.g. `"beartype.beartype"`. |

To turn checking back off, set it to `"False"`:

```{code-block} bash

export UNXT_ENABLE_RUNTIME_TYPECHECKING="False"

```

If you would rather not manage the shell environment, set it from Python — but it must happen **before** importing `unxt`, or anything that imports `unxt`:

```{code-block} python

import os

os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"

```

:::{attention}

Enable runtime type checking during development. For production runs, measure with it **on and off**: the overhead is usually small, but it can affect how long [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) takes to compile.

:::

## Constrain the physical dimension

Annotations on the default `Quantity` constrain dtype and shape but **not** dimension: `Quantity["length"]` is `Quantity`, and the subscript does nothing. To have an argument's dimension checked, use `ParametricQuantity` from the separate [`unxts.parametric`](../packages/unxts.parametric/index) package, which encodes the dimension in its type — see [Dimension annotations for type checking](../packages/unxts.parametric/type-checking).

## See also

- {doc}`../reference/quantity` — what the default `Quantity` does and does not carry in its type.
- {doc}`../explanation/why-quantity-is-non-parametric` — why the default stops at dtype and shape.
- The [`jaxtyping` library][jaxtyping-link] and [the `typing` module](https://docs.python.org/3/library/typing.html).
