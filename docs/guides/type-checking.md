# Type Checking

[typing-Generics-link]: https://typing.readthedocs.io/en/latest/spec/generics.html#generics
[typing-link]: https://docs.python.org/3/library/typing.html
[jaxtyping-link]: https://pypi.org/project/jaxtyping/
[JAX-link]: https://jax.readthedocs.io/en/latest/quickstart.html
[JAX-jit-link]: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit
[plum-link]: https://pypi.org/project/plum-dispatch/

## TL;DR

You can annotate the **dtype** and **shape** of a `Quantity` — checked statically, and at runtime. To _also_ constrain the physical **dimension** of an argument, use the dimension-parametrized `ParametricQuantity` from the separate [`unxts.parametric`](../packages/unxts.parametric/index) package; see [Dimension annotations for type checking](../packages/unxts.parametric/type-checking).

In the following example we define a function over two 1-D, equally-shaped float arrays and return their elementwise ratio:

```{code-block} python

from jaxtyping import Float

import unxt as u

def velocity(
    x: Float[u.Quantity, "N"],
    t: Float[u.Quantity, "N"],
) -> Float[u.Quantity, "N"]:
    return x / t

```

For information on typing in Python see [the built-in `typing` module][typing-link]. Refer to the [`jaxtyping` library][jaxtyping-link] for how to annotate the dtype and shape of a `Quantity`, for example integer arrays or variable / context-dependent shapes. `jaxtyping` also powers `unxt`'s runtime typechecking, discussed next.

## Runtime Type Checking

Using [`jaxtyping`][jaxtyping-link],`unxt` supports runtime type checking, where type annotations are enforced during execution. This is very useful for finding and preventing type-related errors, like passing the wrong type of argument to a function or returning the wrong type of value. To enable runtime type checking on all of `unxt`, set the environment variable `UNXT_ENABLE_RUNTIME_TYPECHECKING` to `beartype.beartype` or any other runtime typecheck backend supported by [`jaxtyping`][jaxtyping-link].

```{code-block} bash

# Enable runtime type checking
export UNXT_ENABLE_RUNTIME_TYPECHECKING="beartype.beartype"

```

:::{attention}

We recommend enabling runtime type checking during development. <br> For normal use, try enabling **and disabling** runtime type checking to assess any performance impact.

:::

The performance overhead associated with runtime type checking should be small but isn't always -- in particular it can affect the time for [`JAX`][JAX-link] to [`jit`][JAX-jit-link] code. To turn off runtime type checking set the environment variable to `None`.

```{code-block} bash

# Disable runtime type checking
export UNXT_ENABLE_RUNTIME_TYPECHECKING="None"

```

Absent the environment variable, this is the default.

:::{tip}

You can set environment variables directly in Python. Execute the following before importing `unxt` (or any library that imports `unxt`).

```{code-block} python

import os

os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"

```

:::

In the background `unxt` checks for the `UNXT_ENABLE_RUNTIME_TYPECHECKING` environment variable and passes it to [`jaxtyping`][jaxtyping-link]'s [import hook](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook). `jaxtyping` also offers function-specific checking through the [`jaxtyped` decorator](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.jaxtyped).

Here's an example:

```{code-block} python
>>> from jaxtyping import Shaped, jaxtyped
>>> from beartype import beartype as typechecker  # or use any supported typechecker

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

The check earns its keep when an argument violates its annotation. Both parameters above share the axis name `"N"`, so arrays whose shapes disagree are rejected before the body runs:

```{code-block} python
>>> x2 = u.Q([2.0, 3.0], "m")  # shape (2,)
>>> t2 = u.Q([1.0], "s")       # shape (1,)

>>> try:
...     velocity(x2, t2)
... except Exception as e:
...     print(type(e).__name__)
TypeCheckError

```

The enforcement here comes from the explicit `@jaxtyped(typechecker=...)` decorator. Setting `UNXT_ENABLE_RUNTIME_TYPECHECKING` applies the same checking across `unxt` — and to your own annotated functions via the import hook — without decorating each one by hand. With no typechecker active the annotations are inert: the call would simply broadcast `(2,)` against `(1,)`.

## Dimension annotations

The default {class}`unxt.quantity.Quantity` carries **dtype and shape** in its type, but no dimension — so `Quantity[<dimension>]` **does nothing** (it is informational only). To additionally check the physical **dimension** of an argument (e.g. `up.PQ["length"]`), use `ParametricQuantity` from the separate [`unxts.parametric`](../packages/unxts.parametric/index) package, which encodes the dimension in its _type_. For construction, dimension inference/checking, and the parametric-class theory, see [Dimension annotations for type checking](../packages/unxts.parametric/type-checking).
