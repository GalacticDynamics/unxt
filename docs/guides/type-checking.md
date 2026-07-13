# Type Checking

[typing-Generics-link]: https://typing.readthedocs.io/en/latest/spec/generics.html#generics
[typing-link]: https://docs.python.org/3/library/typing.html
[jaxtyping-link]: https://pypi.org/project/jaxtyping/
[JAX-link]: https://jax.readthedocs.io/en/latest/quickstart.html
[JAX-jit-link]: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit
[plum-link]: https://pypi.org/project/plum-dispatch/

## TL;DR

You can tell functions about the dtype, shape, and dimensions of a `ParametricQuantity`. The dtype and shape information can be checked statically, and all three can be checked at runtime. (Runtime _dimension_ checking is what the parametric `ParametricQuantity` -- from the `unxts.parametric` package, imported here as `up` so `ParametricQuantity` is `up.PQ` -- adds over the lightweight default `Quantity`.)

In the following example we will define a function that operates on two length-'N' (1-D and equally-shaped) float dtype arrays. The function takes a length and a time and returns a velocity.

```{code-block} python

from jaxtyping import Float

import unxt as u
import unxts.parametric as up

def function(
    x: Float[up.PQ["length"], "N"],
    t: Float[up.PQ["time"], "N"],
) -> Float[up.PQ["speed"], "N"]:
    return x / t

```

For information on typing in Python see [the built-in `typing` module][typing-link]. Refer to the [`jaxtyping` library][jaxtyping-link] for information on how to annotate the dtype and shape of a ParametricQuantity, for example integer arrays or variable / context-dependent shapes. `jaxtyping` also powers `unxt`'s runtime typechecking, discussed next.

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
>>> import unxts.parametric as up

>>> @jaxtyped(typechecker=typechecker)
... def velocity(
...     x: Shaped[up.PQ["length"], "N"],
...     t: Shaped[up.PQ["time"], "N"],
... ) -> Shaped[up.PQ["speed"], "N"]:
...     return x / t

>>> x = up.PQ([2.], "m")
>>> t = up.PQ([1.], "s")

>>> velocity(x, t)
ParametricQuantity(Array([2.], dtype=float32), unit='m / s')

```

## Dimension annotations

The examples above constrain the physical **dimension** of an argument (e.g. `up.PQ["length"]`). That capability comes from `ParametricQuantity`, which encodes the dimension in its _type_ and lives in the separate [`unxts.parametric`](../packages/unxts.parametric/index.md) package. For how `ParametricQuantity` is constructed, how dimensions are inferred and checked, and the parametric-class theory, see [Dimension annotations for type checking](../packages/unxts.parametric/index.md#dimension-annotations-for-type-checking).

The default {class}`unxt.quantity.Quantity` (and the base {class}`unxt.quantity.AbstractQuantity`) are **not** parametric — their type carries dtype and shape but no dimension, so annotations still check dtype and shape.

:::{note}

`Quantity[<dimension>]` **does nothing** and is for informational purposes only.

:::
