# Type Checking

[typing-Generics-link]:
  https://typing.readthedocs.io/en/latest/spec/generics.html#generics
[typing-link]: https://docs.python.org/3/library/typing.html
[jaxtyping-link]: https://pypi.org/project/jaxtyping/
[JAX-link]: https://jax.readthedocs.io/en/latest/quickstart.html
[JAX-jit-link]:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit
[plum-link]: https://pypi.org/project/plum-dispatch/

## TL;DR

You can tell functions about the dtype, shape, and dimensions of a `Quantity`.
The dtype and shape information can be checked statically, and all three can be
checked at runtime.

In the following example we will define a function that operates on two
length-'N' (1-D and equally-shaped) float dtype arrays. The function takes a
length and a time and returns a velocity.

```{code-block} python

from jaxtyping import Float

import unxt as u

def function(
    x: Float[u.Quantity["length"], "N"],
    t: Float[u.Quantity["time"], "N"],
) -> Float[u.Quantity["speed"], "N"]:
    return x / t

```

For information on typing in Python see [the built-in `typing`
module][typing-link]. Refer to the [`jaxtyping` library][jaxtyping-link] for
information on how to annotate the dtype and shape of a Quantity, for example
integer arrays or variable / context-dependent shapes. `jaxtyping` also powers
`unxt`'s runtime typechecking, discussed next.

## Runtime Type Checking

Using [`jaxtyping`][jaxtyping-link],`unxt` supports runtime type checking, where
type annotations are enforced during execution. This is very useful for finding
and preventing type-related errors, like passing the wrong type of argument to a
function or returning the wrong type of value. To enable runtime type checking
on all of `unxt`, set the environment variable
`UNXT_ENABLE_RUNTIME_TYPECHECKING` to `beartype.beartype` or any other runtime
typecheck backend supported by [`jaxtyping`][jaxtyping-link].

```{code-block} bash

# Enable runtime type checking
export UNXT_ENABLE_RUNTIME_TYPECHECKING="beartype.beartype"

```

:::{attention}

We recommend enabling runtime type checking during development. <br> For normal
use, try enabling **and disabling** runtime type checking to asses any
performance impact.

:::

The performance overhead associated with runtime type checking should be small
but isn't always -- in particular it can effect the time for [`JAX`][JAX-link]
to [`jit`][JAX-jit-link] code. To turn off runtime type checking set the
environment variable to `None`.

```{code-block} bash

# Disable runtime type checking
export UNXT_ENABLE_RUNTIME_TYPECHECKING="None"

```

Absent the environment variable, this is the default.

:::{tip}

You can set environment variables directly in Python. Execute the following
before importing `unxt` (or any library that imports `unxt`).

```{code-block} python

import os

os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"

```

:::

In the background `unxt` checks for the `UNXT_ENABLE_RUNTIME_TYPECHECKING`
environment variable and passes it to [`jaxtyping`][jaxtyping-link]'s
[import hook](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook).
`jaxtyping` also offers function-specific checking through the
[`jaxtyped` decorator](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.jaxtyped).

Here's an example:

```{code-block} python
>>> from jaxtyping import Shaped, jaxtyped
>>> from beartype import beartype as typechecker  # or use any supported typechecker

>>> import unxt as u

>>> @jaxtyped(typechecker=typechecker)
... def velocity(
...     x: Shaped[u.Quantity["length"], "N"],
...     t: Shaped[u.Quantity["time"], "N"],
... ) -> Shaped[u.Quantity["speed"], "N"]:
...     return x / t

>>> x = u.Q([2.], "m")
>>> t = u.Q([1.], "s")

>>> velocity(x, t)
Quantity(Array([2.], dtype=float32), unit='m / s')

```

## Dimension Annotations to Quantity

In the previous sections `Quantity` annotations had strings specifying the
dimensions of that Quantity. Let's explore this a little more deeply.

First the theory. Python classes can be 'parametric', where the class is
parametrized by a set of metadata. The most common example of this is for
[generics][typing-Generics-link] in the [builtin `typing` library][typing-link]
where the metadata is type information about a function or object. This is
useful for static type checking. However we are not limited to only type
information. Classes can implement any form of parametric design (see
[here](https://docs.python.org/3/reference/datamodel.html#object.__class_getitem__)).
We use the library [`plum`][plum-link], on which `unxt` depends, to
[enhance](https://beartype.github.io/plum/parametric.html) Python's parametric
functionality and enable `Quantity` classes to be parametrized by their unit's
dimensions in a way that can be checked by runtime type checkers.

Now for some examples.

```{code-block} python
>>> import unxt as u
```

When a `Quantity` is constructed it is parametrized by the unit's dimension.
This can be specified explicitly

```{code-block}
>>> u.Quantity["length"](1, "m")
Quantity['length'](Array(1, dtype=int32, ...), unit='m')
```

or inferred.

```{code-block}
>>> u.Q(1, "m")
Quantity['length'](Array(1, dtype=int32, ...), unit='m')
```

When given explicitly Quantity will check the input dimensions. Here a
length-parametrized Quantity is (correctly) refusing dimensions of time.

```{code-block} python
>>> try:
...     u.Quantity["length"](1, "s")
... except Exception as e:
...     print(e)
Physical type mismatch.
```

That should catch some bugs!

The act of filling a `Quantity`'s parameters and its construction may be
separated

```{code-block} python
>>> LengthQuantity = u.Quantity["length"]
>>> LengthQuantity
<class 'unxt...Quantity[PhysicalType('length')]'>
```

This parametric design is how `unxt` supports runtime type checking.

In `unxt` not all Quantity classes are parametric. The base class,
{class}`unxt.quantity.AbstractQuantity` is not parametric, nor is the concrete
class {class}`unxt.quantity.BareQuantity`.  Parametric classes incur a small
performance overhead (generally eliminated in [`jit`ted code][JAX-jit-link]),
which ultra-performance-optimized code might want to avoid, at the cost of
inference and checking of the dimensions.

:::{note}

`BareQuantity[<dimension>]` **does nothing** and is for informational purposes
only.

:::

Check out [`plum`](https://beartype.github.io/plum/parametric.html) to explore
more powerful features of parametric classes.
