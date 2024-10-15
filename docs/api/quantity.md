# Quantities

```{eval-rst}
.. automodule:: unxt.quantity
    :show-inheritance:
    :members:

```

```{code-block} python

>>> import jax.numpy as jnp
>>> from unxt import Quantity

>>> x = Quantity(jnp.arange(1, 5, dtype=float), "kpc")
>>> x
Quantity['length'](Array([1., 2., 3., 4.], dtype=float32), unit='kpc')

# Addition / Subtraction
>>> x + x
Quantity['length'](Array([2., 4., 6., 8.], dtype=float32), unit='kpc')

# Multiplication / Division
>>> 2 * x
Quantity['length'](Array([2., 4., 6., 8.], dtype=float32), unit='kpc')

>>> y = Quantity(jnp.arange(4, 8, dtype=float), "Gyr")

>>> x / y
Quantity['speed'](Array([0.25     , 0.4      , 0.5      , 0.5714286], dtype=float32), unit='kpc / Gyr')

# Exponentiation
>>> x**2
Quantity['area'](Array([ 1.,  4.,  9., 16.], dtype=float32), unit='kpc2')

# Unit Checking on operations
>>> try:
...    x + y
... except Exception as e:
...     print(e)
'Gyr' (time) and 'kpc' (length) are not convertible

```

`unxt` is built on [`quax`][quax], which enables custom array-ish objects in
JAX. For convenience we use the [`quaxed`][quaxed] library, which is just a
`quax.quaxify` wrapper around `jax` to avoid boilerplate code.

```{code-block} python

>>> from quaxed import grad, vmap
>>> import quaxed.numpy as jnp

>>> jnp.square(x)
Quantity['area'](Array([ 1.,  4.,  9., 16.], dtype=float32), unit='kpc2')

>>> jnp.power(x, 3)
Quantity['volume'](Array([ 1.,  8., 27., 64.], dtype=float32), unit='kpc3')

>>> vmap(grad(lambda x: x**3))(x)
Quantity['area'](Array([ 3., 12., 27., 48.], dtype=float32), unit='kpc2')

```

Since `Quantity` is parametric, it can do runtime dimension checking!

```{code-block} python

>>> LengthQuantity = Quantity["length"]
>>> LengthQuantity(2, "km")
Quantity['length'](Array(2, dtype=int32, weak_type=True), unit='km')

>>> try:
...     LengthQuantity(2, "s")
... except ValueError as e:
...     print(e)
Physical type mismatch.

```
