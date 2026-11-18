# How to call `jnp` code on `unitful` latents

Inside a `unitful` model the sampled values are quantities, and plain `jax.numpy` functions do not take quantities. This guide shows two ways to compute with them without rewriting the model.

```python
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic

```

## For a few lines, use `quaxed.numpy`

`quaxed.numpy` has the same functions as `jax.numpy`, and they take quantities:

```python
>>> import quaxed.numpy as qnp

>>> def model():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return qnp.sqrt(m**2 + u.Q(1.0, "kpc") ** 2)

>>> handlers.seed(unitful(model, galactic), 0)().unit
Unit("kpc")

```

## For an existing `jnp` function, wrap it with `quax.quaxify`

Keep the function as it is, and call its `quaxify`-wrapped version on the latents:

```python
>>> import quax

>>> def chi(x, m):
...     return jnp.sum(jnp.square(x - m))

>>> def model():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return quax.quaxify(chi)(u.Q(jnp.array([1.0, 2.0]), "kpc"), m)

>>> handlers.seed(unitful(model, galactic), 0)().unit
Unit("kpc2")

```

Wrap helper functions only. Wrapping the whole model with `quaxify` does not work; {ref}`About unitful <numpyro-why-a-handler>` explains why.
