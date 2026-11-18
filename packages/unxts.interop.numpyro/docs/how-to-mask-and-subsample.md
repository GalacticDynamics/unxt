# How to mask or subsample `Quantity` observations

This guide shows how to observe part of a `Quantity` data set, with an `obs_mask=` or with a subsampled plate.

```python
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> from numpyro.infer.util import log_density
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic

```

## Mask observations

Pass `rewrap=False`, so the model gets bare numbers in the unit of `usys`, and wrap the latent back into a `Quantity` where the distribution needs it:

```python
>>> def masked(x):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     mask = jnp.array([True, False, True])
...     d = dist.Normal(u.Q(m, "kpc"), u.Q(1.0, "kpc"))
...     numpyro.sample("x", d, obs=x, obs_mask=mask)

>>> x = u.Q(jnp.array([1.0, 1.2, 0.9]), "kpc")
>>> M = unitful(masked, galactic, rewrap=False)
>>> p = {"m": jnp.asarray(1.0), "x_unobserved": jnp.zeros(3)}
>>> logp, _ = log_density(M, (x,), {}, p)
>>> round(float(logp), 1)
-5.3

```

NumPyro splits the site into `x_observed` and `x_unobserved`. Use those names in `units=` and when reading `M.site_units`, where `x` itself has no unit.

## Subsample observations

Index the data with the plate's indices instead of calling `numpyro.subsample`, which does not accept a `Quantity`:

```python
>>> def sub_model(x):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     with numpyro.plate("n", 100, subsample_size=10) as idx:
...         numpyro.sample("x", dist.Normal(m, u.Q(1.0, "kpc")), obs=x[idx])

>>> x = u.Q(jnp.linspace(0.0, 2.0, 100), "kpc")
>>> tr = handlers.trace(handlers.seed(unitful(sub_model, galactic), 0)).get_trace(x)
>>> tr["x"]["value"].shape
(10,)

```
