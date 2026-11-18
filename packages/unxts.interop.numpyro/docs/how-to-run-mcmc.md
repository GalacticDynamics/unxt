# How to run MCMC on `Quantity` data

This guide shows how to run several chains, run a sampler more than once, and set initial values for a `unitful` model with `Quantity` data. Until a NumPyro release includes [pyro-ppl/numpyro#2436](https://github.com/pyro-ppl/numpyro/pull/2436), NumPyro can raise `OverflowError` when it reuses a compiled sampler with a `Quantity` argument. The workarounds below avoid that.

```python
>>> import functools
>>> import jax
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro.infer import MCMC, NUTS, init_to_value
>>> import unxt as u
>>> from unxts.interop.numpyro import attach_units, unitful

>>> def model(x):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     with numpyro.plate("n", x.shape[0]):
...         numpyro.sample("x", dist.Normal(m, s), obs=x)

>>> x = u.Q(1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(0), (50,)), "kpc")
>>> galactic = u.unitsystems.galactic

```

## Run several chains

Pass `chain_method="vectorized"`:

```python
>>> M = unitful(model, galactic)
>>> mcmc = MCMC(
...     NUTS(M), num_warmup=100, num_samples=100, num_chains=2,
...     chain_method="vectorized", progress_bar=False,
... )
>>> mcmc.run(jax.random.PRNGKey(1), x)
>>> mcmc.get_samples()["m"].shape
(200,)

```

If you need sequential chains, which are the default for several chains on one CPU device, close over the data so it is not an argument of `mcmc.run`:

```python
>>> M = unitful(functools.partial(model, x), galactic)
>>> mcmc = MCMC(
...     NUTS(M), num_warmup=100, num_samples=100, num_chains=2,
...     chain_method="sequential", progress_bar=False,
... )
>>> mcmc.run(jax.random.PRNGKey(1))
>>> mcmc.get_samples()["m"].shape
(200,)

```

## Run the sampler again

Make a new `MCMC` object for each run with `Quantity` data, and let `mcmc.run` do the warmup rather than calling `mcmc.warmup` first.

## Set initial values

Give `init_to_value` bare numbers in the units of `usys`:

```python
>>> init = init_to_value(values={"m": 1.0, "s": 0.2})
>>> M = unitful(model, galactic)
>>> mcmc = MCMC(
...     NUTS(M, init_strategy=init), num_warmup=100, num_samples=100,
...     progress_bar=False,
... )
>>> mcmc.run(jax.random.PRNGKey(1), x)
>>> attach_units(mcmc.get_samples(), M)["s"].unit
Unit("kpc")

```

Here 1.0 and 0.2 are kpc, the `galactic` unit of length. {ref}`Behaviour of unitful <numpyro-inference>` says what goes wrong with `Quantity` initial values.
