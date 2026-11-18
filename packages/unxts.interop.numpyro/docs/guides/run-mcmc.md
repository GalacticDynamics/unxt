# Run MCMC on `Quantity` data

NumPyro can raise `OverflowError` when it reuses a compiled sampler with a `Quantity` argument. {ref}`The inference reference <numpyro-inference>` lists when this happens. The steps below avoid it until a NumPyro release includes the fix.

The examples use this model and data:

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

Pass `chain_method="vectorized"` to `MCMC`:

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

If you need sequential chains, close over the data. The data is then not an argument of `mcmc.run`:

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

Make a new `MCMC` object for each run with `Quantity` data. Let `mcmc.run` do the warmup, instead of calling `mcmc.warmup` first.

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

Here 1.0 and 0.2 are in kpc, the `galactic` unit of length. {ref}`The inference reference <numpyro-inference>` explains when `Quantity` initial values fail.
