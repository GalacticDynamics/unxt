# Fit a mean and a width to data with units

In this tutorial I fit a mean and a width to a set of distances measured in kiloparsecs. The data and the priors are `unxt` quantities, I run NUTS with the usual NumPyro calls, and at the end I put the units back on the posterior samples. You need `unxt`, `unxts.interop.numpyro` and `numpyro` installed.

## Set up

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro.infer import MCMC, NUTS
>>> import unxt as u
>>> from unxts.interop.numpyro import attach_units, unitful

```

## Make some data

I draw 200 distances around 1 kpc with a spread of 0.2 kpc, and wrap them as a `Quantity`.

```python
>>> noise = jax.random.normal(jax.random.PRNGKey(0), (200,))
>>> x = u.Q(1.0 + 0.2 * noise, "kpc")
>>> x.shape, x.unit
((200,), Unit("kpc"))

```

## Write the model

The model has a mean `m`, a width `s` and one observed site `x`. The prior on `m` is a normal distribution with a mean of 0 kpc and a width of 3 kpc, and the prior on `s` is a half-normal with a scale of 1 kpc. Both take `Quantity` parameters, and the observed value is the `Quantity` I made above.

```python
>>> def model(x=None):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     with numpyro.plate("n", 200):
...         numpyro.sample("x", dist.Normal(m, s), obs=x)

```

Inside the model `m` and `s` are quantities in kpc, so `dist.Normal(m, s)` has the units it needs without any further work.

## Wrap it in `unitful`

`unitful` takes the model and a unit system. At each site it strips the quantities into that unit system, so NumPyro sees plain numbers, and it hands the sampled value back to the model as a quantity.

```python
>>> M = unitful(model, u.unitsystems.galactic)

```

## Run NUTS

I give NUTS the wrapped model and pass the data as the model argument, as I would for any NumPyro model.

```python
>>> mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200, progress_bar=False)
>>> mcmc.run(jax.random.PRNGKey(1), x)
>>> samples = mcmc.get_samples()
>>> samples["m"].shape
(200,)

```

The samples are plain arrays, which are the numbers NumPyro worked with. They are in the galactic unit of length, which is kpc.

## Put the units back

`attach_units` takes the samples and the handler. The handler remembers the unit of every site it saw while the model ran, so it can wrap each array in the right one.

```python
>>> samples = attach_units(samples, M)
>>> samples["m"].unit
Unit("kpc")
>>> round(float(u.ustrip("kpc", samples["m"].mean())), 1)
1.0

```

The posterior mean of `m` is close to 1 kpc, which is the value I put into the data. I can read the width in another unit by asking for it.

```python
>>> 150.0 < float(u.ustrip("pc", samples["s"].mean())) < 250.0
True

```

The width comes out between 150 and 250 pc, which is the 0.2 kpc I used for the spread.

## Where to go next

The [guide](guide) covers how `unitful` combines with other NumPyro handlers and what to do with `scan` and subsampling.
