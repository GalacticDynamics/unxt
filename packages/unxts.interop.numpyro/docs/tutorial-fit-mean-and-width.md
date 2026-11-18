# Fit a mean and a width to data with units

In this tutorial we fit a mean and a width to 200 distances measured in kiloparsecs. We write the model with `unxt` quantities, run NUTS on it with the usual NumPyro calls, and get the posterior back in units. You need `unxt[interop-numpyro]` installed.

## Set up

```python
>>> import jax
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro.infer import MCMC, NUTS
>>> import unxt as u
>>> from unxts.interop.numpyro import attach_units, unitful

```

## Make some data

We draw 200 distances around 1 kpc with a spread of 0.2 kpc, and make them a `Quantity`:

```python
>>> noise = jax.random.normal(jax.random.PRNGKey(0), (200,))
>>> x = u.Q(1.0 + 0.2 * noise, "kpc")
>>> x.shape, x.unit
((200,), Unit("kpc"))

```

## Write the model

The model has a mean `m`, a width `s` and the observed distances `x`. The priors are a normal distribution centred on 0 kpc with a width of 3 kpc for `m`, and a half-normal with a scale of 1 kpc for `s`:

```python
>>> def model(x=None):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     with numpyro.plate("n", 200):
...         numpyro.sample("x", dist.Normal(m, s), obs=x)

```

This is an ordinary NumPyro model, except that every parameter and the data are quantities. Notice that `m` and `s` are used directly as the parameters of `dist.Normal(m, s)`: inside the model they are quantities in kpc.

## Wrap it in `unitful`

We wrap the model in `unitful` and give it the `galactic` unit system:

```python
>>> M = unitful(model, u.unitsystems.galactic)

```

`M` is the model NumPyro will run. At each site it turns the quantities into plain numbers in `galactic` units, and gives the sampled values back to the model as quantities.

## Run NUTS

We pass `M` to NUTS and the data to `mcmc.run`, as for any NumPyro model:

```python
>>> mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200, progress_bar=False)
>>> mcmc.run(jax.random.PRNGKey(1), x)
>>> samples = mcmc.get_samples()
>>> samples["m"].shape
(200,)

```

These samples are plain arrays, the numbers NumPyro worked with. They are in kpc, the `galactic` unit of length, but nothing on them says so yet.

## Put the units back

`attach_units` takes the samples and the handler. `M` recorded the unit of every site while the model ran, so it can wrap each array in the right one:

```python
>>> samples = attach_units(samples, M)
>>> samples["m"].unit
Unit("kpc")
>>> round(float(u.ustrip("kpc", samples["m"].mean())), 1)
1.0

```

The posterior mean of `m` is close to 1 kpc, the value we put into the data. Because `s` is now a `Quantity`, we can read it in parsecs:

```python
>>> 150.0 < float(u.ustrip("pc", samples["s"].mean())) < 250.0
True

```

The width is between 150 and 250 pc, close to the 0.2 kpc spread we drew the data with.

## What we did

We wrote a NumPyro model in units, wrapped it in `unitful` with a unit system, sampled it with NUTS, and put the units back on the posterior with `attach_units`.

## Next steps

- [How to run MCMC on `Quantity` data](how-to-run-mcmc) covers several chains and initial values.
- [How to fit a model with SVI](how-to-fit-with-svi) does the same fit with variational inference.
- [About `unitful`](explanation) explains why the handler strips units into a unit system.
