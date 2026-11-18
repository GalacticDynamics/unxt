# How to fit a model with SVI

This guide shows how to run stochastic variational inference on a `unitful` model, with an autoguide or with a guide you write, and how to get the fitted values back with units.

```python
>>> import jax
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro.infer import SVI, Predictive, Trace_ELBO
>>> from numpyro.infer.autoguide import AutoNormal
>>> from numpyro.optim import Adam
>>> import unxt as u
>>> from unxts.interop.numpyro import attach_units, unitful

>>> def model():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     numpyro.sample("x", dist.Normal(m, s), obs=u.Q([1.0, 1.2, 0.9], "kpc"))

>>> M = unitful(model, u.unitsystems.galactic)

```

## With an autoguide

Build the autoguide from the wrapped model.

```python
>>> svi = SVI(M, AutoNormal(M), Adam(0.01), Trace_ELBO())
>>> res = svi.run(jax.random.PRNGKey(0), 100, progress_bar=False)

```

## With your own guide

Write the guide with `Quantity` parameters, and wrap it in a `unitful` built from the model's unit system, `M.usys`. Both handlers then read the bare values SVI passes between them in the same units.

```python
>>> positive = dist.constraints.positive

>>> def guide():
...     loc = numpyro.param("m_loc", u.Q(1.0, "kpc"))
...     scale = numpyro.param("m_scale", u.Q(0.1, "kpc"), constraint=positive)
...     numpyro.sample("m", dist.Normal(loc, scale))
...     s_map = numpyro.param("s_map", u.Q(0.2, "kpc"), constraint=positive)
...     numpyro.sample("s", dist.Delta(s_map))

>>> G = unitful(guide, M.usys)
>>> svi = SVI(M, G, Adam(0.01), Trace_ELBO())
>>> res = svi.run(jax.random.PRNGKey(0), 100, progress_bar=False)

```

If a `param` needs a constraint with `Quantity` bounds, give its init value as a `Quantity` too, so `unitful` knows the site's dimension.

## Get the results with units

Use the guide's handler for the fitted parameters:

```python
>>> attach_units(svi.get_params(res.state), G)["m_loc"].unit
Unit("kpc")

```

Use the model's handler for posterior samples drawn from the guide:

```python
>>> post = Predictive(G, params=res.params, num_samples=10)(jax.random.PRNGKey(1))
>>> attach_units(post, M)["m"].unit
Unit("kpc")

```

`attach_units` cannot tell which handler the samples came from, so a handler with another unit system puts the wrong units on them without an error. {ref}`Behaviour of unitful <numpyro-inference>` lists the errors a mismatched guide raises.
