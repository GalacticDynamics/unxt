# Fit a model with SVI

Stochastic variational inference (SVI) fits a guide to the wrapped model. The guide can be an autoguide or one you write. Either way, `attach_units` puts the units back on the results.

The examples use this model:

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

## Use an autoguide

Build the autoguide from the wrapped model:

```python
>>> svi = SVI(M, AutoNormal(M), Adam(0.01), Trace_ELBO())
>>> res = svi.run(jax.random.PRNGKey(0), 100, progress_bar=False)

```

## Write your own guide

Write the guide with `Quantity` parameters. Wrap it in its own `unitful`, built from the model's unit system, `M.usys`:

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

The two handlers now share one unit system. SVI passes {term}`bare values <bare value>` between them, and both read those values in the same units.

If a `param` constraint has `Quantity` bounds, give the init value as a `Quantity` too.

## Read the results with units

For the fitted parameters, pass the guide's handler to `attach_units`:

```python
>>> attach_units(svi.get_params(res.state), G)["m_loc"].unit
Unit("kpc")

```

For posterior samples drawn from the guide, pass the model's handler:

```python
>>> post = Predictive(G, params=res.params, num_samples=10)(jax.random.PRNGKey(1))
>>> attach_units(post, M)["m"].unit
Unit("kpc")

```

`attach_units` cannot tell which handler produced the samples. The wrong handler gives wrong units, with no error. {doc}`../reference/errors` lists the errors that a mismatched guide raises during the fit.
