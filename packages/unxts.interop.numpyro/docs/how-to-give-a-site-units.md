# How to give units to a site whose distribution has no unit rule

This guide shows how to use `units=` for a site whose distribution cannot carry the unit in its parameters, such as a `LogNormal` of a length, whose `loc` and `scale` are logs. [Distribution unit rules](unit-rules) lists which families have a rule.

```python
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic

```

## Name the site's dimension

Write the distribution with bare parameters in the unit of `usys`, and pass the site's dimension in `units=`:

```python
>>> def model():
...     return numpyro.sample("r", dist.LogNormal(0.0, 1.0))

>>> r = handlers.seed(unitful(model, galactic, units={"r": "length"}), 0)()
>>> r.unit
Unit("kpc")

```

The latent comes back in kpc, the `galactic` unit of length, and the bare `loc` and `scale` are read in that unit. You can also write the `usys` unit itself, `units={"r": "kpc"}`. Another unit of the same dimension raises, because the bare parameters would be read in the wrong unit:

```python
>>> unitful(model, galactic, units={"r": "pc"})
Traceback (most recent call last):
    ...
unxts.interop.numpyro.UnitfulError: Site 'r': units= gives pc, but bare parameters are read in the unit of usys; pass units={'r': 'kpc'}.

```

## Name the site as `unitful` sees it

If the model uses `handlers.scope` inside `unitful`, include the prefix in the key. If the `scope` is outside `unitful`, leave the prefix out: for `handlers.scope(unitful(model, usys), "a")` write `units={"r": ...}`, not `"a/r"`.

If the site is behind an `if`, the key raises `UnitfulError` until some call of the handler has reached the site.

## For a censored distribution

Give it a bare base distribution and pass `units=` for the site:

```python
>>> import jax.numpy as jnp

>>> def censored():
...     mask = jnp.array([False, True])
...     d = dist.LeftCensoredDistribution(dist.Normal(0.0, 1.0), censored=mask)
...     numpyro.sample("x", d, obs=u.Q([0.5, 0.0], "kpc"))

>>> tr = handlers.trace(unitful(censored, galactic, units={"x": "kpc"})).get_trace()
>>> tr["x"]["value"]
Array([0.5, 0. ], dtype=float32)

```

Do the same for `MixtureGeneral`, `ImproperUniform`, `Pareto`, and a `TransformedDistribution` with a `ComposeTransform`.
