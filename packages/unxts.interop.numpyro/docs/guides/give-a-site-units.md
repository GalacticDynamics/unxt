# Give units to a site with no unit rule

Some distributions cannot carry a unit in their parameters. A `LogNormal` of a length is one: its `loc` and `scale` are logarithms. For such a site, name its dimension with `units=`. {doc}`../reference/unit-rules` lists the families that have a {term}`unit rule`.

```python
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic

```

## Name the site's dimension

Write the distribution with bare parameters in the unit of `usys`. Pass the site's dimension in `units=`:

```python
>>> def model():
...     return numpyro.sample("r", dist.LogNormal(0.0, 1.0))

>>> r = handlers.seed(unitful(model, galactic, units={"r": "length"}), 0)()
>>> r.unit
Unit("kpc")

```

The value comes back in kpc, the `galactic` unit of length. The bare `loc` and `scale` are read in kpc too.

Instead of the dimension, you can pass the `usys` unit itself, `units={"r": "kpc"}`. Any other unit of that dimension raises:

```python
>>> unitful(model, galactic, units={"r": "pc"})
Traceback (most recent call last):
    ...
unxts.interop.numpyro.UnitfulError: Site 'r': units= gives pc, but bare parameters are read in the unit of usys; pass units={'r': 'kpc'}.

```

## Name the site as `unitful` sees it

If a `scope` inside `unitful` adds a prefix, include the prefix in the key.

If the `scope` is outside `unitful`, leave the prefix out. For `handlers.scope(unitful(model, usys), "a")`, write `units={"r": ...}`, not `"a/r"`.

If the site is behind an `if`, the key raises `UnitfulError` until a call reaches the site.

## Use a censored distribution

Give it a bare base distribution, and pass `units=` for the site:

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

The same steps work for `MixtureGeneral`, `ImproperUniform`, `Pareto`, and a `TransformedDistribution` with a `ComposeTransform`.
