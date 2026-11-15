# How to use unitful in your own models

This page collects the tasks you run into once the [tutorial](tutorial-fit-mean-and-width) works. Every example uses the same setup.

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> from numpyro.infer.reparam import LocScaleReparam
>>> from numpyro.infer.util import log_density
>>> import unxt as u
>>> from unxts.interop.numpyro import UnitfulError, unitful

>>> galactic = u.unitsystems.galactic

>>> def model():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     numpyro.sample("x", dist.Normal(m, s), obs=u.Q([1.0, 1.2, 0.9], "kpc"))

```

## Combine unitful with other handlers

`unitful` has to wrap the model. You write `unitful(model, usys)` and pass that to NumPyro, and you do not use it as a bare `with unitful(...)` context around inference.

Handlers that change how a site is sampled go outside it. That covers `reparam` and `config_enumerate`, which need to see the bare numbers that `unitful` produces.

```python
>>> M = unitful(model, galactic)
>>> rm = handlers.reparam(M, config={"m": LocScaleReparam(0)})
>>> list(handlers.trace(handlers.seed(rm, 0)).get_trace())
['m_decentered', 'm', 's', 'x']

```

Handlers that fix a value, `condition` and `substitute`, work on either side when you give them a quantity. A bare number works only when the handler is outside `unitful`, where you give it in the unit of `usys`. An inner handler with a bare value raises `UnitfulError`.

```python
>>> c = handlers.condition(M, {"m": u.Q(500.0, "pc")})
>>> float(handlers.trace(handlers.seed(c, 0)).get_trace()["m"]["value"])
0.5

```

The value 500 pc is stored as 0.5, which is the number in kpc.

With the handler inside `unitful` the quantity works as well, and a bare value raises.

```python
>>> inner = unitful(handlers.condition(model, {"m": u.Q(500.0, "pc")}), galactic)
>>> float(handlers.trace(handlers.seed(inner, 0)).get_trace()["m"]["value"])
0.5

>>> bare = unitful(handlers.condition(model, {"m": 0.5}), galactic)
>>> handlers.seed(bare, 0)()
Traceback (most recent call last):
    ...
unxts.interop.numpyro._src.rules.UnitfulError: Site 'm': value is bare; give it units of ...

```

Re-entering the same `unitful` instance is not supported, so a model that calls itself through the same handler will not work.

## Use a custom guide

A guide such as `AutoNormal` is built from the wrapped model, `AutoNormal(unitful(model, usys))`, and runs through the same handler. `Predictive` on the guide gives bare samples, and `attach_units(samples, handler)` puts the units on them. NUTS and SVI work the same way, so the posterior you get from either goes through `attach_units`.

A guide that I write myself with `Quantity` parameters needs its own handler, `unitful(guide, usys)`, with the same `usys` as the model. Each handler records the sites it ran. I use the model's handler for `attach_units` on the posterior samples, and the guide's handler on the fitted parameters from `svi.get_params`.

```python
>>> from numpyro.infer import SVI, Predictive, Trace_ELBO
>>> from numpyro.optim import Adam
>>> from unxts.interop.numpyro import attach_units
>>> positive = dist.constraints.positive

>>> def guide():
...     loc = numpyro.param("m_loc", u.Q(1.0, "kpc"))
...     scale = numpyro.param("m_scale", u.Q(0.1, "kpc"), constraint=positive)
...     numpyro.sample("m", dist.Normal(loc, scale))
...     s_map = numpyro.param("s_map", u.Q(0.2, "kpc"), constraint=positive)
...     numpyro.sample("s", dist.Delta(s_map))

>>> M, G = unitful(model, galactic), unitful(guide, galactic)
>>> svi = SVI(M, G, Adam(0.01), Trace_ELBO())
>>> res = svi.run(jax.random.PRNGKey(0), 100, progress_bar=False)
>>> attach_units(svi.get_params(res.state), G)["m_loc"].unit
Unit("kpc")

>>> post = Predictive(G, params=res.params, num_samples=10)(jax.random.PRNGKey(1))
>>> attach_units(post, M)["m"].unit
Unit("kpc")

```

## Give a site its dimension with `units=`

Some distributions have parameters that cannot carry the unit. An `ImproperUniform` over a length has none, and a `LogNormal` of a length has a bare `loc` and `scale` because they are the log of a length. For these I pass `units={"site": unit}`.

```python
>>> def lognormal():
...     return numpyro.sample("r", dist.LogNormal(0.0, 1.0))

>>> r = handlers.seed(unitful(lognormal, galactic, units={"r": "pc"}), 0)()
>>> r.unit
Unit("kpc")

```

The unit in `units=` only gives the site its dimension. The value is still stripped and wrapped in the unit of `usys` for that dimension, which is why `r` has the unit kpc here.

This does not let you use bare numbers where the family's rule names parameters that carry the dimension. A `Normal` with a bare `loc` and a `Quantity` `scale` raises, and so does a `units=` entry for such a site.

```python
>>> def mixed():
...     return numpyro.sample("m", dist.Normal(1.0, u.Q(3.0, "kpc")))

>>> handlers.seed(unitful(mixed, galactic), 0)()
Traceback (most recent call last):
    ...
unxts.interop.numpyro._src.rules.UnitfulError: Site 'm': parameter 'loc' is bare next to Quantity parameters; give it units.

```

A concrete zero is allowed, so `Normal(0.0, u.Q(3.0, "kpc"))` is fine.

## Use scan and masked observations with rewrap=False

`numpyro.contrib.control_flow.scan` traces its body inside a JAX transform, and a `Quantity` cannot cross that boundary as a sampled value. Models that use `obs_mask=` have the same problem. For both I pass `rewrap=False`, so the handler gives the model bare numbers in the unit of `usys`, and I put the units back myself where I need them.

```python
>>> from numpyro.contrib.control_flow import scan

>>> def scan_model(xs):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     mq = u.Q(m, "kpc")
...     def step(carry, xi):
...         numpyro.sample("x", dist.Normal(mq, u.Q(1.0, "kpc")), obs=xi)
...         return carry, None
...     scan(step, 0.0, xs)

>>> xs = u.Q(jnp.array([1.0, 1.2, 0.9]), "kpc")
>>> M = unitful(scan_model, galactic, rewrap=False)
>>> logp, _ = log_density(M, (xs,), {}, {"m": jnp.asarray(1.0)})
>>> round(float(logp), 1)
-4.9

```

## Index data instead of subsampling

`numpyro.subsample` does not work on a `Quantity`. I index the data inside the plate, `x[idx]`, with `idx` from the plate.

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

## Write jnp code on unitful latents

The sampled values are quantities, and plain `jax.numpy` functions do not take quantities. For a few lines I use the `quaxed.numpy` module, which is a drop-in for `jnp` that works on quantities.

```python
>>> import quaxed.numpy as qnp

>>> def with_qnp():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return qnp.sqrt(m**2 + u.Q(1.0, "kpc") ** 2)

>>> handlers.seed(unitful(with_qnp, galactic), 0)().unit
Unit("kpc")

```

For an existing plain-`jnp` function I wrap that function with `quax.quaxify` and call the wrapped version on the latents.

```python
>>> import quax

>>> def chi(x, m):
...     return jnp.sum(jnp.square(x - m))

>>> def with_quaxify():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return quax.quaxify(chi)(u.Q(jnp.array([1.0, 2.0]), "kpc"), m)

>>> handlers.seed(unitful(with_quaxify, galactic), 0)().unit
Unit("kpc2")

```

I only call `quaxify` on helper functions. Wrapping the whole model with it does not work, because NumPyro handlers keep their values outside the transform.

## Read the log density

Stripping in `usys` means the log density is a density per `usys` unit. If you change the unit system, the log density shifts by a constant, so a posterior does not change but a log-likelihood value only compares to another one computed with the same `usys`.

```python
>>> def lm(x):
...     numpyro.sample("x", dist.Normal(u.Q(0.0, "kpc"), u.Q(1.0, "kpc")), obs=x)

>>> x = u.Q(0.5, "kpc")
>>> kpc, _ = log_density(unitful(lm, galactic), (x,), {}, {})
>>> m, _ = log_density(unitful(lm, u.unitsystems.si), (x,), {}, {})
>>> round(float(kpc - m), 1)
44.9

```

The difference is the logarithm of the number of metres in a kiloparsec.

## Things that do not work

- `init_to_value` with `Quantity` values can fail for a model that has no deterministic site, because NumPyro transforms the values while no handler is running. Give `init_to_value` bare numbers in the unit of `usys`.
- Calling `mcmc.run` a second time on the same `MCMC` object with a `Quantity` data argument fails, which is a NumPyro issue. I create a new `MCMC` object for the second run.
- Distributions of the `MultivariateNormal` family with `Quantity` parameters are out of scope.
- Shape parameters have to be bare numbers. That covers `df` of a `StudentT`, `concentration` of a `Gamma` or `Beta`, and mixing probabilities. NumPyro's constructors reject a `Quantity` there, even a dimensionless one, so `dist.StudentT(u.Q(3.0, ""), loc, scale)` raises when it is built. I write `dist.StudentT(3.0, loc, scale)`.
- A `Quantity` set at a site that `unitful` has no unit for raises `UnitfulError`, with the site name in the message.
