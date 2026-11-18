# Sample inside `scan` and `cond`

The sites in a `numpyro.contrib.control_flow.scan` body or a `cond` branch work like any other site. `unitful` runs them under a handler of its own, with the same `usys` and `rewrap`. The carry and the predicate need care. {ref}`The control-flow reference <numpyro-control-flow>` has the rules.

The examples use these imports and data:

```python
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> from numpyro.contrib.control_flow import cond, scan
>>> from numpyro.infer.util import log_density
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic
>>> ys = u.Q(jnp.array([100.0, 300.0, 200.0]), "pc")
>>> p = {"s": jnp.asarray(0.5), "z": jnp.array([0.1, 0.2, 0.3])}

```

## Carry a `Quantity` through `scan`

Give the start value as a `Quantity` in the unit of `usys`, here kpc:

```python
>>> def walk(ys):
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     def step(prev, y):
...         z = numpyro.sample("z", dist.Normal(prev, s))
...         numpyro.sample("y", dist.Normal(z, u.Q(0.1, "kpc")), obs=y)
...         return z, None
...     scan(step, u.Q(0.0, "kpc"), ys)

>>> M = unitful(walk, galactic)
>>> logp, _ = log_density(M, (ys,), {}, p)
>>> round(float(logp), 2)
2.06
>>> M.site_units["z"]
Unit("kpc")

```

A start value in pc fails, because the step returns `z` in kpc. JAX then reports a change in the carry structure.

## Carry a bare number through `scan`

Pass `rewrap=False`. The body then gets bare numbers in the units of `usys`. Add the units where a distribution needs them:

```python
>>> def walk_bare(ys):
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     def step(prev, y):
...         z = numpyro.sample("z", dist.Normal(u.Q(prev, "kpc"), u.Q(s, "kpc")))
...         numpyro.sample("y", dist.Normal(u.Q(z, "kpc"), u.Q(0.1, "kpc")), obs=y)
...         return z, None
...     scan(step, 0.0, ys)

>>> logp, _ = log_density(unitful(walk_bare, galactic, rewrap=False), (ys,), {}, p)
>>> round(float(logp), 2)
2.06

```

The log density is the same as with the `Quantity` carry.

## Branch with `cond`

Sample the same site names, from the same families, in both branches. Pass a bare boolean as the predicate:

```python
>>> def pick(near):
...     def close(_):
...         return numpyro.sample("d", dist.Normal(u.Q(1.0, "kpc"), u.Q(0.1, "kpc")))
...     def far(_):
...         return numpyro.sample("d", dist.Normal(u.Q(8.0, "kpc"), u.Q(0.1, "kpc")))
...     return cond(near, close, far, None)

>>> handlers.seed(unitful(pick, galactic), 0)(True).unit
Unit("kpc")

```

If the predicate compares quantities, strip them first. Write `u.ustrip("kpc", r) < 5.0`, not `r < u.Q(5.0, "kpc")`.
