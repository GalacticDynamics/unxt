# How to combine `unitful` with other NumPyro handlers

This guide shows where to put `condition`, `reparam`, `do` and the other NumPyro handlers relative to `unitful`. The full table is in {ref}`Behaviour of unitful <numpyro-handlers>`.

```python
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro import handlers
>>> from numpyro.infer.reparam import LocScaleReparam
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> galactic = u.unitsystems.galactic

>>> def model():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     numpyro.sample("x", dist.Normal(m, s), obs=u.Q([1.0, 1.2, 0.9], "kpc"))

>>> M = unitful(model, galactic)

```

## Fix a site's value with `condition` or `substitute`

Give the value as a `Quantity`. It works with the handler on either side of `unitful`.

```python
>>> c = handlers.condition(M, {"m": u.Q(500.0, "pc")})
>>> float(handlers.trace(handlers.seed(c, 0)).get_trace()["m"]["value"])
0.5

```

The trace holds 0.5 because 500 pc is 0.5 kpc, the `galactic` unit of length.

If you have a bare number for a site with a dimension, put the handler outside `unitful` and give the number in the unit of `usys`. Inside `unitful` it raises `UnitfulError`.

## Reparameterize a site

Put `reparam` outside `unitful`.

```python
>>> rm = handlers.reparam(M, config={"m": LocScaleReparam(0)})
>>> list(handlers.trace(handlers.seed(rm, 0)).get_trace())
['m_decentered', 'm', 's', 'x']

```

## Intervene with `do` or put a prior on a param with `lift`

Put these inside `unitful`, around the bare model. With `do`, the model gets the `Quantity` you give in place of the sample, in the unit you gave it:

```python
>>> def shifted():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return m + u.Q(1.0, "kpc")

>>> intervened = unitful(handlers.do(shifted, {"m": u.Q(500.0, "pc")}), galactic)
>>> round(float(u.ustrip("kpc", handlers.seed(intervened, 0)())), 3)
1.5

```

For `lift`, write `unitful(handlers.lift(model, prior=prior), usys)`. If `lift` has to go outside, give its prior bare parameters in the unit of `usys`.

## Use one unit system per model

Wrap the model in a single `unitful`. A second `unitful` inside it with another `usys` does not convert between the two, and with the same `usys` the outer one records no units, so `attach_units` has to use the inner one.
