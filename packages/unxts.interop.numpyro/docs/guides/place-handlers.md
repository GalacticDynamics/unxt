# Combine `unitful` with other handlers

Each NumPyro handler goes either inside `unitful` or outside it. Inside, it sees quantities. Outside, it sees {term}`bare values <bare value>`. {ref}`The handler table <numpyro-handlers>` lists every handler, and {doc}`../concepts/handler-order` explains the rule.

The examples use this model:

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

## Fix a value with `condition` or `substitute`

Give the value as a `Quantity`. The handler can then go on either side of `unitful`.

```python
>>> c = handlers.condition(M, {"m": u.Q(500.0, "pc")})
>>> float(handlers.trace(handlers.seed(c, 0)).get_trace()["m"]["value"])
0.5

```

The trace holds 0.5, because 500 pc is 0.5 kpc.

If the value is a bare number, put the handler outside `unitful`. Give the number in the unit of `usys`.

## Reparameterize a site

Put `reparam` outside `unitful`:

```python
>>> rm = handlers.reparam(M, config={"m": LocScaleReparam(0)})
>>> list(handlers.trace(handlers.seed(rm, 0)).get_trace())
['m_decentered', 'm', 's', 'x']

```

## Intervene with `do`

Put `do` inside `unitful`, around the bare model. The model then gets the `Quantity` you give, in the unit you give it:

```python
>>> def shifted():
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     return m + u.Q(1.0, "kpc")

>>> intervened = unitful(handlers.do(shifted, {"m": u.Q(500.0, "pc")}), galactic)
>>> round(float(u.ustrip("kpc", handlers.seed(intervened, 0)())), 3)
1.5

```

## Put a prior on a `param` with `lift`

Put `lift` inside `unitful`, as `unitful(handlers.lift(model, prior=prior), usys)`. The prior can then have `Quantity` parameters.

If `lift` must go outside `unitful`, give the prior bare parameters in the unit of `usys`.

## Use one `unitful` per model

Wrap the model in a single `unitful`. An inner `unitful` would strip the sites first, and the outer one would record no units.
