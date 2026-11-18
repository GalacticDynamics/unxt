# Behaviour of `unitful`

The call signatures are in the {doc}`API <api>`. The per-family dimension rules are in {doc}`unit-rules`. Every error message, with its cause and fix, is in {doc}`errors`.

## Arguments

`unitful(model, usys, *, rewrap=True, units=None)` wraps a model. NumPyro inference takes the wrapped model in place of the model.

| Argument | Meaning |
| --- | --- |
| `model` | The NumPyro model. Required. |
| `usys` | A `unxt` unit system, or its name, such as `"galactic"`. Required. |
| `rewrap` | If true, the model gets each sampled value as a `Quantity` in its {term}`site unit`. If false, it gets the {term}`bare value`. |
| `units` | A map from site name to dimension, for families with no {term}`unit rule`. See [the `units=` keyword](#the-units-keyword). |

| Attribute | Meaning |
| --- | --- |
| `handler.usys` | The unit system. |
| `handler.site_units` | A read-only map from each site name the handler has run to its site unit. `None` marks a site with no unit. |

`unitful` works only as a wrapper. It does not support use as a `with` context around inference. It does not support a model that calls itself through the same `unitful` instance.

## Sites

| Site | Behaviour |
| --- | --- |
| {term}`latent site` | `unitful` strips the distribution's parameters in the site unit. The model gets the value as a `Quantity` in the site unit, or bare with `rewrap=False`. |
| {term}`observed site` | `unitful` also strips the observed value. At a site with a dimension, the observed value must be a `Quantity`. |
| `deterministic` | `unitful` strips a `Quantity` value in `usys` and records its unit. A bare value passes through. |
| `param` | `unitful` strips a `Quantity` init value in `usys`. A callable init must return a bare value in the unit of `usys`. |
| `factor` | The `log_factor` must be bare or a dimensionless `Quantity`. |

A `param` constraint can have `Quantity` bounds, such as `interval(u.Q(0.0, "pc"), u.Q(2.0, "kpc"))`. The init value must then be a `Quantity`, because it gives the site its dimension. `unitful` strips the init value and the bounds in the `usys` unit of that dimension.

A value can be a single `Quantity` array. It cannot be a container that holds quantities, such as `{"a": u.Q(1.0, "pc")}`.

A site with `obs_mask=` needs `rewrap=False`. NumPyro splits it into `<name>_observed` and `<name>_unobserved`. `units=` uses those two names. `site_units` holds both, plus `<name>` itself with `None`.

`numpyro.subsample` does not accept a `Quantity`. Indexing the data with the plate's indices, `x[idx]`, works.

(numpyro-the-units-keyword)=

## The `units=` keyword

`units={"<site>": <value>}` gives a {term}`sample site` a dimension.

| Part | Rule |
| --- | --- |
| Value | A dimension, such as `"length"`, or the `usys` unit of that dimension, such as `"kpc"` under `galactic`. Another unit of the same dimension raises `UnitfulError`. |
| Key | The site name as `unitful` sees it. A `scope` inside `unitful` is part of the name. A `scope` outside `unitful` is not. |
| Reach | Each key must name a site that some call of the handler has reached. The check runs when a call returns, and it counts every earlier call. |
| Site kind | Sample sites only. |
| Family | A family with no unit rule, including a discrete one, takes its dimension from `units=`. A family with a unit rule takes it from the rule, and `units=` must agree. |

The site's parameters are read in the site unit, and its value is wrapped in it.

(numpyro-handlers)=

## Handlers

| Handler | Position | Values it takes |
| --- | --- | --- |
| `condition`, `substitute` | inside or outside | A `Quantity` on either side. At a site with a dimension, a bare value only outside, read in the unit of `usys`. At a site with no unit, a bare value on either side. |
| `do` | inside | A `Quantity`. The model gets it in the unit given. |
| `reparam` | outside |  |
| `lift` | inside | A prior with `Quantity` parameters. An outside `lift` needs a prior with bare parameters in the unit of `usys`. With a `Quantity` prior, scoring fails, so `log_density` and MCMC raise a `TypeError`. Sampling fails for some families, such as `HalfNormal` and `Uniform`, and works for others. |
| `scale` | inside or outside | A bare or dimensionless scale. |
| `block` | outside | The autoguides run the model under `block`, and an outside `condition` still applies under it. A `block` inside `unitful` hides its sites from `unitful`, as from any outer handler. Those sites are not stripped, so they need bare parameters. |
| `scope` | inside or outside |  |
| `seed` | anywhere |  |
| `trace` | outside | An inner `trace` records each site's stripped distribution next to its rewrapped `Quantity` value. That site cannot be scored. |
| `unitful` | one per model | An inner `unitful` strips the sites first, so the outer one records no units. `attach_units` then needs the inner one. |

A `reparam` inside `unitful` fails. The error is either a unit conversion error that does not name the site, or a `UnitfulError` about the new site, such as `m_decentered`.

(numpyro-control-flow)=

## Control flow

The body of a `numpyro.contrib.control_flow.scan` and each branch of a `cond` run under their own `unitful`, with the same `usys` and `rewrap`.

| Construct | Rule |
| --- | --- |
| `scan` carry | A `Quantity` carry starts in the `usys` unit and keeps it through every step. A start value in another unit of the same dimension fails with a JAX error about the carry structure. A bare start value stays bare, so it fails next to `Quantity` parameters. With `rewrap=False`, the start value must be bare, in the `usys` unit. It works if the body wraps the carry in a `Quantity` before using it. |
| `cond` branches | Both branches sample the same site names, from the same families, with the same site unit. |
| `cond` predicate | A bare boolean. A comparison of two quantities gives a boolean `Quantity`, which JAX rejects. |

(numpyro-inference)=

## Inference

| Tool | Behaviour |
| --- | --- |
| `MCMC`, `NUTS` | `mcmc.get_samples()` gives bare values in their site units. |
| `SVI`, autoguide | An autoguide built from the wrapped model, such as `AutoNormal(unitful(model, usys))`, runs through the same handler. |
| `SVI`, custom guide | A guide with `Quantity` parameters runs under its own handler, `unitful(guide, model_handler.usys)`. |
| `Predictive` | Gives bare samples. |
| `log_likelihood` | Gives log densities per `usys` unit. They are not site values. |
| `init_to_value` | Takes bare values in the unit of `usys`. |

`init_to_value` converts `Quantity` values only in a model with a `deterministic` site. NumPyro then runs the model through the handlers to transform them. In a model without one, NumPyro transforms the values while no handler runs. A `Quantity` at a site with a constrained support, such as a `HalfNormal` scale, then fails with a JAX `TypeError`. At an unconstrained site, it can come back as a `Quantity` in the unit given, next to bare samples at the other sites. `mcmc.print_summary()` fails on such a mixed dict.

For a custom guide:

- The model reads replayed values in its own `usys`. A guide site with bare parameters has no unit, and its values are read in the unit of `usys`.
- A guide with `Quantity` parameters and no handler fails inside NumPyro, because nothing strips its parameters. The error depends on the family. A `HalfNormal` site raises a `UnitConversionError` when it samples. A `Delta` site raises `TypeError: log requires ndarray or scalar arguments` when it scores.
- A `VonMises` model site is in radians under any `usys`. Its guide site must then be a `VonMises` too, unless `usys` measures angles in radians.

`attach_units(samples, handler)` wraps each site's array in the site unit `handler` recorded. Keys the handler never saw, such as `_sig_log_det`, pass through, and so do values that are already quantities. Each array becomes a plain `Quantity`, also at a site that gave the model a `u.Angle`. `attach_units` cannot check that the samples came from `handler`. A handler with another `usys` puts the wrong units on them, with no error.

### NumPyro issue with cached samplers

Reusing an `MCMC` object's compiled sampler with a `Quantity` argument can raise `OverflowError`. NumPyro tracks this as [pyro-ppl/numpyro#2434](https://github.com/pyro-ppl/numpyro/issues/2434), with a fix proposed in [#2436](https://github.com/pyro-ppl/numpyro/pull/2436). It affects:

- a second chain with `chain_method="sequential"`, the default for several chains on one CPU device;
- a second `mcmc.run` on the same `MCMC` object;
- `mcmc.warmup` followed by `mcmc.run`.

{doc}`../guides/run-mcmc` gives the workarounds.

## Log density

`unitful` strips every value in `usys`, so each log density is per `usys` unit. Changing `usys` shifts the log density by a constant. Here the same observation is scored in kiloparsecs and in metres:

```python
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from numpyro.infer.util import log_density
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> def model(x):
...     numpyro.sample("x", dist.Normal(u.Q(0.0, "kpc"), u.Q(1.0, "kpc")), obs=x)

>>> x = u.Q(0.5, "kpc")
>>> in_kpc, _ = log_density(unitful(model, u.unitsystems.galactic), (x,), {}, {})
>>> in_m, _ = log_density(unitful(model, u.unitsystems.si), (x,), {}, {})
>>> round(float(in_kpc - in_m), 1)
44.9

```

The shift, 44.9, is the natural log of the number of metres in a kiloparsec. {doc}`../concepts/unit-systems` explains why.

## Import-time patches

Importing `unxts.interop.numpyro` patches two NumPyro functions. Each patch is installed only if a probe at import finds the bug in the installed NumPyro.

| Function | Bug | Fix in NumPyro |
| --- | --- | --- |
| `numpyro.util.not_jax_tracer` | Misses a tracer inside a pytree, such as a `Quantity` under `jit`. | [pyro-ppl/numpyro#2428](https://github.com/pyro-ppl/numpyro/pull/2428) |
| `numpyro.distributions.util._reshape` | Rejects an array-like with its own `reshape`, such as a `Quantity`. | [pyro-ppl/numpyro#2429](https://github.com/pyro-ppl/numpyro/pull/2429) |

The patched functions behave like the originals for every other input.
