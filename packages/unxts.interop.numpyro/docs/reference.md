# Behaviour of `unitful`

How `unitful` treats each kind of site, each NumPyro handler and each inference tool. The call signatures are in the [API](api), and the per-family dimension rules are in [Distribution unit rules](unit-rules).

## The handler

`unitful(model, usys, *, rewrap=True, units=None)` wraps a model. NumPyro inference takes the wrapped model in place of the model.

| Argument | Meaning |
| --- | --- |
| `model` | The NumPyro model. Required. |
| `usys` | A `unxt` unit system, or its name, such as `"galactic"`. Required. Every site is stripped in this unit system. |
| `rewrap` | If true, the model gets each sampled value back as a `Quantity` in the `usys` unit of the site's dimension, or in radians for a `VonMises` site. If false, it gets the bare number in that unit. |
| `units` | A map from site name to dimension, for sites whose family has no unit rule. See [the `units=` keyword](#the-units-keyword). |

`unitful` works only as a wrapper. Using it as a bare `with unitful(...)` context around inference is not supported, and neither is a model that calls itself through the same `unitful` instance.

`handler.site_units` maps each site name the handler has run to its unit, with `None` for a site with no unit. `handler.usys` is the unit system.

## Site values

| Site | What `unitful` does |
| --- | --- |
| `sample`, latent | Strips the distribution's parameters in `usys`. The value returns to the model as a `Quantity` in the `usys` unit of the site's dimension (radians for `VonMises`), or bare with `rewrap=False`. |
| `sample`, observed | Strips the observed value in the same unit. At a site with a dimension the observed value has to be a `Quantity`; a bare one raises `UnitfulError`. |
| `deterministic` | Strips a `Quantity` value in `usys` and records its unit. A bare value passes through. |
| `param` | Strips a `Quantity` init value in `usys`. A callable init has to return a bare number in the unit of `usys`. |
| `factor` | The `log_factor` has to be bare or a dimensionless `Quantity`. |

A `param` constraint can have `Quantity` bounds, as in `interval(u.Q(0.0, "pc"), u.Q(2.0, "kpc"))`. The init value has to be a `Quantity` too, because it gives the site its dimension. The init value and the bounds are stripped in the `usys` unit of that dimension. A bound of another dimension raises `UnitfulError`.

These raise `UnitfulError`, with the site name in the message:

- A `deterministic`, `param` or observed value that is a container holding quantities, such as `{"a": u.Q(1.0, "pc")}` or `[u.Q(1.0, "pc"), u.Q(2.0, "pc")]`. A single `Quantity` array is accepted.
- A `Quantity` with a dimension set at a site that has no unit. A dimensionless `Quantity` there is stripped.
- A value whose dimension differs from the site's.

A site with `obs_mask=` needs `rewrap=False`. NumPyro splits it into `<name>_observed` and `<name>_unobserved`, and those are its names for `units=`. `site_units` holds both, plus `<name>` itself with `None`.

`numpyro.subsample` does not accept a `Quantity`. Indexing the data with the plate's indices, `x[idx]`, does.

(numpyro-the-units-keyword)=

## The `units=` keyword

`units={"site": value}` gives a site a dimension.

| Part | Rule |
| --- | --- |
| Value | A dimension, such as `"length"`, or the unit of `usys` for that dimension, such as `"kpc"` under `galactic`. Another unit of the same dimension, such as `"pc"` under `galactic`, raises `UnitfulError`, because the site's bare parameters are read in the unit of `usys`. |
| Key | A site name as `unitful` sees it. A `scope` inside `unitful` is part of the name; a `scope` outside it is not. A key that matches only a name renamed outside `unitful` raises `UnitfulError`. |
| Reach | Every key has to name a sample site that some call of the handler has reached, or the call raises `UnitfulError` when the model returns. The check adds up over calls of the same handler. |
| Site kind | Sample sites only. A key that names a `deterministic`, `param` or `factor` site raises `UnitfulError`. |
| Family | For a family with a unit rule, the rule decides the dimension. `units=` there raises `UnitfulError` if the carrying parameters are all bare, or if it disagrees with the rule. For a family with no rule, including a discrete one, `units=` sets the dimension. |

The site's value is stripped and wrapped in the unit of `usys`, as for any site.

(numpyro-handlers)=

## Handlers

Where each NumPyro handler goes relative to `unitful`, and what values it takes.

| Handler | Position | Notes |
| --- | --- | --- |
| `condition`, `substitute` | inside or outside | A `Quantity` value works on either side. At a site with a dimension, a bare value works only outside, read in the unit of `usys`; inside, it raises `UnitfulError`. At a site with no unit, a bare value works on either side. |
| `do` | inside | Outside, `do` sends a copy of the site after `unitful` has stripped it, and `unitful` raises `UnitfulError`. |
| `reparam` | outside | Inside, it fails with a unit conversion error from the reparameterizer, which does not name the site, or with `UnitfulError` about the new site, such as `m_decentered`. |
| `lift` | inside | Outside, a `Quantity` prior fails inside NumPyro with a `TypeError`. An outside `lift` works with a prior whose parameters are bare numbers in the unit of `usys`. |
| `scale` | inside or outside | The scale has to be bare or a dimensionless `Quantity`. A scale with a dimension raises `UnitfulError`. |
| `block` | inside or outside | The autoguides run the model under `block`; an outside `condition` still applies under it. |
| `scope` | inside or outside | Changes the site names `units=` sees only when it is inside. |
| `seed`, `trace` | anywhere |  |
| `unitful` | inside another `unitful` | With the same `usys` the sites work, but the outer handler sees them already stripped and records no unit; `attach_units` needs the innermost handler. With a different `usys` nothing is converted and the outer handler records no units. |

(numpyro-control-flow)=

## Control flow

The body of a `numpyro.contrib.control_flow.scan` and each branch of a `cond` run under a `unitful` of their own with the same `usys` and `rewrap`. Their sites follow the rules on this page.

| Construct | Rule |
| --- | --- |
| `scan` carry | A `Quantity` carry has to start in the unit of `usys` and keep that unit through every step. A start value in another unit of the same dimension fails with a JAX error about the carry structure. A bare start value stays bare, so a latent built from it next to `Quantity` parameters raises `UnitfulError`. With `rewrap=False` the body gets bare numbers and a bare start value works. |
| `cond` branches | Both branches have to sample the same site names from the same families, and each site has to have the same unit in both. |
| `cond` predicate | A bare boolean. A comparison of two quantities is a boolean `Quantity`, which JAX rejects as a predicate. |

(numpyro-inference)=

## Inference

| Tool | Behaviour |
| --- | --- |
| `MCMC`, `NUTS` | `mcmc.get_samples()` gives bare numbers in the units of `usys`. `attach_units(samples, handler)` puts the units on them. |
| `SVI` with an autoguide | An autoguide built from the wrapped model, as `AutoNormal(unitful(model, usys))`, runs through the same handler. |
| `SVI` with a custom guide | A guide with `Quantity` parameters runs under its own handler, `unitful(guide, model_handler.usys)`. See below. |
| `Predictive` | Gives bare samples. `attach_units` puts the units on them. |
| `log_likelihood` | Gives log densities per `usys` unit. They are not site values, so they stay bare. |
| `init_to_value` | Takes bare numbers in the unit of `usys`. `Quantity` values are converted only in a model with a `deterministic` site, where NumPyro runs the model through the handlers to transform them. In a model without one, NumPyro transforms them while no handler is running: a site with a constrained support, such as a `HalfNormal` scale, fails with a JAX `TypeError`, and an unconstrained site can come back as a `Quantity` in the unit given while the other sites come back bare. A mixed sample dict like that breaks `mcmc.print_summary()`. |

For a custom guide:

- SVI replays the guide's bare values into the model, which reads them in its own `usys`. A guide handler with another `usys` raises `UnitfulError` at the first replayed site, and so does a guide site in another unit from the model's site.
- A guide site with bare parameters has no unit, and its values are read in the unit of `usys`.
- A guide with `Quantity` parameters and no handler fails inside NumPyro with a JAX `TypeError`, such as `log requires ndarray or scalar arguments`, because SVI computes the guide's log density before the model runs.
- A `VonMises` model site is in radians under any `usys`, so its guide site has to be a `VonMises` too when the angle unit of `usys` is not radians.

`attach_units(samples, handler)` wraps each site in the unit `handler` recorded for it. Keys the handler never saw, such as `_sig_log_det`, and values that are already quantities pass through. It raises `UnitfulError` if the handler has not run yet. It wraps in plain `Quantity` objects, also for a site that gave the model a `u.Angle`. It cannot check that the samples came from `handler`: a handler with another `usys` puts the wrong units on them without an error. Posterior samples go with the model's handler, and the output of `svi.get_params` with the guide's.

### Known NumPyro issue

Reusing an `MCMC` object's compiled sampler with a `Quantity` argument can raise `OverflowError` ([pyro-ppl/numpyro#2434](https://github.com/pyro-ppl/numpyro/issues/2434), with a fix proposed in [#2436](https://github.com/pyro-ppl/numpyro/pull/2436)). That happens for a second chain with `chain_method="sequential"`, which is the default for several chains on one CPU device, for a second `mcmc.run`, and for `mcmc.warmup` followed by `mcmc.run`. [How to run MCMC on Quantity data](how-to-run-mcmc) has the workarounds.

## Log density

`unitful` strips every value in `usys`, so the log density is a density per `usys` unit. Changing `usys` shifts it by a constant: the posterior is unchanged, and a log density compares only with another computed in the same `usys`. [About unitful](explanation) explains why.

## Import-time patches

Importing `unxts.interop.numpyro` patches two NumPyro functions, on NumPyro versions that lack the fixes:

| Function | Bug | Fix |
| --- | --- | --- |
| `numpyro.util.not_jax_tracer` | Misses a tracer inside a pytree, such as a `Quantity` under `jit` | [pyro-ppl/numpyro#2428](https://github.com/pyro-ppl/numpyro/pull/2428) |
| `numpyro.distributions.util._reshape` | Rejects an array-like with its own `reshape`, such as a `Quantity` | [pyro-ppl/numpyro#2429](https://github.com/pyro-ppl/numpyro/pull/2429) |

Each patch is installed only if a probe at import shows the installed function has the bug. The patched functions behave like the originals for every other input.
