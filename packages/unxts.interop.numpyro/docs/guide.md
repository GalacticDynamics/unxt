# How to use unitful in a model

This page collects the tasks I ran into once the [tutorial](tutorial-fit-mean-and-width) worked. Every example uses the same setup.

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

`unitful` has to wrap the model. I write `unitful(model, usys)` and pass that to NumPyro. I do not use it as a bare `with unitful(...)` context around inference.

Handlers that change how a site is sampled go outside it. That covers `reparam` and `config_enumerate`, which need to see the bare numbers that `unitful` produces. A `reparam` inside `unitful` fails with a unit conversion error from inside the reparameterizer, and the message does not name the site.

`handlers.do` goes inside, as `unitful(handlers.do(model, data), usys)`. Outside, `do` sends a copy of the site again after `unitful` has stripped it, and `unitful` raises `UnitfulError` saying so.

```python
>>> M = unitful(model, galactic)
>>> rm = handlers.reparam(M, config={"m": LocScaleReparam(0)})
>>> list(handlers.trace(handlers.seed(rm, 0)).get_trace())
['m_decentered', 'm', 's', 'x']

```

Handlers that fix a value, `condition` and `substitute`, work on either side when I give them a quantity. A bare number works only when the handler is outside `unitful`, and I give it in the unit of `usys`. An inner handler with a bare value raises `UnitfulError`.

```python
>>> c = handlers.condition(M, {"m": u.Q(500.0, "pc")})
>>> float(handlers.trace(handlers.seed(c, 0)).get_trace()["m"]["value"])
0.5

```

The value 500 pc is stored as 0.5, which is the number in kpc. This also works when inference wraps the model in `handlers.block`, as the autoguides do, so `AutoNormal(c)` trains on `c`.

With the handler inside `unitful` the quantity works as well, and a bare value raises.

```python
>>> inner = unitful(handlers.condition(model, {"m": u.Q(500.0, "pc")}), galactic)
>>> float(handlers.trace(handlers.seed(inner, 0)).get_trace()["m"]["value"])
0.5

>>> bare = unitful(handlers.condition(model, {"m": 0.5}), galactic)
>>> handlers.seed(bare, 0)()
Traceback (most recent call last):
    ...
unxts.interop.numpyro._src.rules.UnitfulError: Site 'm': value is bare. A bare value would be read in kpc, the unit of usys; wrap the data in its own unit, as in u.Q(value, <your unit>).

```

Re-entering the same `unitful` instance is not supported, so a model that calls itself through the same handler will not work.

## Use a custom guide

A guide such as `AutoNormal` is built from the wrapped model, `AutoNormal(unitful(model, usys))`, and runs through the same handler. `Predictive` on the guide gives bare samples, and `attach_units(samples, handler)` puts the units on them. NUTS and SVI work the same way, so the posterior I get from either goes through `attach_units`.

A guide that I write myself with `Quantity` parameters needs its own handler, `unitful(guide, usys)`, with the same `usys` as the model. SVI replays the guide's bare values into the model, and the model reads them in its own `usys`. So I build the guide's handler from the model's, as `unitful(guide, M.usys)`. A guide handler with a different `usys` raises `UnitfulError` at the first replayed site. A guide with `Quantity` parameters and no handler fails inside NumPyro with a JAX `TypeError`, such as `log requires ndarray or scalar arguments`, because SVI computes the guide's log density before the model runs.

Each handler records the sites it ran. I use the model's handler for `attach_units` on the posterior samples, and the guide's handler on the fitted parameters from `svi.get_params`. `attach_units` cannot check this, because the samples are bare numbers. A handler with another `usys` puts the wrong units on them without an error.

A `param` constraint can have `Quantity` bounds, as in `interval(u.Q(0.0, "pc"), u.Q(2.0, "kpc"))`. The bounds are stripped in the unit of the init value, so that value has to be a `Quantity` too. A callable init has to return a bare number in the unit of `usys`, or I pass the `Quantity` itself as the init value.

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

>>> M = unitful(model, galactic)
>>> G = unitful(guide, M.usys)
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

>>> r = handlers.seed(unitful(lognormal, galactic, units={"r": "length"}), 0)()
>>> r.unit
Unit("kpc")

```

The `units=` value gives the site its dimension. I write the dimension, such as `"length"`, or the unit of `usys` for it. The keys are the site names as `unitful` sees them. Every key has to name a site that the call reaches, so a site that only some calls sample, such as one behind an `if`, raises `UnitfulError` on a call that skips it. A `scope` inside `unitful` is part of the name, and a `scope` outside it is not, so for `handlers.scope(unitful(model, usys), "a")` I write `units={"r": ...}`, not `"a/r"`. A key that only matches a name renamed outside `unitful` raises `UnitfulError`. The value is stripped and wrapped in the unit of `usys`, which is why `r` has the unit kpc here. The bare `loc` and `scale` are read in that unit too, so another unit of the same dimension raises. Under `galactic`, `"pc"` would silently mean kpc.

```python
>>> unitful(lognormal, galactic, units={"r": "pc"})
Traceback (most recent call last):
    ...
unxts.interop.numpyro._src.rules.UnitfulError: Site 'r': units= gives pc, but bare parameters are read in the unit of usys; pass units={'r': 'kpc'}.

```

This does not let me use bare numbers where the family's rule names parameters that carry the dimension. A `Normal` with a bare `loc` and a `Quantity` `scale` raises, and so does a `units=` entry for such a site.

```python
>>> def mixed():
...     return numpyro.sample("m", dist.Normal(1.0, u.Q(3.0, "kpc")))

>>> handlers.seed(unitful(mixed, galactic), 0)()
Traceback (most recent call last):
    ...
unxts.interop.numpyro._src.rules.UnitfulError: Site 'm': parameter 'loc' is bare next to Quantity parameters; give it units.

```

A concrete zero or infinity is allowed, so `Normal(0.0, u.Q(3.0, "kpc"))` is fine. For a bound on one side only I leave the other bound out, as in `TruncatedNormal(loc, scale, high=u.Q(2.0, "kpc"))`. A `jnp.zeros(3)` built inside the model is a tracer under `jit` and NUTS, so it raises there even though it passes when I call the model directly. For a zero-valued bare parameter I write `0.0` or `np.zeros(3)`.

## Sample inside scan and cond

`numpyro.contrib.control_flow.scan` runs its body under `handlers.block`, so `unitful` never sees the body's sites directly. When a model calls `scan`, `unitful` wraps the body in a handler of its own with the same `usys` and `rewrap`. Sites in the body then work like sites anywhere else. A latent sampled in the body comes back as a `Quantity` in the unit of `usys`, or as a bare number in that unit with `rewrap=False`. Here the walk carries a `Quantity`, so its start value is one too.

```python
>>> from numpyro.contrib.control_flow import scan

>>> def walk(ys):
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     def step(prev, y):
...         z = numpyro.sample("z", dist.Normal(prev, s))
...         numpyro.sample("y", dist.Normal(z, u.Q(0.1, "kpc")), obs=y)
...         return z, None
...     scan(step, u.Q(0.0, "kpc"), ys)

>>> ys = u.Q(jnp.array([100.0, 300.0, 200.0]), "pc")
>>> M = unitful(walk, galactic)
>>> p = {"s": jnp.asarray(0.5), "z": jnp.array([0.1, 0.2, 0.3])}
>>> logp, _ = log_density(M, (ys,), {}, p)
>>> round(float(logp), 2)
2.06
>>> M.site_units["z"]
Unit("kpc")

```

An observed value in the body still has to be a `Quantity` when the site has a dimension. A `Quantity` start value has to be in the unit of `usys`, as `u.Q(0.0, "kpc")` is here. JAX needs the carry to keep its unit through every step, and the step returns a latent in the `usys` unit. A start value in `pc` fails with a JAX error about the carry structure. A bare start value fails too, because the step gives `z` a bare `prev` next to a `Quantity` scale, and `unitful` raises `UnitfulError`.

```python
>>> def bare_walk(ys):
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     def step(prev, y):
...         z = numpyro.sample("z", dist.Normal(prev, s))
...         numpyro.sample("y", dist.Normal(z, u.Q(0.1, "kpc")), obs=y)
...         return z, None
...     scan(step, 0.0, ys)

>>> try:
...     log_density(unitful(bare_walk, galactic), (ys,), {}, p)
... except UnitfulError as e:
...     print(e)
Site 'z': parameter 'loc' is bare next to Quantity parameters; give it units. If it is a zero or an infinity, write it as a concrete 0.0 or np.zeros(...), not jnp.zeros(...) inside the model. Inside a scan, a carry that starts bare stays bare: give the start value as a Quantity in the usys unit, or pass rewrap=False and add the units in the body.

```

With `rewrap=False` the body gets bare numbers in the `usys` unit, so a bare start value works. I add the units back in the body where a distribution needs them, and get the same log density as above.

```python
>>> def walk_nowrap(ys):
...     s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
...     def step(prev, y):
...         z = numpyro.sample("z", dist.Normal(u.Q(prev, "kpc"), u.Q(s, "kpc")))
...         numpyro.sample("y", dist.Normal(u.Q(z, "kpc"), u.Q(0.1, "kpc")), obs=y)
...         return z, None
...     scan(step, 0.0, ys)

>>> logp, _ = log_density(unitful(walk_nowrap, galactic, rewrap=False), (ys,), {}, p)
>>> round(float(logp), 2)
2.06

```

`numpyro.contrib.control_flow.cond` runs its branches the same way, and `unitful` wraps each branch in the same kind of handler. NumPyro needs both branches to sample the same site names from the same family, and `unitful` needs a site to have the same unit in both. The predicate has to be a bare boolean. A comparison of two quantities, such as `r < u.Q(5.0, "kpc")`, is a boolean `Quantity`, and JAX rejects it as a predicate. I compare bare numbers, as in `u.ustrip("kpc", r) < 5.0`.

```python
>>> from numpyro.contrib.control_flow import cond

>>> def pick(near):
...     def close(_):
...         return numpyro.sample("d", dist.Normal(u.Q(1.0, "kpc"), u.Q(0.1, "kpc")))
...     def far(_):
...         return numpyro.sample("d", dist.Normal(u.Q(8.0, "kpc"), u.Q(0.1, "kpc")))
...     return cond(near, close, far, None)

>>> handlers.seed(unitful(pick, galactic), 0)(True).unit
Unit("kpc")

```

## Use masked observations with rewrap=False

A site with `obs_mask=` fails when `unitful` gives the model a `Quantity`. I pass `rewrap=False`, so the handler gives the model bare numbers in the unit of `usys`, and I put the units back myself where I need them. NumPyro splits such a site into `x_observed` and `x_unobserved`, so a `units=` entry for it uses those two names.

```python
>>> def masked(x):
...     m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
...     mq = u.Q(m, "kpc")
...     mask = jnp.array([True, False, True])
...     numpyro.sample("x", dist.Normal(mq, u.Q(1.0, "kpc")), obs=x, obs_mask=mask)

>>> x = u.Q(jnp.array([1.0, 1.2, 0.9]), "kpc")
>>> M = unitful(masked, galactic, rewrap=False)
>>> p = {"m": jnp.asarray(1.0), "x_unobserved": jnp.zeros(3)}
>>> logp, _ = log_density(M, (x,), {}, p)
>>> round(float(logp), 1)
-5.3

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

Stripping in `usys` means the log density is a density per `usys` unit. If I change the unit system, the log density shifts by a constant, so a posterior does not change but a log-likelihood value only compares to another one computed with the same `usys`.

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

A `VonMises` site is the exception. Its density is defined on radians, so it is always stripped and wrapped in radians, whatever the angle unit of `usys` is.

## Things that do not work

- `init_to_value` with `Quantity` values works for some models and fails for others, because NumPyro transforms the values while no handler is running. Where it works, MCMC can hand back the samples as quantities in the unit I gave, not in the unit of `usys`. I give `init_to_value` bare numbers in the unit of `usys`.
- Several chains run one after the other (`chain_method="sequential"`, which is what the default becomes on a single CPU device) fail when the data argument is a `Quantity`. This is a NumPyro bug, [pyro-ppl/numpyro#2434](https://github.com/pyro-ppl/numpyro/issues/2434), fixed by [#2436](https://github.com/pyro-ppl/numpyro/pull/2436) but not yet released. Until a release has the fix I pass `chain_method="vectorized"`, or I close over the data with `functools.partial(model, x)` so it is not an argument of `mcmc.run`.
- Calling `mcmc.run` a second time on the same `MCMC` object with a `Quantity` data argument fails with an `OverflowError`, which is a NumPyro issue. Calling `mcmc.warmup(...)` and then `mcmc.run(...)` with a `Quantity` data argument fails the same way. I create a new `MCMC` object for the second run, and I let `mcmc.run` do the warmup.
- A `unitful` inside another `unitful` with a different `usys` does not convert between them. The inner one strips everything, so the outer one records no units. I use one `unitful` with one `usys`.
- A `unitful` inside another one with the same `usys` works, but the outer one sees the inner sites already stripped and records no unit for them. `attach_units` needs the innermost handler that ran the sites.
- A `deterministic` or `param` value that is a container holding quantities, such as `{"a": u.Q(1.0, "pc")}`, raises `UnitfulError`. I give each quantity a site of its own.
- Distributions of the `MultivariateNormal` family with `Quantity` parameters are out of scope.
- Shape parameters, such as `df` of a `StudentT`, `concentration` of a `Gamma` or `VonMises`, and mixing probabilities, carry no dimension. A dimensionless `Quantity` there is stripped, and a dimensioned one raises `UnitfulError`. The `StudentT`, `Beta`, `Categorical` and `Dirichlet` constructors reject even a dimensionless `Quantity`, so `dist.StudentT(u.Q(3.0, ""), loc, scale)` raises when it is built. I write `dist.StudentT(3.0, loc, scale)`.
- A site whose dimension-carrying parameters are dimensionless `Quantity` objects is a dimensionless site. Bare parameters and a bare observed value there are read as plain numbers, so `dist.Normal(a + b * (r / u.Q(1.0, "kpc")), s)` with bare `a`, `b`, `s` and bare data works. A latent at such a site comes back as a `Quantity` with unit `""`, or bare with `rewrap=False`.
- A dimensionless `Quantity` is stripped wherever the site has no unit. That covers an observed value at a unitless site and the parameters of a family with no unit rule, such as the logits of a `Bernoulli` built from `beta * x`.
- A distribution with no unit rule that holds a `Quantity` with a dimension anywhere raises `UnitfulError`. That covers `MixtureGeneral`, the censored distributions and a `ComposeTransform`. I write its parameters as bare numbers in the unit of `usys` and pass `units=`. For a censored distribution that means a bare base distribution, as in `LeftCensoredDistribution(dist.Normal(0.0, 1.0), censored=mask)` with `units={"x": "kpc"}`. The `censored=` argument is a boolean mask, so a `Quantity` there fails in NumPyro's constructor.
- A `Quantity` set at a site that `unitful` has no unit for raises `UnitfulError`, with the site name in the message.
- `handlers.scale`, inside or outside `unitful`, takes a bare or dimensionless `Quantity` scale. A scale with a dimension raises `UnitfulError`.
- `attach_units` is for site values: samples, `Predictive` output and fitted parameters. The output of `numpyro.infer.log_likelihood` is log densities per `usys` unit, so I leave it bare.
