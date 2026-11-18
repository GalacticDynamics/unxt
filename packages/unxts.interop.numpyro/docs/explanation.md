# About `unitful`

This page explains why `unitful` is a NumPyro handler that strips units into a unit system. That one choice is behind most of the rules in [Behaviour of `unitful`](reference): why `usys` is required, why some handlers go inside it and others outside, and why the posterior samples come back bare.

(numpyro-why-a-handler)=

## Why a handler

NumPyro computes on plain arrays. Its distributions take the log of a scale, its samplers move bare numbers around, and its traces store bare values. A `Quantity` is a JAX pytree, so some of this happens to work on quantities, but most of it doesn't.

One way around that would be to run the whole model under `quax.quaxify`, so every `jnp` call inside it sees quantities. That doesn't work, because NumPyro handlers keep their state outside the function being transformed. A `trace` records values from inside the transform, so tracers leak out of it, and the values the handlers see have lost their units.

So `unitful` stops the units at the edge of each site instead. When a model calls `numpyro.sample`, `unitful` sees the message before NumPyro does. It strips the distribution's parameters and any observed value to bare numbers, lets NumPyro sample and score on those, and gives the model back a `Quantity`, so the model can compute in units while NumPyro works on plain numbers.

That is also why `quaxify` is still useful, on a smaller scale. The values a model gets back are quantities, so a helper function written in plain `jnp` can be wrapped with `quaxify` and called on them. [How to call jnp code on unitful latents](how-to-use-jnp-code) shows this.

## Why a unit system

Stripping a `Quantity` needs a unit to express it in, and `unitful` takes that unit from `usys`: a length becomes a number of kpc under `galactic` and a number of metres under `si`.

The choice changes the numbers NumPyro sees, and so it changes the log density. A density over a length has units of one over that length, so the same data scored in kpc and in metres differ by the log of the number of metres in a kiloparsec.

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

The shift is a constant, so the posterior doesn't change. But a log-likelihood value only compares with another computed in the same unit system. `usys` is a required argument so that this choice is written down in the model's code, rather than taken from whatever unit the data happened to arrive in.

The unit system is also how `unitful` reads a bare number. When a handler outside `unitful` sets a site to `0.5`, or a family with no unit rule has bare parameters, the number has no unit of its own. `unitful` reads it in the unit of `usys`. That is why a `units=` entry has to name the `usys` unit and not just any unit of the right dimension: under `galactic`, `units={"r": "pc"}` would make bare parameters that are really kpc look like pc.

## Why the position of other handlers matters

NumPyro runs a site's message through the handler stack from the innermost handler out, and runs the results back from the outermost in. `unitful` strips the units on the way out. So every handler inside `unitful` sees quantities, and every handler outside it sees bare numbers.

That one fact gives the placement rules:

- `reparam` changes how a site is sampled. It rebuilds the distribution from its parameters with plain arithmetic, so it needs bare numbers, and goes outside.
- `condition` and `substitute` set a value. Inside `unitful` the value is stripped like any other, so it has to be a `Quantity`. Outside, the site is already bare, so a bare number works there, read in the unit of `usys`, and a `Quantity` is stripped by an extra step that `unitful` adds at the outer end of the stack.
- `do` sends a copy of the site through the stack a second time. Outside `unitful` that copy has already been stripped, and `unitful` refuses to strip it twice.
- `lift` turns a `param` site into a sample site with a prior. Outside `unitful` that happens after `unitful` has handled the site as a `param`, so the prior's parameters are never stripped and have to be bare.

## Why a custom guide needs the same unit system

SVI runs the guide, then replays the guide's sampled values into the model. Those values are bare numbers. The model's handler reads them in its own `usys`, so the guide has to have produced them in that `usys` too. If the guide's handler used parsecs where the model's uses kiloparsecs, every replayed value would be off by a factor of 1000 with no error from NumPyro. `unitful` checks the unit of each replayed site and raises instead. Building the guide's handler as `unitful(guide, M.usys)` makes the two agree by construction.

## Why the samples come back bare

The samples NumPyro collects are the values in its trace, and those are the bare numbers it worked with. Wrapping them would mean changing what MCMC and SVI store, which a handler can't do. So the handler records the unit of each site as it runs, and `attach_units` uses that record to wrap the arrays afterwards.

The record is the only link between the arrays and their units. A bare array doesn't say which unit system it came from, so `attach_units` can't tell if it was given the wrong handler.
