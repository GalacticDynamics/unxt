# Units at a sample site

NumPyro computes on plain arrays. Its distributions take logarithms of their scales, its samplers move plain numbers, and its traces store them. None of these steps expects a value with a unit. `unitful` keeps the units in the model and the plain numbers in NumPyro, and it does the exchange at each {term}`sample site`.

## Stripping and wrapping

A model calls `numpyro.sample` with a distribution and, for an {term}`observed site`, a value. NumPyro passes this call to each active handler as a message. `unitful` reads the message before NumPyro samples or scores it.

At the site, `unitful` finds the site's dimension from its distribution. The {term}`unit rule` of the distribution's family names the {term}`carrying parameters <carrying parameter>`, and their dimension is the site's. The unit system then gives the unit of that dimension, the {term}`site unit`.

`unitful` strips every `Quantity` in the message to a {term}`bare value` in the site unit. NumPyro samples and scores those bare values. Afterwards `unitful` wraps the sampled value in the site unit and returns it to the model. With `rewrap=False`, the model gets the bare value instead.

```{admonition} In plain words
Every distribution gets its numbers in one agreed unit per dimension, and the model gets its answers back with units attached.
```

## Why not transform the whole model

JAX can run a function on quantities if `quax.quaxify` wraps it. Wrapping a whole NumPyro model that way does not work. NumPyro's handlers keep their state outside the function that JAX transforms. A `trace` would then record values from inside the transform, and the handlers would see values without their units.

`unitful` avoids the problem by acting at the edge of each site. The model between sites still computes on quantities. `quaxify` stays useful there, on helper functions written in plain `jax.numpy`. The {doc}`../guides/use-jnp-code` guide shows how.

## Where the units go after a fit

MCMC and SVI collect the values in NumPyro's trace. Those are the bare values NumPyro worked with, so the posterior samples have no units. A handler cannot change what MCMC and SVI store.

Instead, `unitful` records the site unit of every site it runs. `attach_units` reads that record and wraps each array in its site unit. The record is the only link between an array and its unit. A bare array cannot say which unit system produced it, so `attach_units` cannot detect the wrong handler.

## Related pages

- {doc}`unit-systems` explains why the unit system matters.
- {doc}`../reference/unit-rules` lists the carrying parameters of each family.
- {doc}`../reference/behaviour` lists what `unitful` does at each kind of site.
