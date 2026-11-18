# Troubleshooting

Each entry names a symptom, its cause and its fix. A `UnitfulError` names its site, and {doc}`../reference/errors` lists every message. The errors below come from NumPyro or JAX, so their messages do not mention `unitful`.

## A `Quantity` reached plain JAX code

<!-- prettier-ignore-start -->
`TypeError: <function> requires ndarray or scalar arguments, got <class 'unxt._src.quantity.quantity.Quantity'>`
: **Cause.** A `Quantity` reached a function that accepts only plain arrays. The function name says where.
  **Fix.** Find the entry below that matches where the error happens.

In a `jax.numpy` call inside the model, such as `sqrt`
: **Fix.** Use `quaxed.numpy`, or wrap the function with `quax.quaxify` ({doc}`use-jnp-code`).

In `broadcast_to` or `reshape`, while building a `StudentT`, `Beta`, `Dirichlet` or `Pareto`
: **Cause.** NumPyro's constructor rejects a `Quantity` parameter. A `Categorical` raises `AttributeError: 'Quantity' object has no attribute 'sum'` instead.
  **Fix.** Pass the parameter as a bare number ({doc}`../reference/unit-rules`).

In `take`, from `numpyro.subsample`
: **Fix.** Index the data with the plate's indices ({doc}`mask-and-subsample`).

In `where`, at a site with `obs_mask=`
: **Fix.** Pass `rewrap=False` ({doc}`mask-and-subsample`).

In `log`, during SVI
: **Cause.** The guide has `Quantity` parameters but no `unitful` handler.
  **Fix.** Wrap the guide as `unitful(guide, M.usys)` ({doc}`fit-with-svi`).
<!-- prettier-ignore-end -->

## Other errors

<!-- prettier-ignore-start -->
`OverflowError` from `mcmc.run`
: **Cause.** NumPyro reused a compiled sampler with a `Quantity` argument.
  **Fix.** Use vectorized chains, or close over the data ({doc}`run-mcmc`).

`TypeError: Error interpreting argument to convert_element_type as a JAX value`, from a `cond`
: **Cause.** The predicate is a boolean `Quantity`, from comparing two quantities.
  **Fix.** Strip the quantities before comparing them ({doc}`use-scan-and-cond`).

A JAX error about the carry structure, from a `scan`
: **Cause.** The carry started in a unit other than the `usys` unit.
  **Fix.** Give the start value in the `usys` unit ({doc}`use-scan-and-cond`).

`UnitConversionError`, such as `'kpc' (length) and '' (dimensionless) are not convertible`, with `reparam`
: **Cause.** `reparam` is inside `unitful`.
  **Fix.** Move `reparam` outside `unitful` ({doc}`place-handlers`).

A `TypeError` at the start of MCMC, with `init_to_value`
: **Cause.** NumPyro transformed a `Quantity` initial value outside any handler.
  **Fix.** Give `init_to_value` bare values in the unit of `usys` ({doc}`run-mcmc`).

`mcmc.print_summary()` fails after a run with `init_to_value`
: **Cause.** One site came back as a `Quantity` and the others came back bare.
  **Fix.** Give `init_to_value` bare values in the unit of `usys`.
<!-- prettier-ignore-end -->

## Pitfalls without an error

These mistakes give wrong numbers and raise nothing.

<!-- prettier-ignore-start -->
The posterior has the wrong units
: **Cause.** `attach_units` got a handler other than the one that ran the samples.
  **Fix.** Pass the handler of the model or guide that ran the sites: `M` for MCMC samples, `G` for guide parameters and `Predictive(G, ...)` samples.

A bare value set outside `unitful` is off by a constant factor
: **Cause.** `unitful` reads it in the unit of `usys`, not in the unit you meant.
  **Fix.** Pass the value as a `Quantity`.

Log densities from two runs disagree by a constant
: **Cause.** The runs used different unit systems ({doc}`../concepts/unit-systems`).
  **Fix.** Compare only log densities computed under the same `usys`.
<!-- prettier-ignore-end -->
