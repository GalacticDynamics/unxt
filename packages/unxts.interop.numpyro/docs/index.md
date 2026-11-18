# `unxts.interop.numpyro`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-fit-mean-and-width
how-to-run-mcmc
how-to-fit-with-svi
how-to-place-handlers
how-to-give-a-site-units
how-to-use-scan-and-cond
how-to-mask-and-subsample
how-to-use-jnp-code
reference
unit-rules
api
explanation
```

`unxts.interop.numpyro` runs [NumPyro](https://num.pyro.ai/) models written with `unxt` quantities. You write the distributions with `Quantity` parameters and pass `Quantity` data, wrap the model in the `unitful` handler with a unit system, and fit it with the usual NumPyro inference. The model gets its sampled values back as quantities, and `attach_units` puts the units on the posterior.

```python
>>> import numpyro
>>> import numpyro.distributions as dist
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> def model():
...     return numpyro.sample("m", dist.Normal(u.Q(0.0, "pc"), u.Q(3.0, "pc")))

>>> m = numpyro.handlers.seed(unitful(model, u.unitsystems.galactic), 0)()
>>> m.unit
Unit("kpc")

```

The distribution is written in parsecs, and the sample comes back in kiloparsecs, the length unit of the `galactic` unit system.

## Install

Install it with `unxt` through the `interop-numpyro` [extra](https://peps.python.org/pep-0508/#extras), so that `unxt`, `numpyro` and this package resolve to versions that work together:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-numpyro]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[interop-numpyro]"
```

:::

::::

To install the package on its own, use `uv add unxts.interop.numpyro` or `pip install unxts.interop.numpyro`.

## Tutorial

- [Fit a mean and a width to data with units](tutorial-fit-mean-and-width). Start here. You fit a model to `Quantity` data with NUTS and get the posterior back in units.

## How-to guides

- [How to run MCMC on `Quantity` data](how-to-run-mcmc): several chains, repeated runs and initial values.
- [How to fit a model with SVI](how-to-fit-with-svi): autoguides, your own guide, and fitted parameters with units.
- [How to combine `unitful` with other NumPyro handlers](how-to-place-handlers): `condition`, `substitute`, `reparam`, `do` and `lift`.
- [How to give units to a site whose distribution has no unit rule](how-to-give-a-site-units): `LogNormal`, censored distributions and others, with `units=`.
- [How to sample inside `scan` and `cond`](how-to-use-scan-and-cond): carries and branches.
- [How to mask or subsample `Quantity` observations](how-to-mask-and-subsample): `obs_mask=` and subsampled plates.
- [How to call `jnp` code on `unitful` latents](how-to-use-jnp-code): `quaxed.numpy` and `quax.quaxify`.

## Reference

- [Behaviour of `unitful`](reference): what happens at each kind of site, where each NumPyro handler goes, inference, and the NumPyro patches applied at import.
- [Distribution unit rules](unit-rules): which parameters carry a site's dimension, family by family.
- [API](api): `unitful`, `attach_units` and `UnitfulError`.

## Explanation

- [About `unitful`](explanation): why it is a handler, why it needs a unit system, and why handler order matters.

## See also

- [NumPyro documentation](https://num.pyro.ai/)
- [unxt documentation](https://unxt.readthedocs.io/)
