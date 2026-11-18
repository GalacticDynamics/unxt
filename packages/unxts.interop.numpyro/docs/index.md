# `unxts.interop.numpyro`

**Write NumPyro models in physical units.**

[NumPyro](https://num.pyro.ai/) computes on plain numbers, so a model with units normally strips them by hand. Every parameter, data array and prior then carries an unwritten unit. `unxts.interop.numpyro` lets the model keep its `unxt` quantities. The `unitful` handler converts them to plain numbers in one unit system at each sample site. NumPyro samples and scores those numbers. The model gets its sampled values back as quantities, and `attach_units` puts the units on the posterior.

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

The prior is written in parsecs. The sample comes back in kiloparsecs, the length unit of the `galactic` unit system.

## Where to start

- **New to `unitful`.** Fit a model to `Quantity` data in the {doc}`tutorial <tutorial-fit-mean-and-width>`.
- **Moving an existing NumPyro model.** Give its distributions `Quantity` parameters, wrap it, and check the {doc}`handler placement <guides/place-handlers>`. A family without a unit rule needs {doc}`units= <guides/give-a-site-units>`.
- **An error to fix.** A `UnitfulError` is listed in {doc}`reference/errors`. An error from NumPyro or JAX is in {doc}`guides/troubleshooting`.

## Install

Install it with `unxt`, through the `interop-numpyro` [extra](https://peps.python.org/pep-0508/#extras). The extra resolves `unxt`, `numpyro` and this package to versions that work together.

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

To install the package on its own, run `uv add unxts.interop.numpyro` or `pip install unxts.interop.numpyro`.

## Sections

- **{doc}`Tutorial <tutorial-fit-mean-and-width>`.** Fit a mean and a width to distances in kiloparsecs, from data to posterior.
- **{doc}`Guides <guides/index>`.** Write the model, place the handlers, run MCMC or SVI, read the results, and fix errors.
- **{doc}`Concepts <concepts/index>`.** How `unitful` treats each site, why it needs a unit system, and why handler order matters.
- **Reference.** {doc}`Behaviour <reference/behaviour>`, {doc}`unit rules <reference/unit-rules>`, {doc}`errors <reference/errors>`, the {doc}`glossary <reference/glossary>` and the {doc}`API <reference/api>`.

For units, dimensions and unit systems, see the [`unxt` documentation](https://unxt.readthedocs.io/). For models and inference, see the [NumPyro documentation](https://num.pyro.ai/).

```{toctree}
:hidden:

tutorial-fit-mean-and-width
guides/index
concepts/index
reference/behaviour
reference/unit-rules
reference/errors
reference/glossary
reference/api
```
